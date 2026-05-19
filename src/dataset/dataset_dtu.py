import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset, Dataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from .dataset_re10k import DatasetRE10kCfg as DatasetScannetCfg

import numpy as np
import os
from PIL import Image

from torch.utils.data import get_worker_info
from torch.distributed import init_process_group, get_rank, get_world_size


from torch.utils.data import Dataset
import os, re
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
from termcolor import colored
from einops import repeat

from .scene_transform import get_boundingbox


# @dataclass
# class DatasetScannetCfg(DatasetCfgCommon):
#     name: Literal["re10k", "scannet"]
#     roots: list[Path]
#     baseline_epsilon: float
#     max_fov: float
#     make_baseline_1: bool
#     augment: bool

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class DatasetDTU(Dataset):
    cfg: DatasetScannetCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetScannetCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
    # def __init__(self, cfg, stage, n_views=3, img_wh=(640, 512),
    #              split_filepath=None, 
    #              pair_filepath=None, 
    #              N_rays=1024,
    #              test_ref_views=[]):

        self.root_dir = cfg.roots[0]
        self.split = stage

        self.img_wh = cfg.image_shape[::-1]
        img_wh = cfg.image_shape[::-1]
        self.num_all_imgs = 49
        self.n_views = cfg.n_views 
        self.N_rays = 1024

        # self.test_ref_views = [23]  # used for validation
        self.test_ref_views = []  # used for validation
        self.scale_factor = 1.0
        self.scale_mat = np.float32(np.diag([1, 1, 1, 1.0]))

        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'

        self.split_filepath = f'datasets/dtu/lists/{stage}.txt'
        self.pair_filepath = 'datasets/dtu/dtu_pairs.txt'

        print(colored("loading all scenes together", 'red'))
        with open(self.split_filepath) as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        self.all_intrinsics = []  # the cam info of the whole scene
        self.all_extrinsics = []
        self.all_near_fars = []


        self.metas, self.ref_src_pairs = self.build_metas()  # load ref-srcs view pairs info of the scene

        self.allview_ids = [i for i in range(self.num_all_imgs)]

        self.load_cam_info()  # load camera info of DTU, and estimate scale_mat

        self.build_remap()
        self.define_transforms()

        # * bounding box for rendering
        self.bbox_min = np.array([-1.0, -1.0, -1.0])
        self.bbox_max = np.array([1.0, 1.0, 1.0])

        self.img_W, self.img_H = self.img_wh
        h_line = (np.linspace(0,self.img_H-1,self.img_H))*2/(self.img_H-1) - 1
        w_line = (np.linspace(0,self.img_W-1,self.img_W))*2/(self.img_W-1) - 1
        h_mesh, w_mesh = np.meshgrid(h_line, w_line, indexing='ij')
        self.w_mesh_flat = w_mesh.reshape(-1)
        self.h_mesh_flat = h_mesh.reshape(-1)
        self.homo_pixel = np.stack([self.w_mesh_flat, self.h_mesh_flat, np.ones(len(self.h_mesh_flat)), np.ones(len(self.h_mesh_flat))])  #[4,HW]


    def build_remap(self):
        self.remap = np.zeros(np.max(self.allview_ids) + 1).astype('int')
        for i, item in enumerate(self.allview_ids):
            self.remap[item] = i


    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor()])


    def build_metas(self):
        metas = []
        ref_src_pairs = {}
        light_idxs = [3] if 'train' not in self.split else range(7)

        with open(self.pair_filepath) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                ref_src_pairs[ref_view] = src_views

        for light_idx in light_idxs:
            for scan in self.scans:
                with open(self.pair_filepath) as f:
                    num_viewpoint = int(f.readline())
                    # viewpoints (49)
                    for _ in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                        # ! only for validation
                        if len(self.test_ref_views) > 0 and ref_view not in self.test_ref_views:
                            continue

                        metas += [(scan, light_idx, ref_view, src_views)]
        
        # print('metas:', metas)
        print('length:', len(metas))

        return metas, ref_src_pairs


    def read_cam_file(self, filename):
        """
        Load camera file e.g., 00000000_cam.txt
        """
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0]) * self.scale_factor
        depth_max = depth_min + float(lines[11].split()[1]) * 192 * self.scale_factor
        
        self.depth_interval = float(lines[11].split()[1])
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics

        return intrinsics_, extrinsics, [depth_min, depth_max]

    def load_cam_info(self):
        for vid in range(self.num_all_imgs):
            proj_mat_filename = os.path.join(self.root_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4  # * the provided intrinsics is 4x downsampled, now keep the same scale with image
            self.all_intrinsics.append(intrinsic)
            extrinsic[:3, 3] *= self.scale_factor
            self.all_extrinsics.append(extrinsic)
            self.all_near_fars.append(near_far)
        
        self.all_intrinsics_debug = self.all_intrinsics.copy()
        self.all_extrinsics_debug = self.all_extrinsics.copy()


    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (1200, 1600)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        return depth_h


    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)

        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()


    def __len__(self):
        return len(self.metas)


    def __getitem__(self, idx):
        sample = {}
        scan, light_idx, ref_view, src_views = self.metas[idx % len(self.metas)]

        view_ids = [ref_view] + src_views[:self.n_views]

        # print('++++++++++++++view_ids:', view_ids)
        w2c_ref = self.all_extrinsics[self.remap[ref_view]]
        w2c_ref_inv = np.linalg.inv(w2c_ref)

        imgs, depths_h = [], []
        intrinsics, w2cs, near_fars = [], [], []  # record proj mats between views
        extrinsics = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,
                                        f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')
            depth_filename = os.path.join(self.root_dir,
                                          f'Depths_raw/{scan}/depth_map_{vid:04d}.pfm')
            
            img = Image.open(img_filename)
            img = self.transform(img)
            imgs += [img]

            index_mat = self.remap[vid]
            # print(f'+++++++++++vid:{vid}, index_mat:{index_mat}')
            near_fars.append(self.all_near_fars[index_mat])
            intrinsics.append(self.all_intrinsics[index_mat])

            # print('load intrinsic:', self.all_intrinsics[index_mat])

            # issue/7
            w2cs.append(self.all_extrinsics[index_mat] @ w2c_ref_inv)
            extrinsics.append(np.linalg.inv(self.all_extrinsics[index_mat]))

            # print('++++++++++++++depth_filename:', depth_filename)

            if os.path.exists(depth_filename):  # and i == 0
                depth_h = self.read_depth(depth_filename) * self.scale_factor
                depths_h.append(depth_h)

        scale_mat, scale_factor = self.cal_scale_mat(img_hw=[self.img_wh[1], self.img_wh[0]],
                                                     intrinsics=intrinsics, extrinsics=w2cs,
                                                     near_fars=near_fars, factor=1.1)
        near_fars_raw = np.stack(near_fars)
        depths_raw = np.stack(depths_h)
        new_near_fars = []
        new_w2cs = []
        new_c2ws = []
        new_depths_h = []
        new_intrinsics = []
        # print('depths_h:', depths_h)
        for intrinsic, extrinsic, depth in zip(intrinsics, w2cs, depths_h):

            P = intrinsic @ extrinsic @ scale_mat
            P = P[:3, :4]
            new_in, c2w = load_K_Rt_from_P(None, P)

            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)

            new_intrinsics.append(new_in)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2))
            near = dist - 1
            far = dist + 1
            new_near_fars.append([0.95 * near, 1.05 * far])
            new_depths_h.append(depth * scale_factor)
        
        # print(f'scan: {scan}, view_ids: {view_ids}, new_depths_h: {new_depths_h}')
        imgs = torch.stack(imgs).float()
        depths_h = np.stack(new_depths_h)

        w2cs_raw = np.stack(w2cs)
        new_intrinsics = np.stack(new_intrinsics)

        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(new_w2cs), np.stack(new_c2ws), np.stack(new_near_fars)
        extrinsics = np.stack(extrinsics)
        start_idx = 0

        imshape = self.to_tensor(Image.open(img_filename)).shape

        

        sample['images'] = imgs[start_idx:]  # (V, 3, H, W)
        sample['w2cs'] = torch.from_numpy(w2cs.astype(np.float32))[start_idx:]  # (V, 4, 4)
        sample['c2ws'] = torch.from_numpy(c2ws.astype(np.float32))[start_idx:]  # (V, 4, 4)
        sample['near_fars'] = torch.from_numpy(near_fars.astype(np.float32))[start_idx:]  # (V, 2)
        # sample['intrinsics'] = torch.from_numpy(new_intrinsics.astype(np.float32))[start_idx:, :3, :3]  # (V, 3, 3)
        # sample['near_fars'] = torch.from_numpy(near_fars_raw.astype(np.float32))[start_idx:]
        sample['intrinsics'] = torch.from_numpy(intrinsics.astype(np.float32))[start_idx:, :3, :3]  # (V, 3, 3)
        sample['extrinsics'] = torch.from_numpy(extrinsics.astype(np.float32))[start_idx:]

        

        sample['meta'] = str(scan) + "_light" + str(light_idx) + "_refview" + str(ref_view)

        sample['scale_mat'] = torch.from_numpy(scale_mat)
        sample['trans_mat'] = torch.from_numpy(w2c_ref_inv)

        # extrinsics
        intrinsics_pad = repeat(torch.eye(4), "X Y -> L X Y", L = len(sample['w2cs'])).clone()
        intrinsics_pad[:,:3,:3] = sample['intrinsics']
        sample['ref_pose']         = (intrinsics_pad @ sample['w2cs'])[0]     # 4, 4
        sample['source_poses']     = (intrinsics_pad @ sample['w2cs'])[1:] 
        
        # from 0~W to NDC's -1~1
        normalize_matrix = torch.tensor([[1/((self.img_W-1)/2), 0, -1, 0], [0, 1/((self.img_H-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]])
        sample['ref_pose'] = normalize_matrix @ sample['ref_pose']
        sample['source_poses'] = normalize_matrix @ sample['source_poses']
        
        sample['ref_pose_inv'] = torch.inverse(sample['ref_pose'])
        sample['source_poses_inv'] = torch.inverse(sample['source_poses'])
        
        sample['ray_o'] = sample['ref_pose_inv'][:3,-1]      # 3

        tmp_ray_d = (sample['ref_pose_inv'] @ self.homo_pixel)[:3] - sample['ray_o'][:,None]
        tmp_ray_d = tmp_ray_d / torch.linalg.norm(tmp_ray_d, dim=0, keepdim=True)
        sample['ray_d'] = tmp_ray_d

        cam_ray_d = (torch.inverse(normalize_matrix @ intrinsics_pad[0]) @ self.homo_pixel)[:3]
        cam_ray_d = cam_ray_d / torch.linalg.norm(cam_ray_d, dim=0, keepdim=True)
        sample['cam_ray_d'] = cam_ray_d

        depths_h = torch.from_numpy(depths_h.astype(np.float32))[start_idx:]
        V,H,W = depths_h.size()       
        depths_h = depths_h.view(V,-1)
        depths_h = depths_h/cam_ray_d[2:3,:]
        depths_h = depths_h.view(V,H,W)
        sample['depths_h'] = depths_h

        sample['intrinsics'][:, :1] /= imshape[2]
        sample['intrinsics'][:, 1:2] /= imshape[1]

        example = {'target':{}, 'context':{}}

        example['target']['image'] = sample['images'][:1] # 3, 512, 640
        example['context']['image'] = sample['images'][1:] # 3, 3, 512, 640

        example['target']['intrinsics'] = sample['intrinsics'][:1]
        example['context']['intrinsics'] = sample['intrinsics'][1:]

        # example['target']['extrinsics'] = sample['extrinsics'][:1]
        # example['context']['extrinsics'] = sample['extrinsics'][1:]
        example['target']['extrinsics'] = sample['c2ws'][:1]
        example['context']['extrinsics'] = sample['c2ws'][1:]
        # example['target']['extrinsics'] = torch.from_numpy(w2cs_raw.astype(np.float32))[start_idx:][:1]
        # example['context']['extrinsics'] = torch.from_numpy(w2cs_raw.astype(np.float32))[start_idx:][1:]
        # example['target']['extrinsics'] = sample['ref_pose'][None].type(torch.float32)
        # example['context']['extrinsics'] = sample['source_poses'].type(torch.float32)


        example['target']['index'] = np.array(ref_view)
        example['context']['index'] = np.array(src_views)

        example['target']['near'] = sample['near_fars'][:1,0]
        example['target']['far'] = sample['near_fars'][:1,1]
        example['context']['near'] = sample['near_fars'][1:,0]
        example['context']['far'] = sample['near_fars'][1:,1]


        # print('extrinsics:', example['context']['extrinsics'])
        # print('near:', example['context']['near'])
        # print('far:', example['context']['far'])

        example['context']['depth'] = depths_h[1:].unsqueeze(1).type(torch.float32)
        example['target']['depth'] = depths_h[:1].unsqueeze(1).type(torch.float32)
        # print('now depth shape:', depths_h[:1].unsqueeze(1).shape)
        # print('pre depth shape:', torch.from_numpy(depths_h.astype(np.float32))[:1].unsqueeze(1).shape)
        # example['context']['depth'] = torch.from_numpy(depths_h.astype(np.float32))[1:].unsqueeze(1)
        # example['target']['depth'] = torch.from_numpy(depths_h.astype(np.float32))[:1].unsqueeze(1)
        # example['context']['depth'] = torch.from_numpy(depths_raw.astype(np.float32))[1:].unsqueeze(1)
        # example['target']['depth'] = torch.from_numpy(depths_raw.astype(np.float32))[:1].unsqueeze(1)

        # print('context_depth.shape:', example['context']['depth'].shape)
        # print('target_depth.shape:', example['target']['depth'].shape)
        example['scene'] = scan

        # return example
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
        return apply_crop_shim(example, tuple(self.cfg.image_shape))
        
        
     
        # return sample


# class DatasetScannet(IterableDataset):
    # cfg: DatasetScannetCfg
    # stage: Stage
    # view_sampler: ViewSampler

    # to_tensor: tf.ToTensor
    # chunks: list[Path]
    # near: float = 0.1
    # far: float = 1000.0

    # def __init__(
    #     self,
    #     cfg: DatasetScannetCfg,
    #     stage: Stage,
    #     view_sampler: ViewSampler,
    # ) -> None:
    #     super().__init__()
    #     self.cfg = cfg
    #     self.stage = stage
    #     self.view_sampler = view_sampler
    #     self.to_tensor = tf.ToTensor()

    #     # Collect chunks.
    #     self.chunks = []
    #     if self.data_stage != 'test':
    #         for root in cfg.roots:
    #             root = root / self.data_stage
    #             root_chunks = sorted(
    #                 [path for path in root.iterdir()]
    #             )
    #             self.chunks.extend(root_chunks)
    #     else:
    #         root = cfg.roots[0] / self.data_stage
    #         self.chunks = sorted(
    #                 [root / path for path in self.index]
    #             )
    #         # print('self.chunks:', self.chunks)
    #     if self.cfg.overfit_to_scene is not None:
    #         chunk_path = self.index[self.cfg.overfit_to_scene]
    #         self.chunks = [chunk_path] * len(self.chunks)

    # def shuffle(self, lst: list) -> list:
    #     indices = torch.randperm(len(lst))
    #     return [lst[x] for x in indices]

    # def __iter__(self):
    #     # Chunks must be shuffled here (not inside __init__) for validation to show
    #     # random chunks.
    #     worker_info = get_worker_info()
    #     if self.stage in ("train", "val"):
    #         self.chunks = self.shuffle(self.chunks)

    #     # When testing, the data loaders alternate chunks.
    #     worker_info = torch.utils.data.get_worker_info()
    #     if self.stage == "test" and worker_info is not None:
    #         self.chunks = [
    #             chunk
    #             for chunk_index, chunk in enumerate(self.chunks)
    #             if chunk_index % worker_info.num_workers == worker_info.id
    #         ]

    #     # for chunk_path in self.chunks:
    #     #     # Load the chunk.
    #     #     chunk = torch.load(chunk_path)

    #     #     if self.cfg.overfit_to_scene is not None:
    #     #         item = [x for x in chunk if x["key"] == self.cfg.overfit_to_scene]
    #     #         assert len(item) == 1
    #     #         chunk = item * len(chunk)

    #     #     if self.stage in ("train", "val"):
    #     #         chunk = self.shuffle(chunk)
    #     while True:
    #         for path in self.chunks:
    #             # print('+++++++++++++++++path:', path)
    #             scene = str(path).split('/')[-1]
    #             if self.data_stage == 'test':
    #                 path = str(path)[:-2]


    #             imshape = self.to_tensor(Image.open(os.path.join(path, 'color', '0.jpg'))).shape
    #             # print('+++++++++++++++++++++++++++imshape:', imshape)
    #             # extrinsics, intrinsics = self.convert_poses(example["cameras"])
    #             extrinsics = torch.from_numpy(np.load(os.path.join(path, 'extrinsics.npy'))).float()
    #             intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_color.txt'))\
    #                                         [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
    #             intrinsics[:, :1] /= imshape[2]
    #             intrinsics[:, 1:2] /= imshape[1]

    #             depth_imshape = self.to_tensor(Image.open(os.path.join(path, 'depth', '0.png'))).shape
    #             depth_intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_depth.txt'))\
    #                                         [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
    #             # print('depth_imshape:', depth_imshape)
    #             # print('depth_intrinsics:', depth_intrinsics)
    #             depth_intrinsics[:, :1] /= depth_imshape[2]
    #             depth_intrinsics[:, 1:2] /= depth_imshape[1]
    #             # print('intrinsics:', intrinsics)
    #             # print('extrinsics:', extrinsics)
    #             # print('extrinsics.shape:', extrinsics.shape)
    #             # print('intrinsics.shape:', intrinsics.shape)
    #             # scene = example["key"]

    #             # try:
    #             context_indices, target_indices = self.view_sampler.sample(
    #                 scene,
    #                 extrinsics,
    #                 intrinsics,
    #                 )
    #             # print('context_indices:', context_indices)
    #             # print('target_indices:', target_indices)
    #             # except ValueError:
    #             #     # Skip because the example doesn't have enough frames.
    #             #     continue

    #             # Skip the example if the field of view is too wide.
    #             if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
    #                 continue

    #             # Load the images.
    #             # context_images = [
    #             #     example["images"][index.item()] for index in context_indices
    #             # ]
    #             # context_images = self.convert_images(context_images)
    #             # target_images = [
    #             #     example["images"][index.item()] for index in target_indices
    #             # ]
    #             # target_images = self.convert_images(target_images)
    #             context_images = []
    #             target_images = []
    #             for t in ['context', 'target']:
    #                 for idx in eval(f'{t}_indices'):
    #                     img = Image.open(os.path.join(path, 'color', str(idx.numpy())+'.jpg'))
    #                     img = self.to_tensor(img.resize((640, 480)))
    #                     eval(f'{t}_images').append(img[None])
                
    #             if self.cfg.use_depth:
    #                 context_depths = []
    #                 target_depths = []
    #                 for t in ['context', 'target']:
    #                     for idx in eval(f'{t}_indices'):
    #                         img = Image.open(os.path.join(path, 'depth', str(idx.numpy())+'.png'))
    #                         img = (np.asarray(img.resize((640, 480))) / 1000).astype(np.float16)
    #                         img = self.to_tensor(img)
    #                         eval(f'{t}_depths').append(img[None])
    #                         # print('path:', os.path.join(path, 'depth', str(idx.numpy())+'.png'))
    #                 # print('context_depths', context_depths)
    #                 context_depths = torch.cat(context_depths)
    #                 target_depths = torch.cat(target_depths)
                
                
    #             context_images = torch.cat(context_images)
    #             target_images = torch.cat(target_images)
    #             # print('context_images:', context_images)
    #             # print('target_images:', target_images)

    #             # Skip the example if the images don't have the right shape.
    #             # context_image_invalid = context_images.shape[1:] != (3, 480, 640)
    #             # target_image_invalid = target_images.shape[1:] != (3, 480, 640)
    #             # if context_image_invalid or target_image_invalid:
    #             #     print(
    #             #         f"Skipped bad example {example['key']}. Context shape was "
    #             #         f"{context_images.shape} and target shape was "
    #             #         f"{target_images.shape}."
    #             #     )
    #             #     continue

    #             # Resize the world to make the baseline 1.
    #             context_extrinsics = extrinsics[context_indices]
    #             if context_extrinsics.shape[0] == 2 and self.cfg.make_baseline_1:
    #                 a, b = context_extrinsics[:, :3, 3]
    #                 scale = (a - b).norm()
    #                 if scale < self.cfg.baseline_epsilon:
    #                     print(
    #                         f"Skipped {scene} because of insufficient baseline "
    #                         f"{scale:.6f}"
    #                     )
    #                     continue
    #                 extrinsics[:, :3, 3] /= scale
    #             else:
    #                 scale = 1
    #             # scale = 1

    #             example = {
    #                 "context": {
    #                     "extrinsics": extrinsics[context_indices],
    #                     "intrinsics": intrinsics[context_indices],
    #                     "image": context_images,
    #                     "near": self.get_bound("near", len(context_indices)) / scale,
    #                     "far": self.get_bound("far", len(context_indices)) / scale,
    #                     "index": context_indices,
    #                 },
    #                 "target": {
    #                     "extrinsics": extrinsics[target_indices],
    #                     "intrinsics": intrinsics[target_indices],
    #                     "image": target_images,
    #                     "near": self.get_bound("near", len(target_indices)) / scale,
    #                     "far": self.get_bound("far", len(target_indices)) / scale,
    #                     "index": target_indices,
    #                 },
    #                 "scene": scene,
    #             }
    #             if self.cfg.use_depth:
    #                 example['context']['depth'] = context_depths
    #                 example['context']['depth_intrinsics'] = depth_intrinsics[context_indices]
    #                 example['target']['depth'] = target_depths
    #                 example['target']['depth_intrinsics'] = depth_intrinsics[context_indices]
    #             # print('example:', example)
    #             if self.stage == "train" and self.cfg.augment:
    #                 example = apply_augmentation_shim(example)
    #             yield apply_crop_shim(example, tuple(self.cfg.image_shape))

    # def convert_poses(
    #     self,
    #     poses: Float[Tensor, "batch 18"],
    # ) -> tuple[
    #     Float[Tensor, "batch 4 4"],  # extrinsics
    #     Float[Tensor, "batch 3 3"],  # intrinsics
    # ]:
    #     b, _ = poses.shape

    #     # Convert the intrinsics to a 3x3 normalized K matrix.
    #     intrinsics = torch.eye(3, dtype=torch.float32)
    #     intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    #     fx, fy, cx, cy = poses[:, :4].T
    #     intrinsics[:, 0, 0] = fx
    #     intrinsics[:, 1, 1] = fy
    #     intrinsics[:, 0, 2] = cx
    #     intrinsics[:, 1, 2] = cy

    #     # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
    #     w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
    #     w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    #     return w2c.inverse(), intrinsics

    # def convert_images(
    #     self,
    #     images: list[UInt8[Tensor, "..."]],
    # ) -> Float[Tensor, "batch 3 height width"]:
    #     torch_images = []
    #     for image in images:
    #         image = Image.open(BytesIO(image.numpy().tobytes()))
    #         torch_images.append(self.to_tensor(image))
    #     # print('-----------------torch_images.shape:', torch_images.shape)
    #     return torch.stack(torch_images)

    # def get_bound(
    #     self,
    #     bound: Literal["near", "far"],
    #     num_views: int,
    # ) -> Float[Tensor, " view"]:
    #     value = torch.tensor(getattr(self, bound), dtype=torch.float32)
    #     return repeat(value, "-> v", v=num_views)

    # @property
    # def data_stage(self) -> Stage:
    #     if self.cfg.overfit_to_scene is not None:
    #         return "test"
    #     if self.stage == "val":
    #         return "test"
    #     return self.stage

    # @cached_property
    # def index(self) -> dict[str, Path]:
    #     data_stages = [self.data_stage]
    #     if self.cfg.overfit_to_scene is not None:
    #         data_stages = ("test", "train")
        
    #     merged_index = {}
    #     for data_stage in data_stages:
    #         for root in self.cfg.roots:
    #             # Load the root's index.
    #             # with (root / data_stage / "index.json").open("r") as f:
    #             #     index = json.load(f)
    #             with open(root / f'{data_stage}_idx.txt', 'r') as f:
    #                 index = f.read().split('\n')
    #             index = {x: Path(root / data_stage / x) for x in index}

    #             # The constituent datasets should have unique keys.
    #             assert not (set(merged_index.keys()) & set(index.keys()))

    #             # Merge the root's index into the main index.
    #             merged_index = {**merged_index, **index}
    #     # print('-----------------merge_indices:', merged_index)
    #     return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())
