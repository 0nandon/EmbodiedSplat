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
from .view_sampler import ViewSampler, ViewSamplerEvaluation
from .dataset_re10k import DatasetRE10kCfg as DatasetScannetPPCfg
from ..model.encoder.gaussian_model.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from ..model.encoder.gaussian_model.dataset_utils import CameraInfo


import numpy as np
import os
from PIL import Image

from torch.utils.data import get_worker_info
from torch.distributed import init_process_group, get_rank, get_world_size


class DatasetScannetPP(Dataset):
    cfg: DatasetScannetPPCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]

    def __init__(
        self,
        cfg: DatasetScannetPPCfg,
        stage: Stage,
        view_sampler: ViewSampler,
        dense: int = 0,
        test_bev: bool = False,
        colmap: bool = False,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.dense = dense
        self.test_bev = test_bev
        self.colmap = colmap
        self.to_tensor = tf.ToTensor()
        self.coverage = {}
        self.interval = {}
        try:
            self.random = self.view_sampler.cfg.random
        except:
            self.random = False
            
        if self.random:
            self.n_views = np.random.randint(self.cfg.sample_num_start, self.cfg.sample_num_end)
        else:
            self.n_views = None

        # Collect chunks.
        self.chunks = []

        print('-'*20 + f'data root: {cfg.roots[0]}')
        if self.data_stage not in ['test', 'test_fvs']:
            for root in cfg.roots:
                root = root / 'data'
                root_chunks = sorted(
                    [path for path in root.iterdir()]
                )
                self.chunks.extend(root_chunks)
        else:
            root = cfg.roots[0] / 'data'
            self.chunks = sorted(
                    [root / path for path in self.index]
                )

        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)
        
        self.png_depth_scale = 1000.0

        # Dictionaries to store the data for each scene
        self.color_paths = []
        self.depth_paths = []
        self.intrinsics = []
        self.c2ws = []

        # Fetch the sequences to use
        if self.data_stage == "train":
            sequence_file = os.path.join(root, "../splits", "nvs_sem_train.txt")
            bad_scenes = ['303745abc7']
        elif self.data_stage == "val" or self.data_stage == "test":
            sequence_file = os.path.join(root, "../splits", "nvs_sem_val.txt")
            bad_scenes = ['cc5237fd77']
        with open(sequence_file, "r") as f:
            sequences = f.read().splitlines()
    
        sequences = [s for s in sequences if s not in bad_scenes]

        self.chunks = [chunk for chunk in self.chunks if str(chunk).split('/')[-1].split('_')[0] in sequences]

        print(f'Stage {self.data_stage}, length: {len(self.chunks)}')

        P = np.array([
            [1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, 1]]
        ).astype(np.float32)
        if self.colmap:
            rot = np.array([
                    [np.cos(np.pi/2), -np.sin(np.pi/2), 0, 0],
                    [np.sin(np.pi/2), np.cos(np.pi/2), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                ])
            P = rot @ P

            # 90-degree rotation matrix around the Z-axis
            R_z_90 = np.array([[0, -1, 0],
                            [1,  0, 0],
                            [0,  0, 1]])

        # Collect information for every sequence
        scenes_with_no_good_frames = []
        # for sequence in self.sequences:
        for sequence in self.chunks:
            input_processed_folder = sequence

            if self.data_stage in ['test']:
                input_processed_folder = str(input_processed_folder)[:-2]

            scene = str(sequence).split('/')[-1]

            # Load Train & Test Splits
            frame_file = os.path.join(input_processed_folder, "dslr", "train_test_lists.json")
            with open(frame_file, "r") as f:
                train_test_list = json.load(f)

            # Camera Metadata
            cams_metadata_path = f"{input_processed_folder}/dslr/nerfstudio/transforms_undistorted.json"
            with open(cams_metadata_path, "r") as f:
                cams_metadata = json.load(f)
            
            undistort_cams_metadata_path = f"{input_processed_folder}/dslr/nerfstudio/transforms_undistorted.json"
            with open(undistort_cams_metadata_path, "r") as f:
                undistort_cams_metadata = json.load(f)

            # Load the nerfstudio/transforms.json file to check whether each image is blurry
            nerfstudio_transforms_path = f"{input_processed_folder}/dslr/nerfstudio/transforms.json"
            with open(nerfstudio_transforms_path, "r") as f:
                nerfstudio_transforms = json.load(f)

            # Create a reverse mapping from image name to the frame information and nerfstudio transform
            # (as transforms_undistorted.json does not store the frames in the same order as train_test_lists.json)
            file_path_to_frame_metadata = {}
            file_path_to_nerfstudio_transform = {}
            for frame in cams_metadata["frames"]:
                file_path_to_frame_metadata[frame["file_path"].split('/')[-1]] = frame
            for frame in cams_metadata["test_frames"]:
                file_path_to_frame_metadata[frame["file_path"].split('/')[-1]] = frame
            for frame in nerfstudio_transforms["frames"]:
                file_path_to_nerfstudio_transform[frame["file_path"]] = frame

            # Fetch the pose for every frame
            sequence_color_paths = []
            sequence_depth_paths = []
            sequence_c2ws = []
            for train_file_name in train_test_list["train"]:
                is_bad = file_path_to_nerfstudio_transform[train_file_name]["is_bad"]
                if is_bad:
                    continue
                sequence_color_paths.append(f"{input_processed_folder}/dslr/undistorted_images/{train_file_name}")
                sequence_depth_paths.append(f"{input_processed_folder}/dslr/undistorted_depths/{train_file_name.replace('.JPG', '.png')}")
                frame_metadata = file_path_to_frame_metadata[train_file_name]
                c2w = np.array(frame_metadata["transform_matrix"], dtype=np.float32)
                c2w = P @ c2w @ P.T
                if self.colmap:
                    c2w[:3, :3] = c2w[:3, :3] @ R_z_90
                sequence_c2ws.append(c2w)
            
            for test_file_name in train_test_list["test"]:
                sequence_color_paths.append(f"{input_processed_folder}/dslr/undistorted_images/{test_file_name}")
                sequence_depth_paths.append(f"{input_processed_folder}/dslr/undistorted_depths/{test_file_name.replace('.JPG', '.png')}")
                frame_metadata = file_path_to_frame_metadata[test_file_name]
                c2w = np.array(frame_metadata["transform_matrix"], dtype=np.float32)
                c2w = P @ c2w @ P.T
                if self.colmap:
                    c2w[:3, :3] = c2w[:3, :3] @ R_z_90
                sequence_c2ws.append(c2w)

            # Get the intrinsics data for the frame
            K = np.eye(4, dtype=np.float32)
            K[0, 0] = undistort_cams_metadata["fl_x"]
            K[1, 1] = undistort_cams_metadata["fl_y"]
            K[0, 2] = undistort_cams_metadata["cx"]
            K[1, 2] = undistort_cams_metadata["cy"]

            self.color_paths.append(sequence_color_paths)
            self.depth_paths.append(sequence_depth_paths)
            self.c2ws.append(sequence_c2ws)
            self.intrinsics.append(K)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    # def __iter__(self):
    def __getitem__(self, scene_idx):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        path = self.chunks[scene_idx]
        
        scene = str(path).split('/')[-1]
        scene_raw = scene.split('_')[0]
        if self.data_stage in ['test']:
            path = str(path)[:-2]

        filenames = os.listdir(os.path.join(path, 'dslr', 'undistorted_images'))
        imshape = self.to_tensor(Image.open(os.path.join(path, 'dslr', 'undistorted_images', filenames[0]))).shape
        extrinsics = torch.from_numpy(np.array(self.c2ws[scene_idx])).float()
        intrinsics = torch.from_numpy(np.array(self.intrinsics[scene_idx])[None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
        
        if scene_raw not in self.coverage:
            coverage_path = os.path.join(self.cfg.roots[0], 'coverage', scene_raw+'.json')
            with open(coverage_path, 'r') as f:
                coverage = json.load(f)
            self.coverage[scene_raw] = np.array(coverage[list(coverage.keys())[0]])
            interval_path = os.path.join(self.cfg.roots[0], 'interval', scene_raw+'.txt')
            with open(interval_path, 'r') as f:
                interval = f.read().splitlines()
            self.interval[scene_raw] = np.array([int(x) for x in interval])
            
        if os.path.basename(path) in ["59e3f1ea37", "07f5b601ee"]:
            self.n_views = np.random.randint(7, 9)
                
        context_indices, target_indices, fvs_length = self.view_sampler.sample(
            scene,
            extrinsics,
            intrinsics,
            phase = self.cfg.phase,
            test_fvs=self.data_stage == 'test_fvs',
            path=path,
            n_views=self.n_views,
            coverage=self.coverage[scene_raw],
            interval=self.interval[scene_raw],
            )
        
        test_fvs = fvs_length > 0
        intrinsics[:, :1] /= imshape[2]
        intrinsics[:, 1:2] /= imshape[1]
        
        depth_imshape = self.to_tensor(Image.open(os.path.join(path, 'dslr', 'undistorted_depths', filenames[0].replace('.JPG', '.png')))).shape
        depth_intrinsics = intrinsics.clone()
        depth_intrinsics[:, :1] /= depth_imshape[2]
        depth_intrinsics[:, 1:2] /= depth_imshape[1]
        
        # prepare the cropped intrinsic for ft
        h_in, w_in = imshape[1:]
        h_out, w_out = self.cfg.image_shape
        scale_factor = max(1.015*h_out / h_in, 1.015*w_out / w_in)
        h_scaled = round(h_in * scale_factor)
        w_scaled = round(w_in * scale_factor)

        # Adjust the intrinsics to account for the cropping.
        new_intrinsics = intrinsics.clone() 
        new_intrinsics[..., 0, 0] *= w_scaled / w_out  # fx
        new_intrinsics[..., 1, 1] *= h_scaled / h_out  # fy

        fovx = focal2fov(new_intrinsics[0][0, 0], new_intrinsics[0][0, 2] * 2)
        fovy = focal2fov(new_intrinsics[0][1, 1], new_intrinsics[0][1, 2] * 2)
        cams = []        

        example = {'context': [], 'scene': scene}
        if self.dense:
            example['source'] = {'image': [], 'extrinsics': [], 'intrinsics': [], 'index': []}
        scale = 1
        length = len(filenames)
        for context_index in context_indices:
            context_images = []
            context_depths = []
            context_caches = []

            for idx in context_index:
                img = Image.open(self.color_paths[scene_idx][idx])
                img = self.to_tensor(img.resize((640, 480)))
                context_images.append(img[None])
        
                cache = os.path.join(
                    "dataset/scannetpp/train", 
                    os.path.basename(path), 
                    self.cfg.cache,
                    os.path.basename(self.color_paths[scene_idx][idx])[:-4] +'.pt'
                )
                context_caches.append(cache)
                
                img = Image.open(self.depth_paths[scene_idx][idx])
                img = (np.asarray(img.resize((640, 480))) / 1000).astype(np.float16)
                img = self.to_tensor(img)
                context_depths.append(img[None])

                w2c = np.linalg.inv(extrinsics[idx.numpy()])
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]

                cams.append(
                    CameraInfo(
                        uid=idx,
                        R=R,
                        T=T,
                        FovY=fovy,
                        FovX=fovx,
                        image_path=self.color_paths[scene_idx][idx],
                        image_name=str(idx.numpy())+'.jpg',
                        width=512,
                        height=384,
                        intrinsics=intrinsics[idx.numpy()],
                    )
                )
            
            context_images = torch.cat(context_images)
            context_depths = torch.cat(context_depths)
            content = {"extrinsics": extrinsics[context_index],
                        "intrinsics": intrinsics[context_index],
                        "image": context_images,
                        "cache": context_caches,
                        "near": self.get_bound("near", len(context_index)) / scale,
                        "far": self.get_bound("far", len(context_index)) / scale,
                        "index": context_index,
                        "cams": cams,
                        }
            if self.cfg.use_depth:
                content['depth'] = context_depths
                content['depth_intrinsics'] = depth_intrinsics[context_indices]
            example['context'].append(content)
        target_images = []

        if not test_fvs:
            for idx in target_indices:
                img = Image.open(self.color_paths[scene_idx][idx])
                img = self.to_tensor(img.resize((640, 480)))
                target_images.append(img[None])
        else:
            length = len(target_indices)
            for idx in target_indices[:length-fvs_length]:
                img = Image.open(self.color_paths[scene_idx][idx])
                img = self.to_tensor(img.resize((640, 480)))
                target_images.append(img[None])
            for idx in target_indices[length-fvs_length:]:
                img = Image.open(self.color_paths[scene_idx][idx])
                img = self.to_tensor(img.resize((640, 480)))
                target_images.append(img[None])
        
        target_images = torch.cat(target_images)

        if not test_fvs:
            example["target"] = {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / scale,
                    "far": self.get_bound("far", len(target_indices)) / scale,
                    "index": target_indices,
                    "test_fvs": False,
                }
        else:
            length = len(target_indices)
            example["target"] = {
                    "extrinsics": torch.cat([extrinsics[target_indices[:length-fvs_length]],
                                             extrinsics[target_indices[length-fvs_length:]]]),
                    "intrinsics": intrinsics[torch.zeros_like(target_indices, device=target_indices.device)],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / scale,
                    "far": self.get_bound("far", len(target_indices)) / scale,
                    "index": target_indices,
                    "test_fvs": fvs_length,
                }
        
        if self.test_bev:
            example['target']['bev_extrinsics'] = torch.from_numpy(np.load(os.path.join(path, 'bev_extrinsics.npy'))).float()[None]

        if self.cfg.use_depth:
            target_depths = []
            for idx in target_indices:
                img = Image.open(self.depth_paths[scene_idx][idx])
                img = (np.asarray(img.resize((640, 480))) / 1000).astype(np.float16)
                img = self.to_tensor(img)
                target_depths.append(img[None])

            target_depths = torch.cat(target_depths)
            example['target']['depth'] = target_depths
            example['target']['depth_intrinsics'] = depth_intrinsics[context_indices]

        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
        example = apply_crop_shim(example, tuple(self.cfg.image_shape))
        
        if self.random:
            self.n_views = np.random.randint(self.cfg.sample_num_start, self.cfg.sample_num_end)

        return example

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(BytesIO(image.numpy().tobytes()))
            torch_images.append(self.to_tensor(image))

        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self.cfg, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        
        merged_index = {}
        for data_stage in data_stages:
            data_stage = 'val' if data_stage == 'test' else data_stage
            for root in self.cfg.roots:
                # Load the root's index.
                with open(root / 'splits' / f'nvs_sem_{data_stage}.txt', 'r') as f:
                    index = f.read().split('\n')
                try:
                    index.remove('')
                except:
                    pass
                index = {x: Path(root / data_stage / x) for x in index}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}

        if isinstance(self.view_sampler, ViewSamplerEvaluation):
            merged_index = {k: v for k, v in self.view_sampler.index.items() if k[:-2] in merged_index}
        return merged_index

    def __len__(self) -> int:
        return len(self.index.keys())