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
from .dataset_re10k import DatasetRE10kCfg as DatasetScannetCfg
from ..model.encoder.gaussian_model.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from ..model.encoder.gaussian_model.dataset_utils import CameraInfo

import numpy as np
import os
from PIL import Image

from torch.utils.data import get_worker_info
from torch.distributed import init_process_group, get_rank, get_world_size


class DatasetScannet(Dataset):
    cfg: DatasetScannetCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]

    def __init__(
        self,
        cfg: DatasetScannetCfg,
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
        self.to_tensor = tf.ToTensor()
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
                root = root / self.data_stage
                root_chunks = sorted(
                    [path for path in root.iterdir()]
                )
                self.chunks.extend(root_chunks)
        else:
            # print('evaluation index:', self.index)
            root = cfg.roots[0] / self.data_stage
            self.chunks = sorted(
                    [root / path for path in self.index]
                )
            # print('self.chunks:', self.chunks)
        if self.cfg.overfit_to_scene is not None:
            chunk_path = self.index[self.cfg.overfit_to_scene]
            self.chunks = [chunk_path] * len(self.chunks)

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __getitem__(self, idx):
        path = self.chunks[idx]
        
        scene = str(path).split('/')[-1]
        if self.data_stage in ['test']:
            path = str(path)[:-2]

        imshape = self.to_tensor(Image.open(os.path.join(path, 'color', '0.jpg'))).shape
        extrinsics = torch.from_numpy(np.load(os.path.join(path, 'extrinsics.npy'))).float()
        intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_color.txt'))\
                                    [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
        context_indices, target_indices, fvs_length = self.view_sampler.sample(
            scene,
            extrinsics,
            intrinsics,
            phase = self.cfg.phase,
            test_fvs=self.data_stage == 'test_fvs',
            path=path,
            n_views=self.n_views,
        )
        test_fvs = fvs_length > 0
        
        intrinsics[:, :1] /= imshape[2]
        intrinsics[:, 1:2] /= imshape[1]

        depth_imshape = self.to_tensor(Image.open(os.path.join(path, 'depth', '0.png'))).shape
        depth_intrinsics = torch.from_numpy(np.loadtxt(os.path.join(path, 'intrinsic', 'intrinsic_depth.txt'))\
                                    [None,:3,:3].repeat(extrinsics.shape[0], 0)).float()
        depth_intrinsics[:, :1] /= depth_imshape[2]
        depth_intrinsics[:, 1:2] /= depth_imshape[1]

        h_in, w_in = imshape[1:]
        h_out, w_out = self.cfg.image_shape
        scale_factor = max(1.015*h_out / h_in, 1.015*w_out / w_in)
        h_scaled = round(h_in * scale_factor)
        w_scaled = round(w_in * scale_factor)

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
        length = len(os.listdir(os.path.join(path, 'color')))
        for context_index in context_indices:
            context_images = []
            context_depths = []
            context_caches = []

            for idx in context_index:             
                org_img = Image.open(os.path.join(path, 'color', str(idx.numpy())+'.jpg'))
                img = self.to_tensor(org_img.resize((640, 480)))
                context_images.append(img[None])
                
                if 'train' in str(path):
                    cache = os.path.join(
                        self.cfg.roots[0] / "train",
                        os.path.basename(path), 
                        self.cfg.cache,
                        str(idx.numpy())+'.pt'
                    )
                else:
                    cache = os.path.join(
                        self.cfg.roots[0] / "test", 
                        os.path.basename(path), 
                        self.cfg.cache,
                        str(idx.numpy())+'.pt'
                    )
                    
                context_caches.append(cache)

                img = Image.open(os.path.join(path, 'depth', str(idx.numpy())+'.png'))
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
                        image_path=os.path.join(path, 'color', str(idx.numpy())+'.jpg'),
                        image_name=str(idx.numpy())+'.jpg',
                        width=512,
                        height=384,
                        intrinsics=intrinsics[idx.numpy()],
                    )
                )

                if self.dense:
                    idx = idx.numpy()
                    if idx < self.dense:
                        start = 0
                        end = start + 2*self.dense + 1
                    elif idx >= length-self.dense:
                        start = length - 2*self.dense - 1
                        end = length
                    else:
                        start = idx-self.dense
                        end = idx + self.dense + 1
                    src_range = np.arange(start, end)
                    src_range = torch.from_numpy(src_range[src_range != idx])
                    for src_idx in src_range:
                        img = Image.open(os.path.join(path, 'color', str(src_idx.numpy())+'.jpg'))
                        img = self.to_tensor(img.resize((640, 480)))
                        example['source']['image'].append(img[None])
                    example['source']['extrinsics'].append(extrinsics[src_range])
                    example['source']['intrinsics'].append(intrinsics[src_range])
                    example['source']['index'].append(src_range)
            
            if self.dense:
                example['source']['image'] = torch.cat(example['source']['image'])
                example['source']['extrinsics'] = torch.cat(example['source']['extrinsics'])
                example['source']['intrinsics'] = torch.cat(example['source']['intrinsics'])
                example['source']['index'] = torch.cat(example['source']['index'])

            context_images = torch.cat(context_images)
            context_depths = torch.cat(context_depths)

            content = {
                "extrinsics": extrinsics[context_index],
                "intrinsics": intrinsics[context_index],
                "image": context_images,
                "cache": context_caches,
                "near": self.get_bound("near", len(context_index)) / scale,
                "far": self.get_bound("far", len(context_index)) / scale,
                "index": context_index,
                "cams": cams
            }
            if self.cfg.use_depth:
                content['depth'] = context_depths
                content['depth_intrinsics'] = depth_intrinsics[context_indices]
            example['context'].append(content) 
            
        target_images = []
        if not test_fvs:
            for idx in target_indices:
                img = Image.open(os.path.join(path, 'color', str(idx.numpy())+'.jpg'))
                img = self.to_tensor(img.resize((640, 480)))
                target_images.append(img[None])
        else:
            length = len(target_indices)
            for idx in target_indices[:length-fvs_length]:
                img = Image.open(os.path.join(path, 'color', str(idx.numpy())+'.jpg'))
                img = self.to_tensor(img.resize((640, 480)))
                target_images.append(img[None])
            sign = int(path[-1])
            for idx in target_indices[length-fvs_length:]:
                img = Image.open(os.path.join(str(path), 'color', str(idx.numpy())+'.jpg'))
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
            x = torch.from_numpy(np.load(os.path.join(str(path), 'extrinsics.npy'))).float()
            example["target"] = {
                    "extrinsics": torch.cat([extrinsics[target_indices[:length-fvs_length]],
                                             x[target_indices[length-fvs_length:]]]),
                    "intrinsics": intrinsics[torch.zeros_like(target_indices, device=target_indices.device)],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / scale,
                    "far": self.get_bound("far", len(target_indices)) / scale,
                    "index": target_indices,
                    "test_fvs": fvs_length,
                }

        if self.test_bev:
            example['target']['bev_extrinsics'] = torch.from_numpy(np.load("dataset/scannet/test/scene0000_01/bev_extrinsics.npy")).float()[None]

        if self.cfg.use_depth:
            target_depths = []
            for idx in target_indices:
                img = Image.open(os.path.join(path, 'depth', str(idx.numpy())+'.png'))
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
        # print('-----------------torch_images.shape:', torch_images.shape)
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self.cfg, bound), dtype=torch.float32)
        # print('value:', value)
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
            for root in self.cfg.roots:
                # Load the root's index.
                # with (root / data_stage / "index.json").open("r") as f:
                #     index = json.load(f)
                with open(root / f'{data_stage}_idx.txt', 'r') as f:
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
        # print('-----------------merge_indices:', merged_index.keys())
        # print('-----------------self.view_sampler.index:', self.view_sampler.index.keys())
        if isinstance(self.view_sampler, ViewSamplerEvaluation):
            merged_index = {k: v for k, v in self.view_sampler.index.items() if k[:-2] in merged_index}
        return merged_index

    def __len__(self) -> int:
        return len(self.index.keys())
