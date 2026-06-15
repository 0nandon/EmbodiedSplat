from dataclasses import dataclass
from typing import Literal, Optional, List

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from pathlib import Path

from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone, BackboneMultiview
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg
from .common.gaussian_adapter import Gaussians as G
from .encoder import Encoder
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .epipolar.depth_predictor_monocular import DepthPredictorMonocular
from .epipolar.epipolar_transformer import EpipolarTransformer, EpipolarTransformerCfg
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg
from ...global_cfg import get_cfg
import os
from PIL import Image
import matplotlib as mpl
import time
from .gaussian_model.gaussian_model import GaussianModel
from .gaussian_model.loss_utils import l1_loss, ssim
from .gaussian_model.renderer import render
from .gaussian_model.dataset_utils import getNerfppNorm
from .gaussian_model.camera_utils import loadCam
from .gaussian_model.dataset_utils import CameraInfo
from .gaussian_model.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from ...dataset.shims.crop_shim import rescale_and_crop
# from ..model_wrapper import convert_array_to_pil

from .common.keypoint_scorer import FeatureScorer, ContextScorer
from .adapter.cascade_gaussian_adapter import CascadeGaussianAdapter
from .adapter.gaussian_refiner import IterativeGaussianRefiner

from modules.networks import CVEncoder, DepthDecoderPP, ResnetMatchingEncoder
from modules.cost_volume import FeatureVolumeManager, CostVolumeManager, AttentionVolumeManager, AVGFeatureVolumeManager
from sr_utils.generic_utils import (reverse_imagenet_normalize, tensor_B_to_bM,
                                 tensor_bM_to_B)
import timm
from modules.layers import TensorFormatter
import mmcv
import torchvision.transforms as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from .attention.transformer import LocalFeatureTransformer, GRU2D_naive_Wweights

from einops import *
import matplotlib.cm as cm
from PIL import Image, ImageFont, ImageDraw

from ..timer import CudaTimer
from .encoder_config import EncoderEpipolarCfg
from ...third_party.fastsam import FastSAM, FastSAMPrompt
from ...third_party.open_clip_network import OpenCLIPNetwork, OpenCLIPNetworkConfig
from ...third_party.maskadapter.mask_adapter_head import MASKAdapterHead as MaskAdapter
from ...third_party.maskadapter.clip import CLIP

from .backbone.mink_unet import mink_unet
import MinkowskiEngine as ME
from .backbone.multilevel_memory import MultilevelMemory

import tensorflow as tff2
import tensorflow.compat.v1 as tff

import math

def convert_array_to_pil(depth_map, no_text=False):
    # Input: depth_map -> HxW numpy array with depth values 
    # Output: colormapped_im -> HxW numpy array with colorcoded depth values
    mask = depth_map!=0
    disp_map = 1/depth_map
    vmax = np.percentile(disp_map[mask], 95)
    vmin = np.percentile(disp_map[mask], 5)
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    mask_ = np.repeat(np.expand_dims(mask,-1), 3, -1)
    colormapped_im = (mapper.to_rgba(disp_map)[:, :, :3] * 255).astype(np.uint8)
    colormapped_im[~mask_] = 255
    min_depth, max_depth = depth_map[mask].min(), depth_map[mask].max()
    image = Image.fromarray(colormapped_im)
    if not no_text:
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 40)
        draw.text((20,20), '[%.2f, %.2f]'%(min_depth, max_depth), (255,255,255), font=font)
        colormapped_im = np.asarray(image)

    return colormapped_im

def concatenate_tensors(attributes):
    """Helper function to concatenate tensors in batches to manage memory usage."""
    batch_size = 10  # You can adjust this batch size based on your available GPU memory
    concatenated = []
    for i in range(0, len(attributes), batch_size):
        batch = attributes[i:i + batch_size]
        if batch:
            # Concatenate the current batch and immediately rearrange to reduce peak memory usage
            concatenated_batch = torch.cat([
                rearrange(
                    gs,
                    "b v r srf spp xyz -> b (v r srf spp) xyz" if 'xyz' in gs.shape else "b v r srf spp i j -> b (v r srf spp) i j" if 'i' in gs.shape else "b v r srf spp c d_sh -> b (v r srf spp) c d_sh"
                ) for gs in batch
            ], dim=1)
            concatenated.append(concatenated_batch)
    # Final concatenation of all batches
    return torch.cat(concatenated, dim=1)

def transform_points(points, transform):
    """ Applies a 4x4 transformation matrix to a set of points. """
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])  # Convert to homogeneous coordinates
    transformed_points = points_homogeneous @ transform.T  # Apply transformation
    return transformed_points[:, :3]  # Return only XYZ

def create_transformed_pyramid(extrinsic_matrix, scale=0.3, num_points = 100):
    # Define vertices of the pyramid in local coordinates
    apex = np.array([[0, 0, -1]]) * scale  # Apex point
    base = np.array([  # Base points
        [-1, -1, 0],
        [1, -1, 0],
        [1, 1, 0],
        [-1, 1, 0]
    ]) * scale

    # Transform vertices using the extrinsic matrix
    all_points = np.vstack([apex, base])
    transformed_points = transform_points(all_points, extrinsic_matrix)

    # Generate points along the edges
    edges_points = [] # Number of points per edge for visualization
    for i in range(1, 5):
        edges_points.append(np.linspace(transformed_points[0], transformed_points[i], num_points))

    for i in range(1, 5):
        edges_points.append(np.linspace(transformed_points[i], transformed_points[i % 4 + 1], num_points))

    # Flatten the list and convert to Open3D point cloud
    points = np.vstack(edges_points)
    return points.astype(np.float32)

class EfficientGaussians:
    def __init__(
            self, 
            initial_capacity=1000, 
            growth_factor=2.0, 
            feat_dim=64, 
            device='cuda', 
            testing=True,
            export_ply=False
        ):

        self.capacity = initial_capacity if testing else 0
        self.growth_factor = growth_factor
        self.device = device
        self.testing = testing
        self.size = 0

        # Initial allocations with guessed sizes
        self.means = torch.zeros((1, self.capacity, 3), device=device)  # Adjust the dimensionality as needed
        self.covariances = torch.zeros((1, self.capacity, 3, 3), device=device)  # Example shape
        self.harmonics = torch.zeros((1, self.capacity, 3, 9), device=device)  # Adjust based on actual shape
        self.opacities = torch.zeros((1, self.capacity), device=device)
        self.features = torch.zeros((1, self.capacity, feat_dim), device=device)  # Adjust based on actual shape

        self.clip_features_3d = torch.zeros((1, self.capacity, feat_dim), device=device)
        self.clip_features_2d = {
            "feat": [],
            "idx": torch.ones((1, self.capacity, 6), device=device) * -1,
            "weight": torch.zeros((1, self.capacity, 6), device=device),
            "invalid": torch.zeros((1, self.capacity), device=device),
            "img_stamp": [0]
        }

        self.coords = torch.zeros((1, self.capacity, 3), device=device)  # Adjust based on actual shape
        self.densities = torch.zeros((1, self.capacity, 1, 1), device=device)  # Adjust based on actual shape
        self.weights = torch.zeros((1, self.capacity, 1, 1), device=device)  # Adjust based on actual shape
        self.extrinsics = torch.zeros((1, self.capacity, 4, 4), device=device)  # Adjust based on actual shape
        self.depths = torch.zeros((1, self.capacity), device=device)  # Adjust based on actual shape
        self.valid = torch.zeros((1, self.capacity), dtype=torch.bool, device=device)  # Track valid entries

        if export_ply:
            self.scales = torch.zeros((1, self.capacity, 3), device=device)  # Example shape
            self.rotations = torch.zeros((1, self.capacity, 4), device=device)  # Example shape
        
        self.export_ply = export_ply

    def append(
        self,
        means, 
        covariances, 
        harmonics, 
        opacities, 
        features,
        clip_features_3d,
        coords, 
        densities, 
        weights, 
        extrinsics, 
        depths,
        codebooks_idx,
        codebooks_weights,
        mask=None, 
        scales=None, 
        rotations=None,
        features_invalid=None,
        fuse=False,
        refine=False
    ):
        if self.testing:
            if mask is not None:
                columns = self.valid[0].nonzero(as_tuple=True)[0][mask]
                self.valid[:, columns] = False

            invalid_indices = (~self.valid).nonzero(as_tuple=True)[1]
            num_new = means.shape[1]
            self.size = self.size + num_new

            if num_new > invalid_indices.shape[0]:  # If more new items than invalid slots
                needed_capacity = num_new - invalid_indices.shape[0]
                self._expand_storage(max(self.capacity + needed_capacity, int(self.capacity * self.growth_factor)))
                invalid_indices = (~self.valid).nonzero(as_tuple=True)[1]

            # Assuming new_gaussians is an instance of a class with similar attributes
            self.means[0, invalid_indices[:num_new]] = means
            self.covariances[0, invalid_indices[:num_new]] = covariances
            self.harmonics[0, invalid_indices[:num_new]] = harmonics
            self.opacities[0, invalid_indices[:num_new]] = opacities
            self.coords[0, invalid_indices[:num_new]] = coords
            self.densities[0, invalid_indices[:num_new]] = densities
            
            if features is not None:
                self.features[0, invalid_indices[:num_new]] = features
                self.clip_features_3d[0, invalid_indices[:num_new]] = clip_features_3d
                
                self.weights[0, invalid_indices[:num_new]] = weights
                self.extrinsics[0, invalid_indices[:num_new]] = extrinsics
                self.depths[0, invalid_indices[:num_new]] = depths

                size = (self.clip_features_2d["idx"] != -1).sum(dim=-1)[0][invalid_indices[:num_new]]
                filter = size == 6
                size[filter] -= 1

                if fuse:
                    self.clip_features_2d["idx"][0, invalid_indices[:num_new], size] = codebooks_idx
                    self.clip_features_2d["weight"][0, invalid_indices[:num_new]] *= (1 - codebooks_weights)[0, :, None]
                    self.clip_features_2d["weight"][0, invalid_indices[:num_new], size] = codebooks_weights
                else:
                    self.clip_features_2d["idx"][0, invalid_indices[:num_new], size] = codebooks_idx
                    self.clip_features_2d["weight"][0, invalid_indices[:num_new], size] = codebooks_weights
                    
                if (size == 5).sum() > 0:
                    sort, sort_idx = self.clip_features_2d["weight"][0, invalid_indices[:num_new]].sort(dim=-1, descending=True)
                    self.clip_features_2d["weight"][0, invalid_indices[:num_new]] = sort
                    self.clip_features_2d["idx"][0, invalid_indices[:num_new]] = torch.gather(
                        self.clip_features_2d["idx"][0, invalid_indices[:num_new]],
                        index=sort_idx,
                        dim=-1
                    )

            if features_invalid is not None:
                self.clip_features_2d["invalid"][0, invalid_indices[:num_new]] = features_invalid
                
            if scales is not None:
                self.scales[0, invalid_indices[:num_new]] = scales
            
            if rotations is not None:
                self.rotations[0, invalid_indices[:num_new]] = rotations

            self.valid[:, invalid_indices[:num_new]] = True
        else:
            remain_mask = ~mask if mask is not None else torch.ones([self.means.shape[1]], dtype=torch.bool, device=self.means.device)
            valid_tensor = torch.ones((1, means.shape[1]), dtype=torch.bool, device=means.device)
            self.means = torch.cat([self.means[:, remain_mask], means], dim=1)
            self.covariances = torch.cat([self.covariances[:, remain_mask], covariances], dim=1)
            self.harmonics = torch.cat([self.harmonics[:, remain_mask], harmonics], dim=1)
            self.opacities = torch.cat([self.opacities[:, remain_mask], opacities], dim=1)
            self.coords = torch.cat([self.coords[:, remain_mask], coords], dim=1)
            self.densities = torch.cat([self.densities[:, remain_mask], densities], dim=1)
            self.valid = torch.cat([self.valid[:, remain_mask], valid_tensor], dim=1)
            self.clip_features_3d = torch.cat([self.clip_features_3d[:, remain_mask], clip_features_3d], dim=1)
                
            if fuse:
                clip_features_2d_idx = self.clip_features_2d["idx"][:, ~remain_mask]
                clip_features_2d_weight = self.clip_features_2d["weight"][:, ~remain_mask]
                
                size = (clip_features_2d_idx != -1).sum(dim=-1)[0]
                filter = size == 6
                size[filter] -= 1

                clip_features_2d_weight *= (1 - codebooks_weights)[:, :, None]
                clip_features_2d_weight[0, torch.arange(size.shape[0]), size] = codebooks_weights
                clip_features_2d_idx[0, torch.arange(size.shape[0]), size] = codebooks_idx
                
                if (size == 5).sum() > 0:
                    sort, sort_idx = clip_features_2d_weight.sort(dim=-1, descending=True)
                        
                    clip_features_2d_weight = sort
                    clip_features_2d_idx = torch.gather(
                        clip_features_2d_idx,
                        index=sort_idx,
                        dim=-1
                    )
            elif refine:
                clip_features_2d_idx = torch.ones(1, means.shape[1], 6).to(self.device) * -1
                clip_features_2d_weight = torch.zeros(1, means.shape[1], 6).to(self.device)
            else:
                clip_features_2d_idx = torch.ones(1, codebooks_idx.shape[-1], 6).to(self.device) * -1
                clip_features_2d_weight = torch.zeros(1, codebooks_idx.shape[-1], 6).to(self.device)
                
                clip_features_2d_idx[0, :, 0] = codebooks_idx[0]
                clip_features_2d_weight[0, :, 0] = codebooks_weights[0]
                    
            self.clip_features_2d["idx"] = torch.cat([self.clip_features_2d["idx"][:, remain_mask], clip_features_2d_idx], dim=1)
            self.clip_features_2d["weight"] = torch.cat([self.clip_features_2d["weight"][:, remain_mask], clip_features_2d_weight], dim=1)
            
            if features_invalid is not None:
                self.clip_features_2d["invalid"] = torch.cat([self.clip_features_2d["invalid"][:, remain_mask], features_invalid], dim=1)
            else:
                self.clip_features_2d["invalid"] = torch.cat([
                    self.clip_features_2d["invalid"][:, remain_mask], 
                    torch.zeros((1, coords.shape[1]), device=self.device)
                ], dim=1)
                
            if features is not None:   
                self.features = torch.cat([self.features[:, remain_mask], features], dim=1)
                self.weights = torch.cat([self.weights[:, remain_mask], weights], dim=1)
                self.extrinsics = torch.cat([self.extrinsics[:, remain_mask], extrinsics], dim=1)
                self.depths = torch.cat([self.depths[:, remain_mask], depths], dim=1)

    def _expand_storage(self, new_capacity):
        self.means = self._resize_tensor(self.means, new_capacity, device=self.device)
        self.covariances = self._resize_tensor(self.covariances, new_capacity, device=self.device)
        self.harmonics = self._resize_tensor(self.harmonics, new_capacity, device=self.device)
        self.opacities = self._resize_tensor(self.opacities, new_capacity, device=self.device)
        self.features = self._resize_tensor(self.features, new_capacity, device=self.device)
        self.clip_features_3d = self._resize_tensor(self.clip_features_3d, new_capacity, device=self.device)
        self.coords = self._resize_tensor(self.coords, new_capacity, device=self.device)
        self.densities = self._resize_tensor(self.densities, new_capacity, device=self.device)
        self.weights = self._resize_tensor(self.weights, new_capacity, device=self.device)
        self.extrinsics = self._resize_tensor(self.extrinsics, new_capacity, device=self.device)
        self.depths = self._resize_tensor(self.depths, new_capacity, device=self.device)
        self.valid = self._resize_tensor(self.valid, new_capacity, device=self.device, fill=False, dtype=torch.bool)
        
        self.clip_features_2d["idx"] = self._resize_tensor(self.clip_features_2d["idx"], new_capacity, fill=-1, device=self.device)
        self.clip_features_2d["weight"] = self._resize_tensor(self.clip_features_2d["weight"], new_capacity, device=self.device)
        self.clip_features_2d["invalid"] = self._resize_tensor(self.clip_features_2d["invalid"], new_capacity, device=self.device)
        
        if self.export_ply:
            self.scales = self._resize_tensor(self.scales, new_capacity, device=self.device)
            self.rotations = self._resize_tensor(self.rotations, new_capacity, device=self.device)
        
        self.capacity = new_capacity

    @staticmethod
    def _resize_tensor(tensor, new_capacity, fill=0, dtype=None, device='cuda'):
        old_size = tensor.size(1)
        dtype = tensor.dtype if dtype is None else dtype
        new_tensor = torch.full((1, new_capacity, *tensor.shape[2:]), fill, dtype=dtype, device=device)
        new_tensor[:, :old_size] = tensor
        return new_tensor

UseDepthMode = Literal[
    "depth"
]


def save_image(data, path):
    data = data.mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    image = Image.fromarray(data.transpose(1, 2, 0), 'RGB')

    # Save the image
    image.save(path)

def rotation_distance(rotations):
    R1 = rotations.unsqueeze(2) 
    R2 = rotations.unsqueeze(1) 
    R_rel = torch.matmul(R1.transpose(-2, -1), R2) 

    trace = torch.diagonal(R_rel, dim1=-2, dim2=-1).sum(-1) 
    trace = torch.clamp(trace, -1, 3)
    angle = torch.acos((trace - 1) / 2)
    return angle.squeeze(0) 

def calculate_distance_matrix(poses):
    translations = poses[:, :, :3, 3]
    rotations = poses[:, :, :3, :3]
    
    translation_dist = torch.cdist(translations, translations).squeeze(0)
    
    rotation_dist = rotation_distance(rotations)
    
    combined_dist = translation_dist + rotation_dist

    return combined_dist

def positional_encoding(positions, freqs, ori=False):
    '''encode positions with positional encoding
        positions: :math:`(...,D)`
        freqs: int
    Return:
        pts: :math:`(..., 2DF)`
    '''
    freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
    ori_c = positions.shape[-1]
    pts = (positions[..., None] * freq_bands).reshape(positions.shape[:-1] +
                                                      (freqs * positions.shape[-1], ))  # (..., DF)
    if ori:
        pts = torch.cat([positions, torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2+ori_c,))
    else:
        pts = torch.stack([torch.sin(pts), torch.cos(pts)], dim=-1).reshape(pts.shape[:-1]+(pts.shape[-1]*2,))
    return pts

# Function to safely update tensor with new values at specified indices
def safe_update_tensor(tensor, valid, indices, new_values):
    # Get the valid entries in the original tensor
    valid_tensor = tensor[:, valid]

    # Create a new tensor that can receive the new values
    new_tensor = valid_tensor.clone()

    # Assign the new values
    new_tensor[:, indices] = new_values

    # Reconstruct the original tensor by combining updated and non-updated parts
    updated_tensor = tensor.clone()
    updated_tensor[:, valid] = new_tensor

    return updated_tensor

class EmbodiedSplatEncoderTrainUnet3d_Single(Encoder[EncoderEpipolarCfg]):
    backbone: Backbone
    backbone_projection: nn.Sequential
    epipolar_transformer: EpipolarTransformer | None
    depth_predictor: DepthPredictorMonocular
    to_gaussians: nn.Sequential
    gaussian_adapter: GaussianAdapter
    high_resolution_skip: nn.Sequential

    def __init__(self, cfg: EncoderEpipolarCfg, depth_range=[0.5, 15.0]) -> None:
        super().__init__(cfg)
        activation_func = nn.ReLU()
        self.to_tensor = tf.ToTensor()
        
        self.depth_range = depth_range

        self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.backbone = timm.create_model(
            "tf_efficientnetv2_s_in21ft1k", 
            pretrained=True, 
            features_only=True,
        )

        self.backbone.num_ch_enc = self.backbone.feature_info.channels()

        for name, module in self.backbone.named_modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.momentum = 0.01  # A lower value than the typical 0.1

        if cfg.use_epipolar_transformer:
            self.epipolar_transformer = EpipolarTransformer(
                cfg.epipolar_transformer,
                self.backbone.feature_info.channels()[0],
                n_views = cfg.max_batch,
            )
        else:
            self.epipolar_transformer = None
            
        self.high_resolution_skip = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3+self.cfg.load_depth, 64, 7, 1, 3),
                activation_func,
            ),
            nn.Sequential(
                nn.Conv2d(3+self.cfg.load_depth, 64, 6, 2, 2),
                activation_func,
            ),
            nn.Sequential(
                nn.Conv2d(3+self.cfg.load_depth, 64, 8, 4, 2),
                activation_func,
            ),
            nn.Sequential(
                nn.Conv2d(3+self.cfg.load_depth, 64, 16, 8, 4),
                activation_func,
            ),
            nn.Sequential(
                nn.Conv2d(3+self.cfg.load_depth, 64, 32, 16, 8),
                activation_func,
            )
        ])

        self.to_gaussians = nn.Sequential(
            activation_func,
            nn.Linear(
                64,
                cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
            ),
        )
        
        self.gausisans_ch = cfg.num_surfaces * (2 + self.gaussian_adapter.d_in)
        
        self.load_depth = cfg.load_depth
        self.est_depth = cfg.est_depth
        
        if self.cfg.est_depth == 'cost':
            if not self.cfg.wo_cost_volume:
                if not self.cfg.wo_matchnet:
                    self.matching_net = ResnetMatchingEncoder(self.cfg.matchnet_type, self.cfg.matchnet_dim,)

                self.cost_volume = AVGFeatureVolumeManager(
                    matching_height=self.cfg.image_H//4, 
                    matching_width=self.cfg.image_W//4,
                    num_depth_bins=self.cfg.num_depth_candidates,
                    matching_dim_size=self.cfg.matchnet_dim if (not self.cfg.wo_matchnet) else 48,
                    num_source_views=self.cfg.num_views-1,
                    log_plane=self.cfg.log_cv,
                )
                
                if not self.cfg.wo_msd:
                    self.cv_encoder = CVEncoder(
                        num_ch_cv=self.cfg.num_depth_candidates,
                        num_ch_enc=self.backbone.num_ch_enc[1:],
                        num_ch_outs=[64, 128, 256, 384]
                    )
                    dec_num_input_ch = (
                        self.backbone.num_ch_enc[:1] + self.cv_encoder.num_ch_enc
                    )
                else:
                    dec_num_input_ch = self.backbone.num_ch_enc[:1] + \
                                        [self.backbone.num_ch_enc[1]+self.cfg.num_depth_candidates] +\
                                        self.backbone.num_ch_enc[2:]
            else:
                dec_num_input_ch = (self.backbone.num_ch_enc)

            self.depth_decoder = DepthDecoderPP(
                dec_num_input_ch, 
                num_output_channels=1+64,
                n_levels=self.cfg.n_levels,
                use_planes=self.cfg.use_planes,
                near=depth_range[0],
                far=depth_range[1],
                num_samples=self.cfg.num_depth_candidates,
                log_plane=self.cfg.log_plane,
                wo_msd=self.cfg.wo_msd,
                refine=self.cfg.depth_refine,
                num_context_views=self.cfg.num_views,
                low_res=self.cfg.low_res,
            )
            self.max_depth = 2 + 2 * (not self.cfg.wo_msd)
            self.tensor_formatter = TensorFormatter()

            if self.cfg.fusion:
                self.weight_embedding = nn.Sequential(
                    nn.Linear(2, 12), 
                    activation_func,
                    nn.Linear(12, 12),
                )
                self.gru = GRU2D_naive_Wweights(concat_depth=self.cfg.concat_depth)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.clip_dim = self.cfg.clip_dim
        
        self.to_semantic = nn.Sequential(
            activation_func,
            nn.Linear(
                64,
                self.clip_dim
            ),
        )

        if not self.cfg.load_cache:
            if self.cfg.clip_model == "clip":
                self.clip = OpenCLIPNetwork(OpenCLIPNetworkConfig)
                self.clip_process = tf.Compose(
                    [
                        tf.Normalize(
                            mean=[0.48145466, 0.4578275, 0.40821073],
                            std=[0.26862954, 0.26130258, 0.27577711],
                        ),
                    ]
                )
                self.clip_dim = 512
            elif self.cfg.clip_model == "openseg":
                saved_model_path = self.cfg.clip_model_path
                self.openseg = tff2.saved_model.load(saved_model_path, tags=[tff.saved_model.tag_constants.SERVING],)
                self.openseg_text_emb = tff.zeros([1, 1, 768])
                self.clip_dim = 768
            elif self.cfg.clip_model == "maskadapter":
                self.maskadapter = MaskAdapter().eval()
                self.clip = CLIP(model_name=self.cfg.clip_model_type, pretrained=self.cfg.clip_model_path).eval()
                
                self.PIXEL_MEAN = torch.tensor([122.7709383, 116.7460125, 104.09373615]).view(1, -1, 1, 1).to(self.device)
                self.PIXEL_STD = torch.tensor([68.5005327, 66.6321579, 70.32316305]).view(1, -1, 1, 1).to(self.device)

        if self.cfg.semantic_early_fusion:
            self.semantic_proj = nn.Sequential(
                nn.Conv2d(in_channels=self.clip_dim, out_channels=64, kernel_size=1, bias=False),
                activation_func,
            )

        if self.cfg.unet_3d:
            self.unet3d = mink_unet(
                in_channels=self.cfg.unet3d_in_channels, 
                out_channels=self.cfg.unet3d_out_channels,
                D=self.cfg.unet3d_D, 
                arch=self.cfg.unet3d_arch
            )
            
            if self.cfg.pcd_aug:
                self.SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
                self.ROTATION_AUGMENTATION_BOUND = (
                    (-np.pi / 64, np.pi / 64), (-np.pi / 64, np.pi / 64), (-np.pi, np.pi)
                )
                
                self.M_v, self.M_r = None, None

        if self.cfg.memory:
            self.memory = MultilevelMemory(
                in_channels=self.cfg.memory_in_channels,
                queue=self.cfg.memory_queue,
                vmp_layer=self.cfg.memory_vmp_layer,
                norm=self.cfg.memory_norm
            )
        else:
            self.memory = None

        if self.cfg.non_object_embedding:
            self.non_object_embedding = nn.Parameter(
                torch.empty(1, self.clip_dim)
            )
            torch.nn.init.kaiming_uniform_(self.non_object_embedding, a=math.sqrt(5))
            self.non_object_embedding.requires_grad = True

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def compute_matching_feats(
        self, 
        cur_image, 
        src_image, 
        unbatched_matching_encoder_forward=False,
    ):
        """ 
            Computes matching features for the current image (reference) and 
            source images.

            Unfortunately on this PyTorch branch we've noticed that the output 
            of our ResNet matching encoder is not numerically consistent when 
            batching. While this doesn't affect training (the changes are too 
            small), it does change and will affect test scores. To combat this 
            we disable batching through this module when testing and instead 
            loop through images to compute their feautures. This is stable and 
            produces exact repeatable results.

            Args:
                cur_image: image tensor of shape B3HW for the reference image.
                src_image: images tensor of shape BM3HW for the source images.
                unbatched_matching_encoder_forward: disable batching and loops 
                    through iamges to compute feaures.
            Returns:
                matching_cur_feats: tensor of matching features of size bchw for
                    the reference current image.
                matching_src_feats: tensor of matching features of size BMcHW 
                    for the source images.
        """
        
        if unbatched_matching_encoder_forward:
            all_frames_bm3hw = torch.cat([cur_image.unsqueeze(1), src_image], dim=1)
            batch_size, num_views = all_frames_bm3hw.shape[:2]
            all_frames_B3hw = tensor_bM_to_B(all_frames_bm3hw)
            matching_feats = [self.matching_net(f) 
                                    for f in all_frames_B3hw.split(1, dim=0)]

            matching_feats = torch.cat(matching_feats, dim=0)
            matching_feats = tensor_B_to_bM(
                                        matching_feats, 
                                        batch_size=batch_size, 
                                        num_views=num_views,
                                    )

        else:
            # Compute matching features and batch them to reduce variance from 
            # batchnorm when training.
            matching_feats = self.tensor_formatter(
                torch.cat([cur_image.unsqueeze(1), src_image], dim=1),
                apply_func=self.matching_net,
            )

        matching_cur_feats = matching_feats[:, 0]
        matching_src_feats = matching_feats[:, 1:].contiguous()

        return matching_cur_feats, matching_src_feats

    def forward(
        self,
        contexts,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
        is_testing: bool = False,
        export_ply: bool = False,
        dataset_name: str = 'scannet',
        decoder = None,
        source = None,
        dense : int = 0,
        scene = None,
        target_indices = None,
        output_path = None,
        target = None,
        test_bev = False,
        path = None,
    ) -> dict:     
        device = contexts[0]["image"].device
        b, n_views, _, h, w = contexts[0]["image"].shape
        results = {}
        num_context_views = self.cfg.num_views
        
        if self.cfg.est_depth == 'cost':
            our_gaussians = []
            gaussians = []
            coords = []
            results = {}
            
            if (dataset_name == 'scannet' or (not(is_testing)) or self.cfg.num_views > 2):
                self.backbone.train()
            else:
                print('freezing backbone...')

            num = 0
            context = contexts[num]
            context['image_shape'] = (h, w)
            self.cfg.gaussians_per_pixel = 1
            context_intrinsics = context['intrinsics'].clone()
            context_intrinsics[:,:,0] *= (w // 4)
            context_intrinsics[:,:,1] *= (h // 4)

            globals = EfficientGaussians(
                initial_capacity=n_views*10000, 
                growth_factor=1.5, 
                device=context['image'].device, 
                testing=is_testing
            )
        
            cur_indices = torch.arange(n_views, device=context['image'].device)
            cur_intrinsics = context_intrinsics.gather(dim=1, index=cur_indices.view(1,-1,1,1).repeat(b,1,3,3))
            cur_extrinsics = context['extrinsics'].gather(dim=1, index=cur_indices.view(1,-1,1,1).repeat(b,1,4,4))
            cur_image = context['image'].gather(dim=1, index=cur_indices.view(1,-1,1,1,1).repeat(b,1,3,h,w)).view(-1,3,h,w)
            
            cur_feats = self.backbone(cur_image)

            resized = 0
            s = -1 if not self.cfg.low_res else 0

            if not self.cfg.wo_cost_volume: # True
                if self.cfg.use_epipolar_transformer: # False
                    _, sampling = self.epipolar_transformer(
                        rearrange(cur_feats[0], "(b v) c h w -> b v c h w", b=b, v=n_views),
                        context["extrinsics"],
                        context["intrinsics"],
                        context["near"],
                        context["far"],
                    )
                
                full_indices = torch.arange(n_views, device=context['image'].device)[None].repeat(n_views,1)
                
                if not dense:
                    use_local = (n_views > num_context_views) # True
                    if not use_local:
                        src_indices = full_indices[~(full_indices == cur_indices[:,None])].view(1,n_views,n_views-1).repeat(b,1,1)
                    else:
                        slide_mask = torch.zeros((n_views, n_views), dtype=torch.bool, device=full_indices.device)
                        dist_matrix = calculate_distance_matrix(context["extrinsics"])

                        # For each row in the distance matrix, mark the closest 'num_context_views' entries as True
                        _, indices = torch.topk(dist_matrix, min(num_context_views, n_views), largest=False, sorted=False, dim=1)
                        slide_mask.scatter_(1, indices, True)
                        slide_mask[torch.arange(n_views), torch.arange(n_views)] = False    
                        src_indices = full_indices[(~(full_indices == cur_indices[:,None]))*slide_mask].view(1,n_views,min(n_views, num_context_views)-1).repeat(b,1,1)
                        
            bs = self.cfg.max_batch
            fusion_features = []
            depth_outputs_all = {f'depth_pred_s{s}_b1hw': [], 'densities': []}

            for batch in range(min(np.ceil(n_views/bs).astype(int), self.cfg.max_batch_length)):
                n_views_now = min(bs, n_views-batch*bs)
                if not self.cfg.wo_cost_volume: # True
                    matching_cur_feats = cur_feats[1][batch*bs:(batch+1)*bs]
                    dim = matching_cur_feats.shape[-3]
                    if not dense: # True
                        matching_src_feats = rearrange(cur_feats[1], "(b v) c h w -> b v c h w", b=b, v=n_views)[:,None].repeat(1,n_views_now,1,1,1,1).gather(dim=2, index=src_indices[:,batch*bs:(batch+1)*bs,:,None,None,None]\
                                            .repeat(1,1,1,dim,h//4,w//4)).view(-1,n_views-1 if not use_local else min(n_views, num_context_views)-1,dim,h//4,w//4)

                        src_extrinsics = context['extrinsics'][:,None].repeat(1,n_views_now,1,1,1).gather(dim=2, index=src_indices[:,batch*bs:(batch+1)*bs,:,None,None].repeat(1,1,1,4,4))
                        src_intrinsics = context_intrinsics[:,None].repeat(1,n_views_now,1,1,1).gather(dim=2, index=src_indices[:,batch*bs:(batch+1)*bs,:,None,None].repeat(1,1,1,3,3))
                    else:
                        src_images = rearrange(source['image'][:, batch*bs*2*dense:(batch+1)*bs*2*dense], 'b v c h w -> (b v) c h w')
                        matching_src_feats = self.backbone(src_images)[1].view(n_views_now, 2*dense, -1, h//4, w//4)
                        src_indices = source['index'][:, batch*bs*2*dense:(batch+1)*bs*2*dense].view(n_views_now, 2*dense, -1)
                        src_extrinsics = source['extrinsics'][:, batch*bs*2*dense:(batch+1)*bs*2*dense].view(1, n_views_now, 2*dense, 4, 4)
                        src_intrinsics = source['intrinsics'][:, batch*bs*2*dense:(batch+1)*bs*2*dense].view(1, n_views_now, 2*dense, 3, 3)
                        
                    src_cam_t_world = src_extrinsics.inverse()
                    cur_cam_t_world = cur_extrinsics[:, batch*bs:(batch+1)*bs].inverse()
                    src_cam_T_cur_cam = src_cam_t_world @ cur_extrinsics[:, batch*bs:(batch+1)*bs].unsqueeze(2)
                    cur_cam_T_src_cam = cur_cam_t_world.unsqueeze(2) @ src_extrinsics
                    
                    src_cam_T_cur_cam_ = rearrange(src_cam_T_cur_cam, 'b v n x y -> (b v) n x y')
                    cur_cam_T_src_cam_ = rearrange(cur_cam_T_src_cam, 'b v n x y -> (b v) n x y')
                    src_intrinsics_ = rearrange(src_intrinsics, 'b v n x y -> (b v) n x y')
                    cur_intrinsics_ = rearrange(cur_intrinsics[:, batch*bs:(batch+1)*bs], 'b v x y -> (b v) x y')
                    src_K = torch.eye(4, device=context['image'].device)[None,None].repeat(src_intrinsics_.shape[0], src_intrinsics_.shape[1],1,1)
                    src_K[:,:,:3,:3] = src_intrinsics_
                    cur_inverse = torch.eye(4, device=context['image'].device)[None].repeat(cur_intrinsics_.shape[0],1,1)
                    cur_inverse[:,:3,:3] = cur_intrinsics_.inverse()

                    results['cur_invK'] = cur_inverse
                    results['src_K'] = src_K
                    results['cur_wtc'] = cur_extrinsics[:, batch*bs:(batch+1)*bs]
                    results['cur_ctw'] = cur_cam_t_world
                    
                    near = context["near"][:1,0].type_as(src_K).view(1, 1, 1, 1)
                    far = context["far"][:1,0].type_as(src_K).view(1, 1, 1, 1)

                    cost_volume = self.cost_volume(
                        cur_feats=matching_cur_feats,
                        src_feats=matching_src_feats,
                        src_extrinsics=src_cam_T_cur_cam_,
                        src_poses=cur_cam_T_src_cam_,
                        src_Ks=src_K,
                        cur_invK=cur_inverse,
                        min_depth=near,
                        max_depth=far,
                    )
                    
                    cost_volume_features = self.cv_encoder(
                        cost_volume, 
                        [x[batch*bs:(batch+1)*bs] for x in cur_feats[1:]],
                    )

                    depth_outputs = self.depth_decoder([
                        cur_feats[0][batch*bs:(batch+1)*bs]] + cost_volume_features,
                        imgs=cur_image[batch*bs:(batch+1)*bs],
                    )
                else:
                    depth_outputs = self.depth_decoder([
                        x[batch*bs:(batch+1)*bs] for x in cur_feats], 
                        imgs=cur_image[batch*bs:(batch+1)*bs],
                    )

                to_skip = context['image'][:, batch*bs:(batch+1)*bs]
                to_skip = rearrange(to_skip, "b v c h w -> (b v) c h w")

                skip = self.high_resolution_skip[s+1](to_skip)

                if not export_ply:
                    margin = 0
                    xy_ray, _ = sample_image_grid((h//(1+self.cfg.low_res), w//(1+self.cfg.low_res)), device)
                else:
                    margin = 8
                    xy_ray, _ = sample_image_grid((h // (1+self.cfg.low_res) - margin*2, 
                                                    w // (1+self.cfg.low_res) - margin*2), device)
                    xy_ray = xy_ray + torch.tensor([[[margin]]], dtype=torch.float32, device=device)
                    depth_outputs[f'output_pred_s{s}_b1hw'] = depth_outputs[f'output_pred_s{s}_b1hw'][:, :, margin:-margin, margin:-margin]
                    depth_outputs[f'depth_pred_s{s}_b1hw'] = depth_outputs[f'depth_pred_s{s}_b1hw'][:, :, margin:-margin, margin:-margin]
                    depth_outputs[f'depth_weights'] = depth_outputs[f'depth_weights'][:, :, margin:-margin, margin:-margin]
                    
                    if not resized:
                        context[f'depth_s{s}'] = context[f'depth_s{s}'][:, :, :, margin:-margin, margin:-margin]
                        resized = 1
                        context["intrinsics"] = context["intrinsics"] * torch.tensor([[w*1.0 / (w - 2*margin), h*1.0 / (h - 2*margin), 1]], device=device)

                    skip = skip[:, :, margin:-margin, margin:-margin]
                
                if self.cfg.depth_pad:
                    border = 8 if not self.cfg.low_res else 4

                    for name in [f'depth_pred_s{s}_b1hw', f'depth_weights']:
                        depth_outputs[name][:, :, :border, :] = depth_outputs[name][:, :, border, None]
                        depth_outputs[name][:, :, -border:, :] = depth_outputs[name][:, :, -border-1, None]
                        depth_outputs[name][:, :, :, :border] = depth_outputs[name][:, :, :, border, None]
                        depth_outputs[name][:, :, :, -border:] = depth_outputs[name][:, :, :, -border-1, None]
                
                gaussians_feats = rearrange(depth_outputs[f'output_pred_s{s}_b1hw'][:,1:], '(b v) c h w -> b v h w c', b=b, v=n_views_now)
                gaussians_feats = gaussians_feats + rearrange(skip, "(b v) c h w -> b v h w c", b=b, v=n_views_now)
                
                if not self.cfg.larger_weight:
                    densities = nn.Sigmoid()(rearrange(depth_outputs[f'output_pred_s{s}_b1hw'][:,:1], '(b v) c h w -> b v (c h w) () ()', b=b, v=n_views_now))
                else:
                    densities = 1 + torch.exp(rearrange(depth_outputs[f'output_pred_s{s}_b1hw'][:,:1], '(b v) c h w -> b v (c h w) () ()', b=b, v=n_views_now))
                    densities = densities.clip(1,100)

                depth_outputs['densities'] = rearrange(densities, "b v x y z -> (b v) () x y z")
                
                if self.cfg.use_gt_depth:
                    depths_raw = context[f'depth_s0'][0, batch*bs:(batch+1)*bs]
                    depths = depth_outputs[f'depth_pred_s{s}_b1hw']
                    weights = depth_outputs[f'depth_weights']
                    
                    mask = (depths_raw > 1e-3) * (depths_raw < 10)
                    
                    new_depths = torch.zeros_like(depths).to(depths.device)
                    new_depths[mask] = depths_raw[mask].float()
                    new_depths[~mask] = depths[~mask]
                    depth_outputs[f'depth_pred_s{s}_b1hw'] = new_depths
                    
                    new_weights = torch.zeros_like(weights).to(weights.device)
                    new_weights[mask] = 1.
                    new_weights[~mask] = weights[~mask]
                    depth_outputs['depth_weights'] = new_weights

                depths = rearrange(depth_outputs[f'depth_pred_s{s}_b1hw'], "(b v) c h w -> b v (c h w) () ()", b=b)
                weights = rearrange(depth_outputs[f'depth_weights'], "(b v) c h w -> b v (c h w) () ()", b=b)

                for key in depth_outputs:
                    if key not in depth_outputs_all:
                        continue
                    if is_testing:
                        depth_outputs_all[key].append(depth_outputs[key])
                    else:
                        depth_outputs_all[key].append(depth_outputs[key])

                gaussians_feats = rearrange(gaussians_feats, "b v h w c -> b v (h w) c")

                xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
                offset_xy = torch.zeros_like(rearrange(gaussians_feats[..., :2], "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces),
                                                device=gaussians_feats.device)
                xy_ray = xy_ray + offset_xy

                coords = self.gaussian_adapter.forward(
                    rearrange(context["extrinsics"][:, batch*bs:(batch+1)*bs], "b v i j -> b v () () () i j"),
                    rearrange(context["intrinsics"][:, batch*bs:(batch+1)*bs], "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    densities,
                    gaussians_feats,
                    (h // (1+self.cfg.low_res) - margin*2, w // (1+self.cfg.low_res) - margin*2),
                    load_depth=self.cfg.load_depth,
                    fusion=True,
                )

                if not self.cfg.load_cache:
                    feats_input, guidances, guidance_mask, valid = self.extract_fastsam_clip_feats(cur_image[batch*bs:(batch+1)*bs], (h, w))
                else:
                    feats_input, guidances, guidance_mask, valid = self.extract_feats_from_cache(contexts[0]["cache"], (h, w))
                
                feats_input = torch.nn.functional.interpolate(
                    feats_input.permute(0, 3, 1, 2),
                    size=(h // 2, w // 2),
                    mode='nearest'
                ).permute(0, 2, 3, 1)
                
                for i in range(len(guidances)):
                    guidances[i] =  torch.nn.functional.interpolate(
                        guidances[i].permute(0, 3, 1, 2),
                        size=(h // 2, w // 2),
                        mode='nearest'
                    ).permute(0, 2, 3, 1)

                guidance_mask = torch.nn.functional.interpolate(
                    guidance_mask.unsqueeze(1).float(),
                    size=(h // 2, w // 2),
                    mode='nearest'
                )[:, 0, :, :].bool()

                if self.cfg.semantic_early_fusion:
                    clip_residual = self.semantic_proj(rearrange(feats_input, "b h w c -> b c h w"))
                    clip_residual = rearrange(clip_residual, "b c h w -> b (h w) c")
                    clip_gaussians_feats = gaussians_feats[0] + clip_residual

                coords = coords[0, :, :, 0, 0, :]
                feats = gaussians_feats[0] if not self.cfg.semantic_early_fusion else clip_gaussians_feats

                M_v, M_r = self.get_transformation_matrix()
                homo_coords = torch.cat([coords, torch.ones(coords.shape[0], coords.shape[1], 1).to(self.device)], dim=-1)
                rigid_transformation = M_r @ M_v
                voxel_coords = torch.floor(homo_coords @ rigid_transformation.T[:, :3])

                input_dict = {"coords": [], "feats": []}
                inverse_map_collate = []
                for voxel_coord, feat in zip(voxel_coords, feats):
                    voxelization_dict = {   
                        "return_index": True,
                        "return_inverse": True,
                        "coordinates": voxel_coord.contiguous(),
                        "features": feat,
                    }

                    _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
                        **voxelization_dict
                    )
                    inverse_map_collate.append(inverse_map)

                    input_dict["coords"].append(voxel_coord[unique_map].int())
                    input_dict["feats"].append(feat[unique_map])

                coordinates, features= ME.utils.sparse_collate(**input_dict)
                sinput = ME.SparseTensor(
                    coordinates=coordinates,
                    features=features,
                    device=self.device,
                )

                feats_3d, feat_3d_inter_collate = self.unet3d(sinput, None, inter=True)

                if not feats_3d.requires_grad:
                    feats_3d_clone = ME.SparseTensor(
                        features=feats_3d.F.clone().detach(),
                        coordinate_map_key=feats_3d.coordinate_map_key,
                        coordinate_manager=feats_3d.coordinate_manager,
                        tensor_stride=feats_3d.tensor_stride
                    )
                    
                    loss = 0.
                    for idx, feat_3d_inter in enumerate(feat_3d_inter_collate):
                        feats_3d_clone = self.memory.proj[idx](feats_3d_clone)

                        feat_3d_inter_clone = ME.SparseTensor(
                            features=feat_3d_inter.F.clone().detach(),
                            coordinate_map_key=feat_3d_inter.coordinate_map_key,
                            coordinate_manager=feat_3d_inter.coordinate_manager,
                            tensor_stride=feat_3d_inter.tensor_stride
                        )
                    
                        student = torch.cat(feats_3d_clone.decomposed_features, dim=0)
                        teacher = torch.cat(feat_3d_inter_clone.decomposed_features, dim=0)
                    
                        loss += (0.2 * F.mse_loss(student, teacher) + (1 - nn.CosineSimilarity(dim=-1)(student, teacher).mean()))
                else:   
                    loss = 0.

                    for guidance in guidances:
                        loss_cos = []
                        for idx, feat_3d in enumerate(feats_3d.decomposed_features):
                            if valid[idx] == -1:
                                continue

                            feat_3d = self.to_semantic(feat_3d)
                            feat_3d_full = feat_3d[inverse_map_collate[idx]]

                            mask = guidance_mask[idx].flatten()

                            loss_cos.append(
                                1 - nn.CosineSimilarity(dim=-1)(
                                    feat_3d_full[mask], 
                                    rearrange(guidance[idx], "h w d -> (h w) d")[mask]
                                )
                            )
        
                        loss += torch.cat(loss_cos).mean()
        
        return loss
    
    def extract_feats_from_cache(self, caches, image_shape):
        h, w = image_shape

        with torch.no_grad():
            clip_img_feats = []
            guidance = []
            guidance_mask = []
            valid = torch.ones(len(caches))
            for i, cache_path in enumerate(caches):
                cache_path = cache_path[0]

                try:
                    cache = torch.load(cache_path, weights_only=False)
                except:
                    cache = torch.load(cache_path[0], weights_only=False)
            
                invalid = len(cache['fastsam'][0]) == 1 and cache['fastsam'][0][0]['segmentation'].sum() == h*w
                
                if invalid:
                    valid[i] = -1
                    
                    clip_img_feats.append(torch.zeros((h, w, self.clip_dim)).to(self.device))
                    guidance.append(torch.zeros((h, w, self.clip_dim)).to(self.device))
                    guidance_mask.append(torch.zeros((h, w)).to(self.device).bool())
                else:
                    mask_feats = cache['clip'][0]
                    
                    clip_img_feat = torch.zeros((h, w, mask_feats.shape[-1])).to(self.device)
                    count = torch.zeros((h, w)).to(self.device)
                    for feat, mask in zip(mask_feats, cache['fastsam'][0]):
                        clip_img_feat[mask['segmentation'], :] = feat.float()
                        count[mask['segmentation']] += 1

                    guidance_mask.append(count != 0)
                    
                    if self.cfg.non_object_embedding:
                        clip_img_feat[~guidance_mask[-1]] = self.non_object_embedding
                    
                    clip_img_feats.append(clip_img_feat)

        guidance = torch.stack(clip_img_feats).clone()
        clip_img_feats = torch.nn.functional.normalize(torch.stack(clip_img_feats), dim=-1, eps=1e-5)
        guidance_mask = torch.stack(guidance_mask)
        
        return clip_img_feats, [guidance], guidance_mask, valid
    
    def extract_fastsam_clip_feats(self, images, image_shape):
        h, w = image_shape
        
        with torch.no_grad():
            inputs = (images * 255.).clone()
            everything_results = self.fastsam(
                inputs,
                device=self.device,
                retina_masks=True,
                imgsz=(w, h),
                conf=self.cfg.fastsam_conf,
                iou=self.cfg.fastsam_iou
            )
            prompt = self.fastsam_prompt(inputs, everything_results, device=self.device)

            if self.cfg.clip_model == "maskclip":
                clip_feats = self.maskclip.encode_image(self.clip_preprocess(images))[1]
                clip_feats = rearrange(clip_feats, "b (h w) d -> b d h w", h=24, w=24)
                clip_feats = torch.nn.functional.interpolate(
                    clip_feats,
                    size=(h, w),
                    mode='bilinear'
                )
            elif self.cfg.clip_model == "maskadapter":
                clip_image = ((images * 255.) - self.PIXEL_MEAN) / self.PIXEL_STD
                x = self.clip.clip_model.visual.trunk.stem(clip_image.float())
                for i in range(4):
                    x = self.clip.clip_model.visual.trunk.stages[i](x)
                clip_feature = self.clip.clip_model.visual.trunk.norm_pre(x).contiguous()

                clip_feature_dense = self.clip.clip_model.visual.trunk.head.norm(clip_feature)
                clip_feature_dense = self.clip.clip_model.visual.trunk.head.drop(clip_feature_dense.permute(0, 2, 3, 1))
                clip_feature_dense = self.clip.clip_model.visual.head(clip_feature_dense).permute(0, 3, 1, 2)

            clip_img_feats = []
            guidance_mask = []
            valid = torch.ones(len(images))

            for idx, everything_result in enumerate(everything_results):
                format_results, masks_torch = prompt._format_results(everything_result, 0, sort=True)
                
                if self.cfg.clip_model == "clip":
                    cropped_images = prompt._crop_image_torch(format_results, img_idx=idx, resize=(224, 224))
                    cropped_images = self.clip_process(cropped_images)
                    mask_feats = self.clip.encode_image(cropped_images.half())
                elif self.cfg.clip_model== "openseg":
                    image = images[idx]
                    clip_feats = self.extract_openseg_img_feature(
                        image, 
                        self.openseg, 
                        self.openseg_text_emb, 
                        img_size=[h, w], 
                    )
                    
                    clip_feats_flat = rearrange(clip_feats, "h w d -> (h w) d").to(self.device).float()
                    masks_flat = rearrange(masks_torch, "n h w -> n (h w)").to(self.device).float()
                    pooled = masks_flat @ clip_feats_flat
                    
                    denom = masks_flat.sum(dim=-1, keepdim=True).clamp(min=1e-6)
                    mask_feats = pooled / denom
                elif self.cfg.clip_model == "maskadapter":
                    semantic_activation_maps = self.maskadapter(clip_feature_dense[idx:idx+1], masks_torch[None].float())
                    maps_for_pooling = F.interpolate(
                        semantic_activation_maps, 
                        size=clip_feature.shape[-2:],
                        mode='bilinear', 
                        align_corners=False
                    )

                    B, C = clip_feature[idx:idx+1].size(0), clip_feature[idx:idx+1].size(1)
                    N = maps_for_pooling.size(1)
                    num_instances = N // 16
                    maps_for_pooling = F.softmax(F.logsigmoid(maps_for_pooling).view(B, N,-1), dim=-1)
                    pooled_clip_feature = torch.bmm(maps_for_pooling, clip_feature[idx:idx+1].view(B, C, -1).permute(0, 2, 1))

                    batch, num_query, channel = pooled_clip_feature.shape
                    pooled_clip_feature = pooled_clip_feature.reshape(batch*num_query, channel, 1, 1) # fake 2D input
                    x = self.clip.clip_model.visual.trunk.head(pooled_clip_feature)
                    x = self.clip.clip_model.visual.head(x)
                    mask_feats = (x.reshape(B, num_instances, 16, -1).mean(dim=-2).contiguous())[0]
                
                clip_img_feat = torch.zeros((h, w, mask_feats.shape[-1])).to(self.device)
                count = torch.zeros((h, w)).to(self.device)
                for jdx, (feat, mask) in enumerate(zip(mask_feats, format_results)):
                    clip_img_feat[mask['segmentation'], :] += feat.float()
                    count[mask['segmentation']] += 1

                count[count == 0.] = 1e-8
                clip_img_feat /= count.unsqueeze(-1)
                
                clip_img_feats.append(clip_img_feat)
                codebook_caches.append(codebook_cache)

            clip_img_feats = torch.nn.functional.normalize(torch.stack(clip_img_feats), dim=-1, eps=1e-5)
            clip_img_feats = clip_img_feats.permute(0, 3, 1, 2)
            codebook_caches = torch.stack(codebook_caches, dim=0)

        return clip_img_feats, codebook_caches, globals

    def extract_openseg_img_feature(
        self, 
        img, 
        openseg_model, 
        text_emb, 
        img_size=None, 
        regional_pool=True
    ):
        '''Extract per-pixel OpenSeg features.'''
        
        def array_to_bytes(array):
            assert array.ndim == 3 and array.shape[2] == 3, "Expected shape (H, W, 3)"
            assert array.dtype in [np.float32, np.float64], "Array must be float"

            uint8_img = tff.image.convert_image_dtype(array, dtype=tff.uint8, saturate=True)
            png_bytes = tff.io.encode_png(uint8_img).numpy()
            return png_bytes

        np_image_string = array_to_bytes(img.permute(1, 2, 0).cpu().numpy())
        
        results = openseg_model.signatures['serving_default'](
            inp_image_bytes=tff.convert_to_tensor(np_image_string),
            inp_text_emb=text_emb)
        img_info = results['image_info']
        crop_sz = [
            int(img_info[0, 0] * img_info[2, 0]),
            int(img_info[0, 1] * img_info[2, 1])
        ]
        if regional_pool:
            image_embedding_feat = results['ppixel_ave_feat'][:, :crop_sz[0], :crop_sz[1]]
        else:
            image_embedding_feat = results['image_embedding_feat'][:, :crop_sz[0], :crop_sz[1]]
        if img_size is not None:
            feat_2d = tff.cast(tff.image.resize_nearest_neighbor(
                image_embedding_feat, img_size, align_corners=True)[0], dtype=tff.float16).numpy()
        else:
            feat_2d = tff.cast(image_embedding_feat[[0]], dtype=tff.float16).numpy()

        feat_2d = torch.from_numpy(feat_2d)
        
        return feat_2d
    
    def get_transformation_matrix(self):
        import collections
        from scipy.linalg import expm, norm
        
        def M(axis, theta):
            return expm(np.cross(np.eye(3), axis / norm(axis) * theta))
        
        voxelization_matrix, rotation_matrix = np.eye(4), np.eye(4)

        rot_mat = np.eye(3)
        if self.cfg.pcd_aug and self.ROTATION_AUGMENTATION_BOUND is not None:
            if isinstance(self.ROTATION_AUGMENTATION_BOUND, collections.abc.Iterable):
                rot_mats = []
                for axis_ind, rot_bound in enumerate(self.ROTATION_AUGMENTATION_BOUND):
                    theta = 0
                    axis = np.zeros(3)
                    axis[axis_ind] = 1
                    if rot_bound is not None:
                        theta = np.random.uniform(*rot_bound)
                    rot_mats.append(M(axis, theta))

                np.random.shuffle(rot_mats)
                rot_mat = rot_mats[0] @ rot_mats[1] @ rot_mats[2]
            else:
                raise ValueError()
        rotation_matrix[:3, :3] = rot_mat

        scale = 1 / self.cfg.voxel_size
        if self.cfg.pcd_aug and self.SCALE_AUGMENTATION_BOUND is not None:
            scale *= np.random.uniform(*self.SCALE_AUGMENTATION_BOUND)
        np.fill_diagonal(voxelization_matrix[:3, :3], scale)

        return torch.from_numpy(voxelization_matrix).to(self.device).float(), torch.from_numpy(rotation_matrix).to(self.device).float()


    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.epipolar_transformer.self_attention.patch_size
                * self.cfg.epipolar_transformer.downscale,
            )

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return self.epipolar_transformer.epipolar_sampler


class FFNLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        intermediate_dim=2048,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(input_dim, intermediate_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(intermediate_dim, output_dim)

        self.norm = nn.LayerNorm(output_dim)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")