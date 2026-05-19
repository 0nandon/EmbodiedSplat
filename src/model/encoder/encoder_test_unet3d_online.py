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

from .encoder_config import EncoderEpipolarCfg

from ..timer import CudaTimer
from .encoder_config import EncoderEpipolarCfg
from ...third_party.fastsam import FastSAM, FastSAMPrompt
from ...third_party.open_clip_network import OpenCLIPNetwork, OpenCLIPNetworkConfig
from ...third_party.maskadapter.mask_adapter_head import MASKAdapterHead as MaskAdapter
from ...third_party.maskadapter.clip import CLIP
from ...third_party.hiersam import Tools

from .backbone.mink_unet import mink_unet
from .backbone.multilevel_memory import MultilevelMemory
import MinkowskiEngine as ME

import tensorflow as tff2
import tensorflow.compat.v1 as tff

import math
from .common.position_encoding import PositionEmbeddingLearnedMLP

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
            export_ply=False,
            domain='ensemble'
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

        self.domain = domain
        if domain in ['ensemble', '3d']:
            self.clip_features_3d = torch.zeros((1, self.capacity, feat_dim), device=device)
            self.clip_features_2d = {
                "invalid": torch.zeros((1, self.capacity), device=device)
            }
        if domain in ['ensemble', '2d']:
            self.clip_features_2d = {
                "feat": [], # CLIP Global Codebook
                "idx": torch.ones((1, self.capacity, 6), device=device) * -1, # Index cache
                "weight": torch.zeros((1, self.capacity, 6), device=device), # Weight cache
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
        fuse=False
    ):

        if mask is not None:
            columns = self.valid[0].nonzero(as_tuple=True)[0][mask]
            self.valid[:, columns] = False
        # Find indices where new data should be placed
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

            if clip_features_3d is not None:
                self.clip_features_3d[0, invalid_indices[:num_new]] = clip_features_3d

            self.weights[0, invalid_indices[:num_new]] = weights
            self.extrinsics[0, invalid_indices[:num_new]] = extrinsics
            self.depths[0, invalid_indices[:num_new]] = depths

            if codebooks_idx is not None and codebooks_weights is not None:
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

    def _expand_storage(self, new_capacity):
        self.means = self._resize_tensor(self.means, new_capacity, device=self.device)
        self.covariances = self._resize_tensor(self.covariances, new_capacity, device=self.device)
        self.harmonics = self._resize_tensor(self.harmonics, new_capacity, device=self.device)
        self.opacities = self._resize_tensor(self.opacities, new_capacity, device=self.device)
        self.features = self._resize_tensor(self.features, new_capacity, device=self.device)
        self.coords = self._resize_tensor(self.coords, new_capacity, device=self.device)
        self.densities = self._resize_tensor(self.densities, new_capacity, device=self.device)
        self.weights = self._resize_tensor(self.weights, new_capacity, device=self.device)
        self.extrinsics = self._resize_tensor(self.extrinsics, new_capacity, device=self.device)
        self.depths = self._resize_tensor(self.depths, new_capacity, device=self.device)
        self.valid = self._resize_tensor(self.valid, new_capacity, device=self.device, fill=False, dtype=torch.bool)
        
        if self.domain in ['ensemble', '3d']:
            self.clip_features_3d = self._resize_tensor(self.clip_features_3d, new_capacity, device=self.device)
        if self.domain in ['ensemble', '2d']:
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

class EmbodiedSplatEncoderTestUnet3d_Online(Encoder[EncoderEpipolarCfg]):
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

        if cfg.backbone.name == 'dino':
            self.backbone = get_backbone(cfg.backbone, 3+cfg.load_depth)
            self.backbone_projection = nn.Sequential(
                activation_func,
                nn.Linear(self.backbone.d_out, cfg.d_feature),
            )
            if cfg.use_epipolar_transformer:
                self.epipolar_transformer = EpipolarTransformer(
                    cfg.epipolar_transformer,
                    cfg.d_feature,
                    n_views = cfg.max_batch,
                )
            else:
                self.epipolar_transformer = None
            if cfg.est_depth == 'est':
                self.depth_predictor = DepthPredictorMonocular(
                    cfg.d_feature,
                    cfg.num_monocular_samples,
                    cfg.num_surfaces,
                    cfg.use_transmittance,
                )
            else:
                self.opacity_mlp = nn.Sequential(
                    activation_func,
                    nn.Linear(cfg.d_feature, 1),
                    nn.Sigmoid(),
                )
                self.depth_refine = nn.Sequential(
                    activation_func,
                    nn.Linear(
                        cfg.d_feature,
                        cfg.d_feature,
                    ),
                    activation_func,
                    nn.Linear(
                        cfg.d_feature,
                        1,
                    ),
                )
            if cfg.predict_opacity:
                self.to_opacity = nn.Sequential(
                    activation_func,
                    nn.Linear(cfg.d_feature, 1),
                    nn.Sigmoid(),
                )
            self.to_gaussians = nn.Sequential(
                activation_func,
                nn.Linear(
                    cfg.d_feature,
                    cfg.num_surfaces * (2 + self.gaussian_adapter.d_in),
                ),
            )
            
            self.high_resolution_skip = nn.Sequential(
                nn.Conv2d(3+self.cfg.load_depth, cfg.d_feature, 7, 1, 3),
                activation_func,
            )
        elif cfg.backbone.name == 'cost_volume':
            self.backbone = BackboneMultiview(
                feature_channels=cfg.d_feature,
                downscale_factor=cfg.downscale_factor,
                no_cross_attn=cfg.wo_backbone_cross_attn,
                use_epipolar_trans=cfg.use_epipolar_trans,
                limit=cfg.backbone_limit,
            )
            ckpt_path = cfg.unimatch_weights_path
            if get_cfg().mode == 'train':
                if cfg.unimatch_weights_path is None:
                    print("==> Init multi-view transformer backbone from scratch")
                else:
                    print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                    unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                    updated_state_dict = OrderedDict(
                        {
                            k: v
                            for k, v in unimatch_pretrained_model.items()
                            if k in self.backbone.state_dict()
                        }
                    )
                    is_strict_loading = not cfg.wo_backbone_cross_attn
                    self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

            # gaussians convertor
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

            # cost volume based depth predictor
            self.depth_predictor = DepthPredictorMultiView(
                feature_channels=cfg.d_feature,
                upscale_factor=cfg.downscale_factor,
                num_depth_candidates=cfg.num_depth_candidates,
                costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
                costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
                costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
                gaussian_raw_channels=cfg.num_surfaces * (self.gaussian_adapter.d_in + 2),
                gaussians_per_pixel=cfg.gaussians_per_pixel,
                # num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_views=min(get_cfg().dataset.view_sampler.num_context_views, cfg.max_batch),
                depth_unet_feat_dim=cfg.depth_unet_feat_dim,
                depth_unet_attn_res=cfg.depth_unet_attn_res,
                depth_unet_channel_mult=cfg.depth_unet_channel_mult,
                wo_depth_refine=cfg.wo_depth_refine,
                wo_cost_volume=cfg.wo_cost_volume,
                wo_cost_volume_refine=cfg.wo_cost_volume_refine,
            )

            if self.cfg.adaptive_gaussian:
                self.keypoint_scorer = ContextScorer(
                        channels=cfg.score_channels,
                        num_layers=cfg.num_layers,
                        max_num_view=cfg.max_num_view
                    )

                self.cascade_gaussian_adapter = CascadeGaussianAdapter(
                    stages=cfg.cga_stages,
                    opacity_thres=cfg.opacity_thres,
                    split_count=cfg.split_count,
                    scaling_factor=cfg.scaling_factor,
                    opacity_factor=cfg.opacity_factor,
                    num_groups=cfg.cga_num_groups,
                    max_num_view=cfg.max_num_view,
                    num_levels=cfg.cga_num_levels,
                    attn_drop=cfg.attn_drop,
                    num_learnable_pts=cfg.num_learnable_pts,
                    fix_scale=cfg.fix_scale,
                    num_anchors=cfg.cga_num_anchors,
                    score_embed=cfg.score_embed
                )

                self.iterative_refiner = IterativeGaussianRefiner(
                    stages=cfg.igr_stages,
                    num_groups=cfg.igr_num_groups,
                    num_levels=cfg.igr_num_levels,
                    attn_drop=cfg.attn_drop,
                    max_num_view=cfg.max_num_view,
                    num_learnable_pts=cfg.num_learnable_pts,
                    fix_scale=cfg.fix_scale,
                    num_anchors=cfg.igr_num_anchors,
                    embed_dim=cfg.d_feature
                )
        else:
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
                    dec_num_input_ch = (self.backbone.num_ch_enc[:1] + self.cv_encoder.num_ch_enc)
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
            
        if not self.cfg.load_cache:
            if self.cfg.sam_model_type == "sam":
                self.sam = Tools(
                    points_per_side = 32,
                    pred_iou_thresh = 0.7,
                    box_nms_thresh = 0.7,
                    stability_score_thresh = 0.85,
                    crop_n_layers = 1,
                    crop_n_points_downscale_factor = 1,
                    min_mask_region_area = 100,
                    load_sam="pretrained/sam_vit_h_4b8939.pth",
                    load_tracker="pretrained/sam2.1_hiera_base_plus.pt"
                )
            elif self.cfg.sam_model_type == "fastsam":
                self.fastsam = FastSAM(self.cfg.fastsam_model_path)
                self.fastsam_prompt = FastSAMPrompt

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
            elif self.cfg.clip_model == "openseg":
                saved_model_path = self.cfg.clip_model_path
                self.openseg = tff2.saved_model.load(saved_model_path, tags=[tff.saved_model.tag_constants.SERVING],)
                self.openseg_text_emb = tff.zeros([1, 1, 768])
            elif self.cfg.clip_model == "maskadapter":
                self.maskadapter = MaskAdapter()
                self.clip = CLIP(model_name=self.cfg.clip_model_type, pretrained=self.cfg.clip_model_path)
                
                self.PIXEL_MEAN = torch.tensor([122.7709383, 116.7460125, 104.09373615]).view(1, -1, 1, 1).to(self.device)
                self.PIXEL_STD = torch.tensor([68.5005327, 66.6321579, 70.32316305]).view(1, -1, 1, 1).to(self.device)

        self.clip_dim = self.cfg.clip_dim
        
        if self.cfg.openvocab_domain in ["ensemble", "3d"]:
            if self.cfg.semantic_early_fusion:
                self.semantic_proj = nn.Sequential(
                    nn.Conv2d(in_channels=self.clip_dim, out_channels=64, kernel_size=1),
                    activation_func,
                )
            
            self.semantic_gru = GRU2D_naive_Wweights(concat_depth=self.cfg.concat_depth)

            self.to_semantic = nn.Sequential(
                activation_func,
                nn.Linear(
                    64,
                    self.clip_dim
                ),
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
                    vmp_layer=self.cfg.memory_vmp_layer
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
    
    def warmup_construct(self, context, target, decoder, is_testing=False, export_ply=False, test_bev=False):
        device = context["image"].device
        b, n_views, _, h, w = context["image"].shape
        results = {}
        
        num_context_views = self.cfg.num_views
        warmup_n_views = self.cfg.num_views

        context['image_shape'] = (h, w)
        self.cfg.gaussians_per_pixel = 1
        context_intrinsics = context['intrinsics'].clone()
        context_intrinsics[:,:,0] *= (w // 4)
        context_intrinsics[:,:,1] *= (h // 4)

        globals = EfficientGaussians(
            initial_capacity=n_views*10000, 
            growth_factor=1.5, 
            device=context['image'].device, 
            testing=is_testing,
            export_ply=export_ply,
            domain=self.cfg.openvocab_domain
        )

        cur_indices = torch.arange(warmup_n_views, device=context['image'].device)
        cur_intrinsics = context_intrinsics.gather(dim=1, index=cur_indices.view(1,-1,1,1).repeat(b,1,3,3))
        cur_extrinsics = context['extrinsics'].gather(dim=1, index=cur_indices.view(1,-1,1,1).repeat(b,1,4,4))
        cur_image = context['image'].gather(dim=1, index=cur_indices.view(1,-1,1,1,1).repeat(b,1,3,h,w)).view(-1,3,h,w)

        cur_cache = [context['cache'][i] for i in cur_indices]

        cur_feats = self.backbone(cur_image)
        
        resized = 0
        s = -1 if not self.cfg.low_res else 0
        
        full_indices = torch.arange(warmup_n_views, device=context['image'].device)[None].repeat(warmup_n_views,1)
        src_indices = full_indices[~(full_indices == cur_indices[:,None])].view(1,warmup_n_views,warmup_n_views-1).repeat(b,1,1)

        bs = self.cfg.max_batch
        depth_outputs_all = {f'depth_pred_s{s}_b1hw': [], 'densities': []}
        
        for batch in range(min(np.ceil(warmup_n_views/bs).astype(int), self.cfg.max_batch_length)):
            if self.cfg.sam_model_type == "fastsam":
                clip_img_feats, codebook_caches, globals = self.extract_fastsam_clip_feats(
                    cur_image[batch*bs:(batch+1)*bs], 
                    cur_cache[batch*bs:(batch+1)*bs], 
                    (h, w), 
                    globals
                )
            elif self.cfg.sam_model_type == "sam":
                clip_img_feats, codebook_caches, globals = self.extract_sam_clip_feats(
                    cur_image[batch*bs:(batch+1)*bs],
                    cur_cache[batch*bs:(batch+1)*bs],  
                    (h, w), 
                    globals
                )
            
            if self.cfg.openvocab_domain in ['ensemble', '3d']:
                clip_gaussian_feats = torch.nn.functional.interpolate(
                    clip_img_feats,
                    size=(h // 2, w // 2),
                    mode='nearest'
                )
            else:
                clip_gaussian_feats = None

            if self.cfg.openvocab_domain in ['ensemble', '2d']:
                cur_codebook_caches = torch.nn.functional.interpolate(
                    codebook_caches.unsqueeze(1),
                    size=(h // 2, w // 2),
                    mode='nearest'
                ).squeeze(1)
            else:
                cur_codebook_caches = None

            warmup_n_views_now = min(bs, warmup_n_views-batch*bs)
            matching_cur_feats = cur_feats[1][batch*bs:(batch+1)*bs]
            dim = matching_cur_feats.shape[-3]
            
            matching_src_feats = rearrange(cur_feats[1], "(b v) c h w -> b v c h w", b=b, v=warmup_n_views)[:,None].repeat(1,warmup_n_views_now,1,1,1,1).gather(dim=2, index=src_indices[:,batch*bs:(batch+1)*bs,:,None,None,None]\
                                .repeat(1, 1, 1, dim, h // 4, w // 4)).view(-1, warmup_n_views-1, dim, h // 4, w // 4)

            src_extrinsics = context['extrinsics'][:,None].repeat(1,warmup_n_views_now,1,1,1).gather(dim=2, index=src_indices[:,batch*bs:(batch+1)*bs,:,None,None].repeat(1,1,1,4,4))
            src_intrinsics = context_intrinsics[:,None].repeat(1,warmup_n_views_now,1,1,1).gather(dim=2, index=src_indices[:,batch*bs:(batch+1)*bs,:,None,None].repeat(1,1,1,3,3))
            
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

            depth_outputs = self.depth_decoder(
                [cur_feats[0][batch*bs:(batch+1)*bs]] + cost_volume_features,
                imgs=cur_image[batch*bs:(batch+1)*bs]
            )

            to_skip = cur_image[None][:, batch*bs:(batch+1)*bs]
            to_skip = rearrange(to_skip, "b v c h w -> (b v) c h w")
            skip = self.high_resolution_skip[s+1](to_skip)
            
            if not export_ply:
                margin = 0
                xy_ray, _ = sample_image_grid((h//(1+self.cfg.low_res), w//(1+self.cfg.low_res)), device)
            else:
                margin = 8
                xy_ray, _ = sample_image_grid(
                    (h // (1+self.cfg.low_res) - margin*2, w // (1+self.cfg.low_res) - margin*2), 
                    device
                )
                xy_ray = xy_ray + torch.tensor([[[margin]]], dtype=torch.float32, device=device)
                depth_outputs[f'output_pred_s{s}_b1hw'] = depth_outputs[f'output_pred_s{s}_b1hw'][:, :, margin:-margin, margin:-margin]
                depth_outputs[f'depth_pred_s{s}_b1hw'] = depth_outputs[f'depth_pred_s{s}_b1hw'][:, :, margin:-margin, margin:-margin]
                depth_outputs[f'depth_weights'] = depth_outputs[f'depth_weights'][:, :, margin:-margin, margin:-margin]
                    
                if not resized:
                    context[f'depth_s{s}'] = context[f'depth_s{s}'][:, :, :, margin:-margin, margin:-margin]
                    resized = 1
                    context["intrinsics"] = context["intrinsics"] * torch.tensor([[w*1.0 / (w - 2*margin), h*1.0 / (h - 2*margin), 1]], device=device)

                skip = skip[:, :, margin:-margin, margin:-margin]
            
            gaussians_feats = rearrange(depth_outputs[f'output_pred_s{s}_b1hw'][:,1:], '(b v) c h w -> b v h w c', b=b, v=warmup_n_views_now)
            gaussians_feats = gaussians_feats + rearrange(skip, "(b v) c h w -> b v h w c", b=b, v=warmup_n_views_now)
            
            if self.cfg.openvocab_domain in ['ensemble', '3d']:
                if self.cfg.semantic_early_fusion:
                    clip_gaussians_feats_3d = gaussians_feats + self.semantic_proj(clip_gaussian_feats).permute(0, 2, 3, 1)[None]
                else:
                    clip_gaussians_feats_3d = gaussians_feats.clone()
                clip_gaussians_feats_3d = rearrange(clip_gaussians_feats_3d, "b v h w c -> b v (h w) c")
            else:
                clip_gaussians_feats_3d = None 

            densities = nn.Sigmoid()(rearrange(depth_outputs[f'output_pred_s{s}_b1hw'][:,:1], '(b v) c h w -> b v (c h w) () ()', b=b, v=warmup_n_views_now))
            depth_outputs['densities'] = rearrange(densities, "b v x y z -> (b v) () x y z")

            if self.cfg.use_gt_depth:
                depths_raw = context[f'depth_s0'][0, batch*bs:(batch+1)*bs]
                depths = depth_outputs[f'depth_pred_s{s}_b1hw']
                weights = depth_outputs[f'depth_weights']
                    
                mask = (depths_raw > 1e-3) * (depths_raw < 10)
                    
                depths[mask] = depths_raw[mask].float()
                depth_outputs[f'depth_pred_s{s}_b1hw'] = depths
                    
                weights[mask] = 1.
                depth_outputs['depth_weights'] = weights

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
            offset_xy = torch.zeros_like(
                rearrange(gaussians_feats[..., :2], "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces),
                device=gaussians_feats.device
            )
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
                fusion=True
            )

            num_raw_gaussians = gaussians_feats.shape[2] * gaussians_feats.shape[1]
            B = gaussians_feats.shape[0]

            for bb in range(B):
                cur_gs = gaussians_feats[bb:bb+1]
                cur_clip_gs_3d = clip_gaussians_feats_3d[bb:bb+1] if clip_gaussians_feats_3d is not None else None
                cur_coords = coords[bb:bb+1]
                cur_densities = densities[bb:bb+1]
                cur_weights = weights[bb:bb+1]
                cur_depth = rearrange(depth_outputs[f'depth_pred_s{s}_b1hw'], "(b v) c h w -> b v c h w", b=B)[bb]

                if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                    cur_codebook_caches = rearrange(cur_codebook_caches, "b h w ->() b (h w)")

                globals = self.fuse_gaussians(
                    cur_gs,
                    cur_clip_gs_3d,
                    cur_codebook_caches,
                    cur_coords, 
                    cur_densities, 
                    cur_weights, 
                    cur_depth,
                    context["extrinsics"][bb:bb+1, batch*bs:(batch+1)*bs],
                    context["intrinsics"][bb:bb+1, batch*bs:(batch+1)*bs],
                    (h // (1+self.cfg.low_res) - margin*2, w // (1+self.cfg.low_res) - margin*2),
                    vis=self.cfg.vis,
                    globals=globals,
                    remove=self.cfg.remove,
                    use_em=self.cfg.use_em,
                    decoder=decoder,
                    near=context["near"][bb:bb+1, batch*bs:(batch+1)*bs], 
                    far=context["far"][bb:bb+1, batch*bs:(batch+1)*bs],
                    img=context['image'][bb:bb+1, batch*bs:(batch+1)*bs],
                    fusion=self.cfg.fusion,
                    fore_fusion=self.cfg.fore_fusion,
                    concat_depth=self.cfg.concat_depth,
                    use_gru=self.cfg.use_gru,
                    export_ply=export_ply
                )
                    
        depth_preds_all = torch.cat(depth_outputs_all[f'depth_pred_s{s}_b1hw'], dim=0)
        densities_preds_all = torch.cat(depth_outputs_all[f'densities'], dim=0)
        
        if self.cfg.refine_gs:
            for _ in range(self.cfg.refine_times):
                for i in range(len(depth_preds_all)):
                    cur_depth = depth_preds_all[i:i+1]
                    cur_weights = densities_preds_all[i:i+1]
                    if self.cfg.refine_uniform:
                        cur_weights = torch.ones_like(cur_weights, device=cur_weights.device)
                        globals.densities = torch.ones_like(globals.densities, device=globals.densities.device)

                    globals = self.refine_gaussians(
                        globals, 
                        cur_depth, 
                        cur_weights, 
                        context["extrinsics"][:,i:i+1], 
                        context["intrinsics"][:,i:i+1], 
                        (h // (1+self.cfg.low_res) - margin*2, w // (1+self.cfg.low_res) - margin*2),
                        ws=self.cfg.refine_ws, 
                        depth_thres=self.cfg.refine_thres,
                        soft_thres=self.cfg.refine_soft_thres, refine_pp=self.cfg.refine_pp,
                        export_ply=(export_ply or self.cfg.ft),
                        num=i,
                    )
                
        if test_bev:
            edges_means = create_transformed_pyramid(context['extrinsics'][0,batch].cpu().numpy())
            edges_means = torch.from_numpy(edges_means).unsqueeze(0).to(globals.means.device)
            edges_covariances = 0.0001 * torch.eye(3, device=globals.means.device)[None, None].repeat(1, edges_means.shape[1], 1, 1)
            edges_harmonics = torch.zeros([1, edges_means.shape[1], 3, globals.harmonics.shape[-1]], device=globals.means.device)
            edges_opacities = torch.ones([1, edges_means.shape[1]], device=globals.means.device)
            edges_conf = torch.ones([1, edges_means.shape[1]], device=globals.means.device)
            edges_harmonics[:, :, 0, 0] = 1.5

            gaussians = Gaussians(
                means=torch.cat([globals.means[:, globals.valid[0]], edges_means], dim=1),
                covariances=torch.cat([globals.covariances[:, globals.valid[0]], edges_covariances], dim=1),
                harmonics=torch.cat([globals.harmonics[:, globals.valid[0]], edges_harmonics], dim=1), 
                opacities=torch.cat([globals.opacities[:, globals.valid[0]], edges_opacities], dim=1),
                conf=torch.cat([globals.densities[:, globals.valid[0],0,0], edges_conf], dim=1)
            )
                    
            output_bev = decoder.forward(
                gaussians,
                target["bev_extrinsics"],
                target["intrinsics"][:, :1],
                target["near"][:, :1],
                target["far"][:, :1],
                (1440, 1920),
                depth_mode='depth',
                scale_invariant=True,
                background_color=torch.tensor([1, 1, 1], device=device).float().to(globals.means.device),
            )
                    
            color = output_bev.color[0][0]
            save_image(color, os.path.join("", f"warmup.png"))

        warmup_cache = {
            'warmup_indices': cur_indices,
            'warmup_intrinsics': cur_intrinsics,
            'warmup_extrinsics': cur_extrinsics,
            'warmup_img_feats': cur_feats,
        }
           
        return globals, depth_outputs_all, warmup_cache

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
        warmup_n_views = self.cfg.num_views

        globals, depth_outputs_all, warmup_cache = self.warmup_construct(
            contexts[0],
            target, 
            decoder,
            is_testing=is_testing,
            export_ply=export_ply,
            test_bev=test_bev
        )

        if self.cfg.est_depth == 'cost':
            length = len(contexts)
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
            
            all_indices = torch.arange(n_views, device=context['image'].device)[warmup_n_views:]
            all_intrinsics = context_intrinsics.gather(dim=1, index=all_indices.view(1,-1,1,1).repeat(b,1,3,3))
            all_extrinsics = context['extrinsics'].gather(dim=1, index=all_indices.view(1,-1,1,1).repeat(b,1,4,4))
            all_image = context['image'].gather(dim=1, index=all_indices.view(1,-1,1,1,1).repeat(b,1,3,h,w)).view(-1,3,h,w)
            all_cache = [context['cache'][i] for i in all_indices]
            
            cur_indices = warmup_cache['warmup_indices']
            cur_intrinsics = warmup_cache['warmup_intrinsics']
            cur_extrinsics= warmup_cache['warmup_extrinsics']
            cur_feats = warmup_cache['warmup_img_feats']

            # online setting
            for idx in tqdm(range(n_views-warmup_n_views), desc="Processing " + scene[:-2]):
                cur_indices = torch.cat([cur_indices, all_indices[idx:idx+1]], dim=0)
                cur_intrinsics = torch.cat([cur_intrinsics, all_intrinsics[:, idx:idx+1]], dim=1)
                cur_extrinsics = torch.cat([cur_extrinsics, all_extrinsics[:, idx:idx+1]], dim=1)
                cur_image = all_image[idx:idx+1]             
                cur_cache = [all_cache[idx]]

                new_feats = self.backbone(cur_image)
                
                if self.cfg.sam_model_type == "fastsam":
                    new_clip_feats, codebook_caches, globals = self.extract_fastsam_clip_feats(
                        cur_image, cur_cache, (h, w), globals
                    )
                elif self.cfg.sam_model_type == "sam":
                    new_clip_feats, codebook_caches, globals = self.extract_sam_clip_feats(cur_image, cur_cache, (h, w), globals)

                if self.cfg.openvocab_domain in ['ensemble', '3d']:
                    clip_gaussians_feats = torch.nn.functional.interpolate(
                        new_clip_feats,
                        size=(h // 2, w // 2),
                        mode='nearest'
                    )
                else:
                    clip_gaussians_feats = None

                if self.cfg.openvocab_domain in ['ensemble', '2d']:
                    cur_codebook_caches = torch.nn.functional.interpolate(
                        codebook_caches.unsqueeze(1),
                        size=(h // 2, w // 2),
                        mode='nearest'
                    ).squeeze(1)
                    cur_codebook_caches = rearrange(cur_codebook_caches, "b h w -> () b (h w)")
                else:
                    cur_codebook_caches = None

                new_cur_feats = []
                for cur_feat, new_feat in zip(cur_feats, new_feats):
                    new_cur_feats.append(
                        torch.cat([cur_feat, new_feat], dim=0)
                    )
                cur_feats = new_cur_feats

                resized = 0
                s = -1 if not self.cfg.low_res else 0

                cur_n_views = len(cur_indices)
                full_indices = torch.arange(cur_n_views, device=context['image'].device)[None]
            
                slide_mask = torch.zeros((1, cur_n_views), dtype=torch.bool, device=full_indices.device)
                
                if self.cfg.recon_mode == "incremental":
                    dist_matrix = calculate_distance_matrix(cur_extrinsics)[-1:]
                    _, indices = torch.topk(dist_matrix, min(num_context_views, n_views), largest=False, sorted=False, dim=1)
                elif self.cfg.recon_mode == "online":
                    indices = torch.arange(idx+1, idx + self.cfg.num_views + 1)[None].cuda()

                slide_mask.scatter_(1, indices, True)
                slide_mask[0, -1] = False
                src_indices = full_indices[slide_mask][None, None].repeat(b, 1, 1)
                
                matching_cur_feats = cur_feats[1][-1:]
                dim = matching_cur_feats.shape[-3]
                
                matching_src_feats = rearrange(cur_feats[1], "(b v) c h w -> b v c h w", b=b, v=cur_n_views)[:,None].gather(dim=2, index=src_indices[:,:,:,None,None,None] \
                                .repeat(1, 1, 1, dim, h//4, w//4)).view(-1, num_context_views-1, dim, h//4, w//4)
                    
                src_extrinsics = cur_extrinsics[:,None].gather(dim=2, index=src_indices[:,:,:,None,None].repeat(1,1,1,4,4))
                src_intrinsics = cur_intrinsics[:,None].gather(dim=2, index=src_indices[:,:,:,None,None].repeat(1,1,1,3,3))
      
                src_cam_t_world = src_extrinsics.inverse()
                cur_cam_t_world = cur_extrinsics[:, -1:].inverse()
                src_cam_T_cur_cam = src_cam_t_world @ cur_extrinsics[:, -1:].unsqueeze(2)
                cur_cam_T_src_cam = cur_cam_t_world.unsqueeze(2) @ src_extrinsics
                
                src_cam_T_cur_cam_ = rearrange(src_cam_T_cur_cam, 'b v n x y -> (b v) n x y')
                cur_cam_T_src_cam_ = rearrange(cur_cam_T_src_cam, 'b v n x y -> (b v) n x y')
                src_intrinsics_ = rearrange(src_intrinsics, 'b v n x y -> (b v) n x y')
                cur_intrinsics_ = rearrange(cur_intrinsics[:, -1:], 'b v x y -> (b v) x y')
                src_K = torch.eye(4, device=context['image'].device)[None,None].repeat(src_intrinsics_.shape[0], src_intrinsics_.shape[1],1,1)
                src_K[:,:,:3,:3] = src_intrinsics_
                cur_inverse = torch.eye(4, device=context['image'].device)[None].repeat(cur_intrinsics_.shape[0],1,1)
                cur_inverse[:,:3,:3] = cur_intrinsics_.inverse()

                results['cur_invK'] = cur_inverse
                results['src_K'] = src_K
                results['cur_wtc'] = cur_extrinsics[:, -1:]
                results['cur_ctw'] = cur_cam_t_world

                near = context["near"][:1,cur_indices[-1]].type_as(src_K).view(1, 1, 1, 1)
                far = context["far"][:1,cur_indices[-1]].type_as(src_K).view(1, 1, 1, 1)

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
                    [x[-1:] for x in cur_feats[1:]],
                )

                depth_outputs = self.depth_decoder(
                    [cur_feats[0][-1:]] + cost_volume_features,
                    imgs=cur_image
                )
                
                to_skip = cur_image
                skip = self.high_resolution_skip[s+1](to_skip)

                if not export_ply:
                    margin = 0
                    xy_ray, _ = sample_image_grid((h//(1+self.cfg.low_res), w//(1+self.cfg.low_res)), device)
                else:
                    margin = 8
                    xy_ray, _ = sample_image_grid(
                        (h // (1+self.cfg.low_res) - margin*2, w // (1+self.cfg.low_res) - margin*2), 
                        device
                    )
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
                
                gaussians_feats = rearrange(depth_outputs[f'output_pred_s{s}_b1hw'][:,1:], 'b c h w -> b () h w c')
                gaussians_feats = gaussians_feats + rearrange(skip, "b c h w -> b () h w c")

                if self.cfg.openvocab_domain in ['ensemble', '3d']:
                    if self.cfg.semantic_early_fusion:
                        clip_gaussians_feats_3d = gaussians_feats + self.semantic_proj(clip_gaussians_feats).permute(0, 2, 3, 1)[None]
                    else:
                        clip_gaussians_feats_3d = gaussians_feats.clone()
                    clip_gaussians_feats_3d = rearrange(clip_gaussians_feats_3d, "b v h w c -> b v (h w) c")
                else:
                    clip_gaussians_feats_3d = None

                densities = nn.Sigmoid()(rearrange(depth_outputs[f'output_pred_s{s}_b1hw'][:,:1], 'b c h w -> b () (c h w) () ()'))
      
                depth_outputs['densities'] = rearrange(densities, "b v x y z -> (b v) () x y z")
              
                if self.cfg.use_gt_depth:
                    depths_raw = context[f'depth_s0'][0, cur_indices[-1]:cur_indices[-1]+1]
                    depths = depth_outputs[f'depth_pred_s{s}_b1hw']
                    weights = depth_outputs[f'depth_weights']
                    
                    mask = (depths_raw > 1e-3) * (depths_raw < 10)
                    
                    depths[mask] = depths_raw[mask].float()
                    depth_outputs[f'depth_pred_s{s}_b1hw'] = depths
                    
                    weights[mask] = 1.
                    depth_outputs['depth_weights'] = weights

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
                offset_xy = torch.zeros_like(
                    rearrange(gaussians_feats[..., :2], "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces),
                    device=gaussians_feats.device
                )
                xy_ray = xy_ray + offset_xy
                
                coords = self.gaussian_adapter.forward(
                    rearrange(context["extrinsics"][:, cur_indices[-1]:cur_indices[-1]+1], "b v i j -> b v () () () i j"),
                    rearrange(context["intrinsics"][:, cur_indices[-1]:cur_indices[-1]+1], "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    densities,
                    gaussians_feats,
                    (h // (1+self.cfg.low_res) - margin*2, w // (1+self.cfg.low_res) - margin*2),
                    load_depth=self.cfg.load_depth,
                    fusion=True,
                )
        
                num_raw_gaussians = gaussians_feats.shape[2] * gaussians_feats.shape[1]
                B = gaussians_feats.shape[0]
                for bb in range(B):
                    cur_gs = gaussians_feats[bb:bb+1]
                    cur_clip_gs_3d = clip_gaussians_feats_3d[bb:bb+1] if clip_gaussians_feats_3d is not None else None
                    cur_coords = coords[bb:bb+1]
                    cur_densities = densities[bb:bb+1]
                    cur_weights = weights[bb:bb+1]
                    cur_depth = rearrange(depth_outputs[f'depth_pred_s{s}_b1hw'], "(b v) c h w -> b v c h w", b=B)[bb]

                    globals, new_gaussian_mask = self.fuse_gaussians(
                        cur_gs,
                        cur_clip_gs_3d,
                        cur_codebook_caches,
                        cur_coords, 
                        cur_densities, 
                        cur_weights,
                        cur_depth, 
                        context["extrinsics"][bb:bb+1, cur_indices[-1]:cur_indices[-1]+1],
                        context["intrinsics"][bb:bb+1, cur_indices[-1]:cur_indices[-1]+1], 
                        (h // (1+self.cfg.low_res) - margin*2, w // (1+self.cfg.low_res) - margin*2),
                        vis=self.cfg.vis,
                        globals=globals,
                        remove=self.cfg.remove,
                        use_em=self.cfg.use_em,
                        decoder=decoder,
                        near=context["near"][bb:bb+1, cur_indices[-1]:cur_indices[-1]+1], 
                        far=context["far"][bb:bb+1, cur_indices[-1]:cur_indices[-1]+1],
                        img=cur_image[bb:bb+1][:, None],
                        fusion=self.cfg.fusion,
                        fore_fusion=self.cfg.fore_fusion,
                        concat_depth=self.cfg.concat_depth,
                        use_gru=self.cfg.use_gru,
                        export_ply=(export_ply or self.cfg.ft),
                        output_new_idx=True
                    )

                    if self.cfg.vis_gs:
                        gs_model = GaussianModel(sh_degree=self.cfg.sh_degree)
                        gs_model.load_gs(globals)
                        gs_model.save_ply(output_path / 'gaussians' / f'{scene}_{batch}.ply')
                        
                    if self.cfg.vis:
                        cur_gaussians = torch.cat(cur_gaussians, dim=1)
                        N = cur_gaussians.shape[1]
                        harmonics = torch.zeros((B, N, 3, 9), device=cur_gaussians.device)
                        harmonics[:, :, -1, 0] = 1
                        results['gaussians'] = Gaussians(
                            cur_gaussians,
                            torch.zeros([3,3], device=cur_gaussians.device).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1),
                            harmonics,
                            torch.ones((B, N), device=cur_gaussians.device),
                        )
                        results['num_gaussians'] = N
                        print('means.shape:', cur_gaussians.shape)
                        return results
                
                depth_preds_all = torch.cat(depth_outputs_all[f'depth_pred_s{s}_b1hw'], dim=0)
                densities_preds_all = torch.cat(depth_outputs_all[f'densities'], dim=0)
                refine_window = self.cfg.refine_window if depth_preds_all.shape[0] >= self.cfg.refine_window else depth_preds_all.shape[0]
                    
            depth_outputs_all[f'depth_pred_s{s}_b1hw'] = torch.cat(depth_outputs_all[f'depth_pred_s{s}_b1hw'], dim=0)
            depth_outputs_all[f'densities'] = torch.cat(depth_outputs_all[f'densities'], dim=0)
            
            if self.cfg.refine_gs:
                for _ in range(self.cfg.refine_times):
                    for i in range(len(depth_outputs_all[f'depth_pred_s{s}_b1hw'])):
                        cur_depth = depth_outputs_all[f'depth_pred_s{s}_b1hw'][i:i+1]
                        cur_weights = depth_outputs_all[f'densities'][i:i+1]
                        if self.cfg.refine_uniform:
                            cur_weights = torch.ones_like(cur_weights, device=cur_weights.device)
                            globals.densities = torch.ones_like(globals.densities, device=globals.densities.device)

                        globals = self.refine_gaussians(
                            globals, 
                            cur_depth, 
                            cur_weights, 
                            context["extrinsics"][:, i:i+1], 
                            context["intrinsics"][:, i:i+1], 
                            (h // (1+self.cfg.low_res) - margin*2, w // (1+self.cfg.low_res) - margin*2),
                            ws=self.cfg.refine_ws, 
                            depth_thres=self.cfg.refine_thres,
                            soft_thres=self.cfg.refine_soft_thres, 
                            refine_pp=self.cfg.refine_pp,
                            export_ply=(export_ply or self.cfg.ft),
                            num=i,
                        )
            
            if test_bev:
                edges_means_collate = []
                edges_covariances_collate = []
                edges_harmonics_collate = []
                edges_opacities_collate = []
                edges_conf_collate = []
                for j, cur_indice in enumerate(cur_indices):
                    if j % 4 != 0 and j != len(cur_indices) -1:
                        continue
                    edges_means = create_transformed_pyramid(context['extrinsics'][0,cur_indice].cpu().numpy())
                    edges_means = torch.from_numpy(edges_means).unsqueeze(0).to(globals.means.device)
                    edges_covariances = 0.0001 * torch.eye(3, device=globals.means.device)[None, None].repeat(1, edges_means.shape[1], 1, 1)
                    edges_harmonics = torch.zeros([1, edges_means.shape[1], 3, globals.harmonics.shape[-1]], device=globals.means.device)
                    edges_opacities = torch.ones([1, edges_means.shape[1]], device=globals.means.device)
                    edges_conf = torch.ones([1, edges_means.shape[1]], device=globals.means.device)
                    edges_harmonics[:, :, 0, 0] = 1.5
                        
                    edges_means_collate.append(edges_means)
                    edges_covariances_collate.append(edges_covariances)
                    edges_harmonics_collate.append(edges_harmonics)
                    edges_opacities_collate.append(edges_opacities)
                    edges_conf_collate.append(edges_conf)
                    
                edges_means = torch.cat(edges_means_collate, dim=1)
                edges_covariances = torch.cat(edges_covariances_collate, dim=1)
                edges_harmonics = torch.cat(edges_harmonics_collate, dim=1)
                edges_opacities = torch.cat(edges_opacities_collate, dim=1)
                edges_conf = torch.cat(edges_conf_collate, dim=1)
                    
                gaussians = Gaussians(
                    means=torch.cat([globals.means[:, globals.valid[0]], edges_means], dim=1),
                    covariances=torch.cat([globals.covariances[:, globals.valid[0]], edges_covariances], dim=1),
                    harmonics=torch.cat([globals.harmonics[:, globals.valid[0]], edges_harmonics], dim=1),
                    opacities=torch.cat([globals.opacities[:, globals.valid[0]], edges_opacities], dim=1),
                    conf=torch.cat([globals.densities[:, globals.valid[0], 0, 0], edges_conf], dim=1)
                )
                output_bev = decoder.forward(
                    gaussians,
                    target["bev_extrinsics"],
                    target["intrinsics"][:, :1],
                    target["near"][:, :1],
                    target["far"][:, :1],
                    (1440, 1920),
                    depth_mode='depth',
                    scale_invariant=True,
                    background_color=torch.tensor([1, 1, 1], device=device).float().to(globals.means.device),
                )
                color = output_bev.color[0][0]
                save_image(color, os.path.join("", "whole.png"))
            
            our_gaussians = [
                Gaussians(
                    means=globals.means[:, globals.valid[0]], 
                    covariances=globals.covariances[:, globals.valid[0]], 
                    harmonics=globals.harmonics[:, globals.valid[0]], 
                    opacities=globals.opacities[:, globals.valid[0]],
                    conf=globals.densities[:, globals.valid[0],0,0],
                    clip_features_3d = globals.clip_features_3d[:, globals.valid[0]] if self.cfg.openvocab_domain in ['ensemble', '3d'] else None,
                    clip_features_2d = {
                        "feat": globals.clip_features_2d["feat"],
                        "idx": globals.clip_features_2d["idx"][:, globals.valid[0]],
                        "weight": globals.clip_features_2d["weight"][:, globals.valid[0]],
                        "invalid": globals.clip_features_2d["invalid"][:, globals.valid[0]],
                    } if self.cfg.openvocab_domain in ['ensemble', '2d'] else None
                )
            ]
            
            save_path = os.path.join(self.cfg.output_path, scene[:-2])
            
            if self.cfg.use_gt_depth:
                save_path = save_path.replace(os.path.basename(self.cfg.output_path), os.path.basename(self.cfg.output_path) + "_gtdepth")

            os.makedirs(save_path, exist_ok=True)
            
            if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                torch.save(globals, os.path.join(save_path, "ckpt_gs.pt"))
            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                torch.save(self.to_semantic(globals.clip_features_3d[:, globals.valid[0]]), os.path.join(save_path, "ckpt_semantic.pt"))

            num_gaussians = our_gaussians[0].means.shape[1]
            results['gs_ratio'] = num_gaussians / num_raw_gaussians

            try:
                depths_raw = rearrange(context[f'depth_s{s}'], "b v c h w -> b v (h w) c 1")
                results[f'depth_num{num}_s{s}_raw'] = depths_raw
                mask = (depths_raw > 1e-3) * (depths_raw < 10)
                results[f'depth_num{num}_s{s}_mask'] = mask
            except:
                pass

            depths_raw = rearrange(context[f'depth_s{s}'], "b v c h w -> (b v) c h w")
            results[f'depth_num{num}_s{s}_raw_b1hw'] = depths_raw
            mask = (depths_raw > 1e-3) * (depths_raw < 10)
            results[f'depth_num{num}_s{s}_mask_b1hw'] = mask

            depths = depth_outputs_all[f'depth_pred_s{s}_b1hw']
            results[f'log_depth_num{num}_s{s}'] = rearrange(torch.log(depths), "(b v) c h w -> b v (c h w) () ()", b=b)
            results[f'depth_num{num}_s{s}'] = rearrange(depths, "(b v) c h w -> b v (c h w) () ()", b=b)

            results[f'depth_num{num}_s{s}_b1hw'] = depths
            results[f'log_depth_num{num}_s{s}_b1hw'] = torch.log(depths)
            results[f'weight_num{num}_s{s}_b1hw'] = depth_outputs[f'output_pred_s{s}_b1hw'][:,:1]

        # Optionally apply a per-pixel opacity.
        opacity_multiplier = (
            rearrange(self.to_opacity(features), "b v r () -> b v r () ()")
            if self.cfg.predict_opacity
            else 1
        )

        visualization_dump = {}
        
        if export_ply:
            cov3D = globals.covariances[0, globals.valid[0]]
            eigvals, eigvecs = torch.linalg.eigh(cov3D.cpu())
            scales = torch.sqrt(eigvals).cuda()
            rotations = rotmat_to_quat(eigvecs.cuda())

            globals.scales = scales
            globals.rotations = rotations
            
            with torch.inference_mode(not self.cfg.ft):
                gs_model = GaussianModel(sh_degree=self.cfg.sh_degree)
                if self.cfg.est_depth == 'cost':
                    gs_model.load_gs(globals)
                else:
                    gs_model.load_gs_(our_gaussians[0], all_scales, all_rotations)
        
            gs_model.save_ply("gs.ply")
        
        results['visualizations'] = visualization_dump
        results['gaussians'] = our_gaussians
        final_num_gaussians = our_gaussians[0].means.shape[1]
        results['num_gaussians'] = final_num_gaussians

        return results
    
    def backbone_3d(self, coords, feats, voxelization_matrix, rotation_matrix, globals=None):
        if globals is not None:
            global_coords = globals.means[:, globals.valid[0]][0]
            global_feats = globals.clip_features_3d[:, globals.valid[0]][0]

            global_homo_coords = torch.cat([global_coords, torch.ones(global_coords.shape[0], 1).to(self.device)], dim=-1)

            rigid_transformation = rotation_matrix @ voxelization_matrix
            global_voxel_coords = torch.floor(global_homo_coords @ rigid_transformation.T[:, :3])

            voxelization_dict = {   
                "return_index": True,
                "return_inverse": True,
                "coordinates": global_voxel_coords.contiguous(),
                "features": global_feats,
            }

            _, _, unique_map, _ = ME.utils.sparse_quantize(
                **voxelization_dict
            )

            input_dict = {"coords": [global_voxel_coords[unique_map].int()], "feats": [global_feats[unique_map]]}
            coordinates, features = ME.utils.sparse_collate(**input_dict)
            accumulated_feats = ME.SparseTensor(
                coordinates=coordinates,
                features=features,
                device=self.device,
            )
        else:
            accumulated_feats = None

        homo_coords = torch.cat([coords, torch.ones(coords.shape[0], 1).to(self.device)], dim=-1)
        
        rigid_transformation = rotation_matrix @ voxelization_matrix
        voxel_coords = torch.floor(homo_coords @ rigid_transformation.T[:, :3])

        voxelization_dict = {   
            "return_index": True,
            "return_inverse": True,
            "coordinates": voxel_coords.contiguous(),
            "features": feats,
        }
                
        _, _, unique_map, inverse_map = ME.utils.sparse_quantize(
            **voxelization_dict
        )

        input_dict = {"coords": [voxel_coords[unique_map].int()], "feats": [feats[unique_map]]}
        coordinates, features= ME.utils.sparse_collate(**input_dict)
        sinput = ME.SparseTensor(
            coordinates=coordinates,
            features=features,
            device=self.device,
        )
                
        global_clip_gaussians_feat_3d = self.unet3d(sinput, self.memory, accumulated_feats)
        global_clip_gaussians_feat_3d = global_clip_gaussians_feat_3d.decomposed_features[0][None]
        
        return global_clip_gaussians_feat_3d[:, inverse_map, :]

    def extract_fastsam_clip_feats(self, images, caches, image_shape, globals):
        h, w = image_shape

        with torch.no_grad():
            if self.cfg.clip_model in ["lseg", "llava_ov"]:
                everything_results = [
                    torch.load(cache[0], weights_only=False) for cache in caches
                ]
            else:
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

            codebook_caches = []  
            clip_img_feats = []
            for idx, everything_result in enumerate(everything_results):
                if self.cfg.clip_model in ["lseg", "llava_ov"]:
                    format_results = everything_result['fastsam'][0]
                else:
                    format_results, masks_torch = prompt._format_results(everything_result, 0, sort=True)
                    
                if self.cfg.clip_model == "maskclip":
                    clip_feats_flat = rearrange(clip_feats[idx], "d h w -> (h w) d").to(self.device).float()
                    
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
                elif self.cfg.clip_model == "openseg":
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
                elif self.cfg.clip_model in ["lseg", "llava_ov"]:
                    mask_feats = everything_result['clip'][0].to(self.device)

                if self.cfg.openvocab_domain in ['ensemble', '2d']:
                    globals.clip_features_2d['feat'].append(mask_feats)
                    globals.clip_features_2d['img_stamp'].append(globals.clip_features_2d['img_stamp'][-1] + mask_feats.shape[0])
                    codebook_cache = torch.ones((h, w)).to(self.device) * -1

                if self.cfg.openvocab_domain in ['ensemble', '3d']:
                    clip_img_feat = torch.zeros((h, w, mask_feats.shape[-1])).to(self.device)
                    
                for jdx, (feat, mask) in enumerate(zip(mask_feats, format_results)):
                    if self.cfg.openvocab_domain in ['ensemble', '3d']:
                        clip_img_feat[mask['segmentation'], :] = feat.float()

                    if self.cfg.openvocab_domain in ['ensemble', '2d']:
                        codebook_cache[mask['segmentation']] = jdx + globals.clip_features_2d['img_stamp'][-2]
                
                if self.cfg.openvocab_domain in ['ensemble', '3d']:
                    clip_img_feats.append(clip_img_feat)

                if self.cfg.openvocab_domain in ['ensemble', '2d']:
                    codebook_caches.append(codebook_cache)

            if self.cfg.openvocab_domain in ['ensemble', '3d']:
                clip_img_feats = torch.nn.functional.normalize(torch.stack(clip_img_feats), dim=-1, eps=1e-5)
                clip_img_feats = clip_img_feats.permute(0, 3, 1, 2)

            if self.cfg.openvocab_domain in ['ensemble', '2d']:
                codebook_caches = torch.stack(codebook_caches, dim=0)

        if self.cfg.openvocab_domain == "ensemble":
            return clip_img_feats, codebook_caches, globals
        elif self.cfg.openvocab_domain == "2d":
            return None, codebook_caches, globals
        elif self.cfg.openvocab_domain == "3d":
            return clip_img_feats, None, globals

    def extract_sam_clip_feats(self, images, caches, image_shape, globals):
        h, w = image_shape

        with torch.no_grad():
            inputs = (images * 255.).clone().permute(0, 2, 3, 1).cpu().numpy()            

            masks_collate = []
            feats_collate = []
            for input in inputs:
                seg_images, seg_maps = self.sam.sam_encoder(input)
                tiles = seg_images['l']
                tiles = tiles.to("cuda")

                tiles = tiles.reshape(-1, 3, 224, 224)
                clip_embed = self.clip.encode_image(tiles)
                clip_embed = clip_embed.reshape(clip_embed.shape[0] // 4, 4, 512)
                clip_embed = clip_embed.mean(dim=1)

                feats_collate.append(clip_embed)

                masks = []
                for mask_idx in np.unique(seg_maps['l']):
                    if mask_idx == -1:
                        continue

                    masks.append(seg_maps['l'] == mask_idx)
                
                masks = torch.from_numpy(np.stack(masks)).to(self.device)
                masks_collate.append(masks)

            codebook_caches = []
            clip_img_feats = []
            for mask, feats in zip(masks_collate, feats_collate):
                if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                    globals.clip_features_2d['feat'].append(feats)
                    globals.clip_features_2d['img_stamp'].append(globals.clip_features_2d['img_stamp'][-1] + feats.shape[0])
                    codebook_cache = torch.ones((h, w)).to(self.device) * -1
                
                if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                    clip_img_feat = torch.zeros((h, w, feats.shape[-1])).to(self.device)

                for jdx, (feat_, mask_) in enumerate(zip(feats, mask)):
                    if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                        clip_img_feat[mask_, :] = feat_.float()

                    if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                        codebook_cache[mask_] = jdx + globals.clip_features_2d['img_stamp'][-2]

                if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                    clip_img_feats.append(clip_img_feat)

                if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                    codebook_caches.append(codebook_cache)

            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                clip_img_feats = torch.nn.functional.normalize(torch.stack(clip_img_feats), dim=-1, eps=1e-5)
                clip_img_feats = clip_img_feats.permute(0, 3, 1, 2)

            if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                codebook_caches = torch.stack(codebook_caches, dim=0)

        if self.cfg.openvocab_domain == "ensemble":
            return clip_img_feats, codebook_caches, globals
        elif self.cfg.openvocab_domain == "2d":
            return None, codebook_caches, globals
        elif self.cfg.openvocab_domain == "3d":
            return clip_img_feats, None, globals
        
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

    def refine_gaussians(
        self, 
        globals, 
        depths, 
        weights, 
        extrinsics, 
        intrinsics, 
        image_shape, 
        depth_thres=0.3, 
        ws=True, 
        soft_thres=False, 
        refine_pp=False, 
        export_ply=False, 
        num=0,
        new_gaussian_mask=None
    ):
        depths = rearrange(depths, "v c h w -> v (c h w)")
        h, w = image_shape
        
        # global_gaussians = globals['gs']
        global_coords = globals.coords[:, globals.valid[0]]
        global_weights = globals.densities[:, globals.valid[0]]

        i = 0
        extrinsic = extrinsics[0,i]
        intrinsic = intrinsics[0,i].clone()
        intrinsic[:1,:] *= w
        intrinsic[1:2,:] *= h
        focal_length = (intrinsic[0, 0], intrinsic[1, 1])
        principal_point = (intrinsic[0, 2], intrinsic[1, 2])
        principal_point_mat = torch.tensor([principal_point[0], principal_point[1]]).to(intrinsic.device)
        principal_point_mat = principal_point_mat.reshape(1, 2)
        focal_length_mat = torch.tensor([focal_length[0], focal_length[1]]).to(intrinsic.device)
        focal_length_mat = focal_length_mat.reshape(1, 2)
        means1 = torch.cat([global_coords[0], torch.ones_like(global_coords[..., :1][0])], dim=-1).permute(1,0) # [4, 196608]
        post_xy_coords = torch.matmul(extrinsic.inverse(), means1)[:3]
        curr_depths = post_xy_coords[2:3, :]
        post_xy_coords = (post_xy_coords / curr_depths)[:2].permute(1,0)
        curr_depths = curr_depths.squeeze()
        post_xy_coords = post_xy_coords * focal_length_mat.reshape(1,2) + principal_point_mat # [196608, 2]
        pixel_coords = post_xy_coords.round().long()[:,[1,0]]
        valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < h) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < w) & (curr_depths > 0)
        proj_map = - torch.ones((h*w), device=depths.device, dtype=curr_depths.dtype)
        depth_map = torch.ones((h*w), device=depths.device, dtype=curr_depths.dtype) * 10000

        pixel_indices = (pixel_coords[:, 1] + pixel_coords[:, 0]*w)[valid]
        depth_map.scatter_reduce_(0, pixel_indices, curr_depths[valid], reduce='amin')
        
        if soft_thres:
            remove_mask = depths[i] - depth_map > 0
        else:
            remove_mask = depths[i] - depth_map > depth_thres
            
        proj_map = torch.where(depth_map[pixel_indices] == curr_depths[valid])[0]
        fusion_indices = torch.where(remove_mask[pixel_indices])[0]
        fusion_indices_ = fusion_indices[torch.isin(fusion_indices, proj_map)]
        corr_indices = proj_map[torch.isin(proj_map, fusion_indices)]
        valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
        valid_indices.scatter_(0, corr_indices, True)
        mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
        mask[valid] = valid_indices

        if refine_pp:
            curr_depths = curr_depths[valid]
            N = len(curr_depths)
            h, w = 384, 512  # Example dimensions for depth_map
            # gs_feat = torch.randn(N, 3, device=curr_depths.device)
            gs_weights = global_weights[0, valid]
            gs_coords = global_coords[0, valid]

            expanded_depths = depths[i][pixel_indices.squeeze()]

            # Compute the differences and apply the condition
            diff = curr_depths - expanded_depths
            condition_mask = torch.abs(diff) < 0.1

            # Filter indices using the condition mask
            valid_indices = torch.nonzero(condition_mask).squeeze()

            # We need to adjust the indices map to only include the valid indices
            filtered_pixel_indices = pixel_indices[valid_indices].squeeze()
            filtered_gaussian_indices = valid_indices

            # Re-create the sparse tensor with the new filtered indices
            sparse_indices = torch.stack([filtered_pixel_indices, filtered_gaussian_indices], dim=0)
            values = torch.ones_like(filtered_gaussian_indices, dtype=torch.float32)  # We can use ones as dummy values

            if sparse_indices.dim() == 1:
                sparse_indices = sparse_indices.unsqueeze(1)

            # Build the new sparse tensor
            new_indices_map = torch.sparse_coo_tensor(sparse_indices, values, size=(h * w, N))

            # Ensure that the sparse tensor is coalesced
            new_indices_map = new_indices_map.coalesce()

            # Retrieve all non-zero indices (which are the valid Gaussians for each pixel)
            gaussian_indices = new_indices_map.indices()[1]
            new_pixel_indices = new_indices_map.indices()[0]

            # Gather Gaussian features using the valid indices
            gaussian_weights = gs_weights[gaussian_indices]

            # Compute the sum of features for each pixel via a scatter operation
            weight_sums = torch.zeros((h * w, 1, 1), device=curr_depths.device)
            weight_sums.index_add_(0, new_pixel_indices, gaussian_weights)

            curr_weights = weight_sums.unsqueeze(0)
            
            # Find the global gs that are near the closest ones
            expanded_depth_map = depth_map[pixel_indices.squeeze()]
            diff = curr_depths - expanded_depth_map
            condition_mask = torch.abs(diff) < 0.1

            # Filter indices using the condition mask
            valid_indices = torch.nonzero(condition_mask).squeeze()

            # We need to adjust the indices map to only include the valid indices
            filtered_pixel_indices = pixel_indices[valid_indices].squeeze()
            filtered_gaussian_indices = valid_indices

            # Re-create the sparse tensor with the new filtered indices
            sparse_indices = torch.stack([filtered_pixel_indices, filtered_gaussian_indices], dim=0)
            values = torch.ones_like(filtered_gaussian_indices, dtype=torch.float32)  # We can use ones as dummy values

            if sparse_indices.dim() == 1:
                sparse_indices = sparse_indices.unsqueeze(1)

            # Build the new sparse tensor
            new_indices_map = torch.sparse_coo_tensor(sparse_indices, values, size=(h * w, N))

            # Ensure that the sparse tensor is coalesced
            new_indices_map = new_indices_map.coalesce()

            # Retrieve all non-zero indices (which are the valid Gaussians for each pixel)
            gaussian_indices = new_indices_map.indices()[1]
            new_pixel_indices = new_indices_map.indices()[0]

            # Gather Gaussian features using the valid indices
            gaussian_weights = gs_weights[gaussian_indices]

            # Compute the sum of features for each pixel via a scatter operation
            weight_sums = torch.zeros((h * w, 1, 1), device=curr_depths.device)
            weight_sums.index_add_(0, new_pixel_indices, gaussian_weights)

            curr_global_weights = global_weights[:, mask]
            to_add = weight_sums[pixel_indices][fusion_indices_]
            # curr_global_weights = curr_global_weights + to_add
            curr_global_weights = to_add
            multiplier = curr_global_weights[:, 0, 0] / (curr_global_weights[:, 0, 0]+ curr_weights[:, pixel_indices][:, fusion_indices_, 0, 0]+ \
                                                         weights[:, i, pixel_indices][:,fusion_indices_, 0, 0])
        else:
            multiplier = global_weights[:, mask, 0, 0] / (global_weights[:, mask, 0, 0]+ weights[:, i, pixel_indices][:,fusion_indices_, 0, 0])

        if new_gaussian_mask is not None:
            filtered_mask = mask & new_gaussian_mask
            multiplier_mask = filtered_mask[mask]
            mask = filtered_mask
        else:
            multiplier_mask = None
            
        if mask.sum() > 0 and (not self.cfg.vis_refine):
            if ws:
                if multiplier_mask is not None:
                    multiplier = multiplier[:, multiplier_mask]
                    
                globals.append(
                    means=globals.means[:, globals.valid[0]][:, mask],
                    covariances=globals.covariances[:, globals.valid[0]][:, mask],
                    harmonics=globals.harmonics[:, globals.valid[0]][:, mask],
                    opacities=globals.opacities[:, globals.valid[0]][:, mask] * multiplier,
                    features=None,
                    clip_features_3d=None,
                    densities=globals.densities[:, globals.valid[0]][:, mask],
                    weights=None,
                    coords=globals.coords[:, globals.valid[0]][:, mask],
                    extrinsics=None,
                    depths=None,
                    codebooks_idx=None,
                    codebooks_weights=None,
                    mask=mask,
                    scales=globals.scales[:, globals.valid[0]][:, mask] if export_ply else None,
                    rotations=globals.rotations[:, globals.valid[0]][:, mask] if export_ply else None,
                    features_invalid=torch.ones_like(globals.clip_features_2d["invalid"][:, globals.valid[0]][:, mask], device=self.device) * -1
                )
            else:
                globals.append(
                    means=globals.means[:, globals.valid[0]][:, mask],
                    covariances=globals.covariances[:, globals.valid[0]][:, mask],
                    harmonics=globals.harmonics[:, globals.valid[0]][:, mask],
                    opacities=globals.opacities[:, globals.valid[0]][:, mask] * 0,
                    features=None,
                    clip_features_3d=None,
                    densities=globals.densities[:, globals.valid[0]][:, mask],
                    weights=None,
                    coords=globals.coords[:, globals.valid[0]][:, mask],
                    extrinsics=None,
                    depths=None,
                    codebooks_idx=None,
                    codebooks_weights=None,
                    mask=mask,
                    scales=globals.scales[:, globals.valid[0]][:, mask] if export_ply else None,
                    rotations=globals.rotations[:, globals.valid[0]][:, mask] if export_ply else None,
                    features_invalid=torch.ones(globals.clip_features_2d["invalid"][:, globals.valid[0]][:, mask], device=self.device) * -1
                )
        
        return globals


    def fuse_gaussians(
        self, 
        gaussians,
        clip_gaussians_3d,
        codebook_caches,
        coords, 
        densities, 
        weight_emb, 
        depths, 
        extrinsics, 
        intrinsics, 
        image_shape, 
        depth_thres=0.1, 
        limit=100,
        vis=False, 
        globals=None, 
        remove=False, 
        use_em=False, 
        decoder=None,
        near=0.5, 
        far=5.0, 
        img=None, 
        fusion=True, 
        fore_fusion=False, 
        concat_depth=False,
        use_gru=True, 
        depth_fore=15.0, 
        export_ply=False,
        output_new_idx=False
    ):
        
        length = min(gaussians.shape[1], limit)
        depths = rearrange(depths, "v c h w -> v (c h w)")
        initial = globals.valid.sum() == 0
        h, w = image_shape

        if initial:
            global_gaussians_feat = gaussians[:,0]
            
            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                global_clip_gaussians_feat_3d = clip_gaussians_3d[:, 0]
            else:
                global_clip_gaussians_feat_3d = None
            
            if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                global_codebook_caches = codebook_caches[:, 0]
                global_codebook_weights = torch.ones_like(global_codebook_caches).to(self.device)
                invalid = global_codebook_caches == -1
                global_codebook_weights[:, invalid[0]] = 0.
            else:
                global_codebook_caches = None
                global_codebook_weights = None

            global_densities = densities[:, 0]
            global_weight_emb = weight_emb[:, 0]
            global_coords = coords[:,0,:,0,0]
            
            if self.cfg.unet_3d and self.cfg.openvocab_domain in ["ensemble", "3d"]:
                self.M_v, self.M_r = self.get_transformation_matrix()
                global_clip_gaussians_feat_3d = self.backbone_3d(
                    global_coords[0], 
                    global_clip_gaussians_feat_3d[0], 
                    self.M_v, 
                    self.M_r
                )
            
            global_extrinsics = extrinsics[:,0][:,None].repeat(1,global_gaussians_feat.shape[1],1,1)
            global_depths = depths[None, 0]

            global_gaussians = rearrange(
                self.to_gaussians(global_gaussians_feat),
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
                                
            global_gaussians = self.gaussian_adapter.forward(
                rearrange(global_extrinsics, "b r i j -> b () r () () i j"),
                repeat(intrinsics[:,0], "b i j -> b () N () () i j", N=global_gaussians.shape[1]),
                None,
                rearrange(global_depths, "b r -> b () r () ()"),
                nn.Sigmoid()(rearrange(global_gaussians[..., :1], "b r srf c -> b () r srf c")),
                rearrange(global_gaussians[..., 2:], "b r srf c -> b () r srf () c"),
                (h, w),
                load_depth=self.cfg.load_depth,
                fusion=False,
                coords=rearrange(global_coords, "b r c -> b () r () () c"),
            )

            globals.append(
                means=rearrange(
                    global_gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                covariances=rearrange(
                    global_gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                harmonics=rearrange(
                    global_gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                opacities=rearrange(
                    global_gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
                features=global_gaussians_feat,
                clip_features_3d=global_clip_gaussians_feat_3d,
                densities=global_densities,
                weights=global_weight_emb,
                coords=global_coords,
                extrinsics=global_extrinsics,
                depths=global_depths,
                codebooks_idx=global_codebook_caches,
                codebooks_weights=global_codebook_weights,
                mask=None,
                scales=rearrange(
                    global_gaussians.scales,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ) if export_ply else None,
                rotations=rearrange(
                    global_gaussians.rotations,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ) if export_ply else None,
            )
        else:
            pass

        removed = []

        for i in range(initial, length):
            # start = time.time()
            extrinsic = extrinsics[0,i]
            intrinsic = intrinsics[0,i].clone()
            intrinsic[:1,:] *= w
            intrinsic[1:2,:] *= h
            focal_length = (intrinsic[0, 0], intrinsic[1, 1])
            principal_point = (intrinsic[0, 2], intrinsic[1, 2])
            principal_point_mat = torch.tensor([principal_point[0], principal_point[1]]).to(intrinsic.device)
            principal_point_mat = principal_point_mat.reshape(1, 2)
            focal_length_mat = torch.tensor([focal_length[0], focal_length[1]]).to(intrinsic.device)
            focal_length_mat = focal_length_mat.reshape(1, 2)
            global_coords = globals.coords[:, globals.valid[0]]
            means1 = torch.cat([global_coords[0], torch.ones_like(global_coords[..., :1][0])], dim=-1).permute(1,0) # [4, 196608]
            post_xy_coords = torch.matmul(extrinsic.inverse(), means1)[:3]
            curr_depths = post_xy_coords[2:3, :]
            post_xy_coords = (post_xy_coords / curr_depths)[:2].permute(1,0)
            curr_depths = curr_depths.squeeze()
            post_xy_coords = post_xy_coords * focal_length_mat.reshape(1,2) + principal_point_mat # [196608, 2]
            pixel_coords = post_xy_coords.round().long()[:,[1,0]]
            valid = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < h) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < w) & (curr_depths > 0)
            proj_map = - torch.ones((h*w), device=coords.device, dtype=curr_depths.dtype)
            depth_map = torch.ones((h*w), device=coords.device, dtype=curr_depths.dtype) * 10000

            pixel_indices = (pixel_coords[:, 1] + pixel_coords[:, 0]*w)[valid]
            depth_map.scatter_reduce_(0, pixel_indices, curr_depths[valid], reduce='amin')

            if not fore_fusion:
                fusion_mask = torch.abs(depth_map - depths[i]) < torch.clamp_min(depths[i] * 0.05, depth_thres)
            else:
                fusion_mask = (depths[i] - depth_map > - torch.clamp_min(depths[i] * 0.05, depth_thres)) \
                                & (depths[i] - depth_map < torch.clamp_min(depths[i] * 0.05, depth_fore))

            concat_mask = torch.ones_like(depths[i], device=depths[i].device, dtype=torch.bool)
            if fusion:
                concat_mask = ~fusion_mask

            if remove:
                remove_mask = depths[i] - depth_map > torch.clamp_min(depths[i] * 0.05, depth_thres)
                concat_mask = concat_mask & (~remove_mask)
            
            if use_em:
                render = decoder.forward(
                    global_gaussians,
                    extrinsics[:,i:i+1],
                    intrinsics[:,i:i+1],
                    near[:,i:i+1],
                    far[:,i:i+1],
                    (h, w),
                    depth_mode='depth',
                ).color
                
                gt = img[:,i]
                error_map = reduce((gt - render) ** 2, "b v c h w -> b v h w", "mean")

                error_mask = (error_map > 10**(-2.5)).view(-1)
                concat_mask = concat_mask & error_mask

                num_imgs = len(os.listdir('./debug'))
                render = (render.squeeze().permute(1, 2, 0) * 255).byte()
                render = Image.fromarray(render.detach().cpu().numpy(), 'RGB')

                render.save(f'debug/render_{num_imgs}.png')
                
            proj_map = torch.where(depth_map[pixel_indices] == curr_depths[valid])[0]
            fusion_indices = torch.where(fusion_mask[pixel_indices])[0]
            fusion_indices_ = fusion_indices[torch.isin(fusion_indices, proj_map)]
            corr_indices = proj_map[torch.isin(proj_map, fusion_indices)]
            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices
            
            if self.cfg.unet_3d and self.cfg.openvocab_domain in ["ensemble", "3d"]:
                clip_gaussians_3d_unet = self.backbone_3d(coords[0, i, :, 0, 0], clip_gaussians_3d[0, i], self.M_v, self.M_r, globals)
            else:
                clip_gaussians_3d_unet = None
                
            if mask.sum() > 0 and fusion:
                input_weights_emb = positional_encoding(torch.cat([globals.densities[:, globals.valid[0]][:, mask], weight_emb[:, i, pixel_indices][:,fusion_indices_]], dim=-1), 6)
                hidden_weights_emb = positional_encoding(torch.cat([densities[:, i, pixel_indices][:,fusion_indices_], globals.weights[:, globals.valid[0]][:, mask]], dim=-1), 6)
                local_latent = gaussians[:, i, pixel_indices][:,fusion_indices_].unsqueeze(2)
                global_latent = globals.features[:, globals.valid[0]][:, mask].unsqueeze(2)
                local_depth = depths[i][pixel_indices][fusion_indices_]
                global_depth = depth_map[pixel_indices][fusion_indices_]

                if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                    local_clip_latent_3d = clip_gaussians_3d_unet[:, pixel_indices][:,fusion_indices_].unsqueeze(2)
                    global_clip_latent_3d = globals.clip_features_3d[:, globals.valid[0]][:, mask].unsqueeze(2)
                else:
                    local_clip_latent_3d = None
                    global_clip_latent_3d = None
                    
                weights_0 = globals.densities[:, globals.valid[0]][:, mask].repeat(1, 1, 1, 2)
                weights_1 = densities[:, i, pixel_indices][:,fusion_indices_].repeat(1, 1, 1, 2)

                if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                    local_codebook_caches = codebook_caches[:, i, pixel_indices][:,fusion_indices_]
                    local_codebook_weights = (weights_1[..., 1] / (weights_0[..., 1] + weights_1[..., 1]))[..., 0]
                    invalid = local_codebook_caches == -1
                    local_codebook_weights[:, invalid[0]] = 0.
                else:
                    local_codebook_caches = None
                    local_codebook_weights = None

                if use_gru:
                    if not concat_depth:
                        fusion_feat = self.gru(
                            local_latent,
                            global_latent,
                            input_weights_emb,
                            hidden_weights_emb
                        ).squeeze(2)

                        if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                            clip_fusion_feat_3d = self.semantic_gru(
                                local_clip_latent_3d,
                                global_clip_latent_3d,
                                input_weights_emb,
                                hidden_weights_emb
                            ).squeeze(2)
                        else:
                            clip_fusion_feat_3d = None
                    else:
                        fusion_feat = self.gru(
                            torch.cat([local_latent, local_depth.view(1,-1,1,1)], dim=-1),
                            torch.cat([global_latent, global_depth.view(1,-1,1,1)], dim=-1),
                            input_weights_emb,
                            hidden_weights_emb
                        ).squeeze(2)

                        if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                            clip_fusion_feat_3d = self.semantic_gru(
                                torch.cat([local_clip_latent_3d, local_depth.view(1,-1,1,1)], dim=-1),
                                torch.cat([global_clip_latent_3d, global_depth.view(1,-1,1,1)], dim=-1),
                                input_weights_emb,
                                hidden_weights_emb
                            ).squeeze(2)
                        else:
                            clip_fusion_feat_3d = None
                else:
                    fusion_feat = (local_latent * weights_1[...,:1] + global_latent * weights_0[...,:1]) / (weights_0[...,:1] + weights_1[...,:1])
                    fusion_feat = fusion_feat.squeeze(2)

                    if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                        clip_fusion_feat_3d = (local_clip_latent_3d * weights_1[...,:1] + global_clip_latent_3d * weights_0[...,:1]) \
                                            / (weights_0[...,:1] + weights_1[...,:1])
                        clip_fusion_feat_3d = clip_fusion_feat_3d.squeeze(2)
                    else:
                        clip_fusion_feat_3d = None

                update_coords = (
                    global_coords[:, mask] * weights_0[...,1] + coords[:, i, pixel_indices] \
                    [:,fusion_indices_,0,0] * weights_1[...,1]) / (weights_0[...,1] + weights_1[...,1]
                )
                update_extrinsics = (
                    globals.extrinsics[:, globals.valid[0]][:, mask]*weights_0[...,:1] + extrinsics[:, i, None] \
                    * weights_1[...,:1]) / (weights_0[...,:1]+weights_1[...,:1]
                )
                update_depths = (
                    globals.depths[:, globals.valid[0]][:, mask] *weights_0[...,0,0] + \
                    depths[None, i, pixel_indices][:,fusion_indices_] \
                    * weights_1[...,0,0]) / (weights_0[...,0,0] + weights_1[...,0,0]
                )

                update_gaussians = rearrange(
                    self.to_gaussians(fusion_feat),
                    "... (srf c) -> ... srf c",
                    srf=self.cfg.num_surfaces,
                )
                                
                update_gaussians = self.gaussian_adapter.forward(
                    rearrange(update_extrinsics, "b r i j -> b () r () () i j"),
                    repeat(intrinsics[:,0], "b i j -> b () N () () i j", N=update_gaussians.shape[1]),
                    None,
                    rearrange(update_depths, "b r -> b () r () ()"),
                    nn.Sigmoid()(rearrange(update_gaussians[..., :1], "b r srf c -> b () r srf c")),
                    rearrange(update_gaussians[..., 2:], "b r srf c -> b () r srf () c"),
                    (h, w),
                    load_depth=self.cfg.load_depth,
                    fusion=False,
                    coords=rearrange(update_coords, "b r c -> b () r () () c"),
                )

            new_extrinsics = extrinsics[:,i,None].repeat(1,(concat_mask).sum(),1,1)
            new_gaussians_feat = gaussians[:,i][:,concat_mask]
            if self.cfg.openvocab_domain in ["ensemble", "3d"]:
                new_clip_gaussians_feat_3d = clip_gaussians_3d_unet[:,concat_mask]
            else:
                new_clip_gaussians_feat_3d = None
            new_densities = densities[:,i][:,concat_mask]
            new_weight_emb = weight_emb[:,i][:,concat_mask]
            new_coords = coords[:,i][:,concat_mask,0,0]
            new_depths = depths[None,i][:,concat_mask]
            new_gaussians = rearrange(
                self.to_gaussians(gaussians[:,i][:,concat_mask]),
                "... (srf c) -> ... srf c",
                srf=self.cfg.num_surfaces,
            )
            
            if self.cfg.openvocab_domain in ["ensemble", "2d"]:
                new_codebook_caches = codebook_caches[:,i][:,concat_mask]
                new_codebook_weights = torch.ones_like(new_codebook_caches).to(self.device)
                invalid = new_codebook_caches == -1
                new_codebook_weights[:, invalid[0]] = 0.
            else:
                new_codebook_caches = None
                new_codebook_weights = None
                
            new_gaussians = self.gaussian_adapter.forward(
                rearrange(new_extrinsics, "b r i j -> b () r () () i j"),
                repeat(intrinsics[:,0], "b i j -> b () N () () i j", N=new_gaussians.shape[1]),
                None,
                rearrange(new_depths, "b r -> b () r () ()"),
                nn.Sigmoid()(rearrange(new_gaussians[..., :1], "b r srf c -> b () r srf c")),
                rearrange(new_gaussians[..., 2:], "b r srf c -> b () r srf () c"),
                (h, w),
                load_depth=self.cfg.load_depth,
                fusion=False,
                coords=rearrange(new_coords, "b r c -> b () r () () c"),
            )
        
            if mask.sum() > 0 and fusion:
                globals.append(
                    means=rearrange(
                        update_gaussians.means,
                        "b v r srf spp xyz -> b (v r srf spp) xyz",
                    ),
                    covariances=rearrange(
                        update_gaussians.covariances,
                        "b v r srf spp i j -> b (v r srf spp) i j",
                    ),
                    harmonics=rearrange(
                        update_gaussians.harmonics,
                        "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                    ),
                    opacities=rearrange(
                        update_gaussians.opacities,
                        "b v r srf spp -> b (v r srf spp)",
                    ),
                    features=fusion_feat,
                    clip_features_3d=clip_fusion_feat_3d,
                    densities=globals.densities[:, globals.valid[0]][:, mask] + densities[:, i, pixel_indices][:,fusion_indices_],
                    weights=globals.weights[:, globals.valid[0]][:, mask] + weight_emb[:, i, pixel_indices][:,fusion_indices_],
                    coords=update_coords,
                    extrinsics=update_extrinsics,
                    depths=update_depths,
                    codebooks_idx=local_codebook_caches,
                    codebooks_weights=local_codebook_weights,
                    mask=mask,
                    scales=globals.scales[:, globals.valid[0]][:, mask] if export_ply else None,
                    rotations=globals.rotations[:, globals.valid[0]][:, mask] if export_ply else None,
                    fuse=True
                )

            globals.append(
                means=rearrange(
                    new_gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                covariances=rearrange(
                    new_gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                harmonics=rearrange(
                    new_gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                opacities=rearrange(
                    new_gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
                features=new_gaussians_feat,
                clip_features_3d=new_clip_gaussians_feat_3d,
                densities=new_densities,
                weights=new_weight_emb,
                coords=new_coords,
                extrinsics=new_extrinsics,
                depths=new_depths,
                codebooks_idx=new_codebook_caches,
                codebooks_weights=new_codebook_weights,
                mask=None,
                scales=rearrange(
                    new_gaussians.scales,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ) if export_ply else None,
                rotations=rearrange(
                    new_gaussians.rotations,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ) if export_ply else None,
            )

            if output_new_idx: 
                new_mask = torch.ones(concat_mask.sum()).bool().to(mask.device)
                new_mask = torch.cat([mask, new_mask], dim=0)

        if output_new_idx:
            return globals, new_mask    
        
        return globals

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
    
def rotmat_to_quat(R: torch.Tensor) -> torch.Tensor:
    det = torch.det(R)
    R = torch.where(det[..., None, None] < 0, torch.cat([R[..., :2, :], -R[..., 2:3, :]], dim=-2), R)

    t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
    qw = torch.empty_like(t)
    qx = torch.empty_like(t)
    qy = torch.empty_like(t)
    qz = torch.empty_like(t)

    mask = t > 0
    r = torch.sqrt(torch.clamp(t[mask] + 1.0, min=1e-12))
    qw[mask] = 0.5 * r
    r = 0.5 / r
    qx[mask] = (R[mask, 2, 1] - R[mask, 1, 2]) * r
    qy[mask] = (R[mask, 0, 2] - R[mask, 2, 0]) * r
    qz[mask] = (R[mask, 1, 0] - R[mask, 0, 1]) * r

    mask0 = ~mask & (R[..., 0, 0] >= R[..., 1, 1]) & (R[..., 0, 0] >= R[..., 2, 2])
    r = torch.sqrt(torch.clamp(1.0 + R[mask0, 0, 0] - R[mask0, 1, 1] - R[mask0, 2, 2], min=1e-12))
    qx[mask0] = 0.5 * r
    r2 = 0.5 / r
    qy[mask0] = (R[mask0, 0, 1] + R[mask0, 1, 0]) * r2
    qz[mask0] = (R[mask0, 0, 2] + R[mask0, 2, 0]) * r2
    qw[mask0] = (R[mask0, 2, 1] - R[mask0, 1, 2]) * r2

    mask1 = ~mask & ~mask0 & (R[..., 1, 1] >= R[..., 2, 2])
    r = torch.sqrt(torch.clamp(1.0 - R[mask1, 0, 0] + R[mask1, 1, 1] - R[mask1, 2, 2], min=1e-12))
    qy[mask1] = 0.5 * r
    r2 = 0.5 / r
    qx[mask1] = (R[mask1, 0, 1] + R[mask1, 1, 0]) * r2
    qz[mask1] = (R[mask1, 1, 2] + R[mask1, 2, 1]) * r2
    qw[mask1] = (R[mask1, 0, 2] - R[mask1, 2, 0]) * r2

    mask2 = ~mask & ~mask0 & ~mask1
    r = torch.sqrt(torch.clamp(1.0 - R[mask2, 0, 0] - R[mask2, 1, 1] + R[mask2, 2, 2], min=1e-12))
    qz[mask2] = 0.5 * r
    r2 = 0.5 / r
    qx[mask2] = (R[mask2, 0, 2] + R[mask2, 2, 0]) * r2
    qy[mask2] = (R[mask2, 1, 2] + R[mask2, 2, 1]) * r2
    qw[mask2] = (R[mask2, 1, 0] - R[mask2, 0, 1]) * r2

    q = torch.stack([qw, qx, qy, qz], dim=-1)
    q = q / torch.clamp(q.norm(dim=-1, keepdim=True), min=1e-12)

    sign = torch.where(q[..., 0:1] < 0, -1.0, 1.0)
    q = q * sign

    return q