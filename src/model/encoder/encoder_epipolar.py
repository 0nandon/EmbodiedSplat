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

from .encoder_config import EncoderEpipolarCfg

from einops import *
import matplotlib.cm as cm
from PIL import Image, ImageFont, ImageDraw

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
    def __init__(self, initial_capacity=1000, growth_factor=2.0, feat_dim=64, device='cuda', testing=True,#means, covariances, harmonics, opacities, features, coords, densities, extrinsics, depths
                 export_ply=False):
        self.capacity = initial_capacity if testing else 0
        self.growth_factor = growth_factor
        self.device = device
        self.testing = testing
        self.size = 0
        # print('Initalizing with shape:', self.capacity)

        # Initial allocations with guessed sizes
        self.means = torch.zeros((1, self.capacity, 3), device=device)  # Adjust the dimensionality as needed
        self.covariances = torch.zeros((1, self.capacity, 3, 3), device=device)  # Example shape
        self.harmonics = torch.zeros((1, self.capacity, 3, 9), device=device)  # Adjust based on actual shape
        self.opacities = torch.zeros((1, self.capacity), device=device)
        self.features = torch.zeros((1, self.capacity, feat_dim), device=device)  # Adjust based on actual shape
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

    def append(self, means, covariances, harmonics, opacities, features, coords, densities, weights, extrinsics, depths,
               mask=None, scales=None, rotations=None):

        if self.testing:
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
                self.weights[0, invalid_indices[:num_new]] = weights
                self.extrinsics[0, invalid_indices[:num_new]] = extrinsics
                self.depths[0, invalid_indices[:num_new]] = depths

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
            if features is not None:   
                self.features = torch.cat([self.features[:, remain_mask], features], dim=1)
                self.weights = torch.cat([self.weights[:, remain_mask], weights], dim=1)
                self.extrinsics = torch.cat([self.extrinsics[:, remain_mask], extrinsics], dim=1)
                self.depths = torch.cat([self.depths[:, remain_mask], depths], dim=1)
            # print('self.valid.shape:', self.valid.shape)

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
        
        if self.export_ply:
            self.scales = self._resize_tensor(self.scales, new_capacity, device=self.device)
            self.rotations = self._resize_tensor(self.rotations, new_capacity, device=self.device)
        
        self.capacity = new_capacity
        # print('Expanding with shape:', self.capacity)

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
    
class EncoderEpipolar(Encoder[EncoderEpipolarCfg]):
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
            
            self.high_resolution_skip = nn.ModuleList(
                                            [nn.Sequential(
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
                                            )]
                                        )

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

                self.cost_volume = AVGFeatureVolumeManager(matching_height=self.cfg.image_H//4, 
                                                        matching_width=self.cfg.image_W//4,
                                                        num_depth_bins=self.cfg.num_depth_candidates,
                                                        matching_dim_size=self.cfg.matchnet_dim if (not self.cfg.wo_matchnet) else 48,
                                                        num_source_views=self.cfg.num_views-1,
                                                        log_plane=self.cfg.log_cv,)
                
                if not self.cfg.wo_msd:
                    self.cv_encoder = CVEncoder(num_ch_cv=self.cfg.num_depth_candidates,
                                                num_ch_enc=self.backbone.num_ch_enc[1:],
                                                num_ch_outs=[64, 128, 256, 384])
                    dec_num_input_ch = (self.backbone.num_ch_enc[:1] 
                                                    + self.cv_encoder.num_ch_enc)
                else:
                    dec_num_input_ch = self.backbone.num_ch_enc[:1] + \
                                        [self.backbone.num_ch_enc[1]+self.cfg.num_depth_candidates] +\
                                        self.backbone.num_ch_enc[2:]
            else:
                dec_num_input_ch = (self.backbone.num_ch_enc)

            self.depth_decoder = DepthDecoderPP(dec_num_input_ch, 
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
                                                low_res=self.cfg.low_res,)
            self.max_depth = 2 + 2 * (not self.cfg.wo_msd)
            self.tensor_formatter = TensorFormatter()

            if self.cfg.fusion:
                self.weight_embedding = nn.Sequential(nn.Linear(2, 12), 
                                            activation_func,
                                            nn.Linear(12, 12),)
                self.gru = GRU2D_naive_Wweights(concat_depth=self.cfg.concat_depth)

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
        
        # Encode the context images.
        if self.cfg.backbone.name == 'dino':
            context = contexts[0]
            
            our_gaussians = []
            if is_testing:
                all_depths = []

            # print(f'src_indices.shape: {src_indices.shape}')
            bs = self.cfg.max_batch

            total_length = min(np.ceil(n_views/bs).astype(int), self.cfg.max_batch_length)
            # for batch in range(np.ceil(n_views/bs).astype(int)):
            for batch in range(total_length):
                # print(f'processing {batch}/{total_length}')
                if is_testing and total_length > 1:
                    if batch == total_length - 1:
                        rest_length = n_views - batch*bs
                        for j in range(bs-rest_length):
                            context['image'] = torch.cat([context['image'], context['image'][:, -1:]], dim=1)
                            context['intrinsics'] = torch.cat([context['intrinsics'], context['intrinsics'][:, -1:]], dim=1)
                            context['extrinsics'] = torch.cat([context['extrinsics'], context['extrinsics'][:, -1:]], dim=1)
                            context['near'] = torch.cat([context['near'], context['near'][:, -1:]], dim=1)
                            context['far'] = torch.cat([context['far'], context['far'][:, -1:]], dim=1)
                    else:
                        rest_length = bs
                
                cur_context = {'image': context['image'][:, batch*bs:(batch+1)*bs],
                                'intrinsics': context['intrinsics'][:, batch*bs:(batch+1)*bs],
                                'extrinsics': context['extrinsics'][:, batch*bs:(batch+1)*bs],
                                'near': context['near'][:, batch*bs:(batch+1)*bs],
                                'far': context['far'][:, batch*bs:(batch+1)*bs],
                                'depth': context['depth'][:, batch*bs:(batch+1)*bs] if self.cfg.load_depth else None}
                features = self.backbone(cur_context)
                features = rearrange(features, "b v c h w -> b v h w c")
                features = self.backbone_projection(features)
                features = rearrange(features, "b v h w c -> b v c h w")


                # Run the epipolar transformer.
                if self.cfg.use_epipolar_transformer:
                    features, sampling = self.epipolar_transformer(
                        features,
                        cur_context["extrinsics"],
                        cur_context["intrinsics"],
                        cur_context["near"],
                        cur_context["far"],
                    )
                to_skip = cur_context['image']
                skip = rearrange(to_skip, "b v c h w -> (b v) c h w")
                skip = self.high_resolution_skip(skip)
                features = features + rearrange(skip, "(b v) c h w -> b v c h w", b=b)

                h1, w1 = features.shape[-2:]

                features = rearrange(features, "b v c h w -> b v (h w) c")
                if self.cfg.est_depth == 'est':
                    depths, densities = self.depth_predictor.forward(
                        features,
                        cur_context["near"],
                        cur_context["far"],
                        deterministic,
                        1 if (deterministic or n_views > 10) else self.cfg.gaussians_per_pixel,
                    )
                else:
                    depths = rearrange(cur_context['depth'], "b v c h w -> b v (h w) c 1")
                    densities = self.opacity_mlp(features).unsqueeze(-1)
                    self.cfg.gaussians_per_pixel = 1
                
                xy_ray, _ = sample_image_grid((h, w), device)
                xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
                gaussians = self.to_gaussians(features)
                gaussians = rearrange(
                    gaussians,
                    "... (srf c) -> ... srf c",
                    srf=self.cfg.num_surfaces,
                )
                offset_xy = gaussians[..., :2].sigmoid()

                if self.cfg.est_depth == 'est':
                    pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
                    xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
                else:
                    xy_ray = xy_ray + torch.zeros_like(offset_xy, device=offset_xy.device)
                

                gpp = self.cfg.gaussians_per_pixel if n_views <= 10 else 1
                gaussians = self.gaussian_adapter.forward(
                    rearrange(cur_context["extrinsics"], "b v i j -> b v () () () i j"),
                    rearrange(cur_context["intrinsics"], "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    self.map_pdf_to_opacity(densities, global_step) / gpp,
                    rearrange(gaussians[..., 2:], "b v r srf c -> b v r srf () c"),
                    (h, w),
                    load_depth=self.cfg.load_depth,
                    ours=False,
                )

                if is_testing and total_length > 1:
                    all_depths.append(depths[:, :rest_length])
                d_sh = (self.cfg.sh_degree + 1) ** 2
                if not isinstance(our_gaussians, list):
                    our_gaussians = Gaussians(
                        torch.cat([our_gaussians.means, rearrange(
                            gaussians.means,
                            "b v r srf spp xyz -> b (v r srf spp) xyz",
                        )], dim=1),
                        torch.cat([our_gaussians.covariances, rearrange(
                            gaussians.covariances,
                            "b v r srf spp i j -> b (v r srf spp) i j",
                        )], dim=1),
                        torch.cat([our_gaussians.harmonics, rearrange(
                            gaussians.harmonics[..., :d_sh],
                            "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                        )], dim=1),
                        torch.cat([our_gaussians.opacities, rearrange(
                            gaussians.opacities,
                            "b v r srf spp -> b (v r srf spp)",
                        )], dim=1),
                    )
                else:
                    our_gaussians = Gaussians(
                        rearrange(
                            gaussians.means,
                            "b v r srf spp xyz -> b (v r srf spp) xyz",
                        ),
                        rearrange(
                            gaussians.covariances,
                            "b v r srf spp i j -> b (v r srf spp) i j",
                        ),
                        rearrange(
                            gaussians.harmonics,
                            "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                        )[..., :d_sh],
                        rearrange(
                            gaussians.opacities,
                            "b v r srf spp -> b (v r srf spp)",
                        ),
                    )
                    
            our_gaussians = [our_gaussians]
            num_gaussians = our_gaussians[0].means.shape[1]

            results['depth_num0_s0_b1hw'] = rearrange(depths.mean(-1).mean(-1), "b v (h w) -> b v h w", h=384, w=512)
            results['depth_num0_s-1_b1hw'] = rearrange(depths.mean(-1).mean(-1), "b v (h w) -> (b v) () h w", h=384, w=512)
            depths_raw = rearrange(context[f'depth_s-1'], "b v c h w -> (b v) c h w")
            results[f'depth_num0_s0_raw_b1hw'] = depths_raw
            mask = (depths_raw > 1e-3) * (depths_raw < 10)
            results[f'depth_num0_s0_mask_b1hw'] = mask
        elif self.cfg.backbone.name == 'cost_volume':
            our_gaussians = []
            all_depths = []
            all_scales = []
            all_rotations = []

            # Encode the context images.
            context = contexts[0]
            if self.cfg.use_epipolar_trans:
                epipolar_kwargs = {
                    "epipolar_sampler": self.epipolar_sampler,
                    "depth_encoding": self.depth_encoding,
                    "extrinsics": context["extrinsics"],
                    "intrinsics": context["intrinsics"],
                    "near": context["near"],
                    "far": context["far"],
                }
            else:
                epipolar_kwargs = None
            
            # print(f'src_indices.shape: {src_indices.shape}')
            bs = self.cfg.max_batch

            total_length = min(np.ceil(n_views/bs).astype(int), self.cfg.max_batch_length)
            
            # for batch in range(np.ceil(n_views/bs).astype(int)):
            for batch in range(total_length):
                # print(f'processing {batch}/{total_length}')
                if is_testing and total_length > 1:
                    if batch == total_length - 1:
                        rest_length = n_views - batch*bs
                        for j in range(bs-rest_length):
                            context['image'] = torch.cat([context['image'], context['image'][:, -1:]], dim=1)
                            context['intrinsics'] = torch.cat([context['intrinsics'], context['intrinsics'][:, -1:]], dim=1)
                            context['extrinsics'] = torch.cat([context['extrinsics'], context['extrinsics'][:, -1:]], dim=1)
                            context['near'] = torch.cat([context['near'], context['near'][:, -1:]], dim=1)
                            context['far'] = torch.cat([context['far'], context['far'][:, -1:]], dim=1)
                    else:
                        rest_length = bs
                
                trans_features, cnn_features = self.backbone(
                    context["image"][:, batch*bs:(batch+1)*bs],
                    attn_splits=self.cfg.multiview_trans_attn_split,
                    return_cnn_features=True,
                    epipolar_kwargs=epipolar_kwargs,
                )

                # Sample depths from the resulting features.
                in_feats = trans_features
                extra_info = {}
                extra_info['images'] = rearrange(context["image"][:, batch*bs:(batch+1)*bs], "b v c h w -> (v b) c h w")
                gpp = self.cfg.gaussians_per_pixel
                depths, densities, raw_gaussians = self.depth_predictor(
                    in_feats,
                    context["intrinsics"][:, batch*bs:(batch+1)*bs],
                    context["extrinsics"][:, batch*bs:(batch+1)*bs],
                    context["near"][:, batch*bs:(batch+1)*bs],
                    context["far"][:, batch*bs:(batch+1)*bs],
                    gaussians_per_pixel=gpp,
                    deterministic=deterministic,
                    extra_info=extra_info,
                    cnn_features=cnn_features,
                )

                # Convert the features and depths into Gaussians.
                # if not export_ply:
                xy_ray, _ = sample_image_grid((h, w), device)
                xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
                gaussians = rearrange(
                    raw_gaussians,
                    "... (srf c) -> ... srf c",
                    srf=self.cfg.num_surfaces,
                )
                offset_xy = gaussians[..., :2].sigmoid()
                pixel_size = 1 / torch.tensor((w, h), dtype=torch.float32, device=device)
                xy_ray = xy_ray + (offset_xy - 0.5) * pixel_size
                gpp = self.cfg.gaussians_per_pixel
                gaussians = self.gaussian_adapter.forward(
                    rearrange(context["extrinsics"][:, batch*bs:(batch+1)*bs], "b v i j -> b v () () () i j"),
                    rearrange(context["intrinsics"][:, batch*bs:(batch+1)*bs], "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    self.map_pdf_to_opacity(densities, global_step) / gpp,
                    rearrange(
                        gaussians[..., 2:],
                        "b v r srf c -> b v r srf () c",
                    ),
                    (h, w),
                    ours=False,
                    ada=self.cfg.adaptive_gaussian,
                )

                if self.cfg.adaptive_gaussian:
                    # Cascade Gaussian Adapter
                    if self.cfg.use_feat:
                        score_maps, alphas = self.keypoint_scorer(trans_features, h, w)
                    else:
                        score_maps, alphas = self.keypoint_scorer(context["image"][:, batch*bs:(batch+1)*bs])
                    

                    gaussians = self.cascade_gaussian_adapter(
                        origin_gaussians=gaussians,
                        score_maps=score_maps,
                        alphas=alphas,
                        extrinsics=context['extrinsics'][:, batch*bs:(batch+1)*bs],
                        intrinsics=context['intrinsics'][:, batch*bs:(batch+1)*bs],
                        image_size=(h, w)
                    )

                    gaussians = gaussians[0]
                
                # our_gaussians.append(gaussians)
                if is_testing and total_length > 1:
                    all_depths.append(depths[:, :rest_length])
                d_sh = (self.cfg.sh_degree + 1) ** 2
                if not isinstance(our_gaussians, list):
                    our_gaussians = Gaussians(
                        torch.cat([our_gaussians.means, rearrange(
                            gaussians.means,
                            "b v r srf spp xyz -> b (v r srf spp) xyz",
                        ) if not self.cfg.adaptive_gaussian else gaussians.means.unsqueeze(0)], dim=1),
                        torch.cat([our_gaussians.covariances, rearrange(
                            gaussians.covariances,
                            "b v r srf spp i j -> b (v r srf spp) i j",
                        ) if not self.cfg.adaptive_gaussian else gaussians.covariances.unsqueeze(0)], dim=1),
                        torch.cat([our_gaussians.harmonics, rearrange(
                            gaussians.harmonics[..., :d_sh],
                            "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                        ) if not self.cfg.adaptive_gaussian else gaussians.harmonics.unsqueeze(0)[..., :d_sh]], dim=1),
                        torch.cat([our_gaussians.opacities, rearrange(
                            gaussians.opacities,
                            "b v r srf spp -> b (v r srf spp)",
                        ) if not self.cfg.adaptive_gaussian else gaussians.opacities.unsqueeze(0)], dim=1),
                    )
                else:
                    our_gaussians = Gaussians(
                        rearrange(
                            gaussians.means,
                            "b v r srf spp xyz -> b (v r srf spp) xyz",
                        ) if not self.cfg.adaptive_gaussian else gaussians.means.unsqueeze(0),
                        rearrange(
                            gaussians.covariances,
                            "b v r srf spp i j -> b (v r srf spp) i j",
                        ) if not self.cfg.adaptive_gaussian else gaussians.covariances.unsqueeze(0),
                        rearrange(
                            gaussians.harmonics,
                            "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                        )[..., :d_sh] if not self.cfg.adaptive_gaussian else gaussians.harmonics.unsqueeze(0)[..., :d_sh],
                        rearrange(
                            gaussians.opacities,
                            "b v r srf spp -> b (v r srf spp)",
                        ) if not self.cfg.adaptive_gaussian else gaussians.opacities.unsqueeze(0),
                    )
                if (export_ply or self.cfg.ft):
                    all_scales.append(rearrange(
                            gaussians.scales,
                            "b v r srf spp xyz -> b (v r srf spp) xyz",
                        ) if not self.cfg.adaptive_gaussian else gaussians.scales.unsqueeze(0))
                    all_rotations.append(rearrange(
                            gaussians.rotations,
                            "b v r srf spp xyz -> b (v r srf spp) xyz",
                        ) if not self.cfg.adaptive_gaussian else gaussians.rotations.unsqueeze(0))
                    
            our_gaussians = [our_gaussians]
            num_gaussians = our_gaussians[0].means.shape[1]

            if (export_ply or self.cfg.ft):
                all_scales = torch.cat(all_scales, dim=1)
                all_rotations = torch.cat(all_rotations, dim=1)

            if is_testing and total_length > 1:
                all_depths = torch.cat(all_depths, dim=1)
            else:
                all_depths = depths

            results['depth_num0_s0_b1hw'] = rearrange(all_depths, "b v (h w) x y -> b v h (w x y)", h=384, w=512)
            results['depth_num0_s-1_b1hw'] = rearrange(all_depths, "b v (h w) x y -> (b v) () h (w x y)", h=384, w=512)
            depths_raw = rearrange(context[f'depth_s-1'], "b v c h w -> (b v) c h w")
            results[f'depth_num0_s0_raw_b1hw'] = depths_raw
            mask = (depths_raw > 1e-3) * (depths_raw < 10)
            results[f'depth_num0_s0_mask_b1hw'] = mask
        elif self.cfg.est_depth == 'cost':
            start = time.time()
            length = len(contexts)
            our_gaussians = []
            gaussians = []
            coords = []
            results = {}
            
            if (dataset_name == 'scannet' or (not(is_testing)) or self.cfg.num_views > 2):
                self.backbone.train()
            else:
                print('freezing backbone...')
                # self.backbone.eval()

            # for num in range(length):
            num = 0
            context = contexts[num]
            context['image_shape'] = (h, w)
            self.cfg.gaussians_per_pixel = 1
            context_intrinsics = context['intrinsics'].clone()
            context_intrinsics[:,:,0] *= (w // 4)
            context_intrinsics[:,:,1] *= (h // 4)

            globals = EfficientGaussians(initial_capacity=n_views*10000, growth_factor=1.5, 
                                         device=context['image'].device, testing=is_testing,
                                         export_ply=export_ply or self.cfg.ft)
        
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

                    depth_outputs = self.depth_decoder([cur_feats[0][batch*bs:(batch+1)*bs]] + cost_volume_features,
                                                        imgs=cur_image[batch*bs:(batch+1)*bs],)
                else:
                    depth_outputs = self.depth_decoder([x[batch*bs:(batch+1)*bs] for x in cur_feats], 
                                                       imgs=cur_image[batch*bs:(batch+1)*bs],)

                to_skip = context['image'][:, batch*bs:(batch+1)*bs]
                to_skip = rearrange(to_skip, "b v c h w -> (b v) c h w")

                skip = self.high_resolution_skip[s+1](to_skip)

                if not export_ply:
                    margin = 0
                    xy_ray, _ = sample_image_grid((h//(1+self.cfg.low_res), w//(1+self.cfg.low_res)), device)
                else:
                    # margin = 8 if not self.cfg.low_res else 4
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
                        # intrinsics_multiplier = torch.tensor([[w*1.0 / (w - 2*margin), h*1.0 / (h - 2*margin), 1]], device=device)
                        context["intrinsics"] = context["intrinsics"] * torch.tensor([[w*1.0 / (w - 2*margin), h*1.0 / (h - 2*margin), 1]], device=device)

                    skip = skip[:, :, margin:-margin, margin:-margin]
                
                if self.cfg.depth_pad:
                    border = 8 if not self.cfg.low_res else 4
                    # print('padding depth...')

                    # for name in [f'output_pred_s{s}_b1hw', f'depth_pred_s{s}_b1hw', f'depth_weights']:
                    for name in [f'depth_pred_s{s}_b1hw', f'depth_weights']:
                        depth_outputs[name][:, :, :border, :] = depth_outputs[name][:, :, border, None]
                        depth_outputs[name][:, :, -border:, :] = depth_outputs[name][:, :, -border-1, None]
                        depth_outputs[name][:, :, :, :border] = depth_outputs[name][:, :, :, border, None]
                        depth_outputs[name][:, :, :, -border:] = depth_outputs[name][:, :, :, -border-1, None]
                    
                gaussians_feats = rearrange(depth_outputs[f'output_pred_s{s}_b1hw'][:,1:], '(b v) c h w -> b v h w c', b=b, v=n_views_now)#.view(b,n_views_now,h//(2**(s+1)), w//(2**(s+1)),64)
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
                    (h // (1+self.cfg.low_res) - margin*2, 
                        w // (1+self.cfg.low_res) - margin*2),
                    load_depth=self.cfg.load_depth,
                    fusion=True,
                )

                num_raw_gaussians = gaussians_feats.shape[2] * gaussians_feats.shape[1]
                B = gaussians_feats.shape[0]
                for bb in range(B):
                    cur_gs = gaussians_feats[bb:bb+1]
                    cur_coords = coords[bb:bb+1]
                    cur_densities = densities[bb:bb+1]
                    cur_weights = weights[bb:bb+1]
                    cur_depth = rearrange(depth_outputs[f'depth_pred_s{s}_b1hw'], "(b v) c h w -> b v c h w", b=B)[bb]

                    if self.cfg.fusion_pp:
                        globals = self.fuse_gaussians_pp(cur_gs, cur_coords, 
                                                        cur_densities, cur_weights, 
                                                        cur_depth, 
                                                        context["extrinsics"][bb:bb+1, batch*bs:(batch+1)*bs], \
                                                        context["intrinsics"][bb:bb+1, batch*bs:(batch+1)*bs], 
                                                        context['image_shape'],
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
                                                        )
                    else:
                        globals = self.fuse_gaussians(cur_gs, cur_coords, 
                                                        cur_densities, cur_weights, 
                                                        cur_depth, 
                                                        context["extrinsics"][bb:bb+1, batch*bs:(batch+1)*bs], \
                                                        context["intrinsics"][bb:bb+1, batch*bs:(batch+1)*bs], 
                                                        # context['image_shape'],
                                                        (h // (1+self.cfg.low_res) - margin*2, 
                                                            w // (1+self.cfg.low_res) - margin*2),
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
                                                        export_ply=(export_ply or self.cfg.ft),
                                                        )
                    # print('globals:', globals)
                    # print('TIME OF FUSION:', time.time()-start)
                    start = time.time()
                    # print(globals.scales)

                    if self.cfg.vis_gs:
                        gs_model = GaussianModel(sh_degree=self.cfg.sh_degree)
                        gs_model.load_gs(globals)
                        gs_model.save_ply(output_path / 'gaussians' / f'{scene}_{batch}.ply')
                    
                    if test_bev:
                        # os.makedirs(path / scene / f"bev", exist_ok=True)

                        edges_means = create_transformed_pyramid(context['extrinsics'][0,batch].cpu().numpy())
                        edges_means = torch.from_numpy(edges_means).unsqueeze(0).to(globals.means.device)
                        edges_covariances = 0.0001 * torch.eye(3, device=globals.means.device)[None, None].repeat(1, edges_means.shape[1], 1, 1)
                        edges_harmonics = torch.zeros([1, edges_means.shape[1], 3, globals.harmonics.shape[-1]], device=globals.means.device)
                        edges_opacities = torch.ones([1, edges_means.shape[1]], device=globals.means.device)
                        edges_conf = torch.ones([1, edges_means.shape[1]], device=globals.means.device)
                        # edges_harmonics[:, :, 0, 0] = 68. / 255 * 1.5
                        # edges_harmonics[:, :, 1, 0] = 114. / 255 * 1.5
                        # edges_harmonics[:, :, 2, 0] = 196. / 255 * 1.5
                        edges_harmonics[:, :, 0, 0] = 1.5

                        gaussians = Gaussians(means=torch.cat([globals.means[:, globals.valid[0]], edges_means], dim=1),
                                            covariances=torch.cat([globals.covariances[:, globals.valid[0]], edges_covariances], dim=1),
                                            harmonics=torch.cat([globals.harmonics[:, globals.valid[0]], edges_harmonics], dim=1), 
                                            opacities=torch.cat([globals.opacities[:, globals.valid[0]], edges_opacities], dim=1),
                                            conf=torch.cat([globals.densities[:, globals.valid[0],0,0], edges_conf], dim=1))
                        
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
                        # save_image(color, path / scene / f"bev/{batch:0>6}.png")
                        save_image(color, os.path.join("bev", f"{batch:0>6}.png"))
                        # exit(0)
                    

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
                        # print('cur_depth.shape:', cur_depth.shape)
                        globals = self.refine_gaussians(globals, cur_depth, cur_weights, context["extrinsics"][:,i:i+1], 
                                                        context["intrinsics"][:,i:i+1], 
                                                        # context['image_shape'], 
                                                        (h // (1+self.cfg.low_res) - margin*2, 
                                                            w // (1+self.cfg.low_res) - margin*2),
                                                        ws=self.cfg.refine_ws, depth_thres=self.cfg.refine_thres,
                                                        soft_thres=self.cfg.refine_soft_thres, refine_pp=self.cfg.refine_pp,
                                                        export_ply=(export_ply or self.cfg.ft),
                                                        num=i,)
                        
                        if test_bev:
                            os.makedirs(path / scene / f"bev", exist_ok=True)

                            edges_means = create_transformed_pyramid(context['extrinsics'][0,i].cpu().numpy())
                            edges_means = torch.from_numpy(edges_means).unsqueeze(0).to(globals.means.device)
                            edges_covariances = 0.0001 * torch.eye(3, device=globals.means.device)[None, None].repeat(1, edges_means.shape[1], 1, 1)
                            edges_harmonics = torch.zeros([1, edges_means.shape[1], 3, globals.harmonics.shape[-1]], device=globals.means.device)
                            edges_opacities = torch.ones([1, edges_means.shape[1]], device=globals.means.device)
                            edges_conf = torch.ones([1, edges_means.shape[1]], device=globals.means.device)
                            # edges_harmonics[:, :, 0, 0] = 68. / 255 * 1.5
                            # edges_harmonics[:, :, 1, 0] = 114. / 255 * 1.5
                            # edges_harmonics[:, :, 2, 0] = 196. / 255 * 1.5
                            edges_harmonics[:, :, -1, 0] = 1.5

                            gaussians = Gaussians(means=torch.cat([globals.means[:, globals.valid[0]], edges_means], dim=1),
                                                covariances=torch.cat([globals.covariances[:, globals.valid[0]], edges_covariances], dim=1),
                                                harmonics=torch.cat([globals.harmonics[:, globals.valid[0]], edges_harmonics], dim=1), 
                                                opacities=torch.cat([globals.opacities[:, globals.valid[0]], edges_opacities], dim=1),
                                                conf=torch.cat([globals.densities[:, globals.valid[0],0,0], edges_conf], dim=1))
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
                            save_image(color, os.path.join("refine_bev", f"{i:0>6}.png"))
                            # save_image(color, path / scene / f"bev/refine_{i:0>6}.png")
                
                # if self.cfg.vis_refine:
                #     exit(0)

            our_gaussians = [Gaussians(means=globals.means[:, globals.valid[0]], 
                                            covariances=globals.covariances[:, globals.valid[0]], 
                                            harmonics=globals.harmonics[:, globals.valid[0]], 
                                            opacities=globals.opacities[:, globals.valid[0]],
                                            conf=globals.densities[:, globals.valid[0],0,0])]
                        
            num_gaussians = our_gaussians[0].means.shape[1]
            # print('predicted num_gaussians:', num_gaussians)
            results['gs_ratio'] = num_gaussians / num_raw_gaussians
            # gaussians = cur_gaussians
            
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
            # results[f'log_depth_num{num}_s{s}_b1hw'] = rearrange(torch.log(depth_outputs[f'depth_pred_s{s}_b1hw']), "(b v) c h w -> b v (c h w) () ()", b=b)
            # print(f'depth_pred_s{s}_b1hw.shape:', depth_outputs_all[f'depth_pred_s{s}_b1hw'].shape)
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
            # try:
            #     visualization_dump["scales"] = rearrange(
            #         globals.scales[:, globals.valid[0]], "b v r srf spp xyz -> b (v r srf spp) xyz"
            #     )
            #     visualization_dump["rotations"] = rearrange(
            #         globals.rotations[:, globals.valid[0]], "b v r srf spp xyzw -> b (v r srf spp) xyzw"
            #     )
            # except:
            # print('globals.valid:', globals.valid)
            # print('globals.scales:', globals.scales)
            visualization_dump["scales"] = globals.scales[:, globals.valid[0]]
            visualization_dump["rotations"] = globals.rotations[:, globals.valid[0]]

        # if not self.cfg.fusion:

        if self.cfg.ft or export_ply:
        # if self.cfg.ft:
            with torch.inference_mode(not self.cfg.ft):
                # print('!!!!!!!!!!!!!!!!!!sh_degree:', self.cfg.sh_degree)
                gs_model = GaussianModel(sh_degree=self.cfg.sh_degree)
                if self.cfg.est_depth == 'cost':
                    gs_model.load_gs(globals)
                else:
                    gs_model.load_gs_(our_gaussians[0], all_scales, all_rotations)

        if self.cfg.ft:
            all_rendered_conf = None
            if self.cfg.ft_depth_sup:
                # all_rendered_depths, all_rendered_conf = self.render_depths(our_gaussians[0], decoder, scene, context, 
                #                                          ft_dense=self.cfg.ft_dense)
                all_rendered_depths = self.render_depths(our_gaussians[0], decoder, scene, context, 
                                                         ft_dense=self.cfg.ft_dense)
            else:
                all_rendered_depths = None
            with torch.inference_mode(False):
                if self.cfg.est_depth == 'cost':
                    del gaussians, globals
                else:
                    del gaussians, our_gaussians, all_scales, all_rotations

                torch.cuda.empty_cache()
                gs_model = self.finetune_gs(gs_model, context, decoder, self.cfg.ft_iter, scene=scene,
                                            target_indices=target_indices, opr=self.cfg.opr, 
                                            rendered_depths=all_rendered_depths, conf=all_rendered_conf,
                                            test_bev=test_bev, target=target, bev_path=path)
            our_gaussians = [Gaussians(means=gs_model.get_xyz.unsqueeze(0), 
                                        covariances=gs_model.get_covariance().unsqueeze(0), 
                                        harmonics=gs_model.get_features.unsqueeze(0).transpose(-1, -2), 
                                        opacities=gs_model.get_opacity.unsqueeze(0).squeeze(-1))]
        
            visualization_dump["scales"] = gs_model.get_scaling.unsqueeze(0)
            visualization_dump["rotations"] = gs_model.get_rotation.unsqueeze(0)
        if export_ply or self.cfg.ft:
            gs_model.save_ply(output_path / 'gaussians' / f'{scene}.ply')
        
        
        # if self.cfg.use_epipolar_transformer:
        #     visualization_dump["sampling"] = sampling
        
        results['visualizations'] = visualization_dump

        results['gaussians'] = our_gaussians
        
        final_num_gaussians = our_gaussians[0].means.shape[1]
        results['num_gaussians'] = final_num_gaussians
        # print('final_num_gaussians:', final_num_gaussians)
        # print('TIME OF FINAL:', time.time()-start) 
        return results

    def render_depths(self, gaussians, decoder, scene, context, h=384, w=512, ft_dense=True):
        depths = []
        confs = []
        if ft_dense:
            try:
                path = f'dataset/scannet/test/{scene[:-2]}/'
                extrinsics = np.load(os.path.join(path, 'extrinsics.npy')).astype(np.float32)
            except:
                path = f'dataset/scannetpp/data/{scene[:-2]}/'
                extrinsics = np.load(os.path.join(path, 'extrinsics.npy'))
            
            extrinsics = torch.from_numpy(extrinsics).to(context['image'].device)[None]
        else:
            extrinsics = context['extrinsics']
        
        length = 1

        intrinsic = context['intrinsics']
        n_targets = extrinsics.shape[1]


        

        for j in range(np.ceil(n_targets/length).astype(int)):
            # print('gaussians.scales.shape:', gaussians.scales.shape)
            # print('extrinsics.shape:', extrinsics[:, j*length:(j+1)*length].shape)
            output = decoder.forward(
                gaussians,
                extrinsics[:, j*length:(j+1)*length],
                intrinsic[:, :1],
                context["near"][:, :1],
                context["far"][:, :1],
                (h, w),
                depth_mode='depth',
                scale_invariant=True,
            )
            depths.append(output.depth)



            # conf = output.conf
            # conf = torch.log(conf + 1e-3)
            # conf = (conf - conf.min()) / (conf.max() - conf.min())

            # confs.append(conf)
        
        # return depths, confs
        return depths
        

    def finetune_gs(self, gaussians, context, decoder, ft_iter, scene, target_indices, dssim=0.2, opr=200,
                    rendered_depths=None, conf=None, test_bev=False, target=None, bev_path=None):
        if not self.cfg.ft_dense:
            all_length = len(context["cams"])
            length = all_length
            indices = torch.randperm(all_length)
            intrinsic = context['intrinsics'][0, 0]
            fovx = focal2fov(intrinsic[0, 0], intrinsic[0, 2] * 2)
            fovy = focal2fov(intrinsic[1, 1], intrinsic[1, 2] * 2)
        else:
            try:
                path = f'dataset/scannet/test/{scene[:-2]}/'
                extrinsics = np.load(os.path.join(path, 'extrinsics.npy'))
            except:
                path = f'dataset/scannetpp/data/{scene[:-2]}/'
                extrinsics = np.load(os.path.join(path, 'extrinsics.npy'))

            all_length = context['index'][0, -1]
            device = context['image'].device
            intrinsic = context['intrinsics'][0, 0]

            fovx = focal2fov(intrinsic[0, 0], intrinsic[0, 2] * 2)
            fovy = focal2fov(intrinsic[1, 1], intrinsic[1, 2] * 2)

            intrinsic = intrinsic.cpu().numpy()

            indices = torch.randperm(all_length)
            mask = ~torch.isin(indices, target_indices.to(indices.device))
            indices = indices[mask]
            length = len(indices)

        
        index = 0
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        nerf_normalization = getNerfppNorm(context['cams'])
        # gaussians.save_ply('./before.ply')
        if not self.cfg.no_spatial_lr:
            gaussians.spatial_lr_scale = nerf_normalization['radius']
        else:
            gaussians.spatial_lr_scale = 0
        gaussians.training_setup(position_lr_max_steps=ft_iter//3, multiplier=self.cfg.ft_multiplier,)

        ema_loss_for_log = 0.0

        
        opacity_reset_interval = opr

        try:
            with open(os.path.join(path, 'valids.txt'), 'r') as f:
                valids = f.readlines()
                valids = [x.strip() for x in valids]
        except:
            pass

        progress_bar = tqdm(range(ft_iter), desc="Finetuning progress")
        for iteration in range(1, ft_iter+1):
            gaussians.update_learning_rate(iteration)


            if not self.cfg.ft_dense:
                viewpoint_cam = context['cams'][indices[index]]
                viewpoint_cam = loadCam(indices[index], viewpoint_cam, 1, image=context['image'][0, indices[index]],
                                        fovx=fovx, fovy=fovy)
            
            elif indices[index] in context['index'][0]:
                pos = (context['index'][0]==indices[index]).nonzero().item()
                viewpoint_cam = context['cams'][pos]
                viewpoint_cam = loadCam(indices[index], viewpoint_cam, 1, image=context['image'][0, pos],
                                        fovx=fovx, fovy=fovy)



            else:
                # while indices[index] in target_indices:
                #     index += 1

                # print('indices[index]:', indices[index])

                w2c = np.linalg.inv(extrinsics[indices[index]])
                R = np.transpose(w2c[:3, :3])
                T = w2c[:3, 3]
                # print('R.shape:', R.shape)
                # print('T.shape:', T.shape)
                try:
                    image = Image.open(os.path.join(path, 'color', str(indices[index].numpy())+'.jpg'))
                    viewpoint_cam = CameraInfo(
                            uid=indices[index],
                            R=R,
                            T=T,
                            FovY=fovy,
                            FovX=fovx,
                            image_path=os.path.join(path, 'color', str(indices[index].numpy())+'.jpg'),
                            image_name=str(indices[index].numpy())+'.jpg',
                            width=512,
                            height=384,
                            intrinsics=intrinsic,
                        )
                except:
                    image = Image.open(os.path.join(path, 'dslr', 'undistorted_images', str(valids[indices[index].numpy()])))
                    viewpoint_cam = CameraInfo(
                            uid=indices[index],
                            R=R,
                            T=T,
                            FovY=fovy,
                            FovX=fovx,
                            image_path=os.path.join(path, 'dslr', 'undistorted_images', str(indices[index].numpy())),
                            image_name=str(indices[index].numpy()),
                            width=512,
                            height=384,
                            intrinsics=intrinsic,
                        )
                # print('new FOV:', viewpoint_cam.FovX, viewpoint_cam.FovY)
                # print('new R T:', viewpoint_cam.R, viewpoint_cam.T)
                image = self.to_tensor(image.resize((640, 480)))
                image = rescale_and_crop(image, 0, (384, 512), use_depth=True)

                viewpoint_cam = loadCam(indices[index], viewpoint_cam, 1, image=image, torch_input=False)
                
            
            viewpoint_cam.cuda()

            

            # Render
            bg = background
            render_pkg = render(viewpoint_cam, gaussians, background)
            image, viewspace_point_tensor, visibility_filter, radii = (
                        render_pkg["render"],
                        render_pkg["viewspace_points"],
                        render_pkg["visibility_filter"],
                        render_pkg["radii"],
                    )
            

            if not image.requires_grad:
                raise ValueError("Rendered image does not require gradients. Check the rendering process.")

            gt_image = viewpoint_cam.original_image.cuda()

            # Check if gt_image inadvertently requires gradients
            if gt_image.requires_grad:
                print("Warning: gt_image requires gradients, which is unusual.")

            Ll1 = l1_loss(image, gt_image)
            ssim_loss = ssim(image, gt_image)

            # save_image(image, f'debug/{indices[index-1]}.png')
            # save_image(gt_image, f'debug/{indices[index-1]}_gt.png')
            # exit(0)
            
            if rendered_depths is not None:
                # print('rendered_depths[0].shape:', rendered_depths[0].shape)
                # print('len(rendered_depths):', len(rendered_depths))
                # print('depth.shape:', render_pkg['depth'].shape)
                # print('indices[index]:', indices[index])
                # for i in range(len(rendered_depths)):
                #     save_image(torch.from_numpy(convert_array_to_pil(rendered_depths[i][0].cpu().numpy().reshape(384,512), no_text=True).transpose(2,0,1)\
                #                                     .astype(np.float32)/255).to(context["image"].device),
                #                                     f'debug/depth_sup_{i}.png')
                # save_image(torch.from_numpy(convert_array_to_pil(rendered_depths[indices[index]][0].cpu().numpy().reshape(384,512), no_text=True).transpose(2,0,1)\
                #                                 .astype(np.float32)/255).to(context["image"].device),
                #                                 f'debug/depth_sup.png')
                
                # save_image(torch.from_numpy(convert_array_to_pil(render_pkg['depth'].detach().cpu().numpy().reshape(384,512), no_text=True).transpose(2,0,1)\
                #                                 .astype(np.float32)/255).to(context["image"].device),
                #                                 'debug/depth.png')
                if not self.cfg.conf_depth_sup:
                    depth_loss = l1_loss(rendered_depths[indices[index]][0], render_pkg['depth']) * 0.1
                else:
                    # depth_loss = l1_loss(rendered_depths[indices[index]][0]*conf[indices[index]][0], render_pkg['depth']*conf[indices[index]][0]) * 0.1 / conf[indices[index]][0].mean()
                    depth_loss = (torch.abs((rendered_depths[indices[index]][0] - render_pkg['depth']))*conf[indices[index]][0].clone()).mean() * 0.1 / conf[indices[index]][0].clone().mean()
                # exit(0)
            else:
                depth_loss = 0

            loss = (1.0 - dssim) * Ll1 + dssim * (1.0 - ssim_loss) + depth_loss

            # print(f'loss: {loss.item()}, depth_loss: {depth_loss}')

            if not loss.requires_grad:
                raise ValueError("Loss computation does not link to gradients. Check loss components.")

            loss.backward()

            with torch.no_grad():
                if iteration < ft_iter // 2:
                    # Keep track of max radii in image-space for pruning
                    gaussians.max_radii2D[visibility_filter] = torch.max(
                        gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                    )
                    gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                    if iteration % 100 == 0:
                        # print('Densify and Prune...')
                        # size_threshold = 20 if ((iteration > opacity_reset_interval) and (self.cfg.ft_prune)) else None
                        size_threshold = 20 if self.cfg.ft_prune else None
                        gaussians.densify_and_prune(
                            0.0002,
                            0.005,
                            nerf_normalization['radius'],
                            size_threshold,
                            densify=self.cfg.ft_densify,
                        )

                    if self.cfg.ft_reset and iteration % opacity_reset_interval == 0:
                        # print('Reset Opacity...')
                        gaussians.reset_opacity()

                # Optimizer step
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad()
                
                # Progress bar
                ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
                if iteration % 20 == 0:
                    progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                    progress_bar.update(20)
                if iteration == ft_iter:
                    progress_bar.close()


            index += 1
            # print(f'index: {index}/{length}, all_length: {all_length}')
            if index == length:
                # exit(0)
                index = 0
                indices = torch.randperm(all_length)

                if self.cfg.ft_dense:
                    mask = ~torch.isin(indices, target_indices.to(indices.device))
                    indices = indices[mask]
            

            if test_bev and ((iteration-1) % 100 == 0 or iteration == ft_iter):
                os.makedirs(bev_path / scene / f"bev", exist_ok=True)
                
                bev_gaussians = Gaussians(means=gaussians.get_xyz.unsqueeze(0),
                                    covariances=gaussians.get_covariance().unsqueeze(0),
                                    harmonics=gaussians.get_features.unsqueeze(0).transpose(-1, -2), 
                                    opacities=gaussians.get_opacity.unsqueeze(0).squeeze(-1),
                                    conf=None)
                output_bev = decoder.forward(
                    bev_gaussians,
                    target["bev_extrinsics"],
                    target["intrinsics"][:, :1],
                    target["near"][:, :1],
                    target["far"][:, :1],
                    (1440, 1920),
                    depth_mode='depth',
                    scale_invariant=True,
                    background_color=torch.tensor([1, 1, 1]).float().to(bev_gaussians.means.device),
                )
                color = output_bev.color[0][0]
                save_image(color, bev_path / scene / f"bev/ft_{iteration:0>6}.png")


        return gaussians
    
    def refine_gaussians(self, globals, depths, weights, extrinsics, intrinsics, image_shape, 
                         depth_thres=0.3, ws=True, soft_thres=False, refine_pp=False, export_ply=False, num=0):

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

        # remove_mask = depths[i] - depth_map > torch.clamp_min(depths[i] * 0.05, depth_thres)
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

            # print('filtered_pixel_indices.shape:', filtered_pixel_indices.shape)
            # print('filtered_gaussian_indices.shape:', filtered_gaussian_indices.shape)
            # print('sparse_indices.shape:', sparse_indices.shape)
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

            # print('filtered_pixel_indices.shape:', filtered_pixel_indices.shape)
            # print('filtered_gaussian_indices.shape:', filtered_gaussian_indices.shape)
            # print('sparse_indices.shape:', sparse_indices.shape)
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




        if mask.sum() > 0 and (not self.cfg.vis_refine):
            if ws:

                globals.append(means=globals.means[:, globals.valid[0]][:, mask],
                                covariances=globals.covariances[:, globals.valid[0]][:, mask],
                                harmonics=globals.harmonics[:, globals.valid[0]][:, mask],
                                opacities=globals.opacities[:, globals.valid[0]][:, mask] * multiplier,
                                features=None,
                                densities=globals.densities[:, globals.valid[0]][:, mask],
                                weights=None,
                                coords=globals.coords[:, globals.valid[0]][:, mask],
                                extrinsics=None,
                                depths=None,
                                mask=mask,
                                scales=globals.scales[:, globals.valid[0]][:, mask] if export_ply else None,
                                rotations=globals.rotations[:, globals.valid[0]][:, mask] if export_ply else None,
                                )
            else:
                globals.append(means=globals.means[:, globals.valid[0]][:, mask],
                                covariances=globals.covariances[:, globals.valid[0]][:, mask],
                                harmonics=globals.harmonics[:, globals.valid[0]][:, mask],
                                opacities=globals.opacities[:, globals.valid[0]][:, mask] * 0,
                                features=None,
                                densities=globals.densities[:, globals.valid[0]][:, mask],
                                weights=None,
                                coords=globals.coords[:, globals.valid[0]][:, mask],
                                extrinsics=None,
                                depths=None,
                                mask=mask,
                                scales=globals.scales[:, globals.valid[0]][:, mask] if export_ply else None,
                                rotations=globals.rotations[:, globals.valid[0]][:, mask] if export_ply else None,
                                )
            
            # globals['coords'] = global_coords
            # globals['gs'] = global_gaussians
            # globals['densities'] = global_densities
        
        return globals


    def fuse_gaussians(self, gaussians, coords, densities, weight_emb, depths, 
                       extrinsics, intrinsics, image_shape, depth_thres=0.1, limit=100,
                       vis=False, globals=None, remove=False, use_em=False, decoder=None,
                       near=0.5, far=5.0, img=None, fusion=True, fore_fusion=False, concat_depth=False,
                       use_gru=True, depth_fore=15.0, export_ply=False):
        length = min(gaussians.shape[1], limit)
        depths = rearrange(depths, "v c h w -> v (c h w)")
        # initial = len(globals['gs_feat']) == 0
        initial = globals.valid.sum() == 0
        h, w = image_shape
        if initial:
            global_gaussians_feat = gaussians[:,0]
            global_densities = densities[:, 0]
            global_weight_emb = weight_emb[:, 0]
            global_coords = coords[:,0,:,0,0]
            
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
            globals.append( means=rearrange(
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
                            densities=global_densities,
                            weights=global_weight_emb,
                            coords=global_coords,
                            extrinsics=global_extrinsics,
                            depths=global_depths,
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

                # print(f'render.shape: {render.shape}')
                num_imgs = len(os.listdir('./debug'))
                render = (render.squeeze().permute(1, 2, 0) * 255).byte()
                render = Image.fromarray(render.detach().cpu().numpy(), 'RGB')

                # Save the image
                render.save(f'debug/render_{num_imgs}.png')
                

            proj_map = torch.where(depth_map[pixel_indices] == curr_depths[valid])[0]
            fusion_indices = torch.where(fusion_mask[pixel_indices])[0]
            fusion_indices_ = fusion_indices[torch.isin(fusion_indices, proj_map)]
            corr_indices = proj_map[torch.isin(proj_map, fusion_indices)]
            valid_indices = torch.zeros(valid.sum(), device=valid.device, dtype=torch.bool)
            valid_indices.scatter_(0, corr_indices, True)
            mask = torch.zeros_like(valid, device=valid.device, dtype=torch.bool)
            mask[valid] = valid_indices



            if mask.sum() > 0 and fusion:
                input_weights_emb = positional_encoding(torch.cat([globals.densities[:, globals.valid[0]][:, mask], weight_emb[:, i, pixel_indices][:,fusion_indices_]], dim=-1), 6)
                hidden_weights_emb = positional_encoding(torch.cat([densities[:, i, pixel_indices][:,fusion_indices_], globals.weights[:, globals.valid[0]][:, mask]], dim=-1), 6)
                local_latent = gaussians[:, i, pixel_indices][:,fusion_indices_].unsqueeze(2)
                global_latent = globals.features[:, globals.valid[0]][:, mask].unsqueeze(2)
                local_depth = depths[i][pixel_indices][fusion_indices_]
                global_depth = depth_map[pixel_indices][fusion_indices_]

                weights_0 = globals.densities[:, globals.valid[0]][:, mask].repeat(1, 1, 1, 2)
                weights_1 = densities[:, i, pixel_indices][:,fusion_indices_].repeat(1, 1, 1, 2)
                # print('prev_densities.shape:', densities.shape)
                # print('TIME OF PROJECTION:', time.time() - start)
                # start = time.time()
                if use_gru:
                    if not concat_depth:
                        fusion_feat = self.gru(local_latent,
                                            global_latent,
                                            input_weights_emb,
                                            hidden_weights_emb).squeeze(2)
                    else:
                        fusion_feat = self.gru(torch.cat([local_latent, local_depth.view(1,-1,1,1)], dim=-1),
                                            torch.cat([global_latent, global_depth.view(1,-1,1,1)], dim=-1),
                                            input_weights_emb,
                                            hidden_weights_emb).squeeze(2)
                else:
                    fusion_feat = (local_latent * weights_1[...,:1] + global_latent * weights_0[...,:1]) / (weights_0[...,:1] + weights_1[...,:1])
                    fusion_feat = fusion_feat.squeeze(2)
                
                # print('TIME OF GRU:', time.time() - start)
                # start = time.time()

                update_coords = (global_coords[:, mask]*weights_0[...,1] + coords[:, i, pixel_indices]\
                              [:,fusion_indices_,0,0]*weights_1[...,1]) / (weights_0[...,1]+weights_1[...,1])
                update_extrinsics = (globals.extrinsics[:, globals.valid[0]][:, mask]*weights_0[...,:1] + extrinsics[:, i, None]\
                                  *weights_1[...,:1]) / (weights_0[...,:1]+weights_1[...,:1])
                update_depths = (globals.depths[:, globals.valid[0]][:, mask]*weights_0[...,0,0] +
                                                        depths[None, i, pixel_indices][:,fusion_indices_]\
                                                            *weights_1[...,0,0])/(weights_0[...,0,0]+weights_1[...,0,0])
                
                
                # print('TIME OF WEIGHTED SUM:', time.time() - start)
                # start = time.time()
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

                # print('TIME OF UPDATING GAUSSIANS:', time.time() - start)
                # start = time.time()

                globals.append(means=rearrange(
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
                                densities=globals.densities[:, globals.valid[0]][:, mask] + densities[:, i, pixel_indices][:,fusion_indices_],
                                weights=globals.weights[:, globals.valid[0]][:, mask] + weight_emb[:, i, pixel_indices][:,fusion_indices_],
                                coords=update_coords,
                                extrinsics=update_extrinsics,
                                depths=update_depths,
                                mask=mask,
                                scales=globals.scales[:, globals.valid[0]][:, mask] if export_ply else None,
                                rotations=globals.rotations[:, globals.valid[0]][:, mask] if export_ply else None,
                                )
                
                # print('export_ply:', export_ply)

                # print('TIME OF CONCATENATION:', time.time() - start)
                # start = time.time()
            
            
            
            new_extrinsics = extrinsics[:,i,None].repeat(1,(concat_mask).sum(),1,1)
            new_gaussians_feat = gaussians[:,i][:,concat_mask]
            new_densities = densities[:,i][:,concat_mask]
            new_weight_emb = weight_emb[:,i][:,concat_mask]
            new_coords = coords[:,i][:,concat_mask,0,0]
            new_depths = depths[None,i][:,concat_mask]
            new_gaussians = rearrange(
                                    self.to_gaussians(gaussians[:,i][:,concat_mask]),
                                    "... (srf c) -> ... srf c",
                                    srf=self.cfg.num_surfaces,
                                )
                                
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
            # print('TIME OF DECODING NEW GAUSSIANS:', time.time() - start)
            start = time.time()

            globals.append(means=rearrange(
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
                            densities=new_densities,
                            weights=new_weight_emb,
                            coords=new_coords,
                            extrinsics=new_extrinsics,
                            depths=new_depths,
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

            # print('TIME OF APPENDING NEW GAUSSIANS:', time.time() - start)
            start = time.time()
        # globals['densities'] = global_densities
        
        return globals
        # return global_gaussians if not vis else removed, global_coords, global_extrinsics, global_depths

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
