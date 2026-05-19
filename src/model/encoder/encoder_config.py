from dataclasses import dataclass
from typing import Literal, Optional, List

from .common.gaussian_adapter import GaussianAdapterCfg
from .backbone import BackboneCfg
from .epipolar.epipolar_transformer import EpipolarTransformerCfg
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderEpipolarCfg:
    name: Literal[
      "epipolar",
      "encoder_test_unet3d_online",
    ]
    d_feature: int
    num_monocular_samples: int
    num_surfaces: int
    predict_opacity: bool
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    near_disparity: float
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    epipolar_transformer: EpipolarTransformerCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    use_epipolar_transformer: bool
    use_transmittance: bool

    est_depth: Literal["gt", "refine", "est", "cost"]

    # MVSplat params
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool

    # Adaptive Gaussian params
    keypoint_scorer_hidden_dim = [256, 128, 64]
    cga_stages: int = 3
    igr_stages: int = 3
    opacity_thres: float = 0.1
    split_count: int = 1
    scaling_factor: float = 0.5
    opacity_factor: float = 0.5
    cga_num_groups: int = 2
    igr_num_groups: int = 4
    score_embed: int = 32
    use_feat: bool = False
    score_channels = [64, 128, 256, 128, 64, 1]
    num_layers: int = 6
    cga_num_levels: int = 2
    igr_num_levels: int = 4
    cga_num_anchors: int = 25600
    max_num_view: int = 12
    igr_num_anchors: int = 25600
    attn_drop: float = 0.15
    num_learnable_pts: int = 6
    fix_scale = [
        [0.0, 0.0, 0.0],
        [0.45, 0.0, 0.0],
        [-0.45, 0.0, 0.0],
        [0.0, 0.45, 0.0],
        [0.0, -0.45, 0.0],
        [0.0, 0.0, 0.45],
        [0.0, 0.0, -0.45],
      ]

    adaptive_gaussian: bool = False
 
    backbone_limit: int = 3
    num_depth_candidates: int = 64
    unimatch_weights_path: str | None = "checkpoints/gmdepth-scale1-resumeflowthings-scannet-5d9d7964.pth"
    use_pc_encoder: bool = False
    
    load_depth: bool = False
    num_views: int = 2
    image_H: int = 384
    image_W: int = 512
    n_levels: int = -1
    fusion: bool = False
    op1: bool = False
    cv_type: str = 'feat'
    use_planes: bool = True
    log_plane: bool = True
    log_cv: bool = False
    wo_cv_encoder: bool = False
    wo_msd: bool = False
    wo_matchnet: bool = False
    depth_refine: bool = False
    vis: bool = False

    matchnet_type: int = 18
    matchnet_dim: int = 16

    use_gt_depth: bool = False

    max_batch: int = 15
    max_batch_length: int = 1000
    remove: bool = False
    use_em: bool = False
    fore_fusion: bool = False
    concat_depth: bool = False

    use_gru: bool = True
    refine_gs: bool = False
    refine_ws: bool = True
    refine_soft_thres: bool = False
    refine_thres: float = 0.1
    refine_times: int = 1
    refine_window: int = 1

    fusion_pp: bool = False
    refine_pp: bool = False
    refine_uniform: bool = False

    low_res: bool = False

    ft: bool = False
    ft_iter: int = 2000
    ft_dense: bool = False
    opr: int = 3000
    ft_densify: bool = False
    ft_reset: bool = False
    ft_prune: bool = False
    no_spatial_lr: bool = True

    ft_depth_sup: bool = False
    conf_depth_sup: bool = False
    ft_multiplier: float = 1.0

    depth_pad: bool = False
    larger_weight: bool = False

    sh_degree: int = 2
    
    vis_refine: bool = False
    vis_gs: bool = False
    
    use_semantic_fusion: bool = False
    sam_model_type: str = ""
    fastsam_model_path: str = ""
    fastsam_retina_masks: bool = False
    fastsam_conf: float = 0.
    fastsam_iou: float = 0.
    clip_model: str = ""
    clip_model_type: str = ""
    clip_model_path: str = ""
    clip_dim: int = 768
    semantic_early_fusion: bool = False
    non_object_embedding: bool = False

    load_cache: bool = False
    load_clip_model: bool = False
    load_sam_model: bool = False

    unet3d_in_channels: int = 64
    unet3d_out_channels: int = 64
    unet3d_D: int = 3
    unet3d_arch: str = 'MinkUNet14A'
    unet3d_norm_freeze: bool = False
    voxel_size: float = 0.02
    use_openscene: bool = False

    memory: bool = False
    memory_in_channels = [32, 64, 128, 256]
    memory_queue: int = -1
    memory_vmp_layer = [0, 1, 2, 3]
    memory_norm: str = "BN"

    pos_enc: bool = False
    unet_3d: bool = False
    pcd_aug: bool = False

    openvocab_domain: str = "ensemble"
    guidance_type: str = "visual"
    recon_mode: str = "online"
    output_path: str = ""