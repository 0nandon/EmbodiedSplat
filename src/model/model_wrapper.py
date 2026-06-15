from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import moviepy.editor as mpy
import torch
import torch.nn.functional as F
import wandb
from einops import pack, rearrange, repeat
from jaxtyping import Float
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim

from ..dataset.data_module import get_data_shim
from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image
from ..misc.LocalLogger import LOG_PATH, LocalLogger
from ..misc.step_tracker import StepTracker
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)
from ..visualization.camera_trajectory.wobble import (
    generate_wobble,
    generate_wobble_transformation,
)
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode, DecoderOutput
from .encoder import Encoder
from .encoder.encoder_epipolar import UseDepthMode
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .encoder.gaussian_model.gaussian_model import GaussianModel
from .types import Gaussians
from torch.nn.utils import clip_grad_norm_

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import mmcv
import os
import json

from torchvision.utils import save_image as save_image_torch
import MinkowskiEngine as ME

import clip
from ..evaluation.database.class_labels import SCANNET20_CLASS_LABELS

def change_gaussians_dtype(gaussians):
    gaussians.means = gaussians.means.float()
    gaussians.covariances = gaussians.covariances.float()
    gaussians.harmonics = gaussians.harmonics.float()
    gaussians.opacities = gaussians.opacities.float()
    return gaussians

def compute_metrics(rgb_gt, rgb, use_mask=False, pre_mask=None, gt_depth=None):

    zeros = torch.zeros((1,3,1,1), device=rgb.device)
    if use_mask:
        # print(f'rgb.shape: {rgb.shape}, gt_depth.shape: {gt_depth.shape}')
        mask = ((torch.sum((rgb != zeros), dim=1) > 0) | (gt_depth<1e-3) | (gt_depth>10)).min(-1)[0].min(-1)[0]
    else:
        mask = torch.ones((rgb.shape[0],), dtype=torch.bool, device=rgb.device)
    # if pre_mask is not None:
    #     mask = torch.tensor(pre_mask, device=rgb.device, dtype=torch.bool)
    rgb = rgb.clip(min=0, max=1)
    psnr = compute_psnr(rgb_gt, rgb)
    
    # print(f'rgb.shape: {rgb.shape}, rgb_gt.shape: {rgb_gt.shape}')
    n_targets = rgb_gt.shape[0]
    lpips = []
    for j in range(np.ceil(n_targets/20).astype(int)):
        l = len(rgb_gt[j*20:(j+1)*20])
        lpips.append(compute_lpips(rgb_gt[j*20:(j+1)*20], rgb[j*20:(j+1)*20]))
    lpips = torch.cat(lpips, dim=0)

    # try:
    #     lpips = compute_lpips(rgb_gt, rgb)
    # except:
    #     lpips = torch.tensor(0.0, device=rgb.device)
    ssim = compute_ssim(rgb_gt, rgb)
    psnr = (psnr * mask).sum() / (mask.sum()+1e-6)
    lpips = (lpips * mask).sum() / (mask.sum()+1e-6)
    ssim = (ssim * mask).sum() / (mask.sum()+1e-6)
    print('psnr:', psnr, 'lpips:', lpips, 'ssim:', ssim, 'mask:', mask.sum())
    return psnr, lpips, ssim, mask

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

def depth_implicit_loss(encoder_results: dict) -> Float[Tensor, ""]:
    depth_loss = nn.L1Loss()(torch.log(encoder_results[f'depth_num0_s-1'])[encoder_results[f'depth_num0_s-1_mask']], \
                                torch.log(encoder_results[f'depth_num0_s-1_raw'])[encoder_results[f'depth_num0_s-1_mask']])
    
    return depth_loss

def depth_render_loss(prediction, batch) -> Float[Tensor, ""]:
    if not 'depth' in batch['target']:
        return torch.tensor(0.0), torch.tensor(0.0), torch.tensor(0.0)
    target = batch['target']['depth'].squeeze(2)
    # print('target.shape:', target.shape)
    # print('prediction.depth.shape:', prediction.depth.shape)
    pred_bN = rearrange(prediction.depth.clone(), 'b v h w -> (b v) (h w)')
    gt_bN = rearrange(target.clone(), 'b v h w -> (b v) (h w)')[:pred_bN.shape[0]]
    mask = gt_bN > 0.5
    gt_bN[~mask] = torch.nan
    pred_bN[~mask] = torch.nan
    abs_rel_b = torch.nanmean(torch.abs(gt_bN - pred_bN) / gt_bN, dim=1).mean()
    abs_diff_b = torch.nanmean(torch.abs(gt_bN - pred_bN), dim=1).mean()
    thresh_bN = torch.max(torch.stack([(gt_bN / pred_bN), (pred_bN / gt_bN)], 
                                                            dim=2), dim=2)[0]
    a25_val = (thresh_bN < (1.0+0.25)     ).float()
    a25_val[~mask] = torch.nan
    delta_25 = torch.nanmean(a25_val, dim=1).mean()

    a10_val = (thresh_bN < (1.0+0.10)     ).float()
    a10_val[~mask] = torch.nan
    delta_10 = torch.nanmean(a10_val, dim=1).mean()

    # np.save('debug/gt.npy', target.clone().detach().cpu().numpy())
    # np.save('debug/pred.npy', prediction.depth.clone().detach().cpu().numpy())
    # exit(0)

    return abs_diff_b, abs_rel_b, delta_25, delta_10

def depth_pred_loss(encoder_results, batch) -> Float[Tensor, ""]:
    try:
        pred_bN = rearrange(encoder_results[f'depth_num0_s-1_b1hw'].clone(), 'b v h w -> (b v) (h w)')
        gt_bN = rearrange(encoder_results[f'depth_num0_s-1_raw_b1hw'].clone().to(pred_bN.device), 'b v h w -> (b v) (h w)')[:pred_bN.shape[0]]
    except:
        pred_bN = rearrange(encoder_results[f'depth_num0_s0_b1hw'].clone(), 'b v h w -> (b v) (h w)')
        gt_bN = rearrange(encoder_results[f'depth_num0_s0_raw_b1hw'].clone().to(pred_bN.device), 'b v h w -> (b v) (h w)')[:pred_bN.shape[0]]
    # print('pred.shape:', pred_bN.shape)
    # print('gt.shape:', gt_bN.shape)
    # print('pre_pred.range:', pred_bN.min(), pred_bN.max())
    # print('pre_gt.range:', gt_bN.min(), gt_bN.max())
    mask = gt_bN > 0.5
    gt_bN[~mask] = torch.nan
    pred_bN[~mask] = torch.nan
    abs_rel_b = torch.nanmean(torch.abs(gt_bN - pred_bN) / gt_bN, dim=1).mean()
    abs_diff_b = torch.nanmean(torch.abs(gt_bN - pred_bN), dim=1).mean()
    thresh_bN = torch.max(torch.stack([(gt_bN / pred_bN), (pred_bN / gt_bN)], 
                                                            dim=2), dim=2)[0]
    a25_val = (thresh_bN < (1.0+0.25)     ).float()
    a25_val[~mask] = torch.nan
    delta_25 = torch.nanmean(a25_val, dim=1).mean()

    a10_val = (thresh_bN < (1.0+0.10)     ).float()
    a10_val[~mask] = torch.nan
    delta_10 = torch.nanmean(a10_val, dim=1).mean()

    # np.save('debug/gt.npy', target.clone().detach().cpu().numpy())
    # np.save('debug/pred.npy', prediction.depth.clone().detach().cpu().numpy())
    # exit(0)

    return abs_diff_b, abs_rel_b, delta_25, delta_10


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool


@dataclass
class TestCfg:
    output_path: Path
    dense: int = 0
    test_ply: bool = None
    test_fps: bool = False
    test_bev: bool = False
    colmap: bool = False
    eval: bool = False
    stage: str = ""


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    use_depth: UseDepthMode | None
    extended_visualization: bool
    depth_sup: bool = False
    scale_invariant: bool = True
    save_mask: bool = False
    f_count: bool = False
    semantic_sup: bool = False
    semantic_sup_start: int = 3000_000
    semantic_sup_joint_start: int = 3000_000
    stage: str = ""


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: nn.Module
    encoder_visualizer: Optional[EncoderVisualizer]
    decoder: Decoder
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        decoder: Decoder,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        cfg_dict,
        run_dir,
        num_context_views: int = 2,
        dataset_name: str = 'scannet',
        test_bev: bool = False,
    ) -> None:
        super().__init__()

        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.run_dir = run_dir
        self.num_context_views = num_context_views
        self.dataset_name = dataset_name
        self.test_bev = test_bev

        # Set up the model.
        self.encoder = encoder
        self.encoder_visualizer = encoder_visualizer
        self.decoder = decoder
        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()

        self.losses_log = {}
        self.loss_total = []
        self.metrics = {}
        for metric in ['psnr', 'lpips', 'ssim', 'pred_delta_25', 'pred_delta_10', 'render_delta_25', 'render_delta_10']:
            self.metrics[metric] = []
        self.valids = []
        self.masks = {}
        if os.path.exists(f'assets/evaluation_index_scannet_{self.num_context_views}views_mask.json'):
            self.masks = json.load(open(f'assets/evaluation_index_scannet_{self.num_context_views}views_mask.json', 'r'))
        self.test_scene_list = []
        self.test_fvs_list = []

        for k1 in cfg_dict:
            try:
                keys = cfg_dict[k1].keys()
                print(f'{k1}:')
                for k2 in keys:
                    print(f'    {k2}: {cfg_dict[k1][k2]}')
            except:
                print(f'{k1}: {cfg_dict[k1]}')
        
        if test_cfg.eval:
            if dataset_name == 'scannet':
                CLASS_LABELS = [
                    "a {} in a photo".format(cls) for cls in SCANNET20_CLASS_LABELS
                ]
            elif dataset_name == 'scannet200':
                CLASS_LABELS = [
                    "a {} in a photo".format(cls) for cls in SCANNET200_CLASS_LABELS
                ]
            elif dataset_name == 'scannetpp':
                pass # coming soon

            if self.encoder.cfg.clip_model in ["openseg", "lseg"]:
                def build_text_embedding(text):
                    run_on_gpu = torch.cuda.is_available()
                    
                    if self.encoder.cfg.clip_model == "openseg":
                        model, _ = clip.load("ViT-L/14@336px")
                    elif self.encoder.cfg.clip_model == "lseg":
                        model, _ = clip.load("ViT-B/32")
                          
                    with torch.no_grad():
                        text = clip.tokenize(text) #tokenize
                        if run_on_gpu:
                            text = text.cuda()
                        
                        text_embedding = model.encode_text(text)  #embed with text encoder
                        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

                    return text_embedding

                self.classifier = build_text_embedding(CLASS_LABELS)
            elif self.encoder.cfg.clip_model == "maskadapter":
                self.classifier = self.encoder.clip.get_text_classifier(CLASS_LABELS, self.device)
                self.classifier = F.normalize(self.classifier, p=2, dim=-1, eps=1e-5)

    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

    def training_step(self, batch, batch_idx):
        if self.train_cfg.stage == 'online':
            pass
            # torch.cuda.empty_cache()
        
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        
        encoder_results = self.encoder(
            batch["context"], 
            self.global_step, 
            False, 
            is_testing=False, 
            decoder=self.decoder,
            source=batch["source"] if self.test_cfg.dense else None, 
            dense=self.test_cfg.dense
        )
        
        if self.train_cfg.stage == 'online' or self.train_cfg.stage == 'rgb':
            gaussians = encoder_results['gaussians']
        
            if not isinstance(gaussians, list):
                output = self.decoder.forward(
                    gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=self.train_cfg.depth_mode,
                    scale_invariant=self.train_cfg.scale_invariant,
                )
            else:
                output_list = []

                for i, gs in enumerate(gaussians):
                    output_list.append(self.decoder.forward(
                        gs,
                        batch["target"]["extrinsics"][i:i+1],
                        batch["target"]["intrinsics"][i:i+1],
                        batch["target"]["near"][i:i+1],
                        batch["target"]["far"][i:i+1],
                        (h, w),
                        depth_mode=self.train_cfg.depth_mode,
                        scale_invariant=self.train_cfg.scale_invariant
                    ))
                output = DecoderOutput(None, None, None, None, None)
                output.color = torch.cat([x.color for x in output_list], dim=0)

                try:
                    output.depth = torch.cat([x.depth for x in output_list], dim=0)
                except:
                    pass
            output_dr = None
            target_gt = batch["target"]["image"]
        
            if self.train_cfg.stage == 'online':
                idx = gaussians[0].clip_features_2d["idx"]
                invalid = (idx != -1).sum(dim=-1) == 0.
                invalid2 = gaussians[0].clip_features_2d["invalid"]
                invalid2 = (invalid2 == -1)
                invalid_total = invalid | invalid2
        
                pred_features = self.encoder.to_semantic(gaussians[0].clip_features_3d[:, ~invalid_total[0]][0, :])
                pred_features = torch.nn.functional.normalize(pred_features, dim=-1, eps=1e-5)
                batch["target"]['feature'] = {
                    "feat": gaussians[0].clip_features_2d["feat"],
                    "idx": gaussians[0].clip_features_2d["idx"][:, ~invalid_total[0]][0, :],
                    "weight": gaussians[0].clip_features_2d["weight"][:, ~invalid_total[0]][0, :]
                }
                output.feature = pred_features

            psnr_probabilistic = compute_psnr(
                rearrange(target_gt, "b v c h w -> (b v) c h w"),
                rearrange(output.color, "b v c h w -> (b v) c h w"),
            )
            psnr = psnr_probabilistic.mean()
            self.log("train/psnr", psnr, on_step=True, on_epoch=True, sync_dist=True, logger=True)
        
            total_loss = 0
            for loss_fn in self.losses:
            
                if self.train_cfg.stage != 'rgb' and loss_fn.name != 'sem':
                    continue
                
                loss = loss_fn.forward(output, batch, gaussians, encoder_results, self.global_step, output_dr)
                self.log(f"loss/{loss_fn.name}", loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
                total_loss = total_loss + loss
                self.losses_log[loss_fn.name] = self.losses_log.get(loss_fn.name, [])
                self.losses_log[loss_fn.name].append(loss)
            self.log("loss/total", total_loss, on_step=True, on_epoch=True, sync_dist=True, logger=True)
            self.loss_total.append(total_loss)
            context_indices = [x['index'].tolist() for x in batch['context']]
            
            if batch_idx %10 == 0:
                to_print = f"train step {self.global_step}; "+\
                        f"scene = {batch['scene']}; " + \
                        f"context = {context_indices}; " +\
                        f"loss = {torch.mean(torch.tensor(self.loss_total)):.6f} "+\
                        f"psnr = {torch.mean(torch.tensor(psnr)):.2f}"
                for name in self.losses_log:
                    to_print = to_print + f' loss_{name} = {torch.mean(torch.tensor(self.losses_log[name])):.6f}'
                if 'gs_ratio' in encoder_results:
                    to_print = to_print + f' gs_ratio = {torch.mean(torch.tensor(encoder_results["gs_ratio"])):.6f}'
                print(to_print)
                self.losses_log = {}
                self.loss_total = []
        elif self.train_cfg.stage == 'single' or self.train_cfg.stage == 'single_proj':
            total_loss = encoder_results
            self.loss_total.append(total_loss)
            context_indices = [x['index'].tolist() for x in batch['context']]
            
            if batch_idx %10 == 0:
                to_print = f"train step {self.global_step}; "+ \
                        f"scene = {batch['scene']}; " + \
                        f"context = {context_indices}; " + \
                        f"loss = {torch.mean(torch.tensor(self.loss_total)):.6f}"
                print(to_print)
                self.loss_total = []

        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        
        return total_loss

    def optimizer_step(
        self, epoch, batch_idx, optimizer, optimizer_closure,
    ):
        optimizer_closure()
        # Check for NaN or inf values in gradients
        for group in optimizer.param_groups:
            for param in group['params']:
                if param.grad is not None:
                    if torch.any(torch.isnan(param.grad)) or torch.any(torch.isinf(param.grad)):
                        param.grad = None  # Optionally clear gradients to avoid an update
                        print(f"Skipped updating {param} due to NaN or inf values in gradient")
                        return  # Skip the optimizer step entirely

        # If no NaN or inf values are found, proceed with the normal optimizer step
        clip_grad_norm_(self.parameters(), max_norm=0.01)
        optimizer.step()

    def on_after_backward(self):
        # Manually check if gradients are clipped
        max_norm = 0.01  # same as your gradient_clip_val
        total_norm = 0
        for p in self.parameters():
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        if total_norm > max_norm:
            print(f"Gradient norm is too large: {total_norm}")


    def test_step(self, batch, batch_idx):
        if self.test_cfg.stage == 'online' and get_cfg()["model"]["encoder"]["openvocab_domain"] in ["ensemble", "3d"]: 
            for m in self.encoder.unet3d.modules():
                if isinstance(m, ME.MinkowskiBatchNorm):
                    m.train() 
            for m in self.encoder.memory.modules():
                if isinstance(m, ME.MinkowskiBatchNorm):
                    m.train() 

        batch: BatchedExample = self.data_shim(batch)

        (scene,) = batch["scene"]
        print(f'processing {scene}')

        b, v, _, h, w = batch["target"]["image"].shape
        assert b == 1
        if batch_idx % 100 == 0:
            print(f"Test step {batch_idx:0>6}.")
        
        self.test_scene_list.append(scene)
        name = get_cfg()["wandb"]["name"]
        path = self.test_cfg.output_path / name

        if not self.test_cfg.test_ply:
            with self.benchmarker.time("encoder"):
                encoder_results = self.encoder(
                    batch["context"],
                    self.global_step,
                    deterministic=False,
                    is_testing=True,
                    export_ply=self.encoder_visualizer.cfg.export_ply,
                    decoder=self.decoder,
                    source=batch["source"] if self.test_cfg.dense else None,
                    dense=self.test_cfg.dense,
                    scene=batch["scene"][0],
                    target_indices=batch["target"]["index"][0],
                    output_path=self.test_cfg.output_path,
                    target=batch['target'],
                    test_bev=self.test_bev,
                    path = path,
                )
                gaussians = encoder_results['gaussians']
                num_gaussians = encoder_results['num_gaussians']
        
        if not self.test_cfg.test_ply:
            pred_abs_diff, pred_rel_diff, pred_delta_25, pred_delta_10 = depth_pred_loss(encoder_results, batch)
            print(f'pred_abs_diff: {pred_abs_diff}, pred_rel_diff: {pred_rel_diff}, pred_delta_25: {pred_delta_25}, pred_delta_10: {pred_delta_10}')
            self.benchmarker.store('depth_pred_abs_diff', float(pred_abs_diff.detach().cpu().numpy()))
            self.benchmarker.store('depth_pred_rel_diff', float(pred_rel_diff.detach().cpu().numpy()))
            self.benchmarker.store('depth_pred_delta_25', float(pred_delta_25.detach().cpu().numpy()))
            self.benchmarker.store('depth_pred_delta_10', float(pred_delta_10.detach().cpu().numpy()))
    
        if not self.test_cfg.test_ply:
            for i, index, fig in zip(range(len(batch["context"][0]["index"][0])), batch["context"][0]["index"][0], batch["context"][0]["image"][0]):
                s = -1 if not self.encoder.cfg.low_res else 0
                length = len(encoder_results[f"depth_num0_s{s}_b1hw"][0])

                save_image(torch.from_numpy(convert_array_to_pil(encoder_results[f"depth_num0_s{s}_b1hw"][i].cpu().numpy().reshape(h//(1+self.encoder.cfg.low_res),w//(1+self.encoder.cfg.low_res)), no_text=True).transpose(2,0,1)\
                                            .astype(np.float32)/255).to(batch["context"][0]["image"][0].device),
                                            path / scene / f"depth_pred/{index:0>6}.png")
                save_image(torch.from_numpy(convert_array_to_pil(encoder_results[f"depth_num0_s{s}_raw_b1hw"][i].cpu().numpy().reshape(h//(1+self.encoder.cfg.low_res),w//(1+self.encoder.cfg.low_res)), no_text=True).transpose(2,0,1)\
                                            .astype(np.float32)/255).to(batch["context"][0]["image"][0].device),
                                            path / scene / f"depth_pred_gt/{index:0>6}.png")
                save_image(fig, path / scene / f"contexts/{index:0>6}.png")

            if v > 100:
                del encoder_results
                torch.cuda.empty_cache()


        with self.benchmarker.time("decoder", num_calls=v):
            if not isinstance(gaussians, list):
                gaussians = [gaussians]

            if self.test_bev:
                output_bev = self.decoder.forward(
                    gaussians[0],
                    batch["target"]["bev_extrinsics"],
                    batch["target"]["intrinsics"][:, :1],
                    batch["target"]["near"][:, :1],
                    batch["target"]["far"][:, :1],
                    (1440, 1920),
                    depth_mode='depth',
                    scale_invariant=self.train_cfg.scale_invariant,
                    background_color=torch.tensor([1, 1, 1], device=batch["target"]["far"].device).float(),
                )
                index = 0
                color = output_bev.color[0][0]
                save_image(color, path / scene / f"bev/final.png")

                exit(0)

            output_list = []
            n_targets = batch["target"]["extrinsics"].shape[1]
            length = 20 if not self.test_cfg.test_fps else 1
            for i, gs in enumerate(gaussians):
                output = []
                costmap_collate = []
                for j in range(np.ceil(n_targets/length).astype(int)):
                    output.append(self.decoder.forward(
                        gs,
                        batch["target"]["extrinsics"][i:i+1, j*length:(j+1)*length],
                        batch["target"]["intrinsics"][i:i+1, j*length:(j+1)*length],
                        batch["target"]["near"][i:i+1, j*length:(j+1)*length],
                        batch["target"]["far"][i:i+1, j*length:(j+1)*length],
                        (h, w),
                        depth_mode='depth',
                        scale_invariant=self.train_cfg.scale_invariant,
                        render_feature=False
                    ))
                    
                now = DecoderOutput(None, None, None, None, None)
                now.color = torch.cat([x.color for x in output], dim=1)
                now.depth = torch.cat([x.depth for x in output], dim=1)
                now.feature = torch.cat(costmap_collate, dim=1) if len(costmap_collate) > 0 else None
                output_list.append(now)
                
            output = DecoderOutput(None, None, None, None, None)
            output.color = torch.cat([x.color for x in output_list], dim=0)
            try:
                output.depth = torch.cat([x.depth for x in output_list], dim=0)
            except:
                pass
            
            if output_list[0].feature is not None:
                output.feature = torch.cat([x.feature for x in output_list], dim=0)
            
            if self.encoder.cfg.vis_refine:
                output_list = []
                n_targets = batch["context"][0]["extrinsics"].shape[1]
                length = 20 if not self.test_cfg.test_fps else 1
                for i, gs in enumerate(gaussians):
                    output_context = []
                    for j in range(np.ceil(n_targets/length).astype(int)):
                        output_context.append(self.decoder.forward(
                            gs,
                            batch["context"][0]["extrinsics"][i:i+1, j*length:(j+1)*length],
                            batch["context"][0]["intrinsics"][i:i+1, j*length:(j+1)*length],
                            batch["context"][0]["near"][i:i+1, j*length:(j+1)*length],
                            batch["context"][0]["far"][i:i+1, j*length:(j+1)*length],
                            (h//2, w//2),
                            depth_mode='depth',
                            scale_invariant=self.train_cfg.scale_invariant,
                        ))
                    now = DecoderOutput(None, None, None, None, None)
                    now.color = torch.cat([x.color for x in output_context], dim=1)
                    now.depth = torch.cat([x.depth for x in output_context], dim=1)
                    output_list.append(now)
                output_context = DecoderOutput(None, None, None, None, None)
                output_context.color = torch.cat([x.color for x in output_list], dim=0)
                try:
                    output_context.depth = torch.cat([x.depth for x in output_list], dim=0)
                except:
                    pass
        
        render_abs_diff, render_rel_diff, render_delta_25, render_delta_10 = depth_render_loss(output, batch)
        print(f'render_abs_diff: {render_abs_diff}, render_rel_diff: {render_rel_diff}, render_delta_25: {render_delta_25}, render_delta_10: {render_delta_10}')
        self.benchmarker.store('depth_render_abs_diff', float(render_abs_diff.detach().cpu().numpy()))
        self.benchmarker.store('depth_render_rel_diff', float(render_rel_diff.detach().cpu().numpy()))
        self.benchmarker.store('depth_render_delta_25', float(render_delta_25.detach().cpu().numpy()))
        self.benchmarker.store('depth_render_delta_10', float(render_delta_10.detach().cpu().numpy()))
        
        fvs_length = batch["target"]["test_fvs"]
        test_fvs = fvs_length > 0   

        count = 0
        try:
            for index, depth_render, depth_gt in zip(batch["target"]["index"][0], output.depth[0], batch["target"]['depth'][0,:,0]):
                save_image(torch.from_numpy(convert_array_to_pil(depth_render.cpu().numpy(), no_text=True).transpose(2,0,1)\
                                                .astype(np.float32)/255).to(batch["context"][0]["image"][0].device),
                                                path / scene / f"depth_render/{index:0>6}.png")
                save_image(torch.from_numpy(convert_array_to_pil(depth_gt.cpu().numpy(), no_text=True).transpose(2,0,1)\
                                                .astype(np.float32)/255).to(batch["context"][0]["image"][0].device),
                                                path / scene / f"depth_render_gt/{index:0>6}.png")
        except:
            pass

        if self.encoder.cfg.vis_refine:
            for index, depth_render, depth_gt in zip(batch["context"][0]["index"][0], output_context.depth[0], batch["context"][0]['depth'][0,:,0]):
                save_image(torch.from_numpy(convert_array_to_pil(depth_render.cpu().numpy(), no_text=True).transpose(2,0,1)\
                                                .astype(np.float32)/255).to(batch["context"][0]["image"][0].device),
                                                path / scene / f"depth_render_context/{index:0>6}.png")

        for i, (index, color, color_gt) in enumerate(zip(batch["target"]["index"][0], output.color[0], batch["target"]["image"][0])):
            if not test_fvs:
                save_image(color, path / scene / f"color/{index:0>6}.png")
                save_image(color_gt, path / scene / f"color_gt/{index:0>6}.png")
            else:
                if count < batch["target"]["index"][0].shape[0]-fvs_length:
                    #if i - 30 >= 0:
                    save_image(color, path / scene / f"interpolation/{index:0>6}.png")
                    save_image(color_gt, path / scene / f"interpolation_gt/{index:0>6}.png")
                else:
                    save_image(color, path / scene / f"extrapolation/{index:0>6}.png")
                    save_image(color_gt, path / scene / f"extrapolation_gt/{index:0>6}.png")
                count += 1
        
        if not test_fvs:
            if scene in self.masks:
                pre_mask = self.masks[scene]
            else:
                pre_mask = None
            
            psnr, lpips, ssim, masks = compute_metrics(batch["target"]["image"][0], output.color[0], 
                                                       use_mask=self.train_cfg.save_mask,
                                                       gt_depth=batch["target"]["depth"].squeeze())
            valid = masks.sum()
            if pre_mask is None:
                self.masks[scene] = masks.tolist()

            self.benchmarker.store('psnr_inter', float(psnr.detach().cpu().numpy()))
            self.benchmarker.store('lpips_inter', float(lpips.detach().cpu().numpy()))
            self.benchmarker.store('ssim_inter', float(ssim.detach().cpu().numpy()))
            self.benchmarker.store('valid_inter', float(valid))
            self.benchmarker.store('num_gaussians', num_gaussians)
            self.test_fvs_list.append(False)
        else:
            length = batch["target"]["index"][0].shape[0]
            psnr_inter, lpips_inter, ssim_inter, masks_inter = compute_metrics(batch["target"]["image"][0][:length-fvs_length], 
                                                                  output.color[0][:length-fvs_length], use_mask=self.train_cfg.save_mask,
                                                                  gt_depth=batch["target"]["depth"].squeeze()[:length-fvs_length])
            psnr_extra, lpips_extra, ssim_extra, masks_extra = compute_metrics(batch["target"]["image"][0][length-fvs_length:],
                                                                  output.color[0][length-fvs_length:], use_mask=self.train_cfg.save_mask,
                                                                  gt_depth=batch["target"]["depth"].squeeze()[length-fvs_length:])
            valid_inter, valid_extra = masks_inter.sum(), masks_extra.sum()
            self.masks[scene] = torch.cat([masks_inter, masks_extra]).tolist()
            self.benchmarker.store('psnr_inter', float(psnr_inter.detach().cpu().numpy()))
            self.benchmarker.store('lpips_inter', float(lpips_inter.detach().cpu().numpy()))
            self.benchmarker.store('ssim_inter', float(ssim_inter.detach().cpu().numpy()))
            self.benchmarker.store('valid_inter', float(valid_inter))
            self.benchmarker.store('psnr_extra', float(psnr_extra.detach().cpu().numpy()))
            self.benchmarker.store('lpips_extra', float(lpips_extra.detach().cpu().numpy()))
            self.benchmarker.store('ssim_extra', float(ssim_extra.detach().cpu().numpy()))
            self.benchmarker.store('valid_extra', float(valid_extra))
            self.benchmarker.store('num_gaussians', num_gaussians)
            self.test_fvs_list.append(True)
        # if self.encoder_visualizer is not None and not self.test_cfg.test_ply:
        #     for k, image in self.encoder_visualizer.visualize(
        #         encoder_results, batch["context"][0], batch_idx, out_path=self.test_cfg.output_path / 'gaussians',
        #         scene_name=scene,
        #     ).items():
        #         self.logger.log_image(k, [prep_image(image)], step=self.global_step)
        """ """
        
    def on_test_end(self) -> None:
        pass
        # breakpoint()
    
        name = get_cfg()["wandb"]["name"]
        self.benchmarker.dump(self.test_cfg.output_path / name / "benchmark.json")
        self.benchmarker.dump_memory(
            self.test_cfg.output_path / name / "peak_memory.json"
        )
        self.benchmarker.dump_stats(
            self.test_cfg.output_path / name / "stats.json"
        )
        filename = 'test_idx.txt'
        if self.train_cfg.save_mask and not os.path.exists(f'assets/evaluation_index_scannet_{self.num_context_views}views_mask.json'):
            json.dump(self.masks, open(f'assets/evaluation_index_scannet_{self.num_context_views}views_mask.json', 'w'))
        
        for i in range(min(len(self.test_scene_list), len(self.test_fvs_list))):
            if not self.test_cfg.test_ply:
                if self.test_fvs_list[i]:
                    print(self.test_scene_list[i], self.benchmarker.benchmarks['psnr_inter'][i], 
                                    self.benchmarker.benchmarks['ssim_inter'][i],
                                    self.benchmarker.benchmarks['lpips_inter'][i],
                                    self.benchmarker.benchmarks['psnr_extra'][int(np.sum(self.test_fvs_list[:i]))], 
                                    self.benchmarker.benchmarks['ssim_extra'][int(np.sum(self.test_fvs_list[:i]))],
                                    self.benchmarker.benchmarks['lpips_extra'][int(np.sum(self.test_fvs_list[:i]))],
                                    self.benchmarker.benchmarks['depth_pred_abs_diff'][i],
                                    self.benchmarker.benchmarks['depth_pred_rel_diff'][i],
                                    self.benchmarker.benchmarks['depth_pred_delta_25'][i],
                                    self.benchmarker.benchmarks['depth_pred_delta_10'][i],
                                    self.benchmarker.benchmarks['depth_render_abs_diff'][i],
                                    self.benchmarker.benchmarks['depth_render_rel_diff'][i],
                                    self.benchmarker.benchmarks['depth_render_delta_25'][i],
                                    self.benchmarker.benchmarks['depth_render_delta_10'][i])
                else:
                    try:
                        print(self.test_scene_list[i], self.benchmarker.benchmarks['psnr_inter'][i], 
                                        self.benchmarker.benchmarks['ssim_inter'][i],
                                        self.benchmarker.benchmarks['lpips_inter'][i],
                                        self.benchmarker.benchmarks['depth_pred_abs_diff'][i],
                                        self.benchmarker.benchmarks['depth_pred_rel_diff'][i],
                                        self.benchmarker.benchmarks['depth_pred_delta_25'][i],
                                        self.benchmarker.benchmarks['depth_pred_delta_10'][i],
                                        self.benchmarker.benchmarks['depth_render_abs_diff'][i],
                                        self.benchmarker.benchmarks['depth_render_rel_diff'][i],
                                        self.benchmarker.benchmarks['depth_render_delta_25'][i],
                                        self.benchmarker.benchmarks['depth_render_delta_10'][i])
                    except:
                        pass
            else:
                if self.test_fvs_list[i]:
                    print(self.test_scene_list[i], self.benchmarker.benchmarks['psnr_inter'][i], 
                                    self.benchmarker.benchmarks['ssim_inter'][i],
                                    self.benchmarker.benchmarks['lpips_inter'][i],
                                    self.benchmarker.benchmarks['psnr_extra'][int(np.sum(self.test_fvs_list[:i]))], 
                                    self.benchmarker.benchmarks['ssim_extra'][int(np.sum(self.test_fvs_list[:i]))],
                                    self.benchmarker.benchmarks['lpips_extra'][int(np.sum(self.test_fvs_list[:i]))],
                                    self.benchmarker.benchmarks['depth_render_abs_diff'][i],
                                    self.benchmarker.benchmarks['depth_render_rel_diff'][i],
                                    self.benchmarker.benchmarks['depth_render_delta_25'][i],
                                    self.benchmarker.benchmarks['depth_render_delta_10'][i])
                                    
                else:
                    print(self.test_scene_list[i], self.benchmarker.benchmarks['psnr_inter'][i], 
                                    self.benchmarker.benchmarks['ssim_inter'][i],
                                    self.benchmarker.benchmarks['lpips_inter'][i],
                                    self.benchmarker.benchmarks['depth_render_abs_diff'][i],
                                    self.benchmarker.benchmarks['depth_render_rel_diff'][i],
                                    self.benchmarker.benchmarks['depth_render_delta_25'][i],
                                    self.benchmarker.benchmarks['depth_render_delta_10'][i])

        if not self.test_cfg.test_ply:
            if self.num_context_views <= 10:
                psnr_avg = (np.array(self.benchmarker.benchmarks['psnr_inter']) 
                                * np.array(self.benchmarker.benchmarks['valid_inter'])).sum() \
                                / np.array(self.benchmarker.benchmarks['valid_inter']).sum()
                ssim_avg = (np.array(self.benchmarker.benchmarks['ssim_inter']) 
                                * np.array(self.benchmarker.benchmarks['valid_inter'])).sum() \
                                / np.array(self.benchmarker.benchmarks['valid_inter']).sum()
                lpips_avg = (np.array(self.benchmarker.benchmarks['lpips_inter']) 
                                * np.array(self.benchmarker.benchmarks['valid_inter'])).sum() \
                                / np.array(self.benchmarker.benchmarks['valid_inter']).sum()
            else:
                psnr_avg = np.array(self.benchmarker.benchmarks['psnr_inter']).mean()
                ssim_avg = np.array(self.benchmarker.benchmarks['ssim_inter']).mean()
                lpips_avg = np.array(self.benchmarker.benchmarks['lpips_inter']).mean()
            print('psnr_inter_avg:', psnr_avg, 
                'ssim_inter_avg:', ssim_avg,
                'lpips_inter_avg:', lpips_avg,
                'depth_pred_abs_diff_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_pred_abs_diff'])).cpu().numpy(),
                'depth_pred_rel_diff_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_pred_rel_diff'])).cpu().numpy(),
                'depth_pred_delta_25_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_pred_delta_25'])).cpu().numpy(),
                'depth_pred_delta_10_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_pred_delta_10'])).cpu().numpy(),
                'depth_render_abs_diff_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_render_abs_diff'])).cpu().numpy(),
                'depth_render_rel_diff_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_render_rel_diff'])).cpu().numpy(),
                'depth_render_delta_25_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_render_delta_25'])).cpu().numpy(),
                'depth_render_delta_10_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_render_delta_10'])).cpu().numpy())
        else:
            print('psnr_inter_avg:', np.array(self.benchmarker.benchmarks['psnr_inter']).mean(), 
                'ssim_inter_avg:', np.array(self.benchmarker.benchmarks['ssim_inter']).mean(),
                'lpips_inter_avg:', np.array(self.benchmarker.benchmarks['lpips_inter']).mean(),'depth_render_abs_diff_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_render_abs_diff'])).cpu().numpy(),
                'depth_render_rel_diff_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_render_rel_diff'])).cpu().numpy(),
                'depth_render_delta_25_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_render_delta_25'])).cpu().numpy(),
                'depth_render_delta_10_avg:', torch.nanmean(torch.tensor(self.benchmarker.benchmarks['depth_render_delta_10'])).cpu().numpy())
        try:
            print('psnr_extra_avg:', (np.array(self.benchmarker.benchmarks['psnr_extra']) 
                                * np.array(self.benchmarker.benchmarks['valid_extra'])).sum()
                                / np.array(self.benchmarker.benchmarks['valid_extra']).sum(), 
                'ssim_extra_avg:', (np.array(self.benchmarker.benchmarks['ssim_extra']) 
                                    * np.array(self.benchmarker.benchmarks['valid_extra'])).sum()
                                    / np.array(self.benchmarker.benchmarks['valid_extra']).sum(),
                'lpips_extra_avg:', (np.array(self.benchmarker.benchmarks['lpips_extra']) 
                                    * np.array(self.benchmarker.benchmarks['valid_extra'])).sum()
                                    / np.array(self.benchmarker.benchmarks['valid_extra']).sum())
        except:
            pass
        print('num_gaussians_avg:', self.benchmarker.benchmarks['num_gaussians_avg'])

    @rank_zero_only
    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        context_indices = [x['index'].tolist() for x in batch['context']]

        (scene,) = batch["scene"]
        print(
            f"validation step {self.global_step}; "
            f"scene = {batch['scene']}; "
            f"context = {context_indices}"
        )

        # Render Gaussians.
        b, _, _, h, w = batch["target"]["image"].shape
        assert b == 1
        encoder_probabilistic_results = self.encoder(
            batch["context"],
            self.global_step,
            deterministic=False,
            is_testing=True,
            decoder=self.decoder,
            source=batch["source"] if self.test_cfg.dense else None,
            dense=self.test_cfg.dense,
        )
        gaussians_probabilistic = encoder_probabilistic_results['gaussians']
        if not isinstance(gaussians_probabilistic, list):
            output_probabilistic = self.decoder.forward(
                gaussians_probabilistic,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode='depth',
                scale_invariant=self.train_cfg.scale_invariant,
            )

        else:
            output_probabilistic_list = []
            n_targets = batch["target"]["extrinsics"].shape[1]
            for i, gs in enumerate(gaussians_probabilistic):
                output = []
                for j in range(np.ceil(n_targets/20).astype(int)):
                    output.append(self.decoder.forward(
                        gs,
                        batch["target"]["extrinsics"][i:i+1, j*20:(j+1)*20],
                        batch["target"]["intrinsics"][i:i+1, j*20:(j+1)*20],
                        batch["target"]["near"][i:i+1, j*20:(j+1)*20],
                        batch["target"]["far"][i:i+1, j*20:(j+1)*20],
                        (h, w),
                        depth_mode='depth',
                        scale_invariant=self.train_cfg.scale_invariant,
                    ))
                now = DecoderOutput(None, None, None, None, None)
                now.color = torch.cat([x.color for x in output], dim=1)
                now.depth = torch.cat([x.depth for x in output], dim=1)
                output_probabilistic_list.append(now)

            output_probabilistic = DecoderOutput(None, None, None, None, None)
            # try:
            output_probabilistic.color = torch.cat([x.color for x in output_probabilistic_list], dim=0)
            try:
                output_probabilistic.depth = torch.cat([x.depth for x in output_probabilistic_list], dim=0)
            except:
                pass
        output_dr = None
        rgb_probabilistic = output_probabilistic.color[0]

        # Compute validation metrics.
        rgb_gt = batch["target"]["image"][0]
        tag, rgb = "probabilistic", rgb_probabilistic
        psnr, lpips, ssim, masks = compute_metrics(rgb_gt, rgb, use_mask=self.encoder.cfg.use_gt_depth,
                                                       gt_depth=batch["target"]["depth"][0,0])
        valid = masks.sum()
        self.log(f"val/psnr_{tag}", psnr)
        self.log(f"val/lpips_{tag}", lpips)
        self.log(f"val/ssim_{tag}", ssim)
        pred_abs_diff, pred_rel_diff, pred_delta_25, pred_delta_10 = depth_pred_loss(encoder_probabilistic_results, batch)
        self.log(f"val/depth_pred_abs_diff_{tag}", pred_abs_diff)
        self.log(f"val/depth_pred_rel_diff_{tag}", pred_rel_diff)
        self.log(f"val/depth_pred_delta_25_{tag}", pred_delta_25)
        self.log(f"val/depth_pred_delta_10_{tag}", pred_delta_10)

        render_abs_diff, render_rel_diff, render_delta_25, render_delta_10 = depth_render_loss(output_probabilistic, batch)
        self.log(f"val/depth_render_abs_diff_{tag}", render_abs_diff)
        self.log(f"val/depth_render_rel_diff_{tag}", render_rel_diff)
        self.log(f"val/depth_render_delta_25_{tag}", render_delta_25)
        self.log(f"val/depth_render_delta_10_{tag}", render_delta_10)
        sign = 1
        for metric in ['psnr', 'lpips', 'ssim', 'pred_delta_25', 'pred_delta_10', 'render_delta_25', 'render_delta_10'] if sign else ['psnr', 'lpips', 'ssim']:
            self.metrics[metric].append(eval(metric).detach().cpu().numpy())
        self.valids.append(valid.item())

        # Construct comparison image.
        if not self.train_cfg.depth_sup:
            context_figs = []
            for fig in batch["context"][0]["image"][0]:
                context_figs.append(fig)
            if 'depth' in batch["context"]:
                for fig in batch["context"][0]["depth"][0]:
                    context_figs.append(torch.from_numpy(convert_array_to_pil(fig.cpu().numpy()[0]).transpose(2,0,1)\
                                                        .astype(np.float32)/255).to(batch["context"][0]["image"][0].device))
            comparison = hcat(
                add_label(vcat(*context_figs), "Context"),
                add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                add_label(vcat(*rgb_probabilistic), "Target (Probabilistic)"),
            )
        else:
            context_figs = []
            
            context_depth_figs = []
            pred_depth_figs = []
            phase = len(batch["context"])
            for p in range(phase):
                for fig in batch["context"][p]["image"][0]:
                    context_figs.append(fig)
                # for s in range(-1,0):
                s = 0 if self.encoder.cfg.low_res else -1
                length = len(encoder_probabilistic_results[f"depth_num{p}_s{s}"][0])
                for i in range(length):
                    try:
                        context_depth_figs.append(torch.from_numpy(convert_array_to_pil(mmcv.imresize(batch["context"][p][f"depth_s{s}"][0][i][0].cpu().numpy(), (w,h),interpolation='nearest')).transpose(2,0,1)\
                                                                .astype(np.float32)/255).to(batch["context"][p]["image"][0].device))
                    except:
                        pass
                    try:
                        pred_depth_figs.append(torch.from_numpy(convert_array_to_pil(encoder_probabilistic_results[f"depth_num{p}_s{s}"][0][i].cpu().numpy().reshape(h,w)).transpose(2,0,1)\
                                                            .astype(np.float32)/255).to(batch["context"][p]["image"][0].device))
                    except:
                        pred_depth_figs.append(torch.from_numpy(convert_array_to_pil(mmcv.imresize(encoder_probabilistic_results[f"depth_num{p}_s{s}"][0][i].cpu().numpy().reshape(h//(2**(s+1)), w//(2**(s+1))), (w,h),interpolation='nearest'))\
                                                                        .transpose(2,0,1).astype(np.float32)/255).to(batch["context"][p]["image"][0].device))

            if output_dr is not None:
                context_depth_render_figs = []
                for fig in output_dr.depth[0]:
                    context_depth_render_figs.append(torch.from_numpy(convert_array_to_pil(mmcv.imresize(fig.cpu().numpy(), (w,h),interpolation='nearest')).transpose(2,0,1)\
                                                            .astype(np.float32)/255).to(batch["context"][0]["image"][0].device))
                comparison = hcat(
                        add_label(vcat(*context_figs), "Context"),
                        add_label(vcat(*context_depth_figs), "Context GT Depths"),
                        add_label(vcat(*pred_depth_figs), "Depths Predictions"),
                        add_label(vcat(*context_depth_render_figs), "Context Rendered Depths"),
                        add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                        add_label(vcat(*rgb_probabilistic), "Target (Predictions)"),
                    )
                
            else:
                try:
                    comparison = hcat(
                    add_label(vcat(*context_figs), "Context"),
                    add_label(vcat(*context_depth_figs), "Context GT Depths"),
                    add_label(vcat(*pred_depth_figs), "Depths Predictions"),
                    add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                    add_label(vcat(*rgb_probabilistic), "Target (Predictions)"),
                )
                except:
                    comparison = hcat(
                        add_label(vcat(*context_figs), "Context"),
                        add_label(vcat(*pred_depth_figs), "Depths Predictions"),
                        add_label(vcat(*rgb_gt), "Target (Ground Truth)"),
                        add_label(vcat(*rgb_probabilistic), "Target (Predictions)"),
                    )
        try:
            self.logger.log_image(
                "comparison",
                [prep_image(add_border(comparison))],
                step=self.global_step,
                caption=batch["scene"],
            )
        except:
            pass
  

        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                encoder_probabilistic_results, batch["context"][0], self.global_step, out_path=self.test_cfg.output_path
            ).items():
                self.logger.log_image(k, [prep_image(image)], step=self.global_step)


    def on_validation_end(self) -> None:
        with open(self.run_dir + "/val_metrics.txt", "a") as f:
            line = '' 
            for metric in ['psnr', 'lpips', 'ssim', 'pred_delta_25', 'pred_delta_10', 'render_delta_25', 'render_delta_10']:
                # try:
                # line = line + f'{metric}=' + str((np.array(self.metrics[metric])*np.array(self.valids)).sum() / np.array(self.valids).sum()) + ' '
                line = line + f'{metric}=' + str(np.array(self.metrics[metric]).mean()) + ' '
                # except:
                #     pass
            f.write(line + '\n')
            print(line)
        for metric in ['psnr', 'lpips', 'ssim', 'pred_delta_25', 'pred_delta_10', 'render_delta_25', 'render_delta_10']:
            self.metrics[metric] = []
            self.valids = []

    @rank_zero_only
    def render_video_wobble(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(batch, trajectory_fn, "wobble", num_frames=60)

    @rank_zero_only
    def render_video_interpolation(self, batch: BatchedExample) -> None:
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0],
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0],
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(batch, trajectory_fn, "rgb")

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, batch: BatchedExample) -> None:
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                batch["context"]["extrinsics"][0, 1]
                if v == 2
                else batch["target"]["extrinsics"][0, 0],
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                batch["context"]["intrinsics"][0, 1]
                if v == 2
                else batch["target"]["intrinsics"][0, 0],
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            batch,
            trajectory_fn,
            "interpolation_exagerrated",
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = True,
    ) -> None:
        # Render probabilistic estimate of scene.
        gaussians_prob = self.encoder(batch["context"], self.global_step, False, is_testing=False)['gaussians']
        gaussians_det = self.encoder(batch["context"], self.global_step, True, is_testing=False)['gaussians']

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)

        _, _, _, h, w = batch["context"]["image"].shape

        # Color-map the result.
        def depth_map(result):
            near = result[result > 0][:16_000_000].quantile(0.01).log()
            far = result.view(-1)[:16_000_000].quantile(0.99).log()
            result = result.log()
            result = 1 - (result - near) / (far - near)
            return apply_color_map_to_image(result, "turbo")

        near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
        far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
        output_prob = self.decoder.forward(
            gaussians_prob, extrinsics, intrinsics, near, far, (h, w), "depth",
            scale_invariant=self.train_cfg.scale_invariant,
        )
        images_prob = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_prob.color[0], depth_map(output_prob.depth[0]))
        ]
        output_det = self.decoder.forward(
            gaussians_det, extrinsics, intrinsics, near, far, (h, w), "depth",
            scale_invariant=self.train_cfg.scale_invariant,
        )
        images_det = [
            vcat(rgb, depth)
            for rgb, depth in zip(output_det.color[0], depth_map(output_det.depth[0]))
        ]
        images = [
            add_border(
                hcat(
                    add_label(image_prob, "Probabilistic"),
                    add_label(image_det, "Deterministic"),
                )
            )
            for image_prob, image_det in zip(images_prob, images_det)
        ]

        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]
        visualizations = {
            f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
        }

        # Since the PyTorch Lightning doesn't support video logging, log to wandb directly.
        try:
            wandb.log(visualizations)
        except Exception:
            assert isinstance(self.logger, LocalLogger)
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=value._fps)
                dir = LOG_PATH / key
                dir.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(
                    str(dir / f"{self.global_step:0>6}.mp4"), logger=None
                )

    def configure_optimizers(self):
        if not self.encoder.cfg.adaptive_gaussian:
            optimizer = optim.Adam(self.parameters(), lr=self.optimizer_cfg.lr)
        else:
            # Parameters of the main model without the iterative refiner
            base_params = [p for n, p in self.named_parameters() if "iterative_refiner" not in n]
            # Parameters of the iterative refiner
            # refiner_params = [p for n, p in self.named_parameters() if "iterative_refiner" in n]
            refiner_params = list(self.encoder.iterative_refiner.parameters())
            # print('refiner_params:', refiner_params)
            optimizer = optim.Adam([
                {'params': base_params},  # Uses default lr
                {'params': refiner_params, 'lr': 0.1 * self.optimizer_cfg.lr}  # Adjusted learning rate
            ], lr=self.optimizer_cfg.lr)

            # for name, param in self.named_parameters():
            #     if param.requires_grad:
            #         param.register_hook(lambda grad: print(name, grad))

        warm_up_steps = self.optimizer_cfg.warm_up_steps
        if self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, self.optimizer_cfg.lr if not self.encoder.cfg.adaptive_gaussian \
                                else [self.optimizer_cfg.lr, self.optimizer_cfg.lr * 0.1],
                            self.trainer.max_steps + 1,
                            pct_start=0.001,
                            cycle_momentum=False,
                            anneal_strategy='cos',
                        )
        else:
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_after_backward(self):
        if self.encoder.cfg.adaptive_gaussian:
            # Gradient clipping for the iterative refiner
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.1)
