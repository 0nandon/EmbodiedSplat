from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch.nn as nn
import torch

from .losses import MSGradientLoss, MVDepthLoss, NormalsLoss, ScaleInvariantLoss
import torch.nn.functional as F


from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import mmcv
import os

from .utils.geometry_utils import NormalGenerator


def convert_array_to_pil(depth_map):
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
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 40)
    draw.text((20,20), '[%.2f, %.2f]'%(min_depth, max_depth), (255,255,255), font=font)
    colormapped_im = image

    return colormapped_im


@dataclass
class LossDSCfg:
    weight: float
    grad: bool = False
    mvd: bool = False
    ms: bool = False
    use_confidence: bool = False
    alpha: float = 0.2


@dataclass
class LossDSCfgWrapper:
    ds: LossDSCfg


class LossDS(Loss[LossDSCfg, LossDSCfgWrapper]):

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        encoder_results: dict,
        global_step: int,
        dr_prediction: DecoderOutput | None = None,
    ) -> Float[Tensor, ""]:
        # print('depth_mask_ratio:', encoder_results['depths_mask'].sum()/torch.prod(torch.tensor(encoder_results['depths_mask'].shape)))
        depth_loss = 0

        # s = -1
        # if confidence is None:
        # for s in range(-1+self.cfg.ms*4, -2, -1):
        # for s in range(-1, 1):
        #     try:
        s = 0
        depth_pred = encoder_results[f'depth_num0_s{s}_b1hw']
        depth_gt = encoder_results[f'depth_num0_s{s}_raw_b1hw']
        mask = (depth_gt > 1e-3) * (depth_gt < 10)

        if not self.cfg.use_confidence:
            now_loss = nn.L1Loss(reduction='none')(torch.log(depth_pred[mask]+1e-8), torch.log(depth_gt[mask]))
        else:
            now_loss = nn.L1Loss(reduction='none')(torch.log(depth_pred[mask]+1e-8), torch.log(depth_gt[mask]))
            confidence = (1 + torch.exp(encoder_results[f'weight_num0_s{s}_b1hw'][mask])).clip(1, 100)
            # weight by confidence

            # print(f'now_loss.shape: {now_loss.shape}, confidence.shape: {confidence.shape}')
            # print('now_loss.mean():', now_loss.mean())
            now_loss = now_loss * confidence - self.cfg.alpha * torch.log(confidence)

            # # average + nan protection (in case of no valid pixels at all)
            # conf_loss1 = conf_loss1.mean() if conf_loss1.numel() > 0 else 0
            # conf_loss2 = conf_loss2.mean() if conf_loss2.numel() > 0 else 0
        
        depth_loss += now_loss.mean() * self.cfg.weight
            # except:
            #     pass

        # log_depth_pred_resized = F.interpolate(
        #                         encoder_results[f'log_depth_num0_s{s}_b1hw'], 
        #                         size=encoder_results[f'depth_num0_s0_raw_b1hw'].shape[-2:],
        #                         mode="nearest",
        #                     )

        # depth_loss += (1/(2**s)) * nn.L1Loss()(log_depth_pred_resized[encoder_results[f'depth_num0_s{s}_mask_b1hw']], \
        #                     torch.log(encoder_results[f'depth_num0_s{s}_raw_b1hw'])[encoder_results[f'depth_num0_s{s}_mask_b1hw']]) * self.cfg.weight
        
        # num = len(os.listdir('debug')) // 3
        # np.save(f'debug/depth_pred_{num}.npy', torch.exp(log_depth_pred_resized[0,0]).detach().cpu().numpy())
        # np.save(f'debug/depth_mask_{num}.npy', encoder_results[f'depth_num0_s0_mask_b1hw'][0,0].detach().cpu().numpy())
        # np.save(f'debug/depth_gt_{num}.npy', encoder_results[f'depth_num0_s0_raw_b1hw'][0,0].detach().cpu().numpy())
        
        
        # exit(0)
            # print('depth loss s:', s, depth_loss)
        # except:
        #     pass
        # convert_array_to_pil(torch.exp(log_depth_pred_resized[0,0]).detach().cpu().numpy()).save('pred.png')
        # convert_array_to_pil(encoder_results[f'depth_num0_s0_raw_b1hw'][0,0].detach().cpu().numpy()).save('target.png')
        # print('gt.shape:', encoder_results[f'depth_num0_s0_raw_b1hw'].shape)
        # print('pred.shape:', encoder_results[f'depth_num0_s0_b1hw'].shape)

        # compute_normals = NormalGenerator(batch['context']['image_shape'][0] // 2, 
        #                                   batch['context']['image_shape'][1] // 2)

        # normals_gt = compute_normals(depth_gt, batch['context']["invK_s0_b44"])

        # # estimate normals for depth
        # normals_pred = compute_normals(depth_pred, batch['context']["invK_s0_b44"])

        # normals_loss = self.normals_loss(normals_gt, normals_pred)

        if self.cfg.grad:
            depth_loss += MSGradientLoss()(encoder_results[f'depth_num0_s-1_raw_b1hw'], 
                                        encoder_results[f'depth_num0_s-1_b1hw']) * self.cfg.weight      
        
        if self.cfg.mvd:
            mv_depth_loss = MVDepthLoss(
                                batch['context']['image_shape'][0] // 2,
                                batch['context']['image_shape'][1] // 2,
                            )
            depth_loss += mv_depth_loss(
                            depth_pred_b1hw=depth_pred,
                            cur_depth_b1hw=depth_gt,
                            src_depth_bk1hw=src_data["depth_b1hw"],
                            cur_invK_b44=encoder_results['cur_invK'],
                            src_K_bk44=encoder_results['src_K'],
                            cur_world_T_cam_b44=encoder_results['cur_wtc'],
                            src_cam_T_world_bk44=encoder_results['src_ctw'],
                        ) * self.cfg.weight
        # if prediction.depth is not None:
        #     # print('target.shape:', batch['target']['depth'].shape)
        #     # print('rendered_depth.shape:', prediction.depth.shape)
        #     mask = batch['target']['depth'] != 0
        #     depth_loss += 0.1*nn.L1Loss()(prediction.depth*mask, batch['target']['depth']*mask)
        return depth_loss
