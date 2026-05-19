from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch.nn as nn
import torch

from PIL import Image, ImageFont, ImageDraw
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import mmcv



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
class LossDRCfg:
    weight: float


@dataclass
class LossDRCfgWrapper:
    dr: LossDRCfg


class LossDR(Loss[LossDRCfg, LossDRCfgWrapper]):
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
        depth_loss = 0.
        # for s in range(4):
        #     depth_loss += nn.L1Loss()(encoder_results[f'depth_s{s}']*encoder_results[f'depth_s{s}_mask'], \
        #                          encoder_results[f'depth_s{s}_raw']*encoder_results[f'depth_s{s}_mask']) * self.cfg.weight
        if dr_prediction is not None:
            # print('target.shape:', batch['target']['depth'].shape)
            # print('rendered_depth.shape:', prediction.depth.shape)
            # target = batch['target']['depth'].squeeze(2)
            # mask = target != 0
            # print('dr_prediction.depth.shape:', dr_prediction.depth.shape)
            # print('encoder_results[f\'depth_num0_s-1_b1hw\'].shape:', encoder_results[f'depth_num0_s-1_b1hw'].shape)
            pred = encoder_results[f'depth_num0_s-1_b1hw'].squeeze(1)
            render = dr_prediction.depth.squeeze(0)
            mask = (pred < 10) & (render > 0.5)

            convert_array_to_pil(render[0].detach().cpu().numpy()).save('render.png')
            convert_array_to_pil(pred[0].detach().cpu().numpy()).save('pred.png')
            # mask = torch.ones_like(pred, dtype=torch.bool)
            # mask = encoder_results[f'depth_num0_s-1_b1hw'] < 10
            # print('encoder_results[f\'depth_num0_s-1_b1hw\'].shape:', encoder_results[f'depth_num0_s-1_b1hw'].shape,
            #       'dr_prediction.depth.shape:', dr_prediction.depth.shape)
            # print(f'pred.shape: {pred.shape}, render.shape: {render.shape}')
            # print(f'pred_range: {pred[mask].min()}, {pred[mask].max()}')
            # print(f'render_range: {render[mask].min()}, {render[mask].max()}')
            # print(f'loss_raw:', nn.L1Loss()(torch.log(dr_prediction.depth[mask_raw]+1e-8), 
            #                                           torch.log(encoder_results[f'depth_num0_s-1_b1hw'][mask_raw]+1e-8)))
            # print(f'loss_post:', nn.L1Loss()(torch.log(render[mask]+1e-8), 
            #                                           torch.log(pred[mask]+1e-8)))
            depth_loss += self.cfg.weight*nn.L1Loss()(torch.log(render[mask]+1e-8), 
                                                      torch.log(pred[mask]+1e-8))
        return depth_loss
