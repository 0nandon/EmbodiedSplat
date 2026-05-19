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
import os

from .losses import MSGradientLoss, MVDepthLoss, NormalsLoss, ScaleInvariantLoss


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
        depth_loss = 0

        if prediction.depth is not None:
            target = batch['target']['depth'].squeeze(2)
            mask = (target > 1e-3) & (target < 10)
            depth_loss += self.cfg.weight*nn.L1Loss()(torch.log(prediction.depth+1e-8)[mask], torch.log(target)[mask])

        return depth_loss
