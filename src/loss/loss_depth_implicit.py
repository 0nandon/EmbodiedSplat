from dataclasses import dataclass

from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss
import torch.nn as nn
import torch


@dataclass
class LossDICfg:
    weight: float


@dataclass
class LossDICfgWrapper:
    ds: LossDICfg


class LossDI(Loss[LossDICfg, LossDICfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        encoder_results: dict,
        global_step: int,
    ) -> Float[Tensor, ""]:
        depth_loss = nn.L1Loss()(encoder_results[f'coarse_depth_pred'][encoder_results[f'depth_s0_mask']], \
                                    torch.log(encoder_results[f'depth_s0_raw'])[encoder_results[f'depth_s0_mask']]) * self.cfg.weight
        
        return depth_loss
