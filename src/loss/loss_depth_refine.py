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
    ) -> Float[Tensor, ""]:
        depth_loss = 0
        for s in range(4):
            depth_loss += nn.L1Loss()(encoder_results[f'depth_s{s}']*encoder_results[f'depth_s{s}_mask'], \
                                 encoder_results[f'depth_s{s}_raw']*encoder_results[f'depth_s{s}_mask']) * self.cfg.weight
        return depth_loss
