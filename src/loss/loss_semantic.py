from dataclasses import dataclass

from jaxtyping import Float
import torch
from torch import Tensor

from ..dataset.types import BatchedExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossSemCfg:
    weight: float
    mse: bool


@dataclass
class LossSemCfgWrapper:
    sem: LossSemCfg


class LossSem(Loss[LossSemCfg, LossSemCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        encoder_results: dict,
        global_step: int,
        dr_prediction: DecoderOutput | None = None,
    ) -> Float[Tensor, ""]:

        feat_dict = torch.cat(batch["target"]["feature"]["feat"], dim=0)
        feat_dict = torch.nn.functional.normalize(feat_dict, dim=-1, eps=1e-5)

        costmap_dict = (feat_dict @ prediction.feature.t()).t()
        
        idx = batch["target"]["feature"]["idx"]
        idx[idx == -1] = 0.

        costmap_sel = torch.gather(costmap_dict, index=idx[:, :5].long(), dim=-1)

        weight = batch["target"]["feature"]["weight"][:, :5]
        weight /= weight.sum(dim=-1, keepdim=True)
        
        cosine = (costmap_sel * weight).sum(dim=-1)
        loss = 1 - cosine

        return self.cfg.weight * loss.mean()