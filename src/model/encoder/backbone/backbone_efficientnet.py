from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor, nn

from ....dataset.types import BatchedViews
from .backbone import Backbone
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
import timm


@dataclass
class BackboneEfficientNetCfg:
    name: Literal["efficientnet"]
    model: Literal["efficientnet"]
    d_out: int
    resnet_name: Literal["resnet", "resunet"]
    use_depth: bool = False


class BackboneEfficientNet(Backbone[BackboneEfficientNetCfg]):
    def __init__(self, cfg: BackboneEfficientNetCfg, d_in: int) -> None:
        super().__init__(cfg)
        # assert d_in == 3
        self.d_in = d_in
        self.encoder = timm.create_model("tf_efficientnetv2_s_in21ft1k", 
                                            pretrained=True, 
                                            features_only=True)

    def forward(
        self,
        context: BatchedViews,
    ) -> Float[Tensor, "batch view d_out height width"]:
        # Compute features from the DINO-pretrained resnet50.
        b, v, _, h, w = context["image"].shape
        x = rearrange(context["image"], "b v c h w -> (b v) c h w")
        features = self.encoder(x)
        # print('features[0].shape:', features[0].shape)
        # print('features[1].shape:', features[1].shape)
        # print('features[2].shape:', features[2].shape)
        # print('features[3].shape:', features[3].shape)
        # print('features[4].shape:', features[4].shape)

        # Compute features from the DINO-pretrained ViT.
        

        return rearrange(features[0], '(b v) c h w -> b v c h w', b=b, v=v)

    @property
    def patch_size(self) -> int:
        return int("".join(filter(str.isdigit, self.cfg.model)))

    @property
    def d_out(self) -> int:
        return self.cfg.d_out
