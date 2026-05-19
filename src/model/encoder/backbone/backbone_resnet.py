import functools
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torchvision.models import ResNet

from ....dataset.types import BatchedViews
from .backbone import Backbone
from .core.res_unet_plus import ResUnetPlusPlus

@dataclass
class BackboneResnetCfg:
    name: Literal["resnet", "resunet"]
    model: Literal[
        "resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "dino_resnet50"
    ]
    num_layers: int
    use_first_pool: bool
    d_out: int
    use_depth: bool = False


class BackboneResnet(Backbone[BackboneResnetCfg]):
    model: ResNet

    def __init__(self, cfg: BackboneResnetCfg, d_in: int) -> None:
        super().__init__(cfg)

        # print('d_in:', d_in)
        # assert d_in == 3
        self.cfg = cfg
        norm_layer = functools.partial(
            nn.InstanceNorm2d,
            affine=False,
            track_running_stats=False,
        )
        print('+++++++++++backbone cfg:', cfg)

        if cfg.model == "dino_resnet50" and cfg.name == 'resnet':
            self.model = torch.hub.load("facebookresearch/dino:main", "dino_resnet50")
        elif cfg.name == 'resnet':
            self.model = getattr(torchvision.models, cfg.model)(norm_layer=norm_layer, d_in=d_in)
        elif cfg.name == 'resunet':
            self.model = ResUnetPlusPlus(channel=d_in)
        # elif cfg.name == 'efficientnet':
        #     self.model = timm.create_model("tf_efficientnetv2_s_in21ft1k", 
        #                                     pretrained=True, 
        #                                     features_only=True,)
        else:
            raise NotImplementedError

        
        # Set up projections
        if cfg.name == 'resnet':
            self.projections = nn.ModuleDict({})
            for index in range(1, cfg.num_layers):
                key = f"layer{index}"
                block = getattr(self.model, key)
                conv_index = 1
                try:
                    while True:
                        d_layer_out = getattr(block[-1], f"conv{conv_index}").out_channels
                        conv_index += 1
                except AttributeError:
                    pass
                self.projections[key] = nn.Conv2d(d_layer_out, cfg.d_out, 1)

            # Add a projection for the first layer.
            self.projections["layer0"] = nn.Conv2d(
                self.model.conv1.out_channels, cfg.d_out, 1
            )

    def forward(
        self,
        context: BatchedViews,
    ) -> Float[Tensor, "batch view d_out height width"]:
        # Merge the batch dimensions.
        b, v, _, h, w = context["image"].shape
        x = rearrange(context["image"], "b v c h w -> (b v) c h w")
        # print('x.shape:', x.shape)

        if self.cfg.use_depth:
            depth = rearrange(context['depth'], "b v c h w -> (b v) c h w")
            # print('depth.shape:', depth.shape)
            # print('x.shape:', x.shape)
            x = torch.cat([x, depth], dim=1)
        
        if self.cfg.name == 'resnet':
            # Run the images through the resnet.
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            features = [self.projections["layer0"](x)]

            # Propagate the input through the resnet's layers.
            for index in range(1, self.cfg.num_layers):
                key = f"layer{index}"
                if index == 0 and self.cfg.use_first_pool:
                    x = self.model.maxpool(x)
                x = getattr(self.model, key)(x)
                features.append(self.projections[key](x))

            # Upscale the features.
            features = [
                F.interpolate(f, (h, w), mode="bilinear", align_corners=True)
                for f in features
            ]
            features = torch.stack(features).sum(dim=0)
        else:
            features = self.model(x)

        # print('post_x.shape:', features.shape)
        # Separate batch dimensions.
        return rearrange(features, "(b v) c h w -> b v c h w", b=b, v=v)

    @property
    def d_out(self) -> int:
        return self.cfg.d_out
