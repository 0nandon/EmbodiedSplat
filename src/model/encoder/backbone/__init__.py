from typing import Any

from .backbone import Backbone
from .backbone_dino import BackboneDino, BackboneDinoCfg
from .backbone_resnet import BackboneResnet, BackboneResnetCfg
from .backbone_efficientnet import BackboneEfficientNet, BackboneEfficientNetCfg
from .backbone_cost_volume import BackboneMultiview, BackboneMVSCfg

BACKBONES: dict[str, Backbone[Any]] = {
    "resnet": BackboneResnet,
    "dino": BackboneDino,
    "dino_resunet": BackboneDino,
    "efficientnet": BackboneEfficientNet,
    "cost_volume": BackboneMultiview,
}

BackboneCfg = BackboneResnetCfg | BackboneDinoCfg | BackboneEfficientNetCfg | BackboneMVSCfg


def get_backbone(cfg: BackboneCfg, d_in: int) -> Backbone[Any]:
    return BACKBONES[cfg.name](cfg, d_in)
