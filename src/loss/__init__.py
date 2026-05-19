from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper
from .loss_depth_sup import LossDS, LossDSCfgWrapper
#from .loss_depth_reg import LossDR, LossDRCfgWrapper
from .loss_depth_render import LossDR, LossDRCfgWrapper
from .loss_depth_implicit import LossDI, LossDICfgWrapper
from .loss_semantic import LossSem, LossSemCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
    LossDRCfgWrapper: LossDR,
    LossDSCfgWrapper: LossDS,
    LossDICfgWrapper: LossDI,
    LossSemCfgWrapper: LossSem
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper | LossDRCfgWrapper | LossDSCfgWrapper | LossDICfgWrapper | LossSemCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    return [LOSSES[type(cfg)](cfg) for cfg in cfgs]
