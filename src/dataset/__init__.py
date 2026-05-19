from torch.utils.data import Dataset

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .dataset_scannet import DatasetScannet, DatasetScannetCfg
from .dataset_scannetpp import DatasetScannetPP, DatasetScannetPPCfg
from .types import Stage
from .view_sampler import get_view_sampler

DATASETS: dict[str, Dataset] = {
    "re10k": DatasetRE10k,
    "scannet": DatasetScannet,
    "scannetpp": DatasetScannetPP,
}

DatasetCfg = DatasetRE10kCfg

def get_dataset(
    cfg: DatasetCfg,
    stage: Stage,
    step_tracker: StepTracker | None,
    dense: bool = False,
    test_bev: bool = False,
    colmap: bool = False,
) -> Dataset:
    view_sampler = get_view_sampler(
        cfg.view_sampler,
        stage,
        cfg.overfit_to_scene is not None,
        cfg.cameras_are_circular,
        step_tracker,
    )
    return DATASETS[cfg.name](cfg, stage, view_sampler, dense=dense, test_bev=test_bev, colmap=colmap)
