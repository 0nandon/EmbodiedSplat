# Copyright (c) OpenMMLab. All rights reserved.
# Adapted from https://github.com/SamsungLabs/fcaf3d/blob/master/mmdet3d/models/detectors/single_stage_sparse.py # noqa
try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    pass

import torch
import torch.nn as nn

class MultilevelMemory(nn.Module):
    def __init__(self, in_channels=[32, 64, 128, 256]):
        super().__init__()
        
        self.proj = nn.ModuleList([
            ME.MinkowskiLinear(in_channels[0] + 64, in_channels[0]),
            ME.MinkowskiLinear(in_channels[1] + 64, in_channels[1]),
            ME.MinkowskiLinear(in_channels[2] + 64, in_channels[2]),
            ME.MinkowskiLinear(in_channels[3] + 64, in_channels[3])
        ])
    
    def forward(self, xs, memory):
        out = []
        for i, x in enumerate(xs):
            x_temp = ME.SparseTensor(
                coordinate_map_key=x.coordinate_map_key, 
                features=memory.features_at_coordinates(x.coordinates.float()), 
                tensor_stride=x.tensor_stride, 
                coordinate_manager=x.coordinate_manager
            )
            out.append(self.proj[i](ME.cat(x, x_temp)))
        return out