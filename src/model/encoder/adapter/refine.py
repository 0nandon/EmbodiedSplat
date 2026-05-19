from typing import List, Optional, Tuple, NamedTuple
import torch.nn as nn
import torch
from torch import Tensor
from einops import einsum, rearrange
from jaxtyping import Float

from .utils import linear_relu_ln, safe_sigmoid
from src.model.types import Gaussian

class Scale(nn.Module):
    def __init__(self, scale):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float32))

    def forward(self, x):
        # print('self.scale:', self.scale)
        return x * self.scale

class SparseGaussian3DRefinementModule(nn.Module):
    def __init__(
        self,
        embed_dims=128,
        sh_degree=4,
        scale_range=None,
        refine_shs=False,
        phi_activation='sigmoid',
        include_shs=True,
        return_shs=True,
        include_opa=True,
        xyz_coordinate='xyz',
        shift=6.0,
        opa_shift=0.05
    ):
        super(SparseGaussian3DRefinementModule, self).__init__()
        self.embed_dims = embed_dims
        self.sh_degree = sh_degree
        self.return_shs = return_shs
        
        if include_shs and return_shs:
            self.output_dim = 10 + int(include_opa) + 3 * ((sh_degree + 1) ** 2)
        else:
            self.output_dim = 10 + int(include_opa)
        
        self.shs_start = 10 + int(include_opa)
        self.include_opa = include_opa

        self.scale_range = scale_range
        self.phi_activation = phi_activation
        self.shift = shift
        self.opa_shift = opa_shift
        
        if refine_shs:
            assert include_shs and return_shs
            self.refine_state = list(range(self.output_dim))
        else:
            self.refine_state = list(range(10 + int(include_opa)))

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim),
        )


    def get_scale_multiplier(
        self,
        intrinsics: Float[Tensor, "*#batch 3 3"],
        pixel_size: Float[Tensor, "*#batch 2"],
        img_size: Tuple[int, int],
        multiplier: float = 0.1,
    ) -> Float[Tensor, " *batch"]:

        # normalize intrinsic
        h, w = img_size
        normal_intrinsics = intrinsics.clone()
        normal_intrinsics[0, 0] /= w
        normal_intrinsics[0, 2] /= w
        normal_intrinsics[1, 1] /= h
        normal_intrinsics[1, 2] /= h

        xy_multipliers = multiplier * einsum(
            normal_intrinsics[..., :2, :2].inverse(),
            pixel_size,
            "... i j, j -> ... i",
        )
        return xy_multipliers.sum(dim=-1)

    def get_depths(self, pts, poses):

        b, nbr, N, _ = pts.shape
        ones = torch.ones(b, nbr, N, 1, device=pts.device)
        pts_w = torch.cat([pts, ones], dim=-1) # (b, nbr, N, 4)

        w2c = torch.inverse(poses)
        w2c = w2c.unsqueeze(0)  # [b, nbr, 4, 4]
        pts_c = torch.matmul(w2c, pts_w.transpose(-1, -2)).transpose(-1, -2) # (b, nbr, N, 4)
        depths = pts_c[..., 2:3]  # [b, nbr, N, 1]

        return depths
     

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        gaussians: Gaussian,
        anchor_idxs: List
    ):
        output = self.layers(instance_feature + anchor_embed)

        # print('output.shape:', output.shape)
        # print(f'output.isnan: {torch.isnan(output).any()}, output.isinf: {torch.isinf(output).any()}')
        # print('refiner.parameters:', list(self.parameters()))

        
    

        if len(self.refine_state) > 0:

            if torch.isnan(output).any() or torch.isinf(output).any():
                raise ValueError("NaN or Inf found in output")
            if torch.isnan(anchor).any() or torch.isinf(anchor).any():
                raise ValueError("NaN or Inf found in output")

            # print('passed refiner')

            refined_part_output = output[..., self.refine_state] + anchor[..., self.refine_state]
            output = torch.cat([refined_part_output, output[..., len(self.refine_state):]], dim=-1)
        
        rot = torch.nn.functional.normalize(output[..., 6:10], dim=-1, eps=1e-8)
        output = torch.cat([output[..., :6], rot, output[..., 10:]], dim=-1)

        if self.phi_activation == 'sigmoid':
            xyz = safe_sigmoid(output[..., :3])
        elif self.phi_activation == 'loop':
            xy = safe_sigmoid(output[..., :2])
            z = torch.remainder(output[..., 2:3], 1.0)
            xyz = torch.cat([xy, z], dim=-1)
        else:
            raise NotImplementedError

        means_modify = gaussians.means[anchor_idxs]
        d = xyz.norm(-1, keepdim=True)
        xyz = xyz / d.clip(min=1e-8)
        offsets = xyz * (torch.exp(d - self.shift) - torch.exp(torch.zeros_like(d) - self.shift))
        means_modify = means_modify + offsets.squeeze()
        means = gaussians.means
        # means[anchor_idxs] = means_modify
        
        opacities_modify = gaussians.opacities[anchor_idxs]
        opacities_offsets = safe_sigmoid(output[..., 10: (10 + int(self.include_opa))]).squeeze()
        opacities_modify = opacities_modify + self.opa_shift * opacities_offsets
        opacities = gaussians.opacities
        # opacities[anchor_idxs] = opacities_modify
        
        gaussian = Gaussian(
            means=means,
            scales=gaussians.scales,
            rotations=gaussians.rotations,
            opacities=opacities,
            harmonics=gaussians.harmonics,
            covariances=gaussians.covariances
        )

        # if torch.isnan(gaussian).any() or torch.isinf(gaussian).any():
        #     raise ValueError("NaN or Inf found in final output")


        return gaussian
    
    @property
    def d_sh(self) -> int:
        return (self.sh_degree + 1) ** 2