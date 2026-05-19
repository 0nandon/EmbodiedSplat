import torch, numpy as np
import torch.nn.functional as F
import torch.nn as nn

def list_2_tensor(lst, key, tensor: torch.Tensor):
    values = []

    for dct in lst:
        values.append(dct[key])
    if isinstance(values[0], (np.ndarray, list)):
        rst = np.stack(values, axis=0)
    elif isinstance(values[0], torch.Tensor):
        rst = torch.stack(values, dim=0)
    else:
        raise NotImplementedError
    
    return tensor.new_tensor(rst)


def linear_relu_ln(embed_dims, in_loops=1, out_loops=1, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return nn.Sequential(*layers)

def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]
    
    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]
    
    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]


def convert_to_buffer(module: nn.Module, persistent: bool = True):
    # Recurse over child modules.
    for name, child in list(module.named_children()):
        convert_to_buffer(child, persistent)

    # Also re-save buffers to change persistence.
    for name, parameter_or_buffer in (
        *module.named_parameters(recurse=False),
        *module.named_buffers(recurse=False),
    ):
        value = parameter_or_buffer.detach().clone()
        delattr(module, name)
        module.register_buffer(name, value, persistent=persistent)


SIGMOID_MAX = 9.21024
LOGIT_MAX = 0.9999

# SIGMOID_MAX = 3.0
# LOGIT_MAX = 0.99

def safe_sigmoid(tensor):
    tensor = torch.clamp(tensor, -SIGMOID_MAX, SIGMOID_MAX)
    return torch.sigmoid(tensor)

def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))


def spherical2cartesian(anchor, pc_range, phi_activation='loop'):
    if phi_activation == 'sigmoid':
        xyz = safe_sigmoid(anchor[..., :3])
    elif phi_activation == 'loop':
        xy = safe_sigmoid(anchor[..., :2])
        z = torch.remainder(anchor[..., 2:3], 1.0)
        xyz = torch.cat([xy, z], dim=-1)
    else:
        raise NotImplementedError
    rrr = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    theta = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    phi = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xxx = rrr * torch.sin(theta) * torch.cos(phi)
    yyy = rrr * torch.sin(theta) * torch.sin(phi)
    zzz = rrr * torch.cos(theta)
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)
    
    return xyz

def cartesian(anchor, pc_range):
    xyz = safe_sigmoid(anchor[..., :3])
    xxx = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    yyy = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    zzz = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    xyz = torch.stack([xxx, yyy, zzz], dim=-1)
    
    return xyz

def inv_cartesian(xyz, pc_range):

    xxx = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    yyy = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    zzz = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])

    anchor = torch.stack([xxx, yyy, zzz], dim=-1)
    return safe_inverse_sigmoid(anchor)


class SparseGaussian3DEncoder(nn.Module):
    def __init__(self, embed_dims: int):
        
        super().__init__()

        self.embed_dims = embed_dims
        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.xyz_fc = embedding_layer(3)
        self.scale_fc = embedding_layer(3)
        self.rot_fc = embedding_layer(4)
        self.opacity_fc = embedding_layer(1)
        self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, box_3d: torch.Tensor):
        xyz_feat = self.xyz_fc(box_3d[..., :3])
        scale_feat = self.scale_fc(box_3d[..., 3:6])
        rot_feat = self.rot_fc(box_3d[..., 6:10])
        opacity_feat = self.opacity_fc(box_3d[..., 10:11])
        output = xyz_feat + scale_feat + rot_feat + opacity_feat
        output = self.output_fc(output)
        return output