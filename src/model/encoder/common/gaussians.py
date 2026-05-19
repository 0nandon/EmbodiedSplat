import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor
import torch.nn.functional as F


# https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
def quaternion_to_matrix(
    quaternions: Float[Tensor, "*batch 4"],
    eps: float = 1e-8,
) -> Float[Tensor, "*batch 3 3"]:
    # Order changed to match scipy format!
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2 / ((quaternions * quaternions).sum(dim=-1) + eps)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return rearrange(o, "... (i j) -> ... i j", i=3, j=3)

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    # Ensure matrix is a proper rotation matrix of shape (*batch, 3, 3)
    m00, m01, m02 = matrix[..., 0, 0], matrix[..., 0, 1], matrix[..., 0, 2]
    m10, m11, m12 = matrix[..., 1, 0], matrix[..., 1, 1], matrix[..., 1, 2]
    m20, m21, m22 = matrix[..., 2, 0], matrix[..., 2, 1], matrix[..., 2, 2]

    # Compute quaternion components
    q0 = 0.5 * torch.sqrt(torch.clamp(m00 + m11 + m22 + 1, min=0))
    q1 = torch.sign(m21 - m12) * 0.5 * torch.sqrt(torch.clamp(m00 - m11 - m22 + 1, min=0))
    q2 = torch.sign(m02 - m20) * 0.5 * torch.sqrt(torch.clamp(-m00 + m11 - m22 + 1, min=0))
    q3 = torch.sign(m10 - m01) * 0.5 * torch.sqrt(torch.clamp(-m00 - m11 + m22 + 1, min=0))

    # Construct the quaternion tensor
    quaternion = torch.stack((q0, q1, q2, q3), dim=-1)

    # Normalize the quaternion to ensure unit quaternion
    quaternion = quaternion / torch.norm(quaternion, dim=-1, keepdim=True)

    return quaternion

def quaternion_multiply(q, r):
    # Extract components for clearer multiplication
    q0, q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r0, r1, r2, r3 = r[..., 0], r[..., 1], r[..., 2], r[..., 3]

    # Quaternion multiplication formula
    t0 = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
    t1 = r0 * q1 + r1 * q0 + r2 * q3 - r3 * q2
    t2 = r0 * q2 - r1 * q3 + r2 * q0 + r3 * q1
    t3 = r0 * q3 + r1 * q2 - r2 * q1 + r3 * q0

    return torch.stack((t1, t2, t3, t0), dim=-1)

def quaternion_normalize(q):
    norm = torch.sqrt(torch.sum(q ** 2, dim=-1, keepdim=True))
    return q / norm


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rotation_xyzw: Float[Tensor, "*#batch 4"],
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()
    rotation = quaternion_to_matrix(rotation_xyzw)
    return (
        rotation
        @ scale
        @ rearrange(scale, "... i j -> ... j i")
        @ rearrange(rotation, "... i j -> ... j i")
    )
