from __future__ import annotations

import math

import torch


def make_random_rotors(
    n_chunks: int, *, device: torch.device, dtype: torch.dtype, seed: int = 0
) -> torch.Tensor:
    """Create unit quaternions [n_chunks, 4] for 3D chunk rotations."""
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    q = torch.randn((n_chunks, 4), generator=g, dtype=torch.float32)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return q.to(device=device, dtype=dtype)


def chunk3(x: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Pad last dim to multiple of 3 and view as chunks."""
    if x.ndim < 1:
        raise ValueError("Expected tensor with at least one dimension")
    d = x.shape[-1]
    n_chunks = math.ceil(d / 3)
    padded = n_chunks * 3
    pad = padded - d
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    new_shape = (*x.shape[:-1], n_chunks, 3)
    return x.view(new_shape), d


def unchunk3(x_chunks: torch.Tensor, original_dim: int) -> torch.Tensor:
    x = x_chunks.reshape(*x_chunks.shape[:-2], x_chunks.shape[-2] * 3)
    return x[..., :original_dim]


def quaternion_rotate_chunks(v_chunks: torch.Tensor, rotors: torch.Tensor) -> torch.Tensor:
    """
    Rotate 3D vectors by unit quaternion per chunk.
    v_chunks: [..., n_chunks, 3]
    rotors: [n_chunks, 4] in (w, x, y, z)
    """
    w = rotors[:, 0]
    qv = rotors[:, 1:]

    v = v_chunks
    # Broadcast chunk-wise rotor across leading dims.
    qv_expand = qv.view(*([1] * (v.ndim - 2)), qv.shape[0], 3)
    w_expand = w.view(*([1] * (v.ndim - 2)), w.shape[0], 1)

    t = 2.0 * torch.cross(qv_expand, v, dim=-1)
    return v + w_expand * t + torch.cross(qv_expand, t, dim=-1)


def l2_norm_separate(x: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized vectors and stored L2 norms for side-channel storage."""
    n = x.norm(dim=-1, keepdim=True).clamp_min(eps)
    return x / n, n.squeeze(-1)


def l2_norm_restore(x: torch.Tensor, norms: torch.Tensor) -> torch.Tensor:
    return x * norms.unsqueeze(-1)

