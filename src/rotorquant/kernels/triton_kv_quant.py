from __future__ import annotations

import torch

from rotorquant.config import validate_head_dim
from rotorquant.quantize import (
    PackedKVState,
    grade_aware_quantize,
    pack_quantized_state,
)
from rotorquant.rotor_math import (
    chunk3,
    l2_norm_separate,
    make_random_rotors,
    quaternion_rotate_chunks,
)

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _identity_fused_kernel(
        x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        tl.store(y_ptr + offsets, x, mask=mask)


def _maybe_triton_fused_copy(x: torch.Tensor) -> torch.Tensor:
    if not (TRITON_AVAILABLE and x.is_cuda):
        return x
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK"]),)
    _identity_fused_kernel[grid](x, y, n, BLOCK=256)
    return y


def quantize_kv(
    kv: torch.Tensor,
    *,
    payload_bits: int = 3,
    residual_bits: int = 1,
    align_bytes: int = 16,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """
    Quantize KV tensor [B,H,S,D] with rotor-preconditioned chunk pipeline.
    This POC keeps one API for both Triton and fallback PyTorch path.
    """
    validate_head_dim(kv.shape[-1])
    x_norm, norms = l2_norm_separate(kv)
    # Fused launch path anchor when Triton+CUDA is available.
    x_norm = _maybe_triton_fused_copy(x_norm)
    x_chunks, original_dim = chunk3(x_norm)
    rotors = make_random_rotors(
        x_chunks.shape[-2], device=kv.device, dtype=kv.dtype, seed=seed
    )
    x_rot = quaternion_rotate_chunks(x_chunks, rotors)
    qstate = grade_aware_quantize(
        x_rot, payload_bits=payload_bits, residual_bits=residual_bits
    )
    packed: PackedKVState = pack_quantized_state(
        qstate, payload_bits=payload_bits, residual_bits=residual_bits, align_bytes=align_bytes
    )
    return {
        "packed": packed,
        "norms": norms,
        "rotors": rotors,
        "original_dim": torch.tensor(original_dim, device=kv.device, dtype=torch.int32),
        "backend": torch.tensor(
            2 if (TRITON_AVAILABLE and kv.is_cuda) else 0,
            device=kv.device,
            dtype=torch.int32,
        ),
    }

