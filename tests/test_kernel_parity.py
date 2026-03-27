import torch

from rotorquant.kernels.triton_kv_dequant import dequantize_kv
from rotorquant.kernels.triton_kv_quant import quantize_kv
from rotorquant.quantize import grade_aware_dequantize, grade_aware_quantize
from rotorquant.rotor_math import (
    chunk3,
    l2_norm_restore,
    l2_norm_separate,
    make_random_rotors,
    quaternion_rotate_chunks,
    unchunk3,
)


def _reference_roundtrip(kv: torch.Tensor, seed: int = 0) -> torch.Tensor:
    x_norm, norms = l2_norm_separate(kv)
    x_chunks, original_dim = chunk3(x_norm)
    rotors = make_random_rotors(x_chunks.shape[-2], device=kv.device, dtype=kv.dtype, seed=seed)
    x_rot = quaternion_rotate_chunks(x_chunks, rotors)
    qs = grade_aware_quantize(x_rot, payload_bits=3, residual_bits=1)
    x_rot_hat = grade_aware_dequantize(qs, use_residual=True)
    x_hat = unchunk3(x_rot_hat, original_dim)
    return l2_norm_restore(x_hat, norms)


def test_kernel_api_matches_reference():
    kv = torch.randn(2, 4, 16, 128, dtype=torch.float32)
    ref = _reference_roundtrip(kv, seed=42)
    qs = quantize_kv(kv, seed=42)
    got = dequantize_kv(qs)
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5)


def test_kernel_api_matches_reference_d256():
    kv = torch.randn(1, 4, 12, 256, dtype=torch.float32)
    ref = _reference_roundtrip(kv, seed=11)
    qs = quantize_kv(kv, seed=11)
    got = dequantize_kv(qs)
    assert ref.shape == got.shape
    assert torch.allclose(ref, got, atol=1e-5, rtol=1e-5)
