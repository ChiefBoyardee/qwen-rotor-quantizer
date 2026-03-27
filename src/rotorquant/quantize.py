from __future__ import annotations

from dataclasses import dataclass

import torch
from rotorquant.bitpack import pack_bits_aligned, unpack_bits


def _linear_quantize(x: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    levels = (1 << bits) - 1
    xmin = x.amin(dim=-1, keepdim=True)
    xmax = x.amax(dim=-1, keepdim=True)
    scale = (xmax - xmin).clamp_min(1e-8) / levels
    q = torch.round((x - xmin) / scale).clamp(0, levels).to(torch.int32)
    return q, xmin, scale


def _linear_dequantize(q: torch.Tensor, xmin: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.to(scale.dtype) * scale + xmin


def to_grade_groups(v_chunks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Map [..., n_chunks, 3] vector coords into:
    - scalar-like group: [..., n_chunks, 1] (kept as zeros in this POC)
    - bivector-like group: [..., n_chunks, 3] (stores x,y,z channels)
    """
    scalar = torch.zeros_like(v_chunks[..., :1])
    bivector = v_chunks
    return scalar, bivector


def from_grade_groups(_: torch.Tensor, bivector: torch.Tensor) -> torch.Tensor:
    return bivector


def grade_aware_quantize(
    v_chunks: torch.Tensor, payload_bits: int = 3, residual_bits: int = 1
) -> dict[str, torch.Tensor]:
    scalar, bivector = to_grade_groups(v_chunks)
    q_scalar, scalar_min, scalar_scale = _linear_quantize(scalar, payload_bits)
    q_biv, biv_min, biv_scale = _linear_quantize(bivector, payload_bits)

    deq_scalar = _linear_dequantize(q_scalar, scalar_min, scalar_scale)
    deq_biv = _linear_dequantize(q_biv, biv_min, biv_scale)
    recon = from_grade_groups(deq_scalar, deq_biv)
    residual = v_chunks - recon

    # 1-bit residual sign correction as QJL-like lightweight side channel.
    q_res_sign = (residual >= 0).to(torch.uint8)
    res_abs = residual.abs()
    q_res_mag, res_min, res_scale = _linear_quantize(res_abs, residual_bits)

    return {
        "q_scalar": q_scalar,
        "scalar_min": scalar_min,
        "scalar_scale": scalar_scale,
        "q_biv": q_biv,
        "biv_min": biv_min,
        "biv_scale": biv_scale,
        "q_res_sign": q_res_sign,
        "q_res_mag": q_res_mag,
        "res_min": res_min,
        "res_scale": res_scale,
    }


def grade_aware_dequantize(state: dict[str, torch.Tensor], use_residual: bool = True) -> torch.Tensor:
    deq_scalar = _linear_dequantize(
        state["q_scalar"], state["scalar_min"], state["scalar_scale"]
    )
    deq_biv = _linear_dequantize(state["q_biv"], state["biv_min"], state["biv_scale"])
    recon = from_grade_groups(deq_scalar, deq_biv)
    if not use_residual:
        return recon
    res_mag = _linear_dequantize(state["q_res_mag"], state["res_min"], state["res_scale"])
    res = torch.where(state["q_res_sign"] > 0, res_mag, -res_mag)
    return recon + res


@dataclass
class PackedKVState:
    """Packed layout; bump ``rotorquant.schema.ROTORQUANT_PACKED_SCHEMA_VERSION`` on breaking changes."""

    payload_bits: int
    residual_bits: int
    payload_packed: torch.Tensor
    payload_count: int
    payload_raw_nbytes: int
    residual_packed: torch.Tensor
    residual_count: int
    residual_raw_nbytes: int
    residual_mag_packed: torch.Tensor
    residual_mag_count: int
    residual_mag_raw_nbytes: int
    biv_min: torch.Tensor
    biv_scale: torch.Tensor
    res_min: torch.Tensor
    res_scale: torch.Tensor
    tensor_shape: tuple[int, ...]
    align_bytes: int = 16

    def to_dict(self) -> dict[str, torch.Tensor | int | tuple[int, ...]]:
        return {
            "payload_bits": self.payload_bits,
            "residual_bits": self.residual_bits,
            "payload_packed": self.payload_packed,
            "payload_count": self.payload_count,
            "payload_raw_nbytes": self.payload_raw_nbytes,
            "residual_packed": self.residual_packed,
            "residual_count": self.residual_count,
            "residual_raw_nbytes": self.residual_raw_nbytes,
            "residual_mag_packed": self.residual_mag_packed,
            "residual_mag_count": self.residual_mag_count,
            "residual_mag_raw_nbytes": self.residual_mag_raw_nbytes,
            "biv_min": self.biv_min,
            "biv_scale": self.biv_scale,
            "res_min": self.res_min,
            "res_scale": self.res_scale,
            "tensor_shape": self.tensor_shape,
            "align_bytes": self.align_bytes,
        }


def pack_quantized_state(
    qstate: dict[str, torch.Tensor],
    *,
    payload_bits: int = 3,
    residual_bits: int = 1,
    align_bytes: int = 16,
) -> PackedKVState:
    q_biv = qstate["q_biv"].to(torch.int32)
    q_res_sign = qstate["q_res_sign"].to(torch.int32)

    payload_packed, payload_count, payload_raw_nbytes = pack_bits_aligned(
        q_biv, payload_bits, align_bytes=align_bytes
    )
    residual_packed, residual_count, residual_raw_nbytes = pack_bits_aligned(
        q_res_sign, residual_bits, align_bytes=align_bytes
    )
    residual_mag_packed, residual_mag_count, residual_mag_raw_nbytes = pack_bits_aligned(
        qstate["q_res_mag"].to(torch.int32), residual_bits, align_bytes=align_bytes
    )

    return PackedKVState(
        payload_bits=payload_bits,
        residual_bits=residual_bits,
        payload_packed=payload_packed,
        payload_count=payload_count,
        payload_raw_nbytes=payload_raw_nbytes,
        residual_packed=residual_packed,
        residual_count=residual_count,
        residual_raw_nbytes=residual_raw_nbytes,
        residual_mag_packed=residual_mag_packed,
        residual_mag_count=residual_mag_count,
        residual_mag_raw_nbytes=residual_mag_raw_nbytes,
        biv_min=qstate["biv_min"],
        biv_scale=qstate["biv_scale"],
        res_min=qstate["res_min"],
        res_scale=qstate["res_scale"],
        tensor_shape=tuple(qstate["q_biv"].shape),
        align_bytes=align_bytes,
    )


def unpack_quantized_state(packed_state: PackedKVState) -> dict[str, torch.Tensor]:
    q_biv = unpack_bits(
        packed_state.payload_packed[: packed_state.payload_raw_nbytes],
        bits=packed_state.payload_bits,
        original_count=packed_state.payload_count,
    ).reshape(packed_state.tensor_shape)
    q_res_sign = unpack_bits(
        packed_state.residual_packed[: packed_state.residual_raw_nbytes],
        bits=packed_state.residual_bits,
        original_count=packed_state.residual_count,
    ).reshape(packed_state.tensor_shape).to(torch.uint8)
    q_res_mag = unpack_bits(
        packed_state.residual_mag_packed[: packed_state.residual_mag_raw_nbytes],
        bits=packed_state.residual_bits,
        original_count=packed_state.residual_mag_count,
    ).reshape(packed_state.tensor_shape).to(torch.int32)
    scalar_shape = (*packed_state.tensor_shape[:-1], 1)
    zeros_scalar = torch.zeros(
        scalar_shape, dtype=packed_state.biv_scale.dtype, device=packed_state.biv_scale.device
    )
    return {
        "q_scalar": torch.zeros(scalar_shape, dtype=torch.int32, device=q_biv.device),
        "scalar_min": zeros_scalar,
        "scalar_scale": torch.ones_like(zeros_scalar),
        "q_biv": q_biv.to(torch.int32),
        "biv_min": packed_state.biv_min,
        "biv_scale": packed_state.biv_scale,
        "q_res_sign": q_res_sign,
        "q_res_mag": q_res_mag,
        "res_min": packed_state.res_min,
        "res_scale": packed_state.res_scale,
    }

