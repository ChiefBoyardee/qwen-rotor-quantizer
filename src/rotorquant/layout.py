from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVLayout:
    # Canonical layout for this POC:
    # [batch, n_heads, seq, head_dim]
    batch: int
    n_heads: int
    seq_len: int
    head_dim: int
    dtype: torch.dtype
    contiguous: bool


def inspect_kv_layout(kv: torch.Tensor) -> KVLayout:
    if kv.ndim != 4:
        raise ValueError(f"Expected KV tensor rank 4, got {kv.ndim}")
    b, h, s, d = kv.shape
    return KVLayout(
        batch=b,
        n_heads=h,
        seq_len=s,
        head_dim=d,
        dtype=kv.dtype,
        contiguous=kv.is_contiguous(),
    )


def assert_kv_contract(kv: torch.Tensor, expected_head_dim: int) -> None:
    layout = inspect_kv_layout(kv)
    if layout.head_dim != expected_head_dim:
        raise ValueError(
            f"KV head_dim mismatch: expected {expected_head_dim}, got {layout.head_dim}"
        )
    if not layout.contiguous:
        raise ValueError("KV tensor must be contiguous for kernel path")
