from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class LayerMeta:
    layer_idx: int
    layer_type: str
    head_dim: int


class VLLMCacheAdapter(Protocol):
    """
    Contract for a future vLLM integration point.
    Implementations are expected to intercept KV write/read paths.
    """

    def write_kv_prefill(self, meta: LayerMeta, kv: torch.Tensor) -> None:
        """Store FP16/BF16 KV during prefill."""

    def on_prefill_complete(self) -> None:
        """Bulk-quantize accumulated KV blocks before decode.

        Implementations may store packed payload/residual streams internally.
        """

    def read_kv_decode(self, meta: LayerMeta) -> torch.Tensor:
        """Read dequantized KV view for decode attention computation."""

    def expected_layer_type(self, layer_idx: int) -> str:
        """Return the expected layer type from model cadence metadata."""
