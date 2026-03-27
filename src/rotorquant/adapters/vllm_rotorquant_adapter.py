from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from rotorquant.adapters.vllm_adapter_contract import LayerMeta, VLLMCacheAdapter
from rotorquant.qwen_layermap import infer_layer_type, parse_layer_types
from rotorquant.runtime.cache_manager import RotorKVCacheManager


@dataclass
class VLLMPhaseState:
    prefill_complete: bool = False


class RotorQuantVLLMAdapter(VLLMCacheAdapter):
    """
    vLLM-facing adapter shim for this repo.
    It mirrors the lifecycle needed by backend interception hooks.
    """

    def __init__(self, *, layer_types: Sequence[str], n_layers: int) -> None:
        self.layer_types = parse_layer_types(layer_types, n_layers)
        self.cache = RotorKVCacheManager()
        self.phase = VLLMPhaseState(prefill_complete=False)

    def expected_layer_type(self, layer_idx: int) -> str:
        return infer_layer_type(layer_idx, self.layer_types)

    def write_kv_prefill(self, meta: LayerMeta, kv: torch.Tensor) -> None:
        expected = self.expected_layer_type(meta.layer_idx)
        if meta.layer_type != expected:
            raise ValueError(
                f"Layer {meta.layer_idx} type mismatch: expected {expected}, got {meta.layer_type}"
            )
        expected_dim = 128 if expected == "linear_attention" else 256
        if meta.head_dim != expected_dim:
            raise ValueError(
                f"Layer {meta.layer_idx} head_dim mismatch: expected {expected_dim}, got {meta.head_dim}"
            )
        self.cache.write_prefill(meta.layer_idx, meta.layer_type, kv)

    def on_prefill_complete(self) -> None:
        if self.phase.prefill_complete:
            return
        self.cache.bulk_quantize()
        self.phase.prefill_complete = True

    def read_kv_decode(self, meta: LayerMeta) -> torch.Tensor:
        if not self.phase.prefill_complete:
            raise RuntimeError("decode read attempted before prefill completion")
        return self.cache.read_decode(meta.layer_idx)
