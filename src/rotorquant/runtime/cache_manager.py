from __future__ import annotations

from dataclasses import dataclass

import torch

from rotorquant.config import RotorQuantConfig, get_qwen_default_configs
from rotorquant.kernels import dequantize_kv, quantize_kv
from rotorquant.layout import assert_kv_contract


@dataclass
class LayerCacheEntry:
    layer_type: str
    fp16_kv: torch.Tensor | None = None
    quant_state: dict[str, torch.Tensor] | None = None


class RotorKVCacheManager:
    def __init__(self, config: RotorQuantConfig | None = None) -> None:
        self.config = config or get_qwen_default_configs()
        self._layers: dict[int, LayerCacheEntry] = {}

    def write_prefill(self, layer_idx: int, layer_type: str, kv: torch.Tensor) -> None:
        cfg = self.config.for_layer_type(layer_type)
        assert_kv_contract(kv, expected_head_dim=cfg.head_dim)
        self._layers[layer_idx] = LayerCacheEntry(
            layer_type=layer_type, fp16_kv=kv.to(torch.float16).contiguous(), quant_state=None
        )

    def bulk_quantize(self) -> None:
        for idx, entry in self._layers.items():
            if entry.fp16_kv is None:
                continue
            if entry.fp16_kv.device.type not in ("cpu", "cuda"):
                raise RuntimeError(
                    f"Unsupported device {entry.fp16_kv.device.type} for layer {idx}"
                )
            qs = quantize_kv(entry.fp16_kv, payload_bits=3, residual_bits=1, seed=idx)
            entry.quant_state = qs
            entry.fp16_kv = None

    def read_decode(self, layer_idx: int) -> torch.Tensor:
        entry = self._layers[layer_idx]
        if entry.quant_state is None:
            if entry.fp16_kv is None:
                raise RuntimeError(f"Layer {layer_idx} cache is empty")
            return entry.fp16_kv
        return dequantize_kv(entry.quant_state, use_residual=True)

