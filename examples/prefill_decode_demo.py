from __future__ import annotations

import torch

from rotorquant.qwen_layermap import QWEN_HYBRID_PATTERN, infer_layer_type
from rotorquant.runtime.cache_manager import RotorKVCacheManager
from rotorquant.runtime.phase_switch import DecodePhaseSwitch


def main() -> None:
    cache = RotorKVCacheManager()
    phase = DecodePhaseSwitch(cache=cache)

    # Minimal 4-layer demo mirrors one Qwen group cadence.
    for layer_idx in range(4):
        layer_type = infer_layer_type(layer_idx, QWEN_HYBRID_PATTERN)
        d = 128 if layer_type == "linear_attention" else 256
        kv = torch.randn(1, 4, 32, d, dtype=torch.float16)
        cache.write_prefill(layer_idx, layer_type, kv)

    phase.on_prefill_complete()

    for layer_idx in range(4):
        restored = cache.read_decode(layer_idx)
        print(layer_idx, tuple(restored.shape), restored.dtype)


if __name__ == "__main__":
    main()
