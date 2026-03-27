import torch

from rotorquant.qwen_layermap import QWEN_HYBRID_PATTERN, infer_layer_type
from rotorquant.runtime.cache_manager import RotorKVCacheManager
from rotorquant.runtime.phase_switch import DecodePhaseSwitch


def test_mixed_layer_prefill_decode_roundtrip_shapes():
    cache = RotorKVCacheManager()
    phase = DecodePhaseSwitch(cache=cache)

    for layer_idx in range(4):
        layer_type = infer_layer_type(layer_idx, QWEN_HYBRID_PATTERN)
        d = 128 if layer_type == "linear_attention" else 256
        kv = torch.randn(1, 2, 8, d, dtype=torch.float16)
        cache.write_prefill(layer_idx, layer_type, kv)

    phase.on_prefill_complete()
    phase.ensure_decode_ready()

    expected_dims = [128, 128, 128, 256]
    for layer_idx, d in enumerate(expected_dims):
        out = cache.read_decode(layer_idx)
        assert tuple(out.shape) == (1, 2, 8, d)
