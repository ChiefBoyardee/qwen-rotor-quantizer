import torch

from rotorquant.adapters.vllm_adapter_contract import LayerMeta
from rotorquant.adapters.vllm_rotorquant_adapter import RotorQuantVLLMAdapter
from rotorquant.qwen_layermap import QWEN_HYBRID_PATTERN


def test_vllm_adapter_prefill_decode_flow():
    adapter = RotorQuantVLLMAdapter(layer_types=QWEN_HYBRID_PATTERN, n_layers=4)
    dims = [128, 128, 128, 256]
    for idx, d in enumerate(dims):
        layer_type = adapter.expected_layer_type(idx)
        kv = torch.randn(1, 2, 8, d, dtype=torch.float16)
        adapter.write_kv_prefill(
            LayerMeta(layer_idx=idx, layer_type=layer_type, head_dim=d), kv
        )
    adapter.on_prefill_complete()
    for idx, d in enumerate(dims):
        layer_type = adapter.expected_layer_type(idx)
        out = adapter.read_kv_decode(
            LayerMeta(layer_idx=idx, layer_type=layer_type, head_dim=d)
        )
        assert tuple(out.shape) == (1, 2, 8, d)
