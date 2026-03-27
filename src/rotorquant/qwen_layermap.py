from __future__ import annotations

from typing import Sequence


QWEN_HYBRID_PATTERN = (
    "linear_attention",
    "linear_attention",
    "linear_attention",
    "full_attention",
)


def parse_layer_types(layer_types: Sequence[str], n_layers: int) -> list[str]:
    if not layer_types:
        raise ValueError("layer_types must not be empty")
    if len(layer_types) == n_layers:
        return list(layer_types)
    if n_layers % len(layer_types) != 0:
        raise ValueError(
            "n_layers must be divisible by layer_types pattern length in this POC"
        )
    reps = n_layers // len(layer_types)
    return list(layer_types) * reps


def infer_layer_type(layer_idx: int, layer_types: Sequence[str]) -> str:
    if layer_idx < 0:
        raise ValueError("layer_idx must be >= 0")
    if not layer_types:
        raise ValueError("layer_types must not be empty")
    return layer_types[layer_idx % len(layer_types)]
