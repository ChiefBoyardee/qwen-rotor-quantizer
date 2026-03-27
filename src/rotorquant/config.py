from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True)
class LayerQuantConfig:
    layer_type: str
    head_dim: int
    chunk_size: int = 3
    payload_bits: int = 3
    residual_bits: int = 1
    norm_dtype: str = "float16"
    align_bytes: int = 16

    @property
    def n_chunks(self) -> int:
        return ceil(self.head_dim / self.chunk_size)

    @property
    def rotor_params_per_chunk(self) -> int:
        # 8 components for Cl(3,0) multivector representation.
        return 8

    @property
    def rotor_param_count(self) -> int:
        return self.n_chunks * self.rotor_params_per_chunk


@dataclass(frozen=True)
class RotorQuantConfig:
    linear_attention: LayerQuantConfig
    full_attention: LayerQuantConfig

    def for_layer_type(self, layer_type: str) -> LayerQuantConfig:
        if layer_type == "linear_attention":
            return self.linear_attention
        if layer_type == "full_attention":
            return self.full_attention
        raise ValueError(f"Unsupported layer type: {layer_type}")


def get_qwen_default_configs() -> RotorQuantConfig:
    return RotorQuantConfig(
        linear_attention=LayerQuantConfig(layer_type="linear_attention", head_dim=128),
        full_attention=LayerQuantConfig(layer_type="full_attention", head_dim=256),
    )


def validate_head_dim(head_dim: int) -> None:
    if head_dim not in (128, 256):
        raise ValueError(
            f"Unsupported head_dim {head_dim}. Phase 2 supports only 128 and 256."
        )
