from __future__ import annotations

from dataclasses import dataclass

from rotorquant.runtime.cache_manager import RotorKVCacheManager


@dataclass
class DecodePhaseSwitch:
    cache: RotorKVCacheManager
    in_decode: bool = False

    def on_prefill_complete(self) -> None:
        if self.in_decode:
            return
        self.cache.bulk_quantize()
        self.in_decode = True

    def ensure_decode_ready(self) -> None:
        if not self.in_decode:
            raise RuntimeError(
                "Decode phase not active. Call on_prefill_complete() before decode."
            )
