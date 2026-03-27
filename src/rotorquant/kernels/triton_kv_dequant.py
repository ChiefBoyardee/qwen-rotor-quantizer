from __future__ import annotations

import torch

from rotorquant.quantize import grade_aware_dequantize, unpack_quantized_state
from rotorquant.rotor_math import l2_norm_restore, unchunk3


def dequantize_kv(
    qstate: dict[str, torch.Tensor],
    *,
    use_residual: bool = True,
) -> torch.Tensor:
    if "packed" in qstate:
        state = unpack_quantized_state(qstate["packed"])
    else:
        state = qstate
    x_rot = grade_aware_dequantize(state, use_residual=use_residual)
    original_dim = int(qstate["original_dim"].item())
    x = unchunk3(x_rot, original_dim=original_dim)
    return l2_norm_restore(x, qstate["norms"])

