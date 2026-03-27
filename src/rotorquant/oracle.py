from __future__ import annotations

import json
from pathlib import Path

import torch

from rotorquant.kernels import dequantize_kv, quantize_kv


def generate_fixture(
    *,
    seed: int,
    shape: tuple[int, int, int, int],
    device: torch.device,
    out_file: str,
) -> dict[str, float]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    kv = torch.randn(shape, generator=g, dtype=torch.float32, device=device)
    qs = quantize_kv(kv, seed=seed)
    recon = dequantize_kv(qs).to(kv.dtype)
    err = (kv - recon).abs()
    metrics = {
        "seed": float(seed),
        "mean_abs_error": float(err.mean().item()),
        "p95_abs_error": float(torch.quantile(err.reshape(-1), 0.95).item()),
        "max_abs_error": float(err.max().item()),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                kv.reshape(1, -1), recon.reshape(1, -1)
            ).item()
        ),
    }
    p = Path(out_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def replay_fixture(out_file: str) -> dict[str, float]:
    return json.loads(Path(out_file).read_text(encoding="utf-8"))
