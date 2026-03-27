from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rotorquant.kernels import TRITON_AVAILABLE, dequantize_kv, quantize_kv


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def run_once(device: torch.device, head_dim: int, seq: int) -> dict[str, float]:
    kv = torch.randn(1, 8, seq, head_dim, device=device, dtype=torch.float16)

    t0 = time.perf_counter()
    qs = quantize_kv(kv, seed=7)
    t1 = time.perf_counter()
    recon = dequantize_kv(qs)
    t2 = time.perf_counter()

    fp16_bytes = kv.numel() * 2
    packed = qs["packed"]
    payload_bytes = float(packed.payload_raw_nbytes)
    residual_bytes = float(packed.residual_raw_nbytes + packed.residual_mag_raw_nbytes)
    norm_bytes = qs["norms"].numel() * 2
    quant_bytes = payload_bytes + residual_bytes + norm_bytes

    cos = cosine_similarity(kv, recon.to(kv.dtype))
    return {
        "quant_ms": (t1 - t0) * 1000.0,
        "dequant_ms": (t2 - t1) * 1000.0,
        "cosine": cos,
        "memory_reduction_x": fp16_bytes / max(quant_bytes, 1.0),
        "backend": "triton_fused" if int(qs["backend"].item()) == 2 else "fallback",
    }


def run_fallback_once(device: torch.device, head_dim: int, seq: int) -> dict[str, float]:
    # Forced fallback baseline by executing on CPU tensors.
    kv = torch.randn(1, 8, seq, head_dim, device=torch.device("cpu"), dtype=torch.float16)
    t0 = time.perf_counter()
    qs = quantize_kv(kv, seed=7)
    t1 = time.perf_counter()
    _ = dequantize_kv(qs)
    t2 = time.perf_counter()
    return {
        "quant_ms": (t1 - t0) * 1000.0,
        "dequant_ms": (t2 - t1) * 1000.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, choices=[128, 256], default=128)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-cosine", type=float, default=0.95)
    ap.add_argument("--min-memory-x", type=float, default=2.5)
    args = ap.parse_args()

    metrics = run_once(torch.device(args.device), args.head_dim, args.seq)
    baseline = run_fallback_once(torch.device(args.device), args.head_dim, min(args.seq, 1024))
    print(metrics)
    print({"fallback_baseline_ms": baseline})
    print({"triton_available": TRITON_AVAILABLE})

    if metrics["cosine"] < args.min_cosine:
        raise SystemExit(f"FAIL cosine gate: {metrics['cosine']:.4f} < {args.min_cosine:.4f}")
    if metrics["memory_reduction_x"] < args.min_memory_x:
        raise SystemExit(
            f"FAIL memory gate: {metrics['memory_reduction_x']:.2f}x < {args.min_memory_x:.2f}x"
        )
    print("PASS gates")


if __name__ == "__main__":
    main()
