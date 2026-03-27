from __future__ import annotations

import argparse
import json
import pathlib
import sys

import torch

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from rotorquant.kernels import dequantize_kv, quantize_kv


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
    )


def _recon_stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    err = (a - b).abs().reshape(-1)
    return {
        "mean_abs_error": float(err.mean().item()),
        "p95_abs_error": float(torch.quantile(err, 0.95).item()),
        "max_abs_error": float(err.max().item()),
    }


def _retrieval_proxy(a: torch.Tensor, b: torch.Tensor, k: int = 5) -> float:
    # Top-k overlap on absolute activation magnitudes as a retrieval proxy.
    idx_a = torch.topk(a.abs().reshape(-1), k=k).indices
    idx_b = torch.topk(b.abs().reshape(-1), k=k).indices
    inter = len(set(idx_a.tolist()) & set(idx_b.tolist()))
    return inter / float(k)


def _logprob_drift_proxy(a: torch.Tensor, b: torch.Tensor) -> float:
    # Synthetic logits proxy from mean over seq/head dims.
    la = a.mean(dim=(0, 1, 2))
    lb = b.mean(dim=(0, 1, 2))
    pa = torch.log_softmax(la, dim=0)
    pb = torch.log_softmax(lb, dim=0)
    return float((pa - pb).abs().mean().item())


def run_suite(device: torch.device, head_dim: int, seq: int) -> dict[str, float]:
    kv = torch.randn(1, 8, seq, head_dim, dtype=torch.float32, device=device)
    qs = quantize_kv(kv, seed=17)
    recon = dequantize_kv(qs).to(kv.dtype)
    out = {"cosine": _cosine(kv, recon)}
    out.update(_recon_stats(kv, recon))
    out["top1_retrieval"] = _retrieval_proxy(kv, recon, k=1)
    out["top5_retrieval"] = _retrieval_proxy(kv, recon, k=5)
    out["logprob_drift_proxy"] = _logprob_drift_proxy(kv, recon)
    out["long_decode_stability_proxy"] = float(recon.abs().mean().item())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--head-dim", type=int, choices=[128, 256], default=128)
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--min-cosine", type=float, default=0.95)
    ap.add_argument("--max-logprob-drift", type=float, default=0.20)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    metrics = run_suite(torch.device(args.device), args.head_dim, args.seq)
    print(metrics)
    if metrics["cosine"] < args.min_cosine:
        raise SystemExit(f"FAIL cosine gate: {metrics['cosine']:.4f}")
    if metrics["logprob_drift_proxy"] > args.max_logprob_drift:
        raise SystemExit(f"FAIL logprob drift gate: {metrics['logprob_drift_proxy']:.4f}")
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("PASS quality gates")


if __name__ == "__main__":
    main()
