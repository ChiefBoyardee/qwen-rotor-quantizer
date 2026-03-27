from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from benchmark_kv import run_once
from quality_gate_suite import run_suite
import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--out", type=str, default="bench/phase3_report.json")
    args = ap.parse_args()

    device = torch.device(args.device)
    report = {}
    for d in (128, 256):
        perf = run_once(device, d, args.seq)
        quality = run_suite(device, d, args.seq)
        report[f"d{d}"] = {"perf": perf, "quality": quality}
    p = pathlib.Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
