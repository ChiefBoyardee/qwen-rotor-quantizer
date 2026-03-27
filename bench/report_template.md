# RotorQuant POC Benchmark Report

## Environment
- GPU:
- Driver/CUDA:
- PyTorch version:
- Triton enabled:

## Workload
- Head dim: `128|256`
- Sequence length:
- Batch / heads:

## Results
- Backend selected: `triton_fused|fallback`
- Quant latency (ms):
- Dequant latency (ms):
- Fallback quant latency (ms):
- Fallback dequant latency (ms):
- Cosine similarity:
- Memory reduction (x):
- Top-1 retrieval proxy:
- Top-5 retrieval proxy:
- Logprob drift proxy:
- Long decode stability proxy:

## Gates
- Cosine gate (`>= threshold`): PASS/FAIL
- Memory gate (`>= threshold`): PASS/FAIL
- Speed sanity check vs reference path: PASS/FAIL

## Notes
- Observed bottlenecks:
- Follow-up optimization candidates:
