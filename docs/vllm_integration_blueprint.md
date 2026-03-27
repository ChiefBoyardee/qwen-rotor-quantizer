# vLLM Integration Blueprint for RotorQuant POC

## Scope
This document maps the greenfield POC into expected vLLM integration seams
without modifying a vLLM fork yet.

## Target Interception Points
- KV write path during prefill in attention backend implementation.
- Phase transition hook at prefill completion.
- KV read path during decode attention step.

## Adapter Mapping
- `write_kv_prefill(meta, kv)`:
  - validate `meta.layer_type` and `meta.head_dim` (`128|256` for Qwen hybrid)
  - store contiguous FP16 cache blocks
- `on_prefill_complete()`:
  - trigger bulk quantization for all pending layers
  - persist side channels (`norms`, rotors, residual signs/mags)
- `read_kv_decode(meta)`:
  - dequantize on-demand (or return cached dequant view)
  - enforce head_dim parity for mixed linear/full attention stacks

## Required Runtime Metadata
- `layer_idx`
- `layer_type` (`linear_attention` or `full_attention`)
- `head_dim`
- prefill/decode phase state

## Migration Checklist
1. Introduce adapter in a custom attention backend class.
2. Route layer metadata from model config into backend calls.
3. Wire prefill-complete callback from scheduler to adapter.
4. Replace direct paged KV reads with adapter-backed reads in decode path.
5. Add parity tests:
   - FP16 baseline vs quantized decode logits
   - mixed 128/256 layer shape safety
6. Add performance tests:
   - prefill bulk-quantization latency
   - decode tokens/s at long context
