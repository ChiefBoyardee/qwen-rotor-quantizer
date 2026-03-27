from __future__ import annotations

import torch


def aligned_nbytes(raw_nbytes: int, align_bytes: int = 16) -> int:
    if align_bytes <= 0:
        raise ValueError("align_bytes must be > 0")
    rem = raw_nbytes % align_bytes
    return raw_nbytes if rem == 0 else raw_nbytes + (align_bytes - rem)


def pack_bits(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, int]:
    """
    Pack integer tensor to uint8 bitstream.
    Returns (packed_bytes, original_element_count).
    """
    if values.dtype not in (torch.int32, torch.int64, torch.uint8):
        raise ValueError("values must be integer tensor")
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")

    flat = values.reshape(-1).to(torch.int64)
    n = flat.numel()
    total_bits = n * bits
    n_bytes = (total_bits + 7) // 8
    out = torch.zeros(n_bytes, dtype=torch.uint8, device=values.device)

    bit_pos = 0
    mask = (1 << bits) - 1
    for v in flat.tolist():
        v = int(v) & mask
        byte_idx = bit_pos // 8
        shift = bit_pos % 8
        out[byte_idx] |= torch.tensor((v << shift) & 0xFF, dtype=torch.uint8, device=out.device)
        spill = (shift + bits) - 8
        if spill > 0 and byte_idx + 1 < n_bytes:
            out[byte_idx + 1] |= torch.tensor(
                (v >> (bits - spill)) & 0xFF, dtype=torch.uint8, device=out.device
            )
        bit_pos += bits
    return out, n


def pack_bits_aligned(
    values: torch.Tensor, bits: int, align_bytes: int = 16
) -> tuple[torch.Tensor, int, int]:
    packed, n = pack_bits(values, bits)
    raw = int(packed.numel())
    aligned = aligned_nbytes(raw, align_bytes=align_bytes)
    if aligned == raw:
        return packed, n, raw
    out = torch.zeros(aligned, dtype=torch.uint8, device=packed.device)
    out[:raw] = packed
    return out, n, raw


def unpack_bits(
    packed: torch.Tensor, bits: int, original_count: int, *, device: torch.device | None = None
) -> torch.Tensor:
    if bits <= 0 or bits > 8:
        raise ValueError("bits must be in [1, 8]")
    if packed.dtype != torch.uint8:
        raise ValueError("packed must be uint8")

    if device is None:
        device = packed.device
    out = torch.zeros(original_count, dtype=torch.int32, device=device)
    bit_pos = 0
    mask = (1 << bits) - 1
    p = packed.to(torch.int64)
    for i in range(original_count):
        byte_idx = bit_pos // 8
        shift = bit_pos % 8
        v = (int(p[byte_idx].item()) >> shift) & 0xFF
        spill = (shift + bits) - 8
        if spill > 0 and byte_idx + 1 < p.numel():
            v |= (int(p[byte_idx + 1].item()) & ((1 << spill) - 1)) << (bits - spill)
        out[i] = v & mask
        bit_pos += bits
    return out

