import torch

from rotorquant.bitpack import aligned_nbytes, pack_bits, pack_bits_aligned, unpack_bits


def test_pack_unpack_3bit_roundtrip():
    x = torch.randint(0, 8, (257,), dtype=torch.int32)
    packed, n = pack_bits(x, bits=3)
    y = unpack_bits(packed, bits=3, original_count=n)
    assert torch.equal(x, y)


def test_pack_unpack_1bit_roundtrip():
    x = torch.randint(0, 2, (129,), dtype=torch.int32)
    packed, n = pack_bits(x, bits=1)
    y = unpack_bits(packed, bits=1, original_count=n)
    assert torch.equal(x, y)


def test_pack_aligned_padding_and_raw_recovery():
    x = torch.randint(0, 8, (101,), dtype=torch.int32)
    packed, n, raw_nbytes = pack_bits_aligned(x, bits=3, align_bytes=16)
    assert packed.numel() == aligned_nbytes(raw_nbytes, align_bytes=16)
    y = unpack_bits(packed[:raw_nbytes], bits=3, original_count=n)
    assert torch.equal(x, y)
