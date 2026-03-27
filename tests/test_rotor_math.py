import torch

from rotorquant.rotor_math import (
    chunk3,
    l2_norm_restore,
    l2_norm_separate,
    make_random_rotors,
    quaternion_rotate_chunks,
    unchunk3,
)


def test_chunk_round_trip():
    x = torch.randn(2, 5, 7)
    ch, d = chunk3(x)
    y = unchunk3(ch, d)
    assert y.shape == x.shape
    assert torch.allclose(x, y, atol=0, rtol=0)


def test_rotor_preserves_norm_per_chunk():
    x = torch.randn(4, 6, 3)
    rot = make_random_rotors(6, device=x.device, dtype=x.dtype, seed=123)
    y = quaternion_rotate_chunks(x, rot)
    nx = x.norm(dim=-1)
    ny = y.norm(dim=-1)
    assert torch.allclose(nx, ny, atol=1e-5, rtol=1e-5)


def test_l2_norm_separability():
    x = torch.randn(3, 11)
    xn, n = l2_norm_separate(x)
    xr = l2_norm_restore(xn, n)
    assert torch.allclose(x, xr, atol=1e-5, rtol=1e-5)
