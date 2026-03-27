from pathlib import Path

import torch

from rotorquant.oracle import generate_fixture, replay_fixture


def test_oracle_fixture_generate_and_replay(tmp_path: Path):
    out_file = tmp_path / "fixture.json"
    m = generate_fixture(
        seed=5, shape=(1, 4, 16, 128), device=torch.device("cpu"), out_file=str(out_file)
    )
    r = replay_fixture(str(out_file))
    assert "cosine" in m
    assert r["seed"] == 5.0
