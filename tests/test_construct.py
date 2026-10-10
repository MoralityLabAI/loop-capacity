"""Claim B, constructively: fixed ternary weights, any number of passes."""

import torch

from loopcap.construct import (rule110_block, rule110_embed, rule110_readout,
                               s5_block, s5_embed, s5_readout)
from loopcap.tasks import sample


def _data(task, n=256, b=64, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    return sample(task, (b,), n, g, torch.device("cpu"))


def test_rule110_block_solves_exactly_T_positions():
    x, y = _data("ca110")
    blk = rule110_block()
    h0 = rule110_embed(x)
    for T in (1, 7, 64, 255, 256, 400):
        pred = rule110_readout(blk.run(h0, T))
        solved = (pred == y).all(0)
        k = min(T, y.shape[1])
        assert solved[:k].all(), f"T={T}: a position <= T is wrong"
    # the state stays exactly binary
    h = blk.run(h0, 300)
    assert set(h.unique().tolist()) <= {0, 1}


def test_rule110_block_is_small_and_T_independent():
    wc = rule110_block().weight_count()
    assert wc["dense"] < 100


def test_s5_block_solves_T_plus_one_positions():
    x, y = _data("s5", n=128, b=32)
    blk = s5_block()
    h0 = s5_embed(x)
    for T in (0, 1, 10, 127, 200):
        pred = s5_readout(blk.run(h0, T))
        solved = (pred == y).all(0)
        k = min(T + 1, y.shape[1])
        assert solved[:k].all(), f"T={T}"
        if T + 1 < y.shape[1]:
            # beyond the reach of T passes the block is not yet right
            assert not solved[T + 1:].all()
