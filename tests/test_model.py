import pytest
import torch

from loopcap.metrics import n_star
from loopcap.model import ModelStack, tied_arch, untied_compute_arch, untied_matched_arch
from loopcap.quant import Quantizer
from loopcap.tasks import sample

CPU = torch.device("cpu")


@pytest.mark.parametrize("task", ["s5", "ca110", "mod7"])
@pytest.mark.parametrize("d", [32, 64])
@pytest.mark.parametrize("T", [1, 2, 4, 8, 16, 32])
def test_untied_matched_within_two_percent(task, d, T):
    target = tied_arch(task, d, T).param_count()
    got = untied_matched_arch(task, d, T).param_count()
    assert abs(got - target) / target <= 0.02


def test_param_counts():
    assert tied_arch("s5", 32, 8).param_count() == 121 * 32 + 12 * 32 * 32 + 32 * 120
    assert untied_compute_arch("ca110", 32, 4).param_count() == 3 * 32 + 4 * 12 * 32 * 32 + 32 * 2


def test_quantizer_levels():
    torch.manual_seed(0)
    w = torch.randn(3, 64, 64)
    q = Quantizer([32, 4, 1.58], CPU)(w)
    assert torch.equal(q[0], w[0])
    assert q[1].unique().numel() <= 15
    assert q[2].unique().numel() <= 3


def test_models_in_a_stack_are_independent():
    arch = tied_arch("ca110", 16, 3)
    model = ModelStack(arch, [32, 1.58], [1, 2], CPU)
    g = torch.Generator()
    g.manual_seed(0)
    x, y = sample("ca110", (2, 4), 32, g, CPU)
    loss = model.train_loss(x, y)
    loss[0].backward()
    for p in model.parameters():
        assert p.grad[1].abs().sum() == 0
        assert p.grad[0].abs().sum() > 0


def test_tied_eval_matches_training_depth():
    arch = tied_arch("mod7", 16, 4)
    model = ModelStack(arch, [32], [3], CPU)
    g = torch.Generator()
    g.manual_seed(1)
    x, y = sample("mod7", (1, 5), 24, g, CPU)
    out = model.correct_by_position(x, y, [1, 4, 9])
    assert set(out) == {1, 4, 9}
    assert out[4].shape == (1, 24)


def test_n_star():
    assert n_star([1, 1, 0.95, 0.5, 1], 0.9) == 3
    assert n_star([0.2, 1], 0.9) == 0
    assert n_star([1, 1], 0.9) == 2
