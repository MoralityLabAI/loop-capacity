import pytest
import torch

from loopcap.metrics import n_star
from loopcap.model import ModelStack, tied_arch, untied_compute_arch, untied_matched_arch
from loopcap.quant import Quantizer
from loopcap.tasks import sample

CPU = torch.device("cpu")


@pytest.mark.parametrize("task", ["s5", "ca110", "ca110d", "mod7"])
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


def test_per_channel_quantizer_levels():
    torch.manual_seed(1)
    w = torch.randn(2, 16, 8)
    q = Quantizer([4, 1.58], CPU, per_channel=True)(w)
    for j in range(8):
        assert q[1, :, j].unique().numel() <= 3
        assert q[0, :, j].unique().numel() <= 15


def test_band_mask_and_local_model_runs():
    import dataclasses
    from loopcap.model import band_mask
    m = band_mask(6, 3, CPU)
    assert m[5].tolist() == [False, False, False, True, True, True]
    assert m[1].tolist() == [True, True, False, False, False, False]
    arch = dataclasses.replace(tied_arch("mod7", 16, 3), window=2)
    model = ModelStack(arch, [32, 1.58], [1, 2], CPU)
    g = torch.Generator()
    g.manual_seed(0)
    x, y = sample("mod7", (2, 3), 12, g, CPU)
    loss = model.train_loss(x, y)
    assert torch.isfinite(loss).all()


def test_local_window_limits_receptive_field():
    """With window w and T passes, position n cannot see tokens more than T*(w-1) back."""
    import dataclasses
    arch = dataclasses.replace(tied_arch("mod7", 16, 2), window=2)
    model = ModelStack(arch, [32], [1], CPU)
    g = torch.Generator()
    g.manual_seed(1)
    x, _ = sample("mod7", (1, 1), 10, g, CPU)
    y = x.clone()
    y[..., 1] = (y[..., 1] + 1) % 7  # change the first input token
    a = model.last_logits(x, [2])[2]
    b = model.last_logits(y, [2])[2]
    assert torch.allclose(a, b)  # the last position is 10 steps away; reach is 2*(2-1) = 2


def test_mem16_task_and_eval():
    from loopcap.tasks import mem_all_keys, mem_table
    from loopcap.train import eval_memorization
    x, y = mem_all_keys(CPU)
    assert x.shape == (65536, 17) and y.shape == (65536,)
    assert abs(float(y.float().mean()) - 0.5) < 0.01
    g = torch.Generator()
    g.manual_seed(0)
    xs, ys = sample("mem16", (2, 5), 16, g, CPU)
    keys = (xs[..., 1:] * (2 ** torch.arange(15, -1, -1))).sum(-1)
    assert torch.equal(ys[..., -1], mem_table(CPU)[keys])
    model = ModelStack(tied_arch("mem16", 8, 1), [32], [1], CPU)
    r = eval_memorization(model, CPU, [1], amp=False, chunk=16384)[1]
    assert abs(float(r["acc"][0]) - 0.5) < 0.02  # an untrained model is at chance
    assert torch.isfinite(r["bits"]).all()


def test_memrand_frozen_embedding_not_counted_or_trained():
    from loopcap.tasks import TASKS
    from loopcap.train import eval_memorization
    arch = tied_arch("memrand", 16, 2)
    assert arch.param_count() == 12 * 16 * 16 + 16 * 2  # block + readout only
    model = ModelStack(arch, [32, 1.58], [1, 2], CPU)
    assert not model.emb.requires_grad
    assert torch.equal(model.emb[0], model.emb[1])  # same key vectors for every model
    g = torch.Generator()
    g.manual_seed(0)
    x, y = sample("memrand", (2, 8), 1, g, CPU)
    assert x.shape == (2, 8, 2) and y.shape == (2, 8, 1)
    model.train_loss(x, y).sum().backward()
    assert model.emb.grad is None
    r = eval_memorization(model, CPU, [2], amp=False, chunk=16384, task="memrand")[2]
    assert abs(float(r["acc"][0]) - 0.5) < 0.02
    assert TASKS["memrand"].frozen_embed


def test_training_logs_and_evaluates_inside_the_loop(tmp_path):
    """Regression: periodic train logs and evaluations must happen during training."""
    import json
    from loopcap.train import parser, run
    out = tmp_path / "r.json"
    args = parser().parse_args(["--task", "mod7", "--arm", "tied", "--d", "16", "--T", "2", "--bits", "32",
                                "--seeds", "1", "--steps", "6", "--log-every", "2", "--eval-every", "3",
                                "--eval-count", "8", "--probe-count", "8", "--seq", "8", "--device", "cpu",
                                "--no-amp", "--out", str(out)])
    run(args)
    events = [json.loads(l)["event"] for l in out.with_suffix(".log.jsonl").read_text().splitlines()]
    assert events.count("train") >= 3 and events.count("eval") == 2
    r = json.loads(out.read_text())
    assert [h["step"] for h in r["models"][0]["history"]] == [3, 6]


def test_progressive_loss_restarts_from_detached_state():
    arch = tied_arch("mod7", 16, 4)
    model = ModelStack(arch, [32, 1.58], [1, 2], CPU)
    g = torch.Generator()
    g.manual_seed(2)
    x, y = sample("mod7", (2, 3), 10, g, CPU)
    base = model.train_loss(x, y)
    for n in (0, 2, 4):  # restart at depth n, segment ends at n + 3 (up to T + 3)
        loss = model.train_loss(x, y, prog_k=3, prog_n=torch.tensor([n]), prog_alpha=0.5)
        assert torch.isfinite(loss).all() and not torch.allclose(loss, base)
        model.zero_grad()
        loss.sum().backward()
        assert all(p.grad is not None for p in model.parameters())
    # alpha = 0 recovers the plain loss
    assert torch.allclose(model.train_loss(x, y, prog_k=3, prog_n=torch.tensor([1]), prog_alpha=0.0), base)
