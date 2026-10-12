import random

import torch

from loopcap.tasks import S5_PERMS, TASKS, ca110_targets, eval_set, mod7_targets, s5_targets, sample


def test_s5_targets_match_bruteforce():
    rng = random.Random(0)
    g = torch.tensor([[rng.randrange(120) for _ in range(40)] for _ in range(8)])
    got = s5_targets(g)
    for b in range(8):
        p = tuple(range(5))
        for n in range(40):
            a = S5_PERMS[int(g[b, n])]
            p = tuple(a[p[x]] for x in range(5))  # g_n o p_(n-1)
            assert S5_PERMS[int(got[b, n])] == p


def _naive_ca110_column(x):
    """c(0, j) for j < len(x), simulated on an explicit wide window."""
    n = len(x)
    lo, hi = -2 * n - 2, 2 * n + 2
    row = {k: (x[k] if 0 <= k < n else 0) for k in range(lo, hi)}
    col = []
    for j in range(n):
        col.append(row[0])
        new = {}
        for k in range(lo + 1, hi - 1):
            idx = 4 * row[k - 1] + 2 * row[k] + row[k + 1]
            new[k] = (110 >> idx) & 1
        new[lo], new[hi - 1] = 0, 0
        row = new
    return col


def test_ca110_targets_match_naive():
    rng = random.Random(1)
    for _ in range(5):
        x = [rng.randrange(2) for _ in range(30)]
        got = ca110_targets(torch.tensor([x]))[0].tolist()
        assert got == _naive_ca110_column(x)


def test_ca110_is_causal():
    """Changing x_k must not change targets before position k."""
    rng = random.Random(2)
    x = torch.tensor([[rng.randrange(2) for _ in range(64)]])
    base = ca110_targets(x)
    for k in (5, 20, 50):
        y = x.clone()
        y[0, k:] = 1 - y[0, k:]
        assert torch.equal(ca110_targets(y)[0, :k], base[0, :k])


def test_mod7():
    a = torch.tensor([[3, 4, 6, 0, 1]])
    assert mod7_targets(a).tolist() == [[3, 0, 6, 6, 0]]


def test_sample_shapes_and_bos():
    g = torch.Generator()
    g.manual_seed(0)
    for task, spec in TASKS.items():
        n = 1 if task == "memrand" else 16
        x, y = sample(task, (2, 3), n, g, torch.device("cpu"))
        extra = (spec.heads,) if spec.heads else ()
        assert x.shape == (2, 3, n + 1) and y.shape == (2, 3, n, *extra)
        assert (x[..., 0] == spec.bos).all()
        assert int(y.max()) < spec.classes


def test_eval_set_is_fixed():
    a = eval_set("s5", 4, 8, torch.device("cpu"))
    b = eval_set("s5", 4, 8, torch.device("cpu"))
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_ca110_diagonal_matches_naive_spacetime():
    from loopcap.tasks import ca110_diagonal_targets
    rng = random.Random(3)
    n, heads = 20, 12
    x = [rng.randrange(2) for _ in range(n)]
    lo, hi = -3 * n, 3 * n
    rows = [{k: (x[k] if 0 <= k < n else 0) for k in range(lo, hi)}]
    for _ in range(heads):
        r = rows[-1]
        new = {k: (110 >> (4 * r[k - 1] + 2 * r[k] + r[k + 1])) & 1 for k in range(lo + 1, hi - 1)}
        new[lo], new[hi - 1] = 0, 0
        rows.append(new)
    got = ca110_diagonal_targets(torch.tensor([x]), heads)[0]
    for s in range(1, n + 1):
        for k in range(heads):
            # the naive window has unknown cells beyond x_(n-1) set to 0; head k at
            # position s only depends on cells <= s-1, so it must still agree
            assert int(got[s - 1, k]) == rows[k][s - 1 - k]


def test_ca110_diagonal_is_causal_and_extends_column():
    from loopcap.tasks import ca110_diagonal_targets
    rng = random.Random(4)
    x = torch.tensor([[rng.randrange(2) for _ in range(40)]])
    d = ca110_diagonal_targets(x, 16)
    col = ca110_targets(x)
    for s in range(1, 17):  # head s-1 at position s is cell 0 after s-1 steps
        assert int(d[0, s - 1, s - 1]) == int(col[0, s - 1])
    y = x.clone()
    y[0, 25:] = 1 - y[0, 25:]
    assert torch.equal(ca110_diagonal_targets(y, 16)[0, :25], d[0, :25])
