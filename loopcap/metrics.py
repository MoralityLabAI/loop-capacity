from __future__ import annotations

import torch

THRESHOLDS = (0.8, 0.9, 0.99)


def n_star(acc: torch.Tensor | list[float], theta: float = 0.9) -> int:
    """Largest n with acc(m) >= theta for every m <= n (acc is 1-indexed by
    position: acc[0] is n = 1). Returns len(acc) if every position passes."""
    a = torch.as_tensor(acc, dtype=torch.float64)
    bad = (a < theta).nonzero()
    return int(bad[0, 0]) if len(bad) else int(a.numel())
