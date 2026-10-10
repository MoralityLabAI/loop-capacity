"""Task families. Every task is causal: the target at 1-indexed position n
depends on exactly the first n input tokens, so n is the problem size.

Sequences are BOS followed by N input tokens. `inputs` has shape [..., N+1]
(BOS at index 0); `targets` has shape [..., N] and targets[..., n-1] is the
label read out at sequence index n.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

S5_PERMS = list(itertools.permutations(range(5)))
S5_INDEX = {p: i for i, p in enumerate(S5_PERMS)}


def _compose(a: tuple[int, ...], b: tuple[int, ...]) -> tuple[int, ...]:
    """(a o b)(x) = a(b(x))."""
    return tuple(a[b[x]] for x in range(5))


def s5_table() -> torch.Tensor:
    """table[a, b] = index of (perm_a o perm_b)."""
    t = torch.empty(120, 120, dtype=torch.long)
    for i, a in enumerate(S5_PERMS):
        for j, b in enumerate(S5_PERMS):
            t[i, j] = S5_INDEX[_compose(a, b)]
    return t


S5_IDENTITY = S5_INDEX[tuple(range(5))]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    vocab: int  # input vocabulary including BOS (BOS = vocab - 1)
    classes: int
    heads: int = 0  # >0: multi-label task with this many binary heads per position

    @property
    def bos(self) -> int:
        return self.vocab - 1

    @property
    def outputs(self) -> int:
        """Width of the readout layer."""
        return self.heads if self.heads else self.classes


CA_HEADS = 32

TASKS = {
    "s5": TaskSpec("s5", vocab=121, classes=120),
    "ca110": TaskSpec("ca110", vocab=3, classes=2),
    "ca110d": TaskSpec("ca110d", vocab=3, classes=2, heads=CA_HEADS),
    "mod7": TaskSpec("mod7", vocab=8, classes=7),
}

_TABLE_CACHE: dict[str, torch.Tensor] = {}


def _s5_table_on(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = s5_table().to(device)
    return _TABLE_CACHE[key]


def s5_targets(g: torch.Tensor) -> torch.Tensor:
    """p_n = g_n o p_(n-1), p_0 = identity. g: [..., N] long."""
    table = _s5_table_on(g.device)
    flat = g.reshape(-1, g.shape[-1])
    p = torch.full((flat.shape[0],), S5_IDENTITY, dtype=torch.long, device=g.device)
    out = torch.empty_like(flat)
    for n in range(flat.shape[1]):
        p = table[flat[:, n], p]
        out[:, n] = p
    return out.reshape(g.shape)


def rule110_step(row: torch.Tensor) -> torch.Tensor:
    """One rule-110 update on the last axis, zero beyond both ends."""
    left = torch.nn.functional.pad(row[..., :-1], (1, 0))
    right = torch.nn.functional.pad(row[..., 1:], (0, 1))
    idx = (left << 2) | (row << 1) | right
    return (110 >> idx) & 1


def ca110_targets(x: torch.Tensor) -> torch.Tensor:
    """targets[..., j] = c(0, j): cell 0 after j steps of rule 110 started from
    c(k, 0) = x_k for k >= 0 and c(k, 0) = 0 for k < 0.

    c(0, j) depends on spatial cells [-j, j], so simulating on [-(N-1), N-1]
    with zero padding is exact: the padded edges never reach cell 0 within
    N-1 steps (the left background really is zero; the right edge is N-1
    cells away from cell 0).
    """
    flat = x.reshape(-1, x.shape[-1]).long()
    n = flat.shape[1]
    row = torch.cat([torch.zeros(flat.shape[0], n - 1, dtype=torch.long, device=x.device), flat], dim=1)
    out = torch.empty_like(flat)
    centre = n - 1  # index of spatial cell 0
    for j in range(n):
        out[:, j] = row[:, centre]
        if j + 1 < n:
            row = rule110_step(row)
    return out.reshape(x.shape)


def ca110_diagonal_targets(x: torch.Tensor, heads: int = CA_HEADS) -> torch.Tensor:
    """targets[..., s-1, k] = c(s-1-k, k) for k < heads: at every position, the
    spacetime diagonal through (s-1, 0). Head k needs k rule applications and
    depends only on x_0..x_(s-1) (zero background on the left).

    The simulation covers spatial cells [-2*heads, N-1]. The zero padding on
    the left is exact (the background is zero and its infected region is
    inside the window). The right padding is wrong, but its error front
    moves left one cell per step, so at time k it has reached cells
    >= N-1-k+1 and never the needed cells <= N-1-k.
    """
    flat = x.reshape(-1, x.shape[-1]).long()
    n = flat.shape[1]
    left = 2 * heads
    row = torch.cat([torch.zeros(flat.shape[0], left, dtype=torch.long, device=x.device), flat], dim=1)
    out = torch.empty(flat.shape[0], n, heads, dtype=torch.long, device=x.device)
    for k in range(heads):
        out[:, :, k] = row[:, left - k:left - k + n]
        if k + 1 < heads:
            row = rule110_step(row)
    return out.reshape(*x.shape, heads)


def mod7_targets(a: torch.Tensor) -> torch.Tensor:
    return torch.cumsum(a, dim=-1) % 7


def sample(task: str, shape: tuple[int, ...], n: int, generator: torch.Generator,
           device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw inputs [*shape, n+1] (with BOS) and targets [*shape, n]."""
    spec = TASKS[task]
    if task == "s5":
        tok = torch.randint(0, 120, (*shape, n), generator=generator, device=device)
        tgt = s5_targets(tok)
    elif task == "ca110":
        tok = torch.randint(0, 2, (*shape, n), generator=generator, device=device)
        tgt = ca110_targets(tok)
    elif task == "ca110d":
        tok = torch.randint(0, 2, (*shape, n), generator=generator, device=device)
        tgt = ca110_diagonal_targets(tok, spec.heads)
    elif task == "mod7":
        tok = torch.randint(0, 7, (*shape, n), generator=generator, device=device)
        tgt = mod7_targets(tok)
    else:
        raise KeyError(task)
    bos = torch.full((*shape, 1), spec.bos, dtype=torch.long, device=device)
    return torch.cat([bos, tok], dim=-1), tgt


def eval_set(task: str, count: int, n: int, device: torch.device, seed: int = 12345):
    """Held-out set, drawn on the CPU so it is identical on every device."""
    g = torch.Generator()
    g.manual_seed(seed)
    x, y = sample(task, (count,), n, g, torch.device("cpu"))
    return x.to(device), y.to(device)
