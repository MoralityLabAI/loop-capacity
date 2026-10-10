"""Hand-built weight-tied blocks with ternary weights (claim B, constructively).

A *local tied block* acts on a state H in {0,1}^(S x D) by
    U_s  = [H_s, H_(s-1), ..., H_(s-r), 1, 1]       (causal mixing, zero padded)
    H'_s = H_s + W2 . ReLU(W1 . U_s)
with W1, W2 in {-1, 0, 1}. The two constant entries supply biases -1 and -2.
The token mixing at fixed relative offsets is a width-(r+1) causal
convolution, which hard attention at fixed offsets can also implement.

Both blocks below keep the state exactly binary, so iterating them never
accumulates error. Their size does not depend on the number of passes T or on
the sequence length.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .tasks import S5_PERMS, S5_INDEX, TASKS


@dataclass
class LocalTiedBlock:
    channels: int
    offsets: int          # r: reads positions s, s-1, ..., s-r
    w1: torch.Tensor      # [hidden, channels*(r+1) + 2]
    w2: torch.Tensor      # [channels, hidden]

    def __post_init__(self):
        for w in (self.w1, self.w2):
            assert set(w.unique().tolist()) <= {-1, 0, 1}, "weights must be ternary"

    def step(self, h: torch.Tensor) -> torch.Tensor:
        # h: [B, S, D] integer
        parts = [h]
        for k in range(1, self.offsets + 1):
            parts.append(torch.nn.functional.pad(h, (0, 0, k, 0))[:, :-k])
        ones = torch.ones(*h.shape[:2], 2, dtype=h.dtype, device=h.device)
        u = torch.cat(parts + [ones], dim=-1)
        return h + torch.relu(u @ self.w1.T) @ self.w2.T

    def run(self, h: torch.Tensor, passes: int) -> torch.Tensor:
        for _ in range(passes):
            h = self.step(h)
        return h

    def weight_count(self) -> dict:
        return {"dense": self.w1.numel() + self.w2.numel(),
                "nonzero": int((self.w1 != 0).sum() + (self.w2 != 0).sum())}


# ---------------------------------------------------------------- rule 110
# State channels: w (working cell), f (done wave from BOS), a (latched answer).
# Position s at pass t holds w = c(s-1-t, t). The wave f reaches s after s
# passes; at the pass where f_(s-1)=1 and f_s=0, w_s is exactly c(0, s-1), and
# it is latched into a. Rule 110: new = c + r - [c+r-1]_+ - [l+c+r-2]_+.

def rule110_block() -> LocalTiedBlock:
    W, F, A = 0, 1, 2
    D, r = 3, 2

    def col(ch, off):
        return off * D + ch

    one1, one2 = D * (r + 1), D * (r + 1) + 1
    w1 = torch.zeros(6, D * (r + 1) + 2, dtype=torch.long)
    # u1 = [w_(s-1)]_+
    w1[0, col(W, 1)] = 1
    # u2 = [w_(s-1) + w_s - 1]_+
    w1[1, col(W, 1)] = 1; w1[1, col(W, 0)] = 1; w1[1, one1] = -1
    # u3 = [w_(s-2) + w_(s-1) + w_s - 2]_+
    w1[2, col(W, 2)] = 1; w1[2, col(W, 1)] = 1; w1[2, col(W, 0)] = 1; w1[2, one1] = -1; w1[2, one2] = -1
    # u4 = [f_(s-1) - f_s]_+
    w1[3, col(F, 1)] = 1; w1[3, col(F, 0)] = -1
    # u5 = [f_(s-1) - f_s + w_s - 1]_+   (= g * w_s)
    w1[4, col(F, 1)] = 1; w1[4, col(F, 0)] = -1; w1[4, col(W, 0)] = 1; w1[4, one1] = -1
    # u6 = [f_(s-1) - f_s + a_s - 1]_+   (= g * a_s)
    w1[5, col(F, 1)] = 1; w1[5, col(F, 0)] = -1; w1[5, col(A, 0)] = 1; w1[5, one1] = -1
    w2 = torch.zeros(D, 6, dtype=torch.long)
    w2[W, 0], w2[W, 1], w2[W, 2] = 1, -1, -1
    w2[F, 3] = 1
    w2[A, 4], w2[A, 5] = 1, -1
    return LocalTiedBlock(D, r, w1, w2)


def rule110_embed(x: torch.Tensor) -> torch.Tensor:
    """x: [B, N+1] tokens with BOS (=2) at index 0 -> state [B, N+1, 3]."""
    bos = TASKS["ca110"].bos
    h = torch.zeros(*x.shape, 3, dtype=torch.long, device=x.device)
    h[..., 0] = (x == 1).long()
    h[..., 1] = (x == bos).long()
    return h


def rule110_readout(h: torch.Tensor) -> torch.Tensor:
    return h[:, 1:, 2]


# ---------------------------------------------------------------- S5
# State channels: S (25, window product as a 0/1 matrix), G (25, own input),
# beta (BOS flag). Encoding M(p)[i, j] = [p(i) = j], so M(a o b) = M(b) M(a)
# and p_n = g_n o p_(n-1) gives M(p_n) = M(p_(n-1)) M(g_n). The update is
# S'_s = S_(s-1) G_s through 125 AND units; BOS holds the identity.

def _mat(p) -> torch.Tensor:
    m = torch.zeros(5, 5, dtype=torch.long)
    for i in range(5):
        m[i, p[i]] = 1
    return m


S5_MATS = torch.stack([_mat(p) for p in S5_PERMS])  # [120, 5, 5]


def s5_block() -> LocalTiedBlock:
    D, r = 51, 1
    S0, G0, BETA = 0, 25, 50
    one1 = D * (r + 1)

    def col(ch, off):
        return off * D + ch

    hidden = 125 + 25
    w1 = torch.zeros(hidden, D * (r + 1) + 2, dtype=torch.long)
    w2 = torch.zeros(D, hidden, dtype=torch.long)
    u = 0
    for i in range(5):
        for k in range(5):
            for j in range(5):
                # [S_(s-1)[i,j] + G_s[j,k] - 1]_+
                w1[u, col(S0 + 5 * i + j, 1)] = 1
                w1[u, col(G0 + 5 * j + k, 0)] = 1
                w1[u, one1] = -1
                w2[S0 + 5 * i + k, u] = 1
                u += 1
    for i in range(5):
        for k in range(5):
            # copy unit [S_s[i,k] - beta_s]_+, subtracted so S' replaces S
            w1[u, col(S0 + 5 * i + k, 0)] = 1
            w1[u, col(BETA, 0)] = -1
            w2[S0 + 5 * i + k, u] = -1
            u += 1
    return LocalTiedBlock(D, r, w1, w2)


def s5_embed(x: torch.Tensor) -> torch.Tensor:
    bos = TASKS["s5"].bos
    h = torch.zeros(*x.shape, 51, dtype=torch.long, device=x.device)
    is_bos = x == bos
    mats = S5_MATS.to(x.device)[x.clamp(max=119)].flatten(-2)  # [B, S, 25]
    mats = mats * (~is_bos).unsqueeze(-1)
    eye = torch.eye(5, dtype=torch.long, device=x.device).flatten()
    h[..., 0:25] = torch.where(is_bos.unsqueeze(-1), eye, mats)
    h[..., 25:50] = mats
    h[..., 50] = is_bos.long()
    return h


def s5_readout(h: torch.Tensor) -> torch.Tensor:
    scores = h[:, 1:, 0:25] @ S5_MATS.to(h.device).flatten(-2).T  # [B, N, 120]
    return scores.argmax(-1)
