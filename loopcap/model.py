"""A stack of M independent small transformers trained in one process.

All M models share an architecture (width, depth, tying) and differ in seed
and weight bit-width. Weights carry a leading model axis and every matmul is a
batched matmul over it, which keeps a 16 GB shared GPU busy with tiny models.

Architecture of one model:
  e = Embed(x)
  h_0 = 0
  h_t = Block_{k(t)}(h_(t-1) + e)        t = 1..passes  (input re-injection)
  logits_t = Readout(RMSNorm(h_t))       shared readout at every depth
  Block(z) = z + Attn(RMSNorm(z)); z + MLP(RMSNorm(z))
TIED: one block, k(t) = 0. UNTIED: `passes` blocks, k(t) = t-1.
No biases and no norm gains, so every parameter is a quantizable matrix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from .quant import Quantizer
from .tasks import TASKS


@dataclass(frozen=True)
class ArchSpec:
    task: str
    d: int
    hidden: int
    n_blocks: int  # distinct blocks: 1 for tied, == passes for untied
    passes: int
    head_dim: int = 8

    @property
    def heads(self) -> int:
        return self.d // self.head_dim

    def param_count(self) -> int:
        spec = TASKS[self.task]
        per_block = 4 * self.d * self.d + 2 * self.d * self.hidden
        return spec.vocab * self.d + self.n_blocks * per_block + self.d * spec.classes

    def tensor_count(self) -> int:
        return 2 + 4 * self.n_blocks

    def to_dict(self) -> dict:
        return asdict(self) | {"params": self.param_count()}


def tied_arch(task: str, d: int, passes: int) -> ArchSpec:
    return ArchSpec(task, d, 4 * d, 1, passes)


def untied_compute_arch(task: str, d: int, passes: int) -> ArchSpec:
    return ArchSpec(task, d, 4 * d, passes, passes)


def untied_matched_arch(task: str, d: int, passes: int, tol: float = 0.02) -> ArchSpec:
    """`passes` distinct blocks whose total weight count matches tied_arch(d).

    Picks the widest d' (multiple of the head dim) whose MLP width h' still
    satisfies h' >= 2 d' (falling back to h' >= d'), then sets h' so the
    total lands on the target count.
    """
    target = tied_arch(task, d, passes).param_count()
    if passes == 1:
        return tied_arch(task, d, 1)
    spec = TASKS[task]
    head_dim = 8
    best = None
    for min_ratio in (2.0, 1.0, 0.5):
        for dp in range(d, head_dim - 1, -head_dim):
            rest = target - (spec.vocab + spec.classes) * dp - passes * 4 * dp * dp
            if rest <= 0:
                continue
            hp = round(rest / (2 * passes * dp))
            if hp >= min_ratio * dp and hp >= 1:
                best = ArchSpec(task, dp, hp, passes, passes, head_dim)
                break
        if best is not None:
            break
    if best is None:
        raise ValueError(f"cannot match {target} params with {passes} blocks")
    got = best.param_count()
    if abs(got - target) / target > tol:
        raise ValueError(f"matched width misses target: {got} vs {target}")
    return best


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def rope_tables(seq: int, head_dim: int, device, base: float = 10000.0):
    inv = base ** (-torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    ang = torch.arange(seq, device=device, dtype=torch.float32)[:, None] * inv[None, :]
    return torch.cos(ang), torch.sin(ang)  # [S, hd/2]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: [..., S, H, hd]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    c = cos[:, None, :]
    s = sin[:, None, :]
    out = torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    return out.flatten(-2)


def bmm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x: [M, ..., din], w: [M, din, dout] -> [M, ..., dout]."""
    m = x.shape[0]
    lead = x.shape[1:-1]
    y = torch.bmm(x.reshape(m, -1, x.shape[-1]), w)
    return y.view(m, *lead, w.shape[-1])


def block_forward(z: torch.Tensor, wqkv, wo, w1, w2, heads: int, cos, sin) -> torch.Tensor:
    m, b, s, d = z.shape
    hd = d // heads
    qkv = bmm(rms_norm(z), wqkv).view(m, b, s, 3, heads, hd)
    q = apply_rope(qkv[:, :, :, 0], cos, sin)
    k = apply_rope(qkv[:, :, :, 1], cos, sin)
    v = qkv[:, :, :, 2]
    q, k, v = (t.permute(0, 1, 3, 2, 4).reshape(m * b, heads, s, hd) for t in (q, k, v))
    att = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    att = att.view(m, b, heads, s, hd).permute(0, 1, 3, 2, 4).reshape(m, b, s, d)
    z = z + bmm(att, wo)
    z = z + bmm(F.gelu(bmm(rms_norm(z), w1)), w2)
    return z


class ModelStack(nn.Module):
    def __init__(self, arch: ArchSpec, bits: list[float], seeds: list[int], device):
        super().__init__()
        assert len(bits) == len(seeds)
        self.arch = arch
        self.m = len(bits)
        self.bits = list(bits)
        self.seeds = list(seeds)
        spec = TASKS[arch.task]
        d, h = arch.d, arch.hidden
        out_scale = 1.0 / math.sqrt(2 * arch.passes)

        def init(shape, std):
            parts = []
            for i, sd in enumerate(seeds):
                g = torch.Generator()
                g.manual_seed(sd * 1000003 + init.counter)
                parts.append(torch.randn(shape, generator=g) * std)
            init.counter += 1
            return nn.Parameter(torch.stack(parts).to(device))

        init.counter = 0
        self.emb = init((spec.vocab, d), 1.0)
        self.wqkv = nn.ParameterList()
        self.wo = nn.ParameterList()
        self.w1 = nn.ParameterList()
        self.w2 = nn.ParameterList()
        for _ in range(arch.n_blocks):
            self.wqkv.append(init((d, 3 * d), d ** -0.5))
            self.wo.append(init((d, d), d ** -0.5 * out_scale))
            self.w1.append(init((d, h), d ** -0.5))
            self.w2.append(init((h, d), h ** -0.5 * out_scale))
        self.readout = init((d, spec.classes), d ** -0.5)
        self.quant = Quantizer(bits, device)
        self._rope = {}

    def rope(self, seq: int, device):
        if seq not in self._rope:
            self._rope[seq] = rope_tables(seq, self.arch.head_dim, device)
        return self._rope[seq]

    def quantized(self):
        q = self.quant
        blocks = [(q(self.wqkv[i]), q(self.wo[i]), q(self.w1[i]), q(self.w2[i]))
                  for i in range(self.arch.n_blocks)]
        return q(self.emb), blocks, q(self.readout)

    def embed(self, emb_q: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # x: [M, B, S] -> [M, B, S, d]
        m, b, s = x.shape
        idx = x.reshape(m, b * s, 1).expand(-1, -1, emb_q.shape[-1])
        return torch.gather(emb_q, 1, idx).view(m, b, s, -1)

    def block_index(self, t: int) -> int:
        return 0 if self.arch.n_blocks == 1 else t

    def train_loss(self, x: torch.Tensor, y: torch.Tensor, use_checkpoint: bool = True) -> torch.Tensor:
        """Per-model loss [M]: 0.5*CE(final) + 0.5*mean CE over earlier depths."""
        arch = self.arch
        emb_q, blocks, ro_q = self.quantized()
        e = self.embed(emb_q, x)
        cos, sin = self.rope(x.shape[-1], x.device)
        m, b, s, _ = e.shape
        classes = ro_q.shape[-1]
        T = arch.passes
        weights = [1.0] if T == 1 else [0.5 / (T - 1)] * (T - 1) + [0.5]

        def step(h, wqkv, wo, w1, w2):
            hn = block_forward(h + e, wqkv, wo, w1, w2, arch.heads, cos, sin)
            logits = bmm(rms_norm(hn[:, :, 1:]), ro_q)
            ce = F.cross_entropy(logits.reshape(-1, classes), y.reshape(-1), reduction="none")
            return hn, ce.view(m, -1).mean(1)

        h = torch.zeros_like(e)
        total = torch.zeros(m, device=x.device)
        for t in range(T):
            blk = blocks[self.block_index(t)]
            if use_checkpoint and T > 1:
                h, ce = checkpoint(step, h, *blk, use_reentrant=False)
            else:
                h, ce = step(h, *blk)
            total = total + weights[t] * ce
        return total

    @torch.no_grad()
    def correct_by_position(self, x: torch.Tensor, y: torch.Tensor, record: list[int]) -> dict[int, torch.Tensor]:
        """Number of correct predictions per position, [M, N], after each pass
        count in `record`. Tied models may run past their training depth;
        untied models only up to their number of blocks."""
        arch = self.arch
        emb_q, blocks, ro_q = self.quantized()
        e = self.embed(emb_q, x)
        cos, sin = self.rope(x.shape[-1], x.device)
        h = torch.zeros_like(e)
        out = {}
        last = max(record)
        if arch.n_blocks > 1:
            assert last <= arch.n_blocks
        for t in range(last):
            blk = blocks[self.block_index(t)]
            h = block_forward(h + e, *blk, arch.heads, cos, sin)
            if (t + 1) in record:
                pred = bmm(rms_norm(h[:, :, 1:]), ro_q).argmax(-1)
                out[t + 1] = (pred == y).sum(1)
        return out
