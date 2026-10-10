"""Per-model weight fake-quantization with a straight-through estimator.

Weights are stacked along a leading model axis [M, ...]; each model has its
own bit-width:
  32    full precision (no quantization)
  1.58  ternary {-1, 0, 1} * s, s = mean|W| per tensor (BitNet b1.58)
  b>=2  symmetric integers {-(2^(b-1)-1) .. 2^(b-1)-1} * s, s = max|W| / qmax
Scales are per tensor (one fp32 scale per weight tensor) or, with
per_channel, per output feature (one fp32 scale per column).
"""

from __future__ import annotations

import math

import torch

FULL = 32.0
TERNARY = 1.58


def levels(bits: float) -> int:
    """Number of distinct weight values a code can take (inf for fp32)."""
    if bits >= FULL:
        return 0
    if abs(bits - TERNARY) < 1e-6:
        return 3
    return 2 ** int(bits) - 1


def weight_bits(bits: float) -> float:
    """Information per weight: log2(levels), or 32 for float."""
    return FULL if bits >= FULL else math.log2(levels(bits))


class Quantizer:
    """Holds per-model quantization settings for a stack of M models."""

    def __init__(self, bits: list[float], device: torch.device, per_channel: bool = False):
        self.bits = list(bits)
        self.per_channel = per_channel  # one scale per output feature (last axis) instead of per tensor
        m = len(bits)
        is_fp = torch.tensor([b >= FULL for b in bits], device=device)
        is_tern = torch.tensor([abs(b - TERNARY) < 1e-6 for b in bits], device=device)
        qmax = torch.tensor([1.0 if (b >= FULL or abs(b - TERNARY) < 1e-6) else float(2 ** (int(b) - 1) - 1)
                             for b in bits], device=device)
        self.is_fp = is_fp.view(m, 1)
        self.is_tern = is_tern.view(m, 1)
        self.qmax = qmax.view(m, 1)
        self.any_quant = not bool(is_fp.all())
        self.active = True  # False: every model runs in float (warm-start phase)

    def __call__(self, w: torch.Tensor) -> torch.Tensor:
        if not self.any_quant or not self.active:
            return w
        m = w.shape[0]
        if self.per_channel:
            flat = w.reshape(m, -1, w.shape[-1])  # [M, fan_in, out]
            a = flat.detach().abs()
            s_mean = a.mean(dim=1, keepdim=True)
            s_max = a.amax(dim=1, keepdim=True)
            shape = (m, 1, 1)
        else:
            flat = w.reshape(m, -1)
            a = flat.detach().abs()
            s_mean = a.mean(dim=1, keepdim=True)
            s_max = a.amax(dim=1, keepdim=True)
            shape = (m, 1)
        tern, qmax, isfp = self.is_tern.view(shape), self.qmax.view(shape), self.is_fp.view(shape)
        s = torch.where(tern, s_mean, s_max / qmax).clamp_min(1e-8)
        q = torch.clamp(torch.round(flat / s), -qmax, qmax) * s
        q = torch.where(isfp, flat.detach(), q)
        # forward value is exactly q (x - x == 0 in IEEE); gradient is identity
        out = q.detach() + (flat - flat.detach())
        return out.view_as(w)
