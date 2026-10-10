"""Train one job: M models (bit-widths x seeds) of one architecture on one task.

    python -m loopcap.train --task ca110 --arm tied --d 32 --T 8 \
        --bits 32 4 1.58 --seeds 1 2 --steps 6000 --out results/runs/x.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib
from pathlib import Path

import torch

from .metrics import THRESHOLDS, n_star
from .model import ModelStack, tied_arch, untied_compute_arch, untied_matched_arch
from .quant import weight_bits
from .tasks import TASKS, eval_set, sample

ARMS = {"tied": tied_arch, "untied_w": untied_matched_arch, "untied_c": untied_compute_arch}
SEQ = 256


def make_arch(task: str, arm: str, d: int, T: int):
    return ARMS[arm](task, d, T)


def lower_priority() -> None:
    if os.name == "nt":
        import ctypes
        BELOW_NORMAL = 0x4000
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), BELOW_NORMAL)


def lr_at(step: int, total: int, peak: float, warmup: int) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    frac = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


def clip_per_model(params, m: int, max_norm: float) -> torch.Tensor:
    sq = None
    for p in params:
        if p.grad is None:
            continue
        s = p.grad.reshape(m, -1).pow(2).sum(1)
        sq = s if sq is None else sq + s
    norm = sq.sqrt()
    scale = (max_norm / (norm + 1e-6)).clamp(max=1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale.view(m, *([1] * (p.grad.dim() - 1))))
    return norm


def evaluate(model: ModelStack, x_eval, y_eval, record: list[int], chunk: int) -> dict[int, torch.Tensor]:
    """Per-position accuracy [M, N] after each pass count in `record`."""
    m = model.m
    totals: dict[int, torch.Tensor] = {}
    count = x_eval.shape[0]
    for i in range(0, count, chunk):
        xb = x_eval[i:i + chunk].unsqueeze(0).expand(m, -1, -1)
        yb = y_eval[i:i + chunk].unsqueeze(0).expand(m, -1, -1)
        got = model.correct_by_position(xb, yb, record)
        for t, c in got.items():
            totals[t] = totals.get(t, 0) + c.double()
    return {t: (c / count).cpu() for t, c in totals.items()}


def test_passes(arm: str, T: int, max_test: int) -> list[int]:
    if arm != "tied":
        return [T]
    out, t = [], 1
    while t <= max_test:
        out.append(t)
        t *= 2
    if T not in out:
        out.append(T)
    return sorted(out)


def run(args) -> dict:
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(args.threads)
    lower_priority()
    if device.type == "cuda" and args.vram_gb:
        total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(0.95, args.vram_gb / total), device)

    arch = make_arch(args.task, args.arm, args.d, args.T)
    bits = [b for b in args.bits for _ in args.seeds]
    seeds = [s for _ in args.bits for s in args.seeds]
    model = ModelStack(arch, bits, seeds, device)
    m = model.m
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.wd)

    gen = torch.Generator(device=device)
    gen.manual_seed(zlib.crc32(f"{args.task}|{args.arm}|{args.d}|{args.T}|{args.seeds}".encode()))
    x_eval, y_eval = eval_set(args.task, args.eval_count, SEQ, device)
    quick = max(1, args.eval_count // 4)

    log_path = Path(args.out).with_suffix(".log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8")

    def log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    history = []
    t0 = time.time()
    log({"event": "start", "arch": arch.to_dict(), "bits": bits, "seeds": seeds, "args": vars(args)})
    for step in range(args.steps):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = sample(args.task, (m, args.batch), SEQ, gen, device)
        loss = model.train_loss(x, y)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()
        gnorm = clip_per_model(params, m, args.clip)
        opt.step()
        if (step + 1) % args.log_every == 0 or step == 0:
            log({"event": "train", "step": step + 1, "lr": lr, "loss": loss.detach().cpu().tolist(),
                 "gnorm": gnorm.cpu().tolist(), "elapsed": time.time() - t0})
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            acc = evaluate(model, x_eval[:quick], y_eval[:quick], [arch.passes], args.eval_chunk)[arch.passes]
            ns = [n_star(acc[i], 0.9) for i in range(m)]
            history.append({"step": step + 1, "n_star_0.9": ns,
                            "mean_acc": acc.mean(1).tolist()})
            log({"event": "eval", **history[-1], "elapsed": time.time() - t0})
    train_seconds = time.time() - t0

    record = test_passes(args.arm, args.T, args.max_test_passes)
    accs = evaluate(model, x_eval, y_eval, record, args.eval_chunk)
    models = []
    for i in range(m):
        per_t = {}
        for t in record:
            a = accs[t][i]
            per_t[str(t)] = {"acc": [round(float(v), 4) for v in a],
                             "n_star": {str(th): n_star(a, th) for th in THRESHOLDS}}
        W = arch.param_count()
        models.append({
            "bits": bits[i], "seed": seeds[i], "params": W,
            "weight_bits": W * weight_bits(bits[i]),
            "scale_bits": 0 if bits[i] >= 32 else 32 * arch.tensor_count(),
            "final_loss": float(loss[i].detach()),
            "history": [{"step": h["step"], "n_star_0.9": h["n_star_0.9"][i],
                         "mean_acc": h["mean_acc"][i]} for h in history],
            "eval": per_t,
        })
    result = {
        "task": args.task, "arm": args.arm, "d": args.d, "T": args.T,
        "arch": arch.to_dict(), "steps": args.steps, "lr": args.lr, "batch": args.batch,
        "train_seconds": train_seconds, "eval_count": args.eval_count, "seq": SEQ,
        "record": record, "models": models,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    tmp = Path(args.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(result), encoding="utf-8")
    os.replace(tmp, args.out)
    if args.ckpt:
        Path(args.ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"arch": arch.to_dict(), "bits": bits, "seeds": seeds,
                    "state": {k: v.cpu() for k, v in model.state_dict().items()}}, args.ckpt)
    log({"event": "done", "train_seconds": train_seconds,
         "n_star_0.9_at_T": [mm["eval"][str(args.T)]["n_star"]["0.9"] for mm in models]})
    logf.close()
    return result


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--T", type=int, required=True)
    p.add_argument("--bits", type=float, nargs="+", default=[32, 4, 1.58])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-count", type=int, default=2048)
    p.add_argument("--eval-chunk", type=int, default=256)
    p.add_argument("--max-test-passes", type=int, default=128)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--vram-gb", type=float, default=0.0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default="")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
