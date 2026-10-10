"""Train one job: M models (bit-widths x seeds) of one architecture on one task.

    python -m loopcap.train --task ca110d --arm tied --d 32 --T 8 \
        --bits 32 4 1.58 --seeds 1 2 --steps 4000 --out results/runs/x.json

On CUDA the whole training step (forward, backward, per-model clipping,
AdamW) is captured once as a CUDA graph and replayed. On a GPU shared with
other jobs, per-kernel launch latency, not arithmetic, limits these tiny
models, so this matters more than anything else for speed. Training data is
generated in chunks of many steps for the same reason.
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


def make_arch(task: str, arm: str, d: int, T: int):
    return ARMS[arm](task, d, T)


def set_priority(low: bool) -> None:
    if os.name == "nt":
        import ctypes
        cls = 0x4000 if low else 0x20  # BELOW_NORMAL / NORMAL
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), cls)


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


class DataStream:
    """Fresh training data, generated `chunk` steps at a time."""

    def __init__(self, task, m, batch, seq, gen, device, chunk):
        self.args = (task, m, batch, seq, gen, device)
        self.chunk = chunk
        self.i = chunk
        self.x = self.y = None

    def next(self):
        task, m, batch, seq, gen, device = self.args
        if self.i >= self.chunk:
            x, y = sample(task, (self.chunk * m, batch), seq, gen, device)
            self.x = x.view(self.chunk, m, batch, *x.shape[2:])
            self.y = y.view(self.chunk, m, batch, *y.shape[2:])
            self.i = 0
        self.i += 1
        return self.x[self.i - 1], self.y[self.i - 1]


def evaluate(model: ModelStack, x_eval, y_eval, record: list[int], chunk: int,
             amp: bool = False) -> dict[int, torch.Tensor]:
    """Accuracy per problem size [M, sizes] after each pass count in `record`."""
    m = model.m
    totals: dict[int, torch.Tensor] = {}
    count = x_eval.shape[0]
    denom = count * (y_eval.shape[1] if model.multilabel else 1)
    for i in range(0, count, chunk):
        xb = x_eval[i:i + chunk].unsqueeze(0).expand(m, *([-1] * x_eval.dim()))
        yb = y_eval[i:i + chunk].unsqueeze(0).expand(m, *([-1] * y_eval.dim()))
        with torch.autocast(xb.device.type, dtype=torch.bfloat16, enabled=amp):
            got = model.correct_by_position(xb, yb, record)
        for t, c in got.items():
            totals[t] = totals.get(t, 0) + c.double()
    return {t: (c / denom).cpu() for t, c in totals.items()}


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
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", 0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(args.threads)
    set_priority(args.low_priority)
    if device.type == "cuda" and args.vram_gb:
        total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(0.95, args.vram_gb / total), device)
    use_graph = args.graph and device.type == "cuda"

    arch = make_arch(args.task, args.arm, args.d, args.T)
    bits = [b for b in args.bits for _ in args.seeds]
    seeds = [s for _ in args.bits for s in args.seeds]
    model = ModelStack(arch, bits, seeds, device, per_channel=args.scale == "channel")
    m = model.m
    params = list(model.parameters())
    if use_graph:
        lr_t = torch.tensor(args.lr, device=device)
        opt = torch.optim.AdamW(params, lr=lr_t, betas=(0.9, 0.98), weight_decay=args.wd,
                                capturable=True)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.98), weight_decay=args.wd)

    gen = torch.Generator(device=device)
    gen.manual_seed(zlib.crc32(f"{args.task}|{args.arm}|{args.d}|{args.T}|{args.seeds}".encode()))
    stream = DataStream(args.task, m, args.batch, args.seq, gen, device, args.data_chunk)
    x_eval, y_eval = eval_set(args.task, args.eval_count, args.seq, device)
    quick = max(1, args.eval_count // 4)
    # curriculum over problem size, decided on a separate probe set
    sizes = TASKS[args.task].heads or args.seq
    x_probe, y_probe = eval_set(args.task, args.probe_count, args.seq, device, seed=999)
    horizon = [min(sizes, args.cur_start) if args.curriculum else sizes] * m
    size_mask = torch.zeros(m, sizes, device=device)

    def apply_horizon():
        idx = torch.arange(sizes, device=device)[None, :]
        size_mask.copy_((idx < torch.tensor(horizon, device=device)[:, None]).float())

    apply_horizon()

    log_path = Path(args.out).with_suffix(".log.jsonl")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logf = open(log_path, "w", encoding="utf-8")

    def log(rec):
        logf.write(json.dumps(rec) + "\n")
        logf.flush()

    x_s, y_s = (t.clone() for t in stream.next())

    def train_step():
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.amp, cache_enabled=False):
            loss = model.train_loss(x_s, y_s, final_weight=args.final_weight, size_mask=size_mask)
        loss.sum().backward()
        gnorm = clip_per_model(params, m, args.clip)
        opt.step()
        return loss.detach(), gnorm

    def set_lr(step):
        lr = lr_at(step, args.steps, args.lr, args.warmup)
        if use_graph:
            lr_t.fill_(lr)
        else:
            for g in opt.param_groups:
                g["lr"] = lr
        return lr

    q_start = int(round(args.quant_start * args.steps))
    model.quant.active = q_start == 0

    history = []
    t0 = time.time()
    log({"event": "start", "arch": arch.to_dict(), "bits": bits, "seeds": seeds, "args": vars(args)})
    graph = None
    loss = gnorm = None
    warm_until = 3 if use_graph else 0
    side = torch.cuda.Stream() if use_graph else None
    for step in range(args.steps):
        if step == q_start and not model.quant.active:
            # switch every model to its own bit-width; the step changes, so recapture
            model.quant.active = True
            graph = None
            warm_until = step + 3 if use_graph else 0
            log({"event": "quantize", "step": step, "elapsed": time.time() - t0})
        lr = set_lr(step)
        if step > 0:
            x, y = stream.next()
            x_s.copy_(x)
            y_s.copy_(y)
        if not use_graph:
            opt.zero_grad(set_to_none=True)
            loss, gnorm = train_step()
        elif step < warm_until:
            # eager warm-up steps on a side stream, as the CUDA-graph recipe requires
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                opt.zero_grad(set_to_none=True)
                loss, gnorm = train_step()
            torch.cuda.current_stream().wait_stream(side)
        else:
            if graph is None:
                graph = torch.cuda.CUDAGraph()
                opt.zero_grad(set_to_none=True)
                with torch.cuda.graph(graph):
                    loss, gnorm = train_step()
            graph.replay()
        if args.curriculum and (step + 1) % args.cur_every == 0 and min(horizon) < sizes:
            acc_p = evaluate(model, x_probe, y_probe, [arch.passes], args.eval_chunk, args.amp)[arch.passes]
            new = [min(sizes, max(horizon[i], n_star(acc_p[i], 0.9) + args.cur_margin)) for i in range(m)]
            if new != horizon:
                horizon = new
                apply_horizon()
        if (step + 1) % args.log_every == 0 or step == 0:
            log({"event": "train", "step": step + 1, "lr": lr, "loss": loss.cpu().tolist(),
                 "gnorm": gnorm.cpu().tolist(), "horizon": horizon, "elapsed": time.time() - t0})
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            acc = evaluate(model, x_eval[:quick], y_eval[:quick], [arch.passes], args.eval_chunk,
                           args.amp)[arch.passes]
            ns = [n_star(acc[i], 0.9) for i in range(m)]
            history.append({"step": step + 1, "n_star_0.9": ns, "mean_acc": acc.mean(1).tolist()})
            log({"event": "eval", **history[-1], "elapsed": time.time() - t0})
    train_seconds = time.time() - t0

    record = test_passes(args.arm, args.T, args.max_test_passes)
    accs = evaluate(model, x_eval, y_eval, record, args.eval_chunk, args.amp)
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
            "scale_bits": 0 if bits[i] >= 32 else 32 * arch.scale_count(args.scale == "channel"),
            "final_loss": float(loss[i]),
            "history": [{"step": h["step"], "n_star_0.9": h["n_star_0.9"][i],
                         "mean_acc": h["mean_acc"][i]} for h in history],
            "eval": per_t,
        })
    result = {
        "task": args.task, "arm": args.arm, "d": args.d, "T": args.T,
        "arch": arch.to_dict(), "steps": args.steps, "lr": args.lr, "batch": args.batch,
        "train_seconds": train_seconds, "eval_count": args.eval_count, "seq": args.seq,
        "amp": args.amp, "graph": use_graph, "final_weight": args.final_weight,
        "quant_start": args.quant_start, "curriculum": args.curriculum, "scale": args.scale,
        "final_horizon": horizon,
        "record": record, "models": models,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 2 ** 30) if device.type == "cuda" else 0.0,
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
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--final-weight", type=float, default=0.5,
                   help="loss weight on the final depth; the rest is spread over earlier depths")
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-count", type=int, default=2048)
    p.add_argument("--eval-chunk", type=int, default=256)
    p.add_argument("--max-test-passes", type=int, default=128)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--data-chunk", type=int, default=50, help="training steps of data generated at once")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--vram-gb", type=float, default=0.0)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16 autocast for matmuls/attention; residual stream stays fp32")
    p.add_argument("--graph", action=argparse.BooleanOptionalAction, default=True,
                   help="capture the training step as a CUDA graph")
    p.add_argument("--low-priority", action="store_true")
    p.add_argument("--curriculum", action=argparse.BooleanOptionalAction, default=True,
                   help="loss covers problem sizes <= horizon; horizon = max(horizon, n*_probe + margin)")
    p.add_argument("--cur-start", type=int, default=4)
    p.add_argument("--cur-margin", type=int, default=8)
    p.add_argument("--cur-every", type=int, default=100)
    p.add_argument("--probe-count", type=int, default=256)
    p.add_argument("--scale", choices=["tensor", "channel"], default="tensor",
                   help="quantization scale granularity")
    p.add_argument("--quant-start", type=float, default=0.0,
                   help="fraction of steps trained in float before each model's quantizer turns on")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.add_argument("--ckpt", default="")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
