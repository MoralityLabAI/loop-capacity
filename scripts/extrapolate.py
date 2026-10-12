"""Re-evaluate saved tied models at a longer sequence length and more passes.

    python scripts/extrapolate.py --ckpt path/to/mod7_tied_d64_T32.pt --seq 512 \
        --passes 16 32 64 128 256 512 --out results/phase2/extrap_mod7_tied_d64_T32_N512.json

The evaluation set is drawn with its own seed (default 777) so it is disjoint
in practice from the training-time evaluation set.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopcap.metrics import THRESHOLDS, n_star  # noqa: E402
from loopcap.model import ArchSpec, ModelStack  # noqa: E402
from loopcap.tasks import TASKS, eval_set  # noqa: E402
from loopcap.train import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--passes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--eval-count", type=int, default=512)
    ap.add_argument("--eval-seed", type=int, default=777)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--vram-gb", type=float, default=1.4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    if device.type == "cuda" and args.vram_gb:
        total = torch.cuda.get_device_properties(device).total_memory / 2 ** 30
        torch.cuda.set_per_process_memory_fraction(min(0.95, args.vram_gb / total), device)
    torch.backends.cuda.matmul.allow_tf32 = True

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    arch = ArchSpec(**{k: v for k, v in ck["arch"].items() if k != "params"})
    if arch.n_blocks > 1:
        raise SystemExit("only tied models can run more passes than they were trained with")
    model = ModelStack(arch, ck["bits"], ck["seeds"], device)
    model.load_state_dict({k: v.to(device) for k, v in ck["state"].items()})
    model.quant.active = True

    x, y = eval_set(arch.task, args.eval_count, args.seq, device, seed=args.eval_seed)
    t0 = time.time()
    accs = evaluate(model, x, y, args.passes, args.chunk, amp=True)
    out = {"ckpt": os.path.basename(args.ckpt), "arch": ck["arch"], "seq": args.seq,
           "eval_count": args.eval_count, "eval_seed": args.eval_seed, "passes": args.passes,
           "seconds": time.time() - t0, "models": []}
    for i, (b, s) in enumerate(zip(ck["bits"], ck["seeds"])):
        per = {}
        for t in args.passes:
            a = accs[t][i]
            per[str(t)] = {"acc": [round(float(v), 4) for v in a],
                           "n_star": {str(th): n_star(a, th) for th in THRESHOLDS}}
        out["models"].append({"bits": b, "seed": s, "eval": per})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(args.out).with_suffix(".tmp")
    tmp.write_text(json.dumps(out), encoding="utf-8")
    os.replace(tmp, args.out)
    print(json.dumps({m["bits"]: {t: m["eval"][str(t)]["n_star"]["0.9"] for t in args.passes}
                      for m in out["models"]}))


if __name__ == "__main__":
    main()
