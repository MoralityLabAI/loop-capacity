"""Run the phase-2 jobs (docs/preregistration_phase2.md), skipping finished ones.

    python -u scripts/run_phase2.py --ckpt-root <dir with phase-1 checkpoints>

Several workers may run at once; they claim jobs through the same lock files
as run_grid.py. Each job is a list of commands; it counts as done once its
final output exists.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grid import claim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase2"
BITS = ["32", "4", "1.58"]
COMMON = ["--bits", *BITS, "--seeds", "1", "2", "--lr", "0.003", "--quant-start", "0.5"]


def jobs(py: str, ckpt_root: Path) -> list[dict]:
    train = [py, "-u", "-m", "loopcap.train"]
    extrap = [py, "-u", str(ROOT / "scripts" / "extrapolate.py")]
    js = []
    # Q2(i): phase-1 global-attention mod7 checkpoints at N = 512
    for name in ("mod7_tied_d64_T32", "mod7_tied_d32_T32", "mod7_tied_d64_T16", "mod7_tied_d32_T16"):
        out = OUT / f"extrap_{name}_N512.json"
        js.append({"id": f"extrap_{name}", "out": out, "cmds": [
            extrap + ["--ckpt", str(ckpt_root / f"{name}.pt"), "--seq", "512",
                      "--passes", "16", "32", "64", "128", "256", "512", "--out", str(out)]]})
    # Q2(ii): local attention (window 4), and the global control for ca110d64
    for task, window, seq, maxp in (("mod7", 4, 128, 512), ("ca110d64", 4, 64, 256), ("ca110d64", 0, 64, 256)):
        jid = f"{task}_tied_d64_T32_w{window}"
        res = OUT / "runs" / f"{jid}.json"
        ck = ckpt_root / "phase2" / f"{jid}.pt"
        cmds = [train + ["--task", task, "--arm", "tied", "--d", "64", "--T", "32", "--window", str(window),
                         "--seq", str(seq), "--steps", "6000", "--batch", "32",
                         "--max-test-passes", str(min(maxp, 256)), *COMMON,
                         "--out", str(res), "--ckpt", str(ck)]]
        final = res
        if task == "mod7":
            final = OUT / f"extrap_{jid}_N512.json"
            cmds.append(extrap + ["--ckpt", str(ck), "--seq", "512",
                                  "--passes", "16", "32", "64", "128", "256", "512", "--out", str(final)])
        js.append({"id": jid, "out": final, "cmds": cmds, "skip_first_if": res})
    # Q1: memorization capacity against passes, largest first. mem16 (the original
    # design, amendment 1) gave no signal; memrand replaces it.
    mem = [(32, 16), (16, 32), (16, 16), (32, 4), (16, 8), (32, 1), (16, 4), (16, 2), (16, 1)]
    for task, seq in (("mem16", "16"), ("memrand", "1")):
        for d, T in mem:
            jid = f"{task}_tied_d{d}_T{T}"
            res = OUT / "runs" / f"{jid}.json"
            js.append({"id": jid, "out": res, "cmds": [
                train + ["--task", task, "--arm", "tied", "--d", str(d), "--T", str(T), "--seq", seq,
                         *(["--steps", "20000", "--batch", "128", "--eval-every", "2000"] if task == "mem16"
                           else ["--steps", "60000", "--batch", "1024", "--eval-every", "6000"]), *COMMON,
                         "--out", str(res), "--ckpt", str(ckpt_root / "phase2" / f"{jid}.pt")]]})
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=os.environ.get("LOOPCAP_CKPT", str(ROOT / "checkpoints")))
    ap.add_argument("--vram-gb", type=float, default=1.4)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    ckpt_root = Path(args.ckpt_root)
    local = ROOT / "runs_local"
    local.mkdir(exist_ok=True)
    for j in jobs(sys.executable, ckpt_root):
        if Path(j["out"]).exists():
            continue
        lock = local / "locks" / f"p2_{j['id']}.lock"
        if args.dry:
            for c in j["cmds"]:
                print(" ".join(c))
            continue
        if not claim(lock):
            continue
        for attempt in (1, 2):
            print(f"[{time.strftime('%H:%M:%S')}] start {j['id']} (attempt {attempt})", flush=True)
            rc = 0
            with open(local / f"p2_{j['id']}.err", "w", encoding="utf-8") as err:
                for k, c in enumerate(j["cmds"]):
                    if k == 0 and j.get("skip_first_if") and Path(j["skip_first_if"]).exists():
                        continue
                    if "loopcap.train" in " ".join(c):
                        c = c + ["--vram-gb", str(args.vram_gb)]
                    rc = subprocess.call(c, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=err)
                    if rc != 0:
                        break
            print(f"[{time.strftime('%H:%M:%S')}] end {j['id']} rc={rc}", flush=True)
            if rc == 0 and Path(j["out"]).exists():
                break
            time.sleep(30)
        lock.unlink(missing_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] phase 2 finished", flush=True)


if __name__ == "__main__":
    main()
