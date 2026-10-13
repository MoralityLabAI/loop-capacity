"""Run the phase-3 jobs (docs/preregistration_phase3.md): progressive-loss training.

    python -u scripts/run_phase3.py --ckpt-root <checkpoint dir>

Workers claim jobs through lock files, as in run_grid.py.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_grid import claim  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase3"
COMMON = ["--arm", "tied", "--d", "64", "--T", "32", "--bits", "32", "4", "1.58", "--seeds", "1", "2",
          "--lr", "0.003", "--quant-start", "0.5", "--steps", "6000", "--batch", "32",
          "--prog-k", "8", "--prog-alpha", "0.5", "--eval-chunk", "64"]


def jobs(py: str, ckpt_root: Path) -> list[dict]:
    train = [py, "-u", "-m", "loopcap.train"]
    extrap = [py, "-u", str(ROOT / "scripts" / "extrapolate.py")]
    js = []
    for jid, task, window, seq in (("prog_mod7_d64_T32_w4", "mod7", 4, 128),
                                   ("prog_ca110d64_d64_T32_w4", "ca110d64", 4, 64),
                                   ("prog_mod7_d64_T32_w0", "mod7", 0, 128)):
        res = OUT / "runs" / f"{jid}.json"
        ck = ckpt_root / "phase3" / f"{jid}.pt"
        cmds = [train + ["--task", task, "--window", str(window), "--seq", str(seq),
                         "--max-test-passes", "256", *COMMON, "--out", str(res), "--ckpt", str(ck)]]
        final = res
        if task == "mod7":
            final = OUT / f"extrap_{jid}_N512.json"
            cmds.append(extrap + ["--ckpt", str(ck), "--seq", "512",
                                  "--passes", "16", "32", "64", "128", "256", "512", "--out", str(final)])
        js.append({"id": jid, "out": final, "res": res, "cmds": cmds})
    return js


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default=os.environ.get("LOOPCAP_CKPT", str(ROOT / "checkpoints")))
    ap.add_argument("--vram-gb", type=float, default=2.5)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    local = ROOT / "runs_local"
    local.mkdir(exist_ok=True)
    for j in jobs(sys.executable, Path(args.ckpt_root)):
        if Path(j["out"]).exists():
            continue
        if args.dry:
            for c in j["cmds"]:
                print(" ".join(c))
            continue
        lock = local / "locks" / f"p3_{j['id']}.lock"
        if not claim(lock):
            continue
        for attempt in (1, 2):
            print(f"[{time.strftime('%H:%M:%S')}] start {j['id']} (attempt {attempt})", flush=True)
            rc = 0
            with open(local / f"p3_{j['id']}.err", "a", encoding="utf-8") as err:
                err.write(f"=== attempt {attempt} {time.strftime('%H:%M:%S')}\n")
                err.flush()
                for k, c in enumerate(j["cmds"]):
                    if k == 0 and Path(j["res"]).exists():
                        continue  # training already done; only the evaluation is left
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
    print(f"[{time.strftime('%H:%M:%S')}] phase 3 finished", flush=True)


if __name__ == "__main__":
    main()
