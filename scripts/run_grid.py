"""Run the pre-registered grid, one job per subprocess, skipping finished jobs.

    python scripts/run_grid.py --steps 4000 --batch 32 --lr 1e-3 [--only tied] [--dry]

Launch detached on a shared GPU with
    vramorama run --vram 3G --owner loopcap-grid --detach --out runs_local/grid.out -- \
        python -u scripts/run_grid.py ...
Each job writes results/runs/<job>.json (atomically) and <job>.log.jsonl.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TS = [1, 2, 4, 8, 16, 32]
BITS = ["32", "4", "1.58"]
CKPT_ROOT = Path(os.environ.get("LOOPCAP_CKPT", r"C:\Users\patri\Documents\Codex\loop_capacity\ckpt"))


def pid_alive(pid: int) -> bool:
    """Liveness check that never signals the process (os.kill on Windows terminates)."""
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ok = k.GetExitCodeProcess(h, ctypes.byref(code))
    k.CloseHandle(h)
    return bool(ok) and code.value == 259  # STILL_ACTIVE


def claim(lock: Path) -> bool:
    """Atomically claim a job; reclaim locks whose owner died."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(2):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                owner = int(lock.read_text().strip() or 0)
            except (OSError, ValueError):
                owner = 0
            if owner and pid_alive(owner):
                return False
            try:
                lock.unlink()
            except OSError:
                return False
    return False


def job_id(task, arm, d, T, steps_tag=""):
    return f"{task}_{arm}_d{d}_T{T}{steps_tag}"


def grid(include_mod7: bool = True):
    jobs = []
    # 1) primary: TIED on both main families, cheap T first
    for T in TS:
        for task in ("ca110", "s5"):
            for d in (32, 64):
                jobs.append(dict(task=task, arm="tied", d=d, T=T, bits=BITS))
    # 2) UNTIED-W (same parameter count)
    for T in TS[1:]:
        for task in ("ca110", "s5"):
            for d in (32, 64):
                jobs.append(dict(task=task, arm="untied_w", d=d, T=T, bits=BITS))
    # 3) UNTIED-C (same compute), d = 32
    for T in TS[1:]:
        for task in ("ca110", "s5"):
            jobs.append(dict(task=task, arm="untied_c", d=32, T=T, bits=BITS))
    # 4) contrast family, reduced grid
    if include_mod7:
        for T in TS:
            jobs.append(dict(task="mod7", arm="tied", d=32, T=T, bits=["32", "1.58"]))
        for T in TS[1:]:
            jobs.append(dict(task="mod7", arm="untied_w", d=32, T=T, bits=["32", "1.58"]))
    return jobs


def extra_long(steps_mult: int):
    """Training-limited check: 3x steps on a few TIED cells."""
    return [dict(task=task, arm="tied", d=32, T=T, bits=BITS, mult=steps_mult)
            for task in ("ca110", "s5") for T in (8, 32)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seeds", nargs="+", default=["1", "2"])
    ap.add_argument("--only", default="", help="comma list of arms or tasks to keep")
    ap.add_argument("--long", type=int, default=0, help="also run the Nx-steps check")
    ap.add_argument("--no-mod7", action="store_true")
    ap.add_argument("--vram-gb", type=float, default=2.8)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    out_dir = ROOT / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    local = ROOT / "runs_local"
    local.mkdir(exist_ok=True)
    jobs = grid(not args.no_mod7)
    if args.long:
        jobs += extra_long(args.long)
    if args.only:
        keep = set(args.only.split(","))
        jobs = [j for j in jobs if j["arm"] in keep or j["task"] in keep]

    for j in jobs:
        mult = j.get("mult", 1)
        jid = job_id(j["task"], j["arm"], j["d"], j["T"], f"_x{mult}" if mult > 1 else "")
        out = out_dir / f"{jid}.json"
        if out.exists():
            continue
        lock = local / "locks" / f"{jid}.lock"
        if not args.dry and not claim(lock):
            continue
        cmd = [py, "-u", "-m", "loopcap.train", "--task", j["task"], "--arm", j["arm"],
               "--d", str(j["d"]), "--T", str(j["T"]), "--bits", *j["bits"], "--seeds", *args.seeds,
               "--steps", str(args.steps * mult), "--batch", str(args.batch), "--lr", str(args.lr),
               "--vram-gb", str(args.vram_gb), "--out", str(out),
               "--ckpt", str(CKPT_ROOT / f"{jid}.pt")]
        if args.dry:
            print(" ".join(cmd))
            continue
        for attempt in (1, 2):
            (local / f"grid_heartbeat_{os.getpid()}.json").write_text(json.dumps(
                {"job": jid, "attempt": attempt, "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
            print(f"[{time.strftime('%H:%M:%S')}] start {jid} (attempt {attempt})", flush=True)
            with open(local / f"{jid}.err", "w", encoding="utf-8") as err:
                rc = subprocess.call(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=err)
            print(f"[{time.strftime('%H:%M:%S')}] end {jid} rc={rc}", flush=True)
            if rc == 0 and out.exists():
                break
            time.sleep(30)
        lock.unlink(missing_ok=True)
    print(f"[{time.strftime('%H:%M:%S')}] grid finished", flush=True)


if __name__ == "__main__":
    main()
