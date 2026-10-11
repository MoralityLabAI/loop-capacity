#!/usr/bin/env bash
# Per-step time for a few configs, CUDA graph on/off.
PY=${PY:-python}
mkdir -p runs_local/bench
run() { # name, extra args
  "$PY" -u -m loopcap.train --steps 60 --log-every 20 --eval-every 1000 --eval-count 256 --vram-gb 2.4 \
    --out runs_local/bench/$1.json "${@:2}" > runs_local/bench/$1.out 2>&1
  "$PY" - "$1" <<'PYEOF'
import json, sys
name = sys.argv[1]
try:
    r = [json.loads(l) for l in open(f"runs_local/bench/{name}.log.jsonl") if '"train"' in l]
    e = [x["elapsed"] for x in r]
    print(f"{name}: {(e[-1] - e[1]) / (r[-1]['step'] - r[1]['step']):.3f} s/step")
except Exception as ex:
    print(name, "FAILED", ex, open(f"runs_local/bench/{name}.out").read()[-800:])
PYEOF
}
run s5_d64_T32_graph   --task s5 --arm tied --d 64 --T 32 --seq 128
run s5_d64_T32_eager   --task s5 --arm tied --d 64 --T 32 --seq 128 --no-graph
run cad_d32_T8_graph   --task ca110d --arm tied --d 32 --T 8 --seq 64
run cad_d32_T32_graph  --task ca110d --arm tied --d 32 --T 32 --seq 64
run cad_d32_T32_b128   --task ca110d --arm tied --d 32 --T 32 --seq 64 --batch 128
run cad_uc_d32_T32     --task ca110d --arm untied_c --d 32 --T 32 --seq 64
