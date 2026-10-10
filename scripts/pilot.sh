#!/usr/bin/env bash
# Pilot v2 (Amendments 1-2): TIED d=32 T=8, b in {32, 1.58}, float warm-start for half the steps.
PY=${PY:-/c/Users/patri/.venvs/bluebeam/Scripts/python.exe}
mkdir -p results/pilot2
for spec in "ca110d 0.003" "s5 0.001" "s5 0.003" "mod7 0.001" "mod7 0.003"; do
  set -- $spec; task=$1; lr=$2
  seq=128; [ $task = ca110d ] && seq=64
  out=results/pilot2/${task}_tied_d32_T8_lr${lr}_qs.json
  [ -f "$out" ] && continue
  "$PY" -u -m loopcap.train --task $task --arm tied --d 32 --T 8 --bits 32 1.58 --seeds 1 \
    --steps 3000 --lr $lr --batch 32 --seq $seq --quant-start 0.5 --eval-every 250 --vram-gb 1.4 --out "$out"
done
