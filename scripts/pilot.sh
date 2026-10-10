#!/usr/bin/env bash
# Pilot v2 (Amendment 1): TIED d=32 T=8, b in {32, 1.58}, lr in {1e-3, 3e-3}, 3000 steps.
PY=${PY:-/c/Users/patri/.venvs/bluebeam/Scripts/python.exe}
mkdir -p results/pilot2
for task in ca110d s5 mod7; do
  seq=128; [ $task = ca110d ] && seq=64
  for lr in 0.001 0.003; do
    out=results/pilot2/${task}_tied_d32_T8_lr${lr}.json
    [ -f "$out" ] && continue
    "$PY" -u -m loopcap.train --task $task --arm tied --d 32 --T 8 --bits 32 1.58 --seeds 1 \
      --steps 3000 --lr $lr --batch 32 --seq $seq --eval-every 250 --vram-gb 2.0 --out "$out"
  done
done
