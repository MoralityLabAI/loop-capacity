# loop-capacity

Can a weight-tied network that runs more passes solve larger instances of a
fixed-rule problem? Or does the size it can solve level off at a level set by
its weight bits W·b?

- [`docs/note.md`](docs/note.md): the note. It states claims A (counting) and
  B (computation), with proofs and citations. It also covers the experiment,
  the results, and what they imply for extrapolating from parameter counts.
- [`docs/preregistration.md`](docs/preregistration.md): the hypotheses and
  decision rules, committed before any run. Amendments are dated.
- [`results/summary.md`](results/summary.md): the generated tables and
  verdicts. Per-job JSON is in `results/runs/`.
- `figures/`: n\* against T (`nstar_vs_T_*.png`), test-time passes
  (`test_time_*.png`), per-position accuracy (`heatmap_*.png`), n\* against
  W·b, and training curves.

## Layout

| path | what |
|---|---|
| `loopcap/tasks.py` | causal task families: `ca110d` (rule-110 spacetime diagonal, 32 heads), `mod7` (running sum mod 7), `s5` (permutation composition, exploratory), and the original `ca110` (single column, unlearnable in the pilot) |
| `loopcap/model.py` | M independent small transformers trained in one process. Their weights are stacked on a model axis. Arms: tied, untied with matched W, untied with matched compute |
| `loopcap/quant.py` | per-model weight fake-quantization: float, int4 or ternary, with a straight-through estimator |
| `loopcap/train.py` | trains one job (bit-widths × seeds) and evaluates per-position accuracy at several pass counts |
| `loopcap/construct.py` | hand-built weight-tied blocks with ternary weights, showing claim B constructively |
| `scripts/run_grid.py` | the pre-registered grid; resumable |
| `scripts/analyze.py` | decision rules, tables and figures |
| `tests/` | correctness of task generators, the constructions, parameter matching and quantizer levels |

## Running

The code needs Python 3.11+, PyTorch 2.x (CUDA optional), numpy, matplotlib
and pytest.

```bash
python -m pytest
```

Train one job: the tied model, d = 32, 8 passes, three bit-widths × two seeds.

```bash
python -m loopcap.train --task ca110d --arm tied --d 32 --T 8 --bits 32 4 1.58 --seeds 1 2 --steps 6000 --lr 0.003 --seq 64 --quant-start 0.5 --out results/runs/ca110d_tied_d32_T8.json
```

Run the full grid. It skips finished jobs.

```bash
python -u scripts/run_grid.py --steps 6000 --lr 0.003 --quant-start 0.5 --long 3
```

Then build the tables and figures.

```bash
python scripts/analyze.py
```

The authoritative settings are the ones in the preregistration amendments:
lr 3e-3, batch 32, 6,000 steps, half float then quantized, and the
problem-size curriculum.

On a shared RTX 5080 Laptop GPU with CUDA-graph capture:
- ca110d, d = 32, T = 32, 6 models: about 0.22 s per step;
- s5, d = 64, T = 32: about 0.5 s per step.

Several `run_grid.py` workers can share the queue; they claim jobs with lock
files.
