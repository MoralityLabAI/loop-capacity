# Pre-registration, phase 3: does a depth-stability objective make learned scans extrapolate?

Written 2026-10-13, before any phase-3 run, and committed before the runs
start.

## Why

Phase 2 found that learned tied models stop extending their reach soon after
their training depth.

- Most of them stop near the largest problem size their training loss
  covered.
- The rule-110 models (`ca110d64`) collapse when run past about 64 passes.
- Local attention did not change this.

Bansal et al. (NeurIPS 2022) made recurrent convolutional networks
extrapolate by running more iterations at test time. They did it with input
recall, which this model already has (the input is re-injected each pass),
plus a *progressive loss*. That loss trains the network to improve from
arbitrary intermediate states, so its behaviour does not depend on the
iteration number.

## Recipe

On top of the phase-2 training (deep supervision, curriculum, float
warm-start), each step adds a progressive term:

1. Take the detached hidden state after n passes, where n is uniform on
   {0, …, 32} and drawn fresh each step.
2. Run k = 8 more passes from it, with gradients.
3. Take the cross-entropy at the end of that segment.

The loss is ½ (phase-2 loss) + ½ (progressive term). A segment can end at
depth 40, beyond the 32 training passes, so the term also rewards staying
correct past the training depth. This differs from Bansal et al., who keep
n + k at or below the training depth.

The implementation keeps the CUDA graph static: n is a one-element index
tensor (`loopcap/model.py`, `train_loss`).

## Arms

All arms: tied, d = 64, trained with T = 32 passes, three bit-widths × two
seeds, lr 3e-3, 6,000 steps, curriculum, half float then quantized.

| arm | task | attention | baseline (existing, same recipe minus the progressive term) |
|---|---|---|---|
| P-mod7-local | `mod7`, training length 128 | window 4 | phase 2 `mod7_tied_d64_T32_w4` |
| P-mod7-global | `mod7`, training length 128 | full | phase 1 `mod7_tied_d64_T32` (re-evaluated in phase 2) |
| P-ca-local | `ca110d64` | window 4 | phase 2 `ca110d64_tied_d64_T32_w4` |

## Measurement

- `mod7`: n\* on 512-token sequences (seed 777) at T_test ∈ {32, 64, 128,
  256, 512}.
- `ca110d64`: n\* over its 64 heads at T_test ∈ {32, 64, 128, 256}.

Each value is the median over bit-widths and seeds, at θ = 0.9.

## Decision rules

- **Stable** if n\*(T_test) ≥ 0.9 · n\*(32) for every T_test above 32.
- **Extrapolates** if n\*(128) ≥ 2 · n\*(32). This is the same rule as phase 2.
- **Effect against the baseline, at T_test = 128:**
  - *helps* if the progressive / baseline ratio is ≥ 1.5;
  - *hurts* if it is ≤ 1/1.5;
  - otherwise *no clear effect*.
- **Coverage.** For each arm, report the final training horizon next to the
  plateau.

## My predictions

- **All three arms are stable.** In particular, `ca110d64` stops collapsing,
  so on it the progressive loss *helps*. That baseline's n\* at 128 passes is
  only 2.
- **`mod7` does not extrapolate past twice its 32-pass value.** Coverage
  still binds, and its plateau stays near the training horizon. The
  comparison against baseline will show *no clear effect* for both `mod7`
  arms.

## Amendments

(none yet)
