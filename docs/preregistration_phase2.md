# Pre-registration, phase 2: measuring claim A, and what limits extrapolation

Written 2026-10-12, before any phase-2 run. It is committed before the code
that runs it executes. Later changes are dated amendments at the end.

Phase 1 (`docs/note.md`) left two things open:

1. **Claim A was never measured.** Phase 1 measured serial reach, which is
   claim B's notion of capability. Claim A concerns stored information. Its
   bound is hard (bits ≤ W·b + log2 T), and it predicts that passes add
   almost nothing to it.
2. **Learned scans stopped extrapolating.** `mod7` models trained at T = 32
   ran as scans and kept improving past their training depth: d = 64
   ternary went from 34.5 to 61 at 64 passes. But they stalled at about 60
   even at 128 passes. One candidate cause is attention. With global softmax
   attention, picking out one neighbour among n positions needs a score
   margin of about ln n, so a learned scan may lose sharpness as sequences
   grow (Remark 1 of the note). Theorem B's construction uses local mixing,
   which has no such dilution.

## Q1: memorization capacity against passes (claim A)

**Task `mem16`.**

- The input is BOS followed by a 16-bit key, one bit per token.
- The target is a fixed random label bit for that key, read at the last
  position only.
- The table covers all 2^16 = 65,536 keys, with labels drawn once from seed
  4242. It is a uniformly random Boolean function on 16 inputs, so its
  description length is 65,536 bits (corollary A2).

**Training.** Batches sample keys uniformly. The loss is on the last position
only, with the phase-1 depth weighting (½ final depth, ½ mean of earlier
depths). There is no curriculum.

- Optimizer: AdamW, lr 3e-3, batch 128 per model, 20,000 steps.
- Float warm-start as in phase 1: the first half of the steps in float, then
  each model's quantizer switches on.
- Embedding vocabulary is 3 tokens and the readout has 2 classes, so almost
  all parameters sit in the block.

**Measurement.** Stored bits = Σ over all 65,536 keys of (1 − CE_k / ln 2),
where CE_k is the cross-entropy in nats at the last position after T passes.
This is the label information minus the model's residual uncertainty, in the
style of Morris et al. Accuracy is reported too.

**Grid.** TIED, b ∈ {float (b = 16), int4, ternary}, 2 seeds:

- d = 16 (W = 3,152) at T ∈ {1, 2, 4, 8, 16, 32};
- d = 32 (W = 12,448) at T ∈ {1, 4, 16}.

**Hypotheses and rules.**

- **A-bound (hard).** Every model's stored bits ≤ W·b + (fp32 scale bits) +
  log2 T. Any violation beyond measurement noise falsifies the accounting,
  and its cause must be identified. Measurement noise means 2 standard
  errors of the per-key estimate, summed over the 65,536 keys.
- **A-flat vs A-grow, per (d, b) cell, median of seeds.**
  - *Flat* if stored bits at the largest T are within ±25% of those at
    T = 4.
  - *Grows* if they are ≥ 1.5× the T = 4 value.
  - Anything else is *in between*.
  - A-grow would not contradict claim A, because the bound is far above
    observed capacity. It would mean passes improve how efficiently
    parameters are used for storage.
- **Saturation check.** Stored bits are logged every 2,000 steps. A model
  counts as saturated if its last two measurements differ by less than 5%.
  Unsaturated cells are reported as lower bounds.

**My prediction.**

- A-bound holds everywhere, comfortably for float. It is tightest for
  ternary, where I expect below 1 bit per parameter.
- A-flat holds from T = 4 upward.
- T = 1 may store less than T ≥ 4, because a single layer is a poorer
  memorizer at equal W.

## Q2: does local token mixing remove the extrapolation ceiling? (claim B)

**(i) Re-evaluation (no training).** The phase-1 TIED checkpoints of `mod7`
at T ∈ {16, 32} (d ∈ {32, 64}, all bit-widths, both seeds) are evaluated on
512-token sequences (fixed seed 777) at T_test ∈ {16, 32, 64, 128, 256, 512}.

**(ii) Local attention, new training.** These runs are identical to phase 1
(T = 32, lr 3e-3, 6,000 steps, curriculum, float warm-start, three
bit-widths × two seeds), except that attention is restricted to a causal
window of 4 positions:

- `mod7`, d = 64, local (sequence length 128 in training);
- `ca110d64`, d = 64, local. This is the diagonal rule-110 task with
  K = 64 heads, so n\* can reach 64.
- `ca110d64`, d = 64, global attention, as the control.

All are evaluated at T_test ∈ {16, 32, 64, 128, 256}, plus 512 for `mod7`.
`mod7` is evaluated on 512-token sequences, `ca110d64` on the training
length (sizes are heads).

**Hypotheses and rules** (median over bit-widths and seeds, θ = 0.9):

- **Ext-local.** A local-attention model *extrapolates* if
  n\*(T_test = 128) ≥ 2 · n\*(T_test = 32). This is judged separately for
  `mod7` and `ca110d64`.
- **Ext-global.** The phase-1 global `mod7` T = 32 checkpoints *plateau* if
  n\*(T_test = 128, N = 512) < 1.25 · n\*(T_test = 32, N = 512). The
  `ca110d64` global control is judged by the same rule.
- **Attribution.** If local extrapolates and global plateaus on the same
  task, the ceiling is attributed to attention dilution. If both plateau,
  the cause is elsewhere: the training horizon or the learned algorithm.

**My prediction.**

- `mod7`: local extrapolates, n\* close to T_test until it saturates; global
  plateaus near 60.
- `ca110d64`: local extrapolates only partly; global plateaus, as the
  K = 32 models did in phase 1.

## Order and budget

Q2(i) runs first, because it is evaluation only. Then Q2(ii) and Q1 run on
up to three grid workers on the shared GPU. Nothing in phase 1 is changed or
re-analysed.

## Amendments

(none yet)

### Amendment 1 (2026-10-13, after the first Q1 run, before any redesigned Q1 run)

**The Q1 design produced no signal.** Every one of the 54 `mem16` models
stored about 0 bits: final accuracy was 0.50 and the stored-bit estimates
fell between −72 and +5. No bound was violated, but only trivially, and the
flat/grow ratio is undefined.

The design caused this. A key spelled out as 16 binary tokens, with a random
label, is a random truth table, and it has no low-order structure for
gradient descent to follow. Each unit sees a 50/50 label split whatever it
computes. The capacity results this was modelled on (Allen-Zhu & Li; Morris
et al.) give keys their own embeddings.

**Redesign: task `memrand`.**

- **Keys.** There are N = 65,536 keys. Each is a single token, whose
  embedding is a fixed random Gaussian vector (seed 5150) shared by all
  models.
- **Embeddings are input data.** They are frozen, unquantized, and not
  counted in W. W therefore counts only the block and the readout.
- **Labels and sequences.** Each label is a random bit (seed 4243). A
  sequence is BOS followed by the key token, and the label is read at the
  key position.
- **Lineage.** This is the classical capacity setting of random inputs with
  random labels (Cover; Baum). The grid, training settings, measurement and
  decision rules are those of Q1 above, with `memrand` in place of `mem16`.

**Prediction.**

- A-bound holds everywhere.
- Float stores well under 16 bits per parameter. Ternary stays under 1.58.
- Stored bits may rise from T = 1 to T = 4. A deeper effective network
  realizes more functions with the same real-valued weights (Bartlett et
  al.), so this would not contradict claim A. I expect them to be flat
  beyond T = 4.

**Q2 results as they stood when this amendment was written** (reported in
full in the note):

- Local attention did *not* lift the plateau. mod7 d = 64 local reached a
  median n\* of 59.5 at every pass count from 64 to 512, against 51 for
  global.
- Neither `ca110d64` variant extrapolated.
- Two of the six local mod7 models generalized well past their training
  horizon, to n\* = 107 and 126. In those two, accuracy decays gradually
  with distance and has no cliff at the 128-token training length.

### Amendment 2 (2026-10-13, before any `memrand` grid run)

**Training budget for `memrand`.**

- **Smoke tests** (d = 16, T = 4, float, outside the grid):
  - batch 128 × 3,000 steps (6 epochs) stored about 70 bits;
  - batch 1,024 × 20,000 steps (312 epochs) stored 4,334 and 3,752 bits,
    about 1.3 bits per parameter.
- **The change.** The pre-registered 20,000 steps at batch 128 would leave
  the models far from saturation, so `memrand` runs batch 1,024 per model
  for 60,000 steps, about 940 epochs. Stored bits are logged every 6,000
  steps for the saturation check. Everything else is unchanged.

**Bug found and fixed.**

- **What went wrong.** A text patch applied during an interrupted session
  (`loopcap/train.py`, Q2 code) moved the per-step logging and the periodic
  evaluation out of the training loop. As a result, the phase-2 runs before
  this amendment have only the final train log and a single end-of-training
  evaluation.
- **What it did not affect.** Training, the curriculum and the final
  evaluations ran correctly.
- **What is lost.**
  - Mid-training history for those runs is lost.
  - `mem16`'s saturation check therefore cannot be computed. That run gave
    no signal anyway.
  - Phase 1 used older code and is unaffected.
- **The fix.** The code is fixed, and a regression test
  (`tests/test_model.py::test_training_logs_and_evaluates_inside_the_loop`)
  pins the behaviour.
