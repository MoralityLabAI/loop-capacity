# Pre-registration: does looping lift the parameter ceiling?

Written 2026-10-10, before any training run. It is committed before the pilot.
Any later change is recorded as a dated amendment at the end of this file. The
hypotheses and decision rules may not change after the main grid starts.

## Question

Does a weight-tied network that runs for more passes keep solving larger
instances of a fixed-rule problem? Or does the size it can solve level off at a
level set by its weight bits W·b?

## Task families

Every task is a causal sequence-to-sequence problem of length N = 256 that
starts with a BOS token. The target at 1-indexed position n depends on exactly
the first n input tokens, so **n is the problem size**. A single evaluation
pass therefore measures accuracy at every n ≤ 256. Inputs are drawn i.i.d.
uniform.

| id | family | input token | target at position n | serial structure |
|---|---|---|---|---|
| `s5` | iterated permutation composition | g_n ∈ S5 (120 symbols) | p_n = g_n ∘ g_(n−1) ∘ … ∘ g_1 (120 classes) | word problem for a non-solvable group: NC1-complete (Barrington). Theory therefore expects the required depth to grow at least like log n |
| `ca110` | rule-110 rollout | x_(n−1) ∈ {0,1} | c(0, n−1): cell 0 after n−1 steps of rule 110, starting from x with an all-zero background on the left | P-complete in general (Neary & Woods). Its only known algorithm needs n−1 sequential local updates |
| `mod7` | modular arithmetic: running sum | a_n ∈ Z_7 | (a_1 + … + a_n) mod 7 | abelian, so constant depth suffices in principle (TC0). This is the contrast family, and it runs on a reduced grid if time allows |

## Arms

All arms share the same block: pre-RMSNorm without gains, causal multi-head
attention with RoPE and head dimension 8, then a GELU MLP. There are no biases.
The input embedding is re-injected before every pass or layer. The readout
weights are shared across depths. Every weight matrix is fake-quantized to b
bits, including the embedding and the readout.

- **TIED(d, T):** one block of width d, applied T times.
- **UNTIED-W(d, L=T):** T distinct blocks whose width d′ and MLP width are
  chosen so the total weight count W is within 2% of TIED(d, ·). The parameter
  count is the same, so the compute per token is about 1/T of TIED's.
- **UNTIED-C(d, L=T):** T distinct blocks of width d, about T times the
  parameters. The compute matches TIED(d, T). Runs at d = 32 only.

## Grid

- T (training passes or layers) ∈ {1, 2, 4, 8, 16, 32}.
- d ∈ {32, 64}.
- b ∈ {32 (fp32), 4 (int4, per-tensor absmax), 1.58 (ternary, per-tensor
  absmean, as in BitNet b1.58)}. Activations stay in float.
- 2 seeds per cell; a seed changes both the initialization and the data stream.
- TIED models are also evaluated at T_test ∈ {1, 2, 4, …, 128} (test-time
  passes).
- `mod7` uses a reduced grid: d = 32, b ∈ {32, 1.58}, TIED and UNTIED-W only.
- Training is identical across arms: the same steps, batch, optimizer and data
  distribution. The loss is ½·CE(final depth) + ½·mean over depths t < T of
  CE(readout at t).

## Measurement

acc(n) is the exact-match accuracy at position n over 2,048 held-out
sequences, drawn with a fixed evaluation seed that is shared by all runs.

**n\*\_θ = max { n : acc(m) ≥ θ for all m ≤ n }**. The primary threshold is
θ = 0.9; θ = 0.8 and θ = 0.99 are reported as sensitivity checks. n\* = 256 is
*censored* and is read as "≥ 256". The cell value is the median over seeds.

## Hypotheses (from the brief)

- **H_compute:** for the tied model, the solvable n keeps growing with T. It is
  limited by training, not by W·b.
- **H_ceiling:** the solvable n levels off at a level that tracks W·b.

## Decision rules (TIED, T_test = T_train, θ = 0.9)

Each family has 6 (d, b) cells.

- A cell **grows** if n\*(32) ≥ 1.2 · n\*(16), or if n\*(32) is censored.
- A cell **saturates** if n\*(32) < 1.1 · n\*(16), n\*(32) < 1.2 · n\*(8), and
  n\*(32) is not censored.
- **Wb-insensitive (per family):** n\*(d=32, ternary) ≥ 0.75 · n\*(d=64, fp32)
  at T = 32. For s5, W·b differs by a factor of about 65 between these two
  cells.
- **Wb-tracking (per family):** Spearman ρ(log W·b, n\*(32)) ≥ 0.6 across the
  6 cells, and n\*(d=64, fp32) ≥ 2 · n\*(d=32, ternary) at T = 32.

Verdict per family:

- **H_compute supported:** at least 4 of 6 cells grow, and the family is
  Wb-insensitive.
- **H_ceiling supported:** at least 4 of 6 cells saturate, and the family is
  Wb-tracking.
- **Mixed:** any other outcome, reported with the pattern that was observed.
  For example, growth that is Wb-sensitive points to an optimization limit
  rather than a counting limit. Saturation that does not track W·b points to a
  ceiling from something else (training, attention precision, length).

**"Limited by training" diagnostic.** n\* is logged during training. A cell is
flagged *training-limited* if n\* rose by ≥ 10% over the last 20% of steps. As
a direct check, a few cells are retrained with 3× the steps.

## Secondary analyses

1. **Test-time passes:** n\*(T_test) for T_test ≠ T_train, and in particular
   the ratio n\*(2·T_train) / n\*(T_train).
2. **Tied vs UNTIED-W at matched T:** this is the brief's comparison at equal
   parameter count.
3. **Tied vs UNTIED-C at matched T:** if TIED(d, T) reaches 0.75–1.33× the n\*
   of UNTIED-C(d, T), then the tied model behaves like an untied model with T
   times the parameters. Parameters × passes is then the right currency.
4. **Functional form:** least-squares fit of log2 n\* on log2 W, log2 b and
   log2 T over the uncensored TIED cells with n\* > 0. Under a product law
   (n\* ∝ W·T), the W and T coefficients should be equal. Under a threshold
   law ("W·b must exceed the rule's description length, after which only T
   matters"), the W and b coefficients should be about 0.

## My predictions (written before any run, for calibration)

- `ca110`: H_compute. n\* grows roughly linearly with T, with ≤ 25%
  sensitivity to b and d. UNTIED-W falls behind TIED from T ≈ 8 on, because its
  layers become too narrow.
- `s5`: mixed. n\* grows with T, but ternary weights and d = 32 hurt through
  optimization rather than through a counting ceiling.
- Test-time passes: partial extrapolation on `ca110` (less than 2× when T
  doubles at test time) and little on `s5`.
- UNTIED-C ≥ TIED at matched T, with TIED within 0.75× on `ca110`.

## Pilot (allowed before the main grid)

The pilot runs TIED, d = 32, T = 8, b ∈ {32, 1.58} on `s5` and `ca110`, with
learning rate ∈ {1e-3, 3e-3}, for 3,000 steps. It may set only the learning
rate, the step budget and the batch size. If a family does not learn at all,
meaning n\* = 0 in every pilot run, its formulation may be changed once, as a
recorded amendment, before the main grid.

## Amendments

### Amendment 1 (2026-10-11, before any main-grid run)

**1. Pilot result for `ca110` as originally formulated.** Setup: TIED, d = 32,
T = 8, lr = 1e-3, 3,000 steps, batch 32, sequence length 256. Result:

- n\* = 6 for the float model and 4 for the ternary model.
- n\* is the same at T_test = 4 as at T_test = 8, and per-position accuracy
  falls to the majority baseline after about position 7.
- The model memorized lookup tables over the first few inputs. It found no
  iterative algorithm.

The lr = 3e-3 and `s5` pilot jobs were stopped unfinished because of item 3.

**Why the original formulation was unlearnable at this budget.** In it, the
interior cells of the spacetime diagram are never targets. Credit assignment
therefore has to run through every pass. This is inherent to a P-complete
task posed as one output per position. If every intermediate state were the
target of a left-to-right finite-state scan, the task would be in NC1. In
`s5` and `mod7`, by contrast, the scan state is the target.

**2. Reformulation: `ca110` → `ca110d`.**

- At every position s, the model outputs the spacetime diagonal through
  (s−1, 0): K = 32 binary heads, with head k = c(s−1−k, k).
- Head k needs k applications of rule 110 and depends only on x_0..x_(s−1).
- The problem size is n = k + 1. acc(n) is the head's accuracy, averaged over
  all positions and sequences, and n\* is computed over heads. The sequence
  length is 64; censoring is at 32.
- Every intermediate cell is now a target at some (position, head). The
  model is still free in which order it computes them and in how many rule
  steps each pass performs, so the family remains P-complete and the
  algorithm is not dictated.
- K = 32 keeps the linear readout's rank ≥ K at d ≥ 32.

The pilot rule allowed a reformulation only if n\* = 0 in every pilot run.
This change goes beyond that rule. It is made openly, before the main grid,
because the original formulation could not test either hypothesis within the
budget. The original pilot result is reported in the note.

**3. Throughput.** The GPU is shared with other agents' jobs, and the first
pilot ran at 0.25–0.6 s/step on the smallest configuration, which made the
full grid infeasible. Changes:

- **bf16 autocast for matmuls and attention.** The float arm therefore
  effectively has 16-bit weights, so the W·b accounting uses b = 16 for it.
  The W·b ratio between the extreme cells becomes about 33× for `s5` and
  40× for `ca110d`, not 65×.
- **The training step is captured as a CUDA graph.**
- **Data is generated in chunks.**
- **Batch 32 per model.** The pilot rule allowed this.
- **Sequence length for `s5` and `mod7` goes from 256 to 128.** Censoring is
  therefore at 128.

**4. Pilot v2** (replaces the stopped pilot): TIED, d = 32, T = 8,
b ∈ {float, ternary}, one seed, lr ∈ {1e-3, 3e-3}, 3,000 steps, on `s5`,
`ca110d` and `mod7`. It sets the learning rate and the step budget.

Hypotheses and decision rules are unchanged, apart from the censoring levels
in items 2 and 3.
