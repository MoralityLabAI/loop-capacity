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

(none yet)
