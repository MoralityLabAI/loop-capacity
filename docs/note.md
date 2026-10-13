# Does looping lift the parameter ceiling?

*Capability per parameter in weight-tied networks: a theory note and an experiment.*

## Summary

**The two claims are both true. They use different notions of capability.**

- **Claim A (counting).** W weights of b bits realize at most 2^{Wb}
  functions, at any depth and any number of loops. A single weight vector
  fixes the whole family f_T, so looping adds at most log2 T bits. This caps
  *how many* targets a network can hold, equivalently its description length:
  K(f) ≲ Wb + O(log T).
- **Claim B (computation).** A weight-tied block with poly(ℓ) ternary weights
  runs any ℓ-bit program for any number of steps T. This caps nothing about
  *how much computation* the held targets may need.
- **Parameter bits pay for description length; passes pay for running time.**
  The repository checks B constructively. An 84-weight ternary rule-110 block
  and a ternary S5 block solve instances of size T (rule 110) or T + 1 (S5)
  for every T.

**In learned models, neither pre-registered hypothesis passes cleanly.** The
verdict is *mixed* for both families.

- **Rule-110 rollout (`ca110d`).** Reach grows with passes, then levels off.
  The plateau heights are ordered by W·b, but they sit two or more orders of
  magnitude below what the bits could represent. They behave like learning
  ceilings: 3× training steps lift float reach by about 30% (and roughly
  double it on `mod7`), and the precision gap opens at the moment of
  quantization.
- **Running sum mod 7 (`mod7`).** Quantized models trained at T = 32 learned
  a sequential scan. At d = 64, ternary and float converge to the same reach
  (n\* ≈ 35). Running the same weights for 64 passes at test time extends a
  ternary model from 34.5 to 61, with no new parameters. Here passes
  substitute for weight bits outright.
- **Matched-W and matched-compute arms.**
  - A tied model beats an untied model *with the same parameter count* once
    T ≥ 8.
  - It falls short of an untied model *with the same compute*
    (T× the parameters): 12.5 vs 23.5 at T = 32 on `ca110d`.
  - Across all arms, n\* depends on parameters and on parameters × passes
    with comparable exponents, plus weight precision.

**Phase 2** (§6) measured both claims directly.

- **Claim A.** Memorizing random labels, tied models store about 1.8 bits
  per parameter in float, far under the 16-bit bound. Passes leave storage
  flat from T = 4 to T = 32.
- **Claim B.** Extra passes at test time help only out to the problem sizes
  the training loss covered. Two local-attention scans went further, and
  stopped where per-step errors compounded. Local attention did not lift the
  plateau.

**Phase 3** (§7) added a depth-stability ("progressive") loss.

- **Float `mod7` scans with local attention** became stable out to 512 passes
  and reached n\* ≈ 90–98, well past their training horizons.
- **Ternary models got worse.**
- **Rule-110 models still collapse** past about 64 passes.
- **Verdict:** stable only for local `mod7`, and no arm extrapolates by the
  pre-registered 2× rule.

**Implication** (§4): a parameter count bounds memorization, not serial
reach. For looped models track parameters × passes *and* parameters, and
treat both as predictive only once the training recipe demonstrably finds the
iterative algorithm.

Deliverables:

- this note;
- the code in `loopcap/`;
- the pre-registration with dated amendments, `docs/preregistration.md`;
- per-job results in `results/runs/`, and the tables in `results/summary.md`;
- figures in `figures/`.

The plots of solvable n against T are `figures/nstar_vs_T_ca110d.png` and
`figures/nstar_vs_T_mod7.png`. Test-time passes are in
`figures/test_time_*.png`, and the compute and parameter view is
`figures/nstar_vs_compute.png`.


---

## Part 1. Two claims, two notions of capability

**Setting.** An architecture 𝒜 maps a weight vector θ and a pass count T to a
function f_{θ,T}: X → Y. For a weight-tied network,
f_{θ,T} = R_θ ∘ F_θ^T ∘ E_θ. Here E is the embedding, F the shared block
(with the input re-injected each pass), and R the readout. The weights are
*b-bit*: there are W of them, and each takes one of at most 2^b values. This
covers any b-bit number format, floats included, because a b-bit format has at
most 2^b values.

### Claim A: counting

**Theorem A.** Fix an architecture with W weights of b bits each.

1. For every T, the set F_T = { f_{θ,T} : θ } has at most 2^{Wb} elements.
2. The set of whole families { (f_{θ,T})_{T ≥ 1} : θ } also has at most
   2^{Wb} elements.
3. ⋃_{T ≤ T_max} F_T has at most T_max · 2^{Wb} elements.

*Proof.* θ ↦ f_{θ,T} and θ ↦ (f_{θ,T})_T are functions on a set of size
2^{Wb}, so neither image can be larger. A union bound gives (3). ∎

None of these quantities enter the bound: depth, recurrence, width,
activation precision, input length. Activations may even have infinite
precision, because the map from weights to function is still a function on a
finite set. A single weight vector fixes the entire family {f_{θ,T}}_T.
Looping lets each of the 2^{Wb} programs run longer; it does not create new
programs. The only freedom it adds is the choice of T itself, worth
log2 T_max bits.

Three corollaries follow, each for a different notion of capability. In
every one, capability per parameter is at most b:

- **A1 (shattering).** If ⋃_{T≤T_max} F_T shatters N points, then
  2^N ≤ T_max·2^{Wb}. So VC-dim ≤ Wb + log2 T_max, which is at most
  b + (log2 T_max)/W per parameter.
  - For real-valued ReLU weights, VC dimension does grow with depth. It is
    O(WL log W) at depth L, with matching examples of order WL log(W/L)
    (Bartlett, Harvey, Liaw & Mehrabian, JMLR 2019). Bounded bits are what
    cap it.
  - Vardi, Yehudai & Shamir (ICLR 2022, Thm 6.1) give O(WB + W log W) for
    B-bit weights. They also show the matching construction: roughly N/B
    parameters at bit complexity Õ(B) memorize N points, for B ≤ √N.
- **A2 (random targets).** Draw f uniformly from a class of size 2^K. Then
  P[f ∈ ⋃_{T≤T_max} F_T] ≤ T_max·2^{Wb−K}.
  - A uniformly random Boolean function on m input bits has K = 2^m. Fitting
    it with probability δ therefore needs Wb ≥ 2^m − log2 T_max − log2(1/δ).
  - Agreeing on a (1−ε) fraction of inputs needs
    Wb ≥ 2^m(1 − H(ε)) − log2 T_max − log2(1/δ).
- **A3 (description length).** For every θ and T,
  K(f_{θ,T}) ≤ Wb + K(T) + c_𝒜 ≤ Wb + O(log T) + c_𝒜. K is the prefix
  Kolmogorov complexity of the function's table on a fixed finite domain. To
  prove it, describe 𝒜, then θ, then T. **A b-bit network, looped or not, can
  only fit targets whose description length is at most about Wb bits.** This
  is the form of A that meets claim B.

*Empirical anchors.* Measured storage sits well below the ceiling, but the
ceiling is still what those measurements live under:

- Language models store about 2 bits per parameter of synthetic knowledge
  (Allen-Zhu & Li, *Physics of LMs 3.3*, ICLR 2025). That figure survives
  int8 quantization and drops to about 0.7 bits per parameter at int4.
- GPT-style models memorize about 3.6 bits per parameter of uniformly random
  data (Morris et al., *How much can language models memorize?*, ICML 2026).

*The finite-precision assumption is necessary.* With real weights, a single
parameter can carry unboundedly many bits. Looped-transformer universality
results that seem to let loops alone buy approximation power confirm this.
Xu & Sato (ICML 2025) bound the approximation error of a looped transformer by
the target's modulus of continuity at a resolution δ that shrinks as the loop
count grows. But the weight bit-complexity their construction needs grows as
O(δ^{−(N+1)d}). On a rich class such as continuous functions, extra loops
help only if you also pay in bits. That is Theorem A.

### Claim B: computation

**Setting.** M is a one-tape Turing machine with s states and a tape symbols.
Its description length is ℓ = Θ(s·a·log(s·a)). On inputs of length n it halts
within T(n) steps, uses at most S(n) cells, and leaves its output on the tape.
A configuration is encoded as a row of one-hot tokens over
Γ = Σ ∪ (Q × Σ), where a symbol paired with a state marks the head. That gives
dimension D = a(s+1).

**Theorem B.** There is a weight-tied block F_M with three parts:

- weights in {−1, 0, 1};
- local token mixing, where position i reads positions i−1, i and i+1 (a
  width-3 convolution, or attention with a hard relative-position pattern);
- a ReLU MLP with O(s·a²) hidden units.

For every input x and every T ≥ T_M(x), with x padded to S(n),

  R ∘ F_M^T ∘ E(x) = M(x).

The block has O(s²a³) = poly(ℓ) weights of log2 3 bits each, independent of T,
n and S.

*Proof.* The head moves at most one cell per step. So the next symbol at cell
i is a function of the current symbols at cells i−1, i and i+1. Write the
update as new = old + Δ. There are four cases:

- **The head is at i.** The cell holds (q, σ), and δ(q, σ) = (q′, σ′, m).
  One unit detects (q, σ). It contributes −e_(q,σ) plus either e_σ′ (if the
  head moves) or e_(q′,σ′) (if it stays).
- **The head enters from the left.** Cell i−1 holds (q, σ) with m = R, and
  cell i holds τ. One unit computes the AND of the two indicators. It
  contributes −e_τ + e_(q′,τ).
- **The head enters from the right.** This is symmetric.
- **Anything else.** Δ = 0, so copying costs nothing.

On {0,1} inputs, ReLU(u + v − 1) computes the AND of u and v exactly. The
bias −1 comes from a constant input channel. So the MLP maps exact one-hot
configurations to exact one-hot configurations, with {−1, 0, 1} weights in
both layers, and nothing accumulates over passes. That takes s·a units for
the first case and s·a² for each direction of entry.

Make each halting state self-looping, δ(h, σ) = (h, σ, stay). Then its Δ is
0, the halting configuration is a fixed point, and F_M^{T′} = F_M^{T} for
every T′ ≥ T_M(x). The construction in `loopcap/construct.py` builds this
kind of block for rule 110 (with a latch for the answer) and for S5. ∎

Remarks:

1. **Softmax attention.** Ordinary softmax attention can stand in for the hard
   local mixing. Attention puts mass 1−ε on the intended neighbour when the
   score margin is ln(S/ε). That needs weight magnitudes of order log S, so
   O(log log S) more bits per weight. A clipped-ramp step in the MLP restores
   exact one-hot values every pass, so the softmax error does not compound
   over T.
2. **Program in the input instead of the weights.** Giannou, Rajput, Sohn,
   Lee, Lee & Papailiopoulos (*Looped Transformers as Programmable Computers*,
   ICML 2023) build fixed weights that execute any program written into the
   input (the "punchcard"):
   - a SUBLEQ one-instruction computer in 9 layers and 2 heads (the
     informal theorem says "fewer than 13 layers" for all their
     constructions);
   - width O(log n + N), from binary positional codes of length log n;
   - softmax with a large temperature standing in for hardmax.

   There the parameter bits are independent of ℓ as well as of T. The ℓ bits
   live in the prompt. Whether the program is counted as parameters or as
   input is a convention, and that convention decides whether "capability
   per parameter" even looks bounded.
3. **Neighbouring results**, all verified against the papers:
   - Back de Luca & Fountoulakis (ICML 2024): looped transformers of constant
     width simulate Dijkstra, BFS, DFS and Kosaraju for graphs of any size up
     to a limit set by numerical precision, O(1/δ̂) nodes.
   - Saunshi, Dikkala, Li, Kumar & Reddi (ICLR 2025):
     - a looped block of L + O(1) layers, run for m loops, simulates m
       chain-of-thought steps of an L-layer model;
     - a 1-layer block run for log2 n loops composes group elements, at
       constant precision (Thm 5.1).
   - Merrill & Sabharwal (NeurIPS 2025): with polynomial padding, O(log^d n)
     loops give exactly FO-uniform TC^d.
   - Li, Liu, Zhou & Ma (ICLR 2024): a constant-depth transformer with T
     chain-of-thought steps computes any circuit of size T. This is
     non-uniform: the circuit sits in the position embeddings.
   - Pérez, Barceló & Marinković (JMLR 2021): Turing completeness through
     decoding steps. It needs arbitrary-precision rationals, and the weights
     depend on M.
   - Chung & Siegelmann (NeurIPS 2021): a 54-neuron fixed-precision RNN plus
     growing stack memory is universal.
4. **Learning theory.** At fixed W·b the hypothesis class is finite. The Occam
   bound therefore gives a sample complexity of O(Wb), whatever T is. Wei,
   Chen & Ma (NeurIPS 2022) make this precise for Turing machines: transformers
   "statistically meaningfully" approximate machines running for time T, with
   sample complexity poly(k, |A|, log T). Here k is the number of states and
   |A| the alphabet size.
5. **Checked in code on this experiment's own tasks**
   (`loopcap/construct.py`, `tests/test_construct.py`):
   - a ternary rule-110 block with **84 weights** solves exactly n\* = T for
     every T;
   - an S5 block solves n\* = T + 1. Its state is a permutation matrix, it has
     150 hidden units, and all its weights are ternary.

   Both stay exactly binary up to T = 400 on 256-token inputs.

**Corollary B′ (time-bounded description length).** Suppose a family {f_n} is
computed by a program of length ℓ in time T(n). Then a tied network with
poly(ℓ) parameter bits computes every f_n, given T(n) passes. Measure
capability as the largest instance of a structured target that is solved, or
as the target's time complexity. In those terms, capability per parameter bit
is unbounded in T: at fixed W and b, n\*(T) → ∞.

Space is the binding constraint, not time. The state must hold S(n) cells, so
the context length (or width) has to grow with space. Time can be exponential
in space.

### The boundary between A and B

- **A counts targets.** It measures capability as *how many* distinct targets
  are reachable. Equivalent framings are the worst case over a large class, or
  the description length of a single target. The cap is Wb bits, plus
  log2 T_max.
- **B prices computation.** It measures capability as *how costly* a reachable
  target may be to compute: its time complexity, or the size of instance
  solved under a fixed rule. At fixed Wb this is unbounded in T.
- **The two are consistent.** B's targets are few. There are at most 2^ℓ
  programs of length ≤ ℓ, so B spends only poly(ℓ) bits of A's budget.
- **A3 and B are converses, up to polynomial overhead.** A3 says
  K(f) ≲ Wb + O(log T). B says that a program of length ℓ and running time t
  is realizable with poly(ℓ) bits and O(t) passes. **Parameter bits pay for
  description length; passes pay for running time.** In Bennett's terms
  (*Logical depth and physical complexity*, 1988), looping buys logical depth,
  not information.
- **Untied networks at the same W.** A depth-L untied network can run at most
  L serial steps of a local algorithm, and each layer needs its own copy of
  the rule. Its reach is therefore at most about Wb / P_step, where P_step is
  the bit cost of one step, unless the target has a parallel shortcut:
  - Every semiautomaton has a ⌈log2 T⌉-depth shortcut (Liu, Ash, Goel,
    Krishnamurthy & Zhang, ICLR 2023). Solvable ones also have
    constant-depth shortcuts, but those need width that grows with T.
  - For non-solvable groups such as S5, constant depth fails unless
    TC0 = NC1. Log-precision transformers are in L-uniform TC0
    (Merrill & Sabharwal, TACL 2023), and the S5 word problem is
    NC1-complete (Barrington 1989; used this way by Merrill, Petty &
    Sabharwal, ICML 2024).

  For untied models of fixed W, then, both claims predict a ceiling that
  tracks Wb.
- **What neither claim says** is whether gradient descent finds the short
  program. That is what Part 2 measures.

---

## Part 2. Experiment

The design, hypotheses and decision rules were committed before any run in
[`preregistration.md`](preregistration.md) (commit `9bf63af`). Two dated
amendments followed, both before the main grid. They changed the task
formulation, the training protocol and the families. They did not change the
hypotheses or the decision rules. §2.6 explains why each change was needed.

### 2.1 Task families and why these

Every task is causal, and its targets are indexed by a problem size n that
fixes how much serial computation the target needs. The rule never changes.
One evaluation pass measures accuracy at every n at once.

- **`ca110d`: rule-110 rollout.** The input is a row of random bits with an
  all-zero background on the left.
  - At every position s the model outputs 32 binary heads. Head k is the cell
    c(s−1−k, k) of the spacetime diagram, that is, the value k rule-110 steps
    later on the diagonal through (s−1, 0).
  - Problem size n = k + 1: head k needs k rule applications.
  - *Why:* predicting rule 110 is P-complete (Neary & Woods 2006) and rule
    110 is universal (Cook 2004). Unless NC = P, no polylog-depth shortcut
    exists, so the required serial depth grows with n.
  - The rule table has 8 entries, so description length is tiny while
    computation grows linearly. This is the regime of Theorem B.
  - The hand-built block in `construct.py` solves the single-column version
    of this task with 84 ternary weights. The diagonal version would need 32
    more channels: a shift register and latches. That extension is
    straightforward but not implemented.
- **`mod7`: running sum mod 7.** The target at position n is
  (a_1 + … + a_n) mod 7.
  - This is the word problem of an abelian group. Constant depth suffices in
    principle, but the width needed then grows with n (Liu et al. 2023).
  - The natural learned algorithm is a scan, n − 1 steps sequentially or
    about log2 n with doubling. Unlike `ca110d`, its scan state *is* the
    target.
  - The S5 construction in `construct.py` carries over to any finite group:
    replace the permutation matrices with the group's regular
    representation.
- **`s5` (exploratory only): iterated permutation composition.** This is the
  canonical NC1-complete word problem (Barrington 1989), and theory makes it
  the most attractive family. No model learned it at this scale (§2.6), so
  it was moved out of the main grid. Only a small d = 64 check remains.
- **Rejected:**
  - binary addition: AC0, so carry-lookahead gives a constant-depth shortcut;
  - parity: hard to optimize, which would confound the test;
  - sorting: TC0.

### 2.2 Arms

All arms use the same block: pre-RMSNorm, causal multi-head attention with
RoPE (head dimension 8), and a GELU MLP. There are no biases and no norm
gains. The input embedding is re-injected before every pass or layer, and a
single readout is shared across depths. Every weight matrix is quantized,
including the embedding and the readout.

| arm | weights | compute per token | depth |
|---|---|---|---|
| **TIED(d, T)** | one block of width d: W | W·T | T passes |
| **UNTIED-W(d, T)** | T blocks, narrowed so the total W matches within 2% | W | T layers |
| **UNTIED-C(d, T)** | T blocks of width d: about W·T | W·T | T layers |

- **UNTIED-W** is the comparison the brief asks for: the same total parameter
  count.
- **UNTIED-C** matches compute instead. It tests whether a tied model at
  (W, T) behaves like an untied model with W·T parameters, which is what
  "parameters × passes" would mean.

W for TIED: 13,408 (d = 32) and 51,392 (d = 64) on `ca110d`; 12,768 and
50,112 on `mod7`.

### 2.3 Precision

Weights are fake-quantized with a straight-through estimator:

- float: bf16 in the matmuls, counted as b = 16;
- int4: 15 levels, per-tensor absmax scale;
- ternary: 3 levels, per-tensor absmean scale, as in BitNet b1.58.

Every model trains in float for the first half of its steps; then its own
quantizer switches on for the second half. The precision axis therefore asks
whether a b-bit network can *hold* a solution of the task. It does not ask
whether straight-through training can bootstrap one from nothing, which
failed in the pilot (§2.6).

Activations are not quantized: claim A concerns weight bits, and Theorem A
holds at any activation precision. Between the extreme cells, d = 64 float
and d = 32 ternary, W·b differs by 39× on `ca110d` and 40× on `mod7`. Even
the smallest budget, about 20 kbit, is two orders of magnitude above the
133 bits of the hand-built rule-110 block. It is also two orders above the
description length of either rule.

### 2.4 Training and measurement

**Training.**

- Each job trains 6 models in one process: 3 bit-widths × 2 seeds. They
  share an architecture and step budget, but each has its own
  initialization and its own data stream.
- Data is fresh every step, so there is no overfitting regime.
- Optimizer: AdamW at lr 3e-3 with cosine decay, batch 32 per model, 6,000
  steps.
- The loss is ½ × (cross-entropy at the final depth) + ½ × (mean
  cross-entropy over earlier depths).
- A curriculum limits the loss to sizes up to a horizon H. H grows to
  n\*\_probe + 8, measured every 100 steps on a separate probe set. The rule
  is identical for every arm.

**Measurement.** acc(n) is accuracy at problem size n on 2,048 held-out
sequences, the same for every run.

**n\*\_θ = max{n : acc(m) ≥ θ for all m ≤ n}**, with θ = 0.9. θ = 0.8 and
0.99 are kept as sensitivity checks. Censoring is at 32 for `ca110d` and 128
for `mod7`. The cell value is the median over the 2 seeds. Tied models are
also evaluated at T_test ∈ {1, 2, 4, …, 128}.

### 2.5 Hypotheses and decision rules

The hypotheses come from the brief and were stated before any run:

- **H_compute:** for the tied model, solvable n keeps growing with T, limited
  by training rather than by W·b.
- **H_ceiling:** solvable n levels off at a level that tracks W·b.

The rules are pre-registered and apply per family to the 6 (d, b) cells of
TIED, with T_test = T_train:

- A cell **grows** if n\*(32) ≥ 1.2·n\*(16), or if n\*(32) is censored.
- A cell **saturates** if n\*(32) < 1.1·n\*(16), n\*(32) < 1.2·n\*(8), and
  n\*(32) is not censored.
- A family is **Wb-insensitive** if n\*(d=32, ternary) ≥ 0.75·n\*(d=64,
  float).
- A family is **Wb-tracking** if Spearman ρ(log Wb, n\*(32)) ≥ 0.6 and
  n\*(d=64, float) ≥ 2·n\*(d=32, ternary).

The verdict for each family:

- **H_compute** if at least 4 of 6 cells grow and the family is
  Wb-insensitive;
- **H_ceiling** if at least 4 of 6 cells saturate and the family is
  Wb-tracking;
- **mixed** otherwise.

### 2.6 What the pilots showed (before the main grid)

All pilots used TIED, d = 32, T = 8, one seed.

1. **Rule 110 asked as one output per position is not learnable at this
   budget.** The first formulation asked, at position n, only for cell 0
   after n−1 steps.
   - After 3,000 steps n\* was 6, and n\*(T_test = 4) = n\*(T_test = 8).
   - The model had memorized lookup tables over the first few inputs and
     found no iteration.
   - The interior cells of the spacetime diagram are never targets in that
     formulation, so credit assignment has to run through every pass.

   This is structural. If every intermediate state of a computation were the
   target of a left-to-right finite-state scan, the computation would be in
   NC1. A P-complete prediction task posed with one output per position
   therefore *must* hide its intermediate state.

   The diagonal formulation exposes the intermediate cells as targets
   without dictating the algorithm. With it, the float model reached
   n\* = 9 at T = 8: one rule step per pass, exactly the sequential maximum.
2. **Permutation composition was not learned at all.** n\* stayed at 0–1
   with or without the curriculum, at lr 1e-3 or 3e-3. Even the trivial
   position-1 copy stayed at 12–39% without the curriculum. This is
   reported as a learnability limit at d = 32 and 3,000 steps. It is not
   evidence about W·b: by Theorem B a ternary block of about 23k weights
   represents the scan exactly.
3. **Ternary weights trained from scratch stalled** at n\* = 2 on `ca110d`.
   Starting from float improved this to n\* = 5 on `ca110d`, but not on
   `mod7` (n\* = 2). Per-channel scales did not help.

These pilots are where most of the time went. They are also the clearest
evidence on how wide the representation-versus-learning gap is (§3).

---

## 3. Results

All numbers come from `results/summary.md` and `results/runs/*.json`, which
`scripts/analyze.py` generates. n\* is at θ = 0.9 and is the median of 2
seeds. Figures are in `figures/`.

### 3.1 Pre-registered verdicts

| family | cells growing (of 6) | cells saturating (of 6) | Wb-insensitive | Wb-tracking (Spearman) | verdict |
|---|---|---|---|---|---|
| `ca110d` | 0 | 3 | no (5.5 vs 22) | yes (0.94) | **mixed** |
| `mod7` | 4 | 0 | no (4.5 vs 36.5) | yes (0.83) | **mixed** |

Neither hypothesis passes as written:

- In `ca110d`, n\* rises with T and then levels off at heights ordered by W·b.
  Only 3 of 6 cells meet the saturation rule; the other 3 still creep
  upward.
- In `mod7`, most cells keep growing. But the smallest-budget cell (d = 32,
  ternary) never gets going, so the family fails the Wb-insensitivity test.

The pre-registered rules cannot tell a *counting* ceiling from a *learning*
ceiling that happens to scale with W·b. Sections 3.2–3.5 separate the two.

### 3.2 `ca110d`: growth, then plateaus ordered by W·b

TIED n\*:

| d | b | T=1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| 32 | float | 5.5 | 8 | 10 | 11.5 | 13 | 12.5 |
| 32 | int4 | 4 | 5.5 | 6 | 5.5 | 8.5 | 9 |
| 32 | ternary | 3.5 | 4.5 | 5.5 | 6 | 5.5 | 5.5 |
| 64 | float | 7 | 10 | 14.5 | 19 | 21.5 | 22 |
| 64 | int4 | 6 | 7 | 10.5 | 13 | 14.5 | 16.5 |
| 64 | ternary | 5.5 | 7 | 8.5 | 9 | 10 | 11 |

- **Passes help, but with diminishing returns.** For d = 32 float, n\* goes
  from 5.5 at T = 1 to 13 at T = 16. One pass already computes about 4–6
  rule steps; extra passes add fewer than one step each.
- **Width buys steps per pass.** At T = 8, d = 64 reaches 19 against 11.5
  for d = 32.
- **The least-squares fit over the 36 uncensored TIED cells** is
  n\* ∝ W^0.36 · b^0.26 · T^0.24. W and T enter with similar, small exponents.
- **These plateaus are not the counting ceiling.**
  - The smallest budget here, d = 32 ternary, still holds about 21 kbit of
    weights.
  - A hand-built ternary block of 84 weights (about 133 bits) solves any n,
    given n passes (Theorem B, checked in `tests/test_construct.py`).
  - So every learned model sits two or more orders of magnitude below what
    its bits could represent.
- **The precision gap opens at the precision switch.**
  - The ternary-designated models were float until step 3,000. At that point
    they had reached n\* ≈ 11–12, the same as the float runs.
  - Quantizing dropped them to 3–4, and in the remaining 3,000 steps they
    recovered only to 5–6.
  - The W·b ordering is therefore largely *how much of a float solution
    survives quantization* in the time allowed, not how much the bits can
    hold.

### 3.3 `mod7`: shortcuts at small T, scans at large T

TIED n\*:

| d | b | T=1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| 32 | float | 10.5 | 21 | 38 | 13.5 | 23 | 35.5 |
| 32 | int4 | 4.5 | 2.5 | 2.5 | 6.5 | 11 | 25 |
| 32 | ternary | 2.5 | 2 | 2.5 | 3.5 | 4.5 | 4.5 |
| 64 | float | 41 | 36 | 43.5 | 24.5 | 40 | 36.5 |
| 64 | int4 | 4 | 10.5 | 15 | 16.5 | 22.5 | 38 |
| 64 | ternary | 0.5 | 8 | 8.5 | 7.5 | 16.5 | 34.5 |

Float models at T = 1 already reach n\* = 41 (d = 64). Theory says how: a
running sum mod 7 can be done in constant depth by counting with attention.
The catch is that this route needs precision that grows with n. Quantization
removes that route, and at T = 1 int4 and ternary models reach only 0.5–4.5.

With more passes, the quantized models recover the reach by iterating:

- **d = 64 ternary** goes 0.5 → 8.5 → 16.5 → 34.5 at T = 1, 4, 16, 32.
- **d = 64 int4** goes 4 → 15 → 22.5 → 38.

At T = 32 every d = 64 cell, float or quantized, lands at 35–38. The
roughly 10× gap in weight bits between float and ternary has been closed by passes.

### 3.4 Test-time passes

The heaviest-trained models change behaviour (`figures/test_time_*.png`).

**Run at fewer or more passes than they were trained with, `mod7` models
trained at T = 32 trace a sequential scan.** Their n\* follows the pass count
almost one-for-one: 1, 2, 4, 8, 17, 35 at 1–32 passes. They keep going past
the training depth:

| model (trained at T = 32) | n\* at 32 passes | 64 passes | 128 passes |
|---|---|---|---|
| d = 64 ternary | 34.5 | 61 | 60 |
| d = 64 int4 | 38 | 49.5 | 49 |
| d = 32 float | 35.5 | 50 | 45.5 |

This is the hand-built construction's signature (n\* = T + 1), arrived at by
gradient descent. It is the clearest evidence here that a fixed set of b-bit
weights turns extra computation into larger solved instances.

**Models trained at small T learned shortcuts that are not stable.** They
reach n\* > T at their training depth, but collapse to n\* ≈ 0 at 2T. So the
algorithm learned depends on the training depth: shortcuts at shallow T,
scans at deep T.

**`ca110d` models trained at T = 32 are stable but do not improve.** At 64
test passes they give 12 (d = 32 float) and 21.5 (d = 64 float), against
12.5 and 22 at 32. Models trained at T ≤ 16 collapse beyond 2T.

### 3.5 Tied vs untied

**At the same parameter count (UNTIED-W), the tied model wins once T ≥ 8.**

| ca110d n\* | T = 4 | T = 8 | T = 16 | T = 32 |
|---|---|---|---|---|
| tied, d = 32 float | 10 | 11.5 | 13 | 12.5 |
| untied, same W | 9.5 | 7.5 | 7 | 7 |

UNTIED-W peaks at T = 4–8 and then declines as its layers become too narrow.
The decline is steeper at lower precision: ternary falls from 4.5 to 2.5.
Spreading a fixed weight budget over more distinct layers costs capability;
spending it again on more passes does not. The same holds on `mod7`: at
d = 64, T = 16, tied int4 reaches 22.5 against 8.5 untied, and tied ternary
16.5 against 3.5.

**At the same compute (UNTIED-C: same depth, T× the parameters),** the untied model wins, and the gap widens with T.

| `ca110d` n\*, d = 32 | T = 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| tied, float | 8 | 10 | 11.5 | 13 | 12.5 |
| untied same compute, float | 9 | 13.5 | 16.5 | 19.5 | 23.5 |
| tied, ternary | 4.5 | 5.5 | 6 | 5.5 | 5.5 |
| untied same compute, ternary | 6 | 8.5 | 11.5 | 13 | 16 |

- **The tied-to-untied ratio** falls from 0.89 at T = 2 to 0.53 at T = 32.
  From T = 8 on, it is outside the pre-registered 0.75–1.33 band that would
  have made "parameters × passes" the currency.
- **On `mod7` at d = 32 the same holds for the quantized models.** At
  T = 32, tied ternary reaches 4.5 against 27.5 untied, and tied int4 25
  against 37.5. The float comparison is too noisy to read.
- **The tied model is not an untied model with W·T parameters.** Both do the
  same arithmetic per token, at the same depth. At equal compute, distinct
  parameters buy more reach.
- **A wider tied block closes most of the gap.** Tied d = 64 at T = 32 has
  51k parameters and reaches n\* = 22. Untied same-compute d = 32 at T = 32
  has 394k parameters and reaches 23.5. That is the same reach with 8× fewer
  parameters and 4× more compute.

**Across all arms** (`results/summary.md`, `figures/nstar_vs_compute.png`), one
power law fits the uncensored cells:

| family | fit | R² |
|---|---|---|
| `ca110d` | n\* ∝ W^0.24 · (W·T)^0.18 · b^0.26 | 0.85 |
| `mod7` | n\* ∝ W^0.37 · (W·T)^0.39 · b^0.65 | 0.78 |

Compute with b explains more variance than parameters with b: 0.72 against
0.69 on `ca110d`, and 0.72 against 0.62 on `mod7`. Among the float `ca110d`
models alone the split is 0.74 against 0.46. But the best fit needs both.


### 3.6 Is the plateau a counting ceiling or a training ceiling?

Three lines of evidence say training:

1. **Constructions.** Ternary blocks of 84 weights (rule 110) and about 23k
   weights (S5) represent the sequential algorithms exactly, for any T.
   Every trained cell has at least about 20 kbit of weights.
2. **Quantization damage.** The ternary/float gap appears at the
   precision-switch step, not as a capacity limit approached gradually
   (§3.2).
3. **3× training steps** (TIED, d = 32, 18,000 steps instead of 6,000):

   | family, T | float n\* (base → 3×) | int4 | ternary |
   |---|---|---|---|
   | `ca110d`, T = 8 | 11.5 → 15 | 5.5 → 8.5 | 6 → 6 |
   | `ca110d`, T = 16 | 13 → 17 | 8.5 → 10 | 5.5 → 6 |
   | `mod7`, T = 8 | 13.5 → 27 | 6.5 → 11.5 | 3.5 → 1.5 |
   | `mod7`, T = 16 | 23 → 39.5 | 11 → 14.5 | 4.5 → 1.5 |

   Float n\* rises by about 30% on `ca110d` and roughly doubles on `mod7`.

   At the longer budget, `ca110d` n\*(16)/n\*(8) is 1.13 and `mod7`'s is
   1.46. The apparent stall between T = 16 and T = 32 at the base budget is
   therefore partly a training limit.

   int4 gains less. The d = 32 ternary models do not recover at all, and
   longer training leaves `mod7` ternary slightly worse. That fits a failure
   to recover from quantization. It does not fit a capacity limit:
   - those models have about 20 kbit of weights;
   - the rules themselves take under 200 bits to write down (rule 110's
     table is 8 bits, the Z7 addition table 147 bits);
   - the hand-built rule-110 block uses 133 bits.


The pre-registered training-limited flag (n\* still rising by ≥ 10% over the
last 20% of steps) fired for 14 of 72 TIED `ca110d` models and 31 of 72
`mod7` models.

### 3.7 `s5` (exploratory)

Tied, d = 64, float, two seeds, 6,000 steps, with the curriculum.

- **T = 8:** n\* = 1. The models learned only the position-1 copy;
  accuracy at position 2 is at chance (about 1%).
- **T = 32:** also n\* = 1 in both seeds, with position 2 at chance.

Permutation composition is NC1-complete, and theory makes it the most
attractive family. A ternary block of about 23k weights executes it exactly
(`construct.py`). It is also the family where learning failed outright at
this scale.


---

## 4. What this implies for extrapolating from parameter counts

A parameter count measures storage, not logical depth. By claim A, W·b caps
how many distinct functions a network can hold, and looping adds at most
log2 T bits. By claim B, it puts no cap on how much computation those
functions may need: a 133-bit ternary block solves rule-110 rollouts of any
length, given enough passes. Extrapolating "capability per parameter" from
untied models to looped ones will therefore over-predict what looping adds to
memorization and under-predict what it adds to serial reach.

The learned models show that the substitution is real but partial. Its size
is set by what training finds, not by either bound: every ceiling observed
here sat orders of magnitude below both.

- **Where training found an iterative algorithm** (`mod7` scans trained at
  T = 32), passes substituted for weight bits outright. Ternary and float
  models converged (n\* ≈ 35). Running the ternary model for 64 passes at
  test time took it to 61 with no new parameters.
- **Where it did not** (rule 110), reach rose jointly with parameters,
  compute and precision: n\* ∝ W^0.24 · (W·T)^0.18 · b^0.26 across all arms.
  An untied model with the same compute but T× the parameters went 1.1–1.9×
  further than the tied one.

So H_compute's prescription survives in a weaker form.
**Parameters × passes** (weight uses per token, the looped analogue of
"active parameters") predicts serial capability better than parameters
alone. The R² comparisons are:

| fit | parameters × passes | parameters |
|---|---|---|
| rule 110, all arms | 0.72 | 0.69 |
| mod7, all arms | 0.72 | 0.62 |
| rule 110, float models only | 0.74 | 0.46 |

But it is not sufficient on its own:

- track it alongside parameter count and weight precision;
- treat W·b only as the ceiling on storable information;
- trust either predictor only once the training recipe demonstrably finds the
  iterative algorithm.

---

## 5. Limitations

- **Scale.** W ranges over 12k–51k, with two seeds per cell, 6,000 steps,
  and problem sizes up to 32 (`ca110d`) or 128 (`mod7`). Every ceiling seen
  here is far below the counting limit, so this experiment cannot observe
  claim A's bound directly. It can only show that it is not what binds.
- **Design changes.** The design changed twice before the main grid
  (Amendments 1–2) and once during it (Amendment 3). Each change and its
  reason is recorded. The reformulated rule-110 task exposes intermediate
  cells as targets. That makes it a test of whether a looped network can
  *execute* a long serial computation it was shown, not whether it can
  *discover* one from end-to-end supervision alone. The original
  formulation failed exactly at discovery (§2.6).
- **The precision axis tests quantize-then-recover, not native low-bit
  training,** which failed in the pilot.
- **Permutation composition, the family theory likes best, was not learned
  at all.**
- **The float arm runs bf16 matmuls,** so its b is about 16, not 32.
- **Shared GPU.** The grid ran with other agents' jobs, so wall-clock
  timings are not comparable across jobs.

---

## 6. Phase 2: claim A measured, and what limits extrapolation

Phase 2 is pre-registered in [`preregistration_phase2.md`](preregistration_phase2.md)
(commit `f8c53fc`, before any phase-2 run), with two dated amendments. Its
results are in `results/phase2/`, generated by `scripts/analyze_phase2.py`.

### 6.1 What stops learned scans from extrapolating

Phase 1 left a puzzle. The `mod7` scans trained at T = 32 kept improving past
their training depth, but then stalled near n\* ≈ 60. One candidate was
global softmax attention, which has to single out a neighbour among more and
more positions. Theorem B's block avoids this by mixing tokens locally.
Phase 2 tested this candidate in two ways (`figures/phase2_extrapolation.png`):
it re-evaluated the phase-1 checkpoints on 512-token sequences with up to 512
passes, and it retrained with local attention (window 4).

Median n\* over bit-widths and seeds (`mod7` on 512-token sequences;
`ca110d64` on its 64 heads):

| model (all trained at T = 32) | 32 passes | 64 | 128 | 512 |
|---|---|---|---|---|
| `mod7` d = 64, global (phase-1 checkpoints) | 35.5 | 51 | 51 | 50.5 |
| `mod7` d = 64, local, window 4 | 37.5 | 59.5 | 59.5 | 59 |
| `ca110d64` d = 64, global | 16.5 | 14 | 7 | — |
| `ca110d64` d = 64, local, window 4 | 19 | 16 | 2 | — |

Against the pre-registered rules:

- **Local attention does not extrapolate on either task.** On `mod7`,
  n\*(128)/n\*(32) = 1.59, short of the 2 required. On `ca110d64` it is
  0.11.
- **The global `mod7` model** has ratio 1.44, so it formally "does not
  plateau". In fact it is flat from 64 passes on. The 128-versus-32 ratio
  was too coarse a rule.
- **The `ca110d64` global control plateaus** (ratio 0.42).
- **Global checkpoints trained to shallower depth (T = 16) or narrower
  width (d = 32)** degrade once run past about twice their training depth.

So, by the pre-registered attribution, attention dilution is not what stops
extrapolation. Two other things are.

1. **Coverage.** Most models' reach ends close to the largest problem size
   their curriculum ever put in the loss:

   | `mod7` models | horizon at end of training | plateau |
   |---|---|---|
   | global | 42–49 | 39–64 |
   | four of the six local | 45–60 | 24–62 |

   A model that was never asked to be right at size 70 is not right at size
   70, however many passes it gets.
2. **Per-step reliability.** Two local models got far beyond their horizon:
   int4 seed 1 (horizon 46, n\* = 107) and ternary seed 1 (horizon 45,
   n\* = 126). Their accuracy decays gradually with distance and has no
   cliff at the 128-token training length. For the ternary model:

   | position | 60 | 100 | 127 | 160 | 200 | 256 |
   |---|---|---|---|---|---|---|
   | accuracy | 0.96 | 0.94 | 0.90 | 0.82 | 0.66 | 0.39 |

   Reach ends where the per-step errors have compounded past the 10%
   threshold.

This is the gap between Theorem B and gradient descent. The hand-built block
re-binarizes its state exactly on every pass, so it has zero drift and
unbounded reach. A learned scan carries a soft state that leaks a little on
each step. Its reach is then set by how far its loss reached in training, and
by how small its per-step error is. Neither is a W·b limit: the ternary model
with n\* = 126 has 50k weights of 1.58 bits.

### 6.2 Claim A, measured

**The first design gave no signal.** It asked models to memorize a random
Boolean function on 16 key bits (`mem16`), and every one stored about 0
bits. A random truth table spelled out in binary tokens gives gradient
descent nothing to follow.

**The redesign (Amendment 1) uses the classical capacity setting,
`memrand`.**

- There are 65,536 keys. Each is one token whose embedding is a fixed random
  vector. The embeddings count as input data, not as parameters, so they are
  excluded from W.
- Labels are random bits.
- Training: batch 1,024 for 60,000 steps (Amendment 2).
- Stored bits = Σ over all keys of (1 − CE/ln 2), computed on the whole
  table (`figures/phase2_memorization_memrand.png`).

Stored bits per parameter (median of two seeds):

| d (W) | b | T=1 | 2 | 4 | 8 | 16 | 32 | claim-A bound |
|---|---|---|---|---|---|---|---|---|
| 16 (3,104) | float | 1.22 | 1.65 | 1.76 | 1.81 | 1.65 | 1.82 | 16 |
| 16 | int4 | 0.39 | 0.43 | 0.44 | 0.40 | 0.40 | 0.34 | 3.9 |
| 16 | ternary | 0.12 | 0.12 | 0.15 | 0.14 | 0.13 | 0.12 | 1.58 |
| 32 (12,352) | float | 1.65 | — | 1.71 | — | 1.24 | — | 16 |
| 32 | int4 | 0.48 | — | 0.44 | — | 0.38 | — | 3.9 |
| 32 | ternary | 0.14 | — | 0.13 | — | 0.14 | — | 1.58 |

- **The bound holds everywhere, by a wide margin.**
  - Float stores at most 1.82 bits per parameter, int4 at most 0.48, and
    ternary at most 0.15. That is roughly a tenth of b in each case.
  - All 18 float models are saturated by the pre-registered rule: their last
    logged change is 2.5% or less.
  - The quantized measurements oscillate after the precision switch, in both
    directions. 12 of 18 int4 and 13 of 18 ternary models fail the 5% rule,
    so treat them as noisy lower bounds.
- **Passes barely change storage.**
  - At d = 16 float, a second pass helps (1.22 → 1.65 bits per parameter): a
    one-layer network is a poorer memorizer at equal W.
  - From T = 4 to T = 32 storage is flat, a ratio of 1.04.
  - Pre-registered verdict: flat in 5 of 6 cells. The sixth, d = 32 float, is
    "in between" (0.73) because T = 16 stores *less* than T = 4. No cell
    grows.
- **The float level, about 1.8 bits per parameter, matches** what Allen-Zhu &
  Li measured for knowledge stored in language models (about 2 bits per
  parameter).
- **Quantized storage is mostly quantization damage.**
  - At step 30,000, ternary-designated models (still float then) held up to
    17,000 bits.
  - At the switch they fell to about 1,600.
  - They recovered only to about 1,700–1,900, which is 9% of their
    19.7-kbit bound.

### 6.3 The two claims, measured on one architecture

Combined with phase 1, this puts claims A and B side by side on the same
model family:

- **Storage.** Going from 4 to 32 passes leaves stored bits flat: d = 16
  float goes from 1.76 to 1.82 bits per parameter.
- **Serial reach.** The same increase in passes multiplied serial reach.
  `mod7` at d = 64, ternary went from n\* = 8.5 at T = 4 to 34.5 at T = 32,
  and to 61 with 64 passes at test time.

**Passes buy computation, not storage.** That is the boundary drawn in
Part 1, now measured rather than argued. The learned models stay far inside
both theoretical limits:

- They store about a tenth of the bits their weights allow.
- Their serial reach is set by training coverage and per-step reliability,
  not by W·b.

### 6.4 Phase-2 caveats

- **The `mem16` design failed.** It is reported, not interpreted.
- **A text patch moved logging and periodic evaluation out of the training
  loop** for the Q2 runs and `mem16` (Amendment 2). Their final results are
  valid, but their mid-training history is missing. It was fixed before
  `memrand`, with a regression test.
- **Small samples.** Two seeds per cell, with large seed variance in the
  local-attention runs: 24 to 126 within the same configuration.
- **The quantized arms test quantize-then-recover**, as in phase 1.

---

## 7. Phase 3: does a depth-stability objective make learned scans extrapolate?

Pre-registered in [`preregistration_phase3.md`](preregistration_phase3.md)
before any run, with one operational amendment that changes memory settings
only. Results are in `results/phase3/`, generated by
`scripts/analyze_phase3.py`; the figure is `figures/phase3_progressive.png`.

**The recipe** adds a progressive term in the spirit of Bansal et al. (2022)
to the phase-2 training:

1. Take the detached hidden state after a random n ∈ {0, …, 32} passes.
2. Run 8 more passes from it, with gradients.
3. Add ½ × the cross-entropy at the end of that segment to the loss.

A segment can end at depth 40, past the 32 training passes. Each arm is
compared with the same recipe without the progressive term.

Median n\* over bit-widths and seeds (`mod7` on 512-token sequences):

| arm | 32 passes | 64 | 128 | 512 | stable | extrapolates | vs baseline at 128 |
|---|---|---|---|---|---|---|---|
| `mod7`, local | 37 | 63 | 69 | 67.5 | yes | no (1.86×) | no clear effect (1.16) |
| `mod7`, global | 38 | 47.5 | 47 | 31 | no | no | no clear effect (0.92) |
| `ca110d64`, local | 16 | 12.5 | 1 | (1 at 256) | no | no | "hurts" (0.50) |

My predictions were mostly wrong:

- **Stability.** I expected all three arms to become stable. Only `mod7` with
  local attention did.
- **Rule 110.** I expected the progressive term to stop the `ca110d64`
  collapse. It did not.
- **The "hurts" label** compares 1 against 2, which are both collapsed
  states. Neither the progressive run nor the baseline survives 128 passes.

**The median hides a split by precision.** In float, the progressive term
did what it was designed to do:

- **Local `mod7` float models** reach n\* = 98 and 91, against 56 and 62 for
  the baseline. Both go well past their training horizons (47 and 67) and hold
  there out to 512 passes.
- **Ternary models got worse** (23 and 15, against 126 and 24). The progressive
  term does not survive the switch from float to ternary.
- **With global attention**, one float seed reached 112 at 64 passes, then
  collapsed to 8 by 256.

What this adds to §6.1:

- **A stability objective helps exactly where training reaches.**
  - It made float scans iteration-stable out to 512 passes, using local
    attention.
  - It pushed their reach to about twice the problem sizes the loss covered.
    That is more than in any earlier phase.
  - It cannot teach stability far beyond the depths it trains: 40 here. The
    rule-110 models, and one global `mod7` model, still collapse between 64
    and 256 passes.
- **Two things a learned model lacks, compared with the hand-built block:**
  - an exact fixed point after it halts (no drift);
  - an update that keeps its state exactly binary, so nothing leaks.

  Training to depth 40 approximates both only locally.
- **Weight bits are again not what binds.** The float models with n\* = 98
  have the same W as the ternary ones with 23. The difference comes from
  training through quantization, not from the counting bound: the ternary
  bound is 79 kbit, against about 130 bits needed (§6.2).

The natural next step would train at larger depth, or with a contraction
term that makes "solved" positions a fixed point of the block. Neither was
run.
