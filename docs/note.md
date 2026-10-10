# Does looping lift the parameter ceiling?

*Capability per parameter in weight-tied networks: a theory note and an experiment.*

<!-- RESULTS-SUMMARY -->

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
  - For real-valued ReLU weights, VC dimension does grow with depth:
    Θ(WL log W) for depth L (Bartlett, Harvey, Liaw & Mehrabian, JMLR 2019).
    Bounded bits are what cap it.
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
i is a function of the current symbols at cells i−1, i and i+1. There are four
cases:

- the head is at i;
- the head enters i from the left;
- the head enters i from the right;
- none of these, and the cell is copied.

Each case is the AND of at most two one-hot indicators. On {0,1} inputs,
ReLU(u + v − 1) computes the AND of u and v exactly. Write the update as
residual = old + Δ. The new one-hot symbol is a {−1, 0, 1} combination of
these AND units and of copy units. So the MLP maps exact one-hot
configurations to exact one-hot configurations, and nothing accumulates over
passes.

Counting hidden units: there are s·a for the head at i, s·a² for each
direction of entry, and a copies. Make each halting state self-looping,
δ(h, σ) = (h, σ, stay). Then the halting configuration is a fixed point, and
F_M^{T'} = F_M^{T} for every T' ≥ T_M(x). ∎

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
[`preregistration.md`](preregistration.md) (commit `9bf63af`). Departures are
listed there as dated amendments and in §2.6.

### 2.1 Task families and why these

Every task is causal. The target at position n depends on exactly the first
n input tokens, so **n is the problem size**. One pass over a 256-token
sequence therefore measures accuracy at every n ≤ 256. The rule is the same
at every n; only the amount of serial computation grows.

- **`s5`: iterated permutation composition.** The input is n uniform
  permutations of five elements; the target is their running composition
  p_n = g_n ∘ p_(n−1), one of 120 classes.
  - *Why:* it is the canonical NC1-complete word problem (Barrington 1989).
    Fixed-depth log-precision transformers lie in TC0, so unless TC0 = NC1 no
    fixed depth solves all n. Log2 n passes of one looped layer suffice in
    principle (Saunshi et al. 2025, Thm 5.1). The task therefore separates a
    *parallel* solution (depth about log n) from a *sequential* one (depth
    about n), and both show up as different shapes of n\*(T).
- **`ca110`: rule-110 rollout.** The input is n random bits; the target is the
  value of cell 0 after n−1 steps of rule 110, with an all-zero background to
  the left.
  - *Why:* predicting rule 110 is P-complete (Neary & Woods 2006) and rule 110
    is universal (Cook 2004). Unless NC = P, no polylog-depth shortcut exists,
    and the only known general algorithm is n−1 sequential local updates.
  - Its rule table has 8 entries, so the description length is tiny while the
    required computation grows linearly. This is exactly the regime of
    Theorem B, and the hand-built 84-weight ternary block solves it.
- **`mod7`: running sum mod 7.** This is the contrast family.
  - The group is abelian, hence solvable, so in principle constant depth
    suffices (Liu et al. 2023). The catch is that the constant-depth
    construction needs width or precision that grows with n.
  - It is the case where serial depth is *not* strictly required. It runs on
    a reduced grid.
- **Rejected:**
  - binary addition: carry-lookahead puts it in AC0, a constant-depth
    shortcut;
  - parity: the same group structure as `mod7`, but notoriously hard to
    optimize, which would confound the test;
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
| **UNTIED-W(d, T)** | T blocks, narrowed to the same total W (within 2%) | W | T layers |
| **UNTIED-C(d, T)** | T blocks of width d: about W·T | W·T | T layers |

UNTIED-W is the comparison the brief asks for: the same total parameter count.
UNTIED-C matches compute instead. It tests directly whether a tied model at
(W, T) behaves like an untied model with W·T parameters, which is what
"parameters × passes" would mean.

### 2.3 Precision

Weights are fake-quantized during training, with a straight-through
estimator, to one of three formats:

- float: bf16 in the matmuls;
- int4: 15 levels, per-tensor absmax scale;
- ternary: 3 levels, per-tensor absmean scale, as in BitNet b1.58.

The quantized models store W·b bits plus one fp32 scale per weight tensor.
Activations are not quantized: claim A is about weight bits, and Theorem A
holds whatever the activation precision.

### 2.4 Measurement

acc(n) is exact-match accuracy at position n over 2,048 held-out sequences,
the same sequences for every run.

**n\*\_θ = max{n : acc(m) ≥ θ for all m ≤ n}**, with θ = 0.9. The values
θ = 0.8 and 0.99 are kept as sensitivity checks. The value 256 is censored
("solves every position").

Each job trains 6 models in one process: 3 bit-widths × 2 seeds. They share
an architecture and step budget but have independent initializations and
independent data streams. Data is generated fresh every step, so no example
repeats and there is no overfitting regime. The cell value is the median over
seeds. Tied models are also evaluated at T_test ∈ {1, 2, 4, …, 128}.

### 2.5 Hypotheses and decision rules

The hypotheses come from the brief and were stated before any run:

- **H_compute:** for the tied model, solvable n keeps growing with T, limited
  by training rather than by W·b.
- **H_ceiling:** solvable n levels off at a level that tracks W·b.

The operational rules (pre-registered):

- A (d, b) cell **grows** if n\*(32) ≥ 1.2·n\*(16), or if n\*(32) is
  censored.
- A cell **saturates** if n\*(32) < 1.1·n\*(16) and n\*(32) < 1.2·n\*(8).
- A family is **Wb-insensitive** if n\*(d=32, ternary) ≥ 0.75·n\*(d=64,
  float). These two cells differ in W·b by about 30× on s5 and about 40× on
  ca110.
- A family is **Wb-tracking** if Spearman ρ(log Wb, n\*) ≥ 0.6 and the
  high-Wb cell reaches at least 2× the low-Wb cell.

The verdict for each family:

- **H_compute** if at least 4 of 6 cells grow and the family is
  Wb-insensitive;
- **H_ceiling** if at least 4 of 6 cells saturate and the family is
  Wb-tracking;
- **mixed** otherwise.

<!-- RESULTS -->
