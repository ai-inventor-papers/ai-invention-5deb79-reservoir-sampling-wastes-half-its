# ASSUMED, NOT PROVED

Nothing in `Anytime.lean` is declared with the `axiom` keyword. Everything imported from outside
the formalisation enters as an **explicit hypothesis** of the theorem that needs it, so that
`#print axioms` stays at the three standard Lean axioms and a reader can see the exact boundary of
what was machine-checked. `results/print_axioms.json` records the raw output for all 30 named
declarations; none mentions `sorryAx` and none mentions a user axiom.

Below, "standard" means: a textbook result whose proof is not in doubt, only out of scope here.

---

### 1. Knuth–Yao optimality
**Where it enters.** Hypothesis `hKY` of `acc_floor`:
`∀ q, 0 ≤ q → q ≤ 1 → binEnt q ≤ cost q`.

**Claim assumed.** Any algorithm that produces a Bernoulli$(q)$ sample from an i.i.d. fair-coin
source uses at least $h(q)$ flips in expectation, where $h$ is binary entropy in bits.

**Why standard.** This is the lower-bound half of the DDG-tree (discrete distribution generating
tree) analysis. Formalising it requires a model of randomised computation over a coin-flip tree,
which is far outside this file's scope.

**Pointer.** D. E. Knuth and A. C. Yao, *The complexity of nonuniform random number generation*,
in Algorithms and Complexity (1976), pp. 357–428.

---

### 2. Concavity of the eviction objective ⇒ the minimum is at a vertex
**Where it enters.** Only through the numeric hypothesis `hEB : E + 1 ≤ A + B` of
`forest_split_count`; the concavity statement itself never appears in Lean.

**Claim assumed.** $\;P\mapsto\sum_S P(S)\,H(q_S)$ is concave in the transportation plan, so its
minimum over the (compact, convex) transportation polytope is attained at an extreme point.

**Why standard.** $H$ is concave (Mathlib even has `Real.concaveOn_negMulLog`), a non-negative
combination of concave functions is concave, and a concave function on a compact convex polytope
attains its minimum at a vertex.

**Pointer.** R. T. Rockafellar, *Convex Analysis* (1970), Cor. 32.3.2.

---

### 3. A transportation-polytope vertex has forest support
**Where it enters.** Same place: `hEB : E + 1 ≤ A + B` of `forest_split_count`.

**Claim assumed.** The support of a basic feasible solution (vertex) of a transportation problem
on a bipartite graph with $A$ sources and $B$ sinks is a forest, hence has at most $A+B-1$ edges.

**Why standard.** This is the spanning-tree characterisation of bases in the network-simplex
method: the columns of the node–arc incidence matrix indexed by a basis are linearly independent
iff the corresponding arcs contain no cycle.

**Pointer.** R. K. Ahuja, T. L. Magnanti and J. B. Orlin, *Network Flows* (1993), Thm. 11.12
("spanning tree solutions"); or any LP text's treatment of the transportation problem.

---

### 4. Plan feasibility — **NOT** assumed
This was on the plan's assumption list as a contingency. It was **discharged**: `theoremC_existence`
proves constructively, with the textbook $f_{\mathrm{unif}}(S,T)=1/k$ witness, that a plan
satisfying both the row and the column conditions exists for every $1\le k\le i$. The two counting
facts it needs are also proved (`Finset.card_powersetCard` for the $k$ children; `card_parents`,
proved here by an explicit injection $y\mapsto T\cup\{y\}$, for the $i-k+1$ parents). No Birkhoff
or Hall argument is invoked. Theorem C is therefore not vacuous.

---

### 5. `entropy_le_log_card` — **NOT** assumed
This was the plan's fallback (take max-entropy as a hypothesis). It was **discharged**: proved from
scratch by the elementary $\log x \le x-1$ argument. The route is recorded explicitly in
`results/informal_proofs.md` (Lemma 4): *not* Mathlib's measure-theoretic
`measureEntropy_le_log_card`, *not* Jensen via `ConcaveOn.le_map_sum`; only
`Real.log_le_sub_one_of_pos` and `Real.negMulLog_zero` are used.

---

### 6. Modelling assumptions of Theorem B (stated as hypotheses of `per_step_entropy_bound`)
These are not "outside mathematics", but they are *modelling* choices and the reader should see them:
* `hsrc : src.card = Nat.choose i k` — the source mass is uniform on $J(i,k)$, i.e. the induction
  hypothesis of Theorem C holds at step $i$;
* `hzero` — non-split sources have a deterministic eviction (zero entropy);
* `hmax : ∀ S ∈ split, H S ≤ Real.logb 2 k` — each split source's coupling is supported on at most
  $k$ atoms, so Lemma 4 applies. (Lemma 4 is proved; what is *assumed* is only that the support
  has at most $k$ atoms, which is the "evict one of the $k$ current elements" structure.)

---

### 7. Not formalised, stated informally only
* $H_k=\Theta(\log k)$, i.e. that `total_sharp`'s bound really is $\Theta(k\log^2 k)$. The
  *inequality* $\sum \le \log_2(k)\,k\,H_k$ **is** formalised (`total_sharp`), as is
  $H_k\le k$ (`harm_le_self`, which recovers the easy bound); what is not formalised is the
  asymptotic growth rate of $H_k$ itself, which is quoted only to name the order.
* Everything on the plan's explicit non-goal list: minimum-entropy-coupling hardness, Algorithm L,
  floating point, PRNGs, blind-spot detectors.
