# Self-summary: what worked, what failed, what to try next

## Outcome
**40/40 declarations verified sorry-free**, 38 named + 2 anonymous sanity `example`s, all passing
`#print axioms` with only the three standard Lean axioms. Every item on the artifact plan landed,
including both optional stretch items (the constructive existence corollary in §3 and the sharp
total in §8). Spend: **$0**.

## What worked
* **Verify numerically before formalising.** `verify_small_cases.py` checked every statement over
  exact rationals *first* (≈ 42 000 checks, 0 failures), including Theorem C against *randomly
  generated* column-feasible plans. No Lean time was spent on a mis-specified statement.
* **A persistent Lean worker.** `lean_worker.py` imports Mathlib once into a pooled server and then
  serves `jobs/*.lean`. This turned a ~20-minute cold start plus a ~250 s per-call import into
  **2–6 s** per iteration and made ~50 compile cycles affordable.
* **Keeping the routing plan abstract in Theorem C.** Stating "any plan with these column sums" is
  both easier in Lean (no combinatorics in the proof) and strictly stronger on paper. It also made
  visible that the ROW condition is never used.
* **Truncation-free statements.** `card + 1 ≤ B` instead of `card ≤ B - 1`; `ℤ` for the index
  convention; casts discharged once in `_real` corollaries under `k ≤ i`. Natural subtraction never
  caused a single failed goal.
* **Reading the printed residual of a failed `linear_combination`.** The residual told us the exact
  coefficient (`-k`, not `+k`) for `theoremC_accept`; guessing twice would have been wasted.
* **Bisecting an error with no position.** A stuck `AddCommMonoid ?m` reported no location; compiling
  declaration-*prefixes* (32 small jobs) localised it in two rounds.

## What failed
* **Loogle indexes a newer Mathlib than the one we compile against.** Two lemmas it returned —
  `Nat.succ_mul_choose_eq` and `Finset.card_filter_powersetCard_subset` — do not exist at v4.14.0.
  Fixes: `Nat.choose_mul_succ_eq` (which states L0b verbatim) and a from-scratch `card_parents`
  via `Finset.card_image_of_injOn`. **Always confirm a Loogle hit by compiling a `#check`.**
* **`set ... with` + `(Finset.sum_subset h ?_).symm`** produced the stuck typeclass goal. Writing the
  filter out in full and using `Finset.sum_subset` forwards fixed it.
* **`rw ... at *`** rewrote a hypothesis into a tautology and destroyed information needed later.
* **`∑ _j ∈ s, 2`** without a type ascription leaves the monoid a metavariable; `(2:ℕ)` fixes it.
* **`field_simp` sometimes closes the goal**, so a trailing `ring` errors with `no goals to be
  solved`. `try ring` is the safe idiom — an actually-unsolved goal still surfaces.
* **The module docstring must not precede `import Mathlib`** in this runner.

## What to try next
1. **Formalise `H_k = Θ(log k)`** so `total_sharp`'s order is machine-checked end to end, rather than
   only the inequality `≤ log₂(k)·k·H_k`. Mathlib has harmonic-number API worth checking first.
2. **Tighten Theorem B.** The current bound is loose by a factor tending to 2 at `k = 2` because it
   only uses "a vertex has forest support", not the entropy of the *optimal* vertex. Closing it is a
   minimum-entropy-coupling question (see `results/open_problem.md`).
3. **Formalise the Knuth–Yao lower bound** (currently hypothesis `hKY`), which would make the whole
   of Theorem A machine-checked. This needs a DDG-tree model of randomised computation.
4. **Instantiate Theorem C at the hybrid rule H** to prove formally that it is exact exactly on the
   steps with `k ∣ (i-k+1)` — the quantitative half of the open problem.
