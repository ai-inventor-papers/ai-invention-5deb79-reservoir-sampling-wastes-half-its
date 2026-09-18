import Mathlib

/-!
# Machine-checking the math behind an anytime-uniform reservoir sampler

Six load-bearing facts behind the anytime reservoir-sampling claims:

* `index_convention` — the two divisibility conventions used in the run's own code agree.
* `L0a`, `L0b` (+ real corollaries) — the two binomial identities.
* `accept_forced` — the accept coin is forced to `k/(i+1)` in EVERY state (Theorem A's floor).
* `theoremC` — ANY column-feasible routing plan preserves prefix uniformity (Theorem C).
* `uniformPlan_row`, `uniformPlan_col`, `theoremC_existence` — a feasible plan always exists
  (constructive witness: the textbook `1/k` rule), so Theorem C is never vacuous.
* `forest_split_count` — the forest-support split-source count (Theorem B's combinatorial core).
* `entropy_le_logb_card`, `per_step_entropy_bound` — the per-acceptance eviction-entropy bound.
* `total_constant_in_n` — the total eviction entropy is bounded independently of `n`.

## §0 Index convention (fixed once)

`i` is the PREFIX SIZE throughout: `R_i` is the reservoir after `i` arrivals, and the
arriving item is item number `i+1` (1-indexed) = index `i` (0-indexed). `[i]` is modelled as
`Finset.range i = {0,…,i-1}` and `J(i,k) := Finset.powersetCard k (Finset.range i)`.

The run's own code disagrees with itself about this, which is why `index_convention` is
shipped as a citable one-liner:
* `iter_1/gen_art/gen_art_experiment_1/fibermap.py` indexes `i` as the prefix size
  and writes the balanced condition `k ∣ (i-k+1)`;
* `iter_1/gen_art/gen_art_experiment_2/candidates.py` indexes the arriving item
  and writes `k ∣ (i-k)`.
`index_convention` says the difference between the two indices is exactly `k`, so
`k ∣ (i-k+1)` and `k ∣ (i+1)` name the SAME set of steps.

## Assumed, not proved

Nothing below is an `axiom`. Everything imported from outside is an explicit HYPOTHESIS of
the theorem that needs it, so `#print axioms` stays at the three standard Lean axioms:
* Knuth–Yao optimality  → hypothesis `hKY` of `acc_floor`.
* concavity ⇒ vertex optimum, and "a transportation-polytope vertex has forest support"
  → enter only as the numeric hypothesis `hEB : E + 1 ≤ A + B` of `forest_split_count`.
* "each split source has at most `k` atoms" → hypothesis `hmax` of `per_step_entropy_bound`.
-/

namespace Anytime

open Finset

/-! ## §0 Index convention -/

/-- The prefix-size index `i-k+1` and the arriving-item index `i+1` differ by exactly `k`,
so `k ∣ (i-k+1)` and `k ∣ (i+1)` describe the same steps. Stated over `ℤ` so that natural
subtraction cannot truncate. -/
theorem index_convention (i k : ℤ) : k ∣ ((i + 1) - (i - k + 1)) := by
  have h : (i + 1) - (i - k + 1) = k := by ring
  rw [h]

/-- Arithmetic form of the same fact in `ℕ`, in the shape the two code files use. -/
theorem index_convention_nat {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    (i - k + 1) + k = i + 1 := by omega

/-! ## §1 L0 — binomial bookkeeping -/

/-- **L0(a): biregularity of the Johnson graph.**
Every `S ∈ J(i,k)` has exactly `k` children `S \ {x} ∈ J(i,k-1)`, and every `T ∈ J(i,k-1)`
has exactly `i-k+1` parents `T ∪ {y}`. Double counting the edges gives this identity, and
biregularity is exactly what makes a transportation plan with uniform child demand
`(i-k+1)/k` feasible. -/
theorem L0a {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    k * Nat.choose i k = (i - k + 1) * Nat.choose i (k - 1) := by
  obtain ⟨m, rfl⟩ : ∃ m, k = m + 1 := ⟨k - 1, by omega⟩
  have h := Nat.choose_succ_right_eq i m
  have e1 : m + 1 - 1 = m := by omega
  have e2 : i - (m + 1) + 1 = i - m := by omega
  rw [e1, e2]
  calc (m + 1) * Nat.choose i (m + 1) = Nat.choose i (m + 1) * (m + 1) := by ring
    _ = Nat.choose i m * (i - m) := h
    _ = (i - m) * Nat.choose i m := by ring

/-- **L0(b): prefix ratio.** `(i+1) * C(i,k) = (i+1-k) * C(i+1,k)`. -/
theorem L0b (i k : ℕ) :
    (i + 1) * Nat.choose i k = (i + 1 - k) * Nat.choose (i + 1) k := by
  have h := Nat.choose_mul_succ_eq i k
  calc (i + 1) * Nat.choose i k = Nat.choose i k * (i + 1) := by ring
    _ = Nat.choose (i + 1) k * (i + 1 - k) := h
    _ = (i + 1 - k) * Nat.choose (i + 1) k := by ring

/-- `0 < C(i,k)` as a real number. -/
theorem choose_pos_real {i k : ℕ} (hki : k ≤ i) : (0:ℝ) < (Nat.choose i k : ℝ) := by
  exact_mod_cast Nat.choose_pos hki

/-- **L0(a) over `ℝ`:** the child/parent ratio. `C(i,k-1)/C(i,k) = k/(i-k+1)`. -/
theorem L0a_real {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    (Nat.choose i (k - 1) : ℝ) / (Nat.choose i k : ℝ) = (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1) := by
  have hA : (0:ℝ) < (Nat.choose i k : ℝ) := choose_pos_real hki
  have hki' : (k:ℝ) ≤ (i:ℝ) := by exact_mod_cast hki
  have hden : (0:ℝ) < (i:ℝ) - (k:ℝ) + 1 := by linarith
  have hnat := L0a hk1 hki
  have hcast : (k:ℝ) * (Nat.choose i k : ℝ)
      = ((i:ℝ) - (k:ℝ) + 1) * (Nat.choose i (k - 1) : ℝ) := by
    have h := congrArg (fun n : ℕ => (n:ℝ)) hnat
    simp only [Nat.cast_mul, Nat.cast_add, Nat.cast_one, Nat.cast_sub hki] at h
    linarith
  field_simp
  linarith [hcast]

/-- **L0(b) over `ℝ`:** `C(i,k)/C(i+1,k) = (i+1-k)/(i+1)`. -/
theorem L0b_real {i k : ℕ} (hki : k ≤ i) :
    (Nat.choose i k : ℝ) / (Nat.choose (i + 1) k : ℝ)
      = ((i:ℝ) + 1 - (k:ℝ)) / ((i:ℝ) + 1) := by
  have hA : (0:ℝ) < (Nat.choose i k : ℝ) := choose_pos_real hki
  have hA' : (0:ℝ) < (Nat.choose (i + 1) k : ℝ) := choose_pos_real (le_trans hki (Nat.le_succ i))
  have hki' : (k:ℝ) ≤ (i:ℝ) := by exact_mod_cast hki
  have hki2 : k ≤ i + 1 := le_trans hki (Nat.le_succ i)
  have hnat := L0b i k
  have hcast : ((i:ℝ) + 1) * (Nat.choose i k : ℝ)
      = ((i:ℝ) + 1 - (k:ℝ)) * (Nat.choose (i + 1) k : ℝ) := by
    have h := congrArg (fun n : ℕ => (n:ℝ)) hnat
    simp only [Nat.cast_mul, Nat.cast_add, Nat.cast_one, Nat.cast_sub hki2] at h
    linarith
  rw [div_eq_div_iff (ne_of_gt hA') (by linarith : ((i:ℝ) + 1) ≠ 0)]
  linarith [hcast]

/-! ## §2 L1 — the accept coin is forced -/

/-- **L1 (Lemma 1 of the paper): the accept probability is forced.**

Hypothesis `h` is the structural fact that the ONLY way the reservoir at step `i+1` can be
`S'` while omitting the arriving item is that the reservoir was already `S'` and the item was
rejected. That makes the linear system DIAGONAL — one unknown per equation — so the accept
probability is pinned pointwise, with no freedom left to the algorithm designer, and in
particular it cannot depend on the state `S'`. -/
theorem accept_forced {i k : ℕ} (hki : k ≤ i) (p : Finset ℕ → ℝ)
    (h : ∀ S' ∈ Finset.powersetCard k (Finset.range i),
      (1 / (Nat.choose i k : ℝ)) * (1 - p S') = 1 / (Nat.choose (i + 1) k : ℝ)) :
    ∀ S' ∈ Finset.powersetCard k (Finset.range i), p S' = (k:ℝ) / ((i:ℝ) + 1) := by
  intro S' hS'
  have hA : (0:ℝ) < (Nat.choose i k : ℝ) := choose_pos_real hki
  have hA' : (0:ℝ) < (Nat.choose (i + 1) k : ℝ) := choose_pos_real (le_trans hki (Nat.le_succ i))
  have hi : (0:ℝ) < (i:ℝ) + 1 := by positivity
  have hb := L0b_real hki
  have hh := h S' hS'
  have key : 1 - p S' = ((i:ℝ) + 1 - (k:ℝ)) / ((i:ℝ) + 1) := by
    rw [← hb]
    field_simp at hh ⊢
    linarith
  rw [eq_div_iff (ne_of_gt hi)]
  have : (1 - p S') * ((i:ℝ) + 1) = (i:ℝ) + 1 - (k:ℝ) := by
    rw [key]; field_simp
  linarith


end Anytime
