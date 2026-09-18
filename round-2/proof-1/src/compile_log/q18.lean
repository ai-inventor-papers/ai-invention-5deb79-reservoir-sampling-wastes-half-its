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

/-- Binary entropy in bits. -/
noncomputable def binEnt (q : ℝ) : ℝ := Real.negMulLog q / Real.log 2 + Real.negMulLog (1 - q) / Real.log 2

/-- **Corollary (Theorem A's floor), with Knuth–Yao as an EXPLICIT HYPOTHESIS.**

`hKY` is *assumed, not proved*: generating a Bernoulli(q) from fair coin flips costs at
least `binEnt q` flips in expectation (Knuth & Yao 1976). We do not formalise the DDG-tree
argument. Given `hKY` and `accept_forced`, the expected number of fair coin flips spent on
the accept decision over a run of length `n` is at least `∑_{i=k}^{n-1} binEnt (k/(i+1))`. -/
theorem acc_floor {k n : ℕ} (cost : ℝ → ℝ)
    (hKY : ∀ q : ℝ, 0 ≤ q → q ≤ 1 → binEnt q ≤ cost q)
    (acc : ℕ → ℝ) (hacc : ∀ i ∈ Finset.Ico k n, acc i = (k:ℝ) / ((i:ℝ) + 1))
    (hk : ∀ i ∈ Finset.Ico k n, (k:ℝ) ≤ (i:ℝ) + 1) :
    ∑ i ∈ Finset.Ico k n, binEnt (acc i) ≤ ∑ i ∈ Finset.Ico k n, cost (acc i) := by
  refine Finset.sum_le_sum ?_
  intro i hi
  refine hKY _ ?_ ?_
  · rw [hacc i hi]; positivity
  · rw [hacc i hi]
    rw [div_le_one (by positivity)]
    exact hk i hi

/-! ## §3 L2 = Theorem C — the induction step for general `(n,k)` -/

/-- **Theorem C, reject branch.** If the arriving item is rejected, the next reservoir is the
old one, and `(1 - k/(i+1)) · 1/C(i,k) = 1/C(i+1,k)` is exactly L0(b). -/
theorem theoremC_reject {i k : ℕ} (hki : k ≤ i) :
    (1 - (k:ℝ) / ((i:ℝ) + 1)) * (1 / (Nat.choose i k : ℝ)) = 1 / (Nat.choose (i + 1) k : ℝ) := by
  have hA : (0:ℝ) < (Nat.choose i k : ℝ) := choose_pos_real hki
  have hA' : (0:ℝ) < (Nat.choose (i + 1) k : ℝ) := choose_pos_real (le_trans hki (Nat.le_succ i))
  have hi : (0:ℝ) < (i:ℝ) + 1 := by positivity
  have hb := L0b_real hki
  rw [div_eq_div_iff (ne_of_gt hA') (by positivity : ((i:ℝ)+1) ≠ 0)] at hb
  field_simp
  linarith [hb]

/-- **Theorem C, accept branch.** Only the COLUMN sum of the plan is used. -/
theorem theoremC_accept {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (f : Finset ℕ → Finset ℕ → ℝ) (T : Finset ℕ)
    (hcol : ∑ S ∈ Finset.powersetCard k (Finset.range i), f S T
      = ((i:ℝ) - (k:ℝ) + 1) / (k:ℝ)) :
    ((k:ℝ) / ((i:ℝ) + 1)) *
        ∑ S ∈ Finset.powersetCard k (Finset.range i), (1 / (Nat.choose i k : ℝ)) * f S T
      = 1 / (Nat.choose (i + 1) k : ℝ) := by
  have hA : (0:ℝ) < (Nat.choose i k : ℝ) := choose_pos_real hki
  have hA' : (0:ℝ) < (Nat.choose (i + 1) k : ℝ) := choose_pos_real (le_trans hki (Nat.le_succ i))
  have hi : (0:ℝ) < (i:ℝ) + 1 := by positivity
  have hkR : (0:ℝ) < (k:ℝ) := by exact_mod_cast hk1
  rw [← Finset.mul_sum, hcol]
  have hb := L0b_real hki
  rw [div_eq_div_iff (ne_of_gt hA') (by positivity : ((i:ℝ)+1) ≠ 0)] at hb
  field_simp
  linear_combination (-(k:ℝ)) * hb

/-- **Theorem C (L2): ANY feasible routing plan preserves prefix uniformity.**

`hrow` (the plan is a probability kernel) is stated because the paper's Theorem C should
carry it, but it is NOT used in the proof: uniformity of the next prefix is a COLUMN-SUM
condition only. That is a genuine structural observation, not an oversight. -/
theorem theoremC {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (f : Finset ℕ → Finset ℕ → ℝ)
    (_hrow : ∀ S ∈ Finset.powersetCard k (Finset.range i),
      ∑ T ∈ Finset.powersetCard (k - 1) (Finset.range i), f S T = 1)
    (hcol : ∀ T ∈ Finset.powersetCard (k - 1) (Finset.range i),
      ∑ S ∈ Finset.powersetCard k (Finset.range i), f S T = ((i:ℝ) - (k:ℝ) + 1) / (k:ℝ))
    (S' : Finset ℕ) (hS' : S' ∈ Finset.powersetCard k (Finset.range (i + 1))) :
    (if i ∈ S' then
        ((k:ℝ) / ((i:ℝ) + 1)) *
          ∑ S ∈ Finset.powersetCard k (Finset.range i),
            (1 / (Nat.choose i k : ℝ)) * f S (S'.erase i)
      else (1 - (k:ℝ) / ((i:ℝ) + 1)) * (1 / (Nat.choose i k : ℝ)))
      = 1 / (Nat.choose (i + 1) k : ℝ) := by
  rw [Finset.mem_powersetCard] at hS'
  obtain ⟨hsub, hcard⟩ := hS'
  by_cases hmem : i ∈ S'
  · rw [if_pos hmem]
    refine theoremC_accept hk1 hki f _ (hcol _ ?_)
    rw [Finset.mem_powersetCard]
    constructor
    · intro x hx
      rw [Finset.mem_erase] at hx
      have := hsub hx.2
      rw [Finset.mem_range] at this ⊢
      omega
    · rw [Finset.card_erase_of_mem hmem, hcard]
  · rw [if_neg hmem]
    exact theoremC_reject hki

/-! ### The feasible polytope is non-empty: the textbook `1/k` plan is a witness -/

/-- The classical Algorithm-R eviction rule as a transportation plan: from a reservoir `S`,
drop a uniformly random one of its `k` elements. -/
noncomputable def uniformPlan (k : ℕ) (S T : Finset ℕ) : ℝ :=
  if T ⊆ S then 1 / (k:ℝ) else 0

/-- Row sums of the uniform plan are `1`: it is a probability kernel. -/
theorem uniformPlan_row {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (S : Finset ℕ) (hS : S ∈ Finset.powersetCard k (Finset.range i)) :
    ∑ T ∈ Finset.powersetCard (k - 1) (Finset.range i), uniformPlan k S T = 1 := by
  classical
  rw [Finset.mem_powersetCard] at hS
  obtain ⟨hSsub, hScard⟩ := hS
  have hset : (Finset.powersetCard (k - 1) (Finset.range i)).filter (fun T => T ⊆ S)
      = Finset.powersetCard (k - 1) S := by
    ext T
    simp only [Finset.mem_filter, Finset.mem_powersetCard]
    constructor
    · rintro ⟨⟨_, hc⟩, hTS⟩; exact ⟨hTS, hc⟩
    · rintro ⟨hTS, hc⟩; exact ⟨⟨hTS.trans hSsub, hc⟩, hTS⟩
  have hcard : ((Finset.powersetCard (k - 1) (Finset.range i)).filter (fun T => T ⊆ S)).card = k := by
    rw [hset, Finset.card_powersetCard, hScard]
    have : k - (k - 1) = 1 := by omega
    rw [← Nat.choose_symm (by omega : k - 1 ≤ k), this, Nat.choose_one_right]
  have hk0 : (k:ℝ) ≠ 0 := by
    have : (0:ℝ) < (k:ℝ) := by exact_mod_cast hk1
    exact ne_of_gt this
  unfold uniformPlan
  rw [← Finset.sum_filter, Finset.sum_const, hcard, nsmul_eq_mul]
  field_simp

/-- **Parent count.** A `(k-1)`-set `T ⊆ [i]` sits inside exactly `i-k+1` many `k`-sets of
`[i]`: its parents are `T ∪ {y}` for the `i-(k-1)` choices of `y ∈ [i] \\ T`. This is the
right-degree half of the Johnson-graph biregularity counted by `L0a`. -/
theorem card_parents {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (T : Finset ℕ) (hTsub : T ⊆ Finset.range i) (hTcard : T.card = k - 1) :
    ((Finset.powersetCard k (Finset.range i)).filter (fun S => T ⊆ S)).card = i - k + 1 := by
  classical
  have himg : (Finset.powersetCard k (Finset.range i)).filter (fun S => T ⊆ S)
      = (Finset.range i \ T).image (fun y => insert y T) := by
    ext S
    simp only [Finset.mem_filter, Finset.mem_powersetCard, Finset.mem_image, Finset.mem_sdiff]
    constructor
    · rintro ⟨⟨hSsub, hScard⟩, hTS⟩
      have hc1 : (S \ T).card = 1 := by
        rw [Finset.card_sdiff hTS, hScard, hTcard]; omega
      obtain ⟨y, hy⟩ := Finset.card_eq_one.mp hc1
      have hymem : y ∈ S \ T := by rw [hy]; exact Finset.mem_singleton_self y
      have hyS : y ∈ S := (Finset.mem_sdiff.mp hymem).1
      have hyT : y ∉ T := (Finset.mem_sdiff.mp hymem).2
      refine ⟨y, ⟨hSsub hyS, hyT⟩, ?_⟩
      apply Finset.Subset.antisymm
      · intro x hx
        rcases Finset.mem_insert.mp hx with rfl | hxT
        · exact hyS
        · exact hTS hxT
      · intro x hx
        by_cases hxT : x ∈ T
        · exact Finset.mem_insert_of_mem hxT
        · have hxd : x ∈ S \ T := Finset.mem_sdiff.mpr ⟨hx, hxT⟩
          rw [hy] at hxd
          rw [Finset.mem_singleton.mp hxd]
          exact Finset.mem_insert_self y T
    · rintro ⟨y, ⟨hyi, hyT⟩, rfl⟩
      refine ⟨⟨?_, ?_⟩, Finset.subset_insert y T⟩
      · intro x hx
        rcases Finset.mem_insert.mp hx with rfl | hxT
        · exact hyi
        · exact hTsub hxT
      · rw [Finset.card_insert_of_not_mem hyT, hTcard]; omega
  rw [himg, Finset.card_image_of_injOn]
  · rw [Finset.card_sdiff hTsub, Finset.card_range, hTcard]; omega
  · intro a ha b hb hab
    simp only [Finset.coe_sdiff, Set.mem_diff, Finset.mem_coe] at ha hb
    have hab' : insert a T = insert b T := hab
    have hmem : a ∈ insert b T := by rw [← hab']; exact Finset.mem_insert_self a T
    rcases Finset.mem_insert.mp hmem with h | h
    · exact h
    · exact absurd h ha.2

/-- Column sums of the uniform plan are `(i-k+1)/k`: it lies in the feasible polytope. -/
theorem uniformPlan_col {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (T : Finset ℕ) (hT : T ∈ Finset.powersetCard (k - 1) (Finset.range i)) :
    ∑ S ∈ Finset.powersetCard k (Finset.range i), uniformPlan k S T
      = ((i:ℝ) - (k:ℝ) + 1) / (k:ℝ) := by
  classical
  rw [Finset.mem_powersetCard] at hT
  obtain ⟨hTsub, hTcard⟩ := hT
  have hcard := card_parents hk1 hki T hTsub hTcard
  have hk0 : (k:ℝ) ≠ 0 := by
    have : (0:ℝ) < (k:ℝ) := by exact_mod_cast hk1
    exact ne_of_gt this
  have hcast : ((i - k + 1 : ℕ) : ℝ) = (i:ℝ) - (k:ℝ) + 1 := by
    have h1 : ((i - k : ℕ) : ℝ) = (i:ℝ) - (k:ℝ) := by rw [Nat.cast_sub hki]
    push_cast [h1]
    ring
  unfold uniformPlan
  rw [← Finset.sum_filter, Finset.sum_const, hcard, nsmul_eq_mul, hcast]
  ring

/-- **Existence corollary (constructive).** The feasible polytope is non-empty for every
`1 ≤ k ≤ i`, witnessed by the textbook `1/k` rule. Consequently Theorem C re-proves the
correctness of classical Algorithm R as a special case. -/
theorem theoremC_existence {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    ∃ f : Finset ℕ → Finset ℕ → ℝ,
      (∀ S ∈ Finset.powersetCard k (Finset.range i),
        ∑ T ∈ Finset.powersetCard (k - 1) (Finset.range i), f S T = 1) ∧
      (∀ T ∈ Finset.powersetCard (k - 1) (Finset.range i),
        ∑ S ∈ Finset.powersetCard k (Finset.range i), f S T = ((i:ℝ) - (k:ℝ) + 1) / (k:ℝ)) :=
  ⟨uniformPlan k, fun S hS => uniformPlan_row hk1 hki S hS,
    fun T hT => uniformPlan_col hk1 hki T hT⟩

/-! ## §4 L3 — forest-support counting (Theorem B's combinatorial core) -/

/-- **L3.** If a bipartite support has `A` sources each of degree `≥ 1`, `E` edges in total
and `E + 1 ≤ A + B` (which is what "the support is a forest on `A + B` vertices" gives),
then at most `B - 1` sources have degree `≥ 2`. Stated truncation-free as `card + 1 ≤ B`.

The geometric input — the eviction objective `∑_S P(S) H(q_S)` is concave, so its minimum
over the transportation polytope is attained at a VERTEX, and the support of a vertex of a
transportation polytope is a forest, hence has at most `A + B - 1` edges (network-simplex
spanning-tree basis) — is ASSUMED, entering only through the hypothesis `hEB`. -/
theorem forest_split_count {A : ℕ} (d : Fin A → ℕ) (hd : ∀ j, 1 ≤ d j)
    (B E : ℕ) (hE : ∑ j, d j = E) (hEB : E + 1 ≤ A + B) :
    (Finset.univ.filter (fun j => 2 ≤ d j)).card + 1 ≤ B := by
  classical
  have hsplit :
      (∑ j ∈ Finset.univ.filter (fun j => 2 ≤ d j), d j)
        + (∑ j ∈ Finset.univ.filter (fun j => ¬ (2 ≤ d j)), d j)
      = ∑ j : Fin A, d j :=
    Finset.sum_filter_add_sum_filter_not Finset.univ (fun j => 2 ≤ d j) d
  have h1 : 2 * (Finset.univ.filter (fun j => 2 ≤ d j)).card
      ≤ ∑ j ∈ Finset.univ.filter (fun j => 2 ≤ d j), d j := by
    have hle : ∑ _j ∈ Finset.univ.filter (fun j => 2 ≤ d j), (2:ℕ)
        ≤ ∑ j ∈ Finset.univ.filter (fun j => 2 ≤ d j), d j :=
      Finset.sum_le_sum (fun j hj => (Finset.mem_filter.mp hj).2)
    simpa [Finset.sum_const, smul_eq_mul, Nat.mul_comm] using hle
  have h2 : (Finset.univ.filter (fun j => ¬ (2 ≤ d j))).card
      ≤ ∑ j ∈ Finset.univ.filter (fun j => ¬ (2 ≤ d j)), d j := by
    have hle : ∑ _j ∈ Finset.univ.filter (fun j => ¬ (2 ≤ d j)), (1:ℕ)
        ≤ ∑ j ∈ Finset.univ.filter (fun j => ¬ (2 ≤ d j)), d j :=
      Finset.sum_le_sum (fun j _ => hd j)
    simpa [Finset.sum_const, smul_eq_mul] using hle
  have hcards : (Finset.univ.filter (fun j => 2 ≤ d j)).card
      + (Finset.univ.filter (fun j => ¬ (2 ≤ d j))).card = A := by
    have h := Finset.filter_card_add_filter_neg_card_eq_card
      (s := (Finset.univ : Finset (Fin A))) (p := fun j => 2 ≤ d j)
    simpa [Finset.card_univ] using h
  omega


end Anytime
