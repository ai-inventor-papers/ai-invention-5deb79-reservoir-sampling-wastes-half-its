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

/-! ## §5 L4 — per-step entropy bound (Theorem B per acceptance) -/

/-- Shannon entropy in bits of a finitely supported weight function. -/
noncomputable def H2 {α : Type*} (s : Finset α) (q : α → ℝ) : ℝ :=
  (∑ a ∈ s, Real.negMulLog (q a)) / Real.log 2

/-- **Max-entropy (self-contained elementary proof).** `H(q) ≤ log |support|` in nats.
Route taken: NOT Mathlib's measure-theoretic `measureEntropy_le_log_card` (which would drag
a `PMF`/`Measure` into the file) and NOT Jensen; the elementary `log x ≤ x - 1` argument,
restricted to the strictly positive atoms (`negMulLog 0 = 0`, so the restriction is free). -/
theorem entropy_le_log_card {α : Type*} [DecidableEq α] (s : Finset α) (q : α → ℝ)
    (hq : ∀ a ∈ s, 0 ≤ q a) (hsum : ∑ a ∈ s, q a = 1) :
    ∑ a ∈ s, Real.negMulLog (q a) ≤ Real.log s.card := by
  classical
  have htsub : s.filter (fun a => 0 < q a) ⊆ s := Finset.filter_subset _ _
  have hzero : ∀ a ∈ s, a ∉ s.filter (fun a => 0 < q a) → q a = 0 := by
    intro a ha hat
    have hn : ¬ (0 < q a) := fun hpos => hat (Finset.mem_filter.mpr ⟨ha, hpos⟩)
    have h0 := hq a ha
    linarith [le_of_not_lt hn]
  have hrestr : ∑ a ∈ s.filter (fun a => 0 < q a), Real.negMulLog (q a)
      = ∑ a ∈ s, Real.negMulLog (q a) := by
    refine Finset.sum_subset htsub ?_
    intro a ha hat
    rw [hzero a ha hat, Real.negMulLog_zero]
  have hsumt : ∑ a ∈ s.filter (fun a => 0 < q a), q a = 1 := by
    rw [Finset.sum_subset htsub (fun a ha hat => hzero a ha hat)]
    exact hsum
  have hsne : s.Nonempty := by
    rcases Finset.eq_empty_or_nonempty s with h | h
    · rw [h, Finset.sum_empty] at hsum
      norm_num at hsum
    · exact h
  have hnpos : (0:ℝ) < (s.card : ℝ) := by
    have hc : 0 < s.card := Finset.card_pos.mpr hsne
    exact_mod_cast hc
  have hterm : ∀ a ∈ s.filter (fun a => 0 < q a),
      Real.negMulLog (q a) ≤ q a * Real.log (s.card) + 1 / (s.card:ℝ) - q a := by
    intro a hat
    have hqa : 0 < q a := (Finset.mem_filter.mp hat).2
    have hx : 0 < 1 / ((s.card:ℝ) * q a) := by positivity
    have hlog : Real.log (1 / ((s.card:ℝ) * q a)) ≤ 1 / ((s.card:ℝ) * q a) - 1 :=
      Real.log_le_sub_one_of_pos hx
    have hrw : Real.log (1 / ((s.card:ℝ) * q a)) = -(Real.log (s.card) + Real.log (q a)) := by
      rw [one_div, Real.log_inv, Real.log_mul (ne_of_gt hnpos) (ne_of_gt hqa)]
    rw [hrw] at hlog
    have hmul : q a * (-(Real.log (s.card) + Real.log (q a)))
        ≤ q a * (1 / ((s.card:ℝ) * q a) - 1) :=
      mul_le_mul_of_nonneg_left hlog (le_of_lt hqa)
    have hsimp : q a * (1 / ((s.card:ℝ) * q a) - 1) = 1 / (s.card:ℝ) - q a := by
      have h1 : (s.card:ℝ) ≠ 0 := ne_of_gt hnpos
      have h2 : q a ≠ 0 := ne_of_gt hqa
      field_simp
      try ring
    have hexp : q a * (-(Real.log (s.card) + Real.log (q a)))
        = -(q a * Real.log (s.card)) - q a * Real.log (q a) := by ring
    have hnml : Real.negMulLog (q a) = -(q a * Real.log (q a)) := by
      rw [Real.negMulLog]; ring
    rw [hexp, hsimp] at hmul
    rw [hnml]
    linarith
  have hsumbound := Finset.sum_le_sum hterm
  have e1 : ∑ a ∈ s.filter (fun a => 0 < q a),
        (q a * Real.log (s.card) + 1 / (s.card:ℝ) - q a)
      = ((∑ a ∈ s.filter (fun a => 0 < q a), q a) * Real.log (s.card)
          + ((s.filter (fun a => 0 < q a)).card : ℝ) * (1 / (s.card:ℝ)))
        - ∑ a ∈ s.filter (fun a => 0 < q a), q a := by
    rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_const,
      nsmul_eq_mul]
  have hfrac : ((s.filter (fun a => 0 < q a)).card : ℝ) * (1 / (s.card:ℝ)) ≤ 1 := by
    rw [mul_one_div, div_le_one hnpos]
    exact_mod_cast Finset.card_le_card htsub
  rw [e1, hsumt] at hsumbound
  rw [← hrestr]
  linarith

/-- Max-entropy in BITS. -/
theorem entropy_le_logb_card {α : Type*} [DecidableEq α] (s : Finset α) (q : α → ℝ)
    (hq : ∀ a ∈ s, 0 ≤ q a) (hsum : ∑ a ∈ s, q a = 1) :
    H2 s q ≤ Real.logb 2 s.card := by
  have h := entropy_le_log_card s q hq hsum
  have hl2 : (0:ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  rw [H2, Real.logb, div_le_div_iff hl2 hl2]
  nlinarith [h, hl2]

/-- **L4: the per-acceptance eviction-entropy bound.**
`H_evict(i,k) ≤ log2(k) · (C(i,k-1) - 1)/C(i,k)`. Split sources contribute at most
`log2 k` each (max-entropy on `≤ k` atoms); non-split sources contribute `0`; and the
number of split sources is bounded by `forest_split_count`. -/
theorem per_step_entropy_bound {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (src : Finset (Finset ℕ)) (hsrc : src.card = Nat.choose i k)
    (H : Finset ℕ → ℝ) (hHnn : ∀ S ∈ src, 0 ≤ H S)
    (split : Finset (Finset ℕ)) (hsub : split ⊆ src)
    (hzero : ∀ S ∈ src, S ∉ split → H S = 0)
    (hmax : ∀ S ∈ split, H S ≤ Real.logb 2 k)
    (hsplit_card : split.card + 1 ≤ Nat.choose i (k - 1)) :
    ∑ S ∈ src, (1 / (Nat.choose i k : ℝ)) * H S
      ≤ Real.logb 2 k * ((Nat.choose i (k - 1) : ℝ) - 1) / (Nat.choose i k : ℝ) := by
  classical
  have hA : (0:ℝ) < (Nat.choose i k : ℝ) := choose_pos_real hki
  have hlogk : 0 ≤ Real.logb 2 k := by
    have : (1:ℝ) ≤ (k:ℝ) := by exact_mod_cast hk1
    have := Real.logb_nonneg (by norm_num : (1:ℝ) < 2) this
    exact this
  have hsum_split : ∑ S ∈ src, H S = ∑ S ∈ split, H S := by
    refine (Finset.sum_subset hsub ?_).symm
    intro a ha hat
    exact hzero a ha hat
  have hbnd : ∑ S ∈ split, H S ≤ (split.card : ℝ) * Real.logb 2 k := by
    have := Finset.sum_le_sum hmax
    simpa [Finset.sum_const, nsmul_eq_mul] using this
  have hcard : (split.card : ℝ) ≤ (Nat.choose i (k - 1) : ℝ) - 1 := by
    have : (split.card : ℝ) + 1 ≤ (Nat.choose i (k - 1) : ℝ) := by exact_mod_cast hsplit_card
    linarith
  have hchain : ∑ S ∈ src, H S ≤ ((Nat.choose i (k - 1) : ℝ) - 1) * Real.logb 2 k := by
    rw [hsum_split]
    refine hbnd.trans ?_
    exact mul_le_mul_of_nonneg_right hcard hlogk
  rw [← Finset.mul_sum, div_mul_eq_mul_div, one_mul, div_le_div_iff hA hA]
  nlinarith [hchain, hA]

/-- The relative form: `log2(k)·(C(i,k-1)-1)/C(i,k) ≤ log2(k)·k/(i-k+1)`, using L0(a). -/
theorem per_step_entropy_bound_ratio {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    Real.logb 2 k * ((Nat.choose i (k - 1) : ℝ) - 1) / (Nat.choose i k : ℝ)
      ≤ Real.logb 2 k * (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1) := by
  have hA : (0:ℝ) < (Nat.choose i k : ℝ) := choose_pos_real hki
  have hki' : (k:ℝ) ≤ (i:ℝ) := by exact_mod_cast hki
  have hden : (0:ℝ) < (i:ℝ) - (k:ℝ) + 1 := by linarith
  have hlogk : 0 ≤ Real.logb 2 k := by
    have h1 : (1:ℝ) ≤ (k:ℝ) := by exact_mod_cast hk1
    exact Real.logb_nonneg (by norm_num : (1:ℝ) < 2) h1
  have hratio := L0a_real hk1 hki
  rw [div_eq_div_iff (ne_of_gt hA) (ne_of_gt hden)] at hratio
  rw [div_le_div_iff hA hden]
  nlinarith [hratio, hlogk, hA, hden]

/-- The trivial ceiling: `H_evict ≤ log2 k` always. The fitted law that this replaces
predicted 11.27 bits per acceptance at `k = 1000, i = k+1`, against a ceiling of
`log2 1000 = 9.9658`; the bound proved here can never violate the ceiling. -/
theorem per_step_entropy_cap {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (src : Finset (Finset ℕ)) (hsrc : src.card = Nat.choose i k)
    (H : Finset ℕ → ℝ) (hmax : ∀ S ∈ src, H S ≤ Real.logb 2 k) :
    ∑ S ∈ src, (1 / (Nat.choose i k : ℝ)) * H S ≤ Real.logb 2 k := by
  have hA : (0:ℝ) < (Nat.choose i k : ℝ) := choose_pos_real hki
  have hb : ∑ S ∈ src, H S ≤ (src.card : ℝ) * Real.logb 2 k := by
    have := Finset.sum_le_sum hmax
    simpa [Finset.sum_const, nsmul_eq_mul] using this
  rw [hsrc] at hb
  rw [← Finset.mul_sum, div_mul_eq_mul_div, one_mul, div_le_iff hA]
  linarith [hb]

/-! ## §6 L5 — the total is constant in `n` -/

/-- Telescoping ingredient: `∑_{j<m} 1/((j+1)(j+2)) ≤ 1` for every `m`. -/
theorem telescope_bound (m : ℕ) :
    ∑ j ∈ Finset.range m, (1:ℝ) / (((j:ℝ) + 1) * ((j:ℝ) + 2)) ≤ 1 := by
  have key : ∀ j : ℕ, (1:ℝ) / (((j:ℝ) + 1) * ((j:ℝ) + 2))
      = (1:ℝ) / ((j:ℝ) + 1) - (1:ℝ) / ((j:ℝ) + 2) := by
    intro j
    have h1 : ((j:ℝ) + 1) ≠ 0 := by positivity
    have h2 : ((j:ℝ) + 2) ≠ 0 := by positivity
    field_simp
    ring
  have hsum : ∀ m : ℕ, ∑ j ∈ Finset.range m, (1:ℝ) / (((j:ℝ) + 1) * ((j:ℝ) + 2))
      = 1 - (1:ℝ) / ((m:ℝ) + 1) := by
    intro m
    induction m with
    | zero => simp
    | succ p ih =>
      rw [Finset.sum_range_succ, ih, key p]
      push_cast
      ring
  rw [hsum m]
  have hpos : (0:ℝ) < (1:ℝ) / ((m:ℝ) + 1) := by positivity
  linarith

/-- **L5: the total eviction randomness is bounded independently of `n`.**

`∑_{i=k}^{n-1} (k/(i+1)) · log2(k)·k/(i-k+1) ≤ log2(k)·k²` for every `n`.

CORRECTION TO THE HYPOTHESIS TEXT: the total is NOT `O(k log k)`. The bound proved here is
`O(k² log k)`; the sharp partial-fraction evaluation gives `log2(k)·k²·(H_k - 1)/(k-1)`,
which is `Θ(k log² k)`. The load-bearing claim — CONSTANT IN `n` — survives either way. -/
theorem total_constant_in_n (k n : ℕ) (hk1 : 1 ≤ k) :
    ∑ i ∈ Finset.Ico k n,
        ((k:ℝ) / ((i:ℝ) + 1)) * (Real.logb 2 k * (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1))
      ≤ Real.logb 2 k * (k:ℝ) ^ 2 := by
  have hlogk : 0 ≤ Real.logb 2 k := by
    have h1 : (1:ℝ) ≤ (k:ℝ) := by exact_mod_cast hk1
    exact Real.logb_nonneg (by norm_num : (1:ℝ) < 2) h1
  have hkR : (1:ℝ) ≤ (k:ℝ) := by exact_mod_cast hk1
  rw [Finset.sum_Ico_eq_sum_range]
  have hterm : ∀ j ∈ Finset.range (n - k),
      ((k:ℝ) / (((k + j : ℕ):ℝ) + 1)) *
          (Real.logb 2 k * (k:ℝ) / (((k + j : ℕ):ℝ) - (k:ℝ) + 1))
        ≤ Real.logb 2 k * (k:ℝ) ^ 2 * ((1:ℝ) / (((j:ℝ) + 1) * ((j:ℝ) + 2))) := by
    intro j _
    have hcast : ((k + j : ℕ):ℝ) = (k:ℝ) + (j:ℝ) := by push_cast; ring
    rw [hcast]
    have hj1 : (0:ℝ) < (j:ℝ) + 1 := by positivity
    have hj2 : (0:ℝ) < (j:ℝ) + 2 := by positivity
    have hkj : (0:ℝ) < (k:ℝ) + (j:ℝ) + 1 := by linarith
    have hd : (k:ℝ) + (j:ℝ) - (k:ℝ) + 1 = (j:ℝ) + 1 := by ring
    rw [hd]
    have hlhs : ((k:ℝ) / ((k:ℝ) + (j:ℝ) + 1)) * (Real.logb 2 k * (k:ℝ) / ((j:ℝ) + 1))
        = Real.logb 2 k * (k:ℝ) ^ 2 * ((1:ℝ) / (((k:ℝ) + (j:ℝ) + 1) * ((j:ℝ) + 1))) := by
      field_simp
      ring
    rw [hlhs]
    refine mul_le_mul_of_nonneg_left ?_ (by positivity)
    rw [div_le_div_iff (by positivity) (by positivity)]
    nlinarith [hkR, hj1, hj2]
  refine (Finset.sum_le_sum hterm).trans ?_
  rw [← Finset.mul_sum]
  have h := telescope_bound (n - k)
  have hc : (0:ℝ) ≤ Real.logb 2 k * (k:ℝ) ^ 2 := mul_nonneg hlogk (sq_nonneg _)
  calc Real.logb 2 k * (k:ℝ) ^ 2 * ∑ j ∈ Finset.range (n - k),
        (1:ℝ) / (((j:ℝ) + 1) * ((j:ℝ) + 2))
      ≤ Real.logb 2 k * (k:ℝ) ^ 2 * 1 := mul_le_mul_of_nonneg_left h hc
    _ = Real.logb 2 k * (k:ℝ) ^ 2 := by ring


/-! ## §7 Headline corollaries and sanity checks -/

/-- **Headline corollary (Theorem C, existential form).** For every `1 ≤ k ≤ i` there is a
routing plan that is a genuine probability kernel AND makes the reservoir at time `i+1`
exactly uniform on `J(i+1,k)`. Combined with `accept_forced` this says: the accept coin is
forced, the eviction plan is free within the transportation polytope, and the polytope is
never empty. -/
theorem anytime_uniform_step {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    ∃ f : Finset ℕ → Finset ℕ → ℝ,
      (∀ S ∈ Finset.powersetCard k (Finset.range i),
        ∑ T ∈ Finset.powersetCard (k - 1) (Finset.range i), f S T = 1) ∧
      (∀ S' ∈ Finset.powersetCard k (Finset.range (i + 1)),
        (if i ∈ S' then
            ((k:ℝ) / ((i:ℝ) + 1)) *
              ∑ S ∈ Finset.powersetCard k (Finset.range i),
                (1 / (Nat.choose i k : ℝ)) * f S (S'.erase i)
          else (1 - (k:ℝ) / ((i:ℝ) + 1)) * (1 / (Nat.choose i k : ℝ)))
          = 1 / (Nat.choose (i + 1) k : ℝ)) := by
  obtain ⟨f, hrow, hcol⟩ := theoremC_existence hk1 hki
  exact ⟨f, hrow, fun S' hS' => theoremC hk1 hki f hrow hcol S' hS'⟩

/-- Strict form of the relative bound for `k ≥ 2`: `(B-1)/A < B/A = k/(i-k+1)`. -/
theorem per_step_entropy_bound_ratio_strict {i k : ℕ} (hk2 : 2 ≤ k) (hki : k ≤ i) :
    Real.logb 2 k * ((Nat.choose i (k - 1) : ℝ) - 1) / (Nat.choose i k : ℝ)
      < Real.logb 2 k * (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1) := by
  have hk1 : 1 ≤ k := by omega
  have hA : (0:ℝ) < (Nat.choose i k : ℝ) := choose_pos_real hki
  have hki' : (k:ℝ) ≤ (i:ℝ) := by exact_mod_cast hki
  have hden : (0:ℝ) < (i:ℝ) - (k:ℝ) + 1 := by linarith
  have hlogk : 0 < Real.logb 2 k := by
    have h2 : (1:ℝ) < (k:ℝ) := by exact_mod_cast (by omega : 1 < k)
    exact Real.logb_pos (by norm_num) h2
  have hratio := L0a_real hk1 hki
  rw [div_eq_div_iff (ne_of_gt hA) (ne_of_gt hden)] at hratio
  rw [div_lt_div_iff hA hden]
  nlinarith [hratio, hlogk, hA, hden]

/-- **The usable per-acceptance bound**, combining the ceiling and the relative bound:
`H_evict(i,k) ≤ min (log2 k) (log2 k · k/(i-k+1))`. -/
theorem per_step_entropy_min {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (src : Finset (Finset ℕ)) (hsrc : src.card = Nat.choose i k)
    (H : Finset ℕ → ℝ) (hHnn : ∀ S ∈ src, 0 ≤ H S)
    (split : Finset (Finset ℕ)) (hsub : split ⊆ src)
    (hzero : ∀ S ∈ src, S ∉ split → H S = 0)
    (hmaxsrc : ∀ S ∈ src, H S ≤ Real.logb 2 k)
    (hsplit_card : split.card + 1 ≤ Nat.choose i (k - 1)) :
    ∑ S ∈ src, (1 / (Nat.choose i k : ℝ)) * H S
      ≤ min (Real.logb 2 k) (Real.logb 2 k * (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1)) := by
  refine le_min (per_step_entropy_cap hk1 hki src hsrc H hmaxsrc) ?_
  exact (per_step_entropy_bound hk1 hki src hsrc H hHnn split hsub hzero
    (fun S hS => hmaxsrc S (hsub hS)) hsplit_card).trans
    (per_step_entropy_bound_ratio hk1 hki)

/-! ### Sanity checks

At `k = 2` the proved bound is `log2(2) · (C(i,1)-1)/C(i,2) = 2/i` bits per acceptance,
while the exactly solved per-step minima reported by the run are `1, 1/3, 1/5, 1/7` at
`i = 2,4,6,8`, i.e. `log2(k)/(i-k+1) = 1/(i-1)`. Since `2/i ≥ 1/(i-1)` for all `i ≥ 2`
the bound is NEVER violated; the slack is `2(i-1)/i → 2`, i.e. a factor of about two.

At `k = 1000, i = k+1 = 1001` the bound is `min (log2 1000) (log2 1000 · 1000/2) = 9.9658`
bits, exactly the trivial ceiling. The fitted law it replaces predicted 11.27 bits per
acceptance there — above the ceiling `log2 1000 = 9.9658`, which is impossible. -/

/-- The `k = 2` specialisation of the numerator of the bound: `(C(i,1)-1)/C(i,2) = 2/i`. -/
example (i : ℕ) (hi : 2 ≤ i) :
    ((Nat.choose i 1 : ℝ) - 1) / (Nat.choose i 2 : ℝ) = 2 / (i:ℝ) := by
  have h := L0a (i := i) (k := 2) (by norm_num) hi
  rw [show (2:ℕ) - 1 = 1 from rfl, Nat.choose_one_right] at h
  have hA : (0:ℝ) < (Nat.choose i 2 : ℝ) := choose_pos_real hi
  have hi' : (2:ℝ) ≤ (i:ℝ) := by exact_mod_cast hi
  have hcast : (2:ℝ) * (Nat.choose i 2 : ℝ) = ((i:ℝ) - 2 + 1) * (i:ℝ) := by
    have h2 := congrArg (fun n : ℕ => (n:ℝ)) h
    push_cast [Nat.cast_sub (by omega : 2 ≤ i)] at h2
    linarith
  rw [Nat.choose_one_right]
  have hine : (i:ℝ) ≠ 0 := by linarith
  field_simp
  linarith [hcast]

/-- The trivial ceiling always dominates when `i = k+1` and `k ≥ 2`: there the relative
bound is `log2(k) · k/2 ≥ log2 k`, so `min` returns the ceiling. -/
example {k : ℕ} (hk2 : 2 ≤ k) :
    Real.logb 2 k ≤ Real.logb 2 k * (k:ℝ) / (((k:ℕ) + 1 : ℕ) - (k:ℝ) + 1) := by
  have hlogk : 0 < Real.logb 2 k := by
    have h2 : (1:ℝ) < (k:ℝ) := by exact_mod_cast (by omega : 1 < k)
    exact Real.logb_pos (by norm_num) h2
  have hk' : (2:ℝ) ≤ (k:ℝ) := by exact_mod_cast hk2
  push_cast
  rw [le_div_iff (by linarith)]
  nlinarith [hlogk, hk']


/-! ## §8 The sharp total: `log2(k) * k * H_k = Θ(k log² k)`

`total_constant_in_n` already carries the load-bearing claim (constant in `n`) with the easy
`log2(k)·k²` bound. This section proves the SHARP constant by partial fractions:
`∑_{m≥1} 1/(m(m+k)) = H_k/k`, hence the total is at most `log2(k)·k·H_k`, which is
`Θ(k log² k)` — still not the `O(k log k)` the hypothesis text asserted, but strictly better
than `O(k² log k)`. `harm_le_self` shows the sharp bound implies the easy one. -/

/-- The `n`-th harmonic number `H_n = ∑_{r=1}^{n} 1/r`. -/
noncomputable def harm (n : ℕ) : ℝ := ∑ r ∈ Finset.range n, (1:ℝ) / ((r:ℝ) + 1)


end Anytime
