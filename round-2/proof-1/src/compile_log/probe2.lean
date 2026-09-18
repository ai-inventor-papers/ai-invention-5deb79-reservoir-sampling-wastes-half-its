import Mathlib
open Finset

/-! ## §5 L4 — per-step entropy bound (Theorem B per acceptance) -/

/-- Shannon entropy in bits of a finitely supported weight function. -/
noncomputable def H2 {α : Type*} (s : Finset α) (q : α → ℝ) : ℝ :=
  (∑ a ∈ s, Real.negMulLog (q a)) / Real.log 2

/-- **Max-entropy (self-contained elementary proof).** `H(q) ≤ log |support|` in nats.
Route taken: NOT Mathlib's measure-theoretic `measureEntropy_le_log_card` (which would drag
a `PMF`/`Measure` into the file) and NOT Jensen; the elementary `log x ≤ x - 1` argument. -/
theorem entropy_le_log_card {α : Type*} [DecidableEq α] (s : Finset α) (q : α → ℝ)
    (hq : ∀ a ∈ s, 0 ≤ q a) (hsum : ∑ a ∈ s, q a = 1) :
    ∑ a ∈ s, Real.negMulLog (q a) ≤ Real.log s.card := by
  classical
  set t : Finset α := s.filter (fun a => 0 < q a) with ht
  have htsub : t ⊆ s := Finset.filter_subset _ _
  -- the zero atoms contribute nothing
  have hrestr : ∑ a ∈ s, Real.negMulLog (q a) = ∑ a ∈ t, Real.negMulLog (q a) := by
    refine (Finset.sum_subset htsub ?_).symm
    intro a ha hat
    have h0 : q a = 0 := by
      have := hq a ha
      have hn : ¬ (0 < q a) := by
        intro hpos; exact hat (Finset.mem_filter.mpr ⟨ha, hpos⟩)
      linarith [le_of_not_lt hn]
    rw [h0, Real.negMulLog_zero]
  have hsumt : ∑ a ∈ t, q a = 1 := by
    rw [← hsum]
    refine (Finset.sum_subset htsub ?_).symm
    intro a ha hat
    have hn : ¬ (0 < q a) := by
      intro hpos; exact hat (Finset.mem_filter.mpr ⟨ha, hpos⟩)
    linarith [hq a ha, le_of_not_lt hn]
  -- s is nonempty (its weights sum to 1)
  have hsne : s.Nonempty := by
    rcases Finset.eq_empty_or_nonempty s with h | h
    · rw [h] at hsum; simp at hsum
    · exact h
  have hn1 : (1:ℝ) ≤ (s.card : ℝ) := by
    exact_mod_cast Finset.card_pos.mpr hsne
  have hnpos : (0:ℝ) < (s.card : ℝ) := by linarith
  -- per-atom bound
  have hterm : ∀ a ∈ t, Real.negMulLog (q a) ≤ q a * Real.log (s.card) + 1 / (s.card:ℝ) - q a := by
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
  have hsumbound : ∑ a ∈ t, Real.negMulLog (q a)
      ≤ ∑ a ∈ t, (q a * Real.log (s.card) + 1 / (s.card:ℝ) - q a) :=
    Finset.sum_le_sum hterm
  have hrhs : ∑ a ∈ t, (q a * Real.log (s.card) + 1 / (s.card:ℝ) - q a)
      = Real.log (s.card) + (t.card : ℝ) / (s.card:ℝ) - 1 := by
    have e1 : ∑ a ∈ t, (q a * Real.log (s.card) + 1 / (s.card:ℝ) - q a)
        = ((∑ a ∈ t, q a) * Real.log (s.card) + (t.card : ℝ) * (1 / (s.card:ℝ)))
          - ∑ a ∈ t, q a := by
      rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_const,
        nsmul_eq_mul]
    rw [e1, hsumt]
    ring
  have htc : (t.card : ℝ) ≤ (s.card : ℝ) := by
    exact_mod_cast Finset.card_le_card htsub
  have hfrac : (t.card : ℝ) / (s.card:ℝ) ≤ 1 := by
    rw [div_le_one hnpos]; exact htc
  rw [hrhs] at hsumbound
  rw [hrestr]
  linarith [hsumbound, hfrac]

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

