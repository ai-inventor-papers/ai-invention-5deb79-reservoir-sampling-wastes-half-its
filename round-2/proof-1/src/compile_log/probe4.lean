import Mathlib
open Finset

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

