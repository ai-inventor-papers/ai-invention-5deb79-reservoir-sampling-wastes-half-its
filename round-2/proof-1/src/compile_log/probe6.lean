import Mathlib
open Finset

/-- Shannon entropy in bits of a finitely supported weight function. -/
noncomputable def H2 {α : Type*} (s : Finset α) (q : α → ℝ) : ℝ :=
  (∑ a ∈ s, Real.negMulLog (q a)) / Real.log 2

theorem entropy_le_log_card {α : Type*} [DecidableEq α] (s : Finset α) (q : α → ℝ)
    (hq : ∀ a ∈ s, 0 ≤ q a) (hsum : ∑ a ∈ s, q a = 1) :
    ∑ a ∈ s, Real.negMulLog (q a) ≤ Real.log s.card := by sorry

/-- Max-entropy in BITS. -/
theorem entropy_le_logb_card {α : Type*} [DecidableEq α] (s : Finset α) (q : α → ℝ)
    (hq : ∀ a ∈ s, 0 ≤ q a) (hsum : ∑ a ∈ s, q a = 1) :
    H2 s q ≤ Real.logb 2 s.card := by
  have h := entropy_le_log_card s q hq hsum
  have hl2 : (0:ℝ) < Real.log 2 := Real.log_pos (by norm_num)
  rw [H2, Real.logb, div_le_div_iff hl2 hl2]
  nlinarith [h, hl2]

