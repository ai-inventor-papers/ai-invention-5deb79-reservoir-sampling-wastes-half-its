import Mathlib
open Finset


example (t : Finset ℕ) (f g : ℕ → ℝ) : ∑ a ∈ t, (f a - g a) = (∑ a ∈ t, f a) - ∑ a ∈ t, g a := Finset.sum_sub_distrib
example (t : Finset ℕ) (f g : ℕ → ℝ) : ∑ a ∈ t, (f a + g a) = (∑ a ∈ t, f a) + ∑ a ∈ t, g a := Finset.sum_add_distrib
example (t : Finset ℕ) (f : ℕ → ℝ) (c d : ℝ) :
    ∑ a ∈ t, (f a * c + d - f a) = ((∑ a ∈ t, f a) * c + (t.card : ℝ) * d) - ∑ a ∈ t, f a := by
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, ← Finset.sum_mul, Finset.sum_const,
    nsmul_eq_mul]
