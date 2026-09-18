import Mathlib
open Finset

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

