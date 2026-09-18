# Per-lemma status table

Environment: Lean 4 **v4.14.0** + Mathlib (`lean-interact` `TempRequireProject`, REPL commit `a7ca4ec4683e9513127094308c6555555ab4c5d1`), single file, `import Mathlib`.

**All 40 declarations are sorry-free and verified** (38 named + 2 anonymous sanity `example`s). The `#print axioms` column is raw compiler output: every named declaration depends only on the three standard Lean axioms (`propext`, `Classical.choice`, `Quot.sound`) or a strict subset. No `sorryAx`; no user `axiom` is declared anywhere in the file — everything imported from outside is an explicit theorem hypothesis.

Timing = wall-clock elaboration of the file truncated at that declaration (CUMULATIVE; Mathlib already in the base environment; server-warm, so differences below ~0.1 s are noise and the series is not monotone). Whole-file elaboration, 3 repeats: **0.59 / 0.59 / 0.58 s**. The one-off `import Mathlib` build costs ~20 min and is paid once by `lean_worker.py`.

| # | name | sorry-free | `#print axioms` | cumulative compile (s) | key Mathlib lemmas | tactics |
|---|------|-----------|------------------|------------------------|--------------------|---------|
| 1 | `index_convention` | **yes** | `[propext, Quot.sound]` | 0.01 | — | `ring` |
| 2 | `index_convention_nat` | **yes** | `[propext, Quot.sound]` | 0.01 | — | `omega` |
| 3 | `L0a` | **yes** | `[propext, Quot.sound]` | 0.02 | `Nat.choose`, `Nat.choose_succ_right_eq` | `calc`, `omega`, `ring` |
| 4 | `L0b` | **yes** | `[propext]` | 0.02 | `Nat.choose`, `Nat.choose_mul_succ_eq` | `calc`, `ring` |
| 5 | `choose_pos_real` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.02 | `Nat.choose`, `Nat.choose_pos` | `exact_mod_cast` |
| 6 | `L0a_real` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.03 | `Nat.cast_add`, `Nat.cast_mul`, `Nat.cast_one`, `Nat.cast_sub`, `Nat.choose` | `exact_mod_cast`, `field_simp`, `linarith`, `simp` |
| 7 | `L0b_real` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.05 | `Nat.cast_add`, `Nat.cast_mul`, `Nat.cast_one`, `Nat.cast_sub`, `Nat.choose`, `Nat.le_succ` | `exact_mod_cast`, `linarith`, `simp` |
| 8 | `accept_forced` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.08 | `Finset.powersetCard`, `Finset.range`, `Nat.choose`, `Nat.le_succ` | `field_simp`, `linarith`, `positivity` |
| 9 | `binEnt` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.07 | `Real.log`, `Real.negMulLog` | — |
| 10 | `acc_floor` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.07 | `Finset.Ico`, `Finset.sum_le_sum` | `positivity` |
| 11 | `theoremC_reject` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.08 | `Nat.choose`, `Nat.le_succ` | `field_simp`, `induction`, `linarith`, `positivity` |
| 12 | `theoremC_accept` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.09 | `Finset.mul_sum`, `Finset.powersetCard`, `Finset.range`, `Nat.choose`, `Nat.le_succ` | `exact_mod_cast`, `field_simp`, `linear_combination`, `positivity` |
| 13 | `theoremC` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.1 | `Finset.card_erase_of_mem`, `Finset.mem_erase`, `Finset.mem_powersetCard`, `Finset.mem_range`, `Finset.powersetCard`, `Finset.range` | `omega` |
| 14 | `uniformPlan` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.1 | — | — |
| 15 | `uniformPlan_row` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.13 | `Finset.card_powersetCard`, `Finset.mem_filter`, `Finset.mem_powersetCard`, `Finset.powersetCard`, `Finset.range`, `Finset.sum_const` | `exact_mod_cast`, `field_simp`, `omega`, `simp` |
| 16 | `card_parents` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.17 | `Finset.Subset.antisymm`, `Finset.card_eq_one.mp`, `Finset.card_image_of_injOn`, `Finset.card_insert_of_not_mem`, `Finset.card_range`, `Finset.card_sdiff` | `omega`, `simp` |
| 17 | `uniformPlan_col` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.19 | `Finset.mem_powersetCard`, `Finset.powersetCard`, `Finset.range`, `Finset.sum_const`, `Finset.sum_filter`, `Nat.cast_sub` | `exact_mod_cast`, `push_cast`, `ring` |
| 18 | `theoremC_existence` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.19 | `Finset.powersetCard`, `Finset.range` | — |
| 19 | `forest_split_count` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.23 | `Finset.card_univ`, `Finset.filter_card_add_filter_neg_card_eq_card`, `Finset.mem_filter.mp`, `Finset.sum_const`, `Finset.sum_filter_add_sum_filter_not`, `Finset.sum_le_sum` | `omega`, `simpa` |
| 20 | `H2` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.23 | `Real.log`, `Real.negMulLog` | — |
| 21 | `entropy_le_log_card` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.32 | `Finset.card_le_card`, `Finset.card_pos.mpr`, `Finset.eq_empty_or_nonempty`, `Finset.filter_subset`, `Finset.mem_filter.mp`, `Finset.mem_filter.mpr` | `exact_mod_cast`, `field_simp`, `linarith`, `norm_num`, `positivity`, `ring` |
| 22 | `entropy_le_logb_card` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.32 | `Real.log`, `Real.log_pos`, `Real.logb` | `nlinarith`, `norm_num` |
| 23 | `per_step_entropy_bound` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.36 | `Finset.mul_sum`, `Finset.sum_const`, `Finset.sum_le_sum`, `Finset.sum_subset`, `Nat.choose`, `Real.logb` | `exact_mod_cast`, `linarith`, `nlinarith`, `norm_num`, `simpa` |
| 24 | `per_step_entropy_bound_ratio` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.36 | `Nat.choose`, `Real.logb`, `Real.logb_nonneg` | `exact_mod_cast`, `linarith`, `nlinarith`, `norm_num` |
| 25 | `per_step_entropy_cap` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.37 | `Finset.mul_sum`, `Finset.sum_const`, `Finset.sum_le_sum`, `Nat.choose`, `Real.logb` | `linarith`, `simpa` |
| 26 | `telescope_bound` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.39 | `Finset.range`, `Finset.sum_range_succ` | `field_simp`, `induction`, `linarith`, `positivity`, `push_cast`, `ring`, `simp` |
| 27 | `total_constant_in_n` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.45 | `Finset.Ico`, `Finset.mul_sum`, `Finset.range`, `Finset.sum_Ico_eq_sum_range`, `Finset.sum_le_sum`, `Real.logb` | `calc`, `exact_mod_cast`, `field_simp`, `linarith`, `nlinarith`, `norm_num`, `positivity`, `push_cast` |
| 28 | `anytime_uniform_step` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.45 | `Finset.powersetCard`, `Finset.range`, `Nat.choose` | — |
| 29 | `per_step_entropy_bound_ratio_strict` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.46 | `Nat.choose`, `Real.logb`, `Real.logb_pos` | `exact_mod_cast`, `linarith`, `nlinarith`, `norm_num`, `omega` |
| 30 | `per_step_entropy_min` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.46 | `Nat.choose`, `Real.logb` | — |
| 31 | `example_30` | **yes** | `n/a (anonymous `example`, not a named declaration)` | 0.48 | `Nat.cast_sub`, `Nat.choose`, `Nat.choose_one_right` | `exact_mod_cast`, `field_simp`, `linarith`, `norm_num`, `omega`, `push_cast` |
| 32 | `example_31` | **yes** | `n/a (anonymous `example`, not a named declaration)` | 0.7 | `Real.logb`, `Real.logb_pos` | `exact_mod_cast`, `linarith`, `nlinarith`, `norm_num`, `omega`, `push_cast` |
| 33 | `harm` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.5 | `Finset.range` | — |
| 34 | `harm_nonneg` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.53 | `Finset.sum_nonneg` | `positivity` |
| 35 | `harm_le_harm_add` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.57 | `Finset.range_subset.mpr`, `Finset.sum_le_sum_of_subset_of_nonneg` | `omega`, `positivity` |
| 36 | `harm_le_self` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.57 | `Finset.card_range`, `Finset.range`, `Finset.sum_const`, `Finset.sum_le_sum`, `Nat.cast_nonneg` | `linarith`, `positivity`, `simpa` |
| 37 | `harm_shift` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.55 | `Finset.range`, `Finset.sum_congr`, `Finset.sum_range_add` | `push_cast`, `ring`, `ring_nf` |
| 38 | `sum_inv_prod_le_harm_div` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.6 | `Finset.mul_sum`, `Finset.range`, `Finset.sum_congr`, `Finset.sum_sub_distrib`, `Nat.add_comm` | `calc`, `exact_mod_cast`, `field_simp`, `linarith`, `positivity`, `ring` |
| 39 | `total_sharp` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.75 | `Finset.Ico`, `Finset.mul_sum`, `Finset.range`, `Finset.sum_Ico_eq_sum_range`, `Finset.sum_congr`, `Real.logb` | `calc`, `exact_mod_cast`, `field_simp`, `norm_num`, `positivity`, `push_cast`, `ring` |
| 40 | `total_sharp_le_easy` | **yes** | `[propext, Classical.choice, Quot.sound]` | 0.94 | `Real.logb`, `Real.logb_nonneg` | `exact_mod_cast`, `nlinarith`, `norm_num` |

## Statements (verbatim from `Anytime.lean`)

### `index_convention`

```lean
theorem index_convention (i k : ℤ) : k ∣ ((i + 1) - (i - k + 1))
```

### `index_convention_nat`

```lean
theorem index_convention_nat {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    (i - k + 1) + k = i + 1
```

### `L0a`

```lean
theorem L0a {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    k * Nat.choose i k = (i - k + 1) * Nat.choose i (k - 1)
```

### `L0b`

```lean
theorem L0b (i k : ℕ) :
    (i + 1) * Nat.choose i k = (i + 1 - k) * Nat.choose (i + 1) k
```

### `choose_pos_real`

```lean
theorem choose_pos_real {i k : ℕ} (hki : k ≤ i) : (0:ℝ) < (Nat.choose i k : ℝ)
```

### `L0a_real`

```lean
theorem L0a_real {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    (Nat.choose i (k - 1) : ℝ) / (Nat.choose i k : ℝ) = (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1)
```

### `L0b_real`

```lean
theorem L0b_real {i k : ℕ} (hki : k ≤ i) :
    (Nat.choose i k : ℝ) / (Nat.choose (i + 1) k : ℝ)
      = ((i:ℝ) + 1 - (k:ℝ)) / ((i:ℝ) + 1)
```

### `accept_forced`

```lean
theorem accept_forced {i k : ℕ} (hki : k ≤ i) (p : Finset ℕ → ℝ)
    (h : ∀ S' ∈ Finset.powersetCard k (Finset.range i),
      (1 / (Nat.choose i k : ℝ)) * (1 - p S') = 1 / (Nat.choose (i + 1) k : ℝ)) :
    ∀ S' ∈ Finset.powersetCard k (Finset.range i), p S' = (k:ℝ) / ((i:ℝ) + 1)
```

### `binEnt`

```lean
noncomputable def binEnt (q : ℝ) : ℝ := Real.negMulLog q / Real.log 2 + Real.negMulLog (1 - q) / Real.log 2
```

### `acc_floor`

```lean
theorem acc_floor {k n : ℕ} (cost : ℝ → ℝ)
    (hKY : ∀ q : ℝ, 0 ≤ q → q ≤ 1 → binEnt q ≤ cost q)
    (acc : ℕ → ℝ) (hacc : ∀ i ∈ Finset.Ico k n, acc i = (k:ℝ) / ((i:ℝ) + 1))
    (hk : ∀ i ∈ Finset.Ico k n, (k:ℝ) ≤ (i:ℝ) + 1) :
    ∑ i ∈ Finset.Ico k n, binEnt (acc i) ≤ ∑ i ∈ Finset.Ico k n, cost (acc i)
```

### `theoremC_reject`

```lean
theorem theoremC_reject {i k : ℕ} (hki : k ≤ i) :
    (1 - (k:ℝ) / ((i:ℝ) + 1)) * (1 / (Nat.choose i k : ℝ)) = 1 / (Nat.choose (i + 1) k : ℝ)
```

### `theoremC_accept`

```lean
theorem theoremC_accept {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (f : Finset ℕ → Finset ℕ → ℝ) (T : Finset ℕ)
    (hcol : ∑ S ∈ Finset.powersetCard k (Finset.range i), f S T
      = ((i:ℝ) - (k:ℝ) + 1) / (k:ℝ)) :
    ((k:ℝ) / ((i:ℝ) + 1)) *
        ∑ S ∈ Finset.powersetCard k (Finset.range i), (1 / (Nat.choose i k : ℝ)) * f S T
      = 1 / (Nat.choose (i + 1) k : ℝ)
```

### `theoremC`

```lean
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
      = 1 / (Nat.choose (i + 1) k : ℝ)
```

### `uniformPlan`

```lean
noncomputable def uniformPlan (k : ℕ) (S T : Finset ℕ) : ℝ
```

### `uniformPlan_row`

```lean
theorem uniformPlan_row {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (S : Finset ℕ) (hS : S ∈ Finset.powersetCard k (Finset.range i)) :
    ∑ T ∈ Finset.powersetCard (k - 1) (Finset.range i), uniformPlan k S T = 1
```

### `card_parents`

```lean
theorem card_parents {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (T : Finset ℕ) (hTsub : T ⊆ Finset.range i) (hTcard : T.card = k - 1) :
    ((Finset.powersetCard k (Finset.range i)).filter (fun S => T ⊆ S)).card = i - k + 1
```

### `uniformPlan_col`

```lean
theorem uniformPlan_col {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (T : Finset ℕ) (hT : T ∈ Finset.powersetCard (k - 1) (Finset.range i)) :
    ∑ S ∈ Finset.powersetCard k (Finset.range i), uniformPlan k S T
      = ((i:ℝ) - (k:ℝ) + 1) / (k:ℝ)
```

### `theoremC_existence`

```lean
theorem theoremC_existence {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    ∃ f : Finset ℕ → Finset ℕ → ℝ,
      (∀ S ∈ Finset.powersetCard k (Finset.range i),
        ∑ T ∈ Finset.powersetCard (k - 1) (Finset.range i), f S T = 1) ∧
      (∀ T ∈ Finset.powersetCard (k - 1) (Finset.range i),
        ∑ S ∈ Finset.powersetCard k (Finset.range i), f S T = ((i:ℝ) - (k:ℝ) + 1) / (k:ℝ))
```

### `forest_split_count`

```lean
theorem forest_split_count {A : ℕ} (d : Fin A → ℕ) (hd : ∀ j, 1 ≤ d j)
    (B E : ℕ) (hE : ∑ j, d j = E) (hEB : E + 1 ≤ A + B) :
    (Finset.univ.filter (fun j => 2 ≤ d j)).card + 1 ≤ B
```

### `H2`

```lean
noncomputable def H2 {α : Type*} (s : Finset α) (q : α → ℝ) : ℝ
```

### `entropy_le_log_card`

```lean
theorem entropy_le_log_card {α : Type*} [DecidableEq α] (s : Finset α) (q : α → ℝ)
    (hq : ∀ a ∈ s, 0 ≤ q a) (hsum : ∑ a ∈ s, q a = 1) :
    ∑ a ∈ s, Real.negMulLog (q a) ≤ Real.log s.card
```

### `entropy_le_logb_card`

```lean
theorem entropy_le_logb_card {α : Type*} [DecidableEq α] (s : Finset α) (q : α → ℝ)
    (hq : ∀ a ∈ s, 0 ≤ q a) (hsum : ∑ a ∈ s, q a = 1) :
    H2 s q ≤ Real.logb 2 s.card
```

### `per_step_entropy_bound`

```lean
theorem per_step_entropy_bound {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (src : Finset (Finset ℕ)) (hsrc : src.card = Nat.choose i k)
    (H : Finset ℕ → ℝ) (hHnn : ∀ S ∈ src, 0 ≤ H S)
    (split : Finset (Finset ℕ)) (hsub : split ⊆ src)
    (hzero : ∀ S ∈ src, S ∉ split → H S = 0)
    (hmax : ∀ S ∈ split, H S ≤ Real.logb 2 k)
    (hsplit_card : split.card + 1 ≤ Nat.choose i (k - 1)) :
    ∑ S ∈ src, (1 / (Nat.choose i k : ℝ)) * H S
      ≤ Real.logb 2 k * ((Nat.choose i (k - 1) : ℝ) - 1) / (Nat.choose i k : ℝ)
```

### `per_step_entropy_bound_ratio`

```lean
theorem per_step_entropy_bound_ratio {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) :
    Real.logb 2 k * ((Nat.choose i (k - 1) : ℝ) - 1) / (Nat.choose i k : ℝ)
      ≤ Real.logb 2 k * (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1)
```

### `per_step_entropy_cap`

```lean
theorem per_step_entropy_cap {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (src : Finset (Finset ℕ)) (hsrc : src.card = Nat.choose i k)
    (H : Finset ℕ → ℝ) (hmax : ∀ S ∈ src, H S ≤ Real.logb 2 k) :
    ∑ S ∈ src, (1 / (Nat.choose i k : ℝ)) * H S ≤ Real.logb 2 k
```

### `telescope_bound`

```lean
theorem telescope_bound (m : ℕ) :
    ∑ j ∈ Finset.range m, (1:ℝ) / (((j:ℝ) + 1) * ((j:ℝ) + 2)) ≤ 1
```

### `total_constant_in_n`

```lean
theorem total_constant_in_n (k n : ℕ) (hk1 : 1 ≤ k) :
    ∑ i ∈ Finset.Ico k n,
        ((k:ℝ) / ((i:ℝ) + 1)) * (Real.logb 2 k * (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1))
      ≤ Real.logb 2 k * (k:ℝ) ^ 2
```

### `anytime_uniform_step`

```lean
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
          = 1 / (Nat.choose (i + 1) k : ℝ))
```

### `per_step_entropy_bound_ratio_strict`

```lean
theorem per_step_entropy_bound_ratio_strict {i k : ℕ} (hk2 : 2 ≤ k) (hki : k ≤ i) :
    Real.logb 2 k * ((Nat.choose i (k - 1) : ℝ) - 1) / (Nat.choose i k : ℝ)
      < Real.logb 2 k * (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1)
```

### `per_step_entropy_min`

```lean
theorem per_step_entropy_min {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i)
    (src : Finset (Finset ℕ)) (hsrc : src.card = Nat.choose i k)
    (H : Finset ℕ → ℝ) (hHnn : ∀ S ∈ src, 0 ≤ H S)
    (split : Finset (Finset ℕ)) (hsub : split ⊆ src)
    (hzero : ∀ S ∈ src, S ∉ split → H S = 0)
    (hmaxsrc : ∀ S ∈ src, H S ≤ Real.logb 2 k)
    (hsplit_card : split.card + 1 ≤ Nat.choose i (k - 1)) :
    ∑ S ∈ src, (1 / (Nat.choose i k : ℝ)) * H S
      ≤ min (Real.logb 2 k) (Real.logb 2 k * (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1))
```

### `example_30`

```lean
example (i : ℕ) (hi : 2 ≤ i) :
    ((Nat.choose i 1 : ℝ) - 1) / (Nat.choose i 2 : ℝ) = 2 / (i:ℝ)
```

### `example_31`

```lean
example {k : ℕ} (hk2 : 2 ≤ k) :
    Real.logb 2 k ≤ Real.logb 2 k * (k:ℝ) / (((k:ℕ) + 1 : ℕ) - (k:ℝ) + 1)
```

### `harm`

```lean
noncomputable def harm (n : ℕ) : ℝ := ∑ r ∈ Finset.range n, (1:ℝ) / ((r:ℝ) + 1)
```

### `harm_nonneg`

```lean
theorem harm_nonneg (n : ℕ) : 0 ≤ harm n
```

### `harm_le_harm_add`

```lean
theorem harm_le_harm_add (m k : ℕ) : harm m ≤ harm (m + k)
```

### `harm_le_self`

```lean
theorem harm_le_self (n : ℕ) : harm n ≤ (n:ℝ)
```

### `harm_shift`

```lean
theorem harm_shift (k m : ℕ) :
    ∑ j ∈ Finset.range m, (1:ℝ) / ((j:ℝ) + 1 + (k:ℝ)) = harm (k + m) - harm k
```

### `sum_inv_prod_le_harm_div`

```lean
theorem sum_inv_prod_le_harm_div {k : ℕ} (hk1 : 1 ≤ k) (m : ℕ) :
    ∑ j ∈ Finset.range m, (1:ℝ) / (((j:ℝ) + 1) * ((j:ℝ) + 1 + (k:ℝ))) ≤ harm k / (k:ℝ)
```

### `total_sharp`

```lean
theorem total_sharp (k n : ℕ) (hk1 : 1 ≤ k) :
    ∑ i ∈ Finset.Ico k n,
        ((k:ℝ) / ((i:ℝ) + 1)) * (Real.logb 2 k * (k:ℝ) / ((i:ℝ) - (k:ℝ) + 1))
      ≤ Real.logb 2 k * (k:ℝ) * harm k
```

### `total_sharp_le_easy`

```lean
theorem total_sharp_le_easy {k : ℕ} (hk1 : 1 ≤ k) :
    Real.logb 2 k * (k:ℝ) * harm k ≤ Real.logb 2 k * (k:ℝ) ^ 2
```
