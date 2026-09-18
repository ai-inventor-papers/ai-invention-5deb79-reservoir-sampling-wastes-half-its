# Compiler warnings on the final file

`verified: true`, `has_sorries: false`, **0 errors**, 13 warnings — none of them is
`declaration uses 'sorry'`. They are Mathlib deprecation lints and unused-binder lints,
attributed below to the declaration that introduced them (by diffing the warning list across
the 40 cumulative-prefix compiles in `compile_log/q*.json`).

| declaration | warning |
|---|---|
| `index_convention_nat` | unused variable `hk1` |
| `uniformPlan_row` | unused variable `hki` |
| `entropy_le_logb_card` | `div_le_div_iff` has been deprecated, use `div_le_div_iff₀` instead |
| `per_step_entropy_bound` | `div_le_div_iff` has been deprecated, use `div_le_div_iff₀` instead |
| `per_step_entropy_bound` | unused variable `hsrc` |
| `per_step_entropy_bound` | unused variable `hHnn` |
| `per_step_entropy_bound_ratio` | `div_le_div_iff` has been deprecated, use `div_le_div_iff₀` instead |
| `per_step_entropy_cap` | `div_le_iff` has been deprecated, use `div_le_iff₀` instead |
| `per_step_entropy_cap` | unused variable `hk1` |
| `total_constant_in_n` | `div_le_div_iff` has been deprecated, use `div_le_div_iff₀` instead |
| `per_step_entropy_bound_ratio_strict` | `div_lt_div_iff` has been deprecated, use `div_lt_div_iff₀` instead |
| `example_31` | `le_div_iff` has been deprecated, use `le_div_iff₀` instead |
| `total_sharp` | Used `tac1 <;> tac2` where `(tac1; tac2)` would suffice |

## The unused-binder warnings are informative, not sloppy

They say exactly which stated hypotheses the proof does **not** consume, which is the same kind
of structural fact as `theoremC`'s deliberately unused `_hrow` (uniformity of the next prefix is a
column-sum condition only). The hypotheses are kept in the statements because the paper's theorems
should carry them; the linter records that they are not load-bearing for the inequality proved.

## Deprecations

`div_le_div_iff`, `div_lt_div_iff`, `div_le_iff`, `le_div_iff` are deprecated in favour of the
`₀`-suffixed forms in newer Mathlib. They still compile at the pinned v4.14.0; the file is not
written against a newer Mathlib and this is recorded rather than silenced.
