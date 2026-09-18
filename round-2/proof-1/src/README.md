# Machine-checking the sampler's math in Lean

Lean 4 + Mathlib formalisation of the six load-bearing facts behind the anytime
reservoir-sampling claims, plus the paper-ready informal proofs, the assumption boundary, and
the open problem the theorems bracket.

**Headline: 40/40 declarations compile sorry-free.** No `sorry`, no user `axiom`; `#print axioms`
returns only the three standard Lean axioms (`propext`, `Classical.choice`, `Quot.sound`) — or a
strict subset — for all 38 named declarations. Whole-file elaboration: **0.59 s** warm.

| what | where |
|---|---|
| the formalisation | [`proof.lean`](proof.lean) (826 lines; `Anytime.lean` is a byte-identical copy under the working name) |
| per-lemma status table | [`results/lemma_status.md`](results/lemma_status.md), [`results/lemma_status.json`](results/lemma_status.json) |
| `#print axioms` raw output | [`results/print_axioms.json`](results/print_axioms.json) |
| paper-ready informal proofs (all lemmas) | [`results/informal_proofs.md`](results/informal_proofs.md) |
| ASSUMED, NOT PROVED list | [`results/assumed_not_proved.md`](results/assumed_not_proved.md) |
| the open problem | [`results/open_problem.md`](results/open_problem.md) |
| index-convention note (retires defect D5) | [`results/index_convention_note.md`](results/index_convention_note.md) |
| exact-rational pre-check of every statement | [`verify_small_cases.py`](verify_small_cases.py), [`results/small_case_verification.json`](results/small_case_verification.json) |
| pipeline output (schema `exp_proof_out`) | [`proof_out.json`](proof_out.json) |
| compiler warnings, attributed (0 errors, no `sorry` warning) | [`results/compiler_warnings.md`](results/compiler_warnings.md) |
| self-summary: what worked / failed / next | [`results/self_summary.md`](results/self_summary.md) |
| every compiler invocation (inputs + raw JSON) | [`compile_log/`](compile_log/) — 97 Lean files + 97 result JSONs |

## What is proved

| | theorem | content |
|---|---|---|
| §0 | `index_convention`, `index_convention_nat` | prefix-size index and arriving-item index differ by exactly `k`, so `k ∣ (i-k+1)` and `k ∣ (i+1)` name the same steps |
| §1 | `L0a`, `L0b` (+ `_real`) | `k·C(i,k) = (i-k+1)·C(i,k-1)` (Johnson biregularity); `(i+1)·C(i,k) = (i+1-k)·C(i+1,k)` |
| §2 | `accept_forced` | **the accept coin is forced to `k/(i+1)` in every state** — the linear system is diagonal |
| §2 | `acc_floor` | the Knuth–Yao floor, with Knuth–Yao as an explicit hypothesis |
| §3 | `theoremC` (+ `_reject`, `_accept`) | **any column-feasible routing plan preserves prefix uniformity, for all (n,k)** |
| §3 | `theoremC_existence`, `uniformPlan_row/col`, `card_parents` | the feasible polytope is non-empty — constructive witness `f = 1/k`, i.e. textbook Algorithm R |
| §4 | `forest_split_count` | forest support ⇒ at most `B-1` split sources (truncation-free `card + 1 ≤ B`) |
| §5 | `entropy_le_log_card`, `entropy_le_logb_card` | max entropy `H(q) ≤ log|support|`, proved from `log x ≤ x-1` |
| §5 | `per_step_entropy_bound`, `_ratio`, `_ratio_strict`, `_cap`, `per_step_entropy_min` | `H_evict ≤ min(log₂k, log₂(k)·k/(i-k+1))` per acceptance |
| §6 | `telescope_bound`, `total_constant_in_n` | **the total eviction randomness is ≤ log₂(k)·k², independently of n** |
| §8 | `total_sharp`, `sum_inv_prod_le_harm_div`, `harm_*` | the **sharp** total `≤ log₂(k)·k·H_k = Θ(k log²k)`, by partial fractions; `harm_le_self` recovers §6 |
| §7 | `anytime_uniform_step` | combines C and C.1: an exactly-uniform step always exists |

## Three things reported rather than buried

1. **The hypothesis text's `O(k log k)` total is wrong.** The machine-checked sharp bound
   (`total_sharp`) is `log₂(k)·k·H_k = Θ(k log² k)`, not `O(k log k)`; the simpler `O(k² log k)`
   bound (`total_constant_in_n`) is also proved, and `harm_le_self` shows the first implies the
   second. Confirmed numerically at k = 2, 3, 10, 50. Only the *constant-in-n* claim — the
   load-bearing one — survives, and it survives under either bound.
2. **The L4 bound and the exactly solved k=2 minima agree.** The bound is `2/i`, the solved minima
   are `1/(i-1)`, and `2/i ≥ 1/(i-1)` for every `i ≥ 2` — never violated, slack → 2×. Both are
   machine-checked Lean `example`s.
3. **Defect D1's epitaph.** The proved bound cannot exceed the trivial ceiling, because the `min`
   with `log₂ k` is part of the theorem. The fitted law it replaces predicted 11.27 bits per
   acceptance at `k = 1000, i = k+1`, against a ceiling of `log₂ 1000 = 9.9658`.

## Reproducing

```bash
python3 verify_small_cases.py                 # exact-rational pre-check of every statement, ~40 s
.venv/bin/python lean_worker.py               # warms Lean 4 v4.14.0 + Mathlib once (~20 min cold)
cp Anytime.lean jobs/check.lean               # then read jobs/check.json
```

`lean_worker.py` is a thin persistent wrapper around the `aii-lean` skill's `aii_run_lean` /
`aii_lean_suggest` entry points: it imports Mathlib once into a pooled server and then serves
`jobs/*.lean` files, so each iteration costs seconds instead of minutes.

Spend on this artifact: **$0** (no LLM API calls, no GPU).
