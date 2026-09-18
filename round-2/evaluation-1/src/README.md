# A third test, and every number rechecked

Evaluation artifact over the two iteration-1 reservoir-sampling experiments
(`gen_art_experiment_1` = E1, exact-rational prefix DP; `gen_art_experiment_2` = E2,
counted-coin bench). Pure Python / NumPy / SciPy. **$0.00 of LLM spend, 0 LLM calls, no
GPU, no network except the Part-C3 reference check.**

## What this artifact answers

The paper's weakest sentence dismissed the distribution-identity-testing literature
because it "assumes sample access to a distribution rather than an implementation under
audit". An implementation under audit *is* a sample oracle, so the only defensible move is
to run the published tester and report what happens. That is Part A. Parts B and C repair
the bookkeeping and pre-empt the checks a reviewer runs next.

## Parts

| Part | What it does | Output |
|---|---|---|
| **A** | Adds the published **L1 identity / collision tester** (Acharya-Daskalakis-Kamath, arXiv:1507.05952; Diakonikolas-Kane-Nikishkin, arXiv:1410.2266; collision-tester optimality, Diakonikolas-Gouleakis-Peebles-Price) as a **third detector column** on both existing panels, and converts the result into a sample-complexity / state-complexity **separation table**. | `results/stage_a1.json`, `stage_a2.json`, `stage_a3.json` |
| **B** | Machine-recomputes **every number the draft prints** from the two `full_method_out.json` artifacts and marks each MATCH / MISMATCH / UNAVAILABLE with a drop-in corrected sentence. | `claims_ledger.csv`, `patch_table.md`, `results/stage_b.json` |
| **C** | Re-derives the co-inclusion calibration and finds the **smallest (n,k) at which the nominal chi-square rule false-alarms on Algorithm R itself**; **tests** (rather than asserts) the sign-reversal mechanism; reference hygiene. | `results/stage_c1.json`, `stage_c2.json`, `scratch/refs.json` |
| **D** | Constructs a **distributed-merge defect** of the Apache DataSketches #647 shape whose per-item marginals are correct **by construction**, and runs both detectors on it. | `results/stage_d.json` |

## The key algebraic fact that makes Part A runnable

For a uniform reference `q_x = 1/D`, the ADK/DKN chi-square-type statistic collapses to a
function of the collision count only:

```
Z = sum_x [ (N_x - T q_x)^2 - N_x ] / (T q_x)
  = (D/T) * sum_x N_x (N_x - 1) - T
  = (2D/T) * C - T          where C = number of colliding sample pairs
```

so the tester never needs a length-`D` count vector. That is what makes the
`n=1000, k=10` panel, where `D = C(1000,10) ~ 2.63e23`, computable at all. The identity
`2C == sum N_x(N_x-1)` is verified by brute force on a toy domain in `test_l1tester.py`,
and the collapsed form is checked against the general statistic with an explicit length-`D`
count vector.

Exact multinomial null moments (not Poissonised): `E[C] = binom(T,2)/D`,
`Var[C] = binom(T,2)(1/D)(1-1/D)`, `E[Z] = -1` exactly for every `T` and `D`, and
`Var[Z] = 2D(T-1)/T(1-1/D)`.

## Files

```
eval.py             assembler -> eval_out.json (exp_eval_sol_out), published as
                    full_/mini_/preview_eval_out.json; also appends the Part-C1
                    addendum to patch_table.md (idempotently)
l1tester.py         the two published testers + exact null moments + planted alternative
test_l1tester.py    7 unit tests, all green (run: .venv/bin/python test_l1tester.py)
partA.py            A1 calibration, A2 matched panel, A3 replicated panel, A4/A5 tables
partB.py            the claims ledger
partC.py            C1 calibration boundary, C2 sign-reversal mechanism
partD.py            the merge-defect panel
run_c1.py           C1 runner: per-cell checkpointing + bounded null budget
chunk_probe.py      does a single nominal co-inclusion p depend on the trial chunking?
determinism_check.py  re-runs the deterministic stages and compares content hashes
vendored/           E1 and E2 modules, copied read-only and prefixed e1_/e2_;
                    sha256 of every one is recorded in eval_out.json metadata
results/            per-stage JSON
scratch/            codefacts.json (source-code audit), refs.json (reference hygiene),
                    extracted E1/E2 metadata, determinism-check copies
```

## Reproducing

```bash
uv venv .venv --python=3.12 && uv pip install --python .venv/bin/python numpy scipy loguru mpmath
.venv/bin/python test_l1tester.py
.venv/bin/python partA.py --stage a1
.venv/bin/python partA.py --stage a2 --nullR 500
.venv/bin/python partA.py --stage a3 --nullR 200
.venv/bin/python partB.py
.venv/bin/python run_c1.py          # C1: 48-cell sweep + 4 calibrated cells, ~13 min
.venv/bin/python partC.py --stage c2
.venv/bin/python partD.py --T 1000000 --nullR 200
.venv/bin/python eval.py
```

Every stage is seeded (the seed list is in `eval_out.json` `metadata.seeds`). The
deterministic paths (Part B, Part A1, Part C2) were re-run independently and produce
byte-identical output; three Part-C1 nominal-sweep cells were re-run and reproduce their
co-inclusion chi-square and p-value **exactly**; and the whole assembled payload (minus
wall times, the determinism block and the hardware block) hashes to the same sha256 on
two independent runs of `eval.py`. All hashes are in `metadata.determinism`.

One adaptation was needed to a vendored module and is recorded in
`metadata.vendored_adaptations`: `e2_nullband.exact_uniform_subsets` capped its rejection
loop at 200 passes, which is not enough at `(n,k) = (500,50)` or `(2000,100)` where the
per-pass acceptance is `exp(-C(k,2)/n) = 0.086`, so it raised `RuntimeError` mid-sweep.
The cap was raised; the sampling law is unchanged (same rejection sampler, run to
completion), and the cells the truncated version did clear reproduce exactly.

## Headline results

### Part A — the three detectors, and what actually separates them

Matched panel, 18 cells, exact-rational ground truth (accuracy / recall on broken /
false alarm on provably correct):

| Detector | accuracy | recall | false alarm |
|---|---|---|---|
| first-order max-deviation (final reservoir) | 0.833 | 0.667 | 0.000 |
| first-order max-deviation, **anytime** (all prefixes) | 0.833 | 0.667 | 0.000 |
| pairwise co-inclusion | 1.000 | 1.000 | 0.000 |
| **L1 identity / collision tester (final)** | **1.000** | **1.000** | **0.000** |
| **L1 identity / collision tester (anytime)** | **1.000** | **1.000** | **0.000** |

The anytime first-order test reproduces the final-reservoir numbers **exactly** and misses
the **same three** S5-CIRC cells: the blindness is structural, not a power problem.

Replicated panel, 80 runs (10 samplers x 8 seeds, n=1000, k=10, T=1e6,
D = C(1000,10) = 2.63e23). Predicted collision count `lambda = binom(T,2)*||p||_2^2`
against the measured count:

| sampler | support | exact TV | lambda predicted | C measured (8 seeds) | L1 recall |
|---|---|---|---|---|---|
| S0 / S1h / S2 / S2f / S2v / S3 (correct) | C(1000,10) | 0 | 1.9e-12 | all 0 | — (0/48 false alarms) |
| S5e BLOCK-LOCK | n/k = 100 | 1 − 100/D | 5.0e9 | 4.99996e9 … 5.00010e9 | 8/8 |
| S6sys SYSTEMATIC | n/k = 100 | 1 − 100/D | 5.0e9 | 4.99989e9 … 5.00005e9 | 8/8 |
| S5-CIRC | n = 1000 | 1 − 1000/D | 5.0e8 | 4.99944e8 … 5.00019e8 | 8/8 |
| S5g PAIR-LOCK | C(500,5) = 2.55e11 | 1 − C(500,5)/D | **1.96** | 2,3,5,1,1,2,2,**0** | **7/8** |

Measured-versus-predicted collision counts agree to a maximum relative error of
**1.1e-4** wherever `lambda > 1000` — a sharp correctness check on the implementation.

**The finding the plan did not anticipate.** The collision tester's power is governed by
`lambda`, i.e. by `||p||_2^2`, **not** by L1 distance. PAIR-LOCK is exactly as far in L1 as
BLOCK-LOCK (TV ≈ 1 in both cases) but its support is 2.5e9 times larger, so `lambda = 1.96`
and the tester misses it whenever the Poisson count is zero — 1 of 8 seeds, against a
predicted miss probability `exp(−1.96) = 0.141`. Calibrated pairwise co-inclusion flags the
same sampler at `z = 1078.6` on **every** seed.

Worst case versus instance: the guarantee needs `T = c·sqrt(C(1000,10))/eps^2` with the
**measured** `c = 5.49`, i.e. 2.8e14 samples at `eps = 0.1` and 2.8e12 even at the blind
spots' own `eps ≈ 1`. The panel used `T = 1e6`. Both statements are true simultaneously,
and they differ by 6.5–8.5 orders of magnitude.

Tester calibration (Part A1), on an **absolute** geometric T grid so the scaling test is
not circular: fitted exponent of `D` is 0.48–0.54 (theory 0.5), fitted exponent of `eps` is
−1.96 to −2.11 (theory −2), and `c` measured over 16 `(D, eps)` cells is 5.03–6.67
(median 5.49, spread ratio 1.32).

### Part D — the case that defeats *two* of the three detectors

The merge defect takes a **fixed** j = 5 survivors from each size-10 reservoir instead of
`j ~ Hypergeometric`. Per-item marginals are exactly `k/(n1+n2) = 0.01` under **both**
rules — verified empirically — so the first-order test has recall **0.00** by construction.
The defective merge keeps 0.2473 of the full support, so it is far in L1 (TV = 0.7527) but
its L2 norm is only 4.04x the uniform one: the expected collision count rises from 1.9e-12
to 7.7e-12, and the L1 tester measures **0 collisions under both merges** — recall **0.00**.
Pairwise co-inclusion has recall **1.00**. This is the spread-out defect the worst-case
`Theta(sqrt(D)/eps^2)` bound warns about, realised concretely.

### Part B — the claims ledger

18 printed numbers checked: **8 MATCH, 9 MISMATCH, 1 UNAVAILABLE**. The three most
consequential rows are not wrong arithmetic but wrong *provenance*:

1. `H is exactly uniform for k = 2..10 up to n = 20` — **UNAVAILABLE**: the executed grid is
   10 cells, every one with `k <= 5` and `n <= 18`.
2. The fitted constant `fitC = 1.1321` is taken from `k = 2` and called "most
   conservative", but `C` **decreases** in `k`, and at `k = 1000, i = k+1` the fitted law
   predicts 11.27 bits per acceptance against the trivial ceiling `log2(1000) = 9.97`. It is
   an extrapolation, not a bound — and a second, incompatible extrapolation for the same
   cell (24.40 vs 35.79 bits) sits elsewhere in the same JSON.
3. `E1/forced.py:133` uses `min((B-1)/A, log2(k))` as a bits-per-acceptance bound, but
   `(B-1)/A` bounds the *fraction of sources that split*; the missing factor is `log2(k)`.
   Restored, the k=10 bracket is 68.81 / 69.11 bits rather than 20.71 / 20.81. The two
   expressions are algebraically identical at `k = 2`, which is why the solved `k = 2` minima
   matched and the defect stayed invisible.

`patch_table.md` carries every MISMATCH and UNAVAILABLE row with a drop-in replacement
sentence, sorted by severity.

### Part C1 — is the co-inclusion calibration deviation justified?

Yes, but not for the reason the source artifact gave. Over a **48-cell**
`(n,k)` sweep at `T = 2e5`, the pre-registered nominal chi-square `p < 1e-6` rule
false-alarms on Algorithm R — the *provably uniform* sampler — at
**1** cell, `n = 200,
k = 40` (`k/n = 0.20`),
which is the smallest such cell both by `k` and by `k/n`. The declared cell
`(2000, 100)` gives `p = 0.267` here, not the `7.9e-16` that was quoted — a single-run
nominal `p` is not sound evidence, and `chunk_probe.py` shows the same statistic swinging
over 1.3 orders of magnitude when only the trial chunking of one PCG64 stream changes.

The decisive quantity is the statistic's **own simulated null**, built from an exactly
uniform k-subset oracle sharing no code path with the sampler under audit. It is
over-dispersed against the nominal `chi2(df)` reference by up to
**3.45x** in standard deviation (co-inclusion bins
are negatively correlated), so the nominal rule would flag a provably uniform sampler with
probability **0.306** at `n = 2000, k = 100`;
3 of 4 calibrated cells are mis-calibrated by
that test. The **calibrated** own-null rule (`z >= 5`) separates correct from broken
samplers with no overlap everywhere: max `z` among correct **1.817** vs min `z` among
broken **2.65e4**, a gap of **2.65e+04**. No verdict in either artifact
depends on the nominal rule.

### Part C2 — why do three *broken* samplers beat the correct one?

**MECHANISM_CONFIRMED.** The max-deviation statistic is a maximum over *effective bins*,
not over items: a block-locked support gives `m = n/k = 100` bins rather than `n = 1000`,
and the Gumbel location shrinks by `sqrt(2 ln m)/sqrt(2 ln n) = 0.8165`. Simulating both
designs (R = 2000 replicates) puts the observed shift **inside** the predicted 95% interval
for all three samplers. The check also refutes the lazy version of the story: PAIR-LOCK
locks items into `n/2 = 500` pairs, not `n/k` blocks, so assuming `n/k` for all three would
mis-predict it — each sampler's effective bin count has to come from its own support.

### Part C3 — reference hygiene

12 references checked: **7 VERIFIED**,
**2 CORRECTED**, **2
NON_PEER_REVIEWED** (reference [11] is a blog post — attributional only, since the closed
form is re-derived independently), **1 UNRESOLVED** (reference
[25]: printed year 2025 against a DOI in the 2026 IEEE Trans. Inform. Theory volume range,
and no exact DOI string was supplied — the search steps are recorded rather than a guessed
year).
