# Counting the coin flips a stream sampler spends

A reservoir-sampling bench that answers two questions on **identical streams**:

1. **Is the sampler uniform?** Sample `k` items from a stream of unknown length, verify
   uniformity over many trials, and report the max deviation from the expected frequency
   `k/n` — **always next to a simulated extreme-value null band**, never as a bare
   percentage, plus an exact pairwise co-inclusion test that sees the *joint* law.
2. **What does its randomness cost?** Drive every candidate through **one** counted
   fair-coin source under **three accounting regimes** (64-bit-per-decision, Knuth–Yao
   entropy-optimal-per-decision, and interval-recycling), so "this sampler is
   randomness-efficient" is settled by measurement rather than by each side choosing its
   own bookkeeping.

Everything is pure Python / NumPy / SciPy / mpmath. No GPU, no network, no LLM calls,
**$0 spend**.

## Quick start

```bash
uv venv .venv --python=3.12 && uv pip install --python=.venv/bin/python numpy scipy mpmath loguru pytest psutil
PYTHONPATH=. .venv/bin/python -m pytest -c pyproject.toml     # gates 1, 4, 5
.venv/bin/python screen.py                                    # SCREEN panel; freezes screen_ranking.json
.venv/bin/python heldout.py --stage all                       # HELD-OUT panel (verifies the freeze hash)
.venv/bin/python method.py                                    # assembles results/method_out.json
```

`heldout.py` recomputes `screen_ranking.json`'s `content_sha256` on startup and **aborts**
if it changed. That is the pre-registration.

## Modules

| file | what it owns |
|---|---|
| `prngs.py` | graded-quality generators: LCG-low16, xorshift32, PCG64, R250, MT19937, ChaCha20 — scalar bit API *and* vectorised bulk API, all verified bit-for-bit against their scalar recurrences |
| `bits.py` | one counted `BitStream` + the three regimes; raw **and** net-buffer-consumption tag attribution |
| `budget.py` | closed-form budgets, so `n = 1e7` is exact rather than extrapolated |
| `candidates.py` | 16 samplers, each in a scalar (bit-counted) **and** a vectorised (Monte-Carlo) implementation, plus an exact rational transition kernel |
| `exact.py` | `fractions.Fraction` prefix verifier — the gate before any Monte Carlo |
| `s1t.py` | the table-exact minimum-randomisation eviction map (max-flow b-matching) |
| `algol.py` | Algorithm-L float64 audit against an 80-digit mpmath reference |
| `nullband.py` | exactly-uniform k-subset sampler, simulated null bands, co-inclusion machinery |
| `montecarlo.py` | vectorised MC harness + the bit-accounting sweep |
| `screen.py` / `heldout.py` / `method.py` | pre-registered screen, held-out panel, assembly |

## What the streams are

Uniformity of a k-subset sampler is a statement about **index subsets**, so the stream is
the index sequence `1..n`, generated in-process and seeded. There is no external corpus to
fetch; `prngs.py` + `candidates.py` *are* the data generator, and every number is
reproducible from the manifest in `results/method_out.json`.

## Headline results

See `results/method_out.json` → `metadata.headline`, `metadata.verdicts`, and
`metadata.corrections_to_the_hypothesis`.

### The literal deliverable

Algorithm R, `n = 1000`, `k = 10`, `T = 10^6` independent trials, expected frequency `k/n = 0.01`:

> **max deviation = 3.68e-4 (0.0368 percentage points), which is +0.55 σ in a simulated
> null band of 3.48e-4 ± 3.58e-5 (p99 = 4.45e-4), i.e. the 77th percentile of what an
> EXACTLY uniform sampler produces.** The analytic extreme-value cross-check
> `σ·sqrt(2 ln n)` gives 3.70e-4.

A bare "max deviation = 0.04 %" is uninterpretable: the statistic maximises over `n` bins,
so a perfect sampler cannot score 0.

### Method vs baseline (uniformity testing), 6 provably-correct and 3 degenerate samplers

| test | recall on broken | false alarm on correct | accuracy |
|---|---|---|---|
| baseline — first-order max deviation | 0.67 (single run) / **0.00** (48+24 replicated runs) | 0.00 | 0.80 |
| **ours — pairwise co-inclusion** | **1.00** | **0.00** | **1.00** |

Replicated over 8 seeds per sampler: the calibrated co-inclusion statistic sits in
`[-2.09, +2.28] σ` for every provably-correct sampler and in `[5.2e5, 4.7e6] σ` for every
degenerate one. Sharper still — the degenerate samplers score *better* than Algorithm R on
the first-order test, because a block-locked support has only `n/k` independent bins.

### Verdicts

| | verdict | margin |
|---|---|---|
| MAIN (eviction randomness is recoverable) | CONFIRMED-IN-TABLE-EXACT-FORM | 1.16 |
| C1 (first-order testing is blind to joint-law failure) | CONFIRMED | 1.00 |
| C2 (Algorithm L's float64 bias is detectable) | REFUTED-BY-MAGNITUDE | 0.29 |
| C3 (generator quality separates the algorithms) | CONFIRMED, ordering REVERSED | 2.40 |
| C4 (representation dominates coupling) | CONFIRMED | 15.76 |

### The reversal that settles MAIN vs C4

`n = 10^6`, `k = 10`, entropy floor `Σ h(k/i) + log2(k)·E[accepts] = 1493.7` bits:

| sampler | R-naive64 | R-KY | R-recycle |
|---|---|---|---|
| Algorithm R | 64,006,165 (42,852× floor) | 2,000,447 (1,339×) | **1,568 (1.05×)** |
| Algorithm L | 20,336 (13.6×) | 13,485 (9.0×) | 12,751 (8.5×) |
| bottom-k | 53,000,000 | 53,000,000 | 53,000,000 |

Under 64-bit-per-decision accounting Algorithm L beats Algorithm R by **3,148×**; under
entropy-optimal accounting with recycling Algorithm R beats Algorithm L by **8.1×**. Which
sampler is "randomness-efficient" is a property of the accounting regime, not of the
sampler. Bottom-k is `53n` bits in every regime — pure representation.

### Corrections this run makes to the hypothesis it was built to test

1. Algorithm L's `1.0 - W` does **not** suffer catastrophic cancellation (Sterbenz). The
   damaging limit is the opposite one, `W → 0`, and the proposed `V = 1-W` fix targets the
   wrong end; tracking `log W` with `log1mexp` is flat at ~1e-16 for `n = 10^3…10^15`.
2. The sum-mod-k mixture repair is **not** exact (fails at prefix i = 5/7/9 for k = 2/3/4).
   The deterministic rank rule **is** exact whenever `k | (i-k)` — verified in rational
   arithmetic for `k = 2..10` up to `n = 20`.
3. FIFO eviction first fails at prefix **i = 3**, not i = 4.
4. The recycling Bernoulli needs the *conditional residual* pushed back and a *threshold*
   refill, or it degenerates to Knuth–Yao.
5. The graded-PRNG ordering is reversed: bottom-k is the most robust, Algorithm R among
   the worst.
