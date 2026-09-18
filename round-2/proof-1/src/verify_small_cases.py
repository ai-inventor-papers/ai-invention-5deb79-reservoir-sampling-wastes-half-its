#!/usr/bin/env python
"""Exact-rational numeric verification of every statement formalised in Anytime.lean.

Run BEFORE the Lean proofs: a counterexample here means the FORMALISATION is
wrong, not the proof.  All arithmetic is over Fraction (exact), except the
entropy/log checks which use floats with an explicit tolerance.

Conventions (fixed once, see Anytime.lean section 0):
  i   = PREFIX SIZE.  R_i is the reservoir after i arrivals.
  [i] = {0, ..., i-1}  (Lean: Finset.range i)
  J(i,k) = k-subsets of [i]
  The arriving item is index i (0-indexed), i.e. item number i+1 (1-indexed).
"""

from __future__ import annotations

import itertools
import json
import math
import random
from fractions import Fraction as F
from math import comb
from pathlib import Path

OUT = Path(__file__).resolve().parent / "results" / "small_case_verification.json"
random.seed(20260918)

report: dict[str, dict] = {}


def record(name: str, checks: int, failures: list, extra: dict | None = None) -> None:
    report[name] = {
        "checks": checks,
        "failures": failures[:5],
        "n_failures": len(failures),
        "status": "PASS" if not failures else "FAIL",
        **(extra or {}),
    }
    print(f"{name:34s} {report[name]['status']}  ({checks} checks, {len(failures)} failures)")


# ---------------------------------------------------------------- index_convention
fails, n = [], 0
for i in range(0, 60):
    for k in range(1, 60):
        n += 1
        if ((i + 1) - (i - k + 1)) != k:
            fails.append((i, k))
        # the two divisibility conventions in the run's own code agree
        if ((i - k + 1) % k == 0) != ((i + 1) % k == 0):
            fails.append(("dvd", i, k))
record("index_convention", n, fails,
       {"note": "k | (i-k+1)  <->  k | (i+1); the two code files differ only in which index they name."})

# ---------------------------------------------------------------- L0a / L0b
fails, n = [], 0
for i in range(0, 40):
    for k in range(1, i + 1):
        n += 1
        if k * comb(i, k) != (i - k + 1) * comb(i, k - 1):
            fails.append(("L0a", i, k))
record("L0a  k*C(i,k)=(i-k+1)*C(i,k-1)", n, fails)

fails, n = [], 0
for i in range(0, 40):
    for k in range(0, i + 1):
        n += 1
        if (i + 1) * comb(i, k) != (i + 1 - k) * comb(i + 1, k):
            fails.append(("L0b", i, k))
record("L0b  (i+1)*C(i,k)=(i+1-k)*C(i+1,k)", n, fails)

fails, n = [], 0
for i in range(1, 40):
    for k in range(1, i + 1):
        n += 1
        if F(comb(i, k - 1), comb(i, k)) != F(k, i - k + 1):
            fails.append(("ratio", i, k))
record("L0a_real  C(i,k-1)/C(i,k)=k/(i-k+1)", n, fails)

# ---------------------------------------------------------------- L1 accept forced
fails, n = [], 0
for i in range(1, 25):
    for k in range(1, i + 1):
        n += 1
        # solve (1/C(i,k))*(1-p) = 1/C(i+1,k)  for p
        p = 1 - F(comb(i, k), comb(i + 1, k))
        if p != F(k, i + 1):
            fails.append(("L1", i, k, str(p)))
record("accept_forced  p=k/(i+1)", n, fails)

# ---------------------------------------------------------------- L2 = Theorem C
def johnson(i: int, k: int):
    return [frozenset(s) for s in itertools.combinations(range(i), k)]


fails, n = [], 0
uniform_plan_fails = []
for i in range(1, 10):
    for k in range(1, i + 1):
        Jik, Jik1, Ji1k = johnson(i, k), johnson(i, k - 1), johnson(i + 1, k)
        A = comb(i, k)
        # --- (a) the UNIFORM plan f(S,T) = 1/k on every edge is feasible
        row = {S: sum(F(1, k) for T in Jik1 if T <= S) for S in Jik}
        col = {T: sum(F(1, k) for S in Jik if T <= S) for T in Jik1}
        for S in Jik:
            if row[S] != 1:
                uniform_plan_fails.append(("row", i, k, sorted(S), str(row[S])))
        for T in Jik1:
            if col[T] != F(i - k + 1, k):
                uniform_plan_fails.append(("col", i, k, sorted(T), str(col[T])))
        # --- (b) ANY column-feasible plan gives a uniform next prefix
        for trial in range(3):
            f: dict[tuple, F] = {}
            for T in Jik1:
                parents = [S for S in Jik if T <= S]
                w = [F(random.randint(1, 20)) for _ in parents]
                tot = sum(w)
                for S, wi in zip(parents, w):
                    f[(S, T)] = F(i - k + 1, k) * wi / tot
            for Sp in Ji1k:
                n += 1
                if (i) in Sp:  # arriving item (0-indexed index i) was accepted
                    T = frozenset(Sp - {i})
                    val = F(k, i + 1) * sum(F(1, A) * f.get((S, T), F(0)) for S in Jik)
                else:
                    val = (1 - F(k, i + 1)) * F(1, A)
                if val != F(1, comb(i + 1, k)):
                    fails.append(("L2", i, k, sorted(Sp), str(val)))
record("theoremC  P_{i+1}=1/C(i+1,k)", n, fails)
record("theoremC_existence (uniform f=1/k)", 1, uniform_plan_fails)

# ---------------------------------------------------------------- L3 forest split count
fails, n = [], 0
for trial in range(20000):
    A = random.randint(1, 9)
    B = random.randint(1, 9)
    d = [random.randint(1, 5) for _ in range(A)]
    E = sum(d)
    if E > A + B - 1:
        continue
    n += 1
    split = sum(1 for x in d if x >= 2)
    if not (split + 1 <= B):
        fails.append((A, B, d))
record("forest_split_count  |Split|+1<=B", n, fails)

# ---------------------------------------------------------------- entropy_le_log_card
def H2(q: list[float]) -> float:
    return sum(-x * math.log2(x) for x in q if x > 0)


fails, n = [], 0
for trial in range(20000):
    m = random.randint(1, 8)
    w = [random.random() for _ in range(m)]
    tot = sum(w)
    q = [x / tot for x in w]
    n += 1
    if H2(q) > math.log2(m) + 1e-12:
        fails.append((m, q, H2(q)))
record("entropy_le_log_card", n, fails)

# ---------------------------------------------------------------- L4 per-step entropy bound
fails, n = [], 0
detail = []
for i in range(2, 22):
    for k in range(1, min(i, 6) + 1):
        A, B = comb(i, k), comb(i, k - 1)
        for trial in range(30):
            # a worst-case-shaped eviction coupling: at most B-1 split sources,
            # each split source uniform over its k children (max entropy)
            n_split = min(B - 1, A)
            idx = random.sample(range(A), n_split) if n_split > 0 else []
            Hev = sum((1.0 / A) * math.log2(k) for _ in idx) if k > 1 else 0.0
            bound1 = math.log2(k) * (B - 1) / A
            bound2 = math.log2(k) * k / (i - k + 1)
            n += 1
            if Hev > bound1 + 1e-12 or bound1 > bound2 + 1e-12 or Hev > math.log2(k) + 1e-12:
                fails.append((i, k, Hev, bound1, bound2))
        if k == 2:
            detail.append({"i": i, "k": k,
                           "bound_log2k_times_(B-1)/A": math.log2(k) * (B - 1) / A,
                           "solved_min_log2k/(i-k+1)": math.log2(k) / (i - k + 1),
                           "slack_ratio": (math.log2(k) * (B - 1) / A) / (math.log2(k) / (i - k + 1))})
record("per_step_entropy_bound", n, fails, {"k2_slack_table": detail[:6]})

# ---------------------------------------------------------------- L5 total constant in n
fails, n = [], 0
for k in range(1, 40):
    for nmax in [k + 1, k + 10, 100, 1000, 100000]:
        n += 1
        tot = sum((k / (i + 1)) * (math.log2(k) * k / (i - k + 1))
                  for i in range(k, max(nmax, k)))
        if tot > math.log2(k) * k * k + 1e-9:
            fails.append((k, nmax, tot, math.log2(k) * k * k))
record("total_constant_in_n  <= log2(k)*k^2", n, fails)

# telescoping ingredient
fails, n = [], 0
for m in range(0, 2000):
    n += 1
    s = sum(1.0 / ((j + 1) * (j + 2)) for j in range(m))
    if s > 1 + 1e-12:
        fails.append((m, s))
record("telescope  sum 1/((j+1)(j+2))<=1", n, fails)

# ---------------------------------------------------------------- L5 sharp constant
fails, n = [], 0
sharp = []
for k in [2, 3, 10, 50]:
    n += 1
    trunc = sum((k / (i + 1)) * (math.log2(k) * k / (i - k + 1)) for i in range(k, k + 2_000_000))
    Hk = sum(1.0 / m for m in range(1, k + 1))
    exact = math.log2(k) * k * Hk
    sharp.append({"k": k, "truncated_series": round(trunc, 4), "log2k_times_k_times_H_k": round(exact, 4),
                  "proved_bound_log2k_k2": round(math.log2(k) * k * k, 4)})
    if abs(trunc - exact) > 1e-2 * max(1.0, exact):
        fails.append((k, trunc, exact))
record("L5 sharp constant = log2(k)*k*H_k", n, fails, {"table": sharp})

# ---------------------------------------------------------------- defect D1 epitaph
d1 = {"fitted_law_bits_per_acceptance_k1000_i_eq_kplus1": 11.27,
      "trivial_ceiling_log2_1000": math.log2(1000),
      "fitted_law_violates_ceiling": 11.27 > math.log2(1000),
      "our_bound_at_k1000_i1001": min(math.log2(1000), math.log2(1000) * 1000 / (1001 - 1000 + 1)),
      "our_bound_respects_ceiling": True}
report["defect_D1_epitaph"] = d1
print("defect_D1_epitaph:", json.dumps(d1))

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(report, indent=2))
overall = all(v.get("status", "PASS") == "PASS" for v in report.values())
print("\nOVERALL:", "ALL PASS" if overall else "SOME FAILED")
