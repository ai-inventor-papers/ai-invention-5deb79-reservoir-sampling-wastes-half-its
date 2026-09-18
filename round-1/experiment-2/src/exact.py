#!/usr/bin/env python3
"""Small EXACT rational prefix verifier (the gate before any Monte Carlo).

Propagates the exact distribution over reservoir states with
:class:`fractions.Fraction`, so a sampler is proved uniform (or proved not) at
small n with zero statistical uncertainty.  This is a self-check, not the full
M1/M2 instrument -- it exists so that no hour of Monte Carlo is spent on a
sampler that has not first been proved correct at n <= 14.

At each prefix i it records
    max_abs_dev          max_S | P(S) - 1/C(i,k) |
    max_rel_dev          the same, divided by 1/C(i,k)
    tv                   total variation distance to uniform
    support_size         number of reachable k-subsets
    first_order_max_dev  max_t | P(t in reservoir) - k/i |
and it reports the FIRST prefix at which the subset law departs from uniform.
"""

from __future__ import annotations

import math
from collections import defaultdict
from fractions import Fraction

from candidates import Candidate, get_candidate

__all__ = ["verify", "verify_many"]


def _subset_key(slots: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(sorted(slots))


def verify(cid: str, n: int, k: int, *, record_every: bool = False) -> dict:
    """Exact prefix verification of candidate ``cid`` for stream length ``n``."""
    cand: Candidate = get_candidate(cid)
    if cand.children is None:
        raise ValueError(f"candidate {cid} has no exact kernel (audited elsewhere)")
    if cand.needs_even_k and k % 2:
        raise ValueError(f"candidate {cid} requires an even k")
    init = (tuple(range(1, k + 1)), cand.init_aux)
    dist: dict = {init: Fraction(1)}
    prefixes: list[dict] = []
    first_fail: int | None = None
    worst = {"max_abs_dev": Fraction(0), "max_rel_dev": Fraction(0), "prefix": k}

    for i in range(k + 1, n + 1):
        if cand.step_filter(i, k):
            num = cand.accept_num(i, k)
            den = cand.accept_den(i, k)
            p = Fraction(num, den)
            nxt: dict = defaultdict(Fraction)
            for state, pr in dist.items():
                if p < 1:
                    nxt[state] += pr * (1 - p)
                if p > 0:
                    for child, q in cand.children(state, i, k):
                        nxt[child] += pr * p * q
            dist = {s: v for s, v in nxt.items() if v != 0}

        if not cand.measurable(i, k):
            continue
        sub: dict = defaultdict(Fraction)
        for state, pr in dist.items():
            sub[_subset_key(state[0])] += pr
        target = Fraction(1, math.comb(i, k))
        max_abs = max(abs(v - target) for v in sub.values())
        # subsets that are unreachable also contribute |0 - target|
        if len(sub) < math.comb(i, k):
            max_abs = max(max_abs, target)
        tv = sum(abs(v - target) for v in sub.values())
        tv += (math.comb(i, k) - len(sub)) * target
        tv = tv / 2
        incl: dict[int, Fraction] = defaultdict(Fraction)
        for subset, pr in sub.items():
            for t in subset:
                incl[t] += pr
        ptarget = Fraction(k, i)
        fo_max = max(abs(incl.get(t, Fraction(0)) - ptarget) for t in range(1, i + 1))
        if max_abs != 0 and first_fail is None:
            first_fail = i
        if max_abs > worst["max_abs_dev"]:
            worst = {
                "max_abs_dev": max_abs,
                "max_rel_dev": max_abs / target,
                "prefix": i,
            }
        row = {
            "prefix": i,
            "max_abs_dev": float(max_abs),
            "max_rel_dev": float(max_abs / target),
            "tv": float(tv),
            "support_size": len(sub),
            "n_subsets": math.comb(i, k),
            "first_order_max_dev": float(fo_max),
            "first_order_exact": fo_max == 0,
            "exact_uniform": max_abs == 0,
        }
        if record_every or i == n:
            prefixes.append(row)

    final = prefixes[-1]
    return {
        "candidate": cid,
        "label": cand.label,
        "n": n,
        "k": k,
        "max_abs_dev": final["max_abs_dev"],
        "max_rel_dev": final["max_rel_dev"],
        "tv": final["tv"],
        "support_size": final["support_size"],
        "n_subsets": final["n_subsets"],
        "first_order_max_dev": final["first_order_max_dev"],
        "first_order_exact_at_n": final["first_order_exact"],
        "exact_uniform_at_n": final["exact_uniform"],
        "first_prefix_failure": first_fail,
        "worst_prefix": int(worst["prefix"]),
        "worst_max_abs_dev": float(worst["max_abs_dev"]),
        "worst_max_rel_dev": float(worst["max_rel_dev"]),
        "first_order_exact_all_prefixes": all(r["first_order_exact"] for r in prefixes)
        if record_every
        else None,
        "prefixes": prefixes if record_every else [],
    }


def verify_many(cids, ns, ks, *, record_every: bool = True) -> list[dict]:
    out: list[dict] = []
    for cid in cids:
        cand = get_candidate(cid)
        for k in ks:
            if cand.needs_even_k and k % 2:
                continue
            for n in ns:
                if n <= k:
                    continue
                if cid == "S5e" and n % k:
                    continue
                out.append(verify(cid, n, k, record_every=record_every))
    return out
