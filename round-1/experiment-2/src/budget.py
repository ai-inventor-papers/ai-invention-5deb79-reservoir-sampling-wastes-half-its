#!/usr/bin/env python3
"""Closed-form randomness budgets for reservoir sampling.

These make the n = 1e7 numbers EXACT rather than extrapolated, and give every
measured bit count an independent route to be read back against.

Definitions (all in bits, all for a stream of length ``n`` and reservoir ``k``):

ACCEPT_H(n,k)  = sum_{i=k+1}^{n} h(k/i)
    The information-theoretic floor for the accept/reject decisions alone.
    Each item i is accepted with probability exactly k/i, so the decision
    carries h(k/i) bits and no exact sampler can spend less on it.

EVICT_H(n,k)   = log2(k) * sum_{i=k+1}^{n} (k/i)
    The classical eviction cost: every acceptance draws a uniform slot, and the
    expected number of acceptances is sum_{i>k} k/i = k*(H_n - H_k).

KY_ACCEPT(n,k) = 2 * (n - k)
    What an entropy-optimal-PER-DECISION coder spends on the same decisions:
    a Knuth-Yao DDG tree for a Bernoulli costs exactly 2 flips on average for
    EVERY p.  This is the "swamping" constant behind candidate 4.

NAIVE64(n,k)   = 64*(n-k) + 64*k*(H_n - H_k)
    What a deployed library spends: one 64-bit word per decision.

OUTPUT_H(n,k)  = log2(C(n,k))
    The entropy of the answer itself.  ``anytime_overhead`` is how much more a
    *streaming, anytime-correct* sampler must spend than a sampler that knew n.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.special import gammaln

__all__ = [
    "h",
    "harmonic",
    "accept_H",
    "evict_H",
    "ky_accept",
    "naive_total",
    "output_H",
    "budget_table",
]


def h(p: np.ndarray | float) -> np.ndarray | float:
    """Binary entropy in bits, with h(0) = h(1) = 0."""
    p = np.asarray(p, dtype=np.float64)
    out = np.zeros_like(p)
    m = (p > 0.0) & (p < 1.0)
    pm = p[m] if p.ndim else p
    if p.ndim == 0:
        if not m:
            return 0.0
        return float(-p * np.log2(p) - (1 - p) * np.log2(1 - p))
    out[m] = -pm * np.log2(pm) - (1 - pm) * np.log2(1 - pm)
    return out


def harmonic(n: int) -> float:
    """H_n computed by exact summation for small n, Euler-Maclaurin for large."""
    if n <= 10_000_000:
        return float(np.sum(1.0 / np.arange(1, n + 1, dtype=np.float64)))
    g = 0.5772156649015328606
    x = float(n)
    return (
        math.log(x)
        + g
        + 1.0 / (2 * x)
        - 1.0 / (12 * x * x)
        + 1.0 / (120 * x**4)
    )


def accept_H(n: int, k: int) -> float:
    """sum_{i=k+1}^{n} h(k/i) -- chunked so n = 1e7 costs ~0.1 s and O(1) RAM."""
    if n <= k:
        return 0.0
    total = 0.0
    step = 1 << 22
    for lo in range(k + 1, n + 1, step):
        hi = min(lo + step - 1, n)
        i = np.arange(lo, hi + 1, dtype=np.float64)
        p = k / i
        total += float(np.sum(-p * np.log2(p) - (1 - p) * np.log2(1 - p)))
    return total


def _expected_accepts(n: int, k: int) -> float:
    """sum_{i=k+1}^{n} k/i = k*(H_n - H_k) -- the expected number of evictions."""
    if n <= k:
        return 0.0
    return k * (harmonic(n) - harmonic(k))


def evict_H(n: int, k: int) -> float:
    return math.log2(k) * _expected_accepts(n, k) if k > 1 else 0.0


def ky_accept(n: int, k: int) -> float:
    return 2.0 * max(n - k, 0)


def naive_total(n: int, k: int, word_bits: int = 64) -> float:
    return word_bits * max(n - k, 0) + word_bits * _expected_accepts(n, k)


def output_H(n: int, k: int) -> float:
    """log2 C(n,k) via gammaln, cross-checked against math.comb for n <= 1e5."""
    val = (gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)) / math.log(2.0)
    if n <= 100_000:
        exact = math.log2(math.comb(n, k))
        if abs(exact - val) > 1e-6 * max(1.0, abs(exact)):
            raise AssertionError(
                f"output_H mismatch at n={n},k={k}: gammaln {val} vs exact {exact}"
            )
        return exact
    return float(val)


def budget_table(ns: list[int], ks: list[int]) -> list[dict]:
    rows: list[dict] = []
    for n in ns:
        for k in ks:
            if k >= n:
                continue
            a = accept_H(n, k)
            e = evict_H(n, k)
            o = output_H(n, k)
            rows.append(
                {
                    "n": int(n),
                    "k": int(k),
                    "accept_H": a,
                    "evict_H": e,
                    "total_H": a + e,
                    "output_H": o,
                    "ky_accept": ky_accept(n, k),
                    "naive64_total": naive_total(n, k, 64),
                    "naive53_total": naive_total(n, k, 53),
                    "expected_accepts": _expected_accepts(n, k),
                    "evict_frac": e / (a + e) if (a + e) > 0 else 0.0,
                    "anytime_overhead": (a + e) / o if o > 0 else float("nan"),
                }
            )
    return rows
