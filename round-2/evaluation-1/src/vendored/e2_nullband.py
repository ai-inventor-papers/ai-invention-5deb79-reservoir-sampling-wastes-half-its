#!/usr/bin/env python3
"""Extreme-value null band + pairwise co-inclusion machinery (M5b / M5c).

The single most e2_common error in "is my reservoir sampler uniform?" reports is
quoting a bare max-deviation percentage.  ``max_j |C[j]/T - k/n|`` is a maximum
over n bins, so it has a positive expectation under a PERFECTLY uniform
sampler: about ``sigma * sqrt(2 ln n)`` with ``sigma = sqrt(p(1-p)/T)``.  At
n = 1000, k = 10, T = 1e6 that is 3.7e-4 -- i.e. an exactly correct sampler
"deviates by 0.037 percentage points" and a report that does not say so is
uninterpretable.

This module therefore

  * builds an EXACTLY uniform k-subset sampler by rejection (draw k indices with
    replacement, discard rows with a repeat: conditioning a with-replacement
    draw on distinctness makes every k-subset equally likely once order is
    discarded),
  * SIMULATES the null distribution of the max-deviation statistic at the SAME
    T as the measurement, so the band is measured rather than assumed,
  * and does the same for the pairwise co-inclusion statistics, whose bins are
    negatively correlated in a way no closed form captures cleanly.

Co-inclusion compares the observed count of each unordered pair {a,b} against
the e2_exact ``T * k(k-1) / (n(n-1))``.  It is the cheapest statistic that sees the
JOINT law, which is exactly what first-order frequency testing is blind to.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from e2_prngs import BitGen

__all__ = [
    "exact_uniform_subsets",
    "PairAccumulator",
    "first_order_stats",
    "coinclusion_stats",
    "simulate_null",
    "analytic_maxdev",
]


def exact_uniform_subsets(gen: BitGen, T: int, n: int, k: int) -> np.ndarray:
    """``T`` exactly uniform k-subsets of [1..n], as a sorted ``(T,k)`` array.

    Rejection on repeats.  The duplicate rate is ~C(k,2)/n (4.5% at n=1000,
    k=10) so the loop makes ~1.05 passes.  Never materialises a T x n matrix.
    """
    if k > n:
        raise ValueError("k > n")
    out = gen.random_below(T * k, n).reshape(T, k)
    out.sort(axis=1)
    # ADAPTATION (evaluation artifact, 2026-09-18): the upstream E2 module capped the
    # rejection loop at 200 passes.  Per-pass acceptance is prod_{i<k}(1-i/n) ~
    # exp(-C(k,2)/n), which is 0.086 at (n,k)=(500,50) and (2000,100); with T=2e5 rows
    # 200 passes leaves ~1e-3 expected rows unresolved, and the loop raised
    # RuntimeError mid-run.  The cap is raised; the per-pass cost decays geometrically
    # (only the still-duplicated rows are redrawn), so the extra passes are free.  The
    # SAMPLING LAW IS UNCHANGED -- this is exactly the same rejection sampler, run to
    # completion instead of being truncated.
    for _ in range(200_000):
        bad = (np.diff(out, axis=1) == 0).any(axis=1)
        nb = int(bad.sum())
        if nb == 0:
            return out + 1
        fresh = gen.random_below(nb * k, n).reshape(nb, k)
        fresh.sort(axis=1)
        out[bad] = fresh
    raise RuntimeError("rejection sampler failed to clear duplicates")


class PairAccumulator:
    """Dense co-inclusion counts over the ``n*n`` linearised (min,max) index."""

    def __init__(self, n: int, k: int) -> None:
        self.n = n
        self.k = k
        ia, ib = np.triu_indices(k, 1)
        self.ia = ia
        self.ib = ib
        self.counts = np.zeros(n * n, dtype=np.int64)

    def add(self, res_sorted: np.ndarray) -> None:
        z = res_sorted - 1  # 0-based
        lin = (z[:, self.ia] * self.n + z[:, self.ib]).ravel()
        self.counts += np.bincount(lin, minlength=self.n * self.n)


def first_order_stats(counts: np.ndarray, T: int, n: int, k: int) -> dict:
    p = k / n
    freq = counts[1 : n + 1] / T
    dev = np.abs(freq - p)
    return {
        "maxdev_raw": float(dev.max()),
        "maxdev_argmax_item": int(dev.argmax()) + 1,
        "meanabsdev": float(dev.mean()),
        "rmsdev": float(np.sqrt((dev**2).mean())),
        "chi2_first_order": float(((counts[1 : n + 1] - T * p) ** 2 / (T * p * (1 - p))).sum()),
        "n_items_never_seen": int((counts[1 : n + 1] == 0).sum()),
    }


def coinclusion_stats(acc: PairAccumulator, T: int) -> dict:
    """Exact-expectation co-inclusion test against ``k(k-1)/(n(n-1))``."""
    n, k = acc.n, acc.k
    q = k * (k - 1) / (n * (n - 1))
    npairs = n * (n - 1) // 2
    e = T * q
    v = T * q * (1.0 - q)
    obs = acc.counts
    nz = obs[obs > 0]
    # every unobserved pair contributes (0-e)^2
    chi2 = float(((nz - e) ** 2).sum() + (npairs - nz.size) * e * e) / v
    zmax_pos = float((nz.max() - e) / math.sqrt(v)) if nz.size else 0.0
    zmax_neg = float((0 - e) / math.sqrt(v)) if nz.size < npairs else float((nz.min() - e) / math.sqrt(v))
    zmax = max(abs(zmax_pos), abs(zmax_neg))
    df = npairs - (n - 1)
    return {
        "coinc_expected_per_pair": float(e),
        "coinc_q": float(q),
        "coinc_n_pairs": int(npairs),
        "coinc_pairs_observed": int(nz.size),
        "coinc_chi2": chi2,
        "coinc_df": int(df),
        "coinc_chi2_over_df": chi2 / df,
        "coinc_p": float(stats.chi2.sf(chi2, df)),
        "coinc_Zmax": zmax,
        "coinc_Zmax_pos": zmax_pos,
        "coinc_Zmax_neg": zmax_neg,
    }


def analytic_maxdev(T: int, n: int, k: int) -> dict:
    """Independent-Binomial extreme-value cross-check for the null band."""
    p = k / n
    sigma = math.sqrt(p * (1 - p) / T)
    return {
        "sigma_per_bin": sigma,
        "analytic_maxdev_mean": sigma * math.sqrt(2.0 * math.log(n)),
        "analytic_maxdev_note": (
            "E[max |dev|] ~ sigma*sqrt(2 ln n) for n independent Binomial(T,p) bins; "
            "the true bins are negatively correlated (they sum to kT), so the "
            "simulated band is authoritative and this is the read-back check."
        ),
    }


def simulate_null(
    gen: BitGen,
    *,
    n: int,
    k: int,
    T: int,
    B: int,
    chunk: int,
    B_coinc: int = 0,
) -> dict:
    """Empirical null distribution of maxdev (B batches) and of the
    co-inclusion statistics (``B_coinc`` batches, which are ~10x costlier)."""
    if B_coinc > B:
        raise ValueError(
            f"B_coinc={B_coinc} exceeds B={B}: the co-inclusion null is a SUBSET of the "
            "max-deviation batches, so asking for more of them silently yields fewer")
    maxdevs = np.empty(B, dtype=np.float64)
    chi2s: list[float] = []
    zmaxs: list[float] = []
    for b in range(B):
        counts = np.zeros(n + 1, dtype=np.int64)
        acc = PairAccumulator(n, k) if b < B_coinc else None
        done = 0
        while done < T:
            m = min(chunk, T - done)
            sub = exact_uniform_subsets(gen, m, n, k)
            counts += np.bincount(sub.ravel(), minlength=n + 1)
            if acc is not None:
                acc.add(sub)
            done += m
            del sub
        maxdevs[b] = first_order_stats(counts, T, n, k)["maxdev_raw"]
        if acc is not None:
            cs = coinclusion_stats(acc, T)
            chi2s.append(cs["coinc_chi2"])
            zmaxs.append(cs["coinc_Zmax"])
            del acc
    out = {
        "null_B": int(B),
        "null_T": int(T),
        "maxdev_null_mean": float(maxdevs.mean()),
        "maxdev_null_sd": float(maxdevs.std(ddof=1)) if B > 1 else 0.0,
        "maxdev_null_p50": float(np.percentile(maxdevs, 50)),
        "maxdev_null_p95": float(np.percentile(maxdevs, 95)),
        "maxdev_null_p99": float(np.percentile(maxdevs, 99)),
        "maxdev_null_min": float(maxdevs.min()),
        "maxdev_null_max": float(maxdevs.max()),
        "maxdev_null_draws": [float(x) for x in maxdevs],
    }
    out.update(analytic_maxdev(T, n, k))
    out["maxdev_null_vs_analytic_ratio"] = (
        out["maxdev_null_mean"] / out["analytic_maxdev_mean"]
        if out["analytic_maxdev_mean"] > 0
        else float("nan")
    )
    if chi2s:
        out.update(
            {
                "coinc_null_B": int(len(chi2s)),
                "coinc_chi2_null_mean": float(np.mean(chi2s)),
                "coinc_chi2_null_sd": float(np.std(chi2s, ddof=1)) if len(chi2s) > 1 else 0.0,
                "coinc_chi2_null_p99": float(np.percentile(chi2s, 99)),
                "coinc_Zmax_null_mean": float(np.mean(zmaxs)),
                "coinc_Zmax_null_sd": float(np.std(zmaxs, ddof=1)) if len(zmaxs) > 1 else 0.0,
                "coinc_Zmax_null_p99": float(np.percentile(zmaxs, 99)),
                "coinc_Zmax_null_max": float(np.max(zmaxs)),
            }
        )
    return out
