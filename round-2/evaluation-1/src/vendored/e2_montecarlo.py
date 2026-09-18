#!/usr/bin/env python3
"""Vectorised Monte-Carlo harness (M5 / M6) and the bit-accounting sweep (M4).

Monte Carlo is chunked over TRIALS (axis 0) and looped over stream position
(axis 1), never the other way round, and only the ACCEPTED rows are touched at
each position -- so the work is O(k ln(n/k) * T) updates plus one uniform per
row per step, instead of O(n*T) updates.
"""

from __future__ import annotations

import gc
import math
import time

import numpy as np

import e2_budget
from e2_bits import REGIMES, make_source
from e2_candidates import get_candidate
from e2_nullband import (
    PairAccumulator,
    analytic_maxdev,
    coinclusion_stats,
    first_order_stats,
)
from e2_prngs import make_prng

__all__ = ["run_mc", "run_bits", "mc_row_with_band"]


def run_mc(
    cid: str,
    *,
    n: int,
    k: int,
    T: int,
    prng: str = "PCG64",
    seed: int = 20260918,
    chunk: int = 50_000,
    coinclusion: bool = True,
) -> dict:
    """First-order counts and (optionally) full pairwise co-inclusion counts."""
    cand = get_candidate(cid)
    if cand.needs_even_k and k % 2:
        raise ValueError(f"{cid} requires an even k")
    if cand.oracle_n and n % k:
        raise ValueError(f"{cid} requires k | n")
    gen = make_prng(prng, seed)
    counts = np.zeros(n + 1, dtype=np.int64)
    acc = PairAccumulator(n, k) if coinclusion else None
    t0 = time.time()
    done = 0
    while done < T:
        m = min(chunk, T - done)
        res = cand.vector(gen, n, k, m)
        if res.shape != (m, k):
            raise AssertionError(f"{cid} returned {res.shape}, expected {(m, k)}")
        counts += np.bincount(res.ravel(), minlength=n + 1)
        if acc is not None:
            acc.add(np.sort(res, axis=1))
        done += m
        del res
        gc.collect()
    out = {
        "candidate": cid,
        "label": cand.label,
        "family": cand.family,
        "n": n,
        "k": k,
        "T": T,
        "prng": prng,
        "seed": seed,
        "wall_s": round(time.time() - t0, 2),
    }
    out.update(first_order_stats(counts, T, n, k))
    out.update(analytic_maxdev(T, n, k))
    if acc is not None:
        out.update(coinclusion_stats(acc, T))
        del acc
    gc.collect()
    return out


def mc_row_with_band(row: dict, band: dict) -> dict:
    """Express a measured max-deviation in SIMULATED null-band units.

    This is the sentence the whole run was asked for: never a bare percentage.
    """
    out = dict(row)
    sd = band.get("maxdev_null_sd", 0.0)
    mean = band.get("maxdev_null_mean", float("nan"))
    draws = np.asarray(band.get("maxdev_null_draws", []), dtype=np.float64)
    out["maxdev_null_mean"] = mean
    out["maxdev_null_sd"] = sd
    out["maxdev_null_p95"] = band.get("maxdev_null_p95")
    out["maxdev_null_p99"] = band.get("maxdev_null_p99")
    out["z_band"] = (row["maxdev_raw"] - mean) / sd if sd > 0 else float("nan")
    out["exceedance_pct"] = (
        float((draws <= row["maxdev_raw"]).mean() * 100.0) if draws.size else float("nan")
    )
    out["passes_first_order_at_p99"] = bool(
        row["maxdev_raw"] <= band.get("maxdev_null_p99", float("inf"))
    )
    if "coinc_chi2" in row and "coinc_Zmax_null_p99" in band:
        c_mean = band.get("coinc_chi2_null_mean", float("nan"))
        c_sd = band.get("coinc_chi2_null_sd", 0.0)
        out["coinc_Zmax_null_p99"] = band["coinc_Zmax_null_p99"]
        out["coinc_chi2_null_mean"] = c_mean
        out["coinc_chi2_null_sd"] = c_sd
        out["coinc_chi2_null_p99"] = band.get("coinc_chi2_null_p99")
        out["coinc_chi2_z"] = (
            (row["coinc_chi2"] - c_mean) / c_sd if c_sd > 0 else float("nan"))
        # LOOSE rule (reported, not used for the verdict): exceed the p99 of a
        # null band built from only a handful of expensive co-inclusion batches.
        # It has a measurable false-alarm rate and we report it as such.
        out["coinc_fails_at_p99"] = bool(
            row["coinc_Zmax"] > band["coinc_Zmax_null_p99"]
            or row["coinc_chi2"] > band.get("coinc_chi2_null_p99", float("inf"))
        )
        # PRE-REGISTERED rule: the plan fixed the co-inclusion criterion at a
        # NOMINAL chi-square p < 1e-6 before any data were seen.  It is reported
        # at every cell, but it is MIS-CALIBRATED when k/n is not small: the
        # C(n,2) pair counts are strongly negatively correlated (they are
        # constrained to sum to T*C(k,2)), so the chi-square reference
        # distribution is wrong and the rule fires on provably-correct samplers
        # -- measured at n=2000, k=100, where Algorithm R itself gets
        # p = 7.9e-16.  See verdicts.co_inclusion_rule.
        out["coinc_fails_nominal_p"] = bool(row.get("coinc_p", 1.0) < 1e-6)
        # PRIMARY rule (used for every verdict): the SAME statistic referred to
        # its own SIMULATED null, which needs no distributional assumption.
        # Five standard deviations above the simulated null mean is the
        # one-sided analogue of the pre-registered 1e-6.
        z = out.get("coinc_chi2_z")
        out["coinc_fails_calibrated"] = bool(
            isinstance(z, float) and math.isfinite(z) and z >= 5.0)
        out["coinc_fails_prereg"] = out["coinc_fails_calibrated"]
        out["blind_spot"] = bool(
            out["passes_first_order_at_p99"] and out["coinc_fails_calibrated"])
        out["blind_spot_nominal_p_rule"] = bool(
            out["passes_first_order_at_p99"] and out["coinc_fails_nominal_p"])
        out["blind_spot_loose_rule"] = bool(
            out["passes_first_order_at_p99"] and out["coinc_fails_at_p99"])
    return out


def rescore(rows: list[dict], band: dict) -> list[dict]:
    """Re-apply the band/decision rules to stored raw rows (keeps rules in one place)."""
    return [mc_row_with_band(r, band) for r in rows]


def run_bits(
    cid: str,
    *,
    n: int,
    k: int,
    regime: str,
    prng: str = "PCG64",
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> dict:
    """Measured bit cost, averaged over ``seeds``, with raw and net tag splits."""
    if regime not in REGIMES:
        raise KeyError(regime)
    cand = get_candidate(cid)
    if cand.needs_even_k and k % 2:
        raise ValueError(f"{cid} requires an even k")
    if cand.oracle_n and n % k:
        raise ValueError(f"{cid} requires k | n")
    raw_tot, net_tot = [], []
    raw_tags: dict[str, list[float]] = {}
    net_tags: dict[str, list[float]] = {}
    t0 = time.time()
    for sd in seeds:
        src = make_source(regime, prng, sd, label=f"{cid}/{regime}")
        res = cand.scalar(src, n, k)
        if len(set(res)) != k:
            raise AssertionError(f"{cid} produced a reservoir with repeats")
        raw = src.s.tag_totals()
        net = src.net_totals()
        if abs(sum(net.values()) - src.net_total()) > 1e-6:
            raise AssertionError("net tag attribution does not telescope")
        raw_tot.append(src.flips)
        net_tot.append(src.net_total())
        for t, v in raw.items():
            raw_tags.setdefault(t, []).append(float(v))
        for t, v in net.items():
            net_tags.setdefault(t, []).append(float(v))
        del src, res
    a = e2_budget.accept_H(n, k)
    e = e2_budget.evict_H(n, k)
    mean_net = float(np.mean(net_tot))
    return {
        "candidate": cid,
        "label": cand.label,
        "n": n,
        "k": k,
        "regime": regime,
        "prng": prng,
        "n_seeds": len(seeds),
        "wall_s": round(time.time() - t0, 2),
        "flips_total_raw_mean": float(np.mean(raw_tot)),
        "flips_total_raw_sd": float(np.std(raw_tot, ddof=1)) if len(seeds) > 1 else 0.0,
        "flips_total_net_mean": mean_net,
        "flips_total_net_sd": float(np.std(net_tot, ddof=1)) if len(seeds) > 1 else 0.0,
        "raw_by_tag": {t: float(np.mean(v)) for t, v in raw_tags.items()},
        "net_by_tag": {t: float(np.mean(v)) for t, v in net_tags.items()},
        "flips_per_item_net": mean_net / n,
        "closed_form_accept_H": a,
        "closed_form_evict_H": e,
        "closed_form_total_H": a + e,
        "closed_form_naive64": e2_budget.naive_total(n, k, 64),
        "closed_form_ky": e2_budget.ky_accept(n, k) + e,
        "ratio_to_closed_form_entropy": mean_net / (a + e) if (a + e) > 0 else float("nan"),
    }


def bernoulli_variant_costs(
    ps: tuple[tuple[int, int], ...] = ((1, 2), (1, 3), (1, 10), (10, 1000), (1, 100_000)),
    n_calls: int = 300_000,
    seed: int = 4242,
) -> list[dict]:
    """Measure -- never assume -- which recycling Bernoulli is actually cheaper.

    ``RecyclingSource.bernoulli`` draws a uniform on b and pushes the CONDITIONAL
    residual back, so its net cost is h(a/b).  ``bernoulli_dyadic`` runs the
    Knuth-Yao DDG walk drawing its e2_bits from the same buffer, which costs 2
    buffer-e2_bits per call regardless of p.  Both are e2_exact; the plan asked for the
    cheaper one to be reported rather than assumed, so both are measured here.
    """
    import math

    from e2_bits import make_source

    out: list[dict] = []
    for a, b in ps:
        p = a / b
        h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        row = {"a": a, "b": b, "p": p, "binary_entropy_bits": h, "n_calls": n_calls}
        for name, fn in (("recycle_conditional_residual", "bernoulli"),
                         ("recycle_dyadic_KY_walk", "bernoulli_dyadic")):
            src = make_source("recycle", "PCG64", seed)
            hits = 0
            f = getattr(src, fn)
            for _ in range(n_calls):
                hits += f(a, b)
            row[f"{name}_net_bits_per_call"] = src.net_total() / n_calls
            row[f"{name}_raw_bits_per_call"] = src.flips / n_calls
            row[f"{name}_empirical_p"] = hits / n_calls
        row["cheaper"] = (
            "recycle_conditional_residual"
            if row["recycle_conditional_residual_net_bits_per_call"]
            <= row["recycle_dyadic_KY_walk_net_bits_per_call"]
            else "recycle_dyadic_KY_walk")
        row["ratio_dyadic_over_conditional"] = (
            row["recycle_dyadic_KY_walk_net_bits_per_call"]
            / row["recycle_conditional_residual_net_bits_per_call"]
            if row["recycle_conditional_residual_net_bits_per_call"] > 0 else float("inf"))
        out.append(row)
    return out
