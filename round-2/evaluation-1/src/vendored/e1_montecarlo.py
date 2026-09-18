"""Pure-numpy Monte-Carlo harness for reservoir samplers.

Implements Algorithm L (exact skip-based) reservoir sampling over the
stream 1..n with pluggable eviction rules, a blind-spot circular-window
sampler (S5-CIRC), first-order and pairwise (second-order) uniformity
diagnostics, an exact-uniform null band, an independence/normal (Gumbel)
analytic approximation to that band, and a naive per-item Bernoulli(k/i)
ground-truth cross-check.

This module does NOT configure loguru sinks itself; the caller is
responsible for that. The only exception is the `if __name__ == "__main__"`
self-test block below, which configures a stdout sink for its own
standalone run.
"""

from __future__ import annotations

import sys
import time
from typing import Callable, Optional

import numpy as np
from loguru import logger
from scipy.stats import norm

EvictRule = Callable[[np.ndarray, np.ndarray, int, np.random.Generator], np.ndarray]


# ---------------------------------------------------------------------------
# Eviction rules (sampler definitions)
# ---------------------------------------------------------------------------


def rule_uniform(
    res: np.ndarray, arriving: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Classical Algorithm R eviction: uniformly random held slot. Sampler S0."""
    return rng.integers(0, k, size=len(res))


def rule_summodk(
    res: np.ndarray, arriving: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Deterministic eviction: 0-based slot ((sum(res) - 1) mod k). Sampler S1n."""
    return (res.sum(axis=1) - 1) % k


def rule_largest(
    res: np.ndarray, arriving: np.ndarray, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Control C-NOACCEPT: always evict the largest held index."""
    return np.full(len(res), k - 1, dtype=np.int64)


_RULES: dict[str, EvictRule] = {
    "uniform": rule_uniform,
    "summodk": rule_summodk,
    "largest": rule_largest,
}


def _evict_insert(
    res_alive: np.ndarray, arriving: np.ndarray, r: np.ndarray, k: int
) -> np.ndarray:
    """Delete position r (0-based) and append `arriving` at the end, vectorised.

    `arriving` is always larger than every currently-held value, and
    `res_alive` is sorted ascending, so deleting any position and appending
    the new max at the end keeps every row sorted, regardless of r.
    """
    cols = np.arange(k)
    mask = cols[None, :] < r[:, None]
    shifted = np.empty_like(res_alive)
    shifted[:, :-1] = res_alive[:, 1:]
    new_vals = np.where(mask, res_alive, shifted)
    new_vals[:, -1] = arriving
    return new_vals


# ---------------------------------------------------------------------------
# Algorithm L
# ---------------------------------------------------------------------------


def simulate_algo_l(
    n: int,
    k: int,
    T: int,
    evict_rule: EvictRule,
    rng: np.random.Generator,
    chunk: int = 100_000,
) -> np.ndarray:
    """Simulate T independent Algorithm-L reservoir streams over 1..n.

    Returns a (T, k) int64 array of final reservoirs, each row sorted
    ascending, values in 1..n. Vectorised across streams in chunks.
    """
    if k <= 0 or k > n:
        raise ValueError(f"require 0 < k <= n, got k={k}, n={n}")

    out = np.empty((T, k), dtype=np.int64)
    start = 0
    while start < T:
        end = min(start + chunk, T)
        M = end - start
        res = np.tile(np.arange(1, k + 1, dtype=np.int64), (M, 1))
        u0 = rng.random(M)
        with np.errstate(divide="ignore"):
            W = np.exp(np.log(u0) / k)
        idx = np.full(M, k, dtype=np.int64)
        alive = np.ones(M, dtype=bool)

        n_iter = 0
        while alive.any():
            u = rng.random(M)
            with np.errstate(divide="ignore", invalid="ignore"):
                skip_f = np.floor(np.log(u) / np.log1p(-W)) + 1.0
            skip_f = np.where(np.isfinite(skip_f) & (skip_f < n + 1), skip_f, n + 1)
            skip = skip_f.astype(np.int64)
            idx = idx + skip
            alive = alive & (idx <= n)
            if alive.any():
                arriving = idx[alive]
                res_alive = res[alive]
                r = np.asarray(evict_rule(res_alive, arriving, k, rng), dtype=np.int64)
                res[alive] = _evict_insert(res_alive, arriving, r, k)
                n_alive = int(alive.sum())
                W[alive] = W[alive] * np.exp(np.log(rng.random(n_alive)) / k)
            n_iter += 1
        logger.debug(
            f"simulate_algo_l chunk [{start}:{end}) converged in {n_iter} outer iterations"
        )
        out[start:end] = res
        start = end
    return out


def simulate_naive_bernoulli(
    n: int, k: int, T: int, evict_rule: EvictRule, rng: np.random.Generator
) -> np.ndarray:
    """Ground-truth cross-check: explicit per-item Bernoulli(k/i) accept, i=k+1..n.

    Vectorised across the T streams (n outer steps), NOT across items.
    Intentionally slow; intended for small T only.
    """
    if k <= 0 or k > n:
        raise ValueError(f"require 0 < k <= n, got k={k}, n={n}")
    res = np.tile(np.arange(1, k + 1, dtype=np.int64), (T, 1))
    for i in range(k + 1, n + 1):
        u = rng.random(T)
        accept = u < (k / i)
        if accept.any():
            sel = np.nonzero(accept)[0]
            res_a = res[sel]
            arriving = np.full(len(sel), i, dtype=np.int64)
            r = np.asarray(evict_rule(res_a, arriving, k, rng), dtype=np.int64)
            res[sel] = _evict_insert(res_a, arriving, r, k)
    return res


# ---------------------------------------------------------------------------
# S5-CIRC blind-spot sampler
# ---------------------------------------------------------------------------


def simulate_circwindow(n: int, k: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """S5-CIRC fast path: anchor j ~ Uniform{1..n} exactly (a k=1 reservoir),
    reservoir = the circular window of k items starting at j.
    """
    j = rng.integers(1, n + 1, size=T).astype(np.int64)
    t = np.arange(k, dtype=np.int64)
    win = ((j[:, None] - 1 + t[None, :]) % n) + 1
    win.sort(axis=1)
    return win


def simulate_circwindow_stepwise(n: int, k: int, T: int, rng: np.random.Generator) -> np.ndarray:
    """S5-CIRC, genuine per-item anchor update: j <- i with prob 1/i, i = 1..n.

    Vectorised over the T streams; the outer loop runs n steps. Used only
    as a small-n validation cross-check of `simulate_circwindow`.
    """
    j = np.ones(T, dtype=np.int64)
    for i in range(2, n + 1):
        u = rng.random(T)
        accept = u < (1.0 / i)
        if accept.any():
            j = np.where(accept, i, j)
    t = np.arange(k, dtype=np.int64)
    win = ((j[:, None] - 1 + t[None, :]) % n) + 1
    win.sort(axis=1)
    return win


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def first_order_stats(reservoirs: np.ndarray, n: int, k: int) -> dict:
    """Per-item marginal-inclusion-frequency diagnostics."""
    T = reservoirs.shape[0]
    counts = np.bincount(reservoirs.ravel(), minlength=n + 1)[1:].astype(np.int64)
    expected = k / n
    freq = counts / T
    dev = np.abs(freq - expected)
    max_dev = float(dev.max())
    argmax_item = int(np.argmax(dev)) + 1
    mean_abs_dev = float(dev.mean())
    exp_count = T * expected
    chi2 = float(np.sum((counts - exp_count) ** 2 / exp_count))
    counts_summary = {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
    }
    return {
        "max_dev": max_dev,
        "argmax_item": argmax_item,
        "mean_abs_dev": mean_abs_dev,
        "chi2": chi2,
        "counts_summary": counts_summary,
    }


def pair_stats(reservoirs: np.ndarray, n: int, k: int) -> dict:
    """Pairwise co-inclusion diagnostics, chunked to bound peak memory."""
    T = reservoirs.shape[0]
    npairs = n * (n - 1) // 2
    counts = np.zeros(npairs, dtype=np.int64)
    iu0, iu1 = np.triu_indices(k, k=1)
    step = 100_000
    for start in range(0, T, step):
        end = min(start + step, T)
        chunk0 = (reservoirs[start:end] - 1).astype(np.int64)  # 0-based item ids
        a = chunk0[:, iu0]
        b = chunk0[:, iu1]
        pidx = a * n - a * (a + 1) // 2 + (b - a - 1)
        flat = pidx.ravel()
        counts += np.bincount(flat, minlength=npairs)
        del chunk0, a, b, pidx, flat

    p = k * (k - 1) / (n * (n - 1))
    mean = T * p
    sd = np.sqrt(T * p * (1 - p))
    z = (counts - mean) / sd
    abs_z = np.abs(z)
    max_abs_z = float(abs_z.max())
    argmax_idx = int(np.argmax(abs_z))

    a_range = np.arange(n, dtype=np.int64)
    starts = a_range * n - a_range * (a_range + 1) // 2
    a_val = int(np.searchsorted(starts, argmax_idx, side="right") - 1)
    b_val = argmax_idx - int(starts[a_val]) + a_val + 1
    argmax_pair = [a_val + 1, b_val + 1]

    freq = counts / T
    max_dev = float(np.max(np.abs(freq - p)))
    n_zero_pairs = int(np.sum(counts == 0))
    frac_zero_pairs = float(n_zero_pairs / npairs)

    del counts, z, abs_z, freq, starts, a_range
    return {
        "max_abs_z": max_abs_z,
        "argmax_pair": argmax_pair,
        "max_dev": max_dev,
        "n_pairs": int(npairs),
        "n_zero_pairs": n_zero_pairs,
        "frac_zero_pairs": frac_zero_pairs,
    }


# ---------------------------------------------------------------------------
# Exact-uniform null band
# ---------------------------------------------------------------------------


def _uniform_subsets_reject(n: int, k: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """Exact uniform k-subsets of [n] via rejection sampling (fast, exact)."""
    cand = rng.integers(0, n, size=(M, k))
    while True:
        cand.sort(axis=1)
        dup = np.any(cand[:, 1:] == cand[:, :-1], axis=1)
        if not dup.any():
            break
        nbad = int(dup.sum())
        cand[dup] = rng.integers(0, n, size=(nbad, k))
    return (cand + 1).astype(np.int64)


def _uniform_subsets_bottomk(n: int, k: int, M: int, rng: np.random.Generator) -> np.ndarray:
    """Exact uniform k-subsets of [n] via bottom-k random keys (slow reference)."""
    keys = rng.random((M, n))
    idx = np.argpartition(keys, k, axis=1)[:, :k]
    res = (idx + 1).astype(np.int64)
    res.sort(axis=1)
    return res


def null_band(
    n: int, k: int, T: int, R: int, rng: np.random.Generator, method: str = "reject"
) -> dict:
    """Simulate R independent replicate batches of T exact uniform k-subsets
    of [n] and record the per-replicate first-order max_dev and pairwise
    max_abs_z, returning their empirical distributions.
    """
    if method not in ("reject", "bottomk"):
        raise ValueError(f"unknown method: {method}")
    draw = _uniform_subsets_reject if method == "reject" else _uniform_subsets_bottomk

    fo_vals: list[float] = []
    pz_vals: list[float] = []
    for rep in range(R):
        res = draw(n, k, T, rng)
        fo = first_order_stats(res, n, k)
        pr = pair_stats(res, n, k)
        fo_vals.append(fo["max_dev"])
        pz_vals.append(pr["max_abs_z"])
        del res
        logger.debug(
            f"null_band[{method}] replicate {rep + 1}/{R} "
            f"max_dev={fo['max_dev']:.6g} max_abs_z={pr['max_abs_z']:.4g}"
        )

    def _summary(vals: list[float]) -> dict:
        arr = np.asarray(vals, dtype=np.float64)
        return {
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50, method="linear")),
            "p95": float(np.percentile(arr, 95, method="linear")),
            "p99": float(np.percentile(arr, 99, method="linear")),
            "max": float(arr.max()),
            "values": [float(v) for v in vals],
        }

    return {
        "R": R,
        "T": T,
        "method": method,
        "first_order": _summary(fo_vals),
        "pair_z": _summary(pz_vals),
    }


def gumbel_band_analytic(n: int, k: int, T: int) -> dict:
    """Independence/normal (extreme-value) approximation to the null band."""
    p = k / n
    sd_freq = float(np.sqrt(p * (1 - p) / T))

    def _gumbel_ab(m: int) -> tuple[float, float]:
        L = np.log(2 * m)
        b = np.sqrt(2 * L) - (np.log(L) + np.log(4 * np.pi)) / (2 * np.sqrt(2 * L))
        a = 1 / np.sqrt(2 * L)
        return float(a), float(b)

    a1, b1 = _gumbel_ab(n)
    p95_std = b1 + a1 * (-np.log(-np.log(0.95)))

    npairs = n * (n - 1) // 2
    a2, b2 = _gumbel_ab(npairs)
    pair_p95_std = b2 + a2 * (-np.log(-np.log(0.95)))

    return {
        "sd_freq": sd_freq,
        "expected_max_dev": float(b1 * sd_freq),
        "p95_max_dev": float(p95_std * sd_freq),
        "pair_expected_max_z": float(b2),
        "pair_p95_max_z": float(pair_p95_std),
    }


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_mc(
    name: str,
    n: int,
    k: int,
    T: int,
    R: int,
    seed: int,
    sampler: str = "algo_l",
    rule: str = "uniform",
    null: Optional[dict] = None,
) -> dict:
    """Run one candidate sampler through simulate -> diagnostics -> null-band
    comparison, returning a single results dict.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    t0 = time.perf_counter()

    if sampler == "algo_l":
        if rule not in _RULES:
            raise ValueError(f"unknown rule: {rule}")
        reservoirs = simulate_algo_l(n, k, T, _RULES[rule], rng)
    elif sampler == "circwindow":
        reservoirs = simulate_circwindow(n, k, T, rng)
    elif sampler == "circwindow_stepwise":
        reservoirs = simulate_circwindow_stepwise(n, k, T, rng)
    else:
        raise ValueError(f"unknown sampler: {sampler}")

    fo = first_order_stats(reservoirs, n, k)
    pr = pair_stats(reservoirs, n, k)

    if null is None:
        null = null_band(n, k, T, R, rng)

    wall_seconds = time.perf_counter() - t0

    p95_fo = null["first_order"]["p95"]
    p95_pz = null["pair_z"]["p95"]
    max_dev_in_band_units = fo["max_dev"] / p95_fo
    flagged_first_order = bool(fo["max_dev"] > p95_fo)
    pair_max_z_in_band_units = pr["max_abs_z"] / p95_pz
    flagged_second_order = bool(pr["max_abs_z"] > p95_pz)

    npairs = n * (n - 1) // 2
    pair_p_value_normal = float(min(1.0, 2 * npairs * norm.sf(pr["max_abs_z"])))
    # The linear p-value underflows to exactly 0.0 for large z, which reads as "missing"
    # rather than "astronomically significant"; carry the log10 as well.
    pair_log10_p_value = float(
        min(0.0, (norm.logsf(pr["max_abs_z"]) + np.log(2.0 * npairs)) / np.log(10.0))
    )

    logger.info(
        f"run_mc[{name}] n={n} k={k} T={T} sampler={sampler}/{rule} "
        f"max_dev={fo['max_dev']:.4g} ({max_dev_in_band_units:.2f}x p95) "
        f"max_abs_z={pr['max_abs_z']:.3f} ({pair_max_z_in_band_units:.2f}x p95) "
        f"wall={wall_seconds:.2f}s"
    )

    return {
        "candidate": name,
        "n": n,
        "k": k,
        "T": T,
        "seed": seed,
        "sampler": sampler,
        "rule": rule,
        "wall_seconds": wall_seconds,
        "first_order": fo,
        "pair": pr,
        "max_dev_in_band_units": max_dev_in_band_units,
        "flagged_first_order": flagged_first_order,
        "pair_max_z_in_band_units": pair_max_z_in_band_units,
        "flagged_second_order": flagged_second_order,
        "pair_p_value_normal": pair_p_value_normal,
        "pair_log10_p_value": pair_log10_p_value,
    }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

    @logger.catch(reraise=True)
    def _self_test() -> None:
        t_start = time.perf_counter()
        failures: list[str] = []

        # --- check 1: algo_l uniform, single-sample calibration -----------
        n1, k1, T1 = 20, 3, 200_000
        rng1 = np.random.Generator(np.random.PCG64(1001))
        res1 = simulate_algo_l(n1, k1, T1, rule_uniform, rng1)
        fo1 = first_order_stats(res1, n1, k1)
        bound1 = 6 * np.sqrt((k1 / n1) * (1 - k1 / n1) / T1)
        ok1 = fo1["max_dev"] < bound1
        logger.info(
            f"[check1] algo_l(uniform) n={n1} k={k1} T={T1}: "
            f"max_dev={fo1['max_dev']:.6g} bound(6sigma)={bound1:.6g} -> {'PASS' if ok1 else 'FAIL'}"
        )
        if not ok1:
            failures.append("check1")

        # --- check 2: algo_l vs naive_bernoulli agreement ------------------
        n2, k2, T2 = 50, 5, 20_000
        rng2a = np.random.Generator(np.random.PCG64(2001))
        rng2b = np.random.Generator(np.random.PCG64(2002))
        res2a = simulate_algo_l(n2, k2, T2, rule_uniform, rng2a)
        res2b = simulate_naive_bernoulli(n2, k2, T2, rule_uniform, rng2b)
        fo2a = first_order_stats(res2a, n2, k2)
        fo2b = first_order_stats(res2b, n2, k2)
        bound2 = 6 * np.sqrt((k2 / n2) * (1 - k2 / n2) / T2)
        ok2a = fo2a["max_dev"] < bound2
        ok2b = fo2b["max_dev"] < bound2
        counts2a = np.bincount(res2a.ravel(), minlength=n2 + 1)[1:]
        counts2b = np.bincount(res2b.ravel(), minlength=n2 + 1)[1:]
        p2 = k2 / n2
        sd_diff2 = np.sqrt(2 * p2 * (1 - p2) / T2)
        max_diff_sigma2 = float((np.abs(counts2a / T2 - counts2b / T2) / sd_diff2).max())
        ok2c = max_diff_sigma2 < 8
        ok2 = ok2a and ok2b and ok2c
        logger.info(
            f"[check2] algo_l vs naive_bernoulli n={n2} k={k2} T={T2}: "
            f"algo_l max_dev={fo2a['max_dev']:.6g} naive max_dev={fo2b['max_dev']:.6g} "
            f"bound(6sigma)={bound2:.6g} max_elementwise_diff={max_diff_sigma2:.2f}sigma(<8) "
            f"-> {'PASS' if ok2 else 'FAIL'}"
        )
        if not ok2:
            failures.append("check2")

        # --- check 3: null_band reject vs bottomk agreement -----------------
        n3, k3, T3, R3 = 100, 5, 20_000, 6
        rng3a = np.random.Generator(np.random.PCG64(3001))
        rng3b = np.random.Generator(np.random.PCG64(3002))
        nb3_reject = null_band(n3, k3, T3, R3, rng3a, method="reject")
        nb3_bottomk = null_band(n3, k3, T3, R3, rng3b, method="bottomk")
        p95_r = nb3_reject["first_order"]["p95"]
        p95_b = nb3_bottomk["first_order"]["p95"]
        ratio3 = max(p95_r, p95_b) / min(p95_r, p95_b)
        ok3 = ratio3 <= 2.0
        logger.info(
            f"[check3] null_band reject vs bottomk n={n3} k={k3} T={T3} R={R3}: "
            f"p95_reject={p95_r:.6g} p95_bottomk={p95_b:.6g} ratio={ratio3:.3f}(<=2) "
            f"-> {'PASS' if ok3 else 'FAIL'}"
        )
        if not ok3:
            failures.append("check3")

        # --- check 4: circwindow inside null band, near-total pair zeros ---
        n4, k4, T4 = 60, 5, 100_000
        rng4 = np.random.Generator(np.random.PCG64(4001))
        res4 = simulate_circwindow(n4, k4, T4, rng4)
        fo4 = first_order_stats(res4, n4, k4)
        pr4 = pair_stats(res4, n4, k4)
        rng4n = np.random.Generator(np.random.PCG64(4002))
        nb4 = null_band(n4, k4, T4, 8, rng4n, method="reject")
        p99_4 = nb4["first_order"]["p99"]
        ok4a = fo4["max_dev"] <= p99_4
        # NOTE: for a circular window of width k, the number of realizable
        # (ever-nonzero) pairs is exactly (k-1)*n, a deterministic
        # combinatorial fact, not a statistical fluctuation. At n=60,k=5
        # that gives zero_frac = 1 - 2*(k-1)/(n-1) = 0.8644 exactly, so the
        # spec's literal ">0.9" bound is unreachable by any correct
        # implementation at these exact params; using 0.8 here preserves
        # the intended blind-spot demonstration (see final report).
        ok4b = pr4["n_zero_pairs"] > 0.8 * pr4["n_pairs"]
        ok4 = ok4a and ok4b
        logger.info(
            f"[check4] circwindow n={n4} k={k4} T={T4}: max_dev={fo4['max_dev']:.6g} "
            f"null_p99={p99_4:.6g}(in-band={ok4a}) "
            f"n_zero_pairs={pr4['n_zero_pairs']}/{pr4['n_pairs']}={pr4['frac_zero_pairs']:.4f}(>80%={ok4b}) "
            f"-> {'PASS' if ok4 else 'FAIL'}"
        )
        if not ok4:
            failures.append("check4")

        # --- check 5: circwindow fast path vs stepwise cross-check ---------
        n5, k5, T5 = 40, 4, 50_000
        rng5a = np.random.Generator(np.random.PCG64(5001))
        rng5b = np.random.Generator(np.random.PCG64(5002))
        res5a = simulate_circwindow(n5, k5, T5, rng5a)
        res5b = simulate_circwindow_stepwise(n5, k5, T5, rng5b)
        fo5a = first_order_stats(res5a, n5, k5)
        fo5b = first_order_stats(res5b, n5, k5)
        lo5, hi5 = sorted([fo5a["max_dev"], fo5b["max_dev"]])
        ok5 = hi5 <= 3 * lo5 if lo5 > 0 else hi5 == 0.0
        logger.info(
            f"[check5] circwindow vs stepwise n={n5} k={k5} T={T5}: "
            f"fast max_dev={fo5a['max_dev']:.6g} stepwise max_dev={fo5b['max_dev']:.6g} "
            f"(within 3x) -> {'PASS' if ok5 else 'FAIL'}"
        )
        if not ok5:
            failures.append("check5")

        # --- check 6: determinism -------------------------------------------
        rng6a = np.random.Generator(np.random.PCG64(6001))
        rng6b = np.random.Generator(np.random.PCG64(6001))
        res6a = simulate_algo_l(30, 4, 5_000, rule_uniform, rng6a)
        res6b = simulate_algo_l(30, 4, 5_000, rule_uniform, rng6b)
        ok6 = np.array_equal(res6a, res6b)
        logger.info(f"[check6] determinism (same seed, two calls): identical={ok6} -> {'PASS' if ok6 else 'FAIL'}")
        if not ok6:
            failures.append("check6")

        wall_selftest = time.perf_counter() - t_start
        logger.info(f"self-test total wall time: {wall_selftest:.2f}s")

        if failures:
            logger.error(f"SELF-TEST FAILED: {failures}")
            sys.exit(1)
        logger.info("SELF-TEST PASSED: all 6 checks OK")

        # --- acceptance timing: run_mc("S0", n=1000,k=10,T=200000,R=4) -----
        n7, k7, T7, R7 = 1000, 10, 200_000, 4
        rng7 = np.random.Generator(np.random.PCG64(1))
        t0 = time.perf_counter()
        res7 = simulate_algo_l(n7, k7, T7, rule_uniform, rng7)
        t_sim = time.perf_counter() - t0
        t0 = time.perf_counter()
        fo7 = first_order_stats(res7, n7, k7)
        t_fo = time.perf_counter() - t0
        t0 = time.perf_counter()
        pr7 = pair_stats(res7, n7, k7)
        t_pair = time.perf_counter() - t0
        t0 = time.perf_counter()
        _nb7 = null_band(n7, k7, T7, R7, rng7, method="reject")
        t_null = time.perf_counter() - t0
        logger.info(
            f"[timing-breakdown] n={n7} k={k7} T={T7} R={R7}: "
            f"simulate={t_sim:.3f}s first_order={t_fo:.3f}s pair={t_pair:.3f}s "
            f"null={t_null:.3f}s total={t_sim + t_fo + t_pair + t_null:.3f}s "
            f"(fo.max_dev={fo7['max_dev']:.4g} pair.max_abs_z={pr7['max_abs_z']:.3f})"
        )

        result7 = run_mc("S0", n7, k7, T7, R7, seed=1)
        logger.info(
            f"[timing-run_mc] run_mc('S0', n={n7}, k={k7}, T={T7}, R={R7}, seed=1) "
            f"wall_seconds={result7['wall_seconds']:.3f}s "
            f"flagged_first_order={result7['flagged_first_order']} "
            f"flagged_second_order={result7['flagged_second_order']}"
        )

    _self_test()
