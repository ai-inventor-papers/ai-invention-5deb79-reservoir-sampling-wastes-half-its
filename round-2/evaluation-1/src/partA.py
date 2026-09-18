#!/usr/bin/env python3
"""PART A -- the L1 identity / collision tester as a THIRD detector column.

A1  calibration of the tester (simulated null + planted-eps power curve -> measured c in
    T = c * sqrt(D) / eps^2)
A2  panel 1: 18 matched cells (n <= 14, k in {2,4}) against exact-rational ground truth
A3  panel 2: 72 replicated runs (8 seeds x 9 samplers, n=1000, k=10, T=1e6)
A4  the sample-complexity / state-complexity separation table
A5  the anytime (all-prefix) column for every detector
"""

from __future__ import annotations

import gc
import json
import sys
import time
from math import comb, log, log10, log2, sqrt
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendored"))

import l1tester as L

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
RES.mkdir(exist_ok=True)
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(HERE / "logs" / "partA.log"), rotation="30 MB", level="DEBUG")

MATCHED_CELLS = [(12, 2), (12, 4), (14, 4)]
MATCHED_KEYS = ["S0", "S1", "S1n", "S1f", "S5-CIRC", "C-NOACCEPT"]
MATCHED_T = 100_000
MATCHED_SEED = 20260918

BIG_N, BIG_K = 1000, 10
BIG_SAMPLERS = ["S0", "S1h", "S2", "S2f", "S2v", "S3", "S5e", "S5g", "S6sys"]
BIG_BLIND = {"S5e", "S5g", "S6sys"}
BIG_SEEDS = list(range(8))

ALPHA = 0.01
NULL_R_MATCHED = 500


# =======================================================================================
# A1 -- calibration
# =======================================================================================


def a1_null_distribution(D: int, T: int, R: int, seed: int) -> dict:
    """Simulated null of Z and C from an independent EXACT uniform oracle."""
    rng = np.random.Generator(np.random.PCG64(seed))
    zs = np.empty(R)
    cs = np.empty(R)
    for r in range(R):
        atoms = L.uniform_sample_atoms(D, T, rng)
        c, _ = L.collision_count_from_keys(atoms)
        cs[r] = c
        zs[r] = L.z_statistic_uniform(c, T, D)
    mom = L.null_moments_uniform(T, D)
    return {
        "D": int(D),
        "T": int(T),
        "R": int(R),
        "Z_mean": float(zs.mean()),
        "Z_sd": float(zs.std(ddof=1)),
        "Z_p99": float(np.percentile(zs, 99)),
        "C_mean": float(cs.mean()),
        "C_sd": float(cs.std(ddof=1)),
        "C_p99": float(np.percentile(cs, 99)),
        "Z_analytic_mean": mom["E_Z"],
        "Z_analytic_sd": mom["SD_Z"],
        "C_analytic_mean": mom["E_C"],
        "C_analytic_sd": mom["SD_C"],
        "values_Z": [float(v) for v in zs[: min(R, 500)]],
    }


def _power_cell(task: tuple) -> list[dict]:
    """Power of the tester over an ABSOLUTE geometric grid of T for one (D, eps) cell.

    The grid is in absolute T, NOT in multiples of sqrt(D)/eps^2.  Parameterising the grid
    by sqrt(D)/eps^2 would make the scaling test circular: T_90 would be forced onto the
    grid multiple and the fitted exponents would reproduce the theory by construction.
    """
    D, eps, reps, seed, alpha, ratio, T0, T_cap, draw_budget = task
    rows = []
    T = T0
    prev = None
    while T <= T_cap:
        R = int(max(150, min(reps, draw_budget // max(1, 2 * T))))
        rng_n = np.random.Generator(np.random.PCG64(seed + 10_000 + D))
        nullz = np.empty(R)
        for r in range(R):
            c, _ = L.collision_count_from_keys(L.uniform_sample_atoms(D, T, rng_n))
            nullz[r] = L.z_statistic_uniform(c, T, D)
        thr = float(np.percentile(nullz, 100 * (1 - alpha)))
        rng_a = np.random.Generator(np.random.PCG64(seed + 20_000 + D))
        hits = 0
        for r in range(R):
            atoms = L.planted_alternative_sample(D, eps, T, rng_a, half_mask_seed=seed)
            c, _ = L.collision_count_from_keys(atoms)
            if L.z_statistic_uniform(c, T, D) > thr:
                hits += 1
        power = hits / R
        rows.append({"D": int(D), "eps": float(eps), "T": int(T), "replicates": int(R),
                     "power": float(power),
                     "power_se": float(sqrt(max(power * (1 - power), 1e-12) / R)),
                     "threshold_Z": thr,
                     "T_over_sqrtD_over_eps2": float(T * eps**2 / sqrt(D))})
        if power >= 0.9:
            # log-linear interpolation of the 0.9 crossing between the last two grid points
            if prev is not None and prev["power"] < 0.9 < power + 1e-12:
                f = (0.9 - prev["power"]) / (power - prev["power"])
                t90 = float(np.exp(np.log(prev["T"]) + f * (np.log(T) - np.log(prev["T"]))))
            else:
                t90 = float(T)
            rows[-1]["T_90_interpolated"] = t90
            break
        prev = rows[-1]
        T = int(round(T * ratio))
    return rows


def a1_power_curve(*, Ds, epss, reps: int, seed: int, alpha: float = ALPHA,
                   draw_budget: int = 30_000_000, ratio: float = 2 ** 0.5,
                   T0: int = 32, T_cap: int = 4_000_000, workers: int = 12) -> dict:
    """Empirical power of the tester, on an absolute geometric T grid, one process per cell."""
    import multiprocessing as mp

    tasks = [(D, eps, reps, seed, alpha, ratio, T0, T_cap, draw_budget)
             for D in Ds for eps in epss]
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=min(workers, len(tasks))) as pool:
        results = pool.map(_power_cell, tasks)
    rows = [r for chunk in results for r in chunk]

    fits = {}
    for D in Ds:
        for eps in epss:
            sel = [r for r in rows if r["D"] == D and r["eps"] == eps
                   and "T_90_interpolated" in r]
            if sel:
                fits[f"D={D},eps={eps}"] = {
                    "T_90": float(sel[0]["T_90_interpolated"]),
                    "T_90_grid_point": int(sel[0]["T"]),
                    "replicates_at_crossing": int(sel[0]["replicates"]),
                    "c_measured": float(sel[0]["T_90_interpolated"] * eps**2 / sqrt(D)),
                }
    cs = [v["c_measured"] for v in fits.values()]
    return {
        "grid_rows": rows,
        "grid_design": {"absolute_geometric_grid": True, "T0": T0, "ratio": ratio,
                        "T_cap": T_cap,
                        "why": "the grid is in absolute T so the sqrt(D)/eps^2 scaling "
                               "test is not circular"},
        "fits": fits,
        "c_measured_median": float(np.median(cs)) if cs else None,
        "c_measured_min": float(min(cs)) if cs else None,
        "c_measured_max": float(max(cs)) if cs else None,
        "c_spread_ratio": float(max(cs) / min(cs)) if cs else None,
        "law": "T_90 = c * sqrt(D) / eps^2 (Acharya-Daskalakis-Kamath / "
               "Diakonikolas-Kane-Nikishkin; collision testers optimal for uniformity, "
               "Diakonikolas-Gouleakis-Peebles-Price)",
    }


# =======================================================================================
# A2 -- matched panel
# =======================================================================================


def _simulate_matched(key: str, n: int, k: int, T: int, rng) -> np.ndarray:
    """Final reservoirs for one matched-panel candidate, via the vendored E1 harness."""
    import e1_montecarlo as mc
    import e1_verdicts as V

    sampler, rule = V.resolve_rule(key, k)
    if sampler == "circwindow":
        return mc.simulate_circwindow_stepwise(n, k, T, rng)
    return mc.simulate_algo_l(n, k, T, rule, rng)


def _simulate_matched_prefix_ranks(key: str, n: int, k: int, T: int, rng) -> dict[int, np.ndarray]:
    """Snapshot the k-subset RANK of the reservoir at EVERY prefix i = k..n.

    Returns {i: int64 array of length T of ranks in [0, C(i,k)) }.  Only ranks are kept,
    never the reservoirs, so memory is O(T * (n-k)) int64 at n <= 14.
    """
    import e1_montecarlo as mc
    import e1_verdicts as V

    sampler, rule = V.resolve_rule(key, k)
    out: dict[int, np.ndarray] = {}
    if sampler == "circwindow":
        anchor = np.ones(T, dtype=np.int64)
        t = np.arange(k, dtype=np.int64)
        for i in range(1, n + 1):
            if i >= 2:
                anchor = np.where(rng.random(T) < 1.0 / i, i, anchor)
            if i >= k:
                win = ((anchor[:, None] - 1 + t[None, :]) % i) + 1
                win.sort(axis=1)
                out[i] = L.combinadic_rank(win, i, k)
    else:
        res = np.tile(np.arange(1, k + 1, dtype=np.int64), (T, 1))
        out[k] = L.combinadic_rank(res, k, k)
        for i in range(k + 1, n + 1):
            accept = rng.random(T) < (k / i)
            if accept.any():
                sel = np.nonzero(accept)[0]
                res_a = res[sel]
                arriving = np.full(len(sel), i, dtype=np.int64)
                r = np.asarray(rule(res_a, arriving, k, rng), dtype=np.int64)
                res[sel] = mc._evict_insert(res_a, arriving, r, k)
            out[i] = L.combinadic_rank(res, i, k)
    return out


def _l1_null_matched(n: int, k: int, T: int, R: int, seed: int) -> dict:
    """Own-simulated null of the FINAL and ANYTIME L1 statistics, from an exact uniform
    k-subset oracle (numpy Generator.choice without replacement, independent of every
    candidate under audit)."""
    rng = np.random.Generator(np.random.PCG64(seed))
    D = comb(n, k)
    final_z, anytime_z = np.empty(R), np.empty(R)
    # D = C(i,k) == 1 at i == k: a single possible subset, so the statistic is degenerate
    # (SD_Z == 0) and the prefix carries no information.  Excluded from the anytime max.
    prefixes = [i for i in range(k, n + 1) if comb(i, k) >= 2]
    Ds = {i: comb(i, k) for i in prefixes}
    for r in range(R):
        # exact uniform k-subset of [i] independently at every prefix: the anytime null
        # for a sampler that is uniform at every prefix is NOT independent across
        # prefixes, so the null is generated by the classical sampler S0 itself, whose
        # anytime uniformity is certified exactly by E1 gate G1.
        ranks = _simulate_matched_prefix_ranks("S0", n, k, T, rng)
        zs = []
        for i in prefixes:
            c, _ = L.collision_count_from_keys(ranks[i])
            m = L.null_moments_uniform(T, Ds[i])
            zs.append((L.z_statistic_uniform(c, T, Ds[i]) - m["E_Z"]) / m["SD_Z"])
        mf = L.null_moments_uniform(T, D)
        cf, _ = L.collision_count_from_keys(ranks[n])
        final_z[r] = (L.z_statistic_uniform(cf, T, D) - mf["E_Z"]) / mf["SD_Z"]
        anytime_z[r] = max(zs)
        del ranks
    return {
        "R": R,
        "final_values": [float(v) for v in final_z],
        "anytime_values": [float(v) for v in anytime_z],
        "final_p99": float(np.percentile(final_z, 99)),
        "anytime_p99": float(np.percentile(anytime_z, 99)),
        "generator": "S0 CLASSICAL-R (anytime-uniform by exact certificate, E1 gate G1)",
    }


def run_a2(*, T: int = MATCHED_T, null_R: int = NULL_R_MATCHED, quick: bool = False) -> dict:
    import e1_candidates as C1
    import e1_montecarlo as mc
    import e1_verdicts as V
    from e1_exact_dp import run_exact_dp

    cells = MATCHED_CELLS[:1] if quick else MATCHED_CELLS
    keys = MATCHED_KEYS[:3] if quick else MATCHED_KEYS
    rows = []
    for (n, k) in cells:
        t0 = time.perf_counter()
        D = comb(n, k)
        logger.info(f"A2 cell n={n} k={k} D={D} -- null bands (R={null_R})")
        rngn = np.random.Generator(np.random.PCG64(MATCHED_SEED + 991))
        fin_null = mc.null_band(n, k, T, 200 if not quick else 20, rngn)
        pre_null = V.all_prefix_null_band(n, k, T, 100 if not quick else 10, MATCHED_SEED + 5)
        l1_null = _l1_null_matched(n, k, T, null_R, MATCHED_SEED + 3121)
        logger.info(f"A2 cell n={n} k={k} nulls done in {time.perf_counter()-t0:.1f}s")
        for key in keys:
            t1 = time.perf_counter()
            rep = run_exact_dp(C1.build(key, n=n, k=k), n, k)
            truth = "UNIFORM" if rep.i_first_fail is None else "NOT_UNIFORM"

            rng = np.random.Generator(np.random.PCG64(MATCHED_SEED))
            reservoirs = _simulate_matched(key, n, k, T, rng)
            fo = mc.first_order_stats(reservoirs, n, k)
            pr = mc.pair_stats(reservoirs, n, k)
            del reservoirs
            gc.collect()

            counts = V.simulate_prefix_counts(key, n, k, T,
                                              np.random.Generator(np.random.PCG64(MATCHED_SEED + 77)))
            ap = V.all_prefix_statistic(counts, n, k, T)
            del counts

            ranks = _simulate_matched_prefix_ranks(
                key, n, k, T, np.random.Generator(np.random.PCG64(MATCHED_SEED + 1234)))
            per_prefix = []
            for i in range(k, n + 1):
                Di = comb(i, k)
                if Di < 2:
                    continue
                res_i = L.run_tester(ranks[i], T, Di)
                res_i["prefix_i"] = i
                per_prefix.append(res_i)
            fin = L.run_tester(ranks[n], T, D)
            del ranks
            gc.collect()

            anytime_z = max(r["Z_normalised"] for r in per_prefix)

            f_fo, p_fo = V.mc_flag(fo["max_dev"], fin_null["first_order"]["values"], ALPHA)
            f_pr, p_pr = V.mc_flag(pr["max_abs_z"], fin_null["pair_z"]["values"], ALPHA)
            f_ap, p_ap = V.mc_flag(ap["max_z"], pre_null["values"], ALPHA)
            p_l1 = L.mc_pvalue(fin["Z_normalised"], l1_null["final_values"])
            p_l1a = L.mc_pvalue(anytime_z, l1_null["anytime_values"])
            thr_asym = L.asymptotic_threshold(T, D, ALPHA)

            rows.append({
                "n": n, "k": k, "T": T, "D": D, "candidate": key,
                "truth_exact_dp": truth,
                "i_first_fail": rep.i_first_fail,
                "mc_final_first_order": {
                    "verdict": "NOT_UNIFORM" if f_fo else "UNIFORM",
                    "max_dev": fo["max_dev"], "mc_p": p_fo},
                "mc_final_pairwise": {
                    "verdict": "NOT_UNIFORM" if f_pr else "UNIFORM",
                    "max_abs_z": pr["max_abs_z"], "mc_p": p_pr,
                    "frac_zero_pairs": pr["frac_zero_pairs"],
                    "argmax_pair": pr["argmax_pair"]},
                "mc_all_prefix": {
                    "verdict": "NOT_UNIFORM" if f_ap else "UNIFORM",
                    "max_z": ap["max_z"], "mc_p": p_ap},
                "l1_identity_final": {
                    "verdict": "NOT_UNIFORM" if p_l1 <= ALPHA else "UNIFORM",
                    "Z": fin["Z"], "Z_normalised": fin["Z_normalised"],
                    "collisions_C": fin["collisions_C"],
                    "n_distinct": fin["n_distinct"],
                    "mc_p": p_l1,
                    "asymptotic_threshold_Z": thr_asym,
                    "verdict_asymptotic": "NOT_UNIFORM" if fin["Z"] > thr_asym else "UNIFORM"},
                "l1_identity_anytime": {
                    "verdict": "NOT_UNIFORM" if p_l1a <= ALPHA else "UNIFORM",
                    "max_Z_normalised": float(anytime_z),
                    "argmax_prefix": int(max(per_prefix, key=lambda r: r["Z_normalised"])["prefix_i"]),
                    "mc_p": p_l1a,
                    "per_prefix": per_prefix},
                "seconds": time.perf_counter() - t1,
            })
            logger.info(f"A2 {key}@n={n},k={k}: truth={truth} fo={rows[-1]['mc_final_first_order']['verdict']} "
                        f"pw={rows[-1]['mc_final_pairwise']['verdict']} ap={rows[-1]['mc_all_prefix']['verdict']} "
                        f"l1={rows[-1]['l1_identity_final']['verdict']} (C={fin['collisions_C']}, "
                        f"Znorm={fin['Z_normalised']:.3g}) [{time.perf_counter()-t1:.1f}s]")
    return {"rows": rows, "null_matched_l1_p99": None}


def confusion(rows, method_key: str) -> dict:
    tp = fp = tn = fn = 0
    missed, falsely = [], []
    for r in rows:
        truth = r["truth_exact_dp"]
        pred = r[method_key]["verdict"]
        cell = f"{r['candidate']}@n={r['n']},k={r['k']}"
        if truth == "NOT_UNIFORM" and pred == "NOT_UNIFORM":
            tp += 1
        elif truth == "NOT_UNIFORM":
            fn += 1
            missed.append(cell)
        elif pred == "NOT_UNIFORM":
            fp += 1
            falsely.append(cell)
        else:
            tn += 1
    tot = tp + fp + tn + fn
    return {
        "method": method_key, "n_cells": tot,
        "true_positive": tp, "false_negative": fn,
        "false_positive": fp, "true_negative": tn,
        "accuracy": (tp + tn) / tot if tot else None,
        "recall_on_broken_samplers": tp / (tp + fn) if (tp + fn) else None,
        "false_alarm_rate_on_correct_samplers": fp / (tn + fp) if (tn + fp) else None,
        "specificity_on_correct_samplers": tn / (tn + fp) if (tn + fp) else None,
        "missed_cells": missed, "false_alarm_cells": falsely,
    }


# =======================================================================================
# A3 -- panel 2: 72 replicated runs at n=1000, k=10, T=1e6
# =======================================================================================


def sampler_geometry(n: int, k: int) -> dict:
    """Exact support size, ||p||_2^2 and TV(p, uniform) for every panel-2 sampler.

    Every blind-spot sampler here is UNIFORM ON ITS SUPPORT, so ||p||_2^2 = 1/|support|
    and TV = 1 - |support| / C(n,k) exactly.
      S5e BLOCK-LOCK : a classical 1-reservoir over the n/k consecutive blocks of k items
                       -> support = n/k blocks.
      S6sys SYSTEMATIC: random start in 1..n/k -> support = n/k.
      S5g PAIR-LOCK  : a classical (k/2)-reservoir over the n/2 consecutive pairs
                       -> support = C(n/2, k/2).
      S5-CIRC        : circular window of length k anchored anywhere -> support = n.
    """
    D = comb(n, k)
    out = {}
    for key in ["S0", "S1h", "S2", "S2f", "S2v", "S3"]:
        out[key] = {"support": D, "uniform_on_support": True, "family": "correct"}
    out["S5e"] = {"support": n // k, "uniform_on_support": True, "family": "blind_spot"}
    out["S6sys"] = {"support": n // k, "uniform_on_support": True, "family": "blind_spot"}
    out["S5g"] = {"support": comb(n // 2, k // 2), "uniform_on_support": True,
                  "family": "blind_spot"}
    out["S5-CIRC"] = {"support": n, "uniform_on_support": True, "family": "blind_spot"}
    for key, v in out.items():
        s = v["support"]
        v["D_total"] = D
        v["log10_support"] = log10(float(s))
        v["l2_norm_squared"] = 1.0 / s
        v["exact_TV_to_uniform"] = 1.0 - s / D
        v["closed_form"] = (
            "uniform" if v["family"] == "correct"
            else f"support={s}, ||p||_2^2 = 1/{s}, TV = 1 - {s}/C({n},{k})")
    return out


def _simulate_big(key: str, n: int, k: int, T: int, seed: int) -> np.ndarray:
    """Final reservoirs for one panel-2 sampler, from the VENDORED E2 samplers."""
    import e2_candidates as C2
    import e2_prngs as P

    if key == "S5-CIRC":
        import e1_montecarlo as mc
        rng = np.random.Generator(np.random.PCG64(seed))
        return mc.simulate_circwindow(n, k, T, rng)
    gen = P.make_prng("PCG64", seed)
    return C2.get_candidate(key).vector(gen, n, k, T)


def run_a3(*, T: int = 1_000_000, seeds=None, samplers=None, null_R: int = 50) -> dict:
    seeds = BIG_SEEDS if seeds is None else seeds
    samplers = (BIG_SAMPLERS + ["S5-CIRC"]) if samplers is None else samplers
    n, k = BIG_N, BIG_K
    D = comb(n, k)
    geo = sampler_geometry(n, k)
    mom = L.null_moments_uniform(T, D)

    # own simulated null of the L1 statistic from an INDEPENDENT exact uniform oracle
    logger.info(f"A3 null: {null_R} replicates of an exact uniform k-subset oracle at T={T}")
    import e1_montecarlo as mc
    nullC, nullZ = [], []
    t0 = time.perf_counter()
    for r in range(null_R):
        rng = np.random.Generator(np.random.PCG64(900_000 + r))
        rows = mc._uniform_subsets_reject(n, k, T, rng)
        c, _ = L.collision_count_from_keys(L.subset_bytes_key(rows))
        del rows
        gc.collect()
        nullC.append(int(c))
        nullZ.append(L.z_statistic_uniform(c, T, D))
    logger.info(f"A3 null done in {time.perf_counter()-t0:.1f}s: C values {sorted(set(nullC))}")

    rows_out = []
    for key in samplers:
        for seed in seeds:
            t1 = time.perf_counter()
            res = _simulate_big(key, n, k, T, 10_000 + 991 * seed)
            keys_arr = L.subset_bytes_key(res)
            del res
            gc.collect()
            r = L.run_tester(keys_arr, T, D)
            del keys_arr
            gc.collect()
            g = geo[key]
            predicted_C = (T * (T - 1) / 2.0) * g["l2_norm_squared"]
            p_mc = L.mc_pvalue(r["Z"], nullZ)
            thr = L.asymptotic_threshold(T, D, ALPHA)
            rows_out.append({
                "candidate": key, "n": n, "k": k, "T": T, "seed": seed,
                "truth": "NOT_UNIFORM" if g["family"] == "blind_spot" else "UNIFORM",
                "family": g["family"],
                "support": g["support"], "l2_norm_squared": g["l2_norm_squared"],
                "exact_TV_to_uniform": g["exact_TV_to_uniform"],
                "collisions_C_measured": r["collisions_C"],
                "collisions_C_predicted": float(predicted_C),
                "collisions_ratio_measured_over_predicted":
                    (r["collisions_C"] / predicted_C) if predicted_C > 0 else None,
                "n_distinct_measured": r["n_distinct"],
                "Z": r["Z"], "Z_normalised": r["Z_normalised"],
                "poisson_log10_p": r["poisson_log10_p"],
                "l1_mc_p": p_mc,
                "l1_verdict": "NOT_UNIFORM" if p_mc <= ALPHA else "UNIFORM",
                "l1_verdict_asymptotic": "NOT_UNIFORM" if r["Z"] > thr else "UNIFORM",
                "asymptotic_threshold_Z": thr,
                "seconds": time.perf_counter() - t1,
            })
            logger.info(f"A3 {key} seed={seed}: C={r['collisions_C']} "
                        f"(pred {predicted_C:.3g}) distinct={r['n_distinct']} "
                        f"Z={r['Z']:.4g} p={p_mc:.3g} -> {rows_out[-1]['l1_verdict']} "
                        f"[{rows_out[-1]['seconds']:.1f}s]")
    return {
        "rows": rows_out,
        "geometry": geo,
        "null": {"R": null_R, "C_values": nullC, "Z_values": nullZ,
                 "C_analytic_mean": mom["E_C"], "Z_analytic_mean": mom["E_Z"],
                 "Z_analytic_sd": mom["SD_Z"],
                 "oracle": "exact uniform k-subset by rejection (E1 montecarlo._uniform_subsets_reject)"},
        "D": D,
        "log10_D": log10(float(D)),
    }


def worst_case_T(D: float, eps: float, c: float) -> float:
    """T = c * sqrt(D) / eps^2 -- the worst-case sample complexity of the L1 tester."""
    return c * sqrt(D) / eps**2


# =======================================================================================
# A4 -- sample-complexity / state-complexity separation table
# =======================================================================================


def build_separation_table(a2_conf: dict, a3_rows: list, *, c_measured: float,
                           n: int = BIG_N, k: int = BIG_K) -> list[dict]:
    D = float(comb(n, k))
    tv_blind = [r["exact_TV_to_uniform"] for r in a3_rows if r["family"] == "blind_spot"]
    tv_max = max(tv_blind) if tv_blind else 0.986
    # the matched panel's own worst blind spot is S5-CIRC at (n=14,k=4): TV = 1 - n/C(n,k)
    tv_matched = 1.0 - 14.0 / comb(14, 4)

    def big_stats(vk: str) -> tuple[float, float]:
        broken = [r for r in a3_rows if r["family"] == "blind_spot"]
        correct = [r for r in a3_rows if r["family"] == "correct"]
        rec = sum(1 for r in broken if r[vk] == "NOT_UNIFORM") / len(broken) if broken else None
        fa = sum(1 for r in correct if r[vk] == "NOT_UNIFORM") / len(correct) if correct else None
        return rec, fa

    l1_rec, l1_fa = big_stats("l1_verdict")
    rows = [
        {
            "detector": "first-order max-deviation (the field-standard test)",
            "panel": "matched 18 cells (n<=14, k in {2,4}, T=1e5)",
            "state_per_trial": "n counters (one per item)",
            "total_memory_order": "O(n)",
            "streaming_single_pass": True,
            "memory_independent_of_T": True,
            "localises_the_defect": False,
            "detection_threshold_in_trials":
                "O(n / eps_first_order^2) for a FIRST-ORDER-far alternative; "
                "NEVER, at any T, for an alternative whose first-order marginals are exact",
            "measured_accuracy": a2_conf["mc_final_first_order"]["accuracy"],
            "measured_recall_on_broken": a2_conf["mc_final_first_order"]["recall_on_broken_samplers"],
            "measured_false_alarm_rate": a2_conf["mc_final_first_order"]["false_alarm_rate_on_correct_samplers"],
            "T_required_worst_case_eps_0.1": "not applicable (no guarantee against joint defects)",
            "T_required_worst_case_eps_measured_TV": "not applicable (no guarantee against joint defects)",
        },
        {
            "detector": "pairwise co-inclusion (calibrated against its own simulated null)",
            "panel": "matched 18 cells (n<=14, k in {2,4}, T=1e5)",
            "state_per_trial": "C(n,2) pair counters",
            "total_memory_order": "O(n^2)",
            "streaming_single_pass": True,
            "memory_independent_of_T": True,
            "localises_the_defect": True,
            "detection_threshold_in_trials": "O(n^2 / eps_pair^2)",
            "measured_accuracy": a2_conf["mc_final_pairwise"]["accuracy"],
            "measured_recall_on_broken": a2_conf["mc_final_pairwise"]["recall_on_broken_samplers"],
            "measured_false_alarm_rate": a2_conf["mc_final_pairwise"]["false_alarm_rate_on_correct_samplers"],
            "T_required_worst_case_eps_0.1": "not a full L1 test; bounds only the second-order marginals",
            "T_required_worst_case_eps_measured_TV": "not a full L1 test; bounds only the second-order marginals",
        },
        {
            "detector": "L1 identity / collision tester (ADK 2015 / DKN 2015)",
            "panel": "matched 18 cells (n<=14, k in {2,4}, T=1e5)",
            "state_per_trial": "the multiset of observed k-subsets",
            "total_memory_order": "O(min(T, D)) distinct keys",
            "streaming_single_pass": True,
            "memory_independent_of_T": False,
            "localises_the_defect": False,
            "detection_threshold_in_trials": "Theta(sqrt(D)/eps^2), D = C(n,k)",
            "measured_accuracy": a2_conf["l1_identity_final"]["accuracy"],
            "measured_recall_on_broken": a2_conf["l1_identity_final"]["recall_on_broken_samplers"],
            "measured_false_alarm_rate": a2_conf["l1_identity_final"]["false_alarm_rate_on_correct_samplers"],
            "T_required_worst_case_eps_0.1": worst_case_T(float(comb(14, 4)), 0.1, c_measured),
            "T_required_worst_case_eps_measured_TV": worst_case_T(float(comb(14, 4)), tv_matched, c_measured),
            "eps_used_for_measured_TV": tv_matched,
        },
        {
            "detector": "L1 identity / collision tester (ADK 2015 / DKN 2015)",
            "panel": f"replicated panel (n={n}, k={k}, T=1e6)",
            "state_per_trial": "the multiset of observed k-subsets",
            "total_memory_order": "O(min(T, D)) distinct keys = O(T) here",
            "streaming_single_pass": True,
            "memory_independent_of_T": False,
            "localises_the_defect": False,
            "detection_threshold_in_trials": f"Theta(sqrt(C({n},{k}))/eps^2)",
            "measured_accuracy": None,
            "measured_recall_on_broken": l1_rec,
            "measured_false_alarm_rate": l1_fa,
            "T_required_worst_case_eps_0.1": worst_case_T(D, 0.1, c_measured),
            "T_required_worst_case_eps_measured_TV": worst_case_T(D, tv_max, c_measured),
            "eps_used_for_measured_TV": tv_max,
        },
    ]
    for r in rows:
        for fld in ("T_required_worst_case_eps_0.1", "T_required_worst_case_eps_measured_TV"):
            if isinstance(r[fld], float):
                r[fld + "_log10"] = log10(float(r[fld])) if r[fld] > 0 else None
    return rows


def run_a1(*, quick: bool = False) -> dict:
    """A1 -- validation of the tester before the panels."""
    Ds = [1024, 4096] if quick else [1024, 4096, 16384, 65536]
    epss = [0.05, 0.2] if quick else [0.1, 0.2, 0.35, 0.5]
    reps = 60 if quick else 300
    logger.info(f"A1 power curve over D={Ds} eps={epss} reps={reps}")
    power = a1_power_curve(Ds=Ds, epss=epss, reps=reps, seed=4242)
    nulls = {}
    for (n, k) in MATCHED_CELLS:
        D = comb(n, k)
        nulls[f"matched_n{n}_k{k}"] = a1_null_distribution(
            D, 2_000 if quick else MATCHED_T, 50 if quick else 500, 777 + n * 100 + k)
    # sqrt(D) scaling check: regress log T_90 on log D at fixed eps
    scaling = {}
    for eps in epss:
        pts = [(D, v["T_90"]) for key, v in power["fits"].items()
               for D in [int(key.split(",")[0].split("=")[1])]
               if abs(float(key.split("eps=")[1]) - eps) < 1e-12]
        if len(pts) >= 2:
            x = np.log(np.array([p[0] for p in pts], dtype=float))
            y = np.log(np.array([p[1] for p in pts], dtype=float))
            slope = float(np.polyfit(x, y, 1)[0])
            scaling[f"eps={eps}"] = {"n_points": len(pts), "fitted_exponent_of_D": slope,
                                     "theory": 0.5}
    eps_scaling = {}
    for D in Ds:
        pts = [(float(key.split("eps=")[1]), v["T_90"]) for key, v in power["fits"].items()
               if key.startswith(f"D={D},")]
        if len(pts) >= 2:
            x = np.log(np.array([p[0] for p in pts]))
            y = np.log(np.array([p[1] for p in pts]))
            eps_scaling[f"D={D}"] = {"n_points": len(pts),
                                     "fitted_exponent_of_eps": float(np.polyfit(x, y, 1)[0]),
                                     "theory": -2.0}
    return {"power": power, "nulls": nulls, "D_scaling": scaling, "eps_scaling": eps_scaling}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["a1", "a2", "a3"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--nullR", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.stage == "a1":
        out = run_a1(quick=args.quick)
    elif args.stage == "a2":
        out = run_a2(T=args.T or (3_000 if args.quick else MATCHED_T),
                     null_R=args.nullR or (5 if args.quick else NULL_R_MATCHED),
                     quick=args.quick)
        out["confusion"] = {m: confusion(out["rows"], m) for m in
                            ["mc_final_first_order", "mc_final_pairwise", "mc_all_prefix",
                             "l1_identity_final", "l1_identity_anytime"]}
    else:
        out = run_a3(T=args.T or (20_000 if args.quick else 1_000_000),
                     seeds=list(range(args.seeds)) if args.seeds else (list(range(2)) if args.quick else None),
                     null_R=args.nullR or (5 if args.quick else 50))
    out["wall_seconds"] = time.perf_counter() - t0
    dest = Path(args.out) if args.out else RES / f"stage_{args.stage}.json"
    dest.write_text(json.dumps(out, indent=1, default=float))
    logger.info(f"stage {args.stage} -> {dest} in {out['wall_seconds']:.1f}s")
