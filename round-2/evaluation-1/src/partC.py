#!/usr/bin/env python3
"""PART C -- the validity checks a reviewer runs next.

C1  Re-derive the co-inclusion calibration from simulation and find the SMALLEST (n,k) at
    which the pre-registered NOMINAL chi-square rule (p < 1e-6) first false-alarms on
    Algorithm R itself.  That boundary is the justification for the declared deviation.
C2  Test -- not assert -- the mechanism behind the SIGN REVERSAL (three broken samplers
    scoring BETTER than Algorithm R on the first-order max-deviation test), by simulating
    max-of-n versus max-of-(effective bins) binomial deviations with each sampler's
    effective bin count derived from its actual support.
C3  Reference hygiene (executed separately with web tools; loaded here from scratch/refs.json).
"""

from __future__ import annotations

import json
import math
import sys
import time
from math import comb, log, sqrt
from pathlib import Path

import numpy as np
from loguru import logger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendored"))

RES = HERE / "results"
RES.mkdir(exist_ok=True)

NOMINAL_ALPHA = 1e-6
CALIBRATED_Z = 5.0

CORRECT = ["S0", "S1h"]
BROKEN = ["S5e", "S5g", "S6sys"]


# =======================================================================================
# C1 -- co-inclusion calibration boundary
# =======================================================================================


def _chunk_for(k: int) -> int:
    """Trial-chunk size that keeps the dense pair-index buffer under ~2e7 int64 entries.

    PairAccumulator materialises chunk * C(k,2) linearised pair indices at once, so a
    fixed chunk of 50k rows costs 1e9 entries (8 GB) at k=200. Scale it with k instead.
    """
    return int(max(2_000, min(50_000, 2e7 // max(1, comb(k, 2)))))


def c1_nominal_sweep(*, T: int, grid, seed: int = 424_242) -> list[dict]:
    """Run Algorithm R (S0) on every grid cell and record the NOMINAL chi-square p.

    A nominal p < 1e-6 on S0 is a FALSE ALARM by construction: S0 is provably uniform.
    """
    import e2_montecarlo as M

    rows = []
    for (n, k) in grid:
        t0 = time.perf_counter()
        try:
            r = M.run_mc("S0", n=n, k=k, T=T, seed=seed + 17 * n + k,
                         chunk=_chunk_for(k))
        except (ValueError, MemoryError) as exc:
            rows.append({"n": n, "k": k, "T": T, "status": f"SKIPPED: {exc}"})
            logger.warning(f"C1 skip n={n} k={k}: {exc}")
            continue
        rows.append({
            "n": n, "k": k, "T": T, "status": "RUN",
            "k_over_n": k / n,
            "coinc_chi2": r["coinc_chi2"], "coinc_df": r["coinc_df"],
            "coinc_chi2_over_df": r["coinc_chi2_over_df"],
            "coinc_p_nominal": r["coinc_p"],
            "coinc_Zmax": r["coinc_Zmax"],
            "nominal_rule_false_alarm": bool(r["coinc_p"] < NOMINAL_ALPHA),
            "expected_per_pair": r["coinc_expected_per_pair"],
            "wall_s": time.perf_counter() - t0,
        })
        logger.info(f"C1 nominal n={n} k={k} k/n={k/n:.4f} p={r['coinc_p']:.3g} "
                    f"chi2/df={r['coinc_chi2_over_df']:.5f} "
                    f"FA={rows[-1]['nominal_rule_false_alarm']} "
                    f"[{rows[-1]['wall_s']:.1f}s]")
    return rows


def c1_calibrated_cells(*, T: int, cells, null_B: int, null_B_coinc: int,
                        seed: int = 424_242) -> list[dict]:
    """Own-simulated null per cell + every correct and broken sampler, calibrated z."""
    import e2_montecarlo as M
    import e2_nullband as NB
    import e2_prngs as P

    out = []
    for (n, k) in cells:
        t0 = time.perf_counter()
        gen = P.make_prng("PCG64", seed + 991 * n + k)
        band = NB.simulate_null(gen, n=n, k=k, T=T, B=null_B, chunk=_chunk_for(k),
                                B_coinc=null_B_coinc)
        logger.info(f"C1 band n={n} k={k} done in {time.perf_counter()-t0:.1f}s "
                    f"(chi2 null mean {band.get('coinc_chi2_null_mean')})")
        for cid in CORRECT + BROKEN:
            try:
                r = M.run_mc(cid, n=n, k=k, T=T, seed=seed + 7 * n + k,
                             chunk=_chunk_for(k))
            except ValueError as exc:
                out.append({"n": n, "k": k, "candidate": cid, "status": f"SKIPPED: {exc}"})
                continue
            rr = M.mc_row_with_band(r, band)
            out.append({
                "n": n, "k": k, "T": T, "candidate": cid, "status": "RUN",
                "truth": "NOT_UNIFORM" if cid in BROKEN else "UNIFORM",
                "coinc_p_nominal": r["coinc_p"],
                "nominal_verdict": "NOT_UNIFORM" if r["coinc_p"] < NOMINAL_ALPHA else "UNIFORM",
                "coinc_chi2_z_vs_simulated_null": rr.get("coinc_chi2_z"),
                "calibrated_verdict":
                    "NOT_UNIFORM" if (rr.get("coinc_chi2_z") or 0) >= CALIBRATED_Z else "UNIFORM",
                "maxdev_raw": r["maxdev_raw"],
                "maxdev_z_band": rr.get("z_band"),
                "null_chi2_mean": band.get("coinc_chi2_null_mean"),
                "null_chi2_sd": band.get("coinc_chi2_null_sd"),
                "null_B": null_B, "null_B_coinc": null_B_coinc,
            })
            logger.info(f"C1 cal n={n} k={k} {cid}: nominal p={r['coinc_p']:.3g} "
                        f"z={out[-1]['coinc_chi2_z_vs_simulated_null']} "
                        f"-> {out[-1]['calibrated_verdict']}")
    return out


def c1_summary(nominal: list[dict], calibrated: list[dict]) -> dict:
    fa = [r for r in nominal if r.get("nominal_rule_false_alarm")]
    ok = [r for r in nominal if r.get("status") == "RUN" and not r["nominal_rule_false_alarm"]]
    smallest_by_ratio = min(fa, key=lambda r: r["k_over_n"]) if fa else None
    smallest_by_k = min(fa, key=lambda r: (r["k"], r["k_over_n"])) if fa else None
    run = [c for c in calibrated if c.get("status") == "RUN"
           and c["coinc_chi2_z_vs_simulated_null"] is not None]
    corr = [c for c in run if c["truth"] == "UNIFORM"]
    brok = [c for c in run if c["truth"] == "NOT_UNIFORM"]
    max_corr = max((c["coinc_chi2_z_vs_simulated_null"] for c in corr), default=None)
    min_brok = min((c["coinc_chi2_z_vs_simulated_null"] for c in brok), default=None)
    nominal_fa_on_correct = [f"{c['candidate']}@n{c['n']}k{c['k']} (p={c['coinc_p_nominal']:.3g})"
                             for c in corr if c["nominal_verdict"] == "NOT_UNIFORM"]
    return {
        "n_cells_run": sum(1 for r in nominal if r.get("status") == "RUN"),
        "n_cells_nominal_false_alarm": len(fa),
        "false_alarm_cells": [{"n": r["n"], "k": r["k"], "k_over_n": r["k_over_n"],
                               "p": r["coinc_p_nominal"]} for r in fa],
        "clean_cells": [{"n": r["n"], "k": r["k"], "k_over_n": r["k_over_n"],
                         "p": r["coinc_p_nominal"]} for r in ok],
        "smallest_false_alarm_by_k_over_n": smallest_by_ratio,
        "smallest_false_alarm_by_k": smallest_by_k,
        "calibrated_max_z_among_correct": max_corr,
        "calibrated_min_z_among_broken": min_brok,
        "calibrated_gap": (min_brok - max_corr) if (max_corr is not None and min_brok is not None) else None,
        "calibrated_separation_clean": (
            bool(max_corr is not None and min_brok is not None
                 and max_corr < CALIBRATED_Z <= min_brok)),
        "nominal_rule_false_alarms_on_provably_correct_samplers": nominal_fa_on_correct,
        "threshold_calibrated_z": CALIBRATED_Z,
        "threshold_nominal_p": NOMINAL_ALPHA,
    }


# =======================================================================================
# C2 -- sign-reversal mechanism
# =======================================================================================


def effective_bin_count(cid: str, n: int, k: int) -> dict:
    """The number of INDEPENDENT first-order bins each sampler's support actually has.

    The max-deviation statistic maximises |count_x/T - k/n| over the n items, but items
    that are locked together by the support share one and the same count, so the maximum
    is over the number of DISTINCT count-carrying groups, not over n.
      S5e BLOCK-LOCK   : items are locked into n/k blocks of k    -> n/k groups
      S6sys SYSTEMATIC : items are locked into n/k residue classes -> n/k groups
      S5g PAIR-LOCK    : items are locked into n/2 pairs           -> n/2 groups
      S5-CIRC          : every item has its own count              -> n groups
      correct samplers : n groups
    """
    table = {
        "S5e": (n // k, f"n/k = {n}/{k}", "consecutive blocks of k share one count"),
        "S6sys": (n // k, f"n/k = {n}/{k}", "residue classes mod n/k share one count"),
        "S5g": (n // 2, f"n/2 = {n}/2", "consecutive PAIRS share one count"),
        "S5-CIRC": (n, "n", "circular windows give every item its own count"),
    }
    m, formula, why = table.get(cid, (n, "n", "no locking"))
    return {"candidate": cid, "effective_bins": int(m), "formula": formula, "why": why}


def simulate_max_of_bins(*, m_bins: int, group_size: int, T: int, p: float, R: int,
                         seed: int, n_replicates_per_mean: int = 1) -> dict:
    """Empirical distribution of max_j |freq_j - p| when the n items carry only m_bins
    DISTINCT counts.

    Careful with the variance.  Locking items together does NOT reduce the per-item
    variance: for BLOCK-LOCK, item x sits in block b and is in the reservoir exactly when
    block b is selected, so count_x = N_b with N_b ~ Binomial(T, k/n) marginally and
    freq_x = N_b/T has the SAME sd sqrt(p(1-p)/T) as under a correct sampler.  What the
    locking changes is only how many DISTINCT values the maximum ranges over: m_bins
    instead of n.  The whole predicted effect is therefore the extreme-value shrinkage
    E[max of m] / E[max of n] ~ sqrt(2 ln m)/sqrt(2 ln n).

    `group_size` is carried for documentation (n / m_bins) and asserted consistent; it
    does not enter the variance.
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    out = np.empty(R)
    for r in range(R):
        vals = np.empty(n_replicates_per_mean)
        for j in range(n_replicates_per_mean):
            cnt = rng.binomial(T, p, size=m_bins)
            vals[j] = float(np.max(np.abs(cnt / T - p)))
        out[r] = vals.mean()
    sigma = sqrt(p * (1 - p) / T)
    return {"m_bins": int(m_bins), "group_size": int(group_size), "R": int(R),
            "n_replicates_per_mean": int(n_replicates_per_mean),
            "mean": float(out.mean()), "sd": float(out.std(ddof=1)),
            "p025": float(np.percentile(out, 2.5)), "p50": float(np.percentile(out, 50)),
            "p975": float(np.percentile(out, 97.5)),
            "analytic_sigma_per_bin": sigma,
            "analytic_mean_gumbel": sigma * sqrt(2 * log(max(m_bins, 2))),
            "values": [float(v) for v in out[:2000]]}


def c2_sign_reversal(e2_meta: dict, *, n: int = 1000, k: int = 10, T: int = 1_000_000,
                     R: int = 2000) -> dict:
    p = k / n
    rows = e2_meta["replication_calibration"]["rows"]
    obs = {}
    for cid in ["S0", "S1h", "S2", "S3", "S5e", "S5g", "S6sys"]:
        vals = [r["maxdev_raw"] for r in rows if r["candidate"] == cid]
        zs = [r["z_band"] for r in rows if r["candidate"] == cid]
        if vals:
            obs[cid] = {"n_replicates": len(vals), "maxdev_mean": float(np.mean(vals)),
                        "maxdev_sd": float(np.std(vals, ddof=1)),
                        "z_band_mean": float(np.mean(zs)),
                        "headline_single_run": None}
    hl = e2_meta["summary_tables"]["T1_uniformity_deliverable_n1000_k10_T1e6"]
    for row in hl:
        if row["candidate"] in obs:
            obs[row["candidate"]]["headline_single_run"] = {
                "maxdev_raw": row["maxdev_raw"], "z_band": row["z_band"]}

    n_rep = min(len(v["n_replicates"] and [1]) for v in [obs["S0"]]) if False else obs["S0"]["n_replicates"]
    sims = {}
    sims["max_of_n"] = simulate_max_of_bins(m_bins=n, group_size=1, T=T, p=p, R=R,
                                            seed=31337, n_replicates_per_mean=n_rep)
    designs = {}
    for cid in ["S5e", "S5g", "S6sys", "S5-CIRC"]:
        eb = effective_bin_count(cid, n, k)
        gsize = n // eb["effective_bins"]
        sims[f"max_of_{cid}"] = simulate_max_of_bins(
            m_bins=eb["effective_bins"], group_size=gsize, T=T, p=p, R=R,
            seed=31337 + 7 * eb["effective_bins"], n_replicates_per_mean=n_rep)
        designs[cid] = eb

    base = sims["max_of_n"]
    base_vals = np.asarray(base["values"])
    out_rows = []
    for cid, eb in designs.items():
        s_ = sims[f"max_of_{cid}"]
        vals = np.asarray(s_["values"])
        m_ = min(len(vals), len(base_vals))
        diffs = vals[:m_] - base_vals[:m_]          # both are means over n_rep replicates
        pred_shift = float(diffs.mean())
        lo_i, hi_i = (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))
        o = obs.get(cid)
        obs_shift = (o["maxdev_mean"] - obs["S0"]["maxdev_mean"]) if o else None
        within = (lo_i <= obs_shift <= hi_i) if obs_shift is not None else None
        out_rows.append({
            "candidate": cid,
            "effective_bins": eb["effective_bins"],
            "effective_bins_formula": eb["formula"],
            "why": eb["why"],
            "sim_mean_max_dev": s_["mean"], "sim_sd_max_dev": s_["sd"],
            "baseline_sim_mean_max_dev": base["mean"], "baseline_sim_sd": base["sd"],
            "sim_analytic_gumbel_mean": s_["analytic_mean_gumbel"],
            "predicted_shift": pred_shift,
            "predicted_shift_95_interval": [lo_i, hi_i],
            "n_replicates_averaged": n_rep,
            "observed_mean_max_dev": o["maxdev_mean"] if o else None,
            "observed_n_replicates": o["n_replicates"] if o else None,
            "observed_shift_vs_S0": obs_shift,
            "observed_shift_inside_predicted_interval": within,
            "analytic_gumbel_ratio_sqrt2lnm_over_sqrt2lnn":
                sqrt(2 * log(eb["effective_bins"])) / sqrt(2 * log(n)),
            "sim_mean_ratio_to_baseline": s_["mean"] / base["mean"],
            "observed_ratio_to_S0": (o["maxdev_mean"] / obs["S0"]["maxdev_mean"]) if o else None,
        })

    scored_better = [cid for cid in ["S5e", "S5g", "S6sys"]
                     if obs.get(cid) and obs[cid]["headline_single_run"]
                     and obs[cid]["headline_single_run"]["maxdev_raw"]
                     < obs["S0"]["headline_single_run"]["maxdev_raw"]]
    tested = [r for r in out_rows if r["candidate"] in ("S5e", "S5g", "S6sys")]
    n_in = sum(1 for r in tested if r["observed_shift_inside_predicted_interval"])
    direction_ok = all(
        (r["predicted_shift"] < 0) == (r["observed_shift_vs_S0"] < 0)
        for r in tested if r["observed_shift_vs_S0"] is not None)
    if n_in == len(tested) and direction_ok:
        verdict = "MECHANISM_CONFIRMED"
    elif direction_ok and n_in >= 1:
        verdict = "MECHANISM_PARTIAL"
    else:
        verdict = "MECHANISM_REFUTED"
    return {
        "n": n, "k": k, "T": T, "R": R, "p_expected": p,
        "observed_from_artifact": obs,
        "simulations": sims,
        "rows": out_rows,
        "blind_spot_samplers_scoring_better_than_S0_on_the_headline_run": scored_better,
        "n_scoring_better": len(scored_better),
        "verdict": verdict,
        "residual_note": (
            "PAIR-LOCK (S5g) locks items into n/2 = 500 pairs, not n/k = 100 blocks, so "
            "its effective bin count is 5x larger than BLOCK-LOCK's and its predicted "
            "shrinkage is correspondingly smaller. Assuming n/k for all three would "
            "mis-predict S5g."),
        "analytic_shrinkage_sqrt2ln100_over_sqrt2ln1000":
            sqrt(2 * log(100)) / sqrt(2 * log(1000)),
    }


if __name__ == "__main__":
    import argparse

    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(str(HERE / "logs" / "partC.log"), rotation="30 MB", level="DEBUG")

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["c1", "c2"])
    ap.add_argument("--T", type=int, default=200_000)
    ap.add_argument("--nullB", type=int, default=200)
    ap.add_argument("--nullBcoinc", type=int, default=48)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    if args.stage == "c1":
        GRID = [(n, k) for n in (200, 500, 1000, 2000, 5000)
                for k in (5, 10, 15, 20, 25, 30, 40, 50, 75, 100) if k <= n // 4]
        CAL_CELLS = [(1000, 10), (1000, 50), (2000, 100), (500, 50)]
        if args.quick:
            GRID, CAL_CELLS = GRID[:3], [(200, 50)]
        nominal = c1_nominal_sweep(T=args.T, grid=GRID)
        cal = c1_calibrated_cells(T=args.T, cells=CAL_CELLS,
                                  null_B=args.nullB if not args.quick else 8,
                                  null_B_coinc=args.nullBcoinc if not args.quick else 4)
        out = {"nominal_sweep": nominal, "calibrated_cells": cal,
               "summary": c1_summary(nominal, cal),
               "grid": [list(g) for g in GRID], "calibrated_cell_list": [list(c) for c in CAL_CELLS],
               "T": args.T}
    else:
        meta = json.loads((HERE / "scratch" / "E2_metadata.json").read_text())
        out = c2_sign_reversal(meta, R=200 if args.quick else 2000)
    out["wall_seconds"] = time.perf_counter() - t0
    (RES / f"stage_{args.stage}.json").write_text(json.dumps(out, indent=1, default=float))
    logger.info(f"stage {args.stage} written in {out['wall_seconds']:.1f}s")
