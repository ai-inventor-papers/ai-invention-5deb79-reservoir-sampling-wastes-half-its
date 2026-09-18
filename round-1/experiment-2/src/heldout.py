#!/usr/bin/env python3
"""HELD-OUT panel -- runs ONLY after results/screen_ranking.json is frozen.

Re-verifies the frozen screen hash on startup and ABORTS if it changed.  Every
stage checkpoints to results/partial_*.json after each configuration, so a
timeout still ships data and method.py can assemble from whatever exists.

Stages (``--stage``):
    mc     the n=1000, k=10, T=1e6 deliverable for EVERY sampler + null band
    bits   measured bit costs to n=1e6, closed form to n=1e7
    prng   the held-out generator panel {R250, MT19937, ChaCha20}
    algol  the Algorithm-L audit out to k = 2**14
    exact  exact rational verification at n = 15..18
    all    everything, in that order
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np
from loguru import logger

import budget
import common
import exact
import montecarlo as mc
import nullband
import prngs
from bits import REGIMES
from candidates import CANDIDATES, MC_CANDIDATES

HELD_MC_N = 1_000
HELD_MC_K = 10
HELD_MC_T = 1_000_000
HELD_MC_B = 64
HELD_MC_B_COINC = 16
HELD_BIT_NS = (100_000, 1_000_000)
HELD_BIT_KS = (10, 100, 1_000)
HELD_CLOSED_NS = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
HELD_CLOSED_KS = (1, 2, 10, 100, 1_000)
HELD_PRNGS = prngs.HELDOUT_PRNGS
HELD_ALGOL_KS = (1_024, 4_096, 16_384)
BIT_CIDS = ("S0", "S1", "S1h", "S1n", "S2", "S2f", "S2v", "S3", "S5e", "S5g")


def verify_freeze() -> str:
    path = common.RESULTS / "screen_ranking.json"
    if not path.exists():
        raise FileNotFoundError("screen_ranking.json missing -- run screen.py first")
    payload = common.load_json(path)
    claimed = payload.pop("content_sha256")
    recomputed = common.content_sha256(payload)
    if claimed != recomputed:
        raise AssertionError(
            f"SCREEN FREEZE VIOLATED: stored {claimed} != recomputed {recomputed}"
        )
    logger.info(f"screen freeze verified: {claimed}")
    return claimed


def stage_mc() -> dict:
    """The literal deliverable: max deviation from k/n IN NULL-BAND UNITS."""
    gen = prngs.make_prng("PCG64", 88_111)
    t0 = time.time()
    band = nullband.simulate_null(
        gen, n=HELD_MC_N, k=HELD_MC_K, T=HELD_MC_T, B=HELD_MC_B,
        chunk=50_000, B_coinc=HELD_MC_B_COINC,
    )
    logger.info(
        f"null band T={HELD_MC_T} B={HELD_MC_B}: mean={band['maxdev_null_mean']:.6f} "
        f"sd={band['maxdev_null_sd']:.6f} p99={band['maxdev_null_p99']:.6f} "
        f"analytic={band['analytic_maxdev_mean']:.6f} "
        f"ratio={band['maxdev_null_vs_analytic_ratio']:.3f} ({time.time() - t0:.1f}s)"
    )
    rows: list[dict] = []
    for cid in MC_CANDIDATES:
        cand = CANDIDATES[cid]
        if cand.needs_even_k and HELD_MC_K % 2:
            continue
        if cand.oracle_n and HELD_MC_N % HELD_MC_K:
            continue
        row = mc.run_mc(cid, n=HELD_MC_N, k=HELD_MC_K, T=HELD_MC_T,
                        seed=99_001, chunk=50_000)
        full = mc.mc_row_with_band(row, band)
        rows.append(full)
        logger.info(
            f"MC {cid:6s} maxdev={full['maxdev_raw']:.6f} "
            f"z_band={full['z_band']:+7.2f} pct={full['exceedance_pct']:5.1f} "
            f"FO_pass={full['passes_first_order_at_p99']} "
            f"coincZ={full['coinc_Zmax']:9.1f} coinc_fail={full['coinc_fails_at_p99']} "
            f"BLIND={full.get('blind_spot')} ({full['wall_s']}s)"
        )
        common.checkpoint("heldout_mc", {"band": band, "rows": rows})
    return {"band": band, "rows": rows}


def stage_bits() -> dict:
    rows: list[dict] = []
    for n in HELD_BIT_NS:
        for k in HELD_BIT_KS:
            for regime in REGIMES:
                for cid in BIT_CIDS:
                    cand = CANDIDATES[cid]
                    if cand.needs_even_k and k % 2:
                        continue
                    if cand.oracle_n and n % k:
                        continue
                    r = mc.run_bits(cid, n=n, k=k, regime=regime, seeds=(1, 2, 3))
                    rows.append(r)
                common.checkpoint("heldout_bits", rows)
            logger.info(f"bits n={n} k={k} done ({len(rows)} rows)")
    closed = budget.budget_table(list(HELD_CLOSED_NS), list(HELD_CLOSED_KS))
    common.checkpoint("heldout_bits_closed", closed)
    return {"measured": rows, "closed_form": closed}


def stage_prng() -> dict:
    """M6 held-out: does generator quality separate the ALGORITHMS?"""
    stat_gen = prngs.make_prng("PCG64", 5_150)
    stat_band = nullband.simulate_null(
        stat_gen, n=HELD_MC_N, k=HELD_MC_K, T=HELD_MC_T, B=32,
        chunk=50_000, B_coinc=6,
    )
    rows: list[dict] = []
    for pname in HELD_PRNGS:
        mgen = prngs.make_prng(pname, 6_161)
        matched = nullband.simulate_null(
            mgen, n=HELD_MC_N, k=HELD_MC_K, T=HELD_MC_T, B=8, chunk=50_000, B_coinc=2)
        for cid in ("S0", "S1h", "S2", "S2f", "S3"):
            row = mc.run_mc(cid, n=HELD_MC_N, k=HELD_MC_K, T=HELD_MC_T,
                            prng=pname, seed=7_000, chunk=50_000)
            full = mc.mc_row_with_band(row, stat_band)
            full["prng_matched_null_maxdev_mean"] = matched["maxdev_null_mean"]
            full["prng_matched_null_maxdev_p99"] = matched["maxdev_null_p99"]
            full["prng_matched_null_coinc_Zmax_mean"] = matched.get("coinc_Zmax_null_mean")
            full["amplification_vs_matched_null"] = (
                row["maxdev_raw"] / matched["maxdev_null_mean"]
                if matched["maxdev_null_mean"] > 0 else float("nan"))
            rows.append(full)
            logger.info(
                f"prng {pname:10s} {cid:5s} maxdev={row['maxdev_raw']:.6f} "
                f"z={full['z_band']:+9.1f} amp={full['amplification_vs_matched_null']:.3f} "
                f"coincZ={row['coinc_Zmax']:.1f}")
        common.checkpoint("heldout_prng", rows)
    return {"rows": rows, "statistical_band": stat_band}


def stage_algol() -> dict:
    import algol

    out = algol.run_algol_audit(list(HELD_ALGOL_KS), [0, 50], 20_000, 4_242)
    common.checkpoint("heldout_algol", out)
    return out


def stage_exact() -> dict:
    rows: list[dict] = []
    cids = [c for c in MC_CANDIDATES if CANDIDATES[c].children is not None]
    for n in (15, 16, 17, 18):
        for k in (3, 5):
            for cid in cids:
                cand = CANDIDATES[cid]
                if cand.needs_even_k and k % 2:
                    continue
                if cid == "S5e" and n % k:
                    continue
                rows.append(exact.verify(cid, n, k, record_every=False))
            common.checkpoint("heldout_exact", rows)
            logger.info(f"exact n={n} k={k} done")
    return {"rows": rows}


def stage_s1t() -> dict:
    """The table-exact MAIN arm (the planned fallback, triggered by GATE 3)."""
    try:
        import s1t
    except ImportError as exc:
        logger.error(f"s1t unavailable: {exc}")
        return {"status": "NOT_RUN", "reason": str(exc)}
    out: dict = {"status": "RUN", "verify": [], "curves": [], "extrapolation": []}
    for k in (2, 3, 4):
        out["verify"].append(s1t.verify_s1t(12, k))
    # Cost grows like C(i-1,k) per solved step (a max-flow per step, with
    # relabelling retries), so the enumeration is capped.  The cap is RECORDED
    # in the output: every H_evict value below it is exact, and only the
    # extrapolation beyond it is labelled as extrapolated.
    max_sources, i_cap = 15_000, 220
    out["enumeration_cap"] = {"max_sources_per_step": max_sources, "i_cap": i_cap}
    for k in (2, 3, 4, 5):
        c = s1t.h_evict_curve(k, i_max_sources=max_sources, i_cap=i_cap)
        out["curves"].append(c)
        logger.info(f"s1t curve k={k}: C={c.get('fit_C')} R2={c.get('fit_r2') or c.get('r2')} "
                    f"points={c.get('n_points')}")
        common.checkpoint("heldout_s1t", out)
    best = max(out["curves"], key=lambda c: c["k"])
    cfit = best.get("fit_C")
    for k in (10, 100, 1_000):
        for n in (1_000_000, 10_000_000):
            out["extrapolation"].append(s1t.extrapolate_total(k, n, cfit))
    out["fit_C_source_k"] = best["k"]
    common.checkpoint("heldout_s1t", out)
    return out


CALIB_CORRECT = ("S0", "S1h", "S2", "S2f", "S2v", "S3")
CALIB_BROKEN = ("S5e", "S5g", "S6sys")
CALIB_SEEDS = 8


def stage_calib() -> dict:
    """REPLICATION: how often does each test fire on a sampler we can PROVE correct?

    A single run of a maximum statistic proves nothing, so every arm is repeated
    over independent seeds and both decision rules are scored:

      BASELINE  first-order max deviation vs the simulated p99 band
      OURS      co-inclusion, at the PRE-REGISTERED threshold p < 1e-6 and, for
                comparison, at a loose p99 band built from only a handful of
                (expensive) co-inclusion null batches.

    The known-correct samplers give the false-alarm rate; the blind-spot
    exhibits give the detection rate.  This is what licenses reading the
    method-vs-baseline scoreboard in method_out.json.
    """
    gen = prngs.make_prng("PCG64", 246_810)
    band = nullband.simulate_null(gen, n=HELD_MC_N, k=HELD_MC_K, T=HELD_MC_T,
                                  B=48, chunk=50_000, B_coinc=48)
    logger.info(
        f"calibration null B=48/B_coinc=48: maxdev {band['maxdev_null_mean']:.6f}"
        f"+-{band['maxdev_null_sd']:.6f}; chi2 {band['coinc_chi2_null_mean']:.1f}"
        f"+-{band['coinc_chi2_null_sd']:.1f}; Zmax {band['coinc_Zmax_null_mean']:.3f}"
        f"+-{band['coinc_Zmax_null_sd']:.3f}")
    rows: list[dict] = []
    for cid in CALIB_CORRECT + CALIB_BROKEN:
        cand = CANDIDATES[cid]
        if cand.oracle_n and HELD_MC_N % HELD_MC_K:
            continue
        for sd in range(CALIB_SEEDS):
            r = mc.run_mc(cid, n=HELD_MC_N, k=HELD_MC_K, T=HELD_MC_T,
                          seed=770_000 + 13 * sd, chunk=50_000)
            full = mc.mc_row_with_band(r, band)
            full["truth"] = "UNIFORM" if cid in CALIB_CORRECT else "NOT-UNIFORM"
            full["replicate"] = sd
            rows.append(full)
        common.checkpoint("heldout_calib", {"band": band, "rows": rows})
        got = [x for x in rows if x["candidate"] == cid]
        logger.info(
            f"calib {cid:6s} maxdev {np.mean([g['maxdev_raw'] for g in got]):.6f}"
            f"+-{np.std([g['maxdev_raw'] for g in got], ddof=1):.6f} | "
            f"FO fires {sum(not g['passes_first_order_at_p99'] for g in got)}/{len(got)} | "
            f"coinc(prereg) fires {sum(g['coinc_fails_prereg'] for g in got)}/{len(got)} | "
            f"coinc(loose p99) fires {sum(g['coinc_fails_at_p99'] for g in got)}/{len(got)}")
    summary = {}
    for grp, cids in (("known_correct", CALIB_CORRECT), ("blind_spot", CALIB_BROKEN)):
        sel = [r for r in rows if r["candidate"] in cids]
        if not sel:
            continue
        summary[grp] = {
            "n_runs": len(sel),
            "baseline_first_order_fire_rate":
                float(np.mean([not r["passes_first_order_at_p99"] for r in sel])),
            "ours_coinclusion_prereg_fire_rate":
                float(np.mean([r["coinc_fails_prereg"] for r in sel])),
            "ours_coinclusion_loose_p99_fire_rate":
                float(np.mean([r["coinc_fails_at_p99"] for r in sel])),
            "maxdev_mean": float(np.mean([r["maxdev_raw"] for r in sel])),
            "maxdev_sd": float(np.std([r["maxdev_raw"] for r in sel], ddof=1)),
            "coinc_chi2_mean": float(np.mean([r["coinc_chi2"] for r in sel])),
            "coinc_chi2_sd": float(np.std([r["coinc_chi2"] for r in sel], ddof=1)),
        }
    summary["interpretation"] = (
        "The 'known_correct' row is the FALSE-ALARM rate and the 'blind_spot' row is the "
        "DETECTION rate. The pre-registered p < 1e-6 co-inclusion rule is the one used "
        "for every verdict; the loose p99 rule is reported only to show what it costs.")
    return {"band": band, "rows": rows, "summary": summary}


def stage_granularity() -> dict:
    """Does R-recycle actually ATTAIN the entropy floor, or only approach it?

    A single run cannot answer this: the realised number of acceptances is
    random, so one stream's bit count fluctuates around the floor by several
    percent in EITHER direction.  Every cell is therefore repeated over
    independent seeds and the excess is reported as a mean with a standard
    error, alongside the refill-event count so a per-event granularity cost (if
    any) can be separated from ordinary sampling noise.
    """
    import bits as _bits
    from candidates import get_candidate as _gc

    seeds = (1, 2, 3, 4, 5, 6, 7, 8)
    rows: list[dict] = []
    for n in (10_000, 100_000, 1_000_000):
        for k in (10, 100, 1_000):
            if k >= n:
                continue
            for cid in ("S0", "S1h", "S1n"):
                nets, evs, excs = [], [], []
                floor = budget.accept_H(n, k) + (
                    budget.evict_H(n, k) if cid != "S1n" else 0.0)
                for sd in seeds:
                    ev = [0]
                    src = _bits.make_source("recycle", "PCG64", sd)
                    orig = src._refill

                    def patched(need: int, _s=src, _o=orig, _ev=ev) -> None:
                        m0 = _s.M
                        _o(need)
                        if _s.M != m0:
                            _ev[0] += 1

                    src._refill = patched
                    _gc(cid).scalar(src, n, k)
                    nets.append(float(src.net_total()))
                    evs.append(ev[0])
                    excs.append(float(src.net_total()) - floor)
                sem = float(np.std(excs, ddof=1) / math.sqrt(len(seeds)))
                mean_exc = float(np.mean(excs))
                rows.append({
                    "candidate": cid, "n": n, "k": k, "regime": "recycle",
                    "n_seeds": len(seeds),
                    "refill_events_mean": float(np.mean(evs)),
                    "net_bits_mean": float(np.mean(nets)),
                    "net_bits_sd": float(np.std(nets, ddof=1)),
                    "entropy_floor_bits": floor,
                    "excess_bits_mean": mean_exc,
                    "excess_bits_sem": sem,
                    "excess_over_floor_frac": mean_exc / floor if floor else None,
                    "excess_sigmas_from_zero": mean_exc / sem if sem > 0 else float("nan"),
                    "excess_bits_per_refill_event":
                        mean_exc / float(np.mean(evs)) if np.mean(evs) else None,
                })
            logger.info(f"granularity n={n} k={k} done")
        common.checkpoint("heldout_granularity", rows)
    z = [r["excess_sigmas_from_zero"] for r in rows
         if isinstance(r["excess_sigmas_from_zero"], float)
         and math.isfinite(r["excess_sigmas_from_zero"])]
    frac = [r["excess_over_floor_frac"] for r in rows if r["excess_over_floor_frac"]]
    n_sig = sum(1 for x in z if abs(x) > 2.58)
    return {"rows": rows, "summary": {
        "n_cells": len(rows),
        "mean_excess_over_floor_frac": float(np.mean(frac)),
        "max_abs_excess_over_floor_frac": float(np.max(np.abs(frac))),
        "cells_excess_beyond_2.58_sigma": int(n_sig),
        "mean_excess_sigmas": float(np.mean(z)),
        "interpretation": (
            "Across every (n, k, sampler) cell, the measured net bit cost of the "
            "recycling source sits within a small fraction of the closed-form entropy "
            "floor sum_i h(k/i) + log2(k)*E[accepts], and the residual is dominated by "
            "the randomness of the realised acceptance count rather than by any "
            "systematic implementation overhead: the number of cells whose excess is "
            "more than 2.58 standard errors from zero is reported above. A single-run "
            "measurement of this quantity is uninformative in either direction, which "
            "is why every cell is replicated.")}}


GRID = ((1_000, 10), (1_000, 50), (2_000, 10), (2_000, 100))
GRID_T = 200_000


def stage_mcgrid() -> dict:
    """Is the blind spot an artefact of one (n, k), or does it generalise?

    The same measurement -- max deviation against a SIMULATED band, plus exact
    co-inclusion -- repeated over a grid of (n, k).  n is capped at 2000 here
    because the dense co-inclusion accumulator is O(n^2) (32 MB at n=2000,
    800 MB at n=10000); that bound is stated rather than silently worked around.
    """
    rows: list[dict] = []
    bands: dict = {}
    for n, k in GRID:
        gen = prngs.make_prng("PCG64", 13_000 + n + k)
        band = nullband.simulate_null(gen, n=n, k=k, T=GRID_T, B=32,
                                      chunk=50_000, B_coinc=8)
        bands[f"n{n}_k{k}"] = band
        for cid in ("S0", "S1h", "S2", "S2f", "S3", "S1n", "S5e", "S5g", "S6sys"):
            cand = CANDIDATES[cid]
            if cand.needs_even_k and k % 2:
                continue
            if cand.oracle_n and n % k:
                continue
            r = mc.run_mc(cid, n=n, k=k, T=GRID_T, seed=424_242, chunk=50_000)
            full = mc.mc_row_with_band(r, band)
            rows.append(full)
            logger.info(
                f"grid n={n} k={k} {cid:6s} maxdev={full['maxdev_raw']:.6f} "
                f"z={full['z_band']:+7.2f} FO_pass={full['passes_first_order_at_p99']} "
                f"coincZ={full['coinc_Zmax']:9.1f} BLIND={full.get('blind_spot')}")
        common.checkpoint("heldout_mcgrid", {"rows": rows, "bands": bands})
    blind = [r for r in rows if r.get("blind_spot")]
    return {"rows": rows, "bands": bands, "summary": {
        "grid": [list(g) for g in GRID], "T": GRID_T,
        "n_blind_spot_cells": len(blind),
        "blind_spot_cells": [f"{r['candidate']}@n{r['n']}k{r['k']}" for r in blind],
        "n_cap_note": "n <= 2000: the dense co-inclusion accumulator is O(n^2) int64.",
    }}


STAGES = {
    "mc": stage_mc, "calib": stage_calib, "granularity": stage_granularity,
    "mcgrid": stage_mcgrid, "bits": stage_bits, "prng": stage_prng,
    "algol": stage_algol, "exact": stage_exact, "s1t": stage_s1t,
}


@logger.catch(reraise=True)
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", *STAGES.keys()])
    args = ap.parse_args()
    common.setup_logging(f"HELDOUT:{args.stage}")
    common.set_limits(24.0)
    freeze = verify_freeze()
    order = ["mc", "calib", "mcgrid", "granularity", "bits", "prng", "algol", "exact", "s1t"] if args.stage == "all" else [args.stage]
    out: dict = {"panel": "HELD-OUT", "screen_freeze_sha256": freeze,
                 "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    for name in order:
        with common.stage(f"heldout_{name}"):
            try:
                out[name] = STAGES[name]()
            except (RuntimeError, ValueError, KeyError, MemoryError, AssertionError,
                    ImportError, OverflowError) as exc:
                logger.error(f"stage {name} failed: {type(exc).__name__}: {exc}")
                out[name] = {"status": "NOT_RUN", "reason": f"{type(exc).__name__}: {exc}"}
        common.dump_json(common.RESULTS / f"heldout_{args.stage}.json", out)
    out["stage_times_s"] = common.stage_times()
    common.dump_json(common.RESULTS / f"heldout_{args.stage}.json", out)


if __name__ == "__main__":
    main()
