#!/usr/bin/env python3
"""SCREEN panel -- runs FIRST, never touches a held-out configuration.

Writes results/screen_ranking.json, whose ``content_sha256`` is logged BEFORE
heldout.py is launched.  heldout.py recomputes that hash on startup and aborts
if it changed: that is the pre-registration, and it is not optional.
"""

from __future__ import annotations

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

SCREEN_EXACT_N = 12
SCREEN_EXACT_KS = (2, 4)
SCREEN_MC_N = 200
SCREEN_MC_K = 10
SCREEN_MC_TS = (1_000, 10_000, 100_000)
SCREEN_MC_B = 128
SCREEN_BIT_NS = (1_000, 10_000)
SCREEN_BIT_KS = (2, 10)
SCREEN_PRNGS = prngs.SCREEN_PRNGS
SCREEN_ALGOL_KS = (2, 10, 100, 256)

EXACT_CIDS = [c for c in MC_CANDIDATES if CANDIDATES[c].children is not None]
BIT_CIDS = ("S0", "S1", "S1h", "S1n", "S2", "S2f", "S2v", "S3", "S5e", "S5g")


def run_exact_screen() -> list[dict]:
    rows: list[dict] = []
    for cid in EXACT_CIDS:
        cand = CANDIDATES[cid]
        for k in SCREEN_EXACT_KS:
            if cand.needs_even_k and k % 2:
                continue
            if cid == "S5e" and SCREEN_EXACT_N % k:
                continue
            r = exact.verify(cid, SCREEN_EXACT_N, k, record_every=True)
            rows.append(r)
            logger.info(
                f"exact {cid:5s} k={k} maxabs={r['max_abs_dev']:.3e} "
                f"relmax={r['max_rel_dev']:.4f} supp={r['support_size']}/{r['n_subsets']} "
                f"FOmax={r['first_order_max_dev']:.3e} firstfail={r['first_prefix_failure']}"
            )
    return rows


def run_mc_screen() -> tuple[list[dict], dict, list[dict]]:
    """Gradual T scaling with runtime logged at every step (GATE 6)."""
    timings: list[dict] = []
    rows: list[dict] = []
    band: dict = {}
    for T in SCREEN_MC_TS:
        gen = prngs.make_prng("PCG64", 90210 + T)
        t0 = time.time()
        band = nullband.simulate_null(
            gen, n=SCREEN_MC_N, k=SCREEN_MC_K, T=T, B=SCREEN_MC_B,
            chunk=min(T, 50_000), B_coinc=max(4, SCREEN_MC_B // 4),
        )
        band_s = time.time() - t0
        this: list[dict] = []
        for cid in MC_CANDIDATES:
            cand = CANDIDATES[cid]
            if cand.needs_even_k and SCREEN_MC_K % 2:
                continue
            if cand.oracle_n and SCREEN_MC_N % SCREEN_MC_K:
                continue
            row = mc.run_mc(cid, n=SCREEN_MC_N, k=SCREEN_MC_K, T=T,
                            seed=4242 + T, chunk=min(T, 50_000))
            this.append(mc.mc_row_with_band(row, band))
        dt = time.time() - t0
        timings.append({"T": T, "band_s": round(band_s, 2), "total_s": round(dt, 2),
                        "s_per_candidate": round((dt - band_s) / max(len(this), 1), 3)})
        logger.info(f"MC screen T={T}: band {band_s:.1f}s, all candidates {dt - band_s:.1f}s")
        rows = this
        common.checkpoint("screen_mc", {"T": T, "rows": rows, "band": band, "timings": timings})
    return rows, band, timings


def run_bits_screen() -> list[dict]:
    rows: list[dict] = []
    for n in SCREEN_BIT_NS:
        for k in SCREEN_BIT_KS:
            for regime in REGIMES:
                for cid in BIT_CIDS:
                    cand = CANDIDATES[cid]
                    if cand.needs_even_k and k % 2:
                        continue
                    if cand.oracle_n and n % k:
                        continue
                    rows.append(mc.run_bits(cid, n=n, k=k, regime=regime,
                                            seeds=(1, 2, 3, 4, 5)))
        common.checkpoint("screen_bits", rows)
    return rows


def run_prng_screen() -> list[dict]:
    """M6 screen: does generator quality separate the ALGORITHMS?

    Each generator gets its own PRNG-MATCHED null band (the rejection sampler
    driven by the same generator), so a generator that is simply bad is
    separated from an algorithm that AMPLIFIES a bad generator.
    """
    rows: list[dict] = []
    n, k, T = SCREEN_MC_N, SCREEN_MC_K, 100_000
    stat_gen = prngs.make_prng("PCG64", 5150)
    stat_band = nullband.simulate_null(stat_gen, n=n, k=k, T=T, B=32,
                                       chunk=50_000, B_coinc=8)
    for pname in SCREEN_PRNGS:
        mgen = prngs.make_prng(pname, 6161)
        matched = nullband.simulate_null(mgen, n=n, k=k, T=T, B=8,
                                         chunk=50_000, B_coinc=2)
        for cid in ("S0", "S1h", "S2", "S2f", "S3"):
            row = mc.run_mc(cid, n=n, k=k, T=T, prng=pname, seed=7000, chunk=50_000)
            full = mc.mc_row_with_band(row, stat_band)
            full["prng_matched_null_maxdev_mean"] = matched["maxdev_null_mean"]
            full["prng_matched_null_maxdev_p99"] = matched["maxdev_null_p99"]
            full["prng_matched_null_coinc_Zmax_mean"] = matched.get("coinc_Zmax_null_mean")
            full["amplification_vs_matched_null"] = (
                row["maxdev_raw"] / matched["maxdev_null_mean"]
                if matched["maxdev_null_mean"] > 0 else float("nan")
            )
            rows.append(full)
            logger.info(
                f"prng {pname:11s} {cid:5s} maxdev={row['maxdev_raw']:.5f} "
                f"z_band={full['z_band']:.1f} amp={full['amplification_vs_matched_null']:.2f} "
                f"coincZ={row['coinc_Zmax']:.1f}"
            )
        common.checkpoint("screen_prng", rows)
    return rows


def run_algol_screen() -> dict:
    import algol

    return algol.run_algol_audit(list(SCREEN_ALGOL_KS), [0, 50], 20_000, 20260918)


def margin_ratios(*, exact_rows, mc_rows, bit_rows, prng_rows, algol_rows) -> dict:
    """Dimensionless margins, computed on the SCREEN panel only."""
    ex = {(r["candidate"], r["k"]): r for r in exact_rows}
    mcr = {r["candidate"]: r for r in mc_rows}
    bits_at = {
        (r["candidate"], r["n"], r["k"], r["regime"]): r for r in bit_rows
    }

    # ---- MAIN: eviction randomness is wasted ----------------------------------
    s1_exact = all(
        ex[(c, k)]["max_abs_dev"] == 0.0
        for (c, k) in ex if c == "S1"
    )
    main_variant = "S1" if s1_exact else "S1h"
    n_b, k_b = 10_000, 10
    base = bits_at[("S0", n_b, k_b, "recycle")]["flips_total_net_mean"]
    main = bits_at[(main_variant, n_b, k_b, "recycle")]["flips_total_net_mean"]
    free = bits_at[("S1n", n_b, k_b, "recycle")]["flips_total_net_mean"]
    saving = (base - main) / base
    saving_upper = (base - free) / base
    main_ratio = saving / 0.20

    # ---- C4: representation vs coupling ---------------------------------------
    s4d = bits_at[("S0", n_b, k_b, "naive64")]["flips_total_net_mean"]
    s4 = bits_at[("S0", n_b, k_b, "KY")]["flips_total_net_mean"]
    s0_rec = base
    coupling_gap = max(s4 - main, 1e-12)
    c4_ratio = (s4d - s4) / (2.0 * coupling_gap)

    # ---- C2: Algorithm-L float bias -------------------------------------------
    cells = algol_rows.get("cells", [])
    worst_rate_err = 0.0
    for c in cells:
        for key in ("rel_rate_err_max_f64", "rel_rate_err_mean_f64"):
            v = c.get(key)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                worst_rate_err = max(worst_rate_err, float(v))
    s2row = mcr.get("S2", {})
    sigma = s2row.get("sigma_per_bin", float("nan"))
    c2_ratio = (
        abs(s2row.get("maxdev_raw", 0.0) - mcr.get("S0", {}).get("maxdev_raw", 0.0))
        / (3.0 * sigma) if sigma and math.isfinite(sigma) and sigma > 0 else float("nan")
    )

    # ---- C3: does PRNG quality separate the ALGORITHMS? -----------------------
    by_prng: dict[str, dict[str, float]] = {}
    for r in prng_rows:
        by_prng.setdefault(r["prng"], {})[r["candidate"]] = r["maxdev_raw"]
    seps = []
    for p, d in by_prng.items():
        vals = [v for v in d.values() if v > 0]
        if len(vals) >= 2:
            seps.append(math.log10(max(vals) / min(vals)))
    c3_ratio = max(seps) if seps else 0.0

    # ---- C1: first-order blind spot -------------------------------------------
    c1_members = []
    for cid in ("S5e", "S5g", "S6sys", "S1n", "S5a", "S5b", "S5c", "S5f"):
        e = next((ex[(c, k)] for (c, k) in ex if c == cid), None)
        m = mcr.get(cid)
        if e is None or m is None:
            continue
        ok = (
            e.get("first_order_exact_all_prefixes") is True
            and e["support_size"] < 4 * e["n"]
            and bool(m.get("passes_first_order_at_p99"))
            and bool(m.get("coinc_fails_at_p99"))
        )
        c1_members.append({
            "candidate": cid,
            "first_order_exact_all_prefixes": e.get("first_order_exact_all_prefixes"),
            "support_size": e["support_size"],
            "n_subsets": e["n_subsets"],
            "passes_first_order_at_p99": m.get("passes_first_order_at_p99"),
            "coinc_fails_at_p99": m.get("coinc_fails_at_p99"),
            "coinc_p": m.get("coinc_p"),
            "qualifies": ok,
        })
    c1_ratio = 1.0 if any(m["qualifies"] for m in c1_members) else 0.0

    return {
        "MAIN": {
            "ratio": main_ratio,
            "threshold": 0.20,
            "measured_saving_frac": saving,
            "zero_eviction_upper_bound_frac": saving_upper,
            "main_variant_used": main_variant,
            "S1_exact": s1_exact,
            "basis": f"matched R-recycle, n={n_b}, k={k_b}, net bits",
            "S0_net_bits": base,
            "main_net_bits": main,
            "S1n_net_bits": free,
        },
        "C1": {"ratio": c1_ratio, "members": c1_members},
        "C2": {
            "ratio": c2_ratio,
            "worst_rel_rate_err_f64": worst_rate_err,
            "sigma_per_bin": sigma,
        },
        "C3": {"ratio": c3_ratio, "log10_separation_by_prng": {
            p: (math.log10(max(v.values()) / min(x for x in v.values() if x > 0))
                if any(x > 0 for x in v.values()) else 0.0)
            for p, v in by_prng.items()}},
        "C4": {
            "ratio": c4_ratio,
            "naive64_bits": s4d,
            "KY_bits": s4,
            "recycle_S0_bits": s0_rec,
            "main_bits": main,
            "representation_gap": s4d - s4,
            "coupling_gap": coupling_gap,
        },
    }


@logger.catch(reraise=True)
def main() -> None:
    common.setup_logging("SCREEN")
    common.set_limits(24.0)
    gates: list[dict] = []

    with common.stage("GATE2_budget"):
        checks = {10: 0.255, 100: 0.454, 1000: 0.613}
        ok, det = True, []
        for k, expect in checks.items():
            a = budget.accept_H(10**6, k)
            e = budget.evict_H(10**6, k)
            f = e / (a + e)
            det.append(f"k={k}: evict_frac={f:.4f} (expect {expect})")
            ok &= abs(f - expect) < 0.005
        ok &= abs(budget.evict_H(10**6, 100) - 6116) / 6116 < 0.01
        gates.append(common.gate("2-budget", ok, "; ".join(det)))
        if not ok:
            raise AssertionError("GATE 2 failed -- h() or the summation limits are wrong")

    with common.stage("exact_screen"):
        exact_rows = run_exact_screen()
        s0 = [r for r in exact_rows if r["candidate"] == "S0"]
        ctrl = [r for r in exact_rows
                if r["candidate"] in ("S1n", "S5a", "S5b", "S5c", "S5f")]
        ok = all(r["max_abs_dev"] == 0.0 for r in s0) and all(
            r["max_abs_dev"] > 0.0 for r in ctrl)
        gates.append(common.gate(
            "3-exact", ok,
            f"S0 exact at {len(s0)} cells; {len(ctrl)} negative controls all nonzero; "
            f"S1 exact={all(r['max_abs_dev'] == 0.0 for r in exact_rows if r['candidate'] == 'S1')}"))
        common.checkpoint("screen_exact", exact_rows)

    with common.stage("mc_screen"):
        mc_rows, band, timings = run_mc_screen()
        # GATE 6 confirmation signals, measured not assumed:
        #  (a) S0's z_band must be centred on 0 over independent seeds -- a
        #      single draw is itself a random variable, so one seed cannot test
        #      the harness;
        #  (b) S0's maxdev must fall as T^(-1/2);
        #  (c) the blind-spot exhibit's co-inclusion Zmax must GROW as sqrt(T)
        #      while its first-order maxdev does not.  That divergence IS the
        #      candidate-1 claim and it is visible already at T = 1e4.
        zs = []
        for sd in range(8):
            r = mc.run_mc("S0", n=SCREEN_MC_N, k=SCREEN_MC_K, T=SCREEN_MC_TS[-1],
                          seed=31000 + sd, coinclusion=False)
            zs.append(mc.mc_row_with_band(r, band)["z_band"])
        zmean = float(np.mean(zs))
        ts, mds, zmaxs, fos = [], [], [], []
        for T in SCREEN_MC_TS:
            r0 = mc.run_mc("S0", n=SCREEN_MC_N, k=SCREEN_MC_K, T=T, seed=777,
                           coinclusion=False)
            rb = mc.run_mc("S5g", n=SCREEN_MC_N, k=SCREEN_MC_K, T=T, seed=778)
            ts.append(T); mds.append(r0["maxdev_raw"])
            zmaxs.append(rb["coinc_Zmax"]); fos.append(rb["maxdev_raw"])
        slope_md = float(np.polyfit(np.log10(ts), np.log10(mds), 1)[0])
        slope_z = float(np.polyfit(np.log10(ts), np.log10(zmaxs), 1)[0])
        slope_fo = float(np.polyfit(np.log10(ts), np.log10(fos), 1)[0])
        ok = (-2.0 < zmean < 2.0) and abs(slope_md + 0.5) < 0.12 and slope_z > 0.3
        gates.append(common.gate(
            "6-mc-scaling", ok,
            f"S0 mean z_band over 8 seeds = {zmean:.2f} (draws "
            f"{[round(z, 2) for z in zs]}); S0 maxdev ~ T^{slope_md:.3f} "
            f"(expect -0.5); S5g coinc Zmax ~ T^{slope_z:.3f} (expect +0.5) while "
            f"its first-order maxdev ~ T^{slope_fo:.3f}; timings={timings}"))
        mc_scaling = {"T": ts, "S0_maxdev": mds, "S0_maxdev_slope": slope_md,
                      "S5g_coinc_Zmax": zmaxs, "S5g_coinc_Zmax_slope": slope_z,
                      "S5g_first_order_maxdev": fos,
                      "S5g_first_order_slope": slope_fo,
                      "S0_z_band_8_seeds": zs, "S0_z_band_mean": zmean}

    with common.stage("bits_screen"):
        bit_rows = run_bits_screen()
        r = next(x for x in bit_rows
                 if x["candidate"] == "S0" and x["n"] == 10_000 and x["k"] == 10
                 and x["regime"] == "recycle")
        ok = abs(r["flips_total_net_mean"] / r["closed_form_total_H"] - 1.0) < 0.05
        rn = next(x for x in bit_rows if x["candidate"] == "S0" and x["n"] == 10_000
                  and x["k"] == 10 and x["regime"] == "naive64")
        ok &= abs(rn["flips_total_net_mean"] / rn["closed_form_naive64"] - 1.0) < 0.005
        gates.append(common.gate(
            "7-bit-accounting", ok,
            f"recycle/entropy={r['ratio_to_closed_form_entropy']:.4f}, "
            f"naive64/closed={rn['flips_total_net_mean'] / rn['closed_form_naive64']:.5f}"))

    with common.stage("prng_screen"):
        prng_rows = run_prng_screen()

    with common.stage("algol_screen"):
        try:
            algol_rows = run_algol_screen()
        except (ImportError, RuntimeError, ValueError) as exc:
            logger.error(f"algol screen unavailable: {exc}")
            algol_rows = {"cells": [], "edge_cases": [], "notes": {"error": str(exc)}}

    ratios = margin_ratios(exact_rows=exact_rows, mc_rows=mc_rows, bit_rows=bit_rows,
                           prng_rows=prng_rows, algol_rows=algol_rows)
    ranking = sorted(
        [("MAIN", ratios["MAIN"]["ratio"]), ("C1", ratios["C1"]["ratio"]),
         ("C2", ratios["C2"]["ratio"]), ("C3", ratios["C3"]["ratio"]),
         ("C4", ratios["C4"]["ratio"])],
        key=lambda t: (-t[1] if math.isfinite(t[1]) else -1e18),
    )
    payload = {
        "panel": "SCREEN",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {
            "exact_n": SCREEN_EXACT_N, "exact_ks": list(SCREEN_EXACT_KS),
            "mc_n": SCREEN_MC_N, "mc_k": SCREEN_MC_K, "mc_Ts": list(SCREEN_MC_TS),
            "mc_null_B": SCREEN_MC_B, "bit_ns": list(SCREEN_BIT_NS),
            "bit_ks": list(SCREEN_BIT_KS), "prngs": list(SCREEN_PRNGS),
            "algol_ks": list(SCREEN_ALGOL_KS),
        },
        "gates": gates,
        "ranking": [{"candidate": c, "margin_ratio": r} for c, r in ranking],
        "margin_ratios": ratios,
        "mc_screen": mc_rows,
        "mc_null_band": band,
        "mc_timings": timings,
        "mc_scaling": mc_scaling,
        "bits_screen": bit_rows,
        "prng_screen": prng_rows,
        "exact_screen": exact_rows,
        "algol_screen": algol_rows,
        "stage_times_s": common.stage_times(),
    }
    payload["content_sha256"] = common.content_sha256(
        {k: v for k, v in payload.items() if k != "content_sha256"}
    )
    common.dump_json(common.RESULTS / "screen_ranking.json", payload)
    logger.info(f"SCREEN FROZEN -- content_sha256 = {payload['content_sha256']}")
    logger.info(f"ranking = {[(c, round(r, 4) if math.isfinite(r) else r) for c, r in ranking]}")


if __name__ == "__main__":
    main()
