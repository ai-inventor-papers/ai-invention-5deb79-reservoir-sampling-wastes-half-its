#!/usr/bin/env python3
"""SCREEN driver.  Executes ONLY the pre-registered screen panel, then freezes the
candidate ranking to ``screen_ranking.json`` together with a content hash.

``run_heldout.py`` refuses to touch any held-out configuration unless this file exists
and its recorded hash matches its own content.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from math import comb, log2
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from zlib import crc32

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import candidates  # noqa: E402
import fibermap  # noqa: E402
import forced  # noqa: E402
import montecarlo as mc  # noqa: E402
import panels  # noqa: E402
import verdicts  # noqa: E402
from exact_dp import dp_cell  # noqa: E402

def key_seed(key: str) -> int:
    """Deterministic per-candidate seed offset (``hash`` is randomised per process)."""
    return crc32(key.encode()) % 9973


RESULTS = ROOT / "results"
SCREEN_RANKING = ROOT / "screen_ranking.json"
N_WORKERS = 4


# --------------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------------


def run_exact_block(block: dict[str, Any], *, label: str) -> list[dict[str, Any]]:
    tasks = [
        (key, n, k, block["stop_on_first_fail"], True)
        for key in block["candidates"]
        for (n, k) in block["cells"]
        if k <= n
    ]
    logger.info(f"[{label}] exact DP: {len(tasks)} cells on {N_WORKERS} workers")
    out: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(dp_cell, t): t for t in tasks}
        for fut in as_completed(futs):
            key, n, k = futs[fut][0], futs[fut][1], futs[fut][2]
            try:
                row = fut.result()
            except Exception:
                logger.error(f"[{label}] exact DP cell {key} n={n} k={k} FAILED")
                raise
            row["key"] = key
            row["description"] = candidates.DESCRIPTIONS[key]
            out.append(row)
            status = "EXACTLY UNIFORM" if row["i_first_fail"] is None else f"fails at i={row['i_first_fail']}"
            logger.debug(f"[{label}] {key:11s} n={n:2d} k={k}: {status}")
    logger.info(f"[{label}] exact DP done in {time.perf_counter() - t0:.1f}s")
    out.sort(key=lambda r: (r["key"], r["k"], r["n"]))
    return out


def run_fiber_block(cells: list[tuple[int, int]], *, label: str) -> list[dict[str, Any]]:
    tasks = [(i, k, True) for (i, k) in cells]
    logger.info(f"[{label}] fibermap: {len(tasks)} steps on {N_WORKERS} workers")
    t0 = time.perf_counter()
    out: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=mp.get_context("spawn")) as pool:
        futs = {pool.submit(fibermap.analyse_cell, t): t for t in tasks}
        for fut in as_completed(futs):
            i, k, _ = futs[fut]
            try:
                out.append(fut.result())
            except Exception:
                logger.error(f"[{label}] fibermap cell i={i} k={k} FAILED")
                raise
    logger.info(f"[{label}] fibermap done in {time.perf_counter() - t0:.1f}s")
    out.sort(key=lambda r: (r["k"], r["i_prefix"]))
    return out


def run_matched_block(cfg: dict[str, Any], seed: int, *, label: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    for (n, k) in cfg["cells"]:
        rng = np.random.Generator(np.random.PCG64(seed + n * 31 + k))
        final_null = mc.null_band(n, k, cfg["T"], cfg["R"], rng)
        prefix_null = verdicts.all_prefix_null_band(n, k, cfg["T"], cfg["prefix_R"], seed + n + k)
        for key in cfg["candidates"]:
            if k > n:
                continue
            row = verdicts.run_matched_cell(
                key, n, k, cfg["T"], seed + key_seed(key),
                final_null=final_null, prefix_null=prefix_null,
            )
            rows.append(row)
            logger.info(
                f"[{label}] matched {key:9s} n={n} k={k}: truth={row['truth_exact_dp']:12s} "
                f"mc1={row['mc_final_first_order']['verdict']:12s} "
                f"mcpair={row['mc_final_pairwise']['verdict']:12s} "
                f"mcany={row['mc_all_prefix']['verdict']}"
            )
    conf = {m: verdicts.confusion(rows, m) for m in
            ("mc_final_first_order", "mc_final_pairwise", "mc_all_prefix")}
    conf["exact_dp"] = {
        "method": "exact_dp",
        "n_cells": len(rows),
        "accuracy": 1.0,
        "note": "ground truth by construction: exact rational arithmetic is a proof",
    }
    logger.info(f"[{label}] matched verdicts done in {time.perf_counter() - t0:.1f}s")
    return {"rows": rows, "confusion": conf}


def run_mc_block(cfg: dict[str, Any], seed: int, *, label: str) -> dict[str, Any]:
    n, k, T, R = cfg["n"], cfg["k"], cfg["T"], cfg["R"]
    logger.info(f"[{label}] Monte Carlo: n={n} k={k} T={T} R={R}")
    t0 = time.perf_counter()
    rng = np.random.Generator(np.random.PCG64(seed))
    band = mc.null_band(n, k, T, R, rng)
    analytic = mc.gumbel_band_analytic(n, k, T)
    rows = []
    for key in cfg["candidates"]:
        spec = verdicts.MC_SPEC[key]
        rows.append(
            mc.run_mc(
                key, n, k, T, R, seed + key_seed(key),
                sampler=spec["sampler"],
                rule=spec["rule"] if spec["sampler"] == "algo_l" else "uniform",
                null=band,
            )
        )
    for r in rows:
        f1, p1 = verdicts.mc_flag(r["first_order"]["max_dev"], band["first_order"]["values"],
                                  panels.FLAG_ALPHA)
        f2, p2 = verdicts.mc_flag(r["pair"]["max_abs_z"], band["pair_z"]["values"],
                                  panels.FLAG_ALPHA)
        r["flagged_first_order_p95_rule"] = r["flagged_first_order"]
        r["flagged_second_order_p95_rule"] = r["flagged_second_order"]
        r["mc_p_value_first_order"] = p1
        r["mc_p_value_pairwise"] = p2
        r["flagged_first_order"] = f1
        r["flagged_second_order"] = f2
        r["flag_rule"] = (
            f"exact Monte-Carlo p-value (1 + #(null >= obs))/(R+1) <= {panels.FLAG_ALPHA}; "
            f"R={R} null replicates"
        )
    logger.info(f"[{label}] Monte Carlo done in {time.perf_counter() - t0:.1f}s")
    return {
        "n": n, "k": k, "T": T, "R": R,
        "null_band_simulated": band,
        "null_band_analytic_gumbel": analytic,
        "band_cross_check_ratio_p95": (
            band["first_order"]["p95"] / analytic["p95_max_dev"] if analytic["p95_max_dev"] else None
        ),
        "rows": rows,
    }


def run_budget_block(cells: list[tuple[int, int]]) -> dict[str, Any]:
    budgets = {}
    for (n, k) in cells:
        budgets[(n, k)] = forced.bit_budget(n, k)
    return {
        "rows": [budgets[c] for c in cells],
        "hypothesis_constant_checks": forced.check_hypothesis_constants(budgets),
    }


def run_forced_block(cells: list[tuple[int, int]]) -> list[dict[str, Any]]:
    return [forced.forced_accept_check(i, k) for (i, k) in cells]


# --------------------------------------------------------------------------------------
# Margin ratios (pre-registered)
# --------------------------------------------------------------------------------------


def score_candidates(
    exact_rows: list[dict[str, Any]],
    fiber_rows: list[dict[str, Any]],
    budgets: dict[str, Any],
    matched: dict[str, Any],
    mcblock: dict[str, Any],
) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = {}
    for r in exact_rows:
        by_key.setdefault(r["key"], []).append(r)

    def all_uniform(key: str) -> bool:
        rows = by_key.get(key, [])
        return bool(rows) and all(r["i_first_fail"] is None and not r["stopped_early"] for r in rows)

    # --- MAIN ---
    ratios = [r["ratio_to_log2k"] for r in fiber_rows if r["k"] > 1]
    frac_below_quarter = float(np.mean([x <= 0.25 for x in ratios])) if ratios else 0.0
    k10 = next((b for b in budgets["rows"] if b["k"] == 10), None)
    saving = k10["bit_saving_frac_lower_bound"] if k10 else 0.0
    main_reqs = {
        "S1_exactly_uniform_all_cells": all_uniform("S1"),
        "S1f_exactly_uniform_all_cells": all_uniform("S1f"),
        "frac_fiber_steps_with_ratio_le_0.25": frac_below_quarter,
        "frac_fiber_steps_requirement": 0.5,
        "frac_requirement_met": frac_below_quarter >= 0.5,
        "analytic_bit_saving_at_k10": saving,
        "bit_saving_requirement": 0.20,
        "bit_saving_requirement_met": saving >= 0.20,
    }
    main_passed = (
        main_reqs["S1_exactly_uniform_all_cells"]
        and main_reqs["S1f_exactly_uniform_all_cells"]
        and main_reqs["frac_requirement_met"]
        and main_reqs["bit_saving_requirement_met"]
    )
    main_margin = min(
        frac_below_quarter / 0.5 if 0.5 else 0.0,
        saving / 0.20 if 0.20 else 0.0,
    )

    # --- C1 ---
    circ_rows = by_key.get("S5-CIRC", [])
    incl_exact = all(
        all(p["incl_maxdev_float"] == 0.0 for p in r["prefixes"]) for r in circ_rows
    ) and bool(circ_rows)
    tv_min = min(
        (min(p["tv_float"] for p in r["prefixes"] if p["i"] > r["k"]) for r in circ_rows),
        default=0.0,
    )
    tv_max = max((max(p["tv_float"] for p in r["prefixes"]) for r in circ_rows), default=0.0)
    # One distinct circular window per anchor -- except at the degenerate prefix i == k,
    # where every anchor gives the whole of [k] and C(k,k) = 1, so support == 1 == C(i,k)
    # and the prefix is (trivially) exactly uniform.
    support_linear = all(
        all(p["support"] == min(p["i"], p["support_total"]) for p in r["prefixes"])
        for r in circ_rows
    ) and bool(circ_rows)
    circ_mc = next((r for r in mcblock["rows"] if r["candidate"] == "S5-CIRC"), None)
    c1_reqs = {
        "exact_first_order_marginals_zero_deviation": incl_exact,
        "exact_TV_max_over_screened_prefixes": tv_max,
        "exact_TV_requirement": 0.9,
        "TV_requirement_met": tv_max >= 0.9,
        "support_equals_min_i_and_C_i_k_not_C_i_k": support_linear,
        "support_note": (
            "support == i for every prefix i > k; at i == k it is 1 because C(k,k) = 1"
        ),
        "mc_first_order_blind_at_screen_scale": (not circ_mc["flagged_first_order"]) if circ_mc else None,
        "mc_pairwise_flags_at_screen_scale": circ_mc["flagged_second_order"] if circ_mc else None,
        "mc_pairwise_p_value": circ_mc["pair_p_value_normal"] if circ_mc else None,
    }
    c1_passed = bool(
        incl_exact
        and tv_max >= 0.9
        and support_linear
        and circ_mc is not None
        and not circ_mc["flagged_first_order"]
        and circ_mc["flagged_second_order"]
    )
    c1_margin = (tv_max / 0.9) if tv_max else 0.0

    ranking = [
        {
            "candidate": "MAIN_S1",
            "margin_ratio": float(main_margin),
            "passed": bool(main_passed),
            "requirements": main_reqs,
            "evidence_pointers": ["exact[].key in {S1,S1f}", "fibermap[].ratio_to_log2k",
                                  "bit_budgets.rows[k=10].bit_saving_frac_lower_bound"],
        },
        {
            "candidate": "C1_S5CIRC",
            "margin_ratio": float(c1_margin),
            "passed": bool(c1_passed),
            "requirements": c1_reqs,
            "evidence_pointers": ["exact[].key == S5-CIRC", "montecarlo.rows[S5-CIRC]",
                                  "matched_verdicts.confusion"],
        },
    ]
    ranking.sort(key=lambda r: (not r["passed"], -r["margin_ratio"]))
    return {
        "ranking": ranking,
        "matched_confusion_summary": {
            m: {kk: c[kk] for kk in ("n_cells", "accuracy", "recall_on_broken_samplers",
                                     "specificity_on_correct_samplers", "missed_cells")
                if kk in c}
            for m, c in matched["confusion"].items()
        },
        "tv_min_over_screened_prefixes": tv_min,
        "deferred_to_instrument_2": panels.DEFERRED,
        "margins_preregistered": panels.MARGINS,
    }


# --------------------------------------------------------------------------------------


def main() -> dict[str, Any]:
    RESULTS.mkdir(exist_ok=True)
    seed = panels.SCREEN["seed"]
    t0 = time.perf_counter()
    logger.info("=" * 78)
    logger.info("SCREEN PANEL (pre-registered; frozen before any held-out configuration)")
    logger.info("=" * 78)

    exact_main = run_exact_block(panels.SCREEN["exact_dp"]["main"], label="screen")
    exact_ord = run_exact_block(panels.SCREEN["exact_dp"]["ordered"], label="screen-ordered")
    exact_rows = exact_main + exact_ord

    fiber_rows = run_fiber_block(panels.SCREEN["fibermap"]["cells"], label="screen")
    budgets = run_budget_block(panels.SCREEN["bit_budgets"]["cells"])
    forced_rows = run_forced_block(panels.SCREEN["forced_accept"]["cells"])
    mcblock = run_mc_block(panels.SCREEN["montecarlo"], seed, label="screen")
    matched = run_matched_block(panels.SCREEN["matched_verdicts"], seed, label="screen")

    ranking = score_candidates(exact_rows, fiber_rows, budgets, matched, mcblock)

    payload = {
        "panel_definition": {
            "exact_dp": panels.SCREEN["exact_dp"],
            "fibermap_cells": panels.SCREEN["fibermap"]["cells"],
            "matched_verdicts": panels.SCREEN["matched_verdicts"],
            "montecarlo": {k: v for k, v in panels.SCREEN["montecarlo"].items()},
            "bit_budgets": panels.SCREEN["bit_budgets"]["cells"],
            "forced_accept": panels.SCREEN["forced_accept"]["cells"],
            "seed": seed,
        },
        **ranking,
        "wall_seconds": time.perf_counter() - t0,
    }
    # The hash covers everything EXCEPT wall-clock timing, so it doubles as a determinism
    # check: re-running run_screen.py must reproduce it byte for byte.
    wall = payload.pop("wall_seconds")
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    payload["sha256_of_this_file_content"] = hashlib.sha256(body.encode()).hexdigest()
    payload["wall_seconds"] = wall
    payload["hash_excludes"] = ["wall_seconds", "sha256_of_this_file_content", "hash_excludes"]
    SCREEN_RANKING.write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"FROZEN screen_ranking.json  sha256={payload['sha256_of_this_file_content']}")
    for r in payload["ranking"]:
        logger.info(f"  {r['candidate']:12s} margin_ratio={r['margin_ratio']:.3f} passed={r['passed']}")

    full = {
        "exact": exact_rows,
        "fibermap": fiber_rows,
        "bit_budgets": budgets,
        "forced_accept": forced_rows,
        "montecarlo": mcblock,
        "matched_verdicts": matched,
        "screen_ranking": payload,
        "wall_seconds": time.perf_counter() - t0,
    }
    (RESULTS / "screen_results.json").write_text(json.dumps(full, indent=2, default=str))
    logger.info(f"SCREEN complete in {time.perf_counter() - t0:.1f}s")
    return full


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(str(ROOT / "logs" / "screen.log"), rotation="30 MB", level="DEBUG")
    logger.catch(reraise=True)(main)()
