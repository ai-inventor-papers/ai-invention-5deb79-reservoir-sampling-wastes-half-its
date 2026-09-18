#!/usr/bin/env python3
"""HELD-OUT driver.

REFUSES to touch any held-out configuration unless ``screen_ranking.json`` exists and
its recorded sha256 matches its own content.  Only the top-ranked survivor of the screen
is confirmed on the held-out exact panel; every other candidate stays at screen scale and
the output says so per candidate.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import forced  # noqa: E402
import montecarlo as mc  # noqa: E402
import panels  # noqa: E402
import verdicts  # noqa: E402
from run_screen import (  # noqa: E402
    RESULTS,
    SCREEN_RANKING,
    key_seed,
    run_budget_block,
    run_exact_block,
    run_fiber_block,
    run_forced_block,
    run_mc_block,
)


class ScreenNotFrozen(RuntimeError):
    """Raised when the held-out panel is attempted without a verified frozen screen."""


def verify_frozen_screen() -> dict[str, Any]:
    if not SCREEN_RANKING.exists():
        raise ScreenNotFrozen(
            f"{SCREEN_RANKING.name} is absent -- the screen panel must be frozen to disk "
            "before ANY held-out configuration executes.  Run run_screen.py first."
        )
    payload = json.loads(SCREEN_RANKING.read_text())
    recorded = payload.pop("sha256_of_this_file_content", None)
    if recorded is None:
        raise ScreenNotFrozen("screen_ranking.json carries no content hash")
    wall = payload.pop("wall_seconds", None)
    payload.pop("hash_excludes", None)
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    actual = hashlib.sha256(body.encode()).hexdigest()
    if actual != recorded:
        raise ScreenNotFrozen(
            f"screen_ranking.json hash mismatch: recorded {recorded}, recomputed {actual}. "
            "The frozen screen has been edited -- refusing to run held-out."
        )
    payload["sha256_of_this_file_content"] = recorded
    payload["wall_seconds"] = wall
    logger.info(f"frozen screen VERIFIED  sha256={recorded}")
    return payload


def main() -> dict[str, Any]:
    RESULTS.mkdir(exist_ok=True)
    screen = verify_frozen_screen()
    survivors = [r for r in screen["ranking"] if r["passed"]]
    top = survivors[0]["candidate"] if survivors else None
    logger.info("=" * 78)
    logger.info(f"HELD-OUT PANEL   top-ranked screen survivor: {top}")
    logger.info("=" * 78)

    seed = panels.HELDOUT["seed"]
    t0 = time.perf_counter()

    exact_rows = run_exact_block(panels.HELDOUT["exact_dp"]["main"], label="heldout")
    fiber_rows = run_fiber_block(panels.HELDOUT["fibermap"]["cells"], label="heldout")
    budgets = run_budget_block(panels.HELDOUT["bit_budgets"]["cells"])
    forced_rows = run_forced_block(panels.HELDOUT["forced_accept"]["cells"])
    mcblock = run_mc_block(panels.HELDOUT["montecarlo"], seed, label="heldout")

    # The literal user-facing deliverable, isolated as a single headline block.
    n, k, T = panels.HELDOUT["montecarlo"]["n"], panels.HELDOUT["montecarlo"]["k"], panels.HELDOUT["montecarlo"]["T"]
    band = mcblock["null_band_simulated"]
    deliverable = {
        "question": (
            "Implement a reservoir sampler that draws k items uniformly from a stream of "
            "unknown length, verify uniformity empirically over many independent trials, "
            "and report the maximum deviation from the expected inclusion frequency."
        ),
        "configuration": {"n": n, "k": k, "trials_T": T, "expected_frequency_k_over_n": k / n,
                          "null_replicates_R": panels.HELDOUT["montecarlo"]["R"]},
        "null_band_first_order_max_deviation": band["first_order"],
        "null_band_analytic_gumbel": mcblock["null_band_analytic_gumbel"],
        "headline": [
            {
                "candidate": r["candidate"],
                "max_deviation_from_k_over_n": r["first_order"]["max_dev"],
                "max_deviation_relative_to_expected": r["first_order"]["max_dev"] / (k / n),
                "max_deviation_in_null_band_p95_units": r["max_dev_in_band_units"],
                "flagged_by_first_order_test": r["flagged_first_order"],
                "pairwise_max_abs_z": r["pair"]["max_abs_z"],
                "pairwise_p_value_bonferroni": r["pair_p_value_normal"],
                "pairwise_log10_p_value_bonferroni": r["pair_log10_p_value"],
                "flagged_by_pairwise_test": r["flagged_second_order"],
                "fraction_of_pairs_never_co_sampled": r["pair"]["frac_zero_pairs"],
            }
            for r in mcblock["rows"]
        ],
    }

    payload = {
        "screen_ranking_sha256": screen["sha256_of_this_file_content"],
        "top_ranked_screen_survivor": top,
        "panel_definition": {
            "exact_dp": panels.HELDOUT["exact_dp"],
            "fibermap_cells": panels.HELDOUT["fibermap"]["cells"],
            "montecarlo": panels.HELDOUT["montecarlo"],
            "bit_budgets": panels.HELDOUT["bit_budgets"]["cells"],
            "forced_accept": panels.HELDOUT["forced_accept"]["cells"],
            "seed": seed,
        },
        "exact": exact_rows,
        "fibermap": fiber_rows,
        "bit_budgets": budgets,
        "forced_accept": forced_rows,
        "montecarlo": mcblock,
        "literal_deliverable": deliverable,
        "wall_seconds": time.perf_counter() - t0,
    }
    (RESULTS / "heldout_results.json").write_text(json.dumps(payload, indent=2, default=str))
    logger.info(f"HELD-OUT complete in {time.perf_counter() - t0:.1f}s")
    for h in deliverable["headline"]:
        logger.info(
            f"  DELIVERABLE {h['candidate']:11s} max_dev={h['max_deviation_from_k_over_n']:.6f} "
            f"({h['max_deviation_in_null_band_p95_units']:.2f}x null p95) "
            f"flagged_1st={h['flagged_by_first_order_test']} "
            f"pair_z={h['pairwise_max_abs_z']:.1f} flagged_pair={h['flagged_by_pairwise_test']}"
        )
    return payload


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(str(ROOT / "logs" / "heldout.log"), rotation="30 MB", level="DEBUG")
    logger.catch(reraise=True)(main)()
