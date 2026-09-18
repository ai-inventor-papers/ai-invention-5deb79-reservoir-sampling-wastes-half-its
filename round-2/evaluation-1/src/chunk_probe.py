#!/usr/bin/env python3
"""Is a single-run nominal co-inclusion false alarm a property of the CELL, of the SEED,
or of the trial-CHUNK size (i.e. of how the PCG64 stream is partitioned)?

Algorithm R is provably uniform, so every p below is a false alarm when it is small.
"""
from __future__ import annotations

import json
import sys
import time
from math import comb
from pathlib import Path

import numpy as np
from loguru import logger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendored"))
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")

import e2_montecarlo as M

CELLS = [(1000, 50), (2000, 100), (200, 40)]
CHUNKS = [10_000, 16_326, 25_000, 50_000]
SEEDS = [0, 1, 2, 3]
T = 200_000
BASE = 424_242


def main() -> None:
    rows = []
    for (n, k) in CELLS:
        base_seed = BASE + 17 * n + k
        for chunk in CHUNKS:
            t0 = time.perf_counter()
            r = M.run_mc("S0", n=n, k=k, T=T, seed=base_seed, chunk=chunk)
            rows.append({"n": n, "k": k, "vary": "chunk", "chunk": chunk,
                         "seed": base_seed, "coinc_p": r["coinc_p"],
                         "chi2_over_df": r["coinc_chi2_over_df"],
                         "false_alarm": bool(r["coinc_p"] < 1e-6),
                         "wall_s": time.perf_counter() - t0})
            logger.info(f"n={n} k={k} chunk={chunk} seed={base_seed} "
                        f"p={r['coinc_p']:.3g} chi2/df={r['coinc_chi2_over_df']:.5f}")
        for sd in SEEDS:
            t0 = time.perf_counter()
            r = M.run_mc("S0", n=n, k=k, T=T, seed=base_seed + 1000 * (sd + 1), chunk=50_000)
            rows.append({"n": n, "k": k, "vary": "seed", "chunk": 50_000,
                         "seed": base_seed + 1000 * (sd + 1), "coinc_p": r["coinc_p"],
                         "chi2_over_df": r["coinc_chi2_over_df"],
                         "false_alarm": bool(r["coinc_p"] < 1e-6),
                         "wall_s": time.perf_counter() - t0})
            logger.info(f"n={n} k={k} chunk=50000 seed={base_seed + 1000*(sd+1)} "
                        f"p={r['coinc_p']:.3g} chi2/df={r['coinc_chi2_over_df']:.5f}")

    summary = {}
    for (n, k) in CELLS:
        ch = [r for r in rows if r["n"] == n and r["k"] == k and r["vary"] == "chunk"]
        se = [r for r in rows if r["n"] == n and r["k"] == k and r["vary"] == "seed"]
        summary[f"n{n}_k{k}"] = {
            "chunk_sweep_p_values": [r["coinc_p"] for r in ch],
            "chunk_sweep_chi2_over_df": [r["chi2_over_df"] for r in ch],
            "chunk_sweep_n_false_alarms": sum(r["false_alarm"] for r in ch),
            "seed_sweep_p_values": [r["coinc_p"] for r in se],
            "seed_sweep_chi2_over_df": [r["chi2_over_df"] for r in se],
            "seed_sweep_n_false_alarms": sum(r["false_alarm"] for r in se),
            "p_spans_orders_of_magnitude_across_chunks": float(
                np.log10(max(max(r["coinc_p"] for r in ch), 1e-300))
                - np.log10(max(min(r["coinc_p"] for r in ch), 1e-300))),
        }
    out = {"rows": rows, "summary": summary, "T": T,
           "chunks_tried": CHUNKS, "seeds_tried": SEEDS,
           "interpretation": (
               "Algorithm R is provably uniform, so every small p here is a false alarm. "
               "If the p value swings by many orders of magnitude when only the trial-chunk "
               "size changes -- with the SEED held fixed -- then a single-run nominal p is "
               "an artifact of how the PCG64 stream is partitioned, not a property of the "
               "(n,k) cell, and cannot on its own justify declaring the nominal rule "
               "mis-calibrated.")}
    (HERE / "results" / "stage_chunkprobe.json").write_text(json.dumps(out, indent=1,
                                                                      default=float))
    logger.info(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
