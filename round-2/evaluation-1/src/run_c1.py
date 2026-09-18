#!/usr/bin/env python3
"""Part C1 with per-cell checkpointing and a bounded null budget.

Rewritten after the previous attempt died: (a) the vendored rejection sampler raised
RuntimeError at (500,50) / (2000,100) because its 200-pass cap was too small (fixed in
vendored/e2_nullband.py), and (b) the null budget null_B=200 made the (2000,100) band
cost 1620 s alone.  This runner uses null_B=40 / null_B_coinc=16 and writes a partial
checkpoint after EVERY cell, so no wall-clock accident can lose the whole stage.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from loguru import logger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(HERE / "logs" / "partC.log"), rotation="30 MB", level="DEBUG")

import partC as C

T = 200_000
NULL_B, NULL_B_COINC = 40, 16
SEED = 424_242
GRID = [(n, k) for n in (200, 500, 1000, 2000, 5000)
        for k in (5, 10, 15, 20, 25, 30, 40, 50, 75, 100) if k <= n // 4]
CAL_CELLS = [(1000, 10), (1000, 50), (500, 50), (2000, 100)]
DEST = HERE / "results" / "stage_c1.json"
PART = HERE / "results" / "stage_c1_partial.json"


def checkpoint(nominal: list[dict], cal: list[dict], t0: float, path: Path) -> dict:
    out = {"nominal_sweep": nominal, "calibrated_cells": cal,
           "summary": C.c1_summary(nominal, cal),
           "grid": [list(g) for g in GRID],
           "calibrated_cell_list": [list(c) for c in CAL_CELLS],
           "T": T, "null_B": NULL_B, "null_B_coinc": NULL_B_COINC, "seed": SEED,
           "cells_completed": sorted({(r["n"], r["k"]) for r in cal}.__iter__()),
           "wall_seconds": time.perf_counter() - t0}
    path.write_text(json.dumps(out, indent=1, default=float))
    return out


def main() -> None:
    t0 = time.perf_counter()
    nominal: list[dict] = []
    for cell in GRID:
        nominal += C.c1_nominal_sweep(T=T, grid=[cell], seed=SEED)
        checkpoint(nominal, [], t0, PART)
    cal: list[dict] = []
    for cell in CAL_CELLS:
        cal += C.c1_calibrated_cells(T=T, cells=[cell], null_B=NULL_B,
                                     null_B_coinc=NULL_B_COINC, seed=SEED)
        checkpoint(nominal, cal, t0, PART)
    out = checkpoint(nominal, cal, t0, DEST)
    logger.info(f"stage c1 written in {out['wall_seconds']:.1f}s")


if __name__ == "__main__":
    main()
