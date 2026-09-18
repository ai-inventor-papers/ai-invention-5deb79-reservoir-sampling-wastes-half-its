#!/usr/bin/env python3
"""Determinism check: re-run the SCREEN panel and assert the frozen hash reproduces.

The exact DP and the transportation flow have no randomness at all, so their outputs are
bit-identical by construction; the Monte-Carlo blocks are seeded explicitly (PCG64, with
the per-candidate offset derived by crc32 rather than Python's per-process-randomised
``hash``).  The recorded sha256 covers everything in screen_ranking.json except wall-clock
timing, so a matching hash across two independent runs certifies both.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import run_screen  # noqa: E402


def main() -> None:
    first = json.loads(run_screen.SCREEN_RANKING.read_text())
    backup = ROOT / "results" / "screen_ranking_run1.json"
    shutil.copy(run_screen.SCREEN_RANKING, backup)

    run_screen.main()
    second = json.loads(run_screen.SCREEN_RANKING.read_text())

    h1 = first["sha256_of_this_file_content"]
    h2 = second["sha256_of_this_file_content"]
    logger.info(f"run 1 sha256 = {h1}")
    logger.info(f"run 2 sha256 = {h2}")
    if h1 != h2:
        for key in sorted(set(first) | set(second)):
            if key in ("wall_seconds", "sha256_of_this_file_content"):
                continue
            if first.get(key) != second.get(key):
                logger.error(f"field differs across runs: {key}")
        raise AssertionError("SCREEN panel is NOT deterministic")
    logger.info("SCREEN panel is deterministic: identical hash across two independent runs")
    (ROOT / "results" / "determinism_check.json").write_text(
        json.dumps({"run1_sha256": h1, "run2_sha256": h2, "identical": True}, indent=2)
    )


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.catch(reraise=True)(main)()
