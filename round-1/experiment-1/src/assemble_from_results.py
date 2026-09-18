#!/usr/bin/env python3
"""Rebuild method_out.json from the cached results/ artefacts, without re-running anything.

Used to iterate on the output assembly (and to re-validate it) after an expensive run has
already produced ``results/screen_results.json`` and ``results/heldout_results.json``.
``method.py`` performs the same assembly at the end of a full run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import method  # noqa: E402


def main() -> None:
    screen = json.loads((ROOT / "results" / "screen_results.json").read_text())
    heldout = json.loads((ROOT / "results" / "heldout_results.json").read_text())
    gates = json.loads((ROOT / "results" / "gates.json").read_text()) if (
        ROOT / "results" / "gates.json"
    ).exists() else {"passed": None, "note": "gate suite not re-run in this assembly pass"}
    method.assemble(screen, heldout, gates)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.catch(reraise=True)(main)()
