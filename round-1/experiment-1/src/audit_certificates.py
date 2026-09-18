#!/usr/bin/env python3
"""Standalone audit of every shipped fiber-map certificate.

Re-reads each file from disk and re-verifies it with ``fibermap.check_certificate``,
which shares no state with the max-flow solver that produced it: every source appears
exactly once, every target is a genuine child of its source, and every target's load is
exactly (i-k+1)/k.  Also re-derives the sha256 recorded in the file.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import fibermap  # noqa: E402


def main() -> None:
    files = sorted((ROOT / "certificates").glob("cert_i*_k*.json"))
    if not files:
        raise FileNotFoundError("no certificates to audit")
    ok = 0
    report = []
    for f in files:
        d = json.loads(f.read_text())
        cert = [(tuple(s), tuple(t)) for s, t in d["map"]]
        chk = fibermap.check_certificate(cert, d["i"], d["k"])
        h = fibermap.certificate_hash(cert)
        good = chk["valid"] and h == d["sha256"]
        ok += good
        report.append(
            {"file": f.name, "i": d["i"], "k": d["k"], "n_rows": len(cert),
             "per_target_load": d["per_target_load"], "checker_valid": chk["valid"],
             "sha256_matches": h == d["sha256"], "errors": chk["errors"]}
        )
        logger.info(
            f"{f.name:20s} i={d['i']:2d} k={d['k']} rows={len(cert):5d} "
            f"load={d['per_target_load']} valid={chk['valid']} hash_ok={h == d['sha256']}"
        )
    (ROOT / "results" / "certificate_audit.json").write_text(
        json.dumps({"n_certificates": len(files), "n_valid": ok, "rows": report}, indent=2)
    )
    logger.info(f"{ok}/{len(files)} certificates re-verified from disk")
    if ok != len(files):
        raise AssertionError("a shipped certificate failed independent re-verification")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.catch(reraise=True)(main)()
