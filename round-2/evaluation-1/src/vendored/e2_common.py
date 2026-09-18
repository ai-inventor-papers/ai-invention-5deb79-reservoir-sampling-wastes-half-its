#!/usr/bin/env python3
"""Shared plumbing: logging, resource limits, JSON sanitising, checkpoints, hashing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
LOGS = ROOT / "logs"
RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

_STAGE_TIMES: dict[str, float] = {}


def setup_logging(tag: str) -> None:
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(LOGS / "run.log", rotation="30 MB", level="DEBUG", enqueue=False)
    logger.info(f"=== stage {tag} starting ===")


def detect_cpus() -> int:
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError, IndexError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


def container_ram_gb() -> float | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = Path(p).read_text().strip()
            if v != "max" and int(v) < 1_000_000_000_000:
                return int(v) / 1e9
        except (FileNotFoundError, ValueError):
            pass
    return None


def set_limits(ram_gb: float = 24.0) -> dict:
    """Hard RAM cap so an accidental huge allocation raises MemoryError."""
    nbytes = int(ram_gb * 1024**3)
    try:
        resource.setrlimit(resource.RLIMIT_AS, (nbytes, nbytes))
    except (ValueError, OSError) as exc:  # pragma: no cover - platform dependent
        logger.warning(f"could not set RLIMIT_AS: {exc}")
    return {"rlimit_as_gb": ram_gb}


def manifest(cost_usd: float = 0.0) -> dict:
    import mpmath
    import scipy

    try:
        git = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        git = ""
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mpmath": mpmath.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpus_detected": detect_cpus(),
        "container_ram_gb": container_ram_gb(),
        "git": git,
        "cost_usd": cost_usd,
        "llm_calls": 0,
        "note": "No LLM, no GPU and no network are used anywhere in this artifact.",
    }


def sanitize(obj: Any) -> Any:
    """JSON-safe: numpy scalars -> python, non-finite floats -> 'inf'/'-inf'/'nan'."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return sanitize(obj.tolist())
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    return obj


def dump_json(path: Path, payload: Any) -> Path:
    path.write_text(json.dumps(sanitize(payload), indent=2, sort_keys=False))
    logger.info(f"wrote {path.name} ({path.stat().st_size / 1024:.1f} KiB)")
    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def content_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(sanitize(payload), sort_keys=True).encode("utf-8")
    ).hexdigest()


def checkpoint(name: str, payload: Any) -> Path:
    """Write a partial result so a timeout still ships data."""
    return dump_json(RESULTS / f"partial_{name}.json", payload)


class stage:
    """Context manager that logs and records a stage's wall time."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __enter__(self) -> "stage":
        self.t0 = time.time()
        logger.info(f"--- {self.name}: start")
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        dt = time.time() - self.t0
        _STAGE_TIMES[self.name] = round(dt, 2)
        if exc_type is None:
            logger.info(f"--- {self.name}: done in {dt:.1f}s")
        else:
            logger.error(f"--- {self.name}: FAILED after {dt:.1f}s ({exc_type.__name__}: {exc})")
        return False


def stage_times() -> dict[str, float]:
    return dict(_STAGE_TIMES)


def gate(name: str, ok: bool, detail: str) -> dict:
    (logger.info if ok else logger.error)(f"GATE {name}: {'PASS' if ok else 'FAIL'} -- {detail}")
    return {"gate": name, "pass": bool(ok), "detail": detail}
