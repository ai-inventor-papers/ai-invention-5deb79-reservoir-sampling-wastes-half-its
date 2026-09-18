#!/usr/bin/env python
"""Persistent Lean worker: warms Mathlib once, then serves job files.

Drop  jobs/<name>.lean            -> writes jobs/<name>.json   (run result)
Drop  jobs/<name>.suggest.json    -> writes jobs/<name>.suggest.out.json
      (content: {"code": "...", "tactics": "exact?,simp"} )
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

SKILL = Path("/ai-inventor/.claude/skills/aii-lean/scripts")
WS = Path(__file__).resolve().parent
JOBS = WS / "jobs"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SKILL / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    runm = _load("aii_run_lean")
    sugm = _load("aii_lean_suggest")
    print("initializing lean (downloads elan + mathlib on first run)...", flush=True)
    t0 = time.time()
    runm.init_run_lean()
    sugm._config = runm._config
    print(f"READY after {time.time() - t0:.1f}s", flush=True)
    (WS / "logs" / "lean_ready").write_text("ready\n")
    while True:
        for job in sorted(JOBS.glob("*.lean")):
            out = job.with_suffix(".json")
            if out.exists():
                continue
            t = time.time()
            try:
                res = runm.core_run_lean(code=job.read_text())
            except Exception as e:  # noqa: BLE001
                res = {"success": False, "verified": False, "errors": [repr(e)]}
            res["wall_clock_s"] = round(time.time() - t, 2)
            out.write_text(json.dumps(res, indent=2))
            print(f"ran {job.name} verified={res.get('verified')} {res['wall_clock_s']}s", flush=True)
        for job in sorted(JOBS.glob("*.suggest.json")):
            out = job.with_name(job.name.replace(".suggest.json", ".suggest.out.json"))
            if out.exists():
                continue
            spec = json.loads(job.read_text())
            try:
                res = sugm.core_lean_suggest(code=spec["code"], tactics=spec.get("tactics", sugm.DEFAULT_TACTICS))
            except Exception as e:  # noqa: BLE001
                res = {"success": False, "errors": [repr(e)]}
            out.write_text(json.dumps(res, indent=2))
            print(f"suggested {job.name}", flush=True)
        time.sleep(1.0)


if __name__ == "__main__":
    main()
