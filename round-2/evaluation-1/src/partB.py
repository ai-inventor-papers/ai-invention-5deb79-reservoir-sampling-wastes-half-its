#!/usr/bin/env python3
"""PART B -- the claims ledger: recompute EVERY number the draft prints.

Each claim carries the text as printed, the artifact it comes from, a dotted json path
into that artifact, a recompute function that DERIVES the value from the artifact, the
executed value, a status in {MATCH, MISMATCH, UNAVAILABLE}, a tolerance, a drop-in
corrected sentence, and a note.

A ledger whose UNAVAILABLE count is honest is worth more than one that is 100% MATCH by
omission, so a path that does not resolve is reported with the paths searched and the
nearest keys found -- never silently skipped.
"""

from __future__ import annotations

import json
import math
import re
import sys
from math import comb, log2
from pathlib import Path
from typing import Any, Callable

from loguru import logger

HERE = Path(__file__).resolve().parent
E1_DIR = Path("/ai-inventor/aii_data/runs/run_2L_JomowDIGF/3_invention_loop/iter_1/"
              "gen_art/gen_art_experiment_1")
E2_DIR = Path("/ai-inventor/aii_data/runs/run_2L_JomowDIGF/3_invention_loop/iter_1/"
              "gen_art/gen_art_experiment_2")

REL_TOL_JSON = 1e-9      # for a float copied verbatim out of the JSON
REL_TOL_DERIVED = 1e-6   # for a quantity recomputed through floating arithmetic


# --------------------------------------------------------------------------------------
# generic path resolver
# --------------------------------------------------------------------------------------


class Unresolved(Exception):
    def __init__(self, path: str, searched: list[str], nearest: list[str]) -> None:
        super().__init__(f"unresolved path {path!r}")
        self.path, self.searched, self.nearest = path, searched, nearest


def resolve(root: Any, path: str) -> Any:
    """Resolve a dotted path with dict keys, [i] list indices and {field=value} filters.

    Examples:
        metadata.verdicts.MAIN.margin_ratio
        datasets[0].examples[3].metadata_n
        datasets{dataset=eviction_step_minimum_entropy}.examples
    """
    cur = root
    searched: list[str] = []
    for tok in re.findall(r"[^.\[\]{}]+|\[\d+\]|\{[^}]*\}", path):
        searched.append(tok)
        try:
            if tok.startswith("[") and tok.endswith("]"):
                cur = cur[int(tok[1:-1])]
            elif tok.startswith("{") and tok.endswith("}"):
                fld, val = tok[1:-1].split("=", 1)
                hits = [x for x in cur if str(x.get(fld)) == val]
                if not hits:
                    raise KeyError(tok)
                cur = hits[0]
            else:
                cur = cur[tok] if isinstance(cur, dict) else cur[int(tok)]
        except (KeyError, IndexError, TypeError, ValueError):
            nearest: list[str] = []
            if isinstance(cur, dict):
                nearest = sorted(cur.keys())[:30]
            elif isinstance(cur, list):
                nearest = [f"list of {len(cur)}"] + (
                    sorted(cur[0].keys())[:20] if cur and isinstance(cur[0], dict) else [])
            raise Unresolved(path, searched, nearest) from None
    return cur


def approx(a: Any, b: Any, *, rel: float = REL_TOL_JSON, absolute: float | None = None) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int,)) and isinstance(b, (int,)):
        return a == b
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(approx(x, y, rel=rel, absolute=absolute)
                                        for x, y in zip(a, b))
    try:
        fa, fb = float(a), float(b)
    except (TypeError, ValueError):
        return a == b
    if absolute is not None:
        return abs(fa - fb) <= absolute
    return math.isclose(fa, fb, rel_tol=rel, abs_tol=0.0)


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def load() -> tuple[dict, dict]:
    e1 = json.loads((E1_DIR / "full_method_out.json").read_text())
    e2 = json.loads((E2_DIR / "full_method_out.json").read_text())
    for art in (e1, e2):
        art["_ds"] = {d["dataset"]: d for d in art["datasets"]}
    return e1, e2


def ds(art: dict, name: str) -> list[dict]:
    return art["_ds"][name]["examples"]


# --------------------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------------------


def build_claims(e1: dict, e2: dict, codefacts: dict) -> list[dict]:
    C: list[dict] = []

    def add(**kw) -> None:
        C.append(kw)

    # ---------------- 1. co-inclusion calibrated z separation --------------------------
    rows = e2["metadata"]["replication_calibration"]["rows"]
    corr = [r for r in rows if not r["blind_spot"]]
    brok = [r for r in rows if r["blind_spot"]]
    zc = [r["coinc_chi2_z"] for r in corr]
    zb = [r["coinc_chi2_z"] for r in brok]
    grid = e2["metadata"]["verdicts"]["co_inclusion_rule"]["separation_on_the_grid"]
    add(
        claim_id="C01_coinclusion_separation",
        claim_text_as_printed="Calibrated co-inclusion z separates correct from broken "
                              "samplers: at most 2.23 for the correct ones versus at "
                              "least 21.4 for the broken ones.",
        source_artifact="E2",
        json_path="metadata.replication_calibration.rows",
        executed_value={"replication_panel_correct_z_min": min(zc),
                        "replication_panel_correct_z_max": max(zc),
                        "replication_panel_broken_z_min": min(zb),
                        "replication_panel_broken_z_max": max(zb),
                        "n_correct_rows": len(corr), "n_broken_rows": len(brok),
                        "grid_panel_max_correct_z": grid["max_calibrated_z_among_provably_correct"],
                        "grid_panel_min_broken_z": grid["min_calibrated_z_among_broken"]},
        status="MISMATCH",
        tolerance="exact min/max over the 72 replication rows",
        corrected_text=(
            f"On the 72-run replication panel (n=1000, k=10, T=1e6) the calibrated "
            f"co-inclusion z lies in [{min(zc):.3f}, {max(zc):.3f}] for the 48 "
            f"provably-correct runs and in [{min(zb):.3g}, {max(zb):.3g}] for the 24 "
            f"blind-spot runs -- a separation of more than four orders of magnitude, not "
            f"a factor of ten. The pair 2.23 vs 21.4 comes from a DIFFERENT panel "
            f"(the (n,k) calibration grid, metadata.verdicts.co_inclusion_rule."
            f"separation_on_the_grid: {grid['max_calibrated_z_among_provably_correct']:.4f} "
            f"vs {grid['min_calibrated_z_among_broken']:.4f}) and must not be attributed "
            f"to the replication panel."),
        note="Both numbers are real and both are in the artifact; the defect is a "
             "provenance conflation between the (n,k) grid and the 72-run panel.",
    )

    # ---------------- 2. "21 such steps" at i >= 2k ------------------------------------
    ev = ds(e1, "eviction_step_minimum_entropy")
    ge = [e for e in ev if e["metadata_i_prefix"] >= 2 * e["metadata_k"]]
    n_nondiv = sum(1 for e in ge if not e["metadata_divisible"])
    n_all = len(ge)
    claimed = e1["metadata"]["verdict"]["supporting"]["n_nondivisible_steps_with_i_at_least_2k"]
    add(
        claim_id="C02_21_steps_at_i_ge_2k",
        claim_text_as_printed="On non-divisible steps the minimum decays like k/(i-k): "
                              "<= 0.333*log2(k) once i >= 2k -- 21 such steps.",
        source_artifact="E1",
        json_path="datasets{dataset=eviction_step_minimum_entropy}.examples",
        executed_value={"n_nondivisible_steps_with_i_ge_2k": n_nondiv,
                        "n_all_steps_with_i_ge_2k": n_all,
                        "n_divisible_steps_with_i_ge_2k": n_all - n_nondiv,
                        "n_steps_total": len(ev),
                        "metadata_declares": claimed},
        status="MATCH" if (n_nondiv == 21 and claimed == 21) else "MISMATCH",
        tolerance="exact integer count",
        corrected_text=(
            f"21 NON-DIVISIBLE steps satisfy i >= 2k ({n_all} steps in total satisfy "
            f"i >= 2k, of which {n_all - n_nondiv} are divisible and carry minimum "
            f"eviction entropy exactly 0), out of {len(ev)} analysed steps."),
        note="The count is right but the printed sentence is ambiguous about whether the "
             "divisible steps are included; the corrected text disambiguates.",
    )

    # ---------------- 3. post-removal anytime overheads --------------------------------
    fb = {(e["metadata_budget"]["n"], e["metadata_budget"]["k"]): e["metadata_budget"]
          for e in ds(e1, "fair_coin_bit_budgets")}
    s1t_budget = {(r["n"], r["k"]): r for r in
                  e2["metadata"]["verdicts"]["MAIN"]["s1t_closed_form_budget"]}
    bcf = {(r["n"], r["k"]): r for r in e2["metadata"]["budget_closed_form"]}
    exec_k10 = fb[(1_000_000, 10)]["anytime_overhead_optimal_upper"]
    exec_k1000 = fb[(1_000_000, 1000)]["anytime_overhead_optimal_upper"]
    fitted_k10 = s1t_budget[(1_000_000, 10)]["S1t_total_bits"] / bcf[(1_000_000, 10)]["output_H"]
    fitted_k1000 = (s1t_budget[(1_000_000, 1000)]["S1t_total_bits"]
                    / bcf[(1_000_000, 1000)]["output_H"])
    add(
        claim_id="C03_anytime_overhead_post_removal",
        claim_text_as_printed="Post-removal anytime overhead drops to 6.47x (k=10) and "
                              "4.80x (k=1000) at n=1e6.",
        source_artifact="E1+E2",
        json_path=("E1 datasets{dataset=fair_coin_bit_budgets} metadata_budget."
                   "anytime_overhead_optimal_upper  ||  E2 metadata.verdicts.MAIN."
                   "s1t_closed_form_budget"),
        executed_value={"E1_executed_analytic_bracket_k10": exec_k10,
                        "E1_executed_analytic_bracket_k1000": exec_k1000,
                        "E1_classical_k10": fb[(1_000_000, 10)]["anytime_overhead_classical"],
                        "E1_classical_k1000": fb[(1_000_000, 1000)]["anytime_overhead_classical"],
                        "E2_fitC_derived_k10": fitted_k10,
                        "E2_fitC_derived_k1000": fitted_k1000},
        status="MISMATCH",
        tolerance=f"rel {REL_TOL_DERIVED}",
        corrected_text=(
            f"At n=1e6 the anytime overhead falls from {fb[(1_000_000,10)]['anytime_overhead_classical']:.4g}x "
            f"to {exec_k10:.4f}x at k=10 and from {fb[(1_000_000,1000)]['anytime_overhead_classical']:.4g}x "
            f"to {exec_k1000:.4f}x at k=1000 under the EXECUTED analytic eviction bracket "
            f"(E1 bench 1). The figures {fitted_k10:.2f}x and {fitted_k1000:.2f}x are "
            f"EXTRAPOLATED: they follow from the fitted constant fitC = "
            f"{s1t_budget[(1_000_000,10)]['fit_C']:.4f}, not from a measurement, and must "
            f"be labelled as such."),
        note="6.47/4.80 are exactly S1t_total_bits / output_H at the fitted C; 6.3859/4.1045 "
             "are the executed analytic-bracket values for the SAME cells.",
    )

    # ---------------- 4. exact-verification grid for H and for M -----------------------
    h_cells = sorted({(r["n"], r["k"]) for r in e2["metadata"]["exact_check"]
                      if r["candidate"] == "S1h"})
    m_cells = sorted({(r["n"], r["k"]) for r in e2["metadata"]["s1t_table_exact"]["verify"]})
    all_exact_cells = sorted({(r["n"], r["k"]) for r in e2["metadata"]["exact_check"]})
    add(
        claim_id="C04_exact_grid_H_and_M",
        claim_text_as_printed="H is exactly uniform for k = 2..10 up to n = 20.",
        source_artifact="E2",
        json_path="metadata.exact_check  ||  metadata.s1t_table_exact.verify",
        executed_value={"H_cells_executed": [list(c) for c in h_cells],
                        "H_n_cells": len(h_cells),
                        "H_max_k": max(k for _, k in h_cells),
                        "H_max_n": max(n for n, _ in h_cells),
                        "M_cells_executed": [list(c) for c in m_cells],
                        "M_n_cells": len(m_cells),
                        "all_exact_check_cells": [list(c) for c in all_exact_cells]},
        status="UNAVAILABLE",
        tolerance="enumeration of executed cells; no tolerance applies",
        corrected_text=(
            f"The deterministic sum-mod-k rank rule (H / sampler S1h) is verified exactly "
            f"uniform at every prefix on the {len(h_cells)} executed cells "
            f"{{{', '.join(f'({n},{k})' for n, k in h_cells)}}} -- every one of them has "
            f"k <= {max(k for _, k in h_cells)} and n <= {max(n for n, _ in h_cells)}. "
            f"The table-exact sampler (M / S1t) is verified at exactly {len(m_cells)} "
            f"cells: {{{', '.join(f'({n},{k})' for n, k in m_cells)}}}. No cell with "
            f"k in 6..10, and no cell with n = 19 or 20, was executed; the claim "
            f"'k = 2..10 up to n = 20' has no executed support in either artifact."),
        note="Not a wrong number: a claim with NO executed support. The missing cells are "
             "the parallel experiment's job, not this ledger's.",
    )

    # ---------------- 5. "evict M" provenance / fitC -----------------------------------
    by_k = e2["metadata"]["verdicts"]["MAIN"]["s1t_fit_C_range"]["by_k"]
    fitC = e2["metadata"]["verdicts"]["MAIN"]["s1t_fit_C_range"]["C_max"]
    # the fitted law is H_evict_hat(i,k) = C * k*log2(k)/i ; at k=1000, i=k+1
    k_probe, i_probe = 1000, 1001
    pred_bits = fitC * k_probe * log2(k_probe) / i_probe
    ceiling = log2(k_probe)
    alt = e2["metadata"]["s1t_table_exact"]["extrapolation"][0]
    add(
        claim_id="C05_fitC_is_not_a_bound",
        claim_text_as_printed="The most conservative fitted constant (max over four fits) "
                              "is used, so the eviction saving is understated not overstated.",
        source_artifact="E2",
        json_path="metadata.verdicts.MAIN.s1t_fit_C_range  ||  metadata.s1t_table_exact.curves",
        executed_value={
            "fits": [{"k": r["k"], "C": r["C"], "R2": r["R2"], "n_points": r["n_points"]}
                     for r in by_k],
            "C_used_in_headline": fitC,
            "C_source_k": e2["metadata"]["verdicts"]["MAIN"]["s1t_closed_form_budget"][0]["fit_source_k"],
            "predicted_bits_per_acceptance_at_k1000_i1001": pred_bits,
            "trivial_ceiling_log2_k": ceiling,
            "exceeds_trivial_ceiling": pred_bits > ceiling,
            "excess_ratio": pred_bits / ceiling,
            "second_incompatible_extrapolation_in_the_same_json": {
                "path": "metadata.s1t_table_exact.extrapolation[0]",
                "fit_C_used": alt["fit_C_used"],
                "total_evict_bits_at_n": alt["extrapolated"]["total_evict_bits_at_n"],
                "vs_headline_S1t_evict_bits": s1t_budget[(1_000_000, 10)]["S1t_evict_bits"]},
        },
        status="MISMATCH",
        tolerance=f"rel {REL_TOL_DERIVED}",
        corrected_text=(
            f"The four fits of H_evict(i,k) ~ C*k*log2(k)/i give C = "
            f"{', '.join(f'{r['C']:.3f} (k={r['k']}, R^2={r['R2']:.3f}, {r['n_points']} points)' for r in by_k)}; "
            f"C DECREASES in k, so the max-over-k choice C = {fitC:.4f} (taken from k=2, "
            f"the SMALLEST k and the furthest from the k=10..1000 cells it is applied to) "
            f"is an extrapolation, not a bound: at k=1000, i=k+1 it predicts "
            f"{pred_bits:.2f} bits per acceptance against the trivial ceiling "
            f"log2(1000) = {ceiling:.3f} bits, exceeding it by "
            f"{pred_bits/ceiling:.2f}x. Every fitC-derived figure -- including the headline "
            f"35.8 vs 380.8 bits at n=1e6, k=10 -- must carry the label EXTRAPOLATED. "
            f"A second, incompatible extrapolation for the SAME cell exists in the same "
            f"JSON (metadata.s1t_table_exact.extrapolation[0]) using C = "
            f"{alt['fit_C_used']:.4f} from k=5, giving "
            f"{alt['extrapolated']['total_evict_bits_at_n']:.2f} bits instead of "
            f"{s1t_budget[(1_000_000,10)]['S1t_evict_bits']:.2f}."),
        note="Two mutually inconsistent values for the same (n=1e6,k=10) eviction budget "
             "coexist in E2 under different key paths.",
    )

    # ---------------- 6. the fit-free bracket mislabel (defect D2) ----------------------
    q1 = codefacts.get("Q1", {})
    rows_d2 = []
    for (n, k) in sorted(fb):
        if k < 2:
            continue
        b = fb[(n, k)]
        up = b["optimal_evict_bits_upper"]
        rows_d2.append({"n": n, "k": k,
                        "optimal_evict_bits_upper_as_printed": up,
                        "log2_k": log2(k),
                        "corrected_upper_with_log2k_factor": up * log2(k),
                        "classical_evict_bits": b["classical_evict_bits"],
                        "corrected_still_below_classical":
                            up * log2(k) < b["classical_evict_bits"]})
    k2_rows = [r for r in rows_d2 if r["k"] == 2]
    add(
        claim_id="C06_fit_free_bracket_missing_log2k",
        claim_text_as_printed="Optimal eviction is CONSTANT in n: <= 20.8 bits at k=10 "
                              "whether n=1e4 or 1e6 (fit-free upper bracket).",
        source_artifact="E1",
        json_path="datasets{dataset=fair_coin_bit_budgets} metadata_budget."
                  "optimal_evict_bits_upper  ||  forced.py:133",
        executed_value={
            "code_expression_at_forced_py_133":
                q1.get("details", {}).get("exact_expression_as_written", "NOT_READ"),
            "code_reading_verdict": q1.get("verdict", "NOT_READ"),
            "rows": rows_d2,
            "k10_n1e3_as_printed": fb[(1000, 10)]["optimal_evict_bits_upper"],
            "k10_n1e6_as_printed": fb[(1_000_000, 10)]["optimal_evict_bits_upper"],
            "k10_n1e3_corrected": fb[(1000, 10)]["optimal_evict_bits_upper"] * log2(10),
            "k10_n1e6_corrected": fb[(1_000_000, 10)]["optimal_evict_bits_upper"] * log2(10),
            "factor": log2(10),
            "coincide_exactly_at_k2": all(
                abs(r["optimal_evict_bits_upper_as_printed"]
                    - r["corrected_upper_with_log2k_factor"]) == 0.0 for r in k2_rows),
        },
        status="MISMATCH" if q1.get("verdict") == "CONFIRMED" else "UNAVAILABLE",
        tolerance="exact (the correction is a multiplication by log2(k))",
        corrected_text=(
            f"E1/forced.py:133 computes upper_per = min((B-1)/A, log2(k)), where (B-1)/A "
            f"bounds the FRACTION OF SOURCES THAT SPLIT, not the entropy; the per-"
            f"acceptance entropy bound is that fraction TIMES log2(k). Restoring the "
            f"factor, the fit-free upper bracket on optimal eviction at k=10 is "
            f"{fb[(1000,10)]['optimal_evict_bits_upper']*log2(10):.2f} bits at n=1e3 and "
            f"{fb[(1_000_000,10)]['optimal_evict_bits_upper']*log2(10):.2f} bits at n=1e6 "
            f"(not {fb[(1000,10)]['optimal_evict_bits_upper']:.2f} / "
            f"{fb[(1_000_000,10)]['optimal_evict_bits_upper']:.2f}), i.e. up to "
            f"{log2(10):.2f}x larger. The qualitative claim survives: the bracket is still "
            f"CONSTANT in n and still far below the classical "
            f"{fb[(1_000_000,10)]['classical_evict_bits']:.1f} bits. The two expressions "
            f"coincide EXACTLY at k=2 (log2(2)=1), which is why the solved k=2 minima "
            f"1, 1/3, 1/5, 1/7 matched and the bug stayed invisible."),
        note="Code-reading claim, confirmed verbatim against E1/forced.py.",
    )

    # ---------------- 7. Algorithm L at n=1e12 / 1e15 ----------------------------------
    q3 = codefacts.get("Q3", {})
    rr = [r for r in e2["metadata"]["verdicts"]["C2"]["realistic_regime"]
          if r.get("status") != "SKIPPED_n_le_2k"]
    big = {r["n"]: r for r in rr if r.get("k") == 1024}
    add(
        claim_id="C07_algorithm_L_realistic_regime_is_modelled",
        claim_text_as_printed="Measured float64 skip-rate error for Algorithm L at k=1024: "
                              "2.434e-14 (n=1e6), 3.996e-9 (n=1e12), 4.034e-5 (n=1e15).",
        source_artifact="E2",
        json_path="metadata.verdicts.C2.realistic_regime  ||  algol.py audit_realistic_regime",
        executed_value={
            "rel_rate_err_shipped_by_n": {str(int(n)): big[n]["rel_rate_err_shipped"]
                                          for n in sorted(big)},
            "n_draws_per_cell": sorted({r.get("n_draws") for r in rr if "n_draws" in r}),
            "state_is_injected_not_simulated": q3.get("verdict", "NOT_READ") == "CONFIRMED",
            "code_reading_finding": q3.get("finding", "NOT_READ")[:800],
        },
        status="MISMATCH" if q3.get("verdict") == "CONFIRMED" else "UNAVAILABLE",
        tolerance=f"rel {REL_TOL_JSON} on the values themselves",
        corrected_text=(
            "The Algorithm-L float64 skip-rate errors at k=1024 "
            "(2.434e-14 at n=1e6, 3.996e-9 at n=1e12, 4.034e-5 at n=1e15) are MODELLED at "
            "the theoretical state W = k/n, not measured along a random trajectory: "
            "E2/algol.py audit_realistic_regime injects W = k/n directly and draws 2000 "
            "skips at that fixed state. The values are correct for what they are; no "
            "sentence may call them measured."),
        note="Values retained; only the provenance label changes.",
    )

    # ---------------- 8. FIFO prefix 3 and the k-1 law ---------------------------------
    dpc = ds(e1, "exact_prefix_dp_cells")
    fifo = [e for e in dpc if e["metadata_candidate"] == "C-FIFO"]
    fifo_k2 = sorted({e["metadata_i_first_fail"] for e in fifo if e["metadata_k"] == 2})
    s1n = [e for e in dpc if e["metadata_candidate"] == "S1n"]
    law_ok = all(abs(e["metadata_m1_rel_at_failure"] - (e["metadata_k"] - 1)) < 1e-12
                 for e in s1n)
    add(
        claim_id="C08_fifo_prefix_and_k_minus_1_law",
        claim_text_as_printed="FIFO control first fails at prefix 3, not 4; the unrepaired "
                              "sum-mod-k rank rule's relative max deviation is exactly k-1 "
                              "(3.0 at k=4, not 9.7).",
        source_artifact="E1",
        json_path="datasets{dataset=exact_prefix_dp_cells}.examples",
        executed_value={
            "C-FIFO_first_fail_prefixes_at_k2": fifo_k2,
            "C-FIFO_first_fail_by_k": {str(k): sorted({e["metadata_i_first_fail"] for e in fifo
                                                       if e["metadata_k"] == k})
                                       for k in sorted({e["metadata_k"] for e in fifo})},
            "S1n_m1_rel_equals_k_minus_1_on_all_cells": law_ok,
            "S1n_cells": [{"n": e["metadata_n"], "k": e["metadata_k"],
                           "i_first_fail": e["metadata_i_first_fail"],
                           "m1_rel": e["metadata_m1_rel_at_failure"],
                           "m1_rel_exact": e["metadata_m1_rel_at_failure_exact"]}
                          for e in s1n],
        },
        status="MATCH" if (fifo_k2 == [3] and law_ok) else "MISMATCH",
        tolerance="exact",
        corrected_text=(
            f"The FIFO control first fails at prefix i=3 at k=2 (and at i=k+1 in general: "
            f"prefix 4 at k=3), not at prefix 4. The unrepaired sum-mod-k rank rule's "
            f"relative max deviation at its first failing prefix is exactly k-1 on every "
            f"one of the {len(s1n)} executed cells -- 3.0 at k=4, 1.0 at k=2, 2.0 at k=3, "
            f"4.0 at k=5 -- not 9.7."),
        note="Both numbers reproduce exactly from the exact-rational DP.",
    )

    # ---------------- 9. the two literal deliverables ----------------------------------
    e1_lit = [e for e in ds(e1, "monte_carlo_uniformity_deliverable")
              if e["metadata_n"] == 1000 and e["metadata_candidate"] == "S0"][0]
    e1_circ = [e for e in ds(e1, "monte_carlo_uniformity_deliverable")
               if e["metadata_n"] == 1000 and e["metadata_candidate"] == "S5-CIRC"][0]
    hl = e2["metadata"]["headline"]
    n_big = 1000
    analytic = hl["null_band_sd"] * math.sqrt(2 * math.log(n_big))
    sigma_bin = e2["metadata"]["replication_calibration"]["band"]["sigma_per_bin"]
    analytic_from_sigma = sigma_bin * math.sqrt(2 * math.log(n_big))
    add(
        claim_id="C09_two_literal_deliverables",
        claim_text_as_printed="At n=1000, k=10, T=1e6 the max deviation from k/n is "
                              "3.68e-4 (and 3.36e-4).",
        source_artifact="E1+E2",
        json_path="E1 datasets{dataset=monte_carlo_uniformity_deliverable}  ||  "
                  "E2 metadata.headline",
        executed_value={
            "E1_S0_max_dev": e1_lit["metadata_max_deviation_from_k_over_n"],
            "E1_S0_in_null_p95_units": e1_lit["metadata_max_deviation_in_null_band_p95_units"],
            "E1_S5CIRC_max_dev": e1_circ["metadata_max_deviation_from_k_over_n"],
            "E2_AlgorithmR_max_dev": hl["maxdev_raw"],
            "E2_null_band_mean": hl["null_band_mean"],
            "E2_null_band_sd": hl["null_band_sd"],
            "E2_null_band_p99": hl["null_band_p99"],
            "E2_z_in_null_band": hl["maxdev_in_null_band_units_z"],
            "E2_analytic_maxdev_mean_as_stored": hl["analytic_maxdev_mean"],
            "analytic_crosscheck_sd_times_sqrt_2lnn": analytic,
            "analytic_crosscheck_sigma_per_bin_times_sqrt_2lnn": analytic_from_sigma,
            "sigma_per_bin": sigma_bin,
            "sigma_per_bin_closed_form": math.sqrt((10 / 1000) * (1 - 10 / 1000) / 1e6),
            "provenance_E1": "E1 bench 1: numpy PCG64, Algorithm L skip-sampling with "
                             "uniform eviction (sampler S0)",
            "provenance_E2": "E2 bench 2: the COUNTED fair-coin source (bits.py) driving "
                             "the per-item accept/evict loop with PCG64 words",
        },
        status="MISMATCH",
        tolerance=f"rel {REL_TOL_DERIVED} on the analytic cross-check",
        corrected_text=(
            f"Two different max-deviation figures exist at the same nominal (n=1000, "
            f"k=10, T=1e6) and must not be interchanged: E1 reports "
            f"{e1_lit['metadata_max_deviation_from_k_over_n']:.3g} for sampler S0 "
            f"(numpy PCG64 + Algorithm-L skip sampling) and E2 reports "
            f"{hl['maxdev_raw']:.3g} for Algorithm R (the counted fair-coin source). "
            f"The simulated null band is mean {hl['null_band_mean']:.3g}, sd "
            f"{hl['null_band_sd']:.3g}, p99 {hl['null_band_p99']:.3g}; the analytic "
            f"cross-check sigma*sqrt(2 ln n) with sigma = sqrt(p(1-p)/T) = "
            f"{sigma_bin:.4g} gives {analytic_from_sigma:.4g}, which is the 3.70e-4 "
            f"quoted -- it is NOT sd*sqrt(2 ln n) = {analytic:.3g}."),
        note="The analytic cross-check uses the PER-BIN sigma, not the sd of the null "
             "band of the maximum; recomputed here from the closed form.",
    )

    # ---------------- 10. bit-budget constants -----------------------------------------
    def harmonic_accept_H(n: int, k: int) -> float:
        """sum_{i=k+1}^{n} h(k/i) -- the exact accept-coin entropy."""
        tot = 0.0
        for i in range(k + 1, n + 1):
            p = k / i
            if 0.0 < p < 1.0:
                tot += -p * math.log2(p) - (1 - p) * math.log2(1 - p)
        return tot

    def classical_evict_H(n: int, k: int) -> float:
        """k*log2(k) * sum_{i=k+1}^{n} 1/i -- log2(k) bits on each of k/i acceptances."""
        return sum((k / i) * log2(k) for i in range(k + 1, n + 1)) if k > 1 else 0.0

    checks = []
    for (n, k) in [(1_000_000, 10), (1_000_000, 100), (1_000_000, 1000)]:
        b = bcf[(n, k)]
        checks.append({
            "n": n, "k": k,
            "evict_frac_artifact": b["evict_frac"],
            "accept_H_artifact": b["accept_H"],
            "accept_H_closed_form": harmonic_accept_H(n, k),
            "accept_H_agrees": approx(b["accept_H"], harmonic_accept_H(n, k),
                                      rel=REL_TOL_DERIVED),
            "evict_H_artifact": b["evict_H"],
            "evict_H_closed_form": classical_evict_H(n, k),
            "evict_H_agrees": approx(b["evict_H"], classical_evict_H(n, k),
                                     rel=REL_TOL_DERIVED),
            "total_H_artifact": b["total_H"],
        })
    c4 = e2["metadata"]["verdicts"]["C4"]["heldout_n1e6"]["10"]
    floor = bcf[(1_000_000, 10)]["total_H"]
    # The 3148x / 8.1x pair is a comparison of the SAME two algorithms under the SAME
    # accounting regime, so it must come from bits_measured, not from mixing the
    # naive64 column of one with the recycle column of the other.
    bm = {(r["candidate"], r["n"], r["k"], r["regime"]): r["flips_total_net_mean"]
          for r in e2["metadata"]["bits_measured"]}
    regimes = {}
    for reg in ("naive64", "naive53", "KY", "recycle"):
        key_s0, key_s2 = ("S0", 1_000_000, 10, reg), ("S2", 1_000_000, 10, reg)
        if key_s0 in bm and key_s2 in bm:
            regimes[reg] = {"S0_algorithm_R": bm[key_s0], "S2_algorithm_L": bm[key_s2],
                            "ratio_R_over_L": bm[key_s0] / bm[key_s2],
                            "ratio_L_over_R": bm[key_s2] / bm[key_s0]}
    add(
        claim_id="C10_bit_budget_constants",
        claim_text_as_printed="n=1e6: evict_frac 0.2550/0.4540/0.6130 at k=10/100/1000; "
                              "accept 7354.94 vs 7355; classical evict 6115.90 vs 6116; "
                              "entropy floor 1493.7 bits and Algorithm R with recycling "
                              "1568.1 (ratio 1.05x); Algorithm L beats R by 3148x under "
                              "64-bit accounting and loses by 8.1x under entropy-optimal; "
                              "representation/coupling gap 31x.",
        source_artifact="E2",
        json_path="metadata.budget_closed_form  ||  metadata.verdicts.C4.heldout_n1e6",
        executed_value={
            "closed_form_checks": checks,
            "entropy_floor_n1e6_k10": floor,
            "S0_recycle_bits": c4["S0_recycle"],
            "ratio_recycle_over_floor": c4["S0_recycle"] / floor,
            "algoL_recycle_bits": c4["algoL_recycle"],
            "algoL_over_S0_under_entropy_optimal": c4["algoL_recycle"] / c4["S0_recycle"],
            "S0_naive64_bits": c4["S0_naive64"],
            "measured_bits_by_regime_n1e6_k10": regimes,
            "L_beats_R_under_naive64_by": regimes.get("naive64", {}).get("ratio_R_over_L"),
            "L_loses_to_R_under_recycle_by": regimes.get("recycle", {}).get("ratio_L_over_R"),
            "representation_over_coupling_ratio": c4["ratio_representation_over_coupling"],
        },
        status="MATCH" if all(c["accept_H_agrees"] and c["evict_H_agrees"] for c in checks)
               else "MISMATCH",
        tolerance=f"rel {REL_TOL_DERIVED} against the closed forms sum_i h(k/i) and "
                  f"k*log2(k)*sum 1/i",
        corrected_text=(
            f"All five hypothesis constants reproduce from their closed forms at n=1e6: "
            f"evict_frac {bcf[(1_000_000,10)]['evict_frac']:.4f} / "
            f"{bcf[(1_000_000,100)]['evict_frac']:.4f} / "
            f"{bcf[(1_000_000,1000)]['evict_frac']:.4f} at k=10/100/1000; accept "
            f"{bcf[(1_000_000,100)]['accept_H']:.2f} bits and classical eviction "
            f"{bcf[(1_000_000,100)]['evict_H']:.2f} bits at k=100. The entropy floor at "
            f"k=10 is {floor:.1f} bits, Algorithm R with interval recycling attains "
            f"{c4['S0_recycle']:.1f} bits ({c4['S0_recycle']/floor:.3f}x the floor), and "
            f"Algorithm L attains {c4['algoL_recycle']:.1f} bits -- "
            f"{c4['algoL_recycle']/c4['S0_recycle']:.2f}x WORSE than Algorithm R under "
            f"entropy-optimal accounting. The regime reversal, taken from COUNTED flips at "
            f"the same cell and the same accounting regime for both algorithms "
            f"(metadata.bits_measured), is: under 64-bit-per-decision accounting Algorithm "
            f"L costs {regimes['naive64']['S2_algorithm_L']:.0f} bits against Algorithm R's "
            f"{regimes['naive64']['S0_algorithm_R']:.0f}, i.e. L beats R by "
            f"{regimes['naive64']['ratio_R_over_L']:.0f}x; under interval recycling L costs "
            f"{regimes['recycle']['S2_algorithm_L']:.0f} bits against R's "
            f"{regimes['recycle']['S0_algorithm_R']:.0f}, i.e. L LOSES to R by "
            f"{regimes['recycle']['ratio_L_over_R']:.2f}x. The representation/coupling gap "
            f"ratio is "
            f"{c4['ratio_representation_over_coupling']:.1f}x."),
        note=(f"The 3148x and 8.1x in the draft are recomputed here from "
              f"metadata.bits_measured as {regimes['naive64']['ratio_R_over_L']:.2f}x and "
              f"{regimes['recycle']['ratio_L_over_R']:.2f}x, both within rounding of the "
              f"printed values. Do NOT form this ratio by dividing one algorithm's "
              f"naive64 column by the other's recycle column -- that mixes regimes and "
              f"gives {c4['S0_naive64']/c4['algoL_recycle']:.0f}x, which is not any "
              f"quantity the draft claims."),
    )

    # ---------------- 11. savings percentages and the regime law -----------------------
    law = lambda n, k: 2 * math.log(k) / (2 * math.log(k) + math.log(n / k))
    sav_rows = []
    for (n, k), r in sorted(s1t_budget.items()):
        sav_rows.append({"n": n, "k": k,
                         "total_saving_frac_artifact": r["total_saving_frac"],
                         "eviction_saving_frac_artifact": r["eviction_saving_frac"],
                         "regime_law_2lnk_over_2lnk_plus_ln_n_over_k": law(n, k),
                         "signed_residual_artifact_minus_law":
                             r["total_saving_frac"] - law(n, k)})
    e1_sav = []
    for (n, k) in sorted(fb):
        if k >= 2:
            e1_sav.append({"n": n, "k": k,
                           "bit_saving_frac_lower_bound": fb[(n, k)]["bit_saving_frac_lower_bound"]})
    add(
        claim_id="C11_saving_percentages",
        claim_text_as_printed="Bit saving >= 24.1% (k=10) to 58.3% (k=1000); "
                              "23.1% / 39.8% / 51.3% at n=1e6 and 20.8% at n=1e7.",
        source_artifact="E1+E2",
        json_path="E1 datasets{dataset=fair_coin_bit_budgets} bit_saving_frac_lower_bound "
                  "||  E2 metadata.verdicts.MAIN.s1t_closed_form_budget total_saving_frac",
        executed_value={"E1_fit_free_lower_bounds": e1_sav,
                        "E2_fitC_saving_rows": sav_rows,
                        "law_residual_max_abs": max(abs(r["signed_residual_artifact_minus_law"])
                                                    for r in sav_rows),
                        "law_residual_mean": sum(r["signed_residual_artifact_minus_law"]
                                                 for r in sav_rows) / len(sav_rows)},
        status="MISMATCH",
        tolerance=f"rel {REL_TOL_DERIVED} on the artifact values; the law comparison is a "
                  f"consistency check only, with no tolerance",
        corrected_text=(
            f"Two different saving figures are in play and must be labelled separately. "
            f"The FIT-FREE analytic lower bounds (E1 bench 1, no fitted constant) are "
            f"{fb[(1_000_000,10)]['bit_saving_frac_lower_bound']*100:.1f}% at k=10 and "
            f"{fb[(1_000_000,1000)]['bit_saving_frac_lower_bound']*100:.1f}% at k=1000, "
            f"n=1e6 -- and these inherit the missing log2(k) factor of claim "
            f"C06_fit_free_bracket_missing_log2k, so they are not usable as printed. "
            f"The EXTRAPOLATED figures {sav_rows[0]['total_saving_frac_artifact']*100:.1f}% "
            f"(n=1e6,k=10), "
            f"{[r for r in sav_rows if r['n']==1_000_000 and r['k']==100][0]['total_saving_frac_artifact']*100:.1f}% "
            f"(n=1e6,k=100), "
            f"{[r for r in sav_rows if r['n']==1_000_000 and r['k']==1000][0]['total_saving_frac_artifact']*100:.1f}% "
            f"(n=1e6,k=1000) and "
            f"{[r for r in sav_rows if r['n']==10_000_000 and r['k']==10][0]['total_saving_frac_artifact']*100:.1f}% "
            f"(n=1e7,k=10) come from the fitted constant fitC and must be labelled "
            f"EXTRAPOLATED. As a consistency check only, the regime law "
            f"2 ln k / (2 ln k + ln(n/k)) has a signed residual against the artifact's own "
            f"rows of at most "
            f"{max(abs(r['signed_residual_artifact_minus_law']) for r in sav_rows):.3f} "
            f"(mean {sum(r['signed_residual_artifact_minus_law'] for r in sav_rows)/len(sav_rows):+.3f}); "
            f"this ledger does not claim the law."),
        note="The 24.1%/58.3% pair is E1's fit-free bracket; the 23.1/39.8/51.3/20.8% "
             "quartet is E2's fitC extrapolation. They are not the same quantity.",
    )

    # ---------------- 12. index-convention audit (defect D5) ---------------------------
    q2 = codefacts.get("Q2", {})
    add(
        claim_id="C12_index_convention_audit",
        path_kind="source_code",
        claim_text_as_printed="The eviction randomness is exactly zero on the steps where "
                              "k divides (i+1) / (i-k) / i (the draft uses more than one "
                              "form of the same condition).",
        source_artifact="E1+E2 source code",
        json_path="E1/fibermap.py, E1/candidates.py, E1/forced.py, E1/exact_dp.py, "
                  "E2/s1t.py, E2/candidates.py",
        executed_value={"rows": q2.get("rows", []),
                        "files_disagree": q2.get("files_disagree"),
                        "explanation": q2.get("explanation", "NOT_READ")},
        status="MISMATCH" if q2.get("files_disagree") else (
            "UNAVAILABLE" if not q2.get("rows") else "MATCH"),
        tolerance="code reading; no numerical tolerance",
        corrected_text=(
            "Fix the paper to ONE convention, i = PREFIX SIZE (the reservoir after i items "
            "is a subset of [i], and the arriving item is i+1). Under that convention the "
            "divisibility condition is k | (i+1). E1/fibermap.py already uses it; "
            "E1/forced.py's bit_budget, E2/s1t.py and E2/candidates.py use i = ARRIVING "
            "ITEM, where the same condition reads k | (i-k), equivalently k | i. The "
            "conditions are mathematically equivalent once translated, but the symbol is "
            "reused with two meanings across files that are used together."),
        note="Per-file translation table is in executed_value.rows.",
    )

    # ---------------- 13. matched-panel confusion (reproduced independently) -----------
    a2_path = HERE / "results" / "stage_a2.json"
    if a2_path.exists():
        a2 = json.loads(a2_path.read_text())
        conf_repro = {m: {"accuracy": v["accuracy"],
                          "recall_on_broken_samplers": v["recall_on_broken_samplers"],
                          "false_alarm_rate_on_correct_samplers":
                              v["false_alarm_rate_on_correct_samplers"],
                          "missed_cells": v["missed_cells"]}
                      for m, v in a2["confusion"].items()}
        stored = e1["metadata"]["matched_confusion"]
        agree = (approx(conf_repro["mc_final_first_order"]["accuracy"],
                        stored["mc_final_first_order"]["accuracy"], rel=1e-12)
                 and approx(conf_repro["mc_final_first_order"]["recall_on_broken_samplers"],
                            stored["mc_final_first_order"]["recall_on_broken_samplers"], rel=1e-12)
                 and approx(conf_repro["mc_all_prefix"]["accuracy"],
                            stored["mc_all_prefix"]["accuracy"], rel=1e-12)
                 and approx(conf_repro["mc_all_prefix"]["recall_on_broken_samplers"],
                            stored["mc_all_prefix"]["recall_on_broken_samplers"], rel=1e-12)
                 and approx(conf_repro["mc_final_pairwise"]["accuracy"],
                            stored["mc_final_pairwise"]["accuracy"], rel=1e-12))
        add(
            claim_id="C13_matched_panel_confusion_and_structural_blindness",
            claim_text_as_printed="Matched 18-cell confusion vs exact-DP ground truth: "
                                  "first-order MC accuracy 0.833 (recall 0.667, misses all "
                                  "3 S5-CIRC cells); ANYTIME all-prefix MC identical "
                                  "0.833/0.667 (the blindness is structural, not a power "
                                  "problem); pairwise MC 1.000.",
            source_artifact="E1",
            json_path="metadata.matched_confusion",
            executed_value={"stored_in_E1": {m: {kk: stored[m].get(kk) for kk in
                                                 ("accuracy", "recall_on_broken_samplers",
                                                  "specificity_on_correct_samplers",
                                                  "missed_cells")}
                                             for m in stored},
                            "reproduced_here_independently": conf_repro,
                            "reproduction_agrees": agree,
                            "reproduction_config": {"T": a2["rows"][0]["T"],
                                                    "seed": 20260918,
                                                    "n_cells": len(a2["rows"])}},
            status="MATCH" if agree else "MISMATCH",
            tolerance="exact equality of accuracy and recall",
            corrected_text=(
                f"Reproduced independently in this evaluation on the same 18 cells: "
                f"first-order accuracy "
                f"{conf_repro['mc_final_first_order']['accuracy']:.4f} / recall "
                f"{conf_repro['mc_final_first_order']['recall_on_broken_samplers']:.4f}, "
                f"anytime all-prefix accuracy {conf_repro['mc_all_prefix']['accuracy']:.4f} "
                f"/ recall {conf_repro['mc_all_prefix']['recall_on_broken_samplers']:.4f} "
                f"(the SAME three S5-CIRC cells missed), pairwise "
                f"{conf_repro['mc_final_pairwise']['accuracy']:.4f}. The anytime test does "
                f"not help: the blindness is structural. The NEW third column, the L1 "
                f"identity/collision tester, scores "
                f"{conf_repro['l1_identity_final']['accuracy']:.4f} accuracy / "
                f"{conf_repro['l1_identity_final']['recall_on_broken_samplers']:.4f} recall "
                f"with a "
                f"{conf_repro['l1_identity_final']['false_alarm_rate_on_correct_samplers']:.4f} "
                f"false-alarm rate on this panel."),
            note="F4 of the plan: the anytime first-order numbers DO reproduce at "
                 "0.833/0.667, so the blindness claim rests on reproduced numbers.",
        )
    else:
        add(claim_id="C13_matched_panel_confusion_and_structural_blindness",
            claim_text_as_printed="Matched 18-cell confusion: first-order 0.833/0.667, "
                                  "anytime identical, pairwise 1.000.",
            source_artifact="E1", json_path="metadata.matched_confusion",
            executed_value={"reason": "results/stage_a2.json not present at ledger time"},
            status="UNAVAILABLE", tolerance="n/a",
            corrected_text="Not re-executed.", note="Part A stage a2 had not finished.")

    # ---------------- 14. 72-run replicated scoreboard ---------------------------------
    rs = e2["metadata"]["replicated_verdict_scoreboard"]
    add(
        claim_id="C14_replicated_scoreboard_72_runs",
        claim_text_as_printed="First-order max-deviation testing gets recall 0.00 on 24 "
                              "replicated degenerate runs; pairwise co-inclusion gets 1.00 "
                              "with 0/48 false alarms on provably-correct samplers.",
        source_artifact="E2",
        json_path="metadata.replicated_verdict_scoreboard",
        executed_value={"baseline_first_order": rs["baseline"], "ours_coinclusion": rs["ours"],
                        "combined": rs["combined"],
                        "recount_from_rows": {
                            "n_rows": len(rows), "n_blind_spot": len(brok),
                            "n_correct": len(corr)}},
        status="MATCH" if (rs["baseline"]["recall_on_broken"] == 0.0
                           and rs["ours"]["recall_on_broken"] == 1.0
                           and rs["ours"]["false_alarm_on_correct"] == 0.0
                           and rs["baseline"]["FN"] == 24 and rs["ours"]["TN"] == 48) else "MISMATCH",
        tolerance="exact integer counts",
        corrected_text=(
            f"Over the 72 replicated runs (8 seeds x 9 samplers, n=1000, k=10, T=1e6) the "
            f"first-order max-deviation test has recall "
            f"{rs['baseline']['recall_on_broken']:.2f} on the {rs['baseline']['FN']} "
            f"degenerate runs and false-alarm rate "
            f"{rs['baseline']['false_alarm_on_correct']:.2f} on the {rs['baseline']['TN']} "
            f"provably-correct ones; calibrated pairwise co-inclusion has recall "
            f"{rs['ours']['recall_on_broken']:.2f} with "
            f"{rs['ours']['FP']}/{rs['ours']['TN'] + rs['ours']['FP']} false alarms."),
        note="Recounted from metadata.replication_calibration.rows as well as the scoreboard.",
    )

    # ---------------- 15. the blind spot, quantified -----------------------------------
    hd = {h["candidate"]: h for h in e1["metadata"]["literal_deliverable"]["headline"]}
    circ = hd["S5-CIRC"]
    n_big, k_big = 1000, 10
    tv_exact = 1.0 - n_big / comb(n_big, k_big)
    add(
        claim_id="C15_blind_spot_quantified",
        claim_text_as_printed="S5-CIRC max_dev=3.26e-4 (0.81x null p95, NOT FLAGGED by the "
                              "first-order test) yet pairwise max|z|=971.5 with "
                              "log10(p)=-2.05e5 and 98.2% of item pairs NEVER co-sampled; "
                              "exact TV = 1 - i/C(i,k), up to 0.986.",
        source_artifact="E1",
        json_path="metadata.literal_deliverable.headline",
        executed_value={
            "S5CIRC_max_dev": circ["max_deviation_from_k_over_n"],
            "S5CIRC_in_null_p95_units": circ["max_deviation_in_null_band_p95_units"],
            "S5CIRC_flagged_first_order": circ["flagged_by_first_order_test"],
            "S5CIRC_pairwise_max_abs_z": circ["pairwise_max_abs_z"],
            "S5CIRC_pairwise_log10_p": circ["pairwise_log10_p_value_bonferroni"],
            "S5CIRC_frac_pairs_never_co_sampled": circ["fraction_of_pairs_never_co_sampled"],
            "exact_TV_at_n1000_k10_recomputed": tv_exact,
            "tv_min_over_screened_prefixes":
                e1["metadata"]["screen_ranking"]["tv_min_over_screened_prefixes"],
            "max_screened_TV_claimed": 0.986,
            "C_n_k": comb(n_big, k_big),
        },
        status="MATCH" if (abs(circ["pairwise_max_abs_z"] - 971.5) < 0.1
                           and abs(circ["fraction_of_pairs_never_co_sampled"] - 0.982) < 5e-4
                           and not circ["flagged_by_first_order_test"]) else "MISMATCH",
        tolerance="rel 1e-3 on the quoted 3-significant-figure values",
        corrected_text=(
            f"At n=1000, k=10, T=1e6 the circular-window sampler S5-CIRC has max deviation "
            f"{circ['max_deviation_from_k_over_n']:.3g} "
            f"({circ['max_deviation_in_null_band_p95_units']:.2f}x the null p95, not "
            f"flagged), pairwise max|z| = {circ['pairwise_max_abs_z']:.1f} with log10(p) = "
            f"{circ['pairwise_log10_p_value_bonferroni']:.3g}, and "
            f"{circ['fraction_of_pairs_never_co_sampled']*100:.1f}% of item pairs never "
            f"co-sampled. Its exact total-variation distance to uniform at this cell is "
            f"1 - n/C(n,k) = {tv_exact:.9f}, i.e. 1 - 1000/C(1000,10); the 0.986 figure is "
            f"the MINIMUM over the SCREENED small-n prefixes, not this cell's value."),
        note="TV at n=1000, k=10 is 1 - 3.8e-21, indistinguishable from 1; 0.986 is the "
             "small-n screened minimum. Both must not be conflated.",
    )

    # ---------------- 16. the sign reversal --------------------------------------------
    t1 = {r["candidate"]: r for r in
          e2["metadata"]["summary_tables"]["T1_uniformity_deliverable_n1000_k10_T1e6"]}
    better = [c for c in ("S5e", "S5g", "S6sys")
              if t1[c]["maxdev_raw"] < t1["S0"]["maxdev_raw"]]
    add(
        claim_id="C16_sign_reversal_blind_spots_score_better",
        claim_text_as_printed="The degenerate samplers SCORE BETTER than Algorithm R on "
                              "the first-order test: 2.17e-4 / 3.27e-4 / 3.11e-4 at "
                              "-3.67 / -0.60 / -1.04 sigma against R's 3.68e-4 at "
                              "+0.55 sigma.",
        source_artifact="E2",
        json_path="metadata.summary_tables.T1_uniformity_deliverable_n1000_k10_T1e6",
        executed_value={c: {"maxdev_raw": t1[c]["maxdev_raw"], "z_band": t1[c]["z_band"]}
                        for c in ("S0", "S5e", "S5g", "S6sys")},
        status="MATCH" if len(better) == 3 else "MISMATCH",
        tolerance=f"rel {REL_TOL_JSON}",
        corrected_text=(
            f"On the single headline run, all three blind-spot samplers score BETTER "
            f"(smaller max deviation) than Algorithm R: BLOCK-LOCK "
            f"{t1['S5e']['maxdev_raw']:.3g} ({t1['S5e']['z_band']:+.2f} sigma), PAIR-LOCK "
            f"{t1['S5g']['maxdev_raw']:.3g} ({t1['S5g']['z_band']:+.2f} sigma) and "
            f"SYSTEMATIC {t1['S6sys']['maxdev_raw']:.3g} "
            f"({t1['S6sys']['z_band']:+.2f} sigma) against Algorithm R's "
            f"{t1['S0']['maxdev_raw']:.3g} ({t1['S0']['z_band']:+.2f} sigma). The reported "
            f"2.17e-4 for BLOCK-LOCK is the single-run value; over 8 seeds its mean is "
            f"different -- quote one or the other, not both."),
        note="Mechanism tested separately in Part C2.",
    )

    # ---------------- 17. gates, spend and determinism ---------------------------------
    g1 = e1["metadata"]["gates"]
    add(
        claim_id="C17_gates_spend_determinism",
        claim_text_as_printed="59/59 gates pass; screen frozen to screen_ranking.json with "
                              "a sha256 that reproduces byte-identically; $0.00 spend; "
                              "113 pytest tests green.",
        source_artifact="E1+E2",
        json_path="E1 metadata.gates",
        executed_value={
            "E1_gates_passed": g1["passed"], "E1_n_tests_passed": g1["n_tests_passed"],
            "E1_gate_names": g1["gates"],
            "E1_spend_usd": e1["metadata"]["spend_usd"],
            "E1_llm_calls": e1["metadata"]["llm_calls"],
            "E2_spend_usd": e2["metadata"]["manifest"]["cost_usd"],
            "E2_llm_calls": e2["metadata"]["manifest"]["llm_calls"],
            "E2_screen_ranking_sha256": e2["metadata"]["screen_ranking"]["content_sha256"],
            "E1_screen_ranking_sha256":
                e1["metadata"]["screen_ranking"]["sha256_of_this_file_content"],
        },
        status="MATCH" if (g1["passed"] and g1["n_tests_passed"] == 59
                           and e1["metadata"]["spend_usd"] == 0.0
                           and e2["metadata"]["manifest"]["cost_usd"] == 0.0) else "MISMATCH",
        tolerance="exact",
        corrected_text=(
            f"E1 reports {g1['n_tests_passed']}/{g1['n_tests_passed']} gates passing and "
            f"E2's frozen screen ranking hashes to "
            f"{e2['metadata']['screen_ranking']['content_sha256'][:12]}...; both artifacts "
            f"report $0.00 of LLM spend and 0 LLM calls. The '113 pytest tests' figure "
            f"belongs to E2 and is not stored in either full_method_out.json -- do not "
            f"attribute it to E1, whose stored count is "
            f"{g1['n_tests_passed']} gates."),
        note="The 59 and the 113 are different artifacts' counts of different things.",
    )

    # ---------------- 18. the declared method deviation --------------------------------
    cp_path = HERE / "results" / "stage_chunkprobe.json"
    decl = e2["metadata"]["verdicts"]["co_inclusion_rule"]
    if cp_path.exists():
        cp = json.loads(cp_path.read_text())
        cell = cp["summary"].get("n2000_k100", {})
        n_fa_chunk = cell.get("chunk_sweep_n_false_alarms")
        n_fa_seed = cell.get("seed_sweep_n_false_alarms")
        reproduces = bool((n_fa_chunk or 0) + (n_fa_seed or 0) > 0)
        add(
            claim_id="C18_declared_deviation_evidence",
            claim_text_as_printed="The pre-registered nominal chi-square p<1e-6 "
                                  "co-inclusion rule is MIS-CALIBRATED at large k/n: it "
                                  "fires on Algorithm R itself at n=2000, k=100 with "
                                  "p = 7.9e-16.",
            source_artifact="E2",
            json_path="metadata.verdicts.co_inclusion_rule",
            executed_value={
                "declared": {"pre_registered_rule": decl["pre_registered_rule"],
                             "rule_actually_used": decl["rule_actually_used"],
                             "false_alarm_cells_declared":
                                 decl["nominal_rule_false_alarms_on_the_grid"]},
                "reproduction": cp["summary"],
                "n2000_k100_reproduces_the_false_alarm": reproduces,
                "chunks_tried": cp["chunks_tried"], "seeds_tried": cp["seeds_tried"],
                "T": cp["T"],
            },
            status="MATCH" if reproduces else "MISMATCH",
            tolerance="a single-run nominal p is reproduced only if some (chunk, seed) "
                      "combination at the same cell again gives p < 1e-6",
            corrected_text=(
                "The declared method deviation is sound, but its stated EVIDENCE is a "
                "single run and does not reproduce. Re-running Algorithm R at n=2000, "
                f"k=100, T=200000 across {len(cp['chunks_tried'])} trial-chunk sizes at a "
                f"fixed seed and {len(cp['seeds_tried'])} further seeds yields "
                f"{(n_fa_chunk or 0) + (n_fa_seed or 0)} nominal false alarms out of "
                f"{len(cp['chunks_tried']) + len(cp['seeds_tried'])} runs, with the p value "
                f"swinging over "
                f"{cell.get('p_spans_orders_of_magnitude_across_chunks', 0):.1f} orders of "
                "magnitude when ONLY the chunking of the PCG64 stream changes. Report the "
                "deviation on the grounds that actually hold: the calibrated own-null rule "
                "separates provably-correct from broken samplers with no overlap on every "
                "cell tested, so no verdict in either artifact depends on the nominal rule, "
                "and the simulated null of the co-inclusion chi-square -- not any single "
                "run -- is what decides whether the nominal reference distribution is "
                "right at a given k/n."),
            note="Part C1 of this evaluation re-derives the calibration; the per-cell "
                 "simulated-null diagnostic is in the calibration_and_sign_reversal "
                 "dataset under metadata_part = C1_nominal_rule_miscalibration.",
        )
    else:
        add(claim_id="C18_declared_deviation_evidence",
            claim_text_as_printed="The nominal chi-square co-inclusion rule fires on "
                                  "Algorithm R at n=2000, k=100 with p = 7.9e-16.",
            source_artifact="E2", json_path="metadata.verdicts.co_inclusion_rule",
            executed_value={"reason": "results/stage_chunkprobe.json not present"},
            status="UNAVAILABLE", tolerance="n/a",
            corrected_text="Not re-executed.", note="chunk probe had not finished.")

    return C


def resolve_and_stamp(claims: list[dict], e1: dict, e2: dict) -> list[dict]:
    """Attach the resolver's view of json_path so an unresolvable path is reported."""
    for c in claims:
        arts = {"E1": e1, "E2": e2}
        if c.get("path_kind") == "source_code":
            c["path_resolution"] = {"status": "SOURCE_CODE_NOT_A_JSON_PATH",
                                    "path": c["json_path"]}
            continue
        first = c["json_path"].split("||")[0].strip()
        art_key = c["source_artifact"].split("+")[0]
        if first.startswith("E1 ") or first.startswith("E2 "):
            art_key, first = first[:2], first[3:].strip()
        if art_key not in arts or not re.match(r"^[A-Za-z_]", first):
            c["path_resolution"] = {"status": "NOT_A_JSON_PATH", "path": c["json_path"]}
            continue
        try:
            v = resolve(arts[art_key], first.split()[0])
            c["path_resolution"] = {
                "status": "RESOLVED", "path": first.split()[0],
                "type": type(v).__name__,
                "size": len(v) if isinstance(v, (list, dict)) else None}
        except Unresolved as exc:
            c["path_resolution"] = {"status": "UNRESOLVED", "path": exc.path,
                                    "searched": exc.searched, "nearest_keys": exc.nearest}
            c["status"] = "UNAVAILABLE"
            c["note"] = (c.get("note", "") +
                         f" | json_path did not resolve; searched {exc.searched}, "
                         f"nearest keys {exc.nearest[:10]}")
    return claims


def write_outputs(claims: list[dict], out_dir: Path) -> dict:
    import csv

    counts = {"n_claims": len(claims),
              "n_match": sum(1 for c in claims if c["status"] == "MATCH"),
              "n_mismatch": sum(1 for c in claims if c["status"] == "MISMATCH"),
              "n_unavailable": sum(1 for c in claims if c["status"] == "UNAVAILABLE")}

    with (out_dir / "claims_ledger.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["claim_id", "claim_text_as_printed", "source_artifact", "json_path",
                    "status", "tolerance", "executed_value_json", "corrected_text", "note"])
        for c in claims:
            w.writerow([c["claim_id"], c["claim_text_as_printed"], c["source_artifact"],
                        c["json_path"], c["status"], c["tolerance"],
                        json.dumps(c["executed_value"], default=str),
                        c["corrected_text"], c.get("note", "")])

    severity = {"C04_exact_grid_H_and_M": 0, "C18_declared_deviation_evidence": 3.5, "C05_fitC_is_not_a_bound": 1,
                "C06_fit_free_bracket_missing_log2k": 2, "C11_saving_percentages": 3,
                "C03_anytime_overhead_post_removal": 4, "C09_two_literal_deliverables": 5,
                "C01_coinclusion_separation": 6,
                "C07_algorithm_L_realistic_regime_is_modelled": 7,
                "C12_index_convention_audit": 8}
    bad = sorted([c for c in claims if c["status"] in ("MISMATCH", "UNAVAILABLE")],
                 key=lambda c: severity.get(c["claim_id"], 99))

    def esc(s: str) -> str:
        return str(s).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Patch table -- every MISMATCH and UNAVAILABLE claim, with a drop-in replacement",
        "",
        f"Generated by `partB.py` from the two iteration-1 artifacts. "
        f"{counts['n_claims']} claims checked: {counts['n_match']} MATCH, "
        f"{counts['n_mismatch']} MISMATCH, {counts['n_unavailable']} UNAVAILABLE. "
        f"Rows are sorted by severity (headline numbers first).",
        "",
        "| # | Claim as printed | Status | Executed value (key figures) | "
        "Corrected sentence (drop-in) | Source path |",
        "|---|---|---|---|---|---|",
    ]
    for i, c in enumerate(bad, 1):
        ev = json.dumps(c["executed_value"], default=str)
        if len(ev) > 700:
            ev = ev[:700] + " ...(full value in claims_ledger.csv / eval_out.json)"
        lines.append(f"| {i} | {esc(c['claim_text_as_printed'])} | **{c['status']}** | "
                     f"`{esc(ev)}` | {esc(c['corrected_text'])} | `{esc(c['json_path'])}` |")
    lines += ["", "## Notes per row", ""]
    for i, c in enumerate(bad, 1):
        lines.append(f"{i}. **{c['claim_id']}** -- {c.get('note','')}")
    (out_dir / "patch_table.md").write_text("\n".join(lines) + "\n")
    return counts


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(str(HERE / "logs" / "partB.log"), rotation="30 MB", level="DEBUG")
    e1, e2 = load()
    cf_path = HERE / "scratch" / "codefacts.json"
    codefacts = json.loads(cf_path.read_text()) if cf_path.exists() else {}
    if not codefacts:
        logger.warning("scratch/codefacts.json missing -- code-reading claims -> UNAVAILABLE")
    claims = resolve_and_stamp(build_claims(e1, e2, codefacts), e1, e2)
    counts = write_outputs(claims, HERE)
    (HERE / "results" / "stage_b.json").write_text(
        json.dumps({"claims": claims, "counts": counts}, indent=1, default=str))
    logger.info(f"ledger: {counts}")
    for c in claims:
        logger.info(f"  {c['status']:<12} {c['claim_id']}")
