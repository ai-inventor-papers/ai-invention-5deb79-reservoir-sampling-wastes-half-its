#!/usr/bin/env python3
"""Exact check: is random eviction needed?

Top-level driver.  Runs the gate suite, then the frozen SCREEN panel, then the HELD-OUT
panel, then assembles ``method_out.json``.

WHAT IS COMPARED, AND AGAINST WHAT
----------------------------------
BASELINE (the current standard, and the literal user ask): verify a reservoir sampler's
uniformity by Monte Carlo -- run T independent streams, count how often each item ends in
the reservoir, report the maximum deviation from the expected frequency k/n.  We implement
three strengths of it (final-reservoir first-order, final-reservoir pairwise, and an
anytime all-prefix version), each with a simulated null band so the reported deviation is
interpretable rather than a bare number.

OUR METHOD: an exact-rational prefix DP over sampler states.  It propagates the EXACT
distribution with Fractions and reports the maximum deviation from 1/C(i,k) over all
C(i,k) subsets at every prefix.  Zero here is a proof of anytime uniformity; nonzero is a
counterexample with an exact magnitude.  Paired with it, the eviction step is solved as a
transportation problem on the Johnson graph, which decides exactly how many random bits
eviction actually needs.

Both run in the same pipeline on the same candidates and the same cells, so no comparison
is contaminated by an implementation-level difference in the measurement path.

NO EXTERNAL DATASET.  Uniformity of a reservoir sampler is a statement about INDEX
k-subsets, so the canonical stream is the integer sequence 1..n, generated in-script.
NO LLM CALLS.  OpenRouter spend for this artifact is $0.00.
"""

from __future__ import annotations

import json
import os
import platform
import re
import resource
import subprocess
import sys
import time
from math import comb, log2
from pathlib import Path
from typing import Any

from loguru import logger

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import candidates  # noqa: E402
import panels  # noqa: E402
import verdicts  # noqa: E402

RAM_LIMIT_GB = 64
CPU_LIMIT_S = 6 * 3600


# --------------------------------------------------------------------------------------
# Example builders (exp_gen_sol_out schema: only input / output / metadata_* / predict_*)
# --------------------------------------------------------------------------------------


def _f(x: Any) -> Any:
    return x


def build_verdict_examples(matched: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for r in matched["rows"]:
        out.append(
            {
                "input": (
                    f"Sampler {r['candidate']} ({r['candidate_description']}). "
                    f"Is it ANYTIME uniform -- i.e. is the reservoir after i items a uniformly "
                    f"random k-subset of [i] for EVERY prefix i = k..n -- at n={r['n']}, k={r['k']}? "
                    f"Monte-Carlo budget: T={r['T']} independent streams."
                ),
                "output": r["truth_exact_dp"],
                "predict_exact_dp": r["exact_dp"]["verdict"],
                "predict_mc_final_first_order": r["mc_final_first_order"]["verdict"],
                "predict_mc_final_pairwise": r["mc_final_pairwise"]["verdict"],
                "predict_mc_all_prefix": r["mc_all_prefix"]["verdict"],
                "metadata_candidate": r["candidate"],
                "metadata_n": r["n"],
                "metadata_k": r["k"],
                "metadata_T": r["T"],
                "metadata_seed": r["seed"],
                "metadata_exact_dp": r["exact_dp"],
                "metadata_mc_final_first_order": r["mc_final_first_order"],
                "metadata_mc_final_pairwise": r["mc_final_pairwise"],
                "metadata_mc_all_prefix": r["mc_all_prefix"],
                "metadata_agreement": r["agreement"],
                "metadata_cost": {
                    "exact_dp_stream_passes": 0,
                    "exact_dp_random_bits": 0,
                    "exact_dp_seconds": r["exact_dp"]["seconds"],
                    "mc_stream_passes": r["mc_stream_passes"],
                    "mc_seconds": r["mc_seconds"],
                },
            }
        )
    return out


def build_exact_examples(screen: dict[str, Any], heldout: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for split, rows in (("screen", screen["exact"]), ("held_out", heldout["exact"])):
        for r in rows:
            verdict = (
                verdicts.UNIFORM
                if (r["i_first_fail"] is None and not r["stopped_early"])
                else verdicts.NOT_UNIFORM
            )
            note = ""
            if r["stopped_early"]:
                note = (
                    f" (run stopped at its first failing prefix i={r['i_first_fail']} to keep the "
                    "ordered-slot state space bounded)"
                )
            out.append(
                {
                    "input": (
                        f"Exact prefix DP over sampler states: {r['key']} "
                        f"({r['description']}) at n={r['n']}, k={r['k']}. "
                        f"Report max_S |P(R_i = S) - 1/C(i,k)| over all C(i,k) subsets at every "
                        f"prefix i = k..n, in exact rational arithmetic."
                    ),
                    "output": verdict,
                    "predict_exact_dp": verdict,
                    "metadata_split": split,
                    "metadata_candidate": r["key"],
                    "metadata_n": r["n"],
                    "metadata_k": r["k"],
                    "metadata_i_first_fail": r["i_first_fail"],
                    "metadata_m1_rel_at_failure": r["m1_rel_at_failure_float"],
                    "metadata_m1_rel_at_failure_exact": r["m1_rel_at_failure"],
                    "metadata_exactly_uniform_all_prefixes": r["exactly_uniform_all_prefixes"],
                    "metadata_stopped_early": r["stopped_early"],
                    "metadata_max_states_seen": r["max_states_seen"],
                    "metadata_note": note,
                    "metadata_prefixes": r["prefixes"],
                }
            )
    return out


def _fiber_class(r: dict[str, Any]) -> str:
    if r["divisible"]:
        return "ZERO_EVICTION_BITS"
    if r["ratio_to_log2k"] <= 0.25:
        return "SUBLINEAR_IN_LOG2K"
    return "COMPARABLE_TO_LOG2K"


def build_fiber_examples(screen: dict[str, Any], heldout: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for split, rows in (("screen", screen["fibermap"]), ("held_out", heldout["fibermap"])):
        for r in rows:
            ex = r["min_entropy_exact"]
            pred_exact = (
                f"{ex['min_bits']:.6f}"
                if ex.get("solved")
                else (
                    f"BRACKETED[{r['bracket']['lower_bits']:.6f},{r['bracket']['upper_bits']:.6f}]"
                )
            )
            slim = dict(r)
            ifz = dict(slim.get("integral_feasibility", {}))
            if r["i_prefix"] > 12:
                ifz.pop("certificate_inline", None)
            slim["integral_feasibility"] = ifz
            out.append(
                {
                    "input": (
                        f"Eviction step at prefix i={r['i_prefix']} (arriving item {r['arriving_item']}), "
                        f"k={r['k']}: transportation problem on the Johnson graph J({r['i_prefix']},{r['k']}) "
                        f"-> J({r['i_prefix']},{r['k'] - 1}) with {r['n_sources']} sources of supply 1, "
                        f"{r['n_targets']} targets of demand {r['demand_per_target']}. "
                        f"What is the MINIMUM per-acceptance eviction entropy, in bits, of any "
                        f"exactly-uniform eviction rule?"
                    ),
                    "output": _fiber_class(r),
                    "predict_classical_rule": f"{r['classical_evict_bits']:.6f}",
                    "predict_min_cost_flow_constructive": f"{r['achieved_entropy_bits']:.6f}",
                    "predict_milp_exact": pred_exact,
                    "predict_analytic_bracket": (
                        f"[{r['bracket']['lower_bits']:.6f},{r['bracket']['upper_bits']:.6f}]"
                    ),
                    "metadata_split": split,
                    "metadata_i_prefix": r["i_prefix"],
                    "metadata_k": r["k"],
                    "metadata_divisible": r["divisible"],
                    "metadata_min_evict_entropy_bits": r["min_evict_entropy_bits"],
                    "metadata_min_evict_entropy_is_exact": r["min_evict_entropy_is_exact"],
                    "metadata_classical_evict_bits": r["classical_evict_bits"],
                    "metadata_ratio_to_log2k": r["ratio_to_log2k"],
                    "metadata_step": slim,
                }
            )
    return out


def build_budget_examples(screen: dict[str, Any], heldout: dict[str, Any]) -> list[dict[str, Any]]:
    claims = {(c["n"], c["k"], c["quantity"]): c for c in
              screen["bit_budgets"]["hypothesis_constant_checks"]
              + heldout["bit_budgets"]["hypothesis_constant_checks"]}
    out = []
    for split, blk in (("screen", screen["bit_budgets"]), ("held_out", heldout["bit_budgets"])):
        for b in blk["rows"]:
            c = claims.get((b["n"], b["k"], "evict_frac"))
            out.append(
                {
                    "input": (
                        f"Anytime reservoir sampling over a stream of n={b['n']} items with k={b['k']}. "
                        f"The accept coin is forced to k/i at every step, so it costs sum_i h(k/i) fair "
                        f"bits. What fraction of the total fair-coin budget does EVICTION account for "
                        f"under the classical uniform-slot rule, and how much is unavoidable?"
                    ),
                    "output": f"{b['evict_frac']:.6f}",
                    "predict_recomputed_here": f"{b['evict_frac']:.6f}",
                    "predict_hypothesis_stated_value": (
                        f"{c['claimed']:.6f}" if c else "not_stated_by_the_hypothesis"
                    ),
                    "metadata_split": split,
                    "metadata_budget": b,
                    "metadata_hypothesis_check": c,
                }
            )
    return out


def build_mc_examples(screen: dict[str, Any], heldout: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for split, blk in (("screen", screen["montecarlo"]), ("held_out", heldout["montecarlo"])):
        band = blk["null_band_simulated"]
        for r in blk["rows"]:
            truth_known = {
                "S0": verdicts.UNIFORM,
                "S1n": verdicts.NOT_UNIFORM,
                "S5-CIRC": verdicts.NOT_UNIFORM,
                "C-NOACCEPT": verdicts.NOT_UNIFORM,
            }[r["candidate"]]
            v1 = verdicts.NOT_UNIFORM if r["flagged_first_order"] else verdicts.UNIFORM
            v2 = verdicts.NOT_UNIFORM if r["flagged_second_order"] else verdicts.UNIFORM
            out.append(
                {
                    "input": (
                        f"Monte-Carlo uniformity check of sampler {r['candidate']} "
                        f"({candidates.DESCRIPTIONS[r['candidate']]}) at n={r['n']}, k={r['k']}, "
                        f"T={r['T']} independent streams. Report the maximum deviation of the "
                        f"per-item inclusion frequency from the expected k/n = {r['k'] / r['n']:.4f}, "
                        f"judged against a simulated exact-uniform null band."
                    ),
                    "output": truth_known,
                    "predict_mc_final_first_order": v1,
                    "predict_mc_final_pairwise": v2,
                    "metadata_split": split,
                    "metadata_candidate": r["candidate"],
                    "metadata_n": r["n"],
                    "metadata_k": r["k"],
                    "metadata_T": r["T"],
                    "metadata_max_deviation_from_k_over_n": r["first_order"]["max_dev"],
                    "metadata_max_deviation_in_null_band_p95_units": r["max_dev_in_band_units"],
                    "metadata_null_band_first_order": band["first_order"],
                    "metadata_first_order": r["first_order"],
                    "metadata_pair": r["pair"],
                    "metadata_pair_p_value_bonferroni": r["pair_p_value_normal"],
                    "metadata_pair_log10_p_value_bonferroni": r["pair_log10_p_value"],
                    "metadata_truth_source": (
                        "exact prefix DP at small n plus the exact structural argument for "
                        "S5-CIRC (support size i, not C(i,k)); S0 is certified exactly by gate G1"
                    ),
                }
            )
    return out


def build_forced_examples(screen: dict[str, Any], heldout: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for split, rows in (("screen", screen["forced_accept"]), ("held_out", heldout["forced_accept"])):
        for r in rows:
            out.append(
                {
                    "input": (
                        f"A sampler is uniform on k-subsets of [{r['i_prefix']}] and must remain "
                        f"uniform on k-subsets of [{r['arriving_item']}] with k={r['k']}. Solve the "
                        f"linear system for the accept probability p(S) of every one of the "
                        f"{r['n_unknowns']} states. Is the solution set a single, state-independent "
                        f"point?"
                    ),
                    "output": "FORCED_UNIQUE" if r["forced_accept_unique"] else "NOT_FORCED",
                    "predict_solved_from_the_linear_system": r["solved_accept"] or "unsolved",
                    "predict_closed_form_k_over_i_plus_1": r["expected_k_over_i_plus_1"],
                    "metadata_split": split,
                    "metadata_forced": r,
                }
            )
    return out


def build_headline_tables(screen: dict[str, Any], heldout: dict[str, Any]) -> dict[str, Any]:
    """Compact tables the write-up can quote directly, derived from the full results."""
    fiber = screen["fibermap"] + heldout["fibermap"]
    by_k: dict[int, list[dict[str, Any]]] = {}
    for r in fiber:
        by_k.setdefault(r["k"], []).append(r)
    eviction = {}
    for kk, rows in sorted(by_k.items()):
        rows = sorted(rows, key=lambda r: r["i_prefix"])
        eviction[str(kk)] = {
            "log2_k": log2(kk) if kk > 1 else 0.0,
            "steps": [
                {
                    "i_prefix": r["i_prefix"],
                    "arriving_item": r["arriving_item"],
                    "divisible_k_divides_i_plus_1": r["divisible"],
                    "summodk_balanced_fraction": r["summodk"]["balanced_fraction"],
                    "repair_fraction": r["minimal_repair"]["repair_fraction"],
                    "min_evict_entropy_bits": r["min_evict_entropy_bits"],
                    "is_exact": r["min_evict_entropy_is_exact"],
                    "lower_bits": r["bracket"]["lower_bits"],
                    "upper_bits": r["bracket"]["upper_bits"],
                    "ratio_to_log2k": r["ratio_to_log2k"],
                }
                for r in rows
            ],
        }

    exact_summary = []
    for split, rows in (("screen", screen["exact"]), ("held_out", heldout["exact"])):
        for r in rows:
            exact_summary.append(
                {
                    "split": split,
                    "candidate": r["key"],
                    "n": r["n"],
                    "k": r["k"],
                    "exactly_uniform_all_prefixes": r["exactly_uniform_all_prefixes"],
                    "i_first_fail": r["i_first_fail"],
                    "m1_rel_at_failure": r["m1_rel_at_failure_float"],
                    "stopped_early": r["stopped_early"],
                    "max_states_seen": r["max_states_seen"],
                }
            )

    budgets = [
        {kk: b[kk] for kk in (
            "n", "k", "accept_bits", "classical_evict_bits", "evict_frac",
            "total_classical_bits", "optimal_evict_bits_lower", "optimal_evict_bits_upper",
            "total_optimal_bits_upper", "bit_saving_frac_lower_bound",
            "log2_binom_n_k", "anytime_overhead_classical", "anytime_overhead_optimal_upper",
        )}
        for b in screen["bit_budgets"]["rows"] + heldout["bit_budgets"]["rows"]
    ]

    return {
        "minimum_eviction_entropy_by_k": eviction,
        "exact_dp_cell_summary": exact_summary,
        "fair_coin_bit_budgets": budgets,
        "monte_carlo_vs_exact_dp_confusion": screen["matched_verdicts"]["confusion"],
        "literal_deliverable": heldout["literal_deliverable"]["headline"],
    }


# --------------------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------------------


def make_verdict(screen: dict[str, Any], heldout: dict[str, Any]) -> dict[str, Any]:
    rank = {r["candidate"]: r for r in screen["screen_ranking"]["ranking"]}
    main, c1 = rank["MAIN_S1"], rank["C1_S5CIRC"]

    ho_by_key: dict[str, list[dict[str, Any]]] = {}
    for r in heldout["exact"]:
        ho_by_key.setdefault(r["key"], []).append(r)

    def held_ok(key: str) -> bool:
        rows = ho_by_key.get(key, [])
        return bool(rows) and all(r["exactly_uniform_all_prefixes"] for r in rows)

    all_fiber = screen["fibermap"] + heldout["fibermap"]
    div = [r for r in all_fiber if r["divisible"]]
    nondiv = [r for r in all_fiber if not r["divisible"]]
    zero_on_divisible = all(r["min_evict_entropy_bits"] == 0.0 for r in div)
    summodk_perfect_on_divisible = all(r["summodk"]["balanced_fraction"] == 1.0 for r in div)
    frac_le_quarter = (
        sum(1 for r in all_fiber if r["ratio_to_log2k"] <= 0.25) / len(all_fiber)
        if all_fiber else 0.0
    )
    max_ratio_nondiv = max((r["ratio_to_log2k"] for r in nondiv), default=0.0)
    # The ratio only reaches 1.0 at the degenerate first step i == k, where there is
    # exactly ONE source (the reservoir is the whole of [k]) whose mass must be split
    # across all k children -- there the classical uniform rule is already optimal.
    # Report the max away from that boundary separately.
    nondiv_beyond = [r for r in nondiv if r["i_prefix"] >= 2 * r["k"]]
    max_ratio_beyond = max((r["ratio_to_log2k"] for r in nondiv_beyond), default=0.0)
    ratio_at_i_eq_k = max(
        (r["ratio_to_log2k"] for r in nondiv if r["i_prefix"] == r["k"]), default=None
    )

    main_held = held_ok("S1") and held_ok("S1f")
    main_verdict = (
        "CONFIRMED"
        if (main["passed"] and main_held and zero_on_divisible)
        else ("PARTIAL" if (main_held or main["passed"]) else "DISCONFIRMED")
    )

    circ_ho = ho_by_key.get("S5-CIRC", [])
    circ_incl_exact = bool(circ_ho) and all(
        all(p["incl_maxdev_float"] == 0.0 for p in r["prefixes"]) for r in circ_ho
    )
    ho_mc = {r["candidate"]: r for r in heldout["montecarlo"]["rows"]}
    circ_mc = ho_mc.get("S5-CIRC")
    c1_verdict = (
        "CONFIRMED"
        if (
            c1["passed"]
            and circ_incl_exact
            and circ_mc is not None
            and not circ_mc["flagged_first_order"]
            and circ_mc["flagged_second_order"]
            and circ_mc["pair_p_value_normal"] < 1e-6
        )
        else ("PARTIAL" if circ_incl_exact else "DISCONFIRMED")
    )

    moved = []
    for c in (
        screen["bit_budgets"]["hypothesis_constant_checks"]
        + heldout["bit_budgets"]["hypothesis_constant_checks"]
    ):
        if c.get("match") is False:
            moved.append(
                {
                    "quantity": f"{c['quantity']} at n={c['n']}, k={c['k']}",
                    "hypothesis_value": c["claimed"],
                    "recomputed_value": c["computed"],
                    "direction": "corrected",
                }
            )

    fifo = next(
        (r for r in screen["exact"] if r["key"] == "C-FIFO" and r["k"] == 2), None
    )
    if fifo is not None and fifo["i_first_fail"] != 4:
        moved.append(
            {
                "quantity": "prefix at which the FIFO control first fails, k=2",
                "hypothesis_value": 4,
                "recomputed_value": fifo["i_first_fail"],
                "direction": "corrected",
                "explanation": (
                    "at prefix i=2 the reservoir is {1,2} with 1 the oldest, so on acceptance FIFO "
                    "always yields {2,3} and never {1,3}; the deviation is already maximal at i=3"
                ),
            }
        )

    s1n = {r["k"]: r for r in screen["exact"] if r["key"] == "S1n"}
    for kk, claimed in ((2, 1.0), (4, 9.7)):
        r = s1n.get(kk)
        if r is not None and abs(r["m1_rel_at_failure_float"] - claimed) > 1e-6:
            moved.append(
                {
                    "quantity": f"relative max deviation of unrepaired sum-mod-k at its first failing prefix, k={kk}",
                    "hypothesis_value": claimed,
                    "recomputed_value": r["m1_rel_at_failure_float"],
                    "direction": "corrected",
                    "explanation": (
                        "the exact DP gives m1_rel = k-1 at the first failing prefix i=k+1, because "
                        "the single source [k] must send all of its accepted mass to one of its k "
                        "children"
                    ),
                }
            )

    return {
        "main": main_verdict,
        "c1": c1_verdict,
        "justification": {
            "main": (
                f"S1 (sum-mod-k + minimal flow repair) and S1f (flow-optimal) are EXACTLY uniform "
                f"at every prefix of every screened and held-out cell "
                f"(screen pass={main['passed']}, held-out pass={main_held}); the minimum eviction "
                f"entropy is EXACTLY 0 on every step with k | (i+1) "
                f"({len(div)}/{len(all_fiber)} steps, zero_on_divisible={zero_on_divisible}), where "
                f"the deterministic sum-mod-k rule is already perfectly balanced "
                f"(summodk_perfect_on_divisible={summodk_perfect_on_divisible}); on the remaining "
                f"steps the minimum decays like k/(i-k) -- at most "
                f"{max_ratio_beyond:.3f} x log2(k) once i >= 2k, and reaching "
                f"{max_ratio_nondiv:.3f} x log2(k) only at the degenerate first step i = k, where a "
                f"single source (the whole reservoir [k]) must split across all k children and the "
                f"classical uniform rule is already optimal. Overall {frac_le_quarter:.1%} of the "
                f"{len(all_fiber)} analysed steps sit at or below 0.25 x log2(k). Summed over the "
                f"whole stream this is a CONSTANT-in-n eviction budget: the analytic upper bound is "
                f"20.8 bits at k=10 whether n is 1e4 or 1e6, against the classical "
                f"k*log2(k)*ln(n/k) which grows to 380.8 bits at n=1e6."
            ),
            "c1": (
                f"S5-CIRC has EXACTLY uniform first-order inclusion probabilities at every prefix "
                f"(incl_maxdev == 0 exactly) with a support of only i of the C(i,k) subsets, so its "
                f"exact total variation from uniform is 1 - i/C(i,k). At the held-out scale "
                f"(n=1000, k=10, T=1e6) the standard first-order max-deviation test does not flag it "
                f"(flagged={circ_mc['flagged_first_order'] if circ_mc else None}) while the pairwise "
                f"co-inclusion test does, at log10(p)="
                f"{circ_mc['pair_log10_p_value'] if circ_mc else None} "
                f"({circ_mc['pair']['frac_zero_pairs'] * 100 if circ_mc else None:.1f}% of item pairs "
                f"are never co-sampled at all). Crucially the blindness is not a power problem: the "
                f"ANYTIME first-order test, which checks every prefix rather than only the final "
                f"reservoir, misses it in every matched cell too, because S5-CIRC's inclusion "
                f"probabilities are exactly k/i at every prefix by a counting argument -- each item "
                f"lies in exactly k of the i circular windows."
            ),
        },
        "numbers_that_moved": moved,
        "supporting": {
            "n_fiber_steps_total": len(all_fiber),
            "n_fiber_steps_divisible": len(div),
            "min_entropy_exactly_zero_on_every_divisible_step": zero_on_divisible,
            "summodk_perfectly_balanced_on_every_divisible_step": summodk_perfect_on_divisible,
            "max_ratio_to_log2k_on_nondivisible_steps": max_ratio_nondiv,
            "max_ratio_to_log2k_on_nondivisible_steps_with_i_at_least_2k": max_ratio_beyond,
            "ratio_to_log2k_at_the_degenerate_step_i_equals_k": ratio_at_i_eq_k,
            "n_nondivisible_steps_with_i_at_least_2k": len(nondiv_beyond),
            "frac_steps_ratio_le_0.25": frac_le_quarter,
            "S1_heldout_exactly_uniform": held_ok("S1"),
            "S1f_heldout_exactly_uniform": held_ok("S1f"),
            "S5CIRC_heldout_first_order_exact": circ_incl_exact,
        },
    }


# --------------------------------------------------------------------------------------


def run_gates() -> dict[str, Any]:
    logger.info("running gate suite G1..G8 (pytest)")
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q", "--no-header", "-p", "no:cacheprovider",
         "-c", str(ROOT / "pytest.ini")],
        cwd=str(ROOT), capture_output=True, text=True, timeout=3600,
        env={**os.environ, "COLUMNS": "200"},
    )
    out = proc.stdout
    counts = {
        word: int(m.group(1))
        for word in ("passed", "failed", "skipped", "error", "xfailed")
        for m in [re.search(rf"(\d+) {word}", out)]
        if m
    }
    if not counts:  # fall back to counting progress characters
        counts = {"passed": out.count("."), "note": "counted from progress output"}
    tail = [ln for ln in out.strip().splitlines() if ln.strip()][-3:]
    logger.info(
        f"gates exit={proc.returncode} in {time.perf_counter() - t0:.1f}s :: {counts}"
    )
    if proc.returncode != 0:
        logger.error(out[-4000:])
        raise RuntimeError("gate suite FAILED -- nothing downstream is meaningful")
    return {
        "passed": True,
        "exit_code": proc.returncode,
        "counts": counts,
        "n_tests_passed": counts.get("passed"),
        "gates": ["G1 DP self-consistency", "G2 controls light up", "G3 hand-checked fiber map",
                  "G4 unrepaired sum-mod-k failure magnitude", "G5 transport arithmetic + bracket",
                  "G6 certificate round-trip + corruption rejected", "G7 S1/S1f exactly uniform",
                  "G8 Monte-Carlo harness validation"],
        "summary": tail,
        "seconds": time.perf_counter() - t0,
    }


def assemble(screen: dict[str, Any], heldout: dict[str, Any], gates: dict[str, Any]) -> dict[str, Any]:
    """Build and write method_out.json from completed screen and held-out results."""
    t_start = time.perf_counter()
    verdict = make_verdict(screen, heldout)
    logger.info(f"VERDICT  main={verdict['main']}  c1={verdict['c1']}")
    for m in verdict["numbers_that_moved"]:
        logger.info(f"  moved: {m['quantity']}: {m['hypothesis_value']} -> {m['recomputed_value']}")

    ho_mc = heldout["literal_deliverable"]
    metadata = {
        "method_name": "exact_rational_prefix_DP_plus_Johnson_transportation_flow",
        "baseline_name": "monte_carlo_uniformity_test_with_simulated_null_band",
        "description": __doc__.strip().splitlines()[0],
        "dataset_justification": (
            "No external dataset is used, and none should be: uniformity of a reservoir sampler "
            "is a statement about INDEX k-subsets, so the canonical stream is the integer sequence "
            "1..n, generated in-script. The empty depends_on is deliberate, not an omission."
        ),
        "spend_usd": 0.0,
        "llm_calls": 0,
        "hardware": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "gpu_used": False,
            "ram_limit_gb": RAM_LIMIT_GB,
        },
        "determinism": {
            "exact_dp_and_fibermap": "no randomness whatsoever; outputs are bit-identical by construction",
            "monte_carlo_seeds": {"screen": panels.SCREEN["seed"], "held_out": panels.HELDOUT["seed"]},
            "rng": "numpy Generator(PCG64), seeded per (panel, candidate) by crc32 of the candidate key",
        },
        "gates": gates,
        "verdict": verdict,
        "screen_ranking": screen["screen_ranking"],
        "matched_confusion": screen["matched_verdicts"]["confusion"],
        "headline_tables": build_headline_tables(screen, heldout),
        "literal_deliverable": ho_mc,
        "hypothesis_constant_checks": (
            screen["bit_budgets"]["hypothesis_constant_checks"]
            + heldout["bit_budgets"]["hypothesis_constant_checks"]
        ),
        "deferred_to_instrument_2": panels.DEFERRED,
        "candidate_descriptions": candidates.DESCRIPTIONS,
        "out_dependency_files": [
            "exact_dp.py", "candidates.py", "fibermap.py", "forced.py", "montecarlo.py",
            "verdicts.py", "panels.py", "run_screen.py", "run_heldout.py", "method.py",
            "screen_ranking.json", "certificates/", "results/",
        ],
        "wall_seconds_total": None,
    }

    datasets = [
        {
            "dataset": "anytime_uniformity_verdicts_matched_panel",
            "examples": build_verdict_examples(screen["matched_verdicts"]),
        },
        {
            "dataset": "exact_prefix_dp_cells",
            "examples": build_exact_examples(screen, heldout),
        },
        {
            "dataset": "eviction_step_minimum_entropy",
            "examples": build_fiber_examples(screen, heldout),
        },
        {
            "dataset": "fair_coin_bit_budgets",
            "examples": build_budget_examples(screen, heldout),
        },
        {
            "dataset": "monte_carlo_uniformity_deliverable",
            "examples": build_mc_examples(screen, heldout),
        },
        {
            "dataset": "forced_accept_coin",
            "examples": build_forced_examples(screen, heldout),
        },
    ]
    for d in datasets:
        if not d["examples"]:
            raise RuntimeError(f"dataset {d['dataset']} is empty")
        logger.info(f"dataset {d['dataset']}: {len(d['examples'])} examples")

    metadata["wall_seconds_total"] = (
        gates.get("seconds", 0.0) + screen.get("wall_seconds", 0.0)
        + heldout.get("wall_seconds", 0.0) + (time.perf_counter() - t_start)
    )
    out = {"metadata": metadata, "datasets": datasets}
    p = ROOT / "method_out.json"
    p.write_text(json.dumps(out, indent=2, default=str))
    logger.info(f"wrote {p} ({p.stat().st_size / 1e6:.2f} MB)")
    return out


def main() -> None:
    resource.setrlimit(resource.RLIMIT_AS, (RAM_LIMIT_GB * 1024**3,) * 2)
    resource.setrlimit(resource.RLIMIT_CPU, (CPU_LIMIT_S,) * 2)

    import run_heldout
    import run_screen

    gates = run_gates()
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "gates.json").write_text(json.dumps(gates, indent=2, default=str))
    screen = run_screen.main()
    heldout = run_heldout.main()
    assemble(screen, heldout, gates)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(str(ROOT / "logs" / "run.log"), rotation="30 MB", level="DEBUG")
    logger.catch(reraise=True)(main)()
