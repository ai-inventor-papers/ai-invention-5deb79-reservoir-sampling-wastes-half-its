#!/usr/bin/env python3
"""Assemble eval_out.json (exp_eval_sol_out) from the stage outputs of parts A, B, C, D.

Datasets emitted
    detector_panel_three_columns   -- every (sampler, cell/seed) x every detector column
    sample_complexity_separation   -- one row per detector x panel, with state, streaming,
                                      locality, threshold T(n,k) and the T a worst-case
                                      eps-far alternative would need
    claims_ledger                  -- one row per printed number, MATCH/MISMATCH/UNAVAILABLE
    calibration_and_sign_reversal  -- the co-inclusion calibration boundary and the
                                      mechanism check behind the sign reversal
    reference_hygiene              -- one row per cited work
    merge_defect_panel             -- only if Part D executed
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from math import comb, log10, sqrt
from pathlib import Path

import numpy as np
from loguru import logger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "vendored"))
RES = HERE / "results"

E1_DIR = Path("/ai-inventor/aii_data/runs/run_2L_JomowDIGF/3_invention_loop/iter_1/"
              "gen_art/gen_art_experiment_1")
E2_DIR = Path("/ai-inventor/aii_data/runs/run_2L_JomowDIGF/3_invention_loop/iter_1/"
              "gen_art/gen_art_experiment_2")

SEEDS = {
    "matched_panel_mc": 20260918,
    "matched_panel_prefix_offset": 77,
    "matched_panel_l1_offset": 1234,
    "matched_panel_null_l1": 20260918 + 3121,
    "large_panel_per_replicate": "10000 + 991*r for r in 0..7",
    "large_panel_l1_null": "900000 + r for r in 0..R-1",
    "a1_power_curve": 4242,
    "a1_null": "777 + 100*n + k",
    "c1": 424242,
    "c2_simulation": 31337,
    "d_merge": 777,
}


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load_stage(name: str) -> dict | None:
    p = RES / f"stage_{name}.json"
    if not p.exists():
        logger.warning(f"stage {name} missing ({p})")
        return None
    return json.loads(p.read_text())


def ex(inp: str, out: str, **kw) -> dict:
    """One schema-valid example: input/output strings, predict_* strings, eval_* numbers."""
    d = {"input": str(inp), "output": str(out)}
    for k, v in kw.items():
        if k.startswith("predict_"):
            d[k] = str(v)
        elif k.startswith("eval_"):
            d[k] = float(v)
        elif k.startswith("metadata_"):
            d[k] = v
        else:
            raise KeyError(f"bad example field {k!r}")
    return d


# =======================================================================================
# dataset builders
# =======================================================================================


def build_detector_panel(a2: dict, a3: dict) -> list[dict]:
    exs: list[dict] = []
    for r in a2["rows"]:
        truth = r["truth_exact_dp"]
        exs.append(ex(
            f"Sampler {r['candidate']} at n={r['n']}, k={r['k']}, T={r['T']} "
            f"(matched panel, D=C({r['n']},{r['k']})={r['D']}). Is the reservoir "
            f"distribution a uniform k-subset at every prefix?",
            truth,
            predict_exact_dp_ground_truth=truth,
            predict_mc_final_first_order=r["mc_final_first_order"]["verdict"],
            predict_mc_final_pairwise=r["mc_final_pairwise"]["verdict"],
            predict_mc_all_prefix_anytime=r["mc_all_prefix"]["verdict"],
            predict_l1_identity_final=r["l1_identity_final"]["verdict"],
            predict_l1_identity_final_asymptotic=r["l1_identity_final"]["verdict_asymptotic"],
            predict_l1_identity_anytime=r["l1_identity_anytime"]["verdict"],
            metadata_panel="matched_18_cells",
            metadata_candidate=r["candidate"], metadata_n=r["n"], metadata_k=r["k"],
            metadata_T=r["T"], metadata_D=r["D"],
            metadata_i_first_fail=r["i_first_fail"],
            metadata_first_order=r["mc_final_first_order"],
            metadata_pairwise=r["mc_final_pairwise"],
            metadata_anytime=r["mc_all_prefix"],
            metadata_l1_final={k: v for k, v in r["l1_identity_final"].items()},
            metadata_l1_anytime={k: v for k, v in r["l1_identity_anytime"].items()
                                 if k != "per_prefix"},
            metadata_l1_anytime_per_prefix=r["l1_identity_anytime"]["per_prefix"],
            eval_first_order_correct=float(r["mc_final_first_order"]["verdict"] == truth),
            eval_pairwise_correct=float(r["mc_final_pairwise"]["verdict"] == truth),
            eval_anytime_correct=float(r["mc_all_prefix"]["verdict"] == truth),
            eval_l1_final_correct=float(r["l1_identity_final"]["verdict"] == truth),
            eval_l1_anytime_correct=float(r["l1_identity_anytime"]["verdict"] == truth),
            eval_l1_collisions=float(r["l1_identity_final"]["collisions_C"]),
            eval_l1_Z_normalised=float(r["l1_identity_final"]["Z_normalised"]),
        ))
    if a3 is not None:
        for r in a3["rows"]:
            truth = r["truth"]
            exs.append(ex(
                f"Sampler {r['candidate']} at n={r['n']}, k={r['k']}, T={r['T']}, "
                f"seed={r['seed']} (replicated panel, log10 D = {a3['log10_D']:.2f}). "
                f"Is the distribution over k-subsets uniform?",
                truth,
                predict_l1_identity_final=r["l1_verdict"],
                predict_l1_identity_final_asymptotic=r["l1_verdict_asymptotic"],
                metadata_panel="replicated_8_seeds",
                metadata_candidate=r["candidate"], metadata_n=r["n"], metadata_k=r["k"],
                metadata_T=r["T"], metadata_seed=r["seed"], metadata_family=r["family"],
                metadata_support=r["support"],
                metadata_l2_norm_squared=r["l2_norm_squared"],
                metadata_exact_TV_to_uniform=r["exact_TV_to_uniform"],
                metadata_collisions_measured=r["collisions_C_measured"],
                metadata_collisions_predicted=r["collisions_C_predicted"],
                metadata_n_distinct=r["n_distinct_measured"],
                metadata_Z=r["Z"], metadata_poisson_log10_p=r["poisson_log10_p"],
                metadata_l1_mc_p=r["l1_mc_p"],
                eval_l1_final_correct=float(r["l1_verdict"] == truth),
                eval_collisions_ratio_measured_over_predicted=float(
                    r["collisions_ratio_measured_over_predicted"] or 0.0),
                eval_exact_TV=float(r["exact_TV_to_uniform"]),
            ))
    return exs


def per_sampler_l1_analysis(a3: dict) -> list[dict]:
    """Per-sampler recall of the L1 tester, against the Poisson miss probability it implies.

    The tester detects a low-support alternative by observing at least one collision, and
    the collision count is Poisson(lambda) with lambda = binom(T,2)*||p||_2^2.  When lambda
    is large (BLOCK-LOCK, SYSTEMATIC, CIRC: 5e8-5e9) detection is certain.  When lambda is
    O(1) -- which is exactly PAIR-LOCK's case, lambda ~ 1.96 -- the tester MISSES with
    probability exp(-lambda) per run, no matter how far the alternative is in L1.
    """
    import collections
    from math import exp

    agg = collections.defaultdict(list)
    for r in a3["rows"]:
        agg[r["candidate"]].append(r)
    out = []
    for cid, rs in agg.items():
        lam = rs[0]["collisions_C_predicted"]
        n_det = sum(1 for r in rs if r["l1_verdict"] == "NOT_UNIFORM")
        truth_broken = rs[0]["truth"] == "NOT_UNIFORM"
        cs = [r["collisions_C_measured"] for r in rs]
        mean_c = sum(cs) / len(cs)
        out.append({
            "candidate": cid,
            "family": rs[0]["family"],
            "n_runs": len(rs),
            "support": rs[0]["support"],
            "l2_norm_squared": rs[0]["l2_norm_squared"],
            "exact_TV_to_uniform": rs[0]["exact_TV_to_uniform"],
            "lambda_predicted_collisions": lam,
            "collisions_measured": cs,
            "collisions_measured_mean": mean_c,
            "collision_prediction_rel_error": (abs(mean_c - lam) / lam) if lam > 0 else None,
            "l1_detections": n_det,
            "l1_recall": (n_det / len(rs)) if truth_broken else None,
            "l1_false_alarm_rate": (n_det / len(rs)) if not truth_broken else None,
            "poisson_predicted_miss_probability": exp(-lam) if (truth_broken and lam < 700)
                                                  else (0.0 if truth_broken else None),
            "observed_miss_fraction": ((len(rs) - n_det) / len(rs)) if truth_broken else None,
            "detection_is_fragile": bool(truth_broken and lam < 10),
        })
    return sorted(out, key=lambda r: (r["family"], r["candidate"]))


def build_separation(a2: dict, a3: dict, a1: dict) -> list[dict]:
    import partA

    conf = a2["confusion"]
    c_meas = (a1["power"]["c_measured_median"] if a1 and a1["power"]["c_measured_median"]
              else 1.0)
    rows = partA.build_separation_table(conf, a3["rows"] if a3 else [], c_measured=c_meas)
    exs = []
    for r in rows:
        exs.append(ex(
            f"Detector '{r['detector']}' on panel '{r['panel']}': what state does it keep, "
            f"is it streaming, does it localise the defect, what is its detection "
            f"threshold in trials, and what would a worst-case eps-far alternative cost?",
            r["detection_threshold_in_trials"],
            predict_detector=r["detector"],
            predict_state_per_trial=r["state_per_trial"],
            predict_threshold_T_of_n_k=r["detection_threshold_in_trials"],
            metadata_panel=r["panel"],
            metadata_total_memory_order=r["total_memory_order"],
            metadata_streaming_single_pass=r["streaming_single_pass"],
            metadata_memory_independent_of_T=r["memory_independent_of_T"],
            metadata_localises_the_defect=r["localises_the_defect"],
            metadata_T_required_worst_case_eps_0p1=r["T_required_worst_case_eps_0.1"],
            metadata_T_required_worst_case_eps_measured_TV=r[
                "T_required_worst_case_eps_measured_TV"],
            metadata_T_required_worst_case_eps_0p1_log10=r.get(
                "T_required_worst_case_eps_0.1_log10"),
            metadata_T_required_worst_case_eps_TV_log10=r.get(
                "T_required_worst_case_eps_measured_TV_log10"),
            metadata_c_measured_in_T_eq_c_sqrtD_over_eps2=c_meas,
            eval_measured_recall_on_broken=float(r["measured_recall_on_broken"] or 0.0),
            eval_measured_false_alarm_rate=float(r["measured_false_alarm_rate"] or 0.0),
            eval_measured_accuracy=float(r["measured_accuracy"]
                                         if r["measured_accuracy"] is not None else -1.0),
        ))
    return exs, rows


def build_ledger(b: dict) -> list[dict]:
    exs = []
    for c in b["claims"]:
        exs.append(ex(
            c["claim_text_as_printed"],
            c["status"],
            predict_status=c["status"],
            predict_corrected_text=c["corrected_text"],
            metadata_claim_id=c["claim_id"],
            metadata_source_artifact=c["source_artifact"],
            metadata_json_path=c["json_path"],
            metadata_tolerance=c["tolerance"],
            metadata_executed_value=c["executed_value"],
            metadata_note=c.get("note", ""),
            metadata_path_resolution=c.get("path_resolution"),
            eval_is_match=float(c["status"] == "MATCH"),
            eval_is_mismatch=float(c["status"] == "MISMATCH"),
            eval_is_unavailable=float(c["status"] == "UNAVAILABLE"),
        ))
    return exs


def nominal_rule_miscalibration(c1: dict) -> list[dict]:
    """Is the NOMINAL chi-square rule mis-calibrated by construction, or is a single-run
    false alarm just noise?

    A single (n,k,seed) run can exceed p < 1e-6 by chance or by a PRNG-path artifact -- and
    in this sweep it does so irreproducibly (n=1000,k=50 fires under one chunking of the
    PCG64 stream and not under another).  The decisive quantity is the SIMULATED null
    distribution of the co-inclusion chi-square, generated by an exactly uniform k-subset
    oracle that shares no code path with the sampler under audit: if its mean sits well
    above the nominal df, the rule is mis-calibrated at that cell no matter which run you
    look at.
    """
    from scipy.stats import chi2 as chi2dist, norm

    rows = []
    seen = set()
    for r in c1["calibrated_cells"]:
        if r.get("status") != "RUN" or (r["n"], r["k"]) in seen:
            continue
        seen.add((r["n"], r["k"]))
        n, k = r["n"], r["k"]
        df = n * (n - 1) // 2 - (n - 1)
        mu, sd = r["null_chi2_mean"], r["null_chi2_sd"]
        if mu is None or not sd:
            continue
        crit = float(chi2dist.isf(1e-6, df))
        rows.append({
            "n": n, "k": k, "k_over_n": k / n, "df": df,
            "null_chi2_mean_simulated": mu,
            "null_chi2_sd_simulated": sd,
            "null_mean_over_df": mu / df,
            "nominal_chi2_sd_if_calibrated": sqrt(2.0 * df),
            "simulated_sd_over_nominal_sd": sd / sqrt(2.0 * df),
            "nominal_p_at_the_simulated_null_mean": float(chi2dist.sf(mu, df)),
            "nominal_critical_value_at_p_1e-6": crit,
            "implied_false_alarm_rate_of_the_nominal_rule":
                float(norm.sf((crit - mu) / sd)),
            "null_B_coinc": r["null_B_coinc"],
            "verdict": ("NOMINAL_RULE_MISCALIBRATED"
                        if float(norm.sf((crit - mu) / sd)) > 1e-3
                        else "NOMINAL_RULE_CALIBRATED_AT_THIS_CELL"),
        })
    return rows


def build_calibration(c1: dict, c2: dict, a1: dict) -> list[dict]:
    exs = []
    if c1:
        s = c1["summary"]
        for r in nominal_rule_miscalibration(c1):
            exs.append(ex(
                f"Is the pre-registered NOMINAL co-inclusion rule (chi-square p < 1e-6) "
                f"calibrated at n={r['n']}, k={r['k']} (k/n = {r['k_over_n']:.4f})? The "
                f"co-inclusion chi-square has df = {r['df']}; its SIMULATED null, from an "
                f"exactly uniform k-subset oracle over {r['null_B_coinc']} replicates, has "
                f"mean {r['null_chi2_mean_simulated']:.1f} and sd "
                f"{r['null_chi2_sd_simulated']:.1f}.",
                r["verdict"],
                predict_verdict=r["verdict"],
                metadata_part="C1_nominal_rule_miscalibration",
                metadata_n=r["n"], metadata_k=r["k"], metadata_k_over_n=r["k_over_n"],
                metadata_df=r["df"],
                metadata_null_chi2_mean_simulated=r["null_chi2_mean_simulated"],
                metadata_null_chi2_sd_simulated=r["null_chi2_sd_simulated"],
                metadata_nominal_chi2_sd_if_calibrated=r["nominal_chi2_sd_if_calibrated"],
                metadata_nominal_critical_value=r["nominal_critical_value_at_p_1e-6"],
                metadata_null_B_coinc=r["null_B_coinc"],
                eval_null_mean_over_df=float(r["null_mean_over_df"]),
                eval_simulated_sd_over_nominal_sd=float(r["simulated_sd_over_nominal_sd"]),
                eval_implied_false_alarm_rate=float(
                    r["implied_false_alarm_rate_of_the_nominal_rule"]),
                eval_log10_nominal_p_at_null_mean=float(log10(max(
                    r["nominal_p_at_the_simulated_null_mean"], 1e-300))),
            ))
        for r in c1["nominal_sweep"]:
            if r.get("status") != "RUN":
                continue
            exs.append(ex(
                f"Nominal co-inclusion rule (chi-square p < 1e-6) applied to Algorithm R "
                f"itself at n={r['n']}, k={r['k']}, T={r['T']} (k/n = {r['k_over_n']:.4f}). "
                f"Does the pre-registered rule false-alarm on a provably uniform sampler?",
                "FALSE_ALARM" if r["nominal_rule_false_alarm"] else "NO_FALSE_ALARM",
                predict_nominal_rule="FALSE_ALARM" if r["nominal_rule_false_alarm"]
                                    else "NO_FALSE_ALARM",
                metadata_part="C1_nominal_sweep",
                metadata_n=r["n"], metadata_k=r["k"], metadata_T=r["T"],
                metadata_k_over_n=r["k_over_n"],
                metadata_coinc_p_nominal=r["coinc_p_nominal"],
                metadata_coinc_chi2_over_df=r["coinc_chi2_over_df"],
                metadata_coinc_Zmax=r["coinc_Zmax"],
                eval_nominal_false_alarm=float(r["nominal_rule_false_alarm"]),
                eval_log10_p=float(log10(max(r["coinc_p_nominal"], 1e-300))),
            ))
        for r in c1["calibrated_cells"]:
            if r.get("status") != "RUN":
                continue
            exs.append(ex(
                f"Calibrated co-inclusion rule (own simulated null, z >= 5) applied to "
                f"{r['candidate']} at n={r['n']}, k={r['k']}, T={r['T']}. Uniform?",
                r["truth"],
                predict_calibrated_rule=r["calibrated_verdict"],
                predict_nominal_rule=r["nominal_verdict"],
                metadata_part="C1_calibrated_cells",
                metadata_candidate=r["candidate"], metadata_n=r["n"], metadata_k=r["k"],
                metadata_coinc_chi2_z=r["coinc_chi2_z_vs_simulated_null"],
                metadata_coinc_p_nominal=r["coinc_p_nominal"],
                metadata_null_B=r["null_B"], metadata_null_B_coinc=r["null_B_coinc"],
                metadata_maxdev_raw=r["maxdev_raw"], metadata_maxdev_z_band=r["maxdev_z_band"],
                eval_calibrated_correct=float(r["calibrated_verdict"] == r["truth"]),
                eval_nominal_correct=float(r["nominal_verdict"] == r["truth"]),
                eval_calibrated_z=float(r["coinc_chi2_z_vs_simulated_null"] or 0.0),
            ))
        exs.append(ex(
            "Summary of C1: what is the smallest (n,k) at which the pre-registered nominal "
            "chi-square co-inclusion rule first false-alarms on Algorithm R, and does the "
            "calibrated rule separate correct from broken with no overlap?",
            json.dumps(s["smallest_false_alarm_by_k"], default=str)
            if s["smallest_false_alarm_by_k"] else "NO_FALSE_ALARM_ON_THE_GRID",
            predict_calibration_boundary=json.dumps(
                {"by_k": s["smallest_false_alarm_by_k"],
                 "by_k_over_n": s["smallest_false_alarm_by_k_over_n"]}, default=str),
            metadata_part="C1_summary", metadata_summary=s,
            eval_n_cells_run=float(s["n_cells_run"]),
            eval_n_cells_nominal_false_alarm=float(s["n_cells_nominal_false_alarm"]),
            eval_calibrated_gap=float(s["calibrated_gap"] or 0.0),
            eval_calibrated_separation_clean=float(s["calibrated_separation_clean"]),
        ))
    if c2:
        for r in c2["rows"]:
            exs.append(ex(
                f"Sign-reversal mechanism for {r['candidate']}: its support locks the n "
                f"items into {r['effective_bins']} groups ({r['effective_bins_formula']}), "
                f"so the max-deviation statistic is a maximum over "
                f"{r['effective_bins']} -- not n=1000 -- effectively independent "
                f"deviations. Does the predicted shift match the observed one?",
                "INSIDE_PREDICTED_INTERVAL" if r["observed_shift_inside_predicted_interval"]
                else ("OUTSIDE_PREDICTED_INTERVAL"
                      if r["observed_shift_inside_predicted_interval"] is False
                      else "NO_OBSERVATION"),
                predict_mechanism="INSIDE_PREDICTED_INTERVAL"
                                  if r["observed_shift_inside_predicted_interval"]
                                  else "OUTSIDE_PREDICTED_INTERVAL"
                                  if r["observed_shift_inside_predicted_interval"] is False
                                  else "NO_OBSERVATION",
                metadata_part="C2_sign_reversal",
                metadata_candidate=r["candidate"],
                metadata_effective_bins=r["effective_bins"],
                metadata_effective_bins_formula=r["effective_bins_formula"],
                metadata_why=r["why"],
                metadata_sim_mean=r["sim_mean_max_dev"], metadata_sim_sd=r["sim_sd_max_dev"],
                metadata_sim_analytic_gumbel_mean=r["sim_analytic_gumbel_mean"],
                metadata_baseline_sim_mean=r["baseline_sim_mean_max_dev"],
                metadata_predicted_shift=r["predicted_shift"],
                metadata_predicted_shift_95_interval=r["predicted_shift_95_interval"],
                metadata_observed_mean=r["observed_mean_max_dev"],
                metadata_observed_shift=r["observed_shift_vs_S0"],
                metadata_n_replicates_averaged=r["n_replicates_averaged"],
                eval_analytic_gumbel_ratio=float(
                    r["analytic_gumbel_ratio_sqrt2lnm_over_sqrt2lnn"]),
                eval_sim_mean_ratio_to_baseline=float(r["sim_mean_ratio_to_baseline"]),
                eval_observed_ratio_to_S0=float(r["observed_ratio_to_S0"] or 0.0),
                eval_inside_predicted_interval=float(
                    bool(r["observed_shift_inside_predicted_interval"])),
            ))
        exs.append(ex(
            "Verdict on the sign-reversal mechanism: is the claim that a block-locked "
            "support has fewer independent bins, and is therefore rewarded by an "
            "extreme-value statistic, quantitatively supported?",
            c2["verdict"],
            predict_verdict=c2["verdict"],
            metadata_part="C2_summary",
            metadata_scored_better=c2["blind_spot_samplers_scoring_better_than_S0_on_the_headline_run"],
            metadata_residual_note=c2["residual_note"],
            metadata_analytic_shrinkage=c2["analytic_shrinkage_sqrt2ln100_over_sqrt2ln1000"],
            eval_n_scoring_better=float(c2["n_scoring_better"]),
            eval_mechanism_confirmed=float(c2["verdict"] == "MECHANISM_CONFIRMED"),
        ))
    if a1:
        for key, v in a1["power"]["fits"].items():
            D = int(key.split(",")[0].split("=")[1])
            eps = float(key.split("eps=")[1])
            exs.append(ex(
                f"Sample complexity calibration of the L1 identity/collision tester on a "
                f"planted eps-far alternative with D={D} atoms and eps={eps} (exact L1 by "
                f"construction): at what T does power cross 0.9, and what constant c does "
                f"that imply in T = c*sqrt(D)/eps^2?",
                f"T_90={v['T_90']}, c={v['c_measured']:.4f}",
                predict_T_90=str(v["T_90"]),
                metadata_part="A1_power_calibration",
                metadata_D=D, metadata_eps=eps,
                eval_T_90=float(v["T_90"]), eval_c_measured=float(v["c_measured"]),
            ))
        exs.append(ex(
            "Does the measured sample complexity of the implemented tester follow the "
            "published law T = Theta(sqrt(D)/eps^2)?",
            a1["power"]["law"],
            predict_law=a1["power"]["law"],
            metadata_part="A1_summary",
            metadata_D_scaling=a1["D_scaling"], metadata_eps_scaling=a1["eps_scaling"],
            metadata_nulls={k: {kk: vv for kk, vv in v.items() if kk != "values_Z"}
                            for k, v in a1["nulls"].items()},
            eval_c_measured_median=float(a1["power"]["c_measured_median"] or 0.0),
            eval_c_spread_ratio=float(a1["power"]["c_spread_ratio"] or 0.0),
        ))
    return exs


def build_refs(refs: list[dict]) -> list[dict]:
    exs = []
    for r in refs:
        canonical = (r.get("canonical") or "").strip()
        if not canonical:
            canonical = (f"NO AUTHORITATIVE RECORD RESOLVED ({r['status']}). "
                         f"{r.get('note', '')}".strip())
        exs.append(ex(
            f"Reference {r['ref_id']} as printed: {r['as_printed']}. What is the "
            f"authoritative record?",
            canonical,
            predict_status=r["status"],
            predict_canonical=canonical,
            metadata_ref_id=r["ref_id"], metadata_as_printed=r["as_printed"],
            metadata_source_url=r.get("source_url", ""), metadata_note=r.get("note", ""),
            eval_verified=float(r["status"] == "VERIFIED"),
            eval_corrected=float(r["status"] == "CORRECTED"),
            eval_unresolved=float(r["status"] == "UNRESOLVED"),
            eval_non_peer_reviewed=float(r["status"] == "NON_PEER_REVIEWED"),
        ))
    return exs


def merge_geometry(d: dict) -> dict:
    """Support, ||p||_2^2, exact TV and the implied collision count for each merge rule.

    CORRECT merge   : uniform on all C(n1+n2, k) k-subsets.
    DEFECTIVE merge : a uniform j-subset of [n1] times a uniform (k-j)-subset of [n2],
                      i.e. uniform on the C(n1,j)*C(n2,k-j) BALANCED subsets only.
    The defective support is a constant fraction of the full one, so the alternative is
    FAR in L1 but its L2 norm is almost unchanged -- which is precisely the regime the
    collision tester cannot see at any affordable T.
    """
    r0 = d["rows"][0]
    n1, n2, k, j, T = r0["n1"], r0["n2"], r0["k"], r0["j_fixed"], r0["T"]
    n = n1 + n2
    sup_ok = comb(n, k)
    sup_bad = comb(n1, j) * comb(n2, k - j)
    npairs = T * (T - 1) / 2.0
    return {
        "n1": n1, "n2": n2, "n": n, "k": k, "j_fixed": j, "T": T,
        "support_correct": sup_ok,
        "support_defective": sup_bad,
        "support_ratio_defective_over_correct": sup_bad / sup_ok,
        "l2_norm_squared_correct": 1.0 / sup_ok,
        "l2_norm_squared_defective": 1.0 / sup_bad,
        "exact_TV_defective_to_uniform": 1.0 - sup_bad / sup_ok,
        "expected_collisions_correct": npairs / sup_ok,
        "expected_collisions_defective": npairs / sup_bad,
        "measured_collisions": {r["merge"]: r["l1_collisions_C"] for r in d["rows"]},
        "why_the_L1_tester_is_blind_here": (
            f"The defective merge keeps {sup_bad/sup_ok:.4f} of the full support, so its "
            f"L2 norm is only {sup_ok/sup_bad:.3f}x the uniform one and the expected "
            f"collision count at T={T:.0g} rises from {npairs/sup_ok:.2e} to just "
            f"{npairs/sup_bad:.2e} -- both indistinguishable from zero. The alternative is "
            f"nevertheless FAR in L1 (TV = {1.0 - sup_bad/sup_ok:.4f}). This is the "
            f"spread-out defect the worst-case Theta(sqrt(D)/eps^2) bound warns about: at "
            f"eps = {1.0 - sup_bad/sup_ok:.4f} it would need T ~ sqrt(C({n},{k}))/eps^2, "
            f"which is {sqrt(float(sup_ok))/(1.0 - sup_bad/sup_ok)**2:.3g} samples."),
    }


def build_merge(d: dict) -> list[dict]:
    geo = merge_geometry(d)
    exs = []
    for r in d["rows"]:
        exs.append(ex(
            f"Distributed merge of two size-k reservoirs (n1={r['n1']}, n2={r['n2']}, "
            f"k={r['k']}, T={r['T']}): the {r['merge']} rule. Every item's marginal "
            f"inclusion probability is k/(n1+n2) under BOTH rules. Is the merged "
            f"distribution a uniform k-subset of the union?",
            r["truth"],
            predict_first_order=r["first_order_verdict"],
            predict_pairwise=r["pairwise_verdict"],
            metadata_merge=r["merge"], metadata_n1=r["n1"], metadata_n2=r["n2"],
            metadata_k=r["k"], metadata_T=r["T"], metadata_j_fixed=r["j_fixed"],
            metadata_first_order_max_dev=r["first_order_max_dev"],
            metadata_first_order_mc_p=r["first_order_mc_p"],
            metadata_pairwise_max_abs_z=r["pairwise_max_abs_z"],
            metadata_pairwise_mc_p=r["pairwise_mc_p"],
            metadata_pairwise_frac_zero_pairs=r["pairwise_frac_zero_pairs"],
            metadata_l1_collisions=r["l1_collisions_C"], metadata_l1_Z=r["l1_Z"],
            metadata_exact_probabilities=d["exact_probabilities"],
            metadata_marginals_measured=d["marginals_measured"],
            metadata_geometry=geo,
            predict_l1_identity="NOT_UNIFORM" if r["l1_collisions_C"] > 0 else "UNIFORM",
            eval_first_order_correct=float(r["first_order_verdict"] == r["truth"]),
            eval_pairwise_correct=float(r["pairwise_verdict"] == r["truth"]),
            eval_l1_correct=float(
                ("NOT_UNIFORM" if r["l1_collisions_C"] > 0 else "UNIFORM") == r["truth"]),
        ))
    exs.append(ex(
        "Why does the L1 identity/collision tester, which caught every blind-spot sampler "
        "on the replicated panel, miss this merge defect entirely?",
        geo["why_the_L1_tester_is_blind_here"],
        predict_explanation=geo["why_the_L1_tester_is_blind_here"],
        metadata_geometry=geo,
        eval_exact_TV_defective=float(geo["exact_TV_defective_to_uniform"]),
        eval_expected_collisions_defective=float(geo["expected_collisions_defective"]),
        eval_expected_collisions_correct=float(geo["expected_collisions_correct"]),
        eval_support_ratio=float(geo["support_ratio_defective_over_correct"]),
    ))
    return exs


# =======================================================================================
# main
# =======================================================================================


def main() -> None:
    t0 = time.perf_counter()
    a1, a2, a3 = load_stage("a1"), load_stage("a2"), load_stage("a3")
    b = load_stage("b")
    c1, c2 = load_stage("c1"), load_stage("c2")

    d = load_stage("d")
    det_check = load_stage("determinism")
    probe = load_stage("chunkprobe")
    refs_path = HERE / "scratch" / "refs.json"
    refs = json.loads(refs_path.read_text()) if refs_path.exists() else []
    if isinstance(refs, dict):
        refs = refs.get("references") or refs.get("rows") or list(refs.values())

    if a2 is None or b is None:
        raise SystemExit("stage a2 and stage b are mandatory")

    det = build_detector_panel(a2, a3)
    psl = per_sampler_l1_analysis(a3) if a3 else []
    for r in psl:
        det.append(ex(
            f"Per-sampler summary over {r['n_runs']} seeds at n=1000, k=10, T=1e6: "
            f"{r['candidate']} has support {r['support']} (||p||_2^2 = "
            f"{r['l2_norm_squared']:.4g}, exact TV = {r['exact_TV_to_uniform']:.12f}), so "
            f"the expected collision count is lambda = binom(T,2)*||p||_2^2 = "
            f"{r['lambda_predicted_collisions']:.4g}. Does the L1 collision tester detect "
            f"it on every seed?",
            ("DETECTED_ON_EVERY_SEED" if r["l1_recall"] == 1.0 else
             f"DETECTED_ON_{r['l1_detections']}_OF_{r['n_runs']}_SEEDS")
            if r["family"] == "blind_spot" else "CORRECT_SAMPLER_NO_DETECTION_EXPECTED",
            predict_l1_per_sampler_recall=str(r["l1_recall"]),
            metadata_panel="replicated_8_seeds_per_sampler_summary",
            metadata_candidate=r["candidate"], metadata_family=r["family"],
            metadata_support=r["support"],
            metadata_l2_norm_squared=r["l2_norm_squared"],
            metadata_exact_TV=r["exact_TV_to_uniform"],
            metadata_lambda_predicted=r["lambda_predicted_collisions"],
            metadata_collisions_measured=r["collisions_measured"],
            metadata_collision_prediction_rel_error=r["collision_prediction_rel_error"],
            metadata_poisson_predicted_miss_probability=r["poisson_predicted_miss_probability"],
            metadata_observed_miss_fraction=r["observed_miss_fraction"],
            metadata_detection_is_fragile=r["detection_is_fragile"],
            eval_l1_recall=float(r["l1_recall"] if r["l1_recall"] is not None else -1.0),
            eval_lambda=float(r["lambda_predicted_collisions"]),
            eval_collision_prediction_rel_error=float(
                r["collision_prediction_rel_error"] or 0.0),
        ))
    sep_ex, sep_rows = build_separation(a2, a3, a1)
    if d and d.get("status") == "EXECUTED":
        mg = merge_geometry(d)
        c_meas = (a1["power"]["c_measured_median"] if a1 else 1.0)
        n_m, k_m = mg["n"], mg["k"]
        merge_rows = [
            ("first-order max-deviation (the field-standard test)", "n counters",
             "O(n)", True, True, False,
             "NEVER, at any T: the defective merge's per-item marginals are exactly "
             "k/(n1+n2) by construction",
             d["first_order_recall"], d["first_order_false_alarm"], None),
            ("pairwise co-inclusion (calibrated against its own simulated null)",
             "C(n,2) pair counters", "O(n^2)", True, True, True,
             "O(n^2 / eps_pair^2); the defective merge shifts the same-stream pair "
             "probability from k(k-1)/(n(n-1)) to j(j-1)/(n1(n1-1))",
             d["pairwise_recall"], d["pairwise_false_alarm"], None),
            ("L1 identity / collision tester (ADK 2015 / DKN 2015)",
             "the multiset of observed k-subsets", "O(min(T, D)) = O(T) here", True,
             False, False,
             f"Theta(sqrt(C({n_m},{k_m}))/eps^2); at the defect's exact "
             f"eps = {mg['exact_TV_defective_to_uniform']:.4f} that is "
             f"{sqrt(float(mg['support_correct']))/mg['exact_TV_defective_to_uniform']**2 * c_meas:.3g} samples",
             float(sum(1 for r in d["rows"]
                       if r["truth"] == "NOT_UNIFORM" and r["l1_collisions_C"] > 0)
                   / max(1, sum(1 for r in d["rows"] if r["truth"] == "NOT_UNIFORM"))),
             0.0, mg),
        ]
        for (det_name, state, mem, stream, mem_ind, loc, thr, rec, fa, geo) in merge_rows:
            row = {
                "detector": det_name,
                "panel": f"distributed-merge defect (n1=n2=500, k=10, T=1e6)",
                "state_per_trial": state, "total_memory_order": mem,
                "streaming_single_pass": stream, "memory_independent_of_T": mem_ind,
                "localises_the_defect": loc,
                "detection_threshold_in_trials": thr,
                "measured_accuracy": None,
                "measured_recall_on_broken": rec,
                "measured_false_alarm_rate": fa,
                "T_required_worst_case_eps_0.1":
                    (sqrt(float(mg["support_correct"])) / 0.1**2 * c_meas
                     if geo else "not applicable"),
                "T_required_worst_case_eps_measured_TV":
                    (sqrt(float(mg["support_correct"]))
                     / mg["exact_TV_defective_to_uniform"]**2 * c_meas
                     if geo else "not applicable"),
                "eps_used_for_measured_TV": (mg["exact_TV_defective_to_uniform"]
                                             if geo else None),
            }
            for fld in ("T_required_worst_case_eps_0.1",
                        "T_required_worst_case_eps_measured_TV"):
                if isinstance(row[fld], float):
                    row[fld + "_log10"] = log10(row[fld])
            sep_rows.append(row)
            sep_ex.append(ex(
                f"Detector '{row['detector']}' on panel '{row['panel']}': the merge "
                f"defect has per-item marginals identical to the correct merge by "
                f"construction, and keeps a constant fraction "
                f"({mg['support_ratio_defective_over_correct']:.4f}) of the full support. "
                f"What does this detector see?",
                row["detection_threshold_in_trials"],
                predict_detector=row["detector"],
                predict_state_per_trial=row["state_per_trial"],
                predict_threshold_T_of_n_k=row["detection_threshold_in_trials"],
                metadata_panel=row["panel"],
                metadata_total_memory_order=row["total_memory_order"],
                metadata_streaming_single_pass=row["streaming_single_pass"],
                metadata_memory_independent_of_T=row["memory_independent_of_T"],
                metadata_localises_the_defect=row["localises_the_defect"],
                metadata_T_required_worst_case_eps_0p1=row["T_required_worst_case_eps_0.1"],
                metadata_T_required_worst_case_eps_measured_TV=row[
                    "T_required_worst_case_eps_measured_TV"],
                metadata_geometry=mg,
                eval_measured_recall_on_broken=float(rec),
                eval_measured_false_alarm_rate=float(fa),
                eval_measured_accuracy=-1.0,
            ))
    led = build_ledger(b)
    if c1:
        # Part C1 re-derived the calibration the C18 claim is about, so the ledger's
        # corrected sentence is extended with the quantity that actually settles it.
        _mis = nominal_rule_miscalibration(c1)
        _worst = max(_mis, key=lambda r: r["implied_false_alarm_rate_of_the_nominal_rule"])
        for _e in led:
            if _e["metadata_claim_id"] == "C18_declared_deviation_evidence":
                _e["predict_corrected_text"] += (
                    f" Part C1 of this evaluation supplies that simulated null: over "
                    f"{len(_mis)} calibrated cells the co-inclusion chi-square null is "
                    f"over-dispersed by up to "
                    f"{max(r['simulated_sd_over_nominal_sd'] for r in _mis):.2f}x the "
                    f"nominal chi-square(df) standard deviation, so the nominal p < 1e-6 "
                    f"rule has an implied false-alarm rate on a provably uniform sampler "
                    f"of {_worst['implied_false_alarm_rate_of_the_nominal_rule']:.3f} at "
                    f"n = {_worst['n']}, k = {_worst['k']} -- which is why a single run at "
                    f"that cell may or may not fire, and why the rule must not be used. "
                    f"Across a {c1['summary']['n_cells_run']}-cell sweep the nominal rule "
                    f"false-alarms on Algorithm R at "
                    f"{c1['summary']['n_cells_nominal_false_alarm']} cell(s), the smallest "
                    f"being n = {c1['summary']['smallest_false_alarm_by_k']['n']}, "
                    f"k = {c1['summary']['smallest_false_alarm_by_k']['k']}.")
    cal = build_calibration(c1, c2, a1)
    ref_ex = build_refs(refs)

    datasets = [
        {"dataset": "detector_panel_three_columns", "examples": det},
        {"dataset": "sample_complexity_separation", "examples": sep_ex},
        {"dataset": "claims_ledger", "examples": led},
        {"dataset": "calibration_and_sign_reversal", "examples": cal},
    ]
    if ref_ex:
        datasets.append({"dataset": "reference_hygiene", "examples": ref_ex})
    not_executed = []
    if d and d.get("status") == "EXECUTED":
        datasets.append({"dataset": "merge_defect_panel", "examples": build_merge(d)})
    else:
        import partD
        not_executed.append({"part": "D_distributed_merge_defect", **partD.SPEC_IF_NOT_EXECUTED})
    if a1 is None:
        not_executed.append({"part": "A1_power_calibration", "status": "NOT_EXECUTED"})
    if a3 is None:
        not_executed.append({"part": "A3_replicated_panel", "status": "NOT_EXECUTED"})
    if c1 is None:
        not_executed.append({"part": "C1_calibration_boundary", "status": "NOT_EXECUTED"})
    if not refs:
        not_executed.append({"part": "C3_reference_hygiene", "status": "NOT_EXECUTED"})

    scope_limits = [
        {"item": "C1 nominal sweep grid",
         "limit": "k capped at 100 and n at 5000",
         "why": "the dense co-inclusion accumulator materialises chunk*C(k,2) linearised "
                "pair indices; at k=200 that is 1e9 int64 entries (8 GB) per chunk. The "
                "trial-chunk size is scaled as 2e7/C(k,2) to keep the buffer bounded, and "
                "the chunk actually used is recorded, because the chunk turns out to "
                "matter (see metadata.chunk_probe)."},
        {"item": "C1 calibrated cells",
         "limit": ("4 cells rather than the full grid, with "
                   + (f"{c1['null_B']} max-deviation and {c1['null_B_coinc']} "
                      f"co-inclusion null replicates per cell" if c1
                      else "the planned null budget")),
         "why": "an own-simulated null costs one exactly-uniform k-subset oracle run per "
                "replicate, and the co-inclusion replicates dominate: at n=2000, k=100 the "
                "band alone takes 263 s at 16 co-inclusion replicates and 1620 s at the "
                "plan's 48 (measured -- a first attempt at 200/48 was abandoned after the "
                "vendored rejection sampler raised RuntimeError at (500,50), see "
                "metadata.vendored_adaptations). The budget was cut to 40/16 so all four "
                "cells complete; the quantities read off the band are its MEAN and SD, "
                "which 16 replicates estimate to within ~18% relative standard error -- "
                "far finer than the 4-to-6-order-of-magnitude separation being measured. "
                "The declared n=2000,k=100 cell is included."},
        {"item": "A1 eps grid",
         "limit": "eps in {0.1, 0.2, 0.35, 0.5}, D in {1024, 4096, 16384, 65536}",
         "why": "eps = 0.02 would put T_90 at ~4e6 per grid point and ~1e9 draws per cell; "
                "the four eps values retained already pin the exponent of eps to "
                "-1.96..-2.11 and of D to 0.48..0.54."},
        {"item": "A3 L1 null replicates",
         "limit": "200 replicates at T=1e6",
         "why": "200 is the smallest R for which an exact Monte-Carlo p-value can reach "
                "the 0.01 level (1/201 = 0.00498); the null collision count is 0 in every "
                "one of the 200 replicates, matching the analytic lambda = 1.9e-12."},
    ]

    conf = a2["confusion"]
    counts = b["counts"]

    # ---------------- metrics_agg (flat numbers only) ---------------------------------
    m: dict[str, float] = {
        "matched_panel_n_cells": float(len(a2["rows"])),
        "matched_first_order_accuracy": float(conf["mc_final_first_order"]["accuracy"]),
        "matched_first_order_recall": float(conf["mc_final_first_order"]["recall_on_broken_samplers"]),
        "matched_first_order_false_alarm": float(conf["mc_final_first_order"]["false_alarm_rate_on_correct_samplers"]),
        "matched_anytime_accuracy": float(conf["mc_all_prefix"]["accuracy"]),
        "matched_anytime_recall": float(conf["mc_all_prefix"]["recall_on_broken_samplers"]),
        "matched_pairwise_accuracy": float(conf["mc_final_pairwise"]["accuracy"]),
        "matched_pairwise_recall": float(conf["mc_final_pairwise"]["recall_on_broken_samplers"]),
        "matched_l1_final_accuracy": float(conf["l1_identity_final"]["accuracy"]),
        "matched_l1_final_recall": float(conf["l1_identity_final"]["recall_on_broken_samplers"]),
        "matched_l1_final_false_alarm": float(conf["l1_identity_final"]["false_alarm_rate_on_correct_samplers"]),
        "matched_l1_anytime_accuracy": float(conf["l1_identity_anytime"]["accuracy"]),
        "matched_l1_anytime_recall": float(conf["l1_identity_anytime"]["recall_on_broken_samplers"]),
        "ledger_n_claims": float(counts["n_claims"]),
        "ledger_n_match": float(counts["n_match"]),
        "ledger_n_mismatch": float(counts["n_mismatch"]),
        "ledger_n_unavailable": float(counts["n_unavailable"]),
        "ledger_match_rate": float(counts["n_match"] / counts["n_claims"]),
        "spend_usd": 0.0,
        "llm_calls": 0.0,
    }
    if a3:
        broken = [r for r in a3["rows"] if r["family"] == "blind_spot"]
        correct = [r for r in a3["rows"] if r["family"] == "correct"]
        m["large_panel_n_runs"] = float(len(a3["rows"]))
        m["large_panel_l1_recall_on_broken"] = float(
            sum(1 for r in broken if r["l1_verdict"] == "NOT_UNIFORM") / len(broken))
        m["large_panel_l1_false_alarm_on_correct"] = float(
            sum(1 for r in correct if r["l1_verdict"] == "NOT_UNIFORM") / len(correct))
        # split the collision-prediction check by regime: when lambda is large the
        # prediction is a sharp correctness check; when lambda is O(1) the measured count
        # is a Poisson draw and a large relative error is EXPECTED, not a discrepancy.
        det_regime = [r for r in broken if r["collisions_C_predicted"] > 1000]
        poi_regime = [r for r in broken if r["collisions_C_predicted"] <= 1000]
        rr = [r["collisions_ratio_measured_over_predicted"] for r in det_regime
              if r["collisions_ratio_measured_over_predicted"]]
        m["large_panel_collision_prediction_max_rel_error_lambda_gt_1000"] = float(
            max(abs(x - 1.0) for x in rr)) if rr else 0.0
        m["large_panel_n_runs_in_poisson_regime_lambda_le_1000"] = float(len(poi_regime))
        m["large_panel_log10_D"] = float(a3["log10_D"])
        frag = [r for r in psl if r["detection_is_fragile"]]
        m["large_panel_n_fragile_blind_spots"] = float(len(frag))
        m["large_panel_min_per_sampler_l1_recall"] = float(min(
            (r["l1_recall"] for r in psl if r["l1_recall"] is not None), default=1.0))
        m["large_panel_min_blind_spot_TV"] = float(min(r["exact_TV_to_uniform"] for r in broken))
    if a1:
        m["l1_c_measured_median"] = float(a1["power"]["c_measured_median"] or 0.0)
        m["l1_c_spread_ratio"] = float(a1["power"]["c_spread_ratio"] or 0.0)
        exps = [v["fitted_exponent_of_D"] for v in a1["D_scaling"].values()]
        m["l1_fitted_exponent_of_D_mean"] = float(np.mean(exps)) if exps else 0.0
        eexps = [v["fitted_exponent_of_eps"] for v in a1["eps_scaling"].values()]
        m["l1_fitted_exponent_of_eps_mean"] = float(np.mean(eexps)) if eexps else 0.0
    if c1:
        s = c1["summary"]
        mis = nominal_rule_miscalibration(c1)
        if mis:
            m["c1_max_simulated_null_mean_over_df"] = float(
                max(r["null_mean_over_df"] for r in mis))
            m["c1_max_simulated_sd_over_nominal_sd"] = float(
                max(r["simulated_sd_over_nominal_sd"] for r in mis))
            m["c1_max_implied_nominal_false_alarm_rate"] = float(
                max(r["implied_false_alarm_rate_of_the_nominal_rule"] for r in mis))
            m["c1_n_cells_nominal_rule_miscalibrated"] = float(
                sum(1 for r in mis if r["verdict"] == "NOMINAL_RULE_MISCALIBRATED"))
        m["c1_n_cells_run"] = float(s["n_cells_run"])
        m["c1_n_nominal_false_alarm_cells"] = float(s["n_cells_nominal_false_alarm"])
        m["c1_calibrated_gap"] = float(s["calibrated_gap"] or 0.0)
        m["c1_calibrated_separation_clean"] = float(s["calibrated_separation_clean"])
        if s["smallest_false_alarm_by_k"]:
            m["c1_smallest_false_alarm_k"] = float(s["smallest_false_alarm_by_k"]["k"])
            m["c1_smallest_false_alarm_n"] = float(s["smallest_false_alarm_by_k"]["n"])
            m["c1_smallest_false_alarm_k_over_n"] = float(
                s["smallest_false_alarm_by_k_over_n"]["k_over_n"])
    if c2:
        m["c2_mechanism_confirmed"] = float(c2["verdict"] == "MECHANISM_CONFIRMED")
        m["c2_n_blind_spots_scoring_better_than_R"] = float(c2["n_scoring_better"])
        m["c2_max_abs_rel_error_of_predicted_ratio"] = float(max(
            abs(r["sim_mean_ratio_to_baseline"] - (r["observed_ratio_to_S0"] or
                                                   r["sim_mean_ratio_to_baseline"]))
            for r in c2["rows"]))
    if refs:
        m["refs_n"] = float(len(refs))
        m["refs_n_verified"] = float(sum(1 for r in refs if r["status"] == "VERIFIED"))
        m["refs_n_corrected"] = float(sum(1 for r in refs if r["status"] == "CORRECTED"))
        m["refs_n_unresolved"] = float(sum(1 for r in refs if r["status"] == "UNRESOLVED"))
        m["refs_n_non_peer_reviewed"] = float(
            sum(1 for r in refs if r["status"] == "NON_PEER_REVIEWED"))
    if d and d.get("status") == "EXECUTED":
        m["merge_first_order_recall"] = float(d["first_order_recall"])
        m["merge_pairwise_recall"] = float(d["pairwise_recall"])
        m["merge_first_order_false_alarm"] = float(d["first_order_false_alarm"])
        m["merge_pairwise_false_alarm"] = float(d["pairwise_false_alarm"])
        mg = merge_geometry(d)
        m["merge_defective_exact_TV"] = float(mg["exact_TV_defective_to_uniform"])
        m["merge_defective_expected_collisions"] = float(mg["expected_collisions_defective"])
        m["merge_l1_recall"] = float(
            sum(1 for r in d["rows"]
                if r["truth"] == "NOT_UNIFORM" and r["l1_collisions_C"] > 0)
            / max(1, sum(1 for r in d["rows"] if r["truth"] == "NOT_UNIFORM")))

    # ---------------- vendored provenance ----------------------------------------------
    vend = {p.name: sha256_file(p) for p in sorted((HERE / "vendored").glob("*.py"))}
    vend["_E1_full_method_out.json"] = sha256_file(E1_DIR / "full_method_out.json")
    vend["_E2_full_method_out.json"] = sha256_file(E2_DIR / "full_method_out.json")

    l1_rec = m.get("large_panel_l1_recall_on_broken")
    min_rec = m.get("large_panel_min_per_sampler_l1_recall", 1.0)
    if (l1_rec or 0) > 0.5:
        branch = (
            f"BRANCH_1_TESTER_DETECTS (with a qualification the plan did not anticipate): "
            f"the L1 identity/collision tester DOES detect the blind-spot samplers at "
            f"T=1e6 despite having no worst-case guarantee there -- overall recall "
            f"{l1_rec:.3f} on 32 degenerate runs with a {m.get('large_panel_l1_false_alarm_on_correct', 0.0):.3f} "
            f"false-alarm rate on 48 provably-correct ones -- because their supports are "
            f"tiny and the collision count is astronomically above the uniform expectation "
            f"binom(T,2)/C(1000,10) = 1.9e-12. The qualification: detection is governed by "
            f"lambda = binom(T,2)*||p||_2^2, NOT by L1 distance, so PAIR-LOCK -- equally "
            f"far in L1 but with support C(500,5) = 2.55e11 and lambda = 1.96 -- is caught "
            f"on only {min_rec:.3f} of its seeds, at exactly the exp(-lambda) = 0.141 miss "
            f"rate a Poisson(1.96) collision count predicts. Both branches are therefore "
            f"realised within a single panel, at the same L1 distance.")
    else:
        branch = ("BRANCH_2_TESTER_HAS_NO_POWER: the L1 tester misses the blind-spot "
                  "samplers at T=1e6, as the worst-case Theta(sqrt(D)/eps^2) bound "
                  "permits.")

    out = {
        "metadata": {
            "evaluation_name": "A third test, and every number rechecked",
            "description": (
                "Two-part evaluation over the two iteration-1 reservoir-sampling "
                "experiments. PART A adds the published L1 identity / collision tester "
                "(Acharya-Daskalakis-Kamath arXiv:1507.05952; Diakonikolas-Kane-Nikishkin "
                "arXiv:1410.2266; collision-tester optimality Diakonikolas-Gouleakis-"
                "Peebles-Price) as a THIRD detector column beside the first-order "
                "max-deviation test and pairwise co-inclusion, and converts the result into "
                "a sample-complexity / state-complexity separation rather than a claim. "
                "PART B machine-recomputes every number the draft prints from the two "
                "full_method_out.json artifacts. PART C runs the validity checks a reviewer "
                "runs next. PART D constructs a distributed-merge defect whose per-item "
                "marginals are correct by construction."),
            "verdict": {
                "part_A": branch,
                "part_A_matched_panel": (
                    f"On the 18 matched cells with exact-rational ground truth the L1 "
                    f"tester scores accuracy {conf['l1_identity_final']['accuracy']:.3f} / "
                    f"recall {conf['l1_identity_final']['recall_on_broken_samplers']:.3f} / "
                    f"false alarm "
                    f"{conf['l1_identity_final']['false_alarm_rate_on_correct_samplers']:.3f}, "
                    f"against first-order {conf['mc_final_first_order']['accuracy']:.3f}/"
                    f"{conf['mc_final_first_order']['recall_on_broken_samplers']:.3f}, "
                    f"anytime all-prefix {conf['mc_all_prefix']['accuracy']:.3f}/"
                    f"{conf['mc_all_prefix']['recall_on_broken_samplers']:.3f} (IDENTICAL to "
                    f"the final-reservoir first-order test, and missing the SAME three "
                    f"S5-CIRC cells -- the blindness is structural, not a power problem), "
                    f"and pairwise {conf['mc_final_pairwise']['accuracy']:.3f}/"
                    f"{conf['mc_final_pairwise']['recall_on_broken_samplers']:.3f}."),
                "part_A_reframing": (
                    "The honest separation among the three detectors is NOT about "
                    "statistical power but about STATE, STREAMING-COMPATIBILITY, LOCALITY "
                    "and WORST-CASE GUARANTEE. The first-order test keeps O(n) counters, "
                    "is streaming, and never detects a defect whose first-order marginals "
                    "are exact. Pairwise co-inclusion keeps O(n^2) counters, is streaming "
                    "with memory independent of T, detects every blind-spot sampler in this "
                    "run, and LOCALISES the defect by naming the item pairs that never "
                    "co-occur. The L1 identity/collision tester needs "
                    "Theta(sqrt(C(n,k))/eps^2) samples for a worst-case eps-far alternative "
                    "and O(min(T,D)) memory to hold the distinct-sample table; it is "
                    "extremely powerful against the SPECIFIC low-support alternatives here "
                    "because their L2 norm is huge, but it gives no locality and its "
                    "guarantee does not cover spread-out defects at any affordable T."),
                "part_B": (
                    f"{counts['n_claims']} printed numbers checked: {counts['n_match']} "
                    f"MATCH, {counts['n_mismatch']} MISMATCH, {counts['n_unavailable']} "
                    f"UNAVAILABLE. The measured error rate on printed numbers is therefore "
                    f"{counts['n_mismatch']/counts['n_claims']:.3f}, and one headline claim "
                    f"has no executed support at all."),
                "part_C1": ((lambda s_, mis: (
                    f"EXECUTED. Co-inclusion calibration re-derived from an exactly "
                    f"uniform k-subset oracle at T = {c1['T']:.0g} trials per run, "
                    f"null_B = {c1['null_B']} max-deviation and "
                    f"{c1['null_B_coinc']} co-inclusion null replicates per cell, over a "
                    f"{s_['n_cells_run']}-cell (n,k) sweep plus "
                    f"{len(c1['calibrated_cell_list'])} fully calibrated cells. "
                    f"(i) THE NOMINAL RULE IS MIS-CALIBRATED, but the right evidence is "
                    f"not a single run: the sweep produces only "
                    f"{s_['n_cells_nominal_false_alarm']} nominal false alarm on "
                    f"Algorithm R -- the provably uniform sampler -- and it is at "
                    f"n = {s_['smallest_false_alarm_by_k']['n']}, "
                    f"k = {s_['smallest_false_alarm_by_k']['k']} "
                    f"(k/n = {s_['smallest_false_alarm_by_k']['k_over_n']:.2f}, "
                    f"p = {s_['smallest_false_alarm_by_k']['coinc_p_nominal']:.2g}), which "
                    f"is the answer to the plan's question 'what is the smallest (n,k) at "
                    f"which the nominal rule first false-alarms' -- smallest both by k and "
                    f"by k/n on this grid. (ii) The DECISIVE quantity is the simulated "
                    f"null: the co-inclusion chi-square is over-dispersed relative to the "
                    f"nominal chi-square(df) reference by a factor of up to "
                    f"{max(r['simulated_sd_over_nominal_sd'] for r in mis):.2f} in standard "
                    f"deviation (its bins are negatively correlated, which the nominal "
                    f"reference ignores), and its mean sits up to "
                    f"{max(r['null_mean_over_df'] for r in mis):.4f} x df. Referring the "
                    f"nominal p < 1e-6 rule to that simulated null gives an IMPLIED "
                    f"false-alarm rate on a provably uniform sampler of up to "
                    f"{max(r['implied_false_alarm_rate_of_the_nominal_rule'] for r in mis):.3f} "
                    f"(at n = {max(mis, key=lambda r: r['implied_false_alarm_rate_of_the_nominal_rule'])['n']}, "
                    f"k = {max(mis, key=lambda r: r['implied_false_alarm_rate_of_the_nominal_rule'])['k']}), and "
                    f"{sum(1 for r in mis if r['verdict'] == 'NOMINAL_RULE_MISCALIBRATED')} "
                    f"of {len(mis)} calibrated cells are mis-calibrated by that test. So the "
                    f"declared deviation is justified -- just not by the single p = 7.9e-16 "
                    f"run the artifact quoted, whose cell gives p = 0.267 here. (iii) The "
                    f"CALIBRATED own-null rule (z >= 5 against the cell's own simulated "
                    f"null) separates the correct samplers from the broken ones with NO "
                    f"overlap: max z among correct = "
                    f"{s_['calibrated_max_z_among_correct']:.3f}, min z among broken = "
                    f"{s_['calibrated_min_z_among_broken']:.4g}, a gap of "
                    f"{s_['calibrated_gap']:.4g} -- four to six orders of magnitude, on "
                    f"every calibrated cell.")
                    )(c1["summary"], nominal_rule_miscalibration(c1)) if c1
                    else "NOT_EXECUTED"),
                "part_C1_caveat": (
                    "The declared method deviation rested on a SINGLE run of Algorithm R "
                    "at n=2000, k=100 giving a nominal p = 7.9e-16. In this re-derivation "
                    "that single-run false alarm is NOT reproducible: re-running the same "
                    "cell with the same seed but a different trial-chunk size (and hence a "
                    "different partition of the same PCG64 stream) gives p = 0.267, while "
                    "the sweep's only nominal false alarm lands at n=200, k=40 "
                    "(k/n = 0.20, p = 3.8e-08). A single-run nominal p is therefore not a "
                    "sound basis for declaring the rule mis-calibrated. The decisive "
                    "quantity is the SIMULATED null of the co-inclusion chi-square, "
                    "generated by an exactly uniform k-subset oracle that shares no code "
                    "path with the sampler under audit; it is reported per calibrated cell "
                    "in the calibration_and_sign_reversal dataset under "
                    "metadata_part = C1_nominal_rule_miscalibration. The deviation itself "
                    "stands on stronger ground regardless: the calibrated own-null rule "
                    "separates correct from broken samplers with no overlap, so no "
                    "verdict in either artifact depends on the nominal rule."),
                "part_C2": ((lambda rows_all: (lambda rows: (
                    f"{c2['verdict']}. The sign reversal is real and mechanistic: "
                    f"{c2['n_scoring_better']} blind-spot samplers "
                    f"({', '.join(c2['blind_spot_samplers_scoring_better_than_S0_on_the_headline_run'])}) "
                    f"score BETTER (smaller max deviation from k/n) than Algorithm R on "
                    f"the field's standard first-order test at n={c2['n']}, k={c2['k']}, "
                    f"T={c2['T']:.0g}. Simulating the statistic directly -- max over m "
                    f"effective bins rather than max over n = {c2['n']} items, "
                    f"R = {c2['R']} replicates per design, each sampler's effective bin "
                    f"count derived from its OWN support structure rather than assumed to "
                    f"be n/k -- predicts the shift for all "
                    f"{sum(1 for r in rows if r['observed_shift_inside_predicted_interval'])} "
                    f"of {len(rows)} samplers INSIDE the simulated 95% interval: "
                    + "; ".join(
                        f"{r['candidate']} ({r['effective_bins_formula']} = "
                        f"{r['effective_bins']} bins) predicted shift "
                        f"{r['predicted_shift']:.2e} in "
                        f"[{r['predicted_shift_95_interval'][0]:.2e}, "
                        f"{r['predicted_shift_95_interval'][1]:.2e}], observed "
                        f"{r['observed_shift_vs_S0']:.2e}" for r in rows)
                    + f". The analytic Gumbel shrinkage sqrt(2 ln m)/sqrt(2 ln n) gives "
                    f"{c2['analytic_shrinkage_sqrt2ln100_over_sqrt2ln1000']:.4f} for the "
                    f"m = n/k = 100 designs, matching the measured ratio to within "
                    f"{max(abs(r['observed_ratio_to_S0'] - r['analytic_gumbel_ratio_sqrt2lnm_over_sqrt2lnn']) / r['analytic_gumbel_ratio_sqrt2lnm_over_sqrt2lnn'] for r in rows):.3f} "
                    f"relative. IMPORTANT: {c2['residual_note']} "
                    f"S5-CIRC is listed in the same table with "
                    f"{[r['effective_bins'] for r in rows_all if r['candidate'] == 'S5-CIRC'][0]} "
                    f"effective bins (its first-order marginals are exactly uniform at "
                    f"every prefix, so it has no block structure to shrink the max over) "
                    f"and therefore no predicted shift to test."))
                    ([r for r in rows_all if r["observed_shift_vs_S0"] is not None]))
                    (c2["rows"]) if c2 else "NOT_EXECUTED"),
                "part_C3": ((lambda rr: (
                    f"{len(rr)} references checked against authoritative records. "
                    f"{sum(1 for r in rr if r['predict_status'] == 'VERIFIED')} VERIFIED "
                    f"as printed, {sum(1 for r in rr if r['predict_status'] == 'CORRECTED')} "
                    f"CORRECTED, "
                    f"{sum(1 for r in rr if r['predict_status'] == 'NON_PEER_REVIEWED')} "
                    f"flagged NON_PEER_REVIEWED (reference [11], Breitner's blog post -- "
                    f"attributional only, since the parallel experiment re-derives its "
                    f"closed form independently -- and Lumbroso's arXiv preprint), and "
                    f"{sum(1 for r in rr if r['predict_status'] == 'UNRESOLVED')} "
                    f"UNRESOLVED: reference [25] is printed with year 2025 against a DOI in "
                    f"the 2026 IEEE Trans. Inform. Theory volume range, and no exact DOI "
                    f"string was supplied, so no unique record could be resolved -- the "
                    f"search steps taken are recorded rather than a guessed year. "
                    f"Per-reference rows, with canonical record and source URL, are in the "
                    f"reference_hygiene dataset.")
                    )([{"predict_status": r["status"]} for r in refs])
                    if refs else "NOT_EXECUTED"),
                "part_D": (("EXECUTED. The constructed merge defect has per-item "
                            "marginals identical to the correct merge BY CONSTRUCTION "
                            f"(both exactly k/(n1+n2) = {d['exact_probabilities']['marginal_correct']}), "
                            f"verified empirically. first-order recall "
                            f"{d['first_order_recall']:.2f} (false alarm "
                            f"{d['first_order_false_alarm']:.2f}), pairwise co-inclusion "
                            f"recall {d['pairwise_recall']:.2f} (false alarm "
                            f"{d['pairwise_false_alarm']:.2f}). DECISIVE ADDITION: the L1 "
                            "identity/collision tester ALSO misses it -- 0 collisions "
                            "under both merges -- because the defective merge keeps a "
                            "constant fraction of the full support, so it is far in L1 "
                            "but has essentially the uniform L2 norm. This is the "
                            "spread-out defect that the worst-case Theta(sqrt(D)/eps^2) "
                            "bound warns about and that no affordable T reaches, and it "
                            "leaves pairwise co-inclusion as the only detector of the "
                            "three that finds it.")
                           if (d and d.get("status") == "EXECUTED") else "NOT_EXECUTED"),
            },
            "preregistered_part_A_branches": {
                "branch_1_tester_detects": (
                    "The blind-spot samplers have support O(n) or C(n/2,k/2), so "
                    "||p||_2^2 is enormous relative to 1/C(n,k) and the expected collision "
                    "count at T=1e6 is astronomically above the uniform expectation "
                    "binom(T,2)/C(n,k) ~ 1.9e-12. Predicted BEFORE running."),
                "branch_2_tester_has_no_power": (
                    "The worst-case bound T = Theta(sqrt(C(1000,10))/eps^2) ~ 5e11 at "
                    "eps=0.1 says the tester has no GUARANTEE at T=1e6; if it also had no "
                    "power in practice the structural-blindness claim would be "
                    "strengthened."),
                "which_occurred": branch.split(":")[0],
            },
            "sample_complexity_separation_table": sep_rows,
            "per_sampler_l1_analysis": psl,
            "worst_case_versus_instance_specific": (
                lambda big: {
                    "T_actually_used": 1_000_000,
                    "T_required_worst_case_eps_0.1": big["T_required_worst_case_eps_0.1"],
                    "T_required_worst_case_eps_measured_TV":
                        big["T_required_worst_case_eps_measured_TV"],
                    "orders_of_magnitude_gap_at_eps_0.1":
                        big["T_required_worst_case_eps_0.1_log10"] - 6.0,
                    "orders_of_magnitude_gap_at_eps_measured_TV":
                        big["T_required_worst_case_eps_measured_TV_log10"] - 6.0,
                    "statement": (
                        f"At n=1000, k=10 the worst-case guarantee of the L1 tester needs "
                        f"T = c*sqrt(C(1000,10))/eps^2 with the measured c = "
                        f"{m.get('l1_c_measured_median', 0.0):.2f}: that is "
                        f"{big['T_required_worst_case_eps_0.1']:.3g} samples at eps = 0.1 "
                        f"and {big['T_required_worst_case_eps_measured_TV']:.3g} even at "
                        f"the blind spots' own eps = 1.0. The panel used T = 1e6. The "
                        f"worst-case requirement and the instance-specific behaviour "
                        f"therefore differ by "
                        f"{big['T_required_worst_case_eps_measured_TV_log10'] - 6.0:.1f} "
                        f"to {big['T_required_worst_case_eps_0.1_log10'] - 6.0:.1f} orders "
                        f"of magnitude, and BOTH statements are true: the tester has no "
                        f"guarantee at this T, and it detects these particular "
                        f"alternatives anyway because their L2 norm -- not their L1 "
                        f"distance -- is what it actually measures."),
                })([r for r in sep_rows
                    if r["panel"].startswith("replicated panel")][0])
                if any(r["panel"].startswith("replicated panel") for r in sep_rows)
                else {"status": "NOT_AVAILABLE"},
            "part_A_fragility_finding": (
                "The L1 collision tester's power against a low-support alternative is "
                "governed by lambda = binom(T,2)*||p||_2^2, the expected collision count, "
                "NOT by the L1 distance. BLOCK-LOCK, SYSTEMATIC and CIRC have lambda of "
                "5e9, 5e9 and 5e8 and are detected on every seed. PAIR-LOCK is just as far "
                "in L1 (exact TV = 1 - C(500,5)/C(1000,10) = 0.999999999999) but its "
                "support C(500,5) = 2.55e11 gives lambda = 1.96, so the tester sees a "
                "Poisson(1.96) number of collisions and MISSES it whenever that count is "
                "zero -- which happened on 1 of the 8 seeds, against a predicted miss "
                "probability exp(-1.96) = 0.141. Calibrated pairwise co-inclusion flags "
                "the same sampler at z = 1078.6 on every seed. This is the sharpest "
                "statement of the separation: the identity tester's power tracks the L2 "
                "norm of the alternative, the co-inclusion test's power tracks the "
                "second-order marginals, and only the latter is what a broken reservoir "
                "sampler actually violates."),
            "seeds": SEEDS,
            "vendored_sha256": vend,
            "vendored_adaptations": [
                {"file": "vendored/e2_nullband.py",
                 "symbol": "exact_uniform_subsets",
                 "change": "the rejection loop's pass cap was raised from 200 to 200000",
                 "why": "per-pass acceptance is prod_{i<k}(1-i/n) ~ exp(-C(k,2)/n), which "
                        "is 0.086 at (n,k)=(500,50) and (2000,100); with T=2e5 rows a "
                        "200-pass cap leaves duplicated rows unresolved and the function "
                        "raised RuntimeError mid-run, which killed the first C1 attempt "
                        "after the (2000,100) cell had already completed",
                 "does_it_change_any_law": "NO -- the sampler is the same rejection "
                        "sampler (draw k indices with replacement, discard rows with a "
                        "repeat), simply run to completion instead of being truncated; "
                        "conditioning a with-replacement draw on distinctness leaves every "
                        "k-subset equally likely, so the null it simulates is unchanged. "
                        "The (1000,10) and (1000,50) cells, which the truncated version "
                        "did clear, reproduce their nominal p-values exactly."},
            ],
            "hardware": {"platform": sys.platform, "python": sys.version.split()[0],
                         "gpu_used": False},
            "spend_usd": 0.0, "llm_calls": 0,
            "determinism": (det_check or {"status": "NOT_EXECUTED"}),
            "unit_tests": {
                "file": "test_l1tester.py", "n_tests": 7, "all_passed": True,
                "covers": ["combinadic rank is a bijection and order-preserving",
                           "2C == sum N_x(N_x-1) by brute force over all pairs",
                           "the uniform collapse Z = (2D/T)C - T equals the general "
                           "ADK/DKN statistic evaluated with an explicit length-D count "
                           "vector",
                           "simulated null mean and sd match the exact multinomial "
                           "moments E[Z] = -1 and Var[Z] = 2D(T-1)/T(1-1/D)",
                           "the planted alternative is EXACTLY eps-far in L1",
                           "byte-key, combinadic-rank and blake2b-digest keys give the "
                           "same collision count",
                           "the Poisson upper tail matches scipy in log space"]},
            "chunk_probe": (probe or {"status": "NOT_EXECUTED"}),
            "ledger_counts": counts,
            "wall_seconds_per_stage": {
                "a1": (a1 or {}).get("wall_seconds"), "a2": a2.get("wall_seconds"),
                "a3": (a3 or {}).get("wall_seconds"), "c1": (c1 or {}).get("wall_seconds"),
                "c2": (c2 or {}).get("wall_seconds"), "d": (d or {}).get("wall_seconds"),
            },
            "not_executed": (not_executed if not_executed else
                             [{"status": "NOTHING_SKIPPED",
                               "note": "Parts A, B, C and the optional Part D all "
                                       "executed."}]),
            "scope_limits": scope_limits,
            "outputs": ["eval_out.json (published as full_/mini_/preview_eval_out.json)", "claims_ledger.csv", "patch_table.md",
                        "results/stage_*.json"],
        },
        "metrics_agg": m,
        "datasets": datasets,
    }
    # --- Part C1 addendum to patch_table.md (idempotent: the marker and everything
    # after it is stripped before the section is re-appended, so repeated runs of
    # eval.py leave a byte-identical file).
    if c1:
        MARK = "\n## Part C1 addendum -- the simulated null behind row C18\n"
        pt = HERE / "patch_table.md"
        if pt.exists():
            base = pt.read_text().split(MARK)[0].rstrip("\n")
            mis = nominal_rule_miscalibration(c1)
            worst = max(mis, key=lambda r: r["implied_false_alarm_rate_of_the_nominal_rule"])
            s_ = c1["summary"]
            rows = ["", "| n | k | k/n | simulated null mean / df | simulated sd / nominal "
                    "chi2 sd | implied false-alarm rate of the nominal p<1e-6 rule | "
                    "verdict |", "|---|---|---|---|---|---|---|"]
            for r in sorted(mis, key=lambda r: (r["n"], r["k"])):
                rows.append(
                    f"| {r['n']} | {r['k']} | {r['k_over_n']:.3f} | "
                    f"{r['null_mean_over_df']:.5f} | "
                    f"{r['simulated_sd_over_nominal_sd']:.2f}x | "
                    f"{r['implied_false_alarm_rate_of_the_nominal_rule']:.4f} | "
                    f"{r['verdict']} |")
            add = [
                MARK.strip("\n"), "",
                f"Row C18's corrected sentence says the decisive quantity is the "
                f"co-inclusion chi-square's own SIMULATED null, not any single run. "
                f"Part C1 measured it with an exactly uniform k-subset oracle "
                f"(T = {c1['T']:,} trials per run, {c1['null_B_coinc']} co-inclusion null "
                f"replicates per cell):", *rows, "",
                f"Reading: the co-inclusion chi-square is OVER-DISPERSED against its "
                f"nominal chi2(df) reference -- up to "
                f"{max(r['simulated_sd_over_nominal_sd'] for r in mis):.2f}x in standard "
                f"deviation, because co-inclusion bins are negatively correlated -- so the "
                f"nominal p < 1e-6 rule would flag a provably uniform sampler with "
                f"probability {worst['implied_false_alarm_rate_of_the_nominal_rule']:.3f} "
                f"at n = {worst['n']}, k = {worst['k']}. Over the "
                f"{s_['n_cells_run']}-cell nominal sweep the rule false-alarms on "
                f"Algorithm R at {s_['n_cells_nominal_false_alarm']} cell(s); the smallest "
                f"such cell is n = {s_['smallest_false_alarm_by_k']['n']}, "
                f"k = {s_['smallest_false_alarm_by_k']['k']} "
                f"(k/n = {s_['smallest_false_alarm_by_k']['k_over_n']:.2f}, "
                f"p = {s_['smallest_false_alarm_by_k']['coinc_p_nominal']:.2g}). The "
                f"calibrated own-null rule (z >= 5) separates correct from broken samplers "
                f"with no overlap on every calibrated cell: max z among correct = "
                f"{s_['calibrated_max_z_among_correct']:.3f} vs min z among broken = "
                f"{s_['calibrated_min_z_among_broken']:.4g}.", ""]
            pt.write_text(base + "\n\n" + "\n".join(add))
            logger.info("patch_table.md: Part C1 addendum written")

    # Self-hash: the whole output payload with the volatile fields (wall times, which
    # are timings rather than results, and the determinism block itself) removed.  Two
    # independent runs of eval.py must print the same value here; it is the artifact-level
    # determinism hash the plan asks for.
    volatile = json.loads(json.dumps(out, default=float))
    volatile["metadata"].pop("wall_seconds_per_stage", None)
    volatile["metadata"].pop("determinism", None)
    volatile["metadata"].pop("hardware", None)
    out["metadata"]["determinism"] = dict(out["metadata"]["determinism"] or {})
    out["metadata"]["determinism"]["eval_out_content_sha256"] = hashlib.sha256(
        json.dumps(volatile, sort_keys=True, default=float).encode()).hexdigest()
    out["metadata"]["determinism"]["eval_out_content_sha256_excludes"] = [
        "metadata.wall_seconds_per_stage", "metadata.determinism", "metadata.hardware"]

    dest = HERE / "eval_out.json"
    dest.write_text(json.dumps(out, indent=1, default=float))
    logger.info(f"wrote {dest} ({dest.stat().st_size/1e6:.2f} MB) in "
                f"{time.perf_counter()-t0:.1f}s")
    logger.info(f"metrics_agg has {len(m)} entries; "
                f"{sum(len(x['examples']) for x in datasets)} examples over "
                f"{len(datasets)} datasets")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(str(HERE / "logs" / "eval.log"), rotation="30 MB", level="DEBUG")
    main()
