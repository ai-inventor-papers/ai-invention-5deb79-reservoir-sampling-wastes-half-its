#!/usr/bin/env python3
"""PART D (optional) -- a distributed-MERGE defect whose per-item marginals are correct.

Shape of the defect (as in Apache DataSketches issue #647, an open maintainer-authored
reservoir-MERGE defect): two reservoirs of size k, built over streams of sizes n1 and n2,
are merged by a rule that PRESERVES every item's marginal inclusion probability
k/(n1+n2) but induces a dependency between which items survive from the same source
reservoir.

  CORRECT merge   : j ~ Hypergeometric(n1+n2, n1, k) survivors are taken from reservoir 1
                    and k-j from reservoir 2, each uniformly at random.  Because R1 is a
                    uniform k-subset of [n1], a uniform j-subset of R1 is a uniform
                    j-subset of [n1], so the merge is exactly a uniform k-subset of the
                    union.
  DEFECTIVE merge : a FIXED j = round(k*n1/(n1+n2)) survivors are taken from reservoir 1
                    and k-j from reservoir 2.  Marginals are still exactly right --
                    P(x in merged) = (k/n1)*(j/k) = j/n1 = k/(n1+n2) when n1=n2 -- but the
                    merged set always contains exactly j items from stream 1, whereas the
                    correct merge spreads that count hypergeometrically.

Exact second-order consequence at n1 = n2 = n, j = k/2:
  correct  : P(x,y both in) = k(k-1) / ((2n)(2n-1))            for EVERY pair
  defective: P(x,y both in) = j(j-1) / (n(n-1))                for a SAME-stream pair
             P(x,y both in) = (j/n)^2                          for a CROSS-stream pair
That difference is what the pairwise co-inclusion test sees and the first-order test
cannot.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from math import comb, sqrt
from pathlib import Path

import numpy as np
from loguru import logger

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendored"))
RES = HERE / "results"
RES.mkdir(exist_ok=True)


def _uniform_ksubsets(n: int, k: int, T: int, rng: np.random.Generator,
                      offset: int = 0) -> np.ndarray:
    """T independent uniform k-subsets of {offset+1, ..., offset+n}, rows sorted."""
    out = np.empty((T, k), dtype=np.int64)
    done = 0
    while done < T:
        m = min(200_000, T - done)
        cand = rng.integers(1, n + 1, size=(m, k))
        cand.sort(axis=1)
        ok = np.all(np.diff(cand, axis=1) > 0, axis=1)
        good = cand[ok]
        take = min(len(good), T - done)
        out[done:done + take] = good[:take] + offset
        done += take
    return out


def _take_j_of_k(res: np.ndarray, j_arr: np.ndarray, k: int,
                 rng: np.random.Generator) -> list[np.ndarray]:
    """For each row, pick j_arr[row] of the k held items uniformly without replacement."""
    T = res.shape[0]
    keys = rng.random((T, k))
    order = np.argsort(keys, axis=1)
    ranks = np.empty_like(order)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(k), (T, k)).copy(), axis=1)
    mask = ranks < j_arr[:, None]
    return [res[mask], mask]


def merge_panel(*, n1: int = 500, n2: int = 500, k: int = 10, T: int = 1_000_000,
                seed: int = 777, null_R: int = 200) -> dict:
    import e1_montecarlo as mc
    import l1tester as L

    n = n1 + n2
    j_fixed = int(round(k * n1 / n))
    rng = np.random.Generator(np.random.PCG64(seed))

    def build(defective: bool) -> np.ndarray:
        out = np.empty((T, k), dtype=np.int64)
        done = 0
        while done < T:
            m = min(250_000, T - done)
            r1 = _uniform_ksubsets(n1, k, m, rng, offset=0)
            r2 = _uniform_ksubsets(n2, k, m, rng, offset=n1)
            if defective:
                j = np.full(m, j_fixed, dtype=np.int64)
            else:
                j = rng.hypergeometric(n1, n2, k, size=m)
            sel1, _ = _take_j_of_k(r1, j, k, rng)
            sel2, _ = _take_j_of_k(r2, k - j, k, rng)
            block = np.empty((m, k), dtype=np.int64)
            c1 = np.zeros(m + 1, dtype=np.int64)
            np.cumsum(j, out=c1[1:])
            c2 = np.zeros(m + 1, dtype=np.int64)
            np.cumsum(k - j, out=c2[1:])
            for row in range(m):
                block[row, :j[row]] = sel1[c1[row]:c1[row + 1]]
                block[row, j[row]:] = sel2[c2[row]:c2[row + 1]]
            block.sort(axis=1)
            out[done:done + m] = block
            done += m
            del r1, r2, sel1, sel2, block
            gc.collect()
        return out

    # simulated null band, from an exactly uniform k-subset sampler over the union
    rng_n = np.random.Generator(np.random.PCG64(seed + 10_001))
    band = mc.null_band(n, k, T, null_R, rng_n)

    rows = []
    marg = {}
    for defective in (False, True):
        t0 = time.perf_counter()
        res = build(defective)
        fo = mc.first_order_stats(res, n, k)
        pr = mc.pair_stats(res, n, k)
        keys = L.subset_bytes_key(res)
        l1 = L.run_tester(keys, T, comb(n, k))
        counts = np.bincount(res.ravel(), minlength=n + 1)[1:n + 1]
        marg["defective" if defective else "correct"] = {
            "mean_freq": float(counts.mean() / T),
            "max_freq": float(counts.max() / T),
            "min_freq": float(counts.min() / T),
            "expected_freq_k_over_n": k / n,
            "analytic_marginal": j_fixed / n1 if defective else k / n,
        }
        del res, keys
        gc.collect()
        p_fo = (1 + sum(1 for v in band["first_order"]["values"]
                        if v >= fo["max_dev"])) / (null_R + 1)
        p_pr = (1 + sum(1 for v in band["pair_z"]["values"]
                        if v >= pr["max_abs_z"])) / (null_R + 1)
        rows.append({
            "merge": "defective_fixed_split" if defective else "correct_hypergeometric",
            "truth": "NOT_UNIFORM" if defective else "UNIFORM",
            "n1": n1, "n2": n2, "n": n, "k": k, "T": T, "j_fixed": j_fixed,
            "first_order_max_dev": fo["max_dev"],
            "first_order_mc_p": p_fo,
            "first_order_verdict": "NOT_UNIFORM" if p_fo <= 0.01 else "UNIFORM",
            "pairwise_max_abs_z": pr["max_abs_z"],
            "pairwise_mc_p": p_pr,
            "pairwise_verdict": "NOT_UNIFORM" if p_pr <= 0.01 else "UNIFORM",
            "pairwise_frac_zero_pairs": pr["frac_zero_pairs"],
            "l1_collisions_C": l1["collisions_C"],
            "l1_n_distinct": l1["n_distinct"],
            "l1_Z": l1["Z"],
            "seconds": time.perf_counter() - t0,
        })
        logger.info(f"D {rows[-1]['merge']}: max_dev={fo['max_dev']:.4g} p={p_fo:.3g} "
                    f"-> {rows[-1]['first_order_verdict']}; pair z={pr['max_abs_z']:.2f} "
                    f"p={p_pr:.3g} -> {rows[-1]['pairwise_verdict']}; "
                    f"L1 C={l1['collisions_C']} [{rows[-1]['seconds']:.1f}s]")

    exact = {
        "correct_pair_prob_any": k * (k - 1) / (n * (n - 1)),
        "defective_pair_prob_same_stream": j_fixed * (j_fixed - 1) / (n1 * (n1 - 1)),
        "defective_pair_prob_cross_stream": (j_fixed / n1) * ((k - j_fixed) / n2),
        "marginal_correct": k / n,
        "marginal_defective": j_fixed / n1,
        "marginals_identical_by_construction": abs(k / n - j_fixed / n1) < 1e-15,
    }
    broken = [r for r in rows if r["truth"] == "NOT_UNIFORM"]
    good = [r for r in rows if r["truth"] == "UNIFORM"]
    return {
        "rows": rows, "exact_probabilities": exact, "marginals_measured": marg,
        "null_band_R": null_R,
        "first_order_recall": sum(1 for r in broken
                                  if r["first_order_verdict"] == "NOT_UNIFORM") / len(broken),
        "pairwise_recall": sum(1 for r in broken
                               if r["pairwise_verdict"] == "NOT_UNIFORM") / len(broken),
        "first_order_false_alarm": sum(1 for r in good
                                       if r["first_order_verdict"] == "NOT_UNIFORM") / len(good),
        "pairwise_false_alarm": sum(1 for r in good
                                    if r["pairwise_verdict"] == "NOT_UNIFORM") / len(good),
        "status": "EXECUTED",
    }


SPEC_IF_NOT_EXECUTED = {
    "status": "NOT_EXECUTED",
    "defect_shape": (
        "Apache DataSketches issue #647 shape: merge two size-k reservoirs built over "
        "streams of sizes n1 and n2 by taking a FIXED j = round(k*n1/(n1+n2)) survivors "
        "from reservoir 1 instead of j ~ Hypergeometric(n1+n2, n1, k). Per-item marginals "
        "are exactly k/(n1+n2) under both rules; only the joint law differs."),
    "expected_outcome": (
        "first-order max-deviation recall 0 on the defective merge (its marginals are "
        "exact by construction), pairwise co-inclusion recall 1, with the measured "
        "per-item marginals verified identical between the two merges."),
    "exact_second_order_prediction": (
        "correct: P(x,y) = k(k-1)/((n1+n2)(n1+n2-1)) for every pair; defective at "
        "n1=n2=n, j=k/2: P = j(j-1)/(n(n-1)) for a same-stream pair and (j/n)^2 for a "
        "cross-stream pair."),
    "config": {"n1": 500, "n2": 500, "k": 10, "T": 1_000_000},
}

if __name__ == "__main__":
    import argparse

    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
    logger.add(str(HERE / "logs" / "partD.log"), rotation="30 MB", level="DEBUG")
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1_000_000)
    ap.add_argument("--nullR", type=int, default=200)
    args = ap.parse_args()
    t0 = time.perf_counter()
    out = merge_panel(T=args.T, null_R=args.nullR)
    out["wall_seconds"] = time.perf_counter() - t0
    (RES / "stage_d.json").write_text(json.dumps(out, indent=1, default=float))
    logger.info(f"stage d written in {out['wall_seconds']:.1f}s")
