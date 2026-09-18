#!/usr/bin/env python3
"""The eviction step as a transportation problem on the Johnson graph.

At the step where item ``i+1`` arrives, a sampler that is uniform on k-subsets of [i]
and stays uniform on k-subsets of [i+1] must

  * accept with probability exactly k/(i+1)   (e1_forced -- see e1_forced.py), and
  * move the accepted mass from each k-subset S of [i] onto one of its k children
    T = S \\ {x}, so that every (k-1)-subset T of [i] ends up with total load
    (i-k+1)/k when each source's supply is normalised to 1.

That is a transportation problem on the biregular bipartite graph J(i,k) -> J(i,k-1)
(out-degree k, in-degree i-k+1).  This module computes, EXACTLY:

  * whether the demands are integral (``k | (i+1)``), in which case a vertex of the
    polytope sends every source's whole mass to ONE child and the minimum eviction
    entropy is exactly 0;
  * an integral certificate of that (unit-capacity max-flow) plus an independent checker;
  * how balanced the deterministic sum-mod-k rule already is;
  * the minimum number of sources that must be reassigned to repair it (min-cost flow);
  * the exact minimum eviction entropy (MILP on small instances) and a pre-registered
    analytic bracket everywhere else.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from itertools import combinations
from math import comb, log2
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp
from loguru import logger
from scipy.optimize import LinearConstraint, linprog, milp
from scipy.sparse.csgraph import maximum_flow

CERT_DIR = Path(__file__).resolve().parent / "certificates"

# Pre-registered caps so nothing explodes silently.
MILP_MAX_VARS = 10_000
MILP_TIME_LIMIT_S = 150.0
LP_MAX_VARS = 200_000


# --------------------------------------------------------------------------------------
# Structure
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Fiber:
    """The transportation instance for the step (prefix i, arriving item i+1)."""

    i: int
    k: int
    sources: tuple[tuple[int, ...], ...]  # k-subsets of [i], sorted
    targets: tuple[tuple[int, ...], ...]  # (k-1)-subsets of [i], sorted
    src_index: dict[tuple[int, ...], int]
    tgt_index: dict[tuple[int, ...], int]
    edge_src: np.ndarray  # (E,) int32 -- edge e leaves sources[edge_src[e]]
    edge_tgt: np.ndarray  # (E,) int32
    edge_removed: np.ndarray  # (E,) int32 -- which element was dropped
    divisible: bool
    demand_num: int  # (i-k+1)  -- demand numerator in the k-scaled problem
    demand_frac: Fraction  # (i-k+1)/k in the unit-supply problem

    @property
    def n_sources(self) -> int:
        return len(self.sources)

    @property
    def n_targets(self) -> int:
        return len(self.targets)

    @property
    def n_edges(self) -> int:
        return int(self.edge_src.shape[0])


@lru_cache(maxsize=64)
def build_fiber(i: int, k: int) -> Fiber:
    """Build the biregular Johnson transportation instance, asserting its arithmetic."""
    if not (1 <= k <= i):
        raise ValueError(f"need 1 <= k <= i, got i={i}, k={k}")
    sources = tuple(combinations(range(1, i + 1), k))
    targets = tuple(combinations(range(1, i + 1), k - 1))
    src_index = {s: a for a, s in enumerate(sources)}
    tgt_index = {t: b for b, t in enumerate(targets)}

    es: list[int] = []
    et: list[int] = []
    er: list[int] = []
    for a, S in enumerate(sources):
        for x in S:
            T = tuple(y for y in S if y != x)
            es.append(a)
            et.append(tgt_index[T])
            er.append(x)

    A, B = len(sources), len(targets)
    assert A == comb(i, k) and B == comb(i, k - 1)
    # Biregularity / flow conservation: A * k == B * (i-k+1)
    assert A * k == B * (i - k + 1), f"transportation arithmetic broken at i={i},k={k}"
    assert len(es) == A * k

    div_a = (i - k + 1) % k == 0
    div_b = (i + 1) % k == 0
    assert div_a == div_b, (
        f"the two divisibility forms disagree at i={i},k={k}: "
        f"(i-k+1)%k={(i - k + 1) % k}, (i+1)%k={(i + 1) % k}"
    )

    return Fiber(
        i=i,
        k=k,
        sources=sources,
        targets=targets,
        src_index=src_index,
        tgt_index=tgt_index,
        edge_src=np.asarray(es, dtype=np.int32),
        edge_tgt=np.asarray(et, dtype=np.int32),
        edge_removed=np.asarray(er, dtype=np.int32),
        divisible=div_a,
        demand_num=i - k + 1,
        demand_frac=Fraction(i - k + 1, k),
    )


# --------------------------------------------------------------------------------------
# The deterministic sum-mod-k rule
# --------------------------------------------------------------------------------------


def summodk_position(S: tuple[int, ...], k: int) -> int:
    """0-based position of the evicted element: r = ((sum(S)-1) mod k) + 1, r-th SMALLEST."""
    return (sum(S) - 1) % k


def summodk_map(fib: Fiber) -> np.ndarray:
    """(A,) array: target index each source maps to under the deterministic rule."""
    out = np.empty(fib.n_sources, dtype=np.int64)
    k = fib.k
    for a, S in enumerate(fib.sources):
        pos = summodk_position(S, k)
        T = S[:pos] + S[pos + 1 :]
        out[a] = fib.tgt_index[T]
    return out


def summodk_balance(fib: Fiber) -> dict[str, Any]:
    """How balanced is sum-mod-k already?  Reported honestly whatever it is."""
    f = summodk_map(fib)
    loads = np.bincount(f, minlength=fib.n_targets).astype(np.int64)
    required_frac = fib.demand_frac  # in unit-supply terms; deterministic loads are integers
    req_is_int = fib.divisible
    required_int = fib.demand_num // fib.k if req_is_int else None
    if req_is_int:
        dev = np.abs(loads - required_int)
        balanced = int((loads == required_int).sum())
        maxdev = int(dev.max())
        l1 = int(dev.sum())
    else:
        # loads must be integers but the requirement is not -- every target is off.
        lo = fib.demand_num / fib.k
        dev_f = np.abs(loads - lo)
        balanced = 0
        maxdev = float(dev_f.max())
        l1 = float(dev_f.sum())
    hist: dict[str, int] = {}
    for v, c in zip(*np.unique(loads, return_counts=True)):
        hist[str(int(v))] = int(c)
    return {
        "required_load": f"{required_frac.numerator}/{required_frac.denominator}",
        "required_load_float": float(required_frac),
        "required_is_integer": bool(req_is_int),
        "load_histogram": hist,
        "max_abs_load_deviation": maxdev,
        "l1_load_deviation": l1,
        "balanced_fraction": balanced / fib.n_targets,
        "n_targets": fib.n_targets,
        "n_sources": fib.n_sources,
    }


# --------------------------------------------------------------------------------------
# Integral feasibility (divisible steps): unit-capacity max-flow
# --------------------------------------------------------------------------------------


def integral_certificate(fib: Fiber) -> dict[str, Any]:
    """Unit-capacity max-flow proving a perfectly balanced DETERMINISTIC fiber map exists.

    Only meaningful when ``fib.divisible``; for non-divisible steps the demands are
    fractional and no deterministic map can exist (reported as such, with the exact
    reason, not as a failure).
    """
    if not fib.divisible:
        return {
            "divisible": False,
            "feasible": False,
            "reason": (
                f"demand (i-k+1)/k = {fib.demand_num}/{fib.k} is not an integer, so no "
                "deterministic (single-child) eviction rule can be exactly uniform"
            ),
        }
    d = fib.demand_num // fib.k
    A, B, E = fib.n_sources, fib.n_targets, fib.n_edges
    # nodes: 0 = src, 1..A = sources, A+1..A+B = targets, A+B+1 = snk
    N = A + B + 2
    snk = A + B + 1
    rows = np.concatenate(
        [
            np.zeros(A, np.int32),
            fib.edge_src + 1,
            np.arange(A + 1, A + B + 1, dtype=np.int32),
        ]
    )
    cols = np.concatenate(
        [
            np.arange(1, A + 1, dtype=np.int32),
            fib.edge_tgt + 1 + A,
            np.full(B, snk, np.int32),
        ]
    )
    data = np.concatenate([np.ones(A, np.int32), np.ones(E, np.int32), np.full(B, d, np.int32)])
    graph = sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.int32)
    res = maximum_flow(graph, 0, snk)
    value = int(res.flow_value)
    feasible = value == A
    cert: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    if feasible:
        flow = sp.csr_matrix(res.flow)
        indptr, indices, fdata = flow.indptr, flow.indices, flow.data
        for a in range(A):
            chosen = None
            for ptr in range(indptr[a + 1], indptr[a + 2]):
                col, val = int(indices[ptr]), int(fdata[ptr])
                if val > 0 and A + 1 <= col <= A + B:
                    chosen = col - A - 1
                    break
            if chosen is None:
                raise RuntimeError(f"max-flow saturated but source {a} carries no outgoing unit")
            cert.append((fib.sources[a], fib.targets[chosen]))
    return {
        "divisible": True,
        "feasible": feasible,
        "flow_value": value,
        "required_flow": A,
        "per_target_load": d,
        "certificate": cert,
    }


def check_certificate(
    cert: list[tuple[tuple[int, ...], tuple[int, ...]]], i: int, k: int
) -> dict[str, Any]:
    """Independent re-verification of a deterministic fiber map.  Shares no solver state."""
    errors: list[str] = []
    sources = set(combinations(range(1, i + 1), k))
    targets = list(combinations(range(1, i + 1), k - 1))
    d_num, d_den = i - k + 1, k
    if d_num % d_den != 0:
        errors.append("demand is not integral -- a deterministic certificate cannot be valid")
    d = d_num // d_den if d_den and d_num % d_den == 0 else None

    seen: set[tuple[int, ...]] = set()
    load: dict[tuple[int, ...], int] = {t: 0 for t in targets}
    for S, T in cert:
        S, T = tuple(S), tuple(T)
        if S not in sources:
            errors.append(f"{S} is not a k-subset of [{i}]")
            continue
        if S in seen:
            errors.append(f"source {S} appears more than once")
        seen.add(S)
        if not set(T).issubset(set(S)) or len(S) - len(T) != 1:
            errors.append(f"{T} is not a child of {S}")
            continue
        if T not in load:
            errors.append(f"{T} is not a (k-1)-subset of [{i}]")
            continue
        load[T] += 1
    missing = sources - seen
    if missing:
        errors.append(f"{len(missing)} sources are unassigned (e.g. {sorted(missing)[:3]})")
    if d is not None:
        bad = [t for t, v in load.items() if v != d]
        if bad:
            errors.append(f"{len(bad)} targets have load != {d} (e.g. {bad[:3]})")
    return {"valid": not errors, "errors": errors[:10], "n_errors": len(errors)}


def certificate_hash(cert: list[tuple[tuple[int, ...], tuple[int, ...]]]) -> str:
    canon = json.dumps(
        sorted([[list(map(int, s)), list(map(int, t))] for s, t in cert]),
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canon.encode()).hexdigest()


# --------------------------------------------------------------------------------------
# Min-cost transportation: minimal repair of sum-mod-k, and the flow tables
# --------------------------------------------------------------------------------------


def _transport_constraints(fib: Fiber) -> tuple[sp.csr_matrix, np.ndarray]:
    """Equality constraints of the k-SCALED problem: supply k per source, demand i-k+1."""
    A, B, E = fib.n_sources, fib.n_targets, fib.n_edges
    e = np.arange(E, dtype=np.int32)
    rows = np.concatenate([fib.edge_src, fib.edge_tgt + A])
    cols = np.concatenate([e, e])
    data = np.ones(2 * E, dtype=np.float64)
    M = sp.csr_matrix((data, (rows, cols)), shape=(A + B, E))
    b = np.concatenate([np.full(A, fib.k, np.float64), np.full(B, fib.demand_num, np.float64)])
    return M, b


def solve_transport(fib: Fiber, cost: np.ndarray) -> np.ndarray:
    """Exact integral min-cost flow on the k-scaled transportation polytope.

    The constraint matrix is totally unimodular and the right-hand side is integral, so
    every basic feasible solution is integral; HiGHS dual simplex returns a vertex.
    """
    if fib.n_edges > LP_MAX_VARS:
        raise MemoryError(f"{fib.n_edges} LP variables exceeds cap {LP_MAX_VARS}")
    M, b = _transport_constraints(fib)
    res = linprog(
        c=cost,
        A_eq=M,
        b_eq=b,
        bounds=(0, fib.k),
        method="highs-ds",
    )
    if not res.success:
        raise RuntimeError(f"transport LP failed at i={fib.i},k={fib.k}: {res.message}")
    f = np.rint(res.x).astype(np.int64)
    if not np.allclose(res.x, f, atol=1e-6):
        raise RuntimeError(
            f"LP vertex is not integral at i={fib.i},k={fib.k} "
            f"(max frac part {np.abs(res.x - f).max():.3g})"
        )
    _verify_flow(fib, f)
    return f


def _verify_flow(fib: Fiber, f: np.ndarray) -> None:
    """Exact integer verification of a k-scaled flow."""
    src_tot = np.bincount(fib.edge_src, weights=f, minlength=fib.n_sources).astype(np.int64)
    tgt_tot = np.bincount(fib.edge_tgt, weights=f, minlength=fib.n_targets).astype(np.int64)
    if not np.all(src_tot == fib.k):
        bad = int(np.argmax(src_tot != fib.k))
        raise RuntimeError(f"source {bad} sends {src_tot[bad]} units, expected {fib.k}")
    if not np.all(tgt_tot == fib.demand_num):
        bad = int(np.argmax(tgt_tot != fib.demand_num))
        raise RuntimeError(f"target {bad} receives {tgt_tot[bad]}, expected {fib.demand_num}")
    if f.min() < 0 or f.max() > fib.k:
        raise RuntimeError("flow out of bounds")


def flow_entropy(fib: Fiber, f: np.ndarray) -> float:
    """Mean per-acceptance eviction entropy (bits) of a k-scaled flow."""
    nz = f > 0
    q = f[nz] / fib.k
    contrib = -q * np.log2(q)
    total = float(np.bincount(fib.edge_src[nz], weights=contrib, minlength=fib.n_sources).sum())
    return total / fib.n_sources


def minimal_repair(fib: Fiber) -> dict[str, Any]:
    """Smallest perturbation of sum-mod-k that restores exact uniformity.

    Cost 0 on the sum-mod-k edge of each source, cost 1 on every other edge, on the
    k-scaled problem.  The optimal cost is the exact minimum number of supply UNITS
    that must leave their preferred child; the number of sources that are moved or
    split follows from the optimal flow.
    """
    pref_tgt = summodk_map(fib)
    is_pref = pref_tgt[fib.edge_src] == fib.edge_tgt
    cost = np.where(is_pref, 0.0, 1.0)
    f = solve_transport(fib, cost)

    units_moved = int(f[~is_pref].sum())
    pref_units = np.zeros(fib.n_sources, dtype=np.int64)
    np.add.at(pref_units, fib.edge_src[is_pref], f[is_pref])
    n_untouched = int((pref_units == fib.k).sum())
    deg = np.bincount(fib.edge_src[f > 0], minlength=fib.n_sources)
    n_split = int((deg > 1).sum())
    n_moved_whole = int(((deg == 1) & (pref_units == 0)).sum())

    return {
        "min_units_moved": units_moved,
        "min_units_moved_per_source": units_moved / (fib.k * fib.n_sources),
        "n_sources_untouched": n_untouched,
        "n_sources_split": n_split,
        "n_sources_moved_whole": n_moved_whole,
        "repair_fraction": (fib.n_sources - n_untouched) / fib.n_sources,
        "achieved_entropy_bits": flow_entropy(fib, f),
        "max_degree": int(deg.max()),
        "_flow": f,
        "_is_pref": is_pref,
    }


def min_entropy_flow(fib: Fiber) -> dict[str, Any]:
    """A low-entropy exactly-uniform flow (sampler S1f).

    Divisible steps: the integral unit-capacity certificate (deterministic, 0 bits).
    Non-divisible steps: the min-cost repair flow, whose achieved entropy is a
    constructive upper bound on the true minimum.
    """
    if fib.divisible:
        cert = integral_certificate(fib)
        if not cert["feasible"]:
            raise RuntimeError(
                f"integral feasibility FAILED at i={fib.i},k={fib.k} where theory says it "
                f"must hold: flow {cert['flow_value']} < {cert['required_flow']}"
            )
        f = np.zeros(fib.n_edges, dtype=np.int64)
        pos = {(a, b): e for e, (a, b) in enumerate(zip(fib.edge_src, fib.edge_tgt))}
        for S, T in cert["certificate"]:
            f[pos[(fib.src_index[S], fib.tgt_index[T])]] = fib.k
        _verify_flow(fib, f)
        return {"flow": f, "entropy_bits": 0.0, "source": "integral_certificate"}
    rep = minimal_repair(fib)
    return {"flow": rep["_flow"], "entropy_bits": rep["achieved_entropy_bits"], "source": "min_cost_repair"}


# --------------------------------------------------------------------------------------
# Minimum eviction entropy: exact MILP + pre-registered analytic bracket
# --------------------------------------------------------------------------------------


def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)


def entropy_bracket(i: int, k: int) -> dict[str, Any]:
    """Pre-registered bounds on the minimum per-acceptance eviction entropy (bits).

    UPPER: a minimising vertex of a transportation polytope has a forest support, so
    #edges <= A + B - 1 and sum_S H(q_S) <= sum_S (deg_S - 1) <= B - 1.
    LOWER: every vertex coordinate lies in (1/k)Z; when k does not divide (i+1) each
    target's fractional demand forces at least one fractionally-served source, and a
    split source has a part in [1/k, 1-1/k] so H(q_S) >= h(1/k).  A split source serves
    at most k targets, so at least ceil(B/k) sources split.
    """
    A, B = comb(i, k), comb(i, k - 1)
    divisible = (i + 1) % k == 0
    upper = (B - 1) / A
    if divisible:
        lower = 0.0
    else:
        lower = binary_entropy(1.0 / k) * (-(-B // k)) / A
    return {
        "A_sources": A,
        "B_targets": B,
        "divisible": divisible,
        "lower_bits": lower,
        "upper_bits": upper,
        "upper_asymptotic_k_over_i_minus_k_plus_1": k / (i - k + 1),
        "log2_k": log2(k) if k > 1 else 0.0,
    }


def min_entropy_exact(fib: Fiber, *, time_limit: float = MILP_TIME_LIMIT_S) -> dict[str, Any]:
    """Exact minimum per-acceptance eviction entropy via MILP.

    The continuous minimum of the concave objective sits at a vertex, and vertices of
    the k-scaled transportation polytope are integral, so the integer program below IS
    the exact continuous minimum -- not a relaxation of it.

    One-hot u(e,c) for c = 0..k with exact cost -(c/k)log2(c/k).
    """
    k, E = fib.k, fib.n_edges
    nvars = E * (k + 1)
    if fib.divisible:
        # No MILP needed and none is a weaker statement: eviction entropy is non-negative,
        # and the unit-capacity integral certificate ACHIEVES 0, so 0 is the exact minimum.
        return {
            "solved": True,
            "min_bits": 0.0,
            "reason": (
                "integral demands (k | i+1): the unit-capacity max-flow certificate sends every "
                "source's whole mass to one child, achieving 0 bits, and entropy is non-negative, "
                "so 0 is the exact minimum -- no search is required"
            ),
            "proof": "integral_certificate",
        }
    if nvars > MILP_MAX_VARS:
        return {"solved": False, "reason": f"{nvars} MILP vars exceeds cap {MILP_MAX_VARS}", "n_vars": nvars}

    cvals = np.arange(k + 1, dtype=np.float64)
    q = cvals / k
    with np.errstate(divide="ignore", invalid="ignore"):
        unit = np.where(q > 0, -q * np.log2(np.where(q > 0, q, 1.0)), 0.0)
    obj = np.tile(unit, E)

    ei = np.repeat(np.arange(E, dtype=np.int64), k + 1)
    vj = np.arange(nvars, dtype=np.int64)
    onehot = sp.csr_matrix((np.ones(nvars), (ei, vj)), shape=(E, nvars))

    amount = np.tile(cvals, E)
    src_rows = fib.edge_src.astype(np.int64).repeat(k + 1)
    tgt_rows = fib.edge_tgt.astype(np.int64).repeat(k + 1)
    src_M = sp.csr_matrix((amount, (src_rows, vj)), shape=(fib.n_sources, nvars))
    tgt_M = sp.csr_matrix((amount, (tgt_rows, vj)), shape=(fib.n_targets, nvars))

    cons = [
        LinearConstraint(onehot, 1, 1),
        LinearConstraint(src_M, k, k),
        LinearConstraint(tgt_M, fib.demand_num, fib.demand_num),
    ]
    res = milp(
        c=obj,
        constraints=cons,
        integrality=np.ones(nvars),
        bounds=(0, 1),
        options={"time_limit": time_limit, "presolve": True},
    )
    if res.x is None:
        return {"solved": False, "reason": f"MILP status {res.status}: {res.message}", "n_vars": nvars}
    total = float(res.fun)
    gap = float(getattr(res, "mip_gap", 0.0) or 0.0)
    if res.status != 0:
        # Time limit hit with an incumbent: it is still a valid CONSTRUCTIVE upper bound,
        # and mip_dual_bound is a valid lower bound.  Report both, do not claim "exact".
        dual = getattr(res, "mip_dual_bound", None)
        return {
            "solved": False,
            "reason": f"MILP status {res.status}: {res.message}",
            "incumbent_bits": total / fib.n_sources,
            "dual_bound_bits": (float(dual) / fib.n_sources) if dual is not None else None,
            "mip_gap": gap,
            "n_vars": nvars,
        }
    return {
        "solved": True,
        "min_bits": total / fib.n_sources,
        "milp_objective_total": total,
        "mip_gap": gap,
        "n_vars": nvars,
    }


# --------------------------------------------------------------------------------------
# Per-step analysis, cached to disk
# --------------------------------------------------------------------------------------


def analyse_step(i: int, k: int, *, want_milp: bool = True, save_cert: bool = True) -> dict[str, Any]:
    """Everything M3 needs for the step (prefix i, arriving item i+1)."""
    fib = build_fiber(i, k)
    br = entropy_bracket(i, k)
    bal = summodk_balance(fib)
    cert = integral_certificate(fib)

    row: dict[str, Any] = {
        "i_prefix": i,
        "arriving_item": i + 1,
        "k": k,
        "n_sources": fib.n_sources,
        "n_targets": fib.n_targets,
        "n_edges": fib.n_edges,
        "divisible_form_a_i_minus_k_plus_1": (i - k + 1) % k == 0,
        "divisible_form_b_i_plus_1": (i + 1) % k == 0,
        "both_forms_agree": True,
        "divisible": fib.divisible,
        "demand_per_target": f"{fib.demand_num}/{k}",
        "demand_per_target_float": fib.demand_num / k,
        "transport_arithmetic_ok": fib.n_sources * k == fib.n_targets * (i - k + 1),
        "summodk": bal,
        "bracket": br,
    }

    if cert["feasible"]:
        c = cert["certificate"]
        chk = check_certificate(c, i, k)
        h = certificate_hash(c)
        row["integral_feasibility"] = {
            "divisible": True,
            "feasible": True,
            "flow_value": cert["flow_value"],
            "required_flow": cert["required_flow"],
            "per_target_load": cert["per_target_load"],
            "checker": chk,
            "certificate_sha256": h,
            "certificate_n_rows": len(c),
            "certificate_sample": [[list(s), list(t)] for s, t in c[:20]],
            "certificate_inline": (
                [[list(s), list(t)] for s, t in c] if i <= 14 else None
            ),
        }
        if not chk["valid"]:
            raise RuntimeError(f"certificate checker REJECTED i={i},k={k}: {chk['errors']}")
        if save_cert:
            CERT_DIR.mkdir(parents=True, exist_ok=True)
            p = CERT_DIR / f"cert_i{i}_k{k}.json"
            p.write_text(
                json.dumps(
                    {
                        "i": i,
                        "k": k,
                        "per_target_load": cert["per_target_load"],
                        "sha256": h,
                        "map": [[list(s), list(t)] for s, t in c],
                    },
                    separators=(",", ":"),
                )
            )
            row["integral_feasibility"]["certificate_file"] = p.name
    else:
        row["integral_feasibility"] = {k2: v for k2, v in cert.items() if k2 != "certificate"}

    rep = minimal_repair(fib)
    row["minimal_repair"] = {kk: vv for kk, vv in rep.items() if not kk.startswith("_")}
    row["minimal_repair"]["predicted_decay_k_over_i_minus_k"] = k / max(i - k, 1)

    achieved = min_entropy_flow(fib)
    row["achieved_entropy_bits"] = achieved["entropy_bits"]
    row["achieved_entropy_source"] = achieved["source"]

    exact = min_entropy_exact(fib) if want_milp else {"solved": False, "reason": "milp disabled"}
    row["min_entropy_exact"] = exact

    lo, hi = br["lower_bits"], br["upper_bits"]
    best = exact["min_bits"] if exact.get("solved") else None
    inc = exact.get("incumbent_bits")
    if best is None and inc is not None and inc < achieved["entropy_bits"]:
        achieved["entropy_bits"] = inc
        achieved["source"] = "milp_incumbent_time_limited"
        row["achieved_entropy_bits"] = inc
        row["achieved_entropy_source"] = achieved["source"]
    reported = best if best is not None else achieved["entropy_bits"]
    row["min_evict_entropy_bits"] = reported
    row["min_evict_entropy_is_exact"] = best is not None
    row["classical_evict_bits"] = log2(k) if k > 1 else 0.0
    row["ratio_to_log2k"] = (reported / log2(k)) if k > 1 else 0.0

    tol = 1e-9
    if best is not None and not (lo - tol <= best <= hi + tol):
        raise RuntimeError(
            f"BRACKET VIOLATION at i={i},k={k}: exact {best} not in [{lo}, {hi}] -- this is a bug"
        )
    if not (achieved["entropy_bits"] <= hi + tol):
        raise RuntimeError(
            f"BRACKET VIOLATION at i={i},k={k}: achieved {achieved['entropy_bits']} > upper {hi}"
        )
    row["bracket_check_passed"] = True
    return row


def analyse_cell(args: tuple[int, int, bool]) -> dict[str, Any]:
    i, k, want_milp = args
    return analyse_step(i, k, want_milp=want_milp)


# --------------------------------------------------------------------------------------
# Eviction tables consumed by the samplers in e1_candidates.py
# --------------------------------------------------------------------------------------

_TABLE_CACHE: dict[tuple[str, int, int], dict[tuple[int, ...], list[tuple[int, Fraction]]]] = {}


def eviction_table(rule: str, i: int, k: int) -> dict[tuple[int, ...], list[tuple[int, Fraction]]]:
    """Map: sorted k-subset S of [i] -> [(position_evicted_0based, exact probability)].

    ``rule`` is ``"repair"`` (sum-mod-k minimally repaired, sampler S1) or ``"flow"``
    (min-entropy flow, sampler S1f).
    """
    key = (rule, i, k)
    if key in _TABLE_CACHE:
        return _TABLE_CACHE[key]
    fib = build_fiber(i, k)
    if rule == "repair":
        f = minimal_repair(fib)["_flow"]
    elif rule == "flow":
        f = min_entropy_flow(fib)["flow"]
    else:
        raise ValueError(f"unknown eviction table rule {rule!r}")
    _verify_flow(fib, f)

    table: dict[tuple[int, ...], list[tuple[int, Fraction]]] = {}
    nz = np.nonzero(f)[0]
    by_src: dict[int, list[tuple[int, int]]] = {}
    for e in nz:
        by_src.setdefault(int(fib.edge_src[e]), []).append((int(fib.edge_removed[e]), int(f[e])))
    for a, S in enumerate(fib.sources):
        entries = by_src.get(a, [])
        if not entries:
            raise RuntimeError(f"source {S} has no outgoing flow at i={i},k={k}")
        table[S] = [(S.index(x), Fraction(c, fib.k)) for x, c in entries]
        assert sum(p for _, p in table[S]) == 1
    _TABLE_CACHE[key] = table
    return table


__all__ = [
    "Fiber",
    "build_fiber",
    "summodk_position",
    "summodk_map",
    "summodk_balance",
    "integral_certificate",
    "check_certificate",
    "certificate_hash",
    "minimal_repair",
    "min_entropy_flow",
    "min_entropy_exact",
    "entropy_bracket",
    "binary_entropy",
    "analyse_step",
    "analyse_cell",
    "eviction_table",
    "flow_entropy",
]
