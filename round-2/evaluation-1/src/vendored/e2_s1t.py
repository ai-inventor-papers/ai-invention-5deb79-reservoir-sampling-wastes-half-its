#!/usr/bin/env python3
"""S1t: the MINIMUM-RANDOMIZATION eviction rule for sorted-reservoir sampling.

Background
----------
The stream is the index sequence 1..n in increasing order, reservoir size k,
reservoir kept SORTED (an accepted item always exceeds every held item, so
"delete the r-th smallest, then append at the end" preserves sortedness).  At
stream position i the sampler accepts item i with probability exactly k/i.
Conditional on acceptance, the transition must map

    SOURCES  = k-subsets S of [i-1]        (count Ns = C(i-1, k))
    TARGETS  = (k-1)-subsets A of [i-1]    (count Nt = C(i-1, k-1))

uniformly: each source has probability 1/Ns, each target must receive exactly
1/Nt.  A source picks a rank r in {0..k-1} (the index, within the SORTED
source tuple, of the element evicted) and the child state is A U {i} where
A = S minus its r-th smallest element.  Each target has exactly m = i - k
preimages, and Ns/Nt = m/k, so each target must receive exactly m/k units of
source mass.  Write d = m // k, s = m % k.

A deterministic rank map r = (sum(S) - 1) mod k is EXACT whenever s == 0 (this
has already been verified with fractions.Fraction for n <= 14, k = 2..5 --
see the parent 'S1' candidate in e2_candidates.py).  This module supplies the
transition for the remaining steps, where s != 0 and no purely deterministic
map can balance the fibre: S1t assigns the maximum possible number of sources
(d = m // k, when feasible) deterministically via an integral b-matching, and
resolves only the U = Ns - Nt*d leftover sources with a MINIMAL, e2_exact,
integer-flow-derived randomisation -- as opposed to S1's "uniform-int(k)"
fallback, which randomises every s != 0 acceptance in full.

Node layout for every scipy.sparse.csgraph.maximum_flow call in this module:
0 = super-source, 1..Ns = sources, Ns+1..Ns+Nt = targets, Ns+Nt+1 = super-sink
(``Ns``/``Nt`` here mean whatever the CURRENT stage's source/target counts are
-- all Ns sources for stage 1, only the U unmatched ones for stage 2).
"""

from __future__ import annotations

import itertools
import math
from collections import defaultdict
from fractions import Fraction
from typing import Sequence

import numpy as np
from loguru import logger
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import maximum_flow
from scipy.special import polygamma

from e2_budget import evict_H
from e2_candidates import Candidate

__all__ = [
    "colex_rank",
    "colex_unrank",
    "solve_step",
    "s1t_children",
    "verify_s1t",
    "h_evict_curve",
    "extrapolate_total",
]

# --------------------------------------------------------------------------- #
#  1. colex rank / unrank -- the combinatorial number system                  #
# --------------------------------------------------------------------------- #
def colex_rank(subset: Sequence[int], k: int) -> int:
    """Colexicographic rank of a k-subset of the positive integers, in 0..C(N,k)-1.

    ``subset`` elements are 1-indexed (matching stream positions). Internally
    uses the combinatorial number system: for a_1 < a_2 < ... < a_k (0-indexed),
    rank = sum_j C(a_j, j) for j = 1..k -- this enumerates subsets in exactly
    colex order (subsets compared by largest element first, then next-largest).
    """
    a = sorted(subset)
    if len(a) != k:
        raise ValueError(f"subset must have exactly k={k} elements, got {len(a)}")
    rank = 0
    for pos, val in enumerate(a):
        c = val - 1
        j = pos + 1
        if c < j - 1:
            raise ValueError(f"invalid k-subset {subset!r}: not enough room at position {pos}")
        rank += math.comb(c, j)
    return rank


def colex_unrank(rank: int, k: int, universe_size: int) -> tuple[int, ...]:
    """Inverse of :func:`colex_rank`: the rank-th k-subset of {1..universe_size}."""
    total = math.comb(universe_size, k)
    if not (0 <= rank < total):
        raise ValueError(f"rank {rank} out of range for C({universe_size},{k})={total}")
    result: list[int] = []
    r = rank
    hi = universe_size - 1
    for j in range(k, 0, -1):
        lo = j - 1
        high = hi
        while lo < high:
            mid = (lo + high + 1) // 2
            if math.comb(mid, j) <= r:
                lo = mid
            else:
                high = mid - 1
        c = lo
        result.append(c)
        r -= math.comb(c, j)
        hi = c - 1
    result.reverse()
    return tuple(x + 1 for x in result)


def _colex_roundtrip_test(universe_size: int, k: int) -> dict:
    """Round-trip colex_rank/colex_unrank over every k-subset of {1..universe_size}."""
    n_subsets = math.comb(universe_size, k)
    rank_to_subset_ok = True
    subset_to_rank_ok = True
    checked_forward = 0
    checked_backward = 0
    first_forward_failure: int | None = None
    first_backward_failure: list[int] | None = None
    for subset in itertools.combinations(range(1, universe_size + 1), k):
        rk = colex_rank(subset, k)
        back = colex_unrank(rk, k, universe_size)
        checked_forward += 1
        if back != subset:
            subset_to_rank_ok = False
            if first_backward_failure is None:
                first_backward_failure = list(subset)
    for rk in range(n_subsets):
        subset = colex_unrank(rk, k, universe_size)
        back_rank = colex_rank(subset, k)
        checked_backward += 1
        if back_rank != rk:
            rank_to_subset_ok = False
            if first_forward_failure is None:
                first_forward_failure = rk
    return {
        "universe_size": universe_size,
        "k": k,
        "n_subsets": n_subsets,
        "checked_subset_to_rank_to_subset": checked_forward,
        "checked_rank_to_subset_to_rank": checked_backward,
        "subset_roundtrip_ok": subset_to_rank_ok,
        "rank_roundtrip_ok": rank_to_subset_ok,
        "all_ok": subset_to_rank_ok and rank_to_subset_ok,
        "first_backward_failure_subset": first_backward_failure,
        "first_forward_failure_rank": first_forward_failure,
    }


# --------------------------------------------------------------------------- #
#  2. solve_step -- the minimum-randomization transition for one prefix i     #
# --------------------------------------------------------------------------- #
def _build_edges(i: int, k: int) -> tuple[list[tuple[int, ...]], list[tuple[int, ...]], dict, list[tuple[int, int, int]]]:
    """Sources (k-subsets), targets ((k-1)-subsets), target index, and every
    (source_eidx, rank_r, target_eidx) edge of the (k, m)-biregular bipartite
    source/target graph on prefix i-1."""
    sources = list(itertools.combinations(range(1, i), k))
    targets = list(itertools.combinations(range(1, i), k - 1))
    tgt_idx = {t: e for e, t in enumerate(targets)}
    edges: list[tuple[int, int, int]] = []
    for s_e, S in enumerate(sources):
        for r in range(k):
            A = S[:r] + S[r + 1 :]
            edges.append((s_e, r, tgt_idx[A]))
    return sources, targets, tgt_idx, edges


def _maxflow_value_and_flow(n_nodes: int, rows: list[int], cols: list[int], caps: list[int], src: int, sink: int):
    graph = coo_matrix((np.asarray(caps, dtype=np.int32), (rows, cols)), shape=(n_nodes, n_nodes)).tocsr()
    res = maximum_flow(graph, src, sink)
    return int(res.flow_value), res.flow


def _stage1_match(
    Ns: int, Nt: int, edges: list[tuple[int, int, int]], d: int, source_perm: np.ndarray | None = None
) -> tuple[dict[int, int], int]:
    """Integral b-matching: exactly d DISTINCT sources routed fully to each target.

    A max-flow solver returns SOME feasible b-matching of the required VALUE
    (Nt*d), but is free to pick one that happens to exhaust every incident
    source of some target A on OTHER targets, starving A's own stage-2
    residual even though a well-spread matching (e.g. the complement of a
    "balanced" selection) would not.  ``source_perm`` relabels which node id
    each source occupies before the solve, which changes the search order the
    underlying algorithm explores and hence which max-flow it returns; the
    retry loop in :func:`solve_step` tries several such relabelings before
    giving up on this ``d`` and reducing it.

    Returns {source_eidx: rank_r} for the matched sources, and the achieved
    total flow (should equal Nt*d; caller checks).
    """
    if d == 0:
        return {}, 0
    if source_perm is None:
        source_perm = np.arange(Ns)
    n_nodes = Ns + Nt + 2
    src_node, sink_node = 0, Ns + Nt + 1
    rows: list[int] = [src_node] * Ns
    cols: list[int] = list(range(1, Ns + 1))
    caps: list[int] = [1] * Ns
    for s_e, _r, t_e in edges:
        rows.append(int(source_perm[s_e]) + 1)
        cols.append(Ns + 1 + t_e)
        caps.append(1)
    for t_e in range(Nt):
        rows.append(Ns + 1 + t_e)
        cols.append(sink_node)
        caps.append(d)
    flow_value, flow = _maxflow_value_and_flow(n_nodes, rows, cols, caps, src_node, sink_node)
    if flow_value != Nt * d:
        return {}, flow_value
    s_arr = np.array([int(source_perm[s_e]) + 1 for s_e, _r, _t in edges])
    t_arr = np.array([Ns + 1 + t_e for _s, _r, t_e in edges])
    vals = np.asarray(flow[s_arr, t_arr]).flatten()
    matched: dict[int, int] = {}
    for (s_e, r, _t_e), v in zip(edges, vals):
        if v > 0:
            matched[s_e] = r
    return matched, flow_value


def _stage2_flow(
    unmatched_eidx: list[int], Nt: int, edges: list[tuple[int, int, int]], k: int, residual: int
) -> tuple[int, dict[tuple[int, int, int], int]]:
    """k-scaled integer flow: each unmatched source supplies k units, each target
    demands ``residual`` units, each edge capacity k.  Returns the achieved
    total flow (should equal Nt*residual; caller checks) and the per-edge flow."""
    U = len(unmatched_eidx)
    local = {s_e: li for li, s_e in enumerate(unmatched_eidx)}
    n_nodes = U + Nt + 2
    src_node, sink_node = 0, U + Nt + 1
    rows: list[int] = [src_node] * U
    cols: list[int] = list(range(1, U + 1))
    caps: list[int] = [k] * U
    relevant: list[tuple[int, int, int]] = []
    for s_e, r, t_e in edges:
        li = local.get(s_e)
        if li is None:
            continue
        relevant.append((s_e, r, t_e))
        rows.append(li + 1)
        cols.append(U + 1 + t_e)
        caps.append(k)
    for t_e in range(Nt):
        rows.append(U + 1 + t_e)
        cols.append(sink_node)
        caps.append(residual)
    flow_value, flow = _maxflow_value_and_flow(n_nodes, rows, cols, caps, src_node, sink_node)
    edge_flow: dict[tuple[int, int, int], int] = {}
    if relevant:
        r_arr = np.array([local[s_e] + 1 for s_e, _r, _t in relevant])
        c_arr = np.array([U + 1 + t_e for _s, _r, t_e in relevant])
        vals = np.asarray(flow[r_arr, c_arr]).flatten()
        for (s_e, r, t_e), v in zip(relevant, vals):
            if v > 0:
                edge_flow[(s_e, r, t_e)] = int(v)
    return flow_value, edge_flow


def solve_step(i: int, k: int, max_sources: int = 60_000) -> dict:
    """Minimum-randomization transition table for stream position i, reservoir k."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if i <= k:
        raise ValueError(f"i must exceed k (no eviction happens at/before the fill), got i={i}, k={k}")
    Ns = math.comb(i - 1, k)
    if Ns > max_sources:
        raise ValueError(f"C(i-1,k)={Ns} exceeds max_sources={max_sources} at i={i}, k={k}")
    Nt = math.comb(i - 1, k - 1)
    m = i - k
    d_target, s = divmod(m, k)
    sources, targets, tgt_idx, edges = _build_edges(i, k)
    max_rank_entropy = math.log2(k) if k > 1 else 0.0

    if s == 0:
        table: dict[int, int | list[int]] = {}
        per_target_count = [0] * Nt
        for S in sources:
            r = (sum(S) - 1) % k
            rank = colex_rank(S, k)
            table[rank] = r
            A = S[:r] + S[r + 1 :]
            per_target_count[tgt_idx[A]] += 1
        for t_e, cnt in enumerate(per_target_count):
            if cnt != d_target:
                raise AssertionError(
                    f"closed-form sum-mod-k rule violated exactness invariant at i={i}, k={k}, "
                    f"target {targets[t_e]}: got {cnt} deterministic sources, expected d={d_target}"
                )
        return {
            "i": i, "k": k, "m": m, "d_target": d_target, "d_achieved": d_target, "s": s,
            "n_sources": Ns, "n_targets": Nt, "frac_randomised": 0.0, "H_evict_bits": 0.0,
            "max_rank_entropy": max_rank_entropy, "table": table, "feasible": True,
            "stage2_flow_value": 0, "stage2_demand": 0,
        }

    d_achieved: int | None = None
    matched: dict[int, int] = {}
    unmatched_eidx: list[int] = []
    stage2_flow_value = 0
    stage2_demand = 0
    edge_flow: dict[tuple[int, int, int], int] = {}
    max_relabel_attempts = 30
    for d_try in range(d_target, -1, -1):
        found_at_this_d = False
        n_attempts = max_relabel_attempts if d_try > 0 else 1
        for attempt in range(n_attempts):
            if d_try > 0:
                rng = np.random.default_rng((i, k, d_try, attempt))
                source_perm = rng.permutation(Ns)
                m_try, flow1_value = _stage1_match(Ns, Nt, edges, d_try, source_perm=source_perm)
                if flow1_value != Nt * d_try:
                    logger.warning(
                        f"i={i},k={k}: stage-1 b-matching short at d={d_try}, attempt={attempt} "
                        f"(got {flow1_value}, wanted {Nt * d_try})"
                    )
                    continue
            else:
                m_try = {}
            u_try = [e for e in range(Ns) if e not in m_try]
            residual = m - d_try * k
            demand = Nt * residual
            flow2_value, ef_try = _stage2_flow(u_try, Nt, edges, k, residual)
            if flow2_value == demand:
                found_at_this_d = True
                break
            logger.warning(
                f"i={i},k={k}: stage-2 flow short at d={d_try}, attempt={attempt} "
                f"(got {flow2_value}, wanted {demand})"
            )
        if found_at_this_d:
            d_achieved = d_try
            matched = m_try
            unmatched_eidx = u_try
            stage2_flow_value = flow2_value
            stage2_demand = demand
            edge_flow = ef_try
            break
        logger.warning(f"i={i},k={k}: exhausted {n_attempts} relabelings at d={d_try}; retrying with d={d_try - 1}")
    if d_achieved is None:
        raise RuntimeError(
            f"no feasible d found at i={i}, k={k}, m={m} -- even d=0 (fully uniform) failed; "
            "this contradicts the always-feasible uniform-rank fallback and indicates a bug"
        )

    table = {}
    per_target_det = [0] * Nt
    per_target_rand = [0] * Nt
    for s_e, r in matched.items():
        S = sources[s_e]
        table[colex_rank(S, k)] = r
        A = S[:r] + S[r + 1 :]
        per_target_det[tgt_idx[A]] += 1

    y_by_source: dict[int, list[int]] = {e: [0] * k for e in unmatched_eidx}
    for (s_e, r, t_e), flow_val in edge_flow.items():
        y_by_source[s_e][r] = flow_val
        per_target_rand[t_e] += flow_val

    H_sum = 0.0
    for s_e in unmatched_eidx:
        S = sources[s_e]
        y = y_by_source[s_e]
        if sum(y) != k:
            raise AssertionError(f"source {S} y-vector {y} does not sum to k={k} at i={i}, k={k}")
        table[colex_rank(S, k)] = y
        for yy in y:
            if yy > 0:
                p = yy / k
                H_sum += -p * math.log2(p)

    for t_e in range(Nt):
        total = per_target_det[t_e] * k + per_target_rand[t_e]
        if total != m:
            raise AssertionError(
                f"EXACTNESS INVARIANT VIOLATED at i={i}, k={k}, target {targets[t_e]}: "
                f"det*k + stage2_flow = {per_target_det[t_e]}*{k} + {per_target_rand[t_e]} = {total} != m={m}"
            )

    if len(table) != Ns:
        raise AssertionError(f"incomplete table at i={i}, k={k}: {len(table)} entries, expected Ns={Ns}")

    U = len(unmatched_eidx)
    return {
        "i": i, "k": k, "m": m, "d_target": d_target, "d_achieved": d_achieved, "s": s,
        "n_sources": Ns, "n_targets": Nt, "frac_randomised": U / Ns, "H_evict_bits": H_sum / Ns,
        "max_rank_entropy": max_rank_entropy, "table": table, "feasible": True,
        "stage2_flow_value": stage2_flow_value, "stage2_demand": stage2_demand,
    }


# --------------------------------------------------------------------------- #
#  3. s1t_children -- the e2_exact-verifier / sampler transition kernel          #
# --------------------------------------------------------------------------- #
_TABLE_CACHE: dict[tuple[int, int], dict] = {}


def _get_table(i: int, k: int, max_sources: int = 60_000) -> dict[int, int | list[int]]:
    key = (i, k)
    if key not in _TABLE_CACHE:
        _TABLE_CACHE[key] = solve_step(i, k, max_sources=max_sources)
    return _TABLE_CACHE[key]["table"]


def s1t_children(state: tuple[tuple[int, ...], tuple], i: int, k: int):
    """Exact transition kernel for S1t, in the shape of ``Candidate.children``.

    ``state`` is ``(sorted_slots_tuple, aux_tuple)``; yields
    ``((new_sorted_slots_tuple, aux), fractions.Fraction(prob))`` pairs.
    Deterministic sum-mod-k rank when (i-k) % k == 0 (no table lookup needed at
    all); otherwise consults the memoised minimum-randomization table.
    """
    slots, aux = state
    m = i - k
    if m % k == 0:
        r = (sum(slots) - 1) % k
        yield (slots[:r] + slots[r + 1 :] + (i,), aux), Fraction(1)
        return
    table = _get_table(i, k)
    src_rank = colex_rank(slots, k)
    entry = table[src_rank]
    if isinstance(entry, int):
        r = entry
        yield (slots[:r] + slots[r + 1 :] + (i,), aux), Fraction(1)
    else:
        for r, y in enumerate(entry):
            if y:
                yield (slots[:r] + slots[r + 1 :] + (i,), aux), Fraction(y, k)


# --------------------------------------------------------------------------- #
#  4. verify_s1t -- e2_exact rational forward propagation (the acceptance test)  #
# --------------------------------------------------------------------------- #
def verify_s1t(n: int, k: int) -> dict:
    """Exact rational prefix verification of S1t, mirroring e2_exact.verify's loop
    shape but driven locally (e2_candidates.py / e2_exact.py are not touched)."""
    cand = Candidate(
        "S1t", "SUMMODK+MINFLOW", scalar=None, vector=None, children=s1t_children, family="e2_s1t",
        note="Minimum-randomization repair of the sum-mod-k eviction rank.",
    )
    init = (tuple(range(1, k + 1)), cand.init_aux)
    dist: dict = {init: Fraction(1)}
    per_prefix: list[dict] = []
    first_fail: int | None = None

    for i in range(k + 1, n + 1):
        num, den = cand.accept_num(i, k), cand.accept_den(i, k)
        p = Fraction(num, den)
        nxt: dict = defaultdict(Fraction)
        for state, pr in dist.items():
            if p < 1:
                nxt[state] += pr * (1 - p)
            if p > 0:
                for child, q in cand.children(state, i, k):
                    nxt[child] += pr * p * q
        dist = {st: v for st, v in nxt.items() if v != 0}

        sub: dict = defaultdict(Fraction)
        for state, pr in dist.items():
            sub[tuple(sorted(state[0]))] += pr
        n_subsets = math.comb(i, k)
        target = Fraction(1, n_subsets)
        max_abs = max(abs(v - target) for v in sub.values())
        if len(sub) < n_subsets:
            max_abs = max(max_abs, target)
        if max_abs != 0 and first_fail is None:
            first_fail = i
        per_prefix.append(
            {
                "prefix": i,
                "max_abs_dev": float(max_abs),
                "max_rel_dev": float(max_abs / target),
                "support_size": len(sub),
                "n_subsets": n_subsets,
                "exact_uniform": max_abs == 0,
            }
        )

    final = per_prefix[-1]
    return {
        "n": n, "k": k,
        "max_abs_dev": final["max_abs_dev"],
        "max_rel_dev": final["max_rel_dev"],
        "support_size": final["support_size"],
        "n_subsets": final["n_subsets"],
        "first_prefix_failure": first_fail,
        "exact_uniform_at_all_prefixes": all(r["exact_uniform"] for r in per_prefix),
        "per_prefix": per_prefix,
    }


# --------------------------------------------------------------------------- #
#  5. h_evict_curve -- the measured eviction-entropy curve and its fit        #
# --------------------------------------------------------------------------- #
def h_evict_curve(k: int, i_max_sources: int = 60_000, i_cap: int = 400) -> dict:
    """Solve every reachable prefix i for reservoir k, and fit
    H_evict(i,k) ~= C * k*log2(k)/i over the s != 0 (randomised) steps."""
    rows: list[dict] = []
    for i in range(k + 1, i_cap + 1):
        if math.comb(i - 1, k) > i_max_sources:
            break  # C(i-1,k) is monotone increasing in i -- nothing further is reachable
        rows.append(solve_step(i, k, max_sources=i_max_sources))

    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        if row["s"] != 0:
            xs.append(k * math.log2(k) / row["i"])
            ys.append(row["H_evict_bits"])

    if xs:
        xs_a, ys_a = np.asarray(xs), np.asarray(ys)
        denom = float(np.sum(xs_a * xs_a))
        fit_C = float(np.sum(xs_a * ys_a) / denom) if denom > 0 else float("nan")
        y_pred = fit_C * xs_a
        ss_res = float(np.sum((ys_a - y_pred) ** 2))
        ss_tot = float(np.sum((ys_a - np.mean(ys_a)) ** 2))
        fit_r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else (1.0 if ss_res == 0 else float("nan"))
    else:
        fit_C, fit_r2 = float("nan"), float("nan")

    avg_H = float(np.mean([row["H_evict_bits"] for row in rows])) if rows else 0.0
    return {
        "k": k, "i_max_sources": i_max_sources, "i_cap": i_cap,
        "n_rows": len(rows), "n_randomised_points": len(xs),
        "fit_C": fit_C, "fit_R2": fit_r2,
        "avg_H_evict_bits_all_i": avg_H,
        "max_i_solved": rows[-1]["i"] if rows else None,
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
#  6. extrapolate_total -- the closed-form / extrapolated eviction totals     #
# --------------------------------------------------------------------------- #
def extrapolate_total(
    k: int, n: int, fit_C: float, i_max_sources: int = 20_000, i_cap: int = 300
) -> dict:
    """Extrapolated S1t total eviction e2_bits over a stream of length n, plus the
    EXACT partial sum over the i values actually solved, plus the classical
    comparison and the n -> infinity limit."""
    effective_cap = min(i_cap, n)
    curve = h_evict_curve(k, i_max_sources=i_max_sources, i_cap=effective_cap)
    exact_rows = curve["rows"]
    exact_partial_sum = sum((k / row["i"]) * row["H_evict_bits"] for row in exact_rows)

    const = fit_C * (k**2) * math.log2(k) if k > 1 else 0.0
    sum_1_over_i2_finite = float(polygamma(1, k + 1) - polygamma(1, n + 1))
    sum_1_over_i2_inf = float(polygamma(1, k + 1))
    total_extrap_finite_n = const * sum_1_over_i2_finite
    total_extrap_n_to_inf = const * sum_1_over_i2_inf

    classical = evict_H(n, k)
    ratio_finite = total_extrap_finite_n / classical if classical > 0 else float("nan")
    ratio_inf = total_extrap_n_to_inf / classical if classical > 0 else float("nan")

    return {
        "k": k, "n": n, "fit_C_used": fit_C,
        "e2_exact": {
            "basis": "e2_exact",
            "i_max_sources": i_max_sources, "i_cap": effective_cap,
            "max_i_solved": exact_rows[-1]["i"] if exact_rows else None,
            "n_points_solved": len(exact_rows),
            "partial_sum_evict_bits_over_solved_i": exact_partial_sum,
        },
        "extrapolated": {
            "basis": "extrapolated-from-e2_exact",
            "model": "H_evict_hat(i,k) = fit_C * k*log2(k) / i",
            "total_evict_bits_at_n": total_extrap_finite_n,
            "total_evict_bits_as_n_to_infinity": total_extrap_n_to_inf,
            "classical_evict_H_bits_at_n": classical,
            "ratio_s1t_over_classical_at_n": ratio_finite,
            "ratio_s1t_over_classical_as_n_to_infinity": ratio_inf,
        },
    }


# --------------------------------------------------------------------------- #
#  main                                                                       #
# --------------------------------------------------------------------------- #
def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(key): _json_safe(val) for key, val in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, Fraction):
        return _json_safe(float(obj))
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        obj = float(obj)
    if isinstance(obj, float):
        if math.isnan(obj):
            return "nan"
        if math.isinf(obj):
            return "inf" if obj > 0 else "-inf"
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def _slim_curve(curve: dict) -> dict:
    """Drop each row's full transition table for console printing (it is the
    complete sampler -- correct and returned by h_evict_curve/solve_step -- but
    is far too large to usefully print as JSON); keep everything else."""
    slim_rows = []
    for row in curve["rows"]:
        slim = {key: val for key, val in row.items() if key != "table"}
        slim["table_n_entries"] = len(row["table"])
        slim_rows.append(slim)
    out = dict(curve)
    out["rows"] = slim_rows
    return out


@logger.catch(reraise=True)
def main() -> None:
    import json

    output: dict = {}

    logger.info("(a) colex round-trip test on C(12,4)=495 subsets")
    output["colex_roundtrip"] = _colex_roundtrip_test(universe_size=12, k=4)

    logger.info("(b) verify_s1t e2_exact-uniformity acceptance test, k in {2,3,4} at n=12")
    output["verify_s1t"] = {}
    for k in (2, 3, 4):
        res = verify_s1t(n=12, k=k)
        slim = {key: val for key, val in res.items() if key != "per_prefix"}
        slim["per_prefix"] = res["per_prefix"]
        output["verify_s1t"][str(k)] = slim
        logger.info(
            f"  k={k}: max_abs_dev={res['max_abs_dev']!r} exact_uniform_at_all_prefixes="
            f"{res['exact_uniform_at_all_prefixes']} first_prefix_failure={res['first_prefix_failure']}"
        )

    logger.info("(c) h_evict_curve for k in {2,3,4,5}, small caps for speed")
    output["h_evict_curve"] = {}
    fit_by_k: dict[int, float] = {}
    for k in (2, 3, 4, 5):
        curve = h_evict_curve(k, i_max_sources=3000, i_cap=80)
        fit_by_k[k] = curve["fit_C"]
        output["h_evict_curve"][str(k)] = _slim_curve(curve)
        logger.info(
            f"  k={k}: n_rows={curve['n_rows']} n_randomised_points={curve['n_randomised_points']} "
            f"fit_C={curve['fit_C']:.6f} fit_R2={curve['fit_R2']:.6f} "
            f"avg_H_evict_bits_all_i={curve['avg_H_evict_bits_all_i']:.6f}"
        )

    largest_k_solved = max(fit_by_k)
    fit_C_for_extrapolation = fit_by_k[largest_k_solved]
    logger.info(
        f"(d) extrapolate_total for k in {{10,100}} at n in {{1e6,1e7}}, "
        f"using fit_C={fit_C_for_extrapolation:.6f} measured at k={largest_k_solved} "
        "-- extrapolating this fit to k=10/100 is an ASSUMPTION, not a measurement"
    )
    output["extrapolate_total"] = {
        "fit_C_source_k": largest_k_solved,
        "fit_C_value": fit_C_for_extrapolation,
        "note": (
            f"fit_C was measured only up to k={largest_k_solved}; applying it to k=10 and "
            "k=100 below assumes the C*k*log2(k)/i scaling law extrapolates across k, which "
            "has NOT been independently measured at those k."
        ),
        "results": {},
    }
    for k in (10, 100):
        output["extrapolate_total"]["results"][str(k)] = {}
        for n in (1_000_000, 10_000_000):
            res = extrapolate_total(k, n, fit_C_for_extrapolation)
            output["extrapolate_total"]["results"][str(k)][str(n)] = res
            logger.info(
                f"  k={k}, n={n}: extrapolated_total={res['extrapolated']['total_evict_bits_at_n']:.4f} "
                f"classical={res['extrapolated']['classical_evict_H_bits_at_n']:.4f} "
                f"ratio={res['extrapolated']['ratio_s1t_over_classical_at_n']:.6e} "
                f"n_to_inf_limit={res['extrapolated']['total_evict_bits_as_n_to_infinity']:.4f} "
                f"exact_partial_sum(i<={res['e2_exact']['max_i_solved']})="
                f"{res['e2_exact']['partial_sum_evict_bits_over_solved_i']:.6f}"
            )

    print(json.dumps(_json_safe(output), indent=2))


if __name__ == "__main__":
    main()
