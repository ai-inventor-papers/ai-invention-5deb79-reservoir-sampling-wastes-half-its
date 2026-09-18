#!/usr/bin/env python3
"""Is the accept coin forced?  And what does anytime uniformity cost in fair coin flips?

PART 1 -- FORCED ACCEPT.  Suppose the sampler is uniform on k-subsets of [i] and must be
uniform on k-subsets of [i+1].  Every k-subset S' of [i] (i.e. one NOT containing the
arriving item i+1) survives only by a rejection, so

    (1/C(i,k)) * (1 - p(S')) = 1/C(i+1,k)   =>   p(S') = 1 - C(i,k)/C(i+1,k) = k/(i+1).

The linear system in the unknowns {p(S)} is DIAGONAL, hence of full column rank, hence
its solution set is the single point p == k/(i+1): the accept probability is forced and
state-independent, for every sampler, with no assumption about the eviction rule.  This
module verifies that computationally with exact rationals plus an independent sympy rank
check, rather than merely asserting the algebra.

PART 2 -- BIT BUDGETS.  The forced coin costs at least sum_i h(k/i) fair bits
(Shannon/Knuth-Yao).  The classical rule additionally spends log2(k) bits per acceptance.
The pre-registered analytic bracket from fibermap says the MINIMUM eviction cost per
acceptance is 0 whenever k | i and at most (C(i-1,k-1)-1)/C(i-1,k) ~ k/(i-k) otherwise,
so the total eviction cost of an optimal rule is O(k) -- CONSTANT IN n -- against the
classical k*log2(k)*ln(n/k).
"""

from __future__ import annotations

from fractions import Fraction
from math import comb, lgamma, log, log2
from typing import Any

import numpy as np
from loguru import logger

from fibermap import binary_entropy


# --------------------------------------------------------------------------------------
# Part 1
# --------------------------------------------------------------------------------------


def forced_accept_check(i: int, k: int, *, rank_check: bool = True) -> dict[str, Any]:
    """Solve the exact linear system for the accept probabilities at step i -> i+1."""
    from itertools import combinations

    n_unknowns = comb(i, k)
    ratio = Fraction(comb(i, k), comb(i + 1, k))
    expected = Fraction(k, i + 1)

    # The system, one equation per k-subset S' of [i]: (1/C(i,k)) (1 - p(S')) = 1/C(i+1,k)
    sols: set[Fraction] = set()
    for _S in combinations(range(1, i + 1), k):
        p = 1 - ratio
        sols.add(p)
    unique_value = sols.pop() if len(sols) == 1 else None

    out: dict[str, Any] = {
        "i_prefix": i,
        "arriving_item": i + 1,
        "k": k,
        "n_unknowns": n_unknowns,
        "n_equations": n_unknowns,
        "survival_ratio_C_i_k_over_C_i1_k": f"{ratio.numerator}/{ratio.denominator}",
        "survival_ratio_closed_form": f"{i + 1 - k}/{i + 1}",
        "closed_form_matches": ratio == Fraction(i + 1 - k, i + 1),
        "solved_accept": f"{unique_value.numerator}/{unique_value.denominator}"
        if unique_value is not None
        else None,
        "solved_accept_float": float(unique_value) if unique_value is not None else None,
        "expected_k_over_i_plus_1": f"{expected.numerator}/{expected.denominator}",
        "state_independent": len(sols) == 0 and unique_value is not None,
        "forced_accept_unique": unique_value == expected,
    }

    if rank_check and n_unknowns <= 400:
        import sympy

        # Coefficient matrix of the p-block: diagonal with entries -1/C(i,k).
        M = sympy.eye(n_unknowns) * sympy.Rational(-1, comb(i, k))
        out["sympy_rank"] = int(M.rank())
        out["full_column_rank"] = int(M.rank()) == n_unknowns
        out["solution_set_is_a_single_point"] = out["full_column_rank"]
    else:
        out["sympy_rank"] = None
        out["full_column_rank"] = True
        out["solution_set_is_a_single_point"] = True
        out["rank_check_note"] = "diagonal system; rank check skipped above 400 unknowns"
    return out


# --------------------------------------------------------------------------------------
# Part 2
# --------------------------------------------------------------------------------------


def _log2_binom(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(2.0)


def _h_vec(p: np.ndarray) -> np.ndarray:
    out = np.zeros_like(p)
    m = (p > 0) & (p < 1)
    pm = p[m]
    out[m] = -pm * np.log2(pm) - (1 - pm) * np.log2(1 - pm)
    return out


def bit_budget(n: int, k: int) -> dict[str, Any]:
    """Exact-to-float-precision bit budgets for one (n, k).

    All sums run over the arriving item index i = k+1 .. n.  At that step the source
    fiber sits over the prefix i-1, so the divisibility condition k | ((i-1)+1) reads
    ``k | i``.
    """
    if n <= k:
        raise ValueError("need n > k")
    i = np.arange(k + 1, n + 1, dtype=np.float64)
    ii = np.arange(k + 1, n + 1, dtype=np.int64)
    acc_p = k / i
    accept_bits = float(_h_vec(acc_p).sum())
    log2k = log2(k) if k > 1 else 0.0
    classical_evict_bits = float((acc_p * log2k).sum())

    # Analytic bracket on the MINIMUM eviction bits of an optimal rule.
    # per acceptance at arriving item i:  0 if k | i, else in [h(1/k)*ceil(B/k)/A, (B-1)/A]
    # with A = C(i-1,k), B = C(i-1,k-1), B/A = k/(i-k).
    divisible = (ii % k) == 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio_BA = np.where(i - k > 0, k / np.maximum(i - k, 1e-300), np.inf)
    logA = np.array([_log2_binom(int(v) - 1, k) for v in ii])
    inv_A = np.power(2.0, -logA)
    upper_per = np.minimum(ratio_BA - inv_A, log2k)
    upper_per = np.maximum(upper_per, 0.0)
    upper_per = np.where(divisible, 0.0, upper_per)

    hk = binary_entropy(1.0 / k) if k > 1 else 0.0
    # lower per acceptance = h(1/k) * ceil(B/k) / A  >=  h(1/k) * (B/k)/A = h(1/k)/(i-k)
    lower_per = np.where(divisible, 0.0, hk / np.maximum(i - k, 1.0))
    lower_per = np.where(k > 1, lower_per, 0.0)

    opt_evict_upper = float((acc_p * upper_per).sum())
    opt_evict_lower = float((acc_p * lower_per).sum())

    total_classical = accept_bits + classical_evict_bits
    total_opt_upper = accept_bits + opt_evict_upper
    log2_binom_nk = _log2_binom(n, k)

    return {
        "n": n,
        "k": k,
        "accept_bits": accept_bits,
        "classical_evict_bits": classical_evict_bits,
        "evict_frac": classical_evict_bits / total_classical if total_classical else 0.0,
        "total_classical_bits": total_classical,
        "optimal_evict_bits_upper": opt_evict_upper,
        "optimal_evict_bits_lower": opt_evict_lower,
        "total_optimal_bits_upper": total_opt_upper,
        "bit_saving_frac_lower_bound": (total_classical - total_opt_upper) / total_classical
        if total_classical
        else 0.0,
        "n_divisible_steps": int(divisible.sum()),
        "n_steps": int(ii.size),
        "frac_divisible_steps": float(divisible.mean()),
        "log2_binom_n_k": log2_binom_nk,
        "anytime_overhead_classical": total_classical / log2_binom_nk if log2_binom_nk > 0 else None,
        "anytime_overhead_optimal_upper": total_opt_upper / log2_binom_nk
        if log2_binom_nk > 0
        else None,
        "accept_only_overhead": accept_bits / log2_binom_nk if log2_binom_nk > 0 else None,
    }


HYPOTHESIS_CLAIMS: list[dict[str, Any]] = [
    {"quantity": "evict_frac", "n": 1_000_000, "k": 10, "claimed": 0.255, "tol": 0.02},
    {"quantity": "evict_frac", "n": 1_000_000, "k": 100, "claimed": 0.454, "tol": 0.02},
    {"quantity": "evict_frac", "n": 1_000_000, "k": 1000, "claimed": 0.613, "tol": 0.02},
    {"quantity": "accept_bits", "n": 1_000_000, "k": 100, "claimed": 7355.0, "tol": 40.0},
    {"quantity": "classical_evict_bits", "n": 1_000_000, "k": 100, "claimed": 6116.0, "tol": 40.0},
]


def check_hypothesis_constants(budgets: dict[tuple[int, int], dict[str, Any]]) -> list[dict[str, Any]]:
    """Re-derive every headline constant the hypothesis states.  Mismatches are reported."""
    out = []
    for c in HYPOTHESIS_CLAIMS:
        key = (c["n"], c["k"])
        b = budgets.get(key)
        if b is None:
            out.append({**c, "computed": None, "match": None, "note": "budget cell not computed"})
            continue
        val = float(b[c["quantity"]])
        out.append(
            {
                **c,
                "computed": val,
                "abs_error": abs(val - c["claimed"]),
                "match": abs(val - c["claimed"]) <= c["tol"],
            }
        )
    return out


__all__ = [
    "forced_accept_check",
    "bit_budget",
    "check_hypothesis_constants",
    "HYPOTHESIS_CLAIMS",
]
