#!/usr/bin/env python3
"""L1 identity / uniformity testing statistics for k-subset samplers.

Implements the two published testers used as the THIRD detector column:

  (1) The chi-square-type identity-testing statistic of Acharya-Daskalakis-Kamath
      (NIPS 2015, arXiv:1507.05952) and Diakonikolas-Kane-Nikishkin (SODA 2015,
      arXiv:1410.2266):

          Z = sum_x [ (N_x - T q_x)^2 - N_x ] / (T q_x)

      with T samples, observed counts N_x and reference distribution q.

  (2) The collision tester (Goldreich-Ron; optimality for uniformity by
      Diakonikolas-Gouleakis-Peebles-Price, arXiv:1611.03579), whose statistic is the
      number of colliding sample pairs C.

CRITICAL ALGEBRAIC FACT used everywhere below.  For the UNIFORM reference q_x = 1/D,

      sum_x N_x (N_x - 1) = 2 C      (C = number of colliding unordered sample pairs)

  and therefore

      Z = (D/T) * sum_x N_x(N_x-1) - T = (2 D / T) * C - T.

  Derivation (every term kept):
      Z = (D/T) sum_x [ (N_x - T/D)^2 - N_x ]
        = (D/T) [ sum_x N_x^2 - (2T/D) sum_x N_x + D (T/D)^2 - sum_x N_x ]
        = (D/T) [ sum_x N_x(N_x-1) - 2 T^2/D + T^2/D ]
        = (D/T) sum_x N_x(N_x-1) - T.

  Consequence: the tester never needs a length-D count vector, only the multiset of
  observed samples.  That is what makes D = C(1000,10) ~ 2.63e23 runnable at all.

Exact null moments under iid sampling from the uniform q (multinomial, NOT Poissonised):
      E[C]   = binom(T,2) / D
      Var[C] = binom(T,2) * (1/D) * (1 - 1/D)
        (pairs sharing one index have exactly zero covariance: P(a=b=c) = 1/D^2 = p^2)
      E[Z]   = -1                       exactly, for every T and D
      Var[Z] = (2D/T)^2 Var[C] = 2 D (T-1)/T * (1 - 1/D)

When T^2/D is small (the sublinear regime of the n=1000,k=10 panel) C is essentially
Poisson(lambda = binom(T,2)/D) and the normal approximation to Z is worthless, so the
exact Poisson upper-tail p-value is carried alongside the asymptotic threshold.
"""

from __future__ import annotations

import hashlib
from math import comb, exp, isfinite, lgamma, log, log10, sqrt

import numpy as np

# --------------------------------------------------------------------------------------
# Canonical keys for k-subsets
# --------------------------------------------------------------------------------------


def combinadic_rank(rows: np.ndarray, n: int, k: int) -> np.ndarray:
    """Exact combinatorial rank in [0, C(n,k)) of sorted 1-based k-subsets of [n].

    rows: (T, k) int array, each row sorted ascending with values in 1..n.
    Uses the combinadic number system: rank = sum_{j=0}^{k-1} C(n - c_j, k - j) ... in the
    colex-free lexicographic form
        rank = sum_{j=0}^{k-1} sum_{v=prev+1}^{c_j - 1} C(n - v, k - 1 - j)
    Implemented with an exact Python-int table so it is correct for any n,k where C(n,k)
    fits in a Python int; returned as an object array of ints when it overflows int64.
    """
    if rows.ndim != 2 or rows.shape[1] != k:
        raise ValueError(f"rows must be (T,{k}), got {rows.shape}")
    total = comb(n, k)
    T = rows.shape[0]
    use_int64 = total < 2**62
    tbl = [[comb(a, b) for b in range(k + 1)] for a in range(n + 1)]
    if use_int64:
        out = np.zeros(T, dtype=np.int64)
        prev = np.zeros(T, dtype=np.int64)
        for j in range(k):
            cj = rows[:, j].astype(np.int64)
            # sum_{v=prev+1}^{cj-1} C(n-v, k-1-j) -- vectorise with a cumulative table
            rem = k - 1 - j
            cum = np.zeros(n + 2, dtype=np.int64)  # cum[m] = sum_{v=1}^{m} C(n-v, rem)
            acc = 0
            for v in range(1, n + 1):
                acc += tbl[n - v][rem] if (n - v) >= rem else 0
                cum[v] = acc
            cum[n + 1] = acc
            out += cum[np.clip(cj - 1, 0, n)] - cum[np.clip(prev, 0, n)]
            prev = cj
        return out
    raise ValueError("combinadic_rank: C(n,k) exceeds int64; use subset_bytes_key instead")


def subset_bytes_key(rows: np.ndarray) -> np.ndarray:
    """Exact (collision-free) canonical key: the raw sorted row bytes, viewed as void.

    Unlike a digest this cannot collide, so the collision count it yields is EXACT.
    """
    a = np.ascontiguousarray(rows.astype(np.int32))
    return a.view([("", a.dtype)] * a.shape[1]).ravel()


def subset_digest_key(rows: np.ndarray) -> np.ndarray:
    """128-bit blake2b digest of each sorted row, as two uint64 columns.

    Provided because the plan names it; only used as a cross-check of subset_bytes_key
    because a digest can in principle collide and would then inflate C.
    """
    a = np.ascontiguousarray(rows.astype(np.int32))
    out = np.empty((a.shape[0], 2), dtype=np.uint64)
    for i in range(a.shape[0]):
        d = hashlib.blake2b(a[i].tobytes(), digest_size=16).digest()
        out[i, 0] = int.from_bytes(d[:8], "little")
        out[i, 1] = int.from_bytes(d[8:], "little")
    return out


# --------------------------------------------------------------------------------------
# Collision count and the statistic
# --------------------------------------------------------------------------------------


def collision_count_from_keys(keys: np.ndarray) -> tuple[int, int]:
    """Return (C, n_distinct) where C = sum_x N_x (N_x - 1) / 2, computed in O(T log T).

    Exact: no dict, no hashing, just a sort over the canonical keys.
    """
    _, counts = np.unique(keys, return_counts=True)
    c = counts.astype(np.int64)
    pairs = int(np.sum(c * (c - 1) // 2))
    return pairs, int(c.size)


def sum_n_times_n_minus_1(keys: np.ndarray) -> int:
    """sum_x N_x (N_x - 1) directly, for the brute-force identity unit test."""
    _, counts = np.unique(keys, return_counts=True)
    c = counts.astype(object)
    return int(sum(x * (x - 1) for x in c))


def z_statistic_uniform(collisions: int, T: int, D: int) -> float:
    """Z = (2D/T) * C - T for the uniform reference (exact algebra, see module docstring)."""
    return float(2.0 * D / T) * float(collisions) - float(T)


def z_statistic_general(counts: np.ndarray, q: np.ndarray, T: int) -> float:
    """The general ADK/DKN statistic for an arbitrary reference q (used in unit tests)."""
    tq = T * q
    return float(np.sum(((counts - tq) ** 2 - counts) / tq))


def null_moments_uniform(T: int, D: float) -> dict:
    """Exact multinomial null moments of C and Z under q = uniform on D atoms."""
    npairs = T * (T - 1) / 2.0
    ec = npairs / D
    vc = npairs * (1.0 / D) * (1.0 - 1.0 / D)
    return {
        "E_C": ec,
        "SD_C": sqrt(vc),
        "E_Z": -1.0,
        "SD_Z": sqrt(2.0 * D * (T - 1) / T * (1.0 - 1.0 / D)),
        "lambda_poisson": ec,
    }


def _log_poisson_upper_tail(c_obs: int, lam: float) -> float:
    """ln P(Poisson(lam) >= c_obs), summed in log space so it never underflows."""
    if c_obs <= 0:
        return 0.0
    if lam <= 0.0:
        return float("-inf")
    terms: list[float] = []
    j = c_obs
    while True:
        lp = -lam + j * log(lam) - lgamma(j + 1)
        terms.append(lp)
        j += 1
        # stop once the running terms are negligible relative to the first one
        if j > c_obs + 5 and lp < terms[0] - 80.0:
            break
        if j > c_obs + 100000:
            break
    m = max(terms)
    return m + log(sum(exp(t - m) for t in terms))


def poisson_upper_tail(c_obs: int, lam: float) -> float:
    """P(Poisson(lam) >= c_obs); exact, evaluated in log space."""
    lt = _log_poisson_upper_tail(c_obs, lam)
    if lt == float("-inf"):
        return 0.0
    if lt > 0.0:
        return 1.0
    return float(min(1.0, exp(lt)))


def log10_poisson_upper_tail(c_obs: int, lam: float) -> float:
    """log10 P(Poisson(lam) >= c_obs); the linear value underflows for the blind spots."""
    lt = _log_poisson_upper_tail(c_obs, lam)
    if lt == float("-inf"):
        return float("-inf")
    return float(min(0.0, lt / log(10.0)))


def mc_pvalue(observed: float, null_values) -> float:
    """Exact Monte-Carlo p-value (1 + #{null >= obs}) / (R + 1).

    The SAME calibration convention iteration 1 adopted for the other two detectors.
    """
    nv = list(null_values)
    ge = sum(1 for v in nv if v >= observed)
    return (1.0 + ge) / (len(nv) + 1.0)


def asymptotic_threshold(T: int, D: float, alpha: float = 0.01) -> float:
    """Textbook asymptotic (normal) rejection threshold for Z at level alpha."""
    from scipy.stats import norm

    m = null_moments_uniform(T, D)
    return m["E_Z"] + norm.isf(alpha) * m["SD_Z"]


# --------------------------------------------------------------------------------------
# Planted eps-far alternative (for the power / sample-complexity calibration)
# --------------------------------------------------------------------------------------


def planted_alternative_sample(
    D: int, eps: float, T: int, rng: np.random.Generator, *, half_mask_seed: int = 0
) -> np.ndarray:
    """Sample T iid draws from q_eps: mass (1+eps)/D on a random half, (1-eps)/D on the rest.

    ||q_eps - uniform||_1 = (D/2)(eps/D) + (D/2)(eps/D) = eps, EXACTLY, by construction.
    Returned as int64 atom labels in [0, D).
    """
    if D % 2 != 0:
        raise ValueError("planted_alternative_sample requires an even D for an exact eps")
    half = D // 2
    perm_rng = np.random.Generator(np.random.PCG64(half_mask_seed))
    perm = perm_rng.permutation(D)
    hi = rng.random(T) < (1.0 + eps) / 2.0
    idx = np.where(hi, rng.integers(0, half, size=T), half + rng.integers(0, half, size=T))
    return perm[idx]


def uniform_sample_atoms(D: int, T: int, rng: np.random.Generator) -> np.ndarray:
    return rng.integers(0, D, size=T, dtype=np.int64)


def run_tester(keys: np.ndarray, T: int, D: float) -> dict:
    """One full evaluation of both published testers on a sample of canonical keys."""
    c, ndist = collision_count_from_keys(keys)
    z = z_statistic_uniform(c, T, D)
    mom = null_moments_uniform(T, D)
    return {
        "T": int(T),
        "log10_D": log10(float(D)),
        "collisions_C": int(c),
        "n_distinct": int(ndist),
        "sum_Nx_Nx_minus_1": int(2 * c),
        "Z": float(z),
        "Z_null_mean": mom["E_Z"],
        "Z_null_sd": mom["SD_Z"],
        "Z_normalised": float((z - mom["E_Z"]) / mom["SD_Z"]) if mom["SD_Z"] > 0 else 0.0,
        "C_null_mean": mom["E_C"],
        "C_null_sd": mom["SD_C"],
        "poisson_log10_p": log10_poisson_upper_tail(c, mom["lambda_poisson"]),
    }


__all__ = [
    "combinadic_rank",
    "subset_bytes_key",
    "subset_digest_key",
    "collision_count_from_keys",
    "sum_n_times_n_minus_1",
    "z_statistic_uniform",
    "z_statistic_general",
    "null_moments_uniform",
    "poisson_upper_tail",
    "log10_poisson_upper_tail",
    "mc_pvalue",
    "asymptotic_threshold",
    "planted_alternative_sample",
    "uniform_sample_atoms",
    "run_tester",
]
