#!/usr/bin/env python3
"""Unit tests for l1tester.py -- the identity 2C == sum N_x(N_x-1) and the Z algebra."""
from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np

import l1tester as L


def test_combinadic_rank_is_a_bijection() -> None:
    for n, k in [(6, 2), (8, 3), (14, 4), (12, 2)]:
        subs = np.array(sorted(combinations(range(1, n + 1), k)), dtype=np.int64)
        r = L.combinadic_rank(subs, n, k)
        assert r.min() == 0 and r.max() == comb(n, k) - 1, (n, k, r.min(), r.max())
        assert len(set(r.tolist())) == comb(n, k)
        assert np.all(np.diff(r) > 0), "lexicographic order must be preserved"


def test_collision_identity_brute_force() -> None:
    """2C == sum_x N_x(N_x-1), verified by brute force on a toy domain."""
    rng = np.random.Generator(np.random.PCG64(7))
    for D, T in [(10, 40), (23, 200), (101, 500)]:
        atoms = rng.integers(0, D, size=T)
        c, _ = L.collision_count_from_keys(atoms)
        # brute force over all unordered pairs
        brute = sum(1 for a, b in combinations(range(T), 2) if atoms[a] == atoms[b])
        assert c == brute, (D, T, c, brute)
        counts = np.bincount(atoms, minlength=D)
        assert 2 * c == int(np.sum(counts * (counts - 1))), (D, T)
        assert L.sum_n_times_n_minus_1(atoms) == 2 * c


def test_Z_uniform_collapse_matches_general_form() -> None:
    """Z = (2D/T) C - T must equal the general ADK/DKN statistic with q uniform."""
    rng = np.random.Generator(np.random.PCG64(11))
    for D, T in [(10, 60), (37, 400), (101, 1000), (1001, 5000)]:
        atoms = rng.integers(0, D, size=T)
        counts = np.bincount(atoms, minlength=D).astype(float)
        q = np.full(D, 1.0 / D)
        z_gen = L.z_statistic_general(counts, q, T)
        c, _ = L.collision_count_from_keys(atoms)
        z_fast = L.z_statistic_uniform(c, T, D)
        assert abs(z_gen - z_fast) < 1e-6 * max(1.0, abs(z_gen)), (D, T, z_gen, z_fast)


def test_null_moments_match_simulation() -> None:
    D, T, R = 401, 3000, 4000
    rng = np.random.Generator(np.random.PCG64(3))
    zs = []
    for _ in range(R):
        atoms = rng.integers(0, D, size=T)
        c, _ = L.collision_count_from_keys(atoms)
        zs.append(L.z_statistic_uniform(c, T, D))
    zs = np.asarray(zs)
    mom = L.null_moments_uniform(T, D)
    # E[Z] = -1 EXACTLY; 4000 replicates gives an se of sd/sqrt(R)
    se = mom["SD_Z"] / np.sqrt(R)
    assert abs(zs.mean() - mom["E_Z"]) < 4 * se, (zs.mean(), mom["E_Z"], se)
    assert abs(zs.std(ddof=1) / mom["SD_Z"] - 1.0) < 0.05, (zs.std(ddof=1), mom["SD_Z"])


def test_planted_alternative_has_exact_L1_eps() -> None:
    for D in [100, 1000, 10000]:
        for eps in [0.02, 0.05, 0.1, 0.2, 0.5]:
            half = D // 2
            p = np.concatenate([np.full(half, (1 + eps) / D), np.full(half, (1 - eps) / D)])
            assert abs(p.sum() - 1.0) < 1e-12
            assert abs(np.abs(p - 1.0 / D).sum() - eps) < 1e-12


def test_bytes_key_agrees_with_rank_and_digest() -> None:
    rng = np.random.Generator(np.random.PCG64(5))
    n, k, T = 14, 4, 20000
    rows = np.sort(np.array([rng.choice(n, size=k, replace=False) + 1 for _ in range(T)]), axis=1)
    r = L.combinadic_rank(rows, n, k)
    c_rank, d_rank = L.collision_count_from_keys(r)
    c_bytes, d_bytes = L.collision_count_from_keys(L.subset_bytes_key(rows))
    assert (c_rank, d_rank) == (c_bytes, d_bytes)
    dig = L.subset_digest_key(rows[:4000])
    c_dig, _ = L.collision_count_from_keys(dig.view([("a", np.uint64), ("b", np.uint64)]).ravel())
    c_b2, _ = L.collision_count_from_keys(L.subset_bytes_key(rows[:4000]))
    assert c_dig == c_b2


def test_poisson_tail() -> None:
    from scipy.stats import poisson

    for lam, c in [(0.5, 3), (2.0, 5), (1e-6, 1), (1e-12, 2)]:
        got = L.poisson_upper_tail(c, lam)
        want = float(poisson.sf(c - 1, lam))
        assert abs(got - want) <= 1e-9 * max(want, 1e-300) + 1e-300, (lam, c, got, want)
        lg = L.log10_poisson_upper_tail(c, lam)
        assert abs(lg - np.log10(want)) < 1e-6, (lam, c, lg, np.log10(want))


if __name__ == "__main__":
    fails = []
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                fails.append((name, exc))
                print(f"FAIL {name}: {exc}")
    print(f"\n{len(fails)} failure(s)")
    raise SystemExit(1 if fails else 0)
