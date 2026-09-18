#!/usr/bin/env python3
"""Gates G1..G8.  Nothing scales up until the gate below it is green.

Run from the workspace root:  .venv/bin/python -m pytest tests -q
"""

from __future__ import annotations

import sys
from fractions import Fraction
from itertools import combinations
from math import comb, log2
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import candidates  # noqa: E402
import fibermap  # noqa: E402
import forced  # noqa: E402
import montecarlo as mc  # noqa: E402
from exact_dp import run_exact_dp  # noqa: E402


# --------------------------------------------------------------------------------------
# G1 -- DP self-consistency: the baseline must be EXACTLY uniform at every prefix
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 2, 3])
@pytest.mark.parametrize("n", [6, 8, 10])
def test_g1_classical_r_is_exactly_uniform(n: int, k: int) -> None:
    rep = run_exact_dp(candidates.build("S0", n=n, k=k), n, k)
    assert rep.i_first_fail is None
    for r in rep.rows:
        assert r.m1_abs == 0, f"S0 deviates at i={r.i}: {r.m1_abs}"
        assert r.tv == 0
        assert r.support == r.support_total == comb(r.i, k)
        assert r.incl_maxdev == 0


def test_g1_mass_is_exactly_one() -> None:
    # run_exact_dp raises if mass or transition mass ever differs from 1; exercise the
    # table-driven sampler too, whose transition probabilities come from an LP.
    for key in ("S0", "S1", "S1f", "S1n", "S5-CIRC"):
        run_exact_dp(candidates.build(key, n=9, k=3), 9, 3)


# --------------------------------------------------------------------------------------
# G2 -- every negative control must light up
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", candidates.CONTROL_KEYS)
def test_g2_controls_fail_early(key: str) -> None:
    rep = run_exact_dp(candidates.build(key, n=8, k=2), 8, 2)
    assert rep.i_first_fail is not None, f"control {key} scored exactly uniform -- investigate"
    assert rep.i_first_fail <= 8
    assert rep.m1_rel_at_failure > 0


def test_g2_fifo_first_failure_prefix_is_recorded() -> None:
    """The hypothesis reports FIFO first failing at n=4, k=2.  We record what is measured.

    The measured value is 3: at prefix i=2 the reservoir is {1,2} with 1 the oldest, so
    on acceptance FIFO always produces {2,3} and never {1,3}.  The correction is carried
    into method_out.json under ``numbers_that_moved``.
    """
    rep = run_exact_dp(candidates.build("C-FIFO", n=8, k=2), 8, 2)
    assert rep.i_first_fail == 3
    assert rep.m1_rel_at_failure == 1


# --------------------------------------------------------------------------------------
# G3 -- hand-checked fiber map
# --------------------------------------------------------------------------------------


def test_g3_hand_checked_fiber_map_i3_k2() -> None:
    fib = fibermap.build_fiber(3, 2)
    m = fibermap.summodk_map(fib)
    got = {fib.sources[a]: fib.targets[m[a]] for a in range(fib.n_sources)}
    assert got == {(1, 2): (2,), (1, 3): (1,), (2, 3): (3,)}
    assert len(set(got.values())) == 3  # bijection onto the singletons


# --------------------------------------------------------------------------------------
# G4 -- S1n reproduces the reported failure magnitudes
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("k,claimed", [(2, 1.0), (4, 9.7)])
def test_g4_summodk_norepair_failure_magnitude(k: int, claimed: float) -> None:
    rep = run_exact_dp(candidates.build("S1n", n=12, k=k), 12, k)
    assert rep.i_first_fail is not None
    measured = float(rep.m1_rel_at_failure)
    # We do not assert agreement with the hypothesis's number -- we record it.  The test
    # only guarantees the magnitude is finite, positive and reproducible.
    assert measured > 0
    assert np.isfinite(measured)
    print(f"G4 k={k}: measured m1_rel at first failure = {measured:.4f} (hypothesis: {claimed})")


# --------------------------------------------------------------------------------------
# G5 -- transportation arithmetic and the pre-registered entropy bracket
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("k", [2, 3, 4, 5])
def test_g5_transport_arithmetic(k: int) -> None:
    for i in range(k, 16):
        fib = fibermap.build_fiber(i, k)
        assert comb(i, k) * k == comb(i, k - 1) * (i - k + 1)
        assert fib.demand_frac == Fraction(i - k + 1, k)
        assert ((i - k + 1) % k == 0) == ((i + 1) % k == 0)
        assert fib.divisible == ((i + 1) % k == 0)


@pytest.mark.parametrize("i,k", [(4, 2), (5, 2), (6, 3), (7, 3), (9, 4), (10, 4)])
def test_g5_bracket_contains_everything(i: int, k: int) -> None:
    row = fibermap.analyse_step(i, k, save_cert=False)
    lo = row["bracket"]["lower_bits"]
    hi = row["bracket"]["upper_bits"]
    assert row["achieved_entropy_bits"] <= hi + 1e-9
    ex = row["min_entropy_exact"]
    if ex.get("solved"):
        assert lo - 1e-9 <= ex["min_bits"] <= hi + 1e-9
        assert ex["min_bits"] <= row["achieved_entropy_bits"] + 1e-9


# --------------------------------------------------------------------------------------
# G6 -- certificate round-trip, and a checker that actually says no
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("i,k", [(3, 2), (5, 2), (7, 2), (5, 3), (8, 3), (7, 4)])
def test_g6_certificate_round_trip(i: int, k: int) -> None:
    fib = fibermap.build_fiber(i, k)
    if not fib.divisible:
        pytest.skip(f"i={i},k={k} has fractional demand; no deterministic certificate exists")
    cert = fibermap.integral_certificate(fib)
    assert cert["feasible"], f"integral feasibility failed at i={i},k={k} where theory says it holds"
    chk = fibermap.check_certificate(cert["certificate"], i, k)
    assert chk["valid"], chk["errors"]


def test_g6_checker_rejects_a_corrupted_certificate() -> None:
    fib = fibermap.build_fiber(5, 2)
    assert fib.divisible
    cert = list(fibermap.integral_certificate(fib)["certificate"])
    assert fibermap.check_certificate(cert, 5, 2)["valid"]

    # (a) point one source at a target that is not its child
    bad_a = list(cert)
    S, _T = bad_a[0]
    other = next(t for t in fib.targets if not set(t).issubset(set(S)))
    bad_a[0] = (S, other)
    assert not fibermap.check_certificate(bad_a, 5, 2)["valid"]

    # (b) duplicate a source / drop another -> load imbalance
    bad_b = list(cert)
    bad_b[1] = (bad_b[0][0], bad_b[1][1])
    assert not fibermap.check_certificate(bad_b, 5, 2)["valid"]

    # (c) drop a row entirely
    assert not fibermap.check_certificate(cert[:-1], 5, 2)["valid"]


def test_g6_infeasible_reported_with_reason_not_silently() -> None:
    fib = fibermap.build_fiber(4, 2)  # (4+1) % 2 == 1 -> fractional demand
    assert not fib.divisible
    cert = fibermap.integral_certificate(fib)
    assert cert["feasible"] is False
    assert "not an integer" in cert["reason"]


# --------------------------------------------------------------------------------------
# G7 -- the repaired / flow-optimal samplers are exactly uniform end to end
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["S1", "S1f"])
@pytest.mark.parametrize("n,k", [(8, 2), (10, 2), (9, 3), (10, 4)])
def test_g7_repaired_sampler_is_exactly_uniform(key: str, n: int, k: int) -> None:
    rep = run_exact_dp(candidates.build(key, n=n, k=k), n, k)
    assert rep.i_first_fail is None, (
        f"{key} n={n} k={k} first fails at prefix {rep.i_first_fail} "
        f"(m1_rel={float(rep.m1_rel_at_failure)})"
    )
    for r in rep.rows:
        assert r.m1_abs == 0
        assert r.support == r.support_total


# --------------------------------------------------------------------------------------
# S5-CIRC: the blind-spot predictions, checked exactly
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("n,k", [(10, 3), (12, 4)])
def test_s5_circ_exact_first_order_but_broken_joint(n: int, k: int) -> None:
    from exact_dp import observable_distribution, pairwise_inclusion

    rep = run_exact_dp(candidates.build("S5-CIRC", n=n, k=k), n, k)
    for r in rep.rows:
        assert r.incl_maxdev == 0, f"S5-CIRC inclusion vector deviates at i={r.i}"
        # one distinct circular window per anchor, except at i == k where all anchors
        # give the whole of [k] (and C(k,k) = 1, so the prefix is still exactly uniform)
        assert r.support == min(r.i, comb(r.i, k))
        assert r.tv == Fraction(1) - Fraction(r.support, comb(r.i, k))

    obs = observable_distribution(candidates.build("S5-CIRC", n=n, k=k), n, k)
    pi = pairwise_inclusion(obs, n)
    correct = Fraction(k * (k - 1), n * (n - 1))
    zeros = sum(1 for x, y in combinations(range(1, n + 1), 2) if pi.get((x, y), 0) == 0)
    assert zeros > 0, "S5-CIRC should have structurally impossible pairs"
    assert max(abs(v - correct) for v in pi.values()) > 0


# --------------------------------------------------------------------------------------
# Forced accept coin
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("i,k", [(3, 2), (5, 2), (6, 3), (8, 3), (9, 4)])
def test_forced_accept_is_unique(i: int, k: int) -> None:
    out = forced.forced_accept_check(i, k)
    assert out["forced_accept_unique"]
    assert out["state_independent"]
    assert out["solution_set_is_a_single_point"]
    assert out["solved_accept"] == f"{Fraction(k, i + 1).numerator}/{Fraction(k, i + 1).denominator}"


def test_bit_budget_reproduces_hypothesis_constants() -> None:
    budgets = {(1_000_000, k): forced.bit_budget(1_000_000, k) for k in (10, 100, 1000)}
    checks = forced.check_hypothesis_constants(budgets)
    for c in checks:
        assert c["match"] is not None
        print(f"  {c['quantity']} n={c['n']} k={c['k']}: computed {c['computed']:.4f} "
              f"vs claimed {c['claimed']} -> match={c['match']}")
    assert all(c["match"] for c in checks), [c for c in checks if not c["match"]]


# --------------------------------------------------------------------------------------
# G8 -- Monte Carlo harness validation, three ways
# --------------------------------------------------------------------------------------


def test_g8a_skip_generation_matches_naive_bernoulli() -> None:
    rng = np.random.Generator(np.random.PCG64(20260918))
    n, k, T = 60, 5, 30_000
    a = mc.simulate_algo_l(n, k, T, mc.rule_uniform, rng)
    b = mc.simulate_naive_bernoulli(n, k, T, mc.rule_uniform, rng)
    sa = mc.first_order_stats(a, n, k)
    sb = mc.first_order_stats(b, n, k)
    sigma = np.sqrt((k / n) * (1 - k / n) / T)
    assert sa["max_dev"] < 6 * sigma
    assert sb["max_dev"] < 6 * sigma


def test_g8b_null_band_brackets_the_baseline() -> None:
    """S0 IS the null, so the exact Monte-Carlo p-value of its statistic must not be small.

    The band is built by an INDEPENDENT exact uniform-k-subset sampler (rejection on
    distinct draws), never by the code under test, and the decision uses the exact
    Monte-Carlo p-value rather than a percentile estimated from a handful of replicates
    (which at R<=20 is only the 1st-2nd order statistic and would flag the baseline).
    """
    import verdicts as V

    rng = np.random.Generator(np.random.PCG64(7))
    n, k, T, R = 200, 10, 50_000, 120
    band = mc.null_band(n, k, T, R, rng)
    res = mc.simulate_algo_l(n, k, T, mc.rule_uniform, rng)
    dev = mc.first_order_stats(res, n, k)["max_dev"]
    flagged, p = V.mc_flag(dev, band["first_order"]["values"], 0.01)
    assert not flagged, (dev, p, band["first_order"]["p95"])
    assert dev >= band["first_order"]["p50"] * 0.4


def test_g8c_blindspot_money_shot_at_screen_scale() -> None:
    rng = np.random.Generator(np.random.PCG64(11))
    import verdicts as V

    n, k, T, R = 200, 10, 50_000, 120
    band = mc.null_band(n, k, T, R, rng)
    res = mc.simulate_circwindow(n, k, T, rng)
    fo = mc.first_order_stats(res, n, k)
    pr = mc.pair_stats(res, n, k)
    f1, p1 = V.mc_flag(fo["max_dev"], band["first_order"]["values"], 0.01)
    f2, p2 = V.mc_flag(pr["max_abs_z"], band["pair_z"]["values"], 0.01)
    assert not f1, f"first-order test should be BLIND (p={p1})"
    assert f2, f"pairwise test should be decisive (p={p2})"
    assert pr["max_abs_z"] > 10 * band["pair_z"]["p95"]


def test_g8d_determinism() -> None:
    a = mc.simulate_algo_l(50, 4, 5_000, mc.rule_uniform, np.random.Generator(np.random.PCG64(3)))
    b = mc.simulate_algo_l(50, 4, 5_000, mc.rule_uniform, np.random.Generator(np.random.PCG64(3)))
    assert np.array_equal(a, b)


def test_mc_agrees_with_exact_dp_at_small_n() -> None:
    """Cross-validate the Monte Carlo harness against the exact DP on the same sampler."""
    from exact_dp import observable_distribution

    n, k, T = 12, 4, 400_000
    obs = observable_distribution(candidates.build("S1n", n=n, k=k), n, k)
    exact_incl = [Fraction(0)] * (n + 1)
    for S, p in obs.items():
        for x in S:
            exact_incl[x] += p
    rng = np.random.Generator(np.random.PCG64(1234))
    res = mc.simulate_algo_l(n, k, T, mc.rule_summodk, rng)
    counts = np.bincount(res.ravel(), minlength=n + 1)[1:]
    freq = counts / T
    sigma = np.sqrt(0.25 / T)  # generous: p(1-p) <= 1/4
    for x in range(1, n + 1):
        assert abs(freq[x - 1] - float(exact_incl[x])) < 8 * sigma, (
            x, freq[x - 1], float(exact_incl[x])
        )
