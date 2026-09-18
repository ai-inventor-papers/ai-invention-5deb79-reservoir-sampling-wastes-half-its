"""GATE 1 -- bit source correctness.  Must pass before any sweep."""

from __future__ import annotations

import math
from collections import Counter

import pytest
from scipy import stats

from bits import REGIMES, make_source


@pytest.mark.parametrize("a,b", [(1, 2), (1, 3), (10, 1000), (999, 1000), (1, 10**6)])
@pytest.mark.parametrize("regime", REGIMES)
def test_bernoulli_frequency(regime: str, a: int, b: int) -> None:
    n = 400_000 if regime.startswith("naive") else 2_000_000
    src = make_source(regime, "PCG64", 1234 + a + b)
    hits = sum(src.bernoulli(a, b) for _ in range(n))
    p = a / b
    sd = math.sqrt(max(p * (1 - p), 1e-12) / n)
    assert abs(hits / n - p) <= 4 * sd + 1e-9, f"{regime} bern({a}/{b}) off"


@pytest.mark.parametrize("k", [2, 3, 5, 10, 100, 1000])
@pytest.mark.parametrize("regime", REGIMES)
def test_uniform_int_chisq(regime: str, k: int) -> None:
    n = max(20 * k, 200_000)
    src = make_source(regime, "PCG64", 77 + k)
    c = Counter(src.uniform_int(k) for _ in range(n))
    assert set(c) <= set(range(k))
    obs = [c.get(j, 0) for j in range(k)]
    assert stats.chisquare(obs).pvalue > 0.001, f"{regime} uniform_int({k}) chi2 fail"


@pytest.mark.parametrize("a,b", [(1, 2), (1, 1000), (999, 1000), (1, 10**6)])
def test_ky_bernoulli_costs_exactly_two_flips(a: int, b: int) -> None:
    """The load-bearing fact behind candidate 4's swamping claim."""
    n = 1_000_000
    src = make_source("KY", "PCG64", 5)
    for _ in range(n):
        src.bernoulli(a, b)
    assert abs(src.flips / n - 2.0) <= 0.01, f"KY bern({a}/{b}) = {src.flips / n}"


@pytest.mark.parametrize("k", [2, 3, 5, 10, 100, 1000])
def test_ky_uniform_int_cost_band(k: int) -> None:
    n = 200_000
    src = make_source("KY", "PCG64", 9 + k)
    for _ in range(n):
        src.uniform_int(k)
    cost = src.flips / n
    assert math.log2(k) <= cost <= math.log2(k) + 2.0, f"FDR({k}) cost {cost}"


def test_recycling_reaches_the_entropy_floor() -> None:
    """R-recycle must reach log2(3) per uniform_int(3) -- this is where MAIN lives."""
    n = 1_000_000
    src = make_source("recycle", "PCG64", 13)
    for _ in range(n):
        src.uniform_int(3)
    cost = src.net_total() / n
    assert math.log2(3) <= cost <= math.log2(3) + 0.05, f"recycle uniform_int(3) = {cost}"


@pytest.mark.parametrize("a,b", [(1, 2), (1, 3), (10, 1000), (1, 100)])
def test_recycling_bernoulli_tracks_binary_entropy(a: int, b: int) -> None:
    """Net cost per Bernoulli must equal h(p) to within sampling noise.

    The tolerance is two-sided because the realised refill count is itself
    random: the rare branch costs log2(1/p) bits and occurs with probability p,
    so at p * n = O(1) the estimator has O(1) relative variance.  Cases with
    p * n < 100 (e.g. p = 1e-6 at n = 1e6) are deliberately NOT asserted here --
    they are under-powered, not wrong -- and the exact per-branch identity is
    what test_recycling_branch_identity checks instead.
    """
    n = 1_000_000
    p = a / b
    h = -p * math.log2(p) - (1 - p) * math.log2(1 - p)
    assert p * n >= 100, "this test needs the rare branch to be well sampled"
    src = make_source("recycle", "PCG64", 31)
    for _ in range(n):
        src.bernoulli(a, b)
    cost = src.net_total() / n
    assert 0.97 * h <= cost <= 1.03 * h + 0.002, f"recycle bern({a}/{b}) = {cost} vs h = {h}"


def test_recycling_branch_identity() -> None:
    """Exact check of the recycling identity, free of rare-branch variance.

    E[log2 M_new - log2 M] = p*log2(p) + (1-p)*log2(1-p) = -h(p), so the net
    cost per call is h(p) EXACTLY.  Measured here by conditioning on the branch:
    the mean net cost of the accept branch and of the reject branch are compared
    against log2(b/a) and log2(b/(b-a)) separately, so a single rare event
    cannot move the verdict.
    """
    a, b = 1, 64
    src = make_source("recycle", "PCG64", 101)
    cost_hit, cost_miss, n_hit, n_miss = 0.0, 0.0, 0, 0
    for _ in range(200_000):
        before = src.s.flips - src._buffer_bits()
        out = src.bernoulli(a, b)
        delta = (src.s.flips - src._buffer_bits()) - before
        if out:
            cost_hit += delta
            n_hit += 1
        else:
            cost_miss += delta
            n_miss += 1
    assert n_hit > 1000 and n_miss > 1000
    assert abs(cost_hit / n_hit - math.log2(b / a)) < 0.05
    assert abs(cost_miss / n_miss - math.log2(b / (b - a))) < 0.005


@pytest.mark.parametrize("regime", REGIMES)
def test_determinism(regime: str) -> None:
    def run() -> tuple[int, list[int]]:
        src = make_source(regime, "PCG64", 4242)
        out = [src.bernoulli(3, 7) for _ in range(5000)]
        out += [src.uniform_int(11) for _ in range(5000)]
        return src.flips, out

    assert run() == run()


@pytest.mark.parametrize("regime", REGIMES)
def test_tag_attribution_sums(regime: str) -> None:
    src = make_source(regime, "PCG64", 8)
    with src.op("accept"):
        for _ in range(500):
            src.bernoulli(1, 5)
    with src.op("evict"):
        for _ in range(500):
            src.uniform_int(7)
    tot = src.s.tag_totals()
    assert sum(tot.values()) == src.flips
    assert abs(sum(src.net_totals().values()) - src.net_total()) < 1e-9
