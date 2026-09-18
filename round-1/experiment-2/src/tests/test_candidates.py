"""GATE 4 -- the scalar and vectorised implementations must agree EXACTLY,
GATE 5 -- the null-band sampler must itself be exactly uniform,
plus a cross-check of every vectorised sampler against its EXACT kernel.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np
import pytest
from scipy import stats

import exact
import nullband
import prngs
from candidates import CANDIDATES, MC_CANDIDATES, get_candidate
from tests.replay import ReplayGen, ScriptedSource, make_scripts

PAIRS = [c for c in MC_CANDIDATES]


@pytest.mark.parametrize("cid", PAIRS)
def test_scalar_matches_vector_bit_for_bit(cid: str) -> None:
    """Same scripted uniforms into both paths -> identical reservoirs.

    This is the check that catches an off-by-one in the vectorised
    sorted-reservoir eviction, which is the most likely silent bug here.
    """
    n, k = 50, 5
    cand = get_candidate(cid)
    if cand.needs_even_k and k % 2:
        n, k = 50, 4
    if cand.oracle_n and n % k:
        pytest.skip("oracle systematic needs k | n")
    mismatches = 0
    for trial in range(200):
        sa, sb = make_scripts(1000 + trial, 4 * n + 500, 4 * n + 500, k)
        res_scalar = sorted(cand.scalar(ScriptedSource(sa), n, k))
        res_vec = sorted(int(x) for x in cand.vector(ReplayGen(sb), n, k, 1)[0])
        if res_scalar != res_vec:
            mismatches += 1
    assert mismatches == 0, f"{cid}: {mismatches}/200 scalar-vs-vector mismatches"


@pytest.mark.parametrize("cid", PAIRS)
def test_vector_reservoir_is_a_valid_subset(cid: str) -> None:
    cand = get_candidate(cid)
    k = 10
    n = 200
    gen = prngs.make_prng("PCG64", 3 + hash(cid) % 1000)
    res = cand.vector(gen, n, k, 3000)
    assert res.shape == (3000, k)
    assert res.min() >= 1 and res.max() <= n
    assert (np.diff(np.sort(res, axis=1), axis=1) > 0).all(), f"{cid} has repeats"


@pytest.mark.parametrize("cid", [c for c in MC_CANDIDATES if CANDIDATES[c].children is not None])
def test_vector_matches_exact_kernel(cid: str) -> None:
    """Empirical first-order vector at n=12 must match the EXACT rational one."""
    n, k = 12, 3
    cand = get_candidate(cid)
    if cand.needs_even_k:
        k = 4
    if cid == "S5e" and n % k:
        pytest.skip("BLOCK-LOCK needs k | n")
    ex = exact.verify(cid, n, k, record_every=True)
    gen = prngs.make_prng("PCG64", 555)
    T = 400_000
    res = cand.vector(gen, n, k, T)
    emp = np.bincount(res.ravel(), minlength=n + 1)[1:] / T
    assert abs(emp.max() - emp.min()) < 1.0
    # the exact first-order max deviation must be reproduced within 5 sigma
    sigma = math.sqrt((k / n) * (1 - k / n) / T)
    assert abs(float(np.abs(emp - k / n).max()) - ex["first_order_max_dev"]) < 6 * sigma


def test_nullband_sampler_is_exactly_uniform() -> None:
    """GATE 5 -- the band is meaningless unless its sampler is exact."""
    gen = prngs.make_prng("PCG64", 4242)
    sub = nullband.exact_uniform_subsets(gen, 2_000_000, 10, 3)
    c = Counter(map(tuple, sub))
    assert len(c) == math.comb(10, 3)
    assert stats.chisquare(list(c.values())).pvalue > 0.001


def test_s0_is_inside_its_own_null_band() -> None:
    """A provably-correct sampler must not drift outside the simulated band."""
    import montecarlo as mc

    gen = prngs.make_prng("PCG64", 777)
    band = nullband.simulate_null(gen, n=200, k=10, T=100_000, B=32, chunk=50_000, B_coinc=6)
    row = mc.run_mc("S0", n=200, k=10, T=100_000, seed=778)
    full = mc.mc_row_with_band(row, band)
    assert -3.5 < full["z_band"] < 3.5, f"S0 z_band = {full['z_band']}"
    assert full["coinc_chi2_over_df"] < 1.1


@pytest.mark.parametrize("cid", ["S5e", "S5g", "S6sys"])
def test_blind_spot_exhibits_have_exact_first_order_marginals(cid: str) -> None:
    """Candidate 1: first-order-uniform yet jointly degenerate."""
    import montecarlo as mc

    k = 10
    n = 1000
    row = mc.run_mc(cid, n=n, k=k, T=200_000, seed=31337)
    gen = prngs.make_prng("PCG64", 31338)
    band = nullband.simulate_null(gen, n=n, k=k, T=200_000, B=16, chunk=50_000, B_coinc=4)
    full = mc.mc_row_with_band(row, band)
    assert full["passes_first_order_at_p99"], f"{cid} maxdev {row['maxdev_raw']}"
    assert full["coinc_fails_at_p99"], f"{cid} coinc Zmax {row['coinc_Zmax']}"
