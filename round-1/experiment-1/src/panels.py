#!/usr/bin/env python3
"""Pre-registered SCREEN and HELD-OUT panels.

The screen panel is frozen to ``screen_ranking.json`` (with a content hash) BEFORE any
held-out configuration executes; ``run_heldout.py`` refuses to run without it.
"""

from __future__ import annotations

from typing import Any

# --------------------------------------------------------------------------------------
# SCREEN
# --------------------------------------------------------------------------------------

SCREEN: dict[str, Any] = {
    "exact_dp": {
        # set-based candidates and the circular-window sampler (small anchor state)
        "main": {
            "candidates": ["S0", "S1", "S1n", "S1f", "S5-CIRC", "C-OFFBY1", "C-OFFBY1b", "C-NOACCEPT"],
            "cells": [(8, 2), (10, 2), (12, 2), (14, 2), (8, 4), (10, 4), (12, 4), (14, 4)],
            "stop_on_first_fail": False,
        },
        # ordered-slot controls: P(n,k) states, so capped at n<=12, k<=3, stopped at failure
        "ordered": {
            "candidates": ["C-FIFO", "C-RR", "C-SLOT0"],
            "cells": [(10, 2), (12, 2), (10, 3), (12, 3)],
            "stop_on_first_fail": True,
        },
    },
    "fibermap": {
        "cells": (
            [(i, 2) for i in range(2, 14)]
            + [(i, 3) for i in range(3, 14)]
            + [(i, 4) for i in range(4, 14)]
        )
    },
    "matched_verdicts": {
        "candidates": ["S0", "S1", "S1n", "S1f", "S5-CIRC", "C-NOACCEPT"],
        "cells": [(12, 2), (12, 4), (14, 4)],
        "T": 100_000,
        "R": 200,
        "prefix_R": 100,
    },
    "montecarlo": {
        "candidates": ["S0", "S1n", "S5-CIRC", "C-NOACCEPT"],
        "n": 200,
        "k": 10,
        "T": 100_000,
        "R": 200,
    },
    "bit_budgets": {"cells": [(1_000, 2), (1_000, 10), (10_000, 2), (10_000, 10)]},
    "forced_accept": {"cells": [(3, 2), (5, 2), (7, 2), (6, 3), (8, 3), (9, 4), (11, 4)]},
    "seed": 20260918,
}

# --------------------------------------------------------------------------------------
# HELD-OUT  (executed only after screen_ranking.json is frozen and its hash verified)
# --------------------------------------------------------------------------------------

HELDOUT: dict[str, Any] = {
    "exact_dp": {
        "main": {
            "candidates": ["S0", "S1", "S1f", "S1n", "S5-CIRC"],
            "cells": [(15, 3), (16, 3), (17, 3), (18, 3), (15, 5), (16, 5), (17, 5), (18, 5)],
            "stop_on_first_fail": False,
        }
    },
    "fibermap": {"cells": [(i, 3) for i in range(14, 18)] + [(i, 5) for i in range(14, 18)]},
    "montecarlo": {
        # THE LITERAL DELIVERABLE: n=1000, k=10, T=1e6, max deviation with its null band
        "candidates": ["S0", "S1n", "S5-CIRC", "C-NOACCEPT"],
        "n": 1_000,
        "k": 10,
        "T": 1_000_000,
        "R": 100,
    },
    "bit_budgets": {
        "cells": [
            (n, k)
            for n in (100_000, 1_000_000)
            for k in (1, 2, 10, 100, 1000)
            if k < n
        ]
    },
    "forced_accept": {"cells": [(14, 3), (17, 3), (14, 5), (17, 5)]},
    "seed": 20260919,
}

# --------------------------------------------------------------------------------------
# Pre-registered margins (written into screen_ranking.json BEFORE held-out runs)
# --------------------------------------------------------------------------------------

# The plan pre-registered R = 10 (screen) / 20 (held-out) null replicates.  Those were
# raised to 200 / 100 before any verdict was computed, because a p95 estimated from 10-20
# replicates is only the 1st-2nd order statistic: measured at n=200, k=10, T=1e5 the true
# p95 of the max-deviation null is 3.59 sigma while an 8-replicate band put its p99 at
# 3.03 sigma, which would have flagged the provably-uniform baseline.  Decisions use an
# EXACT Monte-Carlo p-value, (1 + #{null >= observed}) / (R + 1) <= 0.01, which needs
# R >= 99 to be attainable at all.  The analytic Gumbel band is reported alongside as an
# independent cross-check.
FLAG_ALPHA = 0.01

MARGINS: dict[str, dict[str, Any]] = {
    "MAIN_S1": {
        "description": (
            "An exactly-uniform anytime reservoir sampler exists whose eviction step "
            "costs far less than the classical log2(k) bits per acceptance."
        ),
        "requirements": [
            "exact DP M1 == 0 at EVERY prefix of EVERY screened cell for S1 and S1f",
            "min eviction entropy / log2(k) <= 0.25 on at least half the screened fiber steps",
            "analytic bit saving at k=10 >= 20% of the classical total",
        ],
    },
    "C1_S5CIRC": {
        "description": (
            "A k-memory sampler exists with exactly-uniform first-order marginals and a "
            "broken joint law -- i.e. the standard first-order uniformity test is blind."
        ),
        "requirements": [
            "exact inclusion vector deviation == 0 at every prefix",
            "exact total variation >= 0.9 on the screened cells",
            "support size O(n), not C(n,k)",
            "the first-order Monte-Carlo test does NOT flag it at T=1e6",
            "the pairwise Monte-Carlo test DOES flag it at p < 1e-6",
        ],
    },
}

DEFERRED: dict[str, str] = {
    "S2_ALGORITHM_L_FLOAT64": (
        "floating-point realisation of Algorithm L and its log1p/expm1 variant is owned by "
        "instrument 2 (numerical-robustness bench); no value is fabricated here"
    ),
    "S3_ARES_BOTTOM_K": (
        "weighted A-Res / bottom-k sampling is owned by instrument 2; only the unweighted "
        "bottom-k construction is used here, and only as an exact null generator"
    ),
    "S4_DDG_R_COUNTED_FLIPS": (
        "measured (counted) fair-coin-flip budgets via a DDG tree, and the flip counts of "
        "deployed reservoir implementations, are owned by instrument 2.  Every bit figure "
        "in this artifact is ANALYTIC (an entropy), never a counted flip"
    ),
}
