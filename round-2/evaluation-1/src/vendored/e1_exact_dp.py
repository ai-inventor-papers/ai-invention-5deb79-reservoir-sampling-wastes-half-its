#!/usr/bin/env python3
"""Exact-rational prefix DP over reservoir-sampler states.

The instrument.  Given any sampler that implements the ``Sampler`` protocol, it
propagates the EXACT distribution over sampler states (Fractions, no floats) from
prefix ``i = k`` up to ``i = n`` and, at every prefix, projects onto the observable
k-subset and measures

  M1  max_S | P(R_i = S) - 1/C(i,k) |     over ALL C(i,k) subsets (not just support)
  M2  total variation, support size, first-order inclusion vector deviation

A value of exactly ``0`` here is a PROOF of anytime uniformity on that prefix, not a
statistical statement.  That is the whole point of using Fractions.
"""

from __future__ import annotations

import gc
from collections import defaultdict
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from math import comb
from typing import Any, Hashable, Iterable, Protocol, Sequence

from loguru import logger

# --------------------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class State:
    """A sampler state.

    ``slots`` is an ORDERED tuple of length k holding the currently retained stream
    indices.  Set-based rules ignore the order; FIFO / round-robin / always-slot-0 do
    not, which is exactly why the DP state is not a frozenset.

    ``aux`` carries any extra bookkeeping (a round-robin pointer, a circular-window
    anchor, ...).  It must be hashable.  Set-based rules use ``()``.
    """

    slots: tuple[int, ...]
    aux: Hashable = ()

    def observable(self) -> frozenset[int]:
        return frozenset(self.slots)


class Sampler(Protocol):
    """Protocol every candidate implements."""

    name: str
    ordered: bool  # True if the state space is ordered slots (much larger)

    def init_states(self, k: int) -> dict[State, Fraction]:
        """Distribution over states after the first k items have arrived."""
        ...

    def transitions(self, state: State, i: int, k: int) -> list[tuple[State, Fraction]]:
        """``i`` is the index of the ARRIVING item (i = k+1 .. n).

        Returns (next_state, probability) pairs whose probabilities sum to exactly 1.
        """
        ...


# --------------------------------------------------------------------------------------
# Report rows
# --------------------------------------------------------------------------------------


def _frac_str(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


@dataclass
class PrefixRow:
    i: int
    m1_abs: Fraction
    m1_rel: Fraction
    tv: Fraction
    support: int
    support_total: int
    incl_maxdev: Fraction
    n_states: int

    def to_json(self) -> dict[str, Any]:
        return {
            "i": self.i,
            "m1_abs": _frac_str(self.m1_abs),
            "m1_abs_float": float(self.m1_abs),
            "m1_rel": _frac_str(self.m1_rel),
            "m1_rel_float": float(self.m1_rel),
            "tv": _frac_str(self.tv),
            "tv_float": float(self.tv),
            "support": self.support,
            "support_total": self.support_total,
            "incl_maxdev": _frac_str(self.incl_maxdev),
            "incl_maxdev_float": float(self.incl_maxdev),
            "n_states": self.n_states,
            "exactly_uniform": self.m1_abs == 0,
        }


@dataclass
class PrefixReport:
    candidate: str
    n: int
    k: int
    rows: list[PrefixRow] = field(default_factory=list)
    i_first_fail: int | None = None
    m1_rel_at_failure: Fraction | None = None
    stopped_early: bool = False
    max_states_seen: int = 0

    def to_json(self, *, include_rows: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "candidate": self.candidate,
            "n": self.n,
            "k": self.k,
            "i_first_fail": self.i_first_fail,
            "m1_rel_at_failure": (
                _frac_str(self.m1_rel_at_failure) if self.m1_rel_at_failure is not None else None
            ),
            "m1_rel_at_failure_float": (
                float(self.m1_rel_at_failure) if self.m1_rel_at_failure is not None else None
            ),
            "exactly_uniform_all_prefixes": self.i_first_fail is None and not self.stopped_early,
            "stopped_early": self.stopped_early,
            "max_states_seen": self.max_states_seen,
            "n_prefixes_checked": len(self.rows),
        }
        if include_rows:
            out["prefixes"] = [r.to_json() for r in self.rows]
        return out


# --------------------------------------------------------------------------------------
# The DP
# --------------------------------------------------------------------------------------


def run_exact_dp(
    sampler: Sampler,
    n: int,
    k: int,
    *,
    stop_on_first_fail: bool = False,
    max_states: int = 4_000_000,
    check_transition_mass: bool = True,
) -> PrefixReport:
    """Propagate the exact state distribution and measure M1/M2 at every prefix."""
    if k < 1 or n < k:
        raise ValueError(f"need 1 <= k <= n, got n={n}, k={k}")

    dist: dict[State, Fraction] = dict(sampler.init_states(k))
    tot0 = sum(dist.values())
    if tot0 != 1:
        raise ValueError(f"{sampler.name}: init_states mass is {tot0}, not 1")

    report = PrefixReport(candidate=sampler.name, n=n, k=k)

    for i in range(k, n + 1):
        if i > k:
            new: dict[State, Fraction] = defaultdict(Fraction)
            for st, p in dist.items():
                trans = sampler.transitions(st, i, k)
                if check_transition_mass:
                    tot = Fraction(0)
                    for _, q in trans:
                        tot += q
                    if tot != 1:
                        raise ValueError(
                            f"{sampler.name}: transition mass {tot} != 1 at state={st}, i={i}"
                        )
                for st2, q in trans:
                    if q:
                        new[st2] += p * q
            dist = dict(new)
            del new
            if len(dist) > max_states:
                raise MemoryError(
                    f"{sampler.name}: {len(dist)} states at i={i} exceeds cap {max_states}"
                )

        mass = sum(dist.values())
        if mass != 1:
            raise ValueError(f"{sampler.name}: mass {mass} != 1 at prefix i={i}")

        report.max_states_seen = max(report.max_states_seen, len(dist))

        # ---- project to the observable k-subset ----
        obs: dict[frozenset[int], Fraction] = defaultdict(Fraction)
        for st, p in dist.items():
            obs[st.observable()] += p

        total = comb(i, k)
        U = Fraction(1, total)

        # M1 ranges over ALL C(i,k) subsets: a sampler whose support MISSES subsets is
        # precisely the failure mode being hunted, so absent subsets contribute |0 - U|.
        maxdev = Fraction(0)
        tv_sum = Fraction(0)
        seen = 0
        for S in combinations(range(1, i + 1), k):
            p = obs.get(frozenset(S), Fraction(0))
            if p:
                seen += 1
            d = abs(p - U)
            tv_sum += d
            if d > maxdev:
                maxdev = d
        if seen != len(obs):
            raise ValueError(
                f"{sampler.name}: support has {len(obs)} sets but only {seen} are valid "
                f"k-subsets of [{i}] -- sampler emitted an out-of-range state"
            )

        incl = [Fraction(0)] * (i + 1)
        for S, p in obs.items():
            for x in S:
                incl[x] += p
        target = Fraction(k, i)
        incl_maxdev = max(abs(incl[x] - target) for x in range(1, i + 1))

        row = PrefixRow(
            i=i,
            m1_abs=maxdev,
            m1_rel=maxdev / U,
            tv=tv_sum / 2,
            support=len(obs),
            support_total=total,
            incl_maxdev=incl_maxdev,
            n_states=len(dist),
        )
        report.rows.append(row)
        del obs

        if maxdev != 0 and report.i_first_fail is None:
            report.i_first_fail = i
            report.m1_rel_at_failure = row.m1_rel
            if stop_on_first_fail:
                report.stopped_early = True
                logger.debug(
                    f"{sampler.name} n={n} k={k}: first failure at i={i} "
                    f"(m1_rel={float(row.m1_rel):.4g}), stopping early"
                )
                break

    del dist
    gc.collect()
    return report


def state_space_size(k: int, i: int, *, ordered: bool) -> int:
    """Number of reachable base states at prefix i (before aux multiplicity)."""
    if ordered:
        size = 1
        for t in range(k):
            size *= i - t
        return size
    return comb(i, k)


def all_k_subsets(i: int, k: int) -> Iterable[tuple[int, ...]]:
    return combinations(range(1, i + 1), k)


def pairwise_inclusion(obs: dict[frozenset[int], Fraction], i: int) -> dict[tuple[int, int], Fraction]:
    """Exact pairwise co-inclusion matrix pi_xy from an observable distribution."""
    pi: dict[tuple[int, int], Fraction] = defaultdict(Fraction)
    for S, p in obs.items():
        ss = sorted(S)
        for a in range(len(ss)):
            for b in range(a + 1, len(ss)):
                pi[(ss[a], ss[b])] += p
    return dict(pi)


def observable_distribution(sampler: Sampler, n: int, k: int) -> dict[frozenset[int], Fraction]:
    """Run the DP to prefix n and return the exact observable distribution there."""
    dist: dict[State, Fraction] = dict(sampler.init_states(k))
    for i in range(k + 1, n + 1):
        new: dict[State, Fraction] = defaultdict(Fraction)
        for st, p in dist.items():
            for st2, q in sampler.transitions(st, i, k):
                if q:
                    new[st2] += p * q
        dist = dict(new)
    obs: dict[frozenset[int], Fraction] = defaultdict(Fraction)
    for st, p in dist.items():
        obs[st.observable()] += p
    return dict(obs)


def dp_cell(args: Sequence[Any]) -> dict[str, Any]:
    """ProcessPool entry point: (sampler_key, n, k, stop_on_first_fail, include_rows)."""
    import e1_candidates  # local import keeps the spawn payload small

    sampler_key, n, k, stop_early, include_rows = args
    sampler = e1_candidates.build(sampler_key, n=n, k=k)
    rep = run_exact_dp(sampler, n, k, stop_on_first_fail=stop_early)
    return rep.to_json(include_rows=include_rows)
