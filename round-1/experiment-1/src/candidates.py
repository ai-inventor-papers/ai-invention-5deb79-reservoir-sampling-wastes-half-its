#!/usr/bin/env python3
"""Candidate reservoir samplers, all against the same ``Sampler`` protocol.

Everything here is measured by the SAME exact DP (``exact_dp.run_exact_dp``), so no
comparison between a candidate and the classical baseline can be contaminated by an
implementation-level difference in the measurement path.

  S0       CLASSICAL-R      the baseline: accept k/i, evict a uniformly random held item
  S1       SUMMODK+REPAIR   deterministic sum-mod-k eviction, minimally repaired by flow
  S1n      SUMMODK-NOREPAIR fully deterministic; the zero-randomness upper bound
  S1f      FLOW-OPTIMAL     min-entropy exactly-uniform eviction from the flow certificates
  S5-CIRC  circular-window  the blind-spot candidate: perfect first-order marginals,
                            broken joint law, genuine O(k) memory

  Negative controls (each must light up):
  C-OFFBY1   accept k/(i-1)              C-OFFBY1b  accept (k-1)/i
  C-FIFO     evict the oldest            C-RR       evict round-robin
  C-SLOT0    always evict slot 0         C-NOACCEPT evict the largest held index
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

from exact_dp import State

import fibermap

# --------------------------------------------------------------------------------------
# Set-based samplers (state = sorted slots, aux = ())
# --------------------------------------------------------------------------------------


class _SetSampler:
    """Base: accept probability ``accept(i, k)``, eviction over positions of the sorted slots."""

    ordered = False
    name = "abstract"

    def accept(self, i: int, k: int) -> Fraction:
        return Fraction(k, i)

    def evict(self, slots: tuple[int, ...], i: int, k: int) -> list[tuple[int, Fraction]]:
        """Return [(0-based position in the SORTED slots, probability)]."""
        raise NotImplementedError

    def init_states(self, k: int) -> dict[State, Fraction]:
        return {State(tuple(range(1, k + 1)), ()): Fraction(1)}

    def transitions(self, state: State, i: int, k: int) -> list[tuple[State, Fraction]]:
        a = self.accept(i, k)
        out: list[tuple[State, Fraction]] = []
        if a != 1:
            out.append((state, 1 - a))
        if a != 0:
            s = state.slots
            for pos, q in self.evict(s, i, k):
                if q == 0:
                    continue
                new = s[:pos] + s[pos + 1 :] + (i,)  # i exceeds every held index -> stays sorted
                out.append((State(new, ()), a * q))
        return out


class ClassicalR(_SetSampler):
    """S0 -- the baseline.  Uniform eviction; log2(k) bits per acceptance."""

    name = "S0_CLASSICAL_R"

    def evict(self, slots, i, k):
        p = Fraction(1, k)
        return [(pos, p) for pos in range(k)]


class SumModK(_SetSampler):
    """S1n -- fully deterministic eviction, r = ((sum(R)-1) mod k) + 1, r-th smallest."""

    name = "S1n_SUMMODK_NOREPAIR"

    def evict(self, slots, i, k):
        return [(fibermap.summodk_position(slots, k), Fraction(1))]


class TableDriven(_SetSampler):
    """S1 / S1f -- eviction read from the exact transportation-flow tables."""

    def __init__(self, rule: str, name: str, n: int, k: int) -> None:
        self.rule = rule
        self.name = name
        self._n = n
        self._k = k
        self._tables: dict[int, dict[tuple[int, ...], list[tuple[int, Fraction]]]] = {}

    def _table(self, i: int, k: int):
        # the arriving item is i, so the source fiber is over the prefix i-1
        if i not in self._tables:
            self._tables[i] = fibermap.eviction_table(self.rule, i - 1, k)
        return self._tables[i]

    def evict(self, slots, i, k):
        return self._table(i, k)[slots]


class EvictLargest(_SetSampler):
    """C-NOACCEPT -- correct accept coin, deterministic 'evict the largest held index'."""

    name = "C_NOACCEPT_EVICT_LARGEST"

    def evict(self, slots, i, k):
        return [(k - 1, Fraction(1))]


class OffByOneAccept(_SetSampler):
    """C-OFFBY1 -- accept k/(i-1) instead of k/i, uniform eviction."""

    name = "C_OFFBY1_ACCEPT_K_OVER_I_MINUS_1"

    def accept(self, i, k):
        a = Fraction(k, i - 1)
        return a if a < 1 else Fraction(1)

    def evict(self, slots, i, k):
        p = Fraction(1, k)
        return [(pos, p) for pos in range(k)]


class OffByOneAcceptB(_SetSampler):
    """C-OFFBY1b -- accept (k-1)/i, uniform eviction."""

    name = "C_OFFBY1b_ACCEPT_K_MINUS_1_OVER_I"

    def accept(self, i, k):
        return Fraction(k - 1, i)

    def evict(self, slots, i, k):
        p = Fraction(1, k)
        return [(pos, p) for pos in range(k)]


# --------------------------------------------------------------------------------------
# Ordered-slot samplers (order of arrival matters -- much larger state space)
# --------------------------------------------------------------------------------------


class _OrderedSampler:
    ordered = True
    name = "abstract-ordered"

    def accept(self, i: int, k: int) -> Fraction:
        return Fraction(k, i)

    def evict_slot(self, state: State, i: int, k: int) -> list[tuple[int, Fraction, Any]]:
        """Return [(slot index to overwrite, probability, new aux)]."""
        raise NotImplementedError

    def init_states(self, k: int) -> dict[State, Fraction]:
        return {State(tuple(range(1, k + 1)), self.init_aux(k)): Fraction(1)}

    def init_aux(self, k: int) -> Any:
        return ()

    def transitions(self, state: State, i: int, k: int) -> list[tuple[State, Fraction]]:
        a = self.accept(i, k)
        out: list[tuple[State, Fraction]] = []
        if a != 1:
            out.append((state, 1 - a))
        if a != 0:
            for slot, q, aux in self.evict_slot(state, i, k):
                if q == 0:
                    continue
                s = list(state.slots)
                s[slot] = i
                out.append((State(tuple(s), aux), a * q))
        return out


class Fifo(_OrderedSampler):
    """C-FIFO -- slots are kept in arrival order; the oldest held item is evicted."""

    name = "C_FIFO_EVICT_OLDEST"

    def transitions(self, state, i, k):
        a = self.accept(i, k)
        out = []
        if a != 1:
            out.append((state, 1 - a))
        if a != 0:
            # drop slot 0 (oldest), shift left, append the new arrival
            new = state.slots[1:] + (i,)
            out.append((State(new, ()), a))
        return out


class RoundRobin(_OrderedSampler):
    """C-RR -- a pointer cycles through the slots."""

    name = "C_RR_ROUND_ROBIN"

    def init_aux(self, k):
        return (0,)

    def evict_slot(self, state, i, k):
        ptr = state.aux[0]
        return [(ptr, Fraction(1), ((ptr + 1) % k,))]


class AlwaysSlot0(_OrderedSampler):
    """C-SLOT0 -- always overwrite slot 0."""

    name = "C_SLOT0_ALWAYS_FIRST"

    def evict_slot(self, state, i, k):
        return [(0, Fraction(1), ())]


# --------------------------------------------------------------------------------------
# The blind-spot candidate
# --------------------------------------------------------------------------------------


def circular_window(anchor: int, size: int, k: int) -> tuple[int, ...]:
    """{((anchor-1+t) mod size) + 1 : t = 0..k-1}, returned sorted."""
    return tuple(sorted(((anchor - 1 + t) % size) + 1 for t in range(k)))


class CircWindow:
    """S5-CIRC -- a k-memory sampler with EXACT first-order marginals and a broken joint law.

    State = a single anchor j, maintained by a correct k=1 reservoir (j <- i with
    probability 1/i).  The reservoir is the circular window of length k starting at j.
    Every item lies in exactly k of the i windows, so P(x in R_i) = k/i exactly for
    every item and every prefix -- yet the support has only i of the C(i,k) subsets.

    Streaming realisation in O(k) memory: hold the k-1 items {1..k-1} (which are exactly
    the wrapped part of any window, since the wrap is always a prefix {1..m} with
    m <= k-1) plus the at most k window items from the anchor onward, plus the counter.
    """

    name = "S5_CIRC_WINDOW"
    ordered = True

    def init_states(self, k: int) -> dict[State, Fraction]:
        p = Fraction(1, k)
        return {State(tuple(range(1, k + 1)), (j,)): p for j in range(1, k + 1)}

    def transitions(self, state: State, i: int, k: int) -> list[tuple[State, Fraction]]:
        j = state.aux[0]
        pj = Fraction(1, i)
        return [
            (State(circular_window(j, i, k), (j,)), 1 - pj),
            (State(circular_window(i, i, k), (i,)), pj),
        ]


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

SET_BASED = {
    "S0": ClassicalR,
    "S1n": SumModK,
    "C-NOACCEPT": EvictLargest,
    "C-OFFBY1": OffByOneAccept,
    "C-OFFBY1b": OffByOneAcceptB,
}

ORDERED = {
    "C-FIFO": Fifo,
    "C-RR": RoundRobin,
    "C-SLOT0": AlwaysSlot0,
    "S5-CIRC": CircWindow,
}

CONTROL_KEYS = ["C-OFFBY1", "C-OFFBY1b", "C-FIFO", "C-RR", "C-SLOT0", "C-NOACCEPT"]
ORDERED_KEYS = set(ORDERED) - {"S5-CIRC"}  # S5-CIRC's aux is a single anchor, so it stays small

DESCRIPTIONS: dict[str, str] = {
    "S0": "baseline classical Algorithm R: accept k/i, evict a uniformly random held item",
    "S1": "sum-mod-k deterministic eviction, minimally repaired by min-cost transportation flow",
    "S1n": "sum-mod-k deterministic eviction, NO repair (zero eviction randomness)",
    "S1f": "flow-optimal eviction: integral certificate where k|(i+1), min-cost repair elsewhere",
    "S5-CIRC": "circular-window sampler: exact first-order marginals, support of size i only",
    "C-OFFBY1": "control: accept k/(i-1) instead of k/i",
    "C-OFFBY1b": "control: accept (k-1)/i instead of k/i",
    "C-FIFO": "control: correct accept coin, evict the oldest held item",
    "C-RR": "control: correct accept coin, evict round-robin by slot pointer",
    "C-SLOT0": "control: correct accept coin, always overwrite slot 0",
    "C-NOACCEPT": "control: correct accept coin, deterministically evict the largest held index",
}


def build(key: str, *, n: int, k: int):
    """Instantiate a candidate by key."""
    if key == "S1":
        return TableDriven("repair", "S1_SUMMODK_REPAIRED", n, k)
    if key == "S1f":
        return TableDriven("flow", "S1f_FLOW_OPTIMAL", n, k)
    if key in SET_BASED:
        return SET_BASED[key]()
    if key in ORDERED:
        return ORDERED[key]()
    raise KeyError(f"unknown candidate {key!r}")


ALL_KEYS = ["S0", "S1", "S1n", "S1f", "S5-CIRC"] + CONTROL_KEYS

__all__ = [
    "build",
    "ALL_KEYS",
    "CONTROL_KEYS",
    "ORDERED_KEYS",
    "DESCRIPTIONS",
    "circular_window",
    "ClassicalR",
    "SumModK",
    "TableDriven",
    "CircWindow",
]
