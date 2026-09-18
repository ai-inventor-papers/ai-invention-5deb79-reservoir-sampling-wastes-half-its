#!/usr/bin/env python3
"""One counted fair-coin source, three accounting regimes.

The whole disagreement between the MAIN hypothesis ("eviction randomness is
wasted") and candidate 4 ("*representation* is wasted") is an accounting
question, so every candidate sampler is driven through the SAME counted bit
object and only the *regime* changes:

R-naive   (:class:`NaiveSource`)      what libraries actually do: one 53- or
                                      64-bit machine word per random decision.
R-KY      (:class:`KnuthYaoSource`)   entropy-optimal PER DECISION, no reuse
                                      across decisions (Knuth-Yao DDG trees;
                                      Lumbroso's Fast Dice Roller for ranges).
R-recycle (:class:`RecyclingSource`)  interval reuse: residual entropy is
                                      returned to a buffer, so the amortised
                                      cost over a whole stream approaches the
                                      sum of the true decision entropies.

Every flip is attributed to exactly one tag in
``accept | evict | repair | representation | other`` and the tags are asserted
to sum to the total.

CORRECTION TO THE PLANNED R-recycle BERNOULLI
---------------------------------------------
The naive recycling Bernoulli ``uniform_int(b) < a`` costs ~log2(b) net bits per
call, *not* h(a/b): the accept/reject decision throws away the residual
information in the drawn integer.  The correct construction pushes the
*conditional residual* back into the buffer --

    out = uniform_int(b)
    if out <  a:  push(out,     a)      -> residual uniform on {0..a-1}
    else:         push(out - a, b - a)  -> residual uniform on {0..b-a-1}

Then the buffer capacity M evolves as M -> M*p (w.p. p) or M*(1-p) (w.p. 1-p),
so E[log2 M_new - log2 M] = -h(p) and the refill loop pulls exactly h(p) fresh
bits per call in expectation.  This is what makes R-recycle reach the entropy
floor, and it is verified numerically in tests/test_bits.py.
"""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

from prngs import BitGen, make_prng

__all__ = [
    "BitStream",
    "NaiveSource",
    "KnuthYaoSource",
    "RecyclingSource",
    "make_source",
    "REGIMES",
    "TAGS",
]

TAGS = ("accept", "evict", "repair", "representation", "other")


class BitStream:
    """Counts every fair coin flip pulled from an underlying :class:`BitGen`."""

    def __init__(self, gen: BitGen, label: str = "") -> None:
        self.gen = gen
        self.label = label
        self.flips = 0
        self.by_tag: dict[str, int] = defaultdict(int)
        self.tag = "other"

    @contextmanager
    def tagged(self, tag: str) -> Iterator[None]:
        if tag not in TAGS:
            raise ValueError(f"unknown flip tag {tag!r}")
        prev = self.tag
        self.tag = tag
        try:
            yield
        finally:
            self.tag = prev

    def bit(self) -> int:
        self.flips += 1
        self.by_tag[self.tag] += 1
        return self.gen.next_bit()

    def bits(self, m: int) -> int:
        """``m`` bits at once (bulk-charged, bulk-pulled) -- MSB first."""
        if m <= 0:
            return 0
        self.flips += m
        self.by_tag[self.tag] += m
        return self.gen.next_bits(m)

    def float64(self) -> float:
        """Exact 53-bit uniform in [0,1), charged as 53 flips."""
        return self.bits(53) * (2.0**-53)

    def tag_totals(self) -> dict[str, int]:
        d = {t: int(self.by_tag.get(t, 0)) for t in TAGS}
        if sum(d.values()) != self.flips:
            raise AssertionError(
                f"tag attribution {sum(d.values())} != total flips {self.flips}"
            )
        return d


class _Source:
    """Common interface: ``bernoulli(a, b)`` and ``uniform_int(k)``, both exact.

    NET vs RAW attribution
    ----------------------
    Under R-naive and R-KY a flip is pulled by the decision that needs it, so
    "which tag pulled the flip" is the whole story.  Under R-recycle it is NOT:
    the buffer is shared, so a flip pulled while deciding an *acceptance* can
    end up paying for a later *eviction*, and raw refill-trigger attribution
    reports eviction as free when it is not.

    :meth:`op` therefore charges each tagged region its NET buffer consumption

        net(region) = (bits pulled inside it) + log2(M_before) - log2(M_after)

    which is additive, regime-independent, and telescopes to
    ``flips_total - log2(M_final)`` over a whole run.  Both the raw and the net
    decompositions are reported, and the run-level identity is asserted.
    """

    regime = "base"

    def __init__(self, stream: BitStream) -> None:
        self.s = stream
        self.net_by_tag: dict[str, float] = {t: 0.0 for t in TAGS}

    def _buffer_bits(self) -> float:
        return 0.0

    @contextmanager
    def op(self, tag: str) -> Iterator[None]:
        """Tag a decision AND charge it its net buffer consumption."""
        f0 = self.s.flips
        b0 = self._buffer_bits()
        with self.s.tagged(tag):
            yield
        self.net_by_tag[tag] += (self.s.flips - f0) + (b0 - self._buffer_bits())

    def net_totals(self) -> dict[str, float]:
        return {t: float(self.net_by_tag[t]) for t in TAGS}

    def net_total(self) -> float:
        return float(self.s.flips - self._buffer_bits())

    @property
    def flips(self) -> int:
        return self.s.flips

    def bernoulli(self, a: int, b: int) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def uniform_int(self, k: int) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def float64(self) -> float:
        return self.s.float64()


class NaiveSource(_Source):
    """R-naive: one machine word per decision, exactly as deployed libraries do.

    ``word_bits=64`` is the headline (a deployed 64-bit word); ``word_bits=53``
    is recorded as ``naive53`` so the claim is robust to that bookkeeping choice.

    The Bernoulli test is the library idiom ``u * b < a`` evaluated on a fresh
    53-bit uniform; the range draw is the library idiom ``word % k``.  Both are
    *slightly* biased (that is the point of the regime) but the bias is
    O(2**-53) and O(k/2**64) respectively, far below anything measured here.
    """

    regime = "R-naive"

    def __init__(self, stream: BitStream, word_bits: int = 64) -> None:
        super().__init__(stream)
        if word_bits not in (53, 64):
            raise ValueError("word_bits must be 53 or 64")
        self.word_bits = word_bits

    def bernoulli(self, a: int, b: int) -> int:
        w = self.s.bits(self.word_bits)
        u = w * (2.0 ** -self.word_bits)
        return 1 if u * b < a else 0

    def uniform_int(self, k: int) -> int:
        if k <= 1:
            return 0
        return self.s.bits(self.word_bits) % k


class KnuthYaoSource(_Source):
    """R-KY: entropy-optimal per decision, with NO reuse between decisions.

    Bernoulli uses the Knuth-Yao DDG tree for a Bernoulli variate, which is
    exactly "compare a uniform binary expansion against the binary expansion of
    p, generated lazily by long division, and stop at the first disagreement".
    Because the two expansions disagree with probability 1/2 at every position,
    the expected cost is EXACTLY 2 flips for every p -- which is the whole point
    of the regime: ``h(k/i) -> 0`` as the stream grows, but the per-decision
    optimum does not.  (Lumbroso, arXiv:1304.1916, reports the same 2-flip
    average.)

    Range draws use Lumbroso's Fast Dice Roller, expected cost < log2(k) + 2.
    """

    regime = "R-KY"

    def bernoulli(self, a: int, b: int) -> int:
        if a <= 0:
            return 0
        if a >= b:
            return 1
        num = a
        while True:
            num *= 2
            if num >= b:
                p_bit = 1
                num -= b
            else:
                p_bit = 0
            u_bit = self.s.bit()
            if u_bit < p_bit:
                return 1
            if u_bit > p_bit:
                return 0

    def uniform_int(self, k: int) -> int:
        if k <= 1:
            return 0
        v, c = 1, 0
        while True:
            v, c = 2 * v, 2 * c + self.s.bit()
            if v >= k:
                if c < k:
                    return c
                v, c = v - k, c - k


class RecyclingSource(_Source):
    """R-recycle: interval reuse (Draper & Saad style entropy recycling).

    State is a pair ``(x, M)`` with ``x`` uniform on ``{0..M-1}``: an entropy
    buffer.  ``uniform_int(k)`` consumes only what it needs and returns the
    residual; ``bernoulli(a, b)`` additionally returns the *conditional*
    residual, which is what makes the net cost equal ``h(a/b)`` (see module
    docstring).

    ``M`` is capped at ``2**cap_bits`` so Python big-int arithmetic stays cheap;
    entropy above the cap is discarded, which can only make the measured cost an
    OVER-estimate, never an under-estimate.  The default cap of 96 bits keeps
    the discarded fraction below 1e-12 bits per call for every configuration
    used here.
    """

    regime = "R-recycle"

    def __init__(self, stream: BitStream, thresh_bits: int = 40, cap_bits: int = 112) -> None:
        super().__init__(stream)
        self.x = 0
        self.M = 1
        self.thresh = 1 << thresh_bits
        self.cap = 1 << cap_bits

    def _buffer_bits(self) -> float:
        return math.log2(self.M) if self.M > 1 else 0.0

    def _refill(self, need: int) -> None:
        """Top the buffer up to ``need * thresh``.

        Refilling only to ``need`` (the naive reading of the recycling scheme)
        leaves a quotient of q = M // k that is O(1) rather than O(M/k), so
        essentially ALL of the buffer is destroyed by every draw and the scheme
        degenerates to Knuth-Yao.  Topping up to a threshold that is 2**40 times
        the requested range keeps ``q ~ M/k``, so exactly ``log2(k)`` bits are
        consumed per range draw, which is what makes the entropy floor
        reachable.  Verified numerically in tests/test_bits.py.
        """
        target = need * self.thresh
        while self.M < target:
            self.x = 2 * self.x + self.s.bit()
            self.M *= 2

    def _push(self, v: int, m: int) -> None:
        """Merge an independent uniform-on-``m`` value back into the buffer."""
        if m <= 1:
            return
        if self.M * m > self.cap:
            return  # discard: conservative (over-counts the true cost)
        self.x = self.x * m + v
        self.M *= m

    def uniform_int(self, k: int) -> int:
        if k <= 1:
            return 0
        while True:
            self._refill(k)
            q, r = divmod(self.M, k)
            if self.x < q * k:
                out = self.x % k
                self.x, self.M = self.x // k, q
                return out
            # rejection branch: the residual is still uniform on {0..r-1}
            self.x -= q * k
            self.M = r

    def bernoulli(self, a: int, b: int) -> int:
        """Exact Bernoulli(a/b) whose NET cost is h(a/b) bits (see module doc)."""
        if a <= 0:
            return 0
        if a >= b:
            return 1
        out = self.uniform_int(b)
        if out < a:
            self._push(out, a)
            return 1
        self._push(out - a, b - a)
        return 0

    def bernoulli_dyadic(self, a: int, b: int) -> int:
        """Alternative: the Knuth-Yao walk, but drawing its bits from the buffer.

        This is the variant the plan asked to compare against, implemented
        honestly: the DDG walk halts at the first disagreement between the
        uniform bit stream and p's binary expansion, so it costs 2 buffer-bits
        per call regardless of p, and the buffer gives no discount because each
        walk bit is genuinely consumed.  Kept so the cheaper of the two is
        REPORTED rather than assumed -- see ``bernoulli_variant_costs`` in the
        bit-measurement stage.
        """
        if a <= 0:
            return 0
        if a >= b:
            return 1
        num = a
        while True:
            num *= 2
            if num >= b:
                p_bit = 1
                num -= b
            else:
                p_bit = 0
            u_bit = self.uniform_int(2)
            if u_bit < p_bit:
                return 1
            if u_bit > p_bit:
                return 0


REGIMES = ("naive64", "naive53", "KY", "recycle")


def make_source(regime: str, prng: str, seed: int, label: str = "") -> _Source:
    """Build a counted source for one of :data:`REGIMES` over a named PRNG."""
    stream = BitStream(make_prng(prng, seed), label=label or f"{regime}/{prng}")
    if regime == "naive64":
        return NaiveSource(stream, word_bits=64)
    if regime == "naive53":
        return NaiveSource(stream, word_bits=53)
    if regime == "KY":
        return KnuthYaoSource(stream)
    if regime == "recycle":
        return RecyclingSource(stream)
    raise KeyError(f"unknown regime {regime!r}; known: {REGIMES}")
