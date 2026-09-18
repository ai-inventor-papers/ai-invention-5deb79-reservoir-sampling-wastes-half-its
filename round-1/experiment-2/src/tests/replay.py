"""Scripted randomness so the scalar and vectorised paths can be compared EXACTLY.

``ScriptedSource`` and ``ReplayGen`` hand out the *same* sequence of uniforms and
slot indices to the two implementations of every sampler, so any off-by-one in
the vectorised sorted-reservoir eviction shows up as a literal mismatch rather
than as a statistical near-miss.
"""

from __future__ import annotations

import numpy as np

from bits import BitStream, _Source
from prngs import BitGen, make_prng


class _Script:
    def __init__(self, floats: list[float], slots: list[int]) -> None:
        self.floats = floats
        self.slots = slots
        self.fi = 0
        self.si = 0

    def nextf(self, m: int) -> np.ndarray:
        out = np.asarray(self.floats[self.fi : self.fi + m], dtype=np.float64)
        if out.size < m:
            raise RuntimeError("script exhausted (floats)")
        self.fi += m
        return out

    def nexts(self, m: int, k: int) -> np.ndarray:
        out = np.asarray(self.slots[self.si : self.si + m], dtype=np.int64) % k
        if out.size < m:
            raise RuntimeError("script exhausted (slots)")
        self.si += m
        return out


class ScriptedSource(_Source):
    regime = "scripted"

    def __init__(self, script: _Script) -> None:
        super().__init__(BitStream(make_prng("PCG64", 0), label="scripted"))
        self.script = script

    def bernoulli(self, a: int, b: int) -> int:
        return 1 if float(self.script.nextf(1)[0]) * b < a else 0

    def uniform_int(self, k: int) -> int:
        if k <= 1:
            return 0
        return int(self.script.nexts(1, k)[0])

    def float64(self) -> float:
        return float(self.script.nextf(1)[0])


class ReplayGen(BitGen):
    name = "replay"
    word_bits = 32

    def __init__(self, script: _Script) -> None:
        super().__init__(0)
        self.script = script

    def _words(self, size: int) -> np.ndarray:  # pragma: no cover - unused
        raise RuntimeError("ReplayGen serves only float64/below")

    def random_float64(self, size: int) -> np.ndarray:
        return self.script.nextf(size)

    def random_below(self, size: int, k: int) -> np.ndarray:
        return self.script.nexts(size, k)


def make_scripts(seed: int, n_floats: int, n_slots: int, kmax: int) -> tuple[_Script, _Script]:
    rng = np.random.default_rng(seed)
    f = [float(x) for x in rng.random(n_floats)]
    s = [int(x) for x in rng.integers(0, kmax, n_slots)]
    return _Script(list(f), list(s)), _Script(list(f), list(s))
