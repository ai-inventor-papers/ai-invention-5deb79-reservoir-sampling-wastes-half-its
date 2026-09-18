#!/usr/bin/env python3
"""Graded-quality pseudo-random bit generators (the M6 panel).

Every generator exposes the SAME interface so that the bit-accounting layer in
``bits.py`` and the vectorised Monte-Carlo harness in ``montecarlo.py`` can be
driven by any member of the panel without changing a line of sampler code:

    next_bit()            -> int in {0,1}        (scalar, buffered)
    next_bits(m)          -> int                 (m bits, MSB first)
    random_u32(size)      -> np.ndarray[uint32]  (bulk)
    random_float64(size)  -> np.ndarray[float64] in [0,1)
    random_below(size, k) -> np.ndarray[int64]   uniform on {0..k-1}

The panel is graded by *known* quality:

SCREEN panel
    LCG-low16   deliberately weak: the LOW 16 bits of a 32-bit Lehmer/LCG have a
                period of only 2**16 and strong lattice structure.
    xorshift32  Marsaglia (13,17,5); passes many smoke tests, fails BigCrush.
    PCG64       NumPy's default bit generator (good).

HELD-OUT panel
    R250        lagged Fibonacci x_i = x_{i-103} XOR x_{i-250}: the exact
                generator of the Ferrenberg-Landau-Wong (1992) Ising failure.
                The panel's designed-to-fail member.
    MT19937     Mersenne Twister (good equidistribution, linear over GF(2)).
    ChaCha20    os.urandom, buffered 64 KiB at a time (cryptographic gold
                standard; the reference "true randomness" arm).

All weak generators are implemented VECTORISED in NumPy (explicit uint32
wraparound) because a Python-loop LCG cannot deliver 1e9 draws in budget.
"""

from __future__ import annotations

import os
from typing import Final

import numpy as np

__all__ = [
    "BitGen",
    "LCGLow16",
    "XorShift32",
    "PCG64Gen",
    "R250",
    "MT19937Gen",
    "ChaCha20Gen",
    "make_prng",
    "SCREEN_PRNGS",
    "HELDOUT_PRNGS",
    "ALL_PRNGS",
]

_U32: Final = np.uint32
_MASK32: Final = np.uint32(0xFFFFFFFF)
_BULK_WORDS: Final = 2048  # words produced per refill of the scalar bit buffer


class BitGen:
    """Base class: bulk uint32 production + a buffered scalar bit interface.

    Subclasses only have to implement :meth:`_words`, which must return exactly
    ``size`` uint32 words of the generator's output stream and advance the
    internal state.  Everything else (scalar bits, floats, bounded integers) is
    derived here so that every panel member is consumed identically.
    """

    name: str = "base"
    #: number of usable bits carried by each word returned by :meth:`_words`
    word_bits: int = 32

    def __init__(self, seed: int) -> None:
        self.seed = int(seed)
        self._buf: np.ndarray = np.empty(0, dtype=_U32)
        self._buf_pos: int = 0
        self._cur: int = 0
        self._cur_left: int = 0

    # ------------------------------------------------------------------ bulk
    def _words(self, size: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def random_u32(self, size: int) -> np.ndarray:
        """``size`` uint32 words.  Words carrying fewer than 32 bits are packed."""
        if self.word_bits == 32:
            return self._words(size)
        # Pack ceil(32/word_bits) short words into each uint32.
        per = (32 + self.word_bits - 1) // self.word_bits
        raw = self._words(size * per).astype(np.uint64)
        out = np.zeros(size, dtype=np.uint64)
        for j in range(per):
            out = (out << np.uint64(self.word_bits)) | raw[j::per][:size]
        return (out & np.uint64(0xFFFFFFFF)).astype(_U32)

    def random_u64(self, size: int) -> np.ndarray:
        hi = self.random_u32(size).astype(np.uint64)
        lo = self.random_u32(size).astype(np.uint64)
        return (hi << np.uint64(32)) | lo

    def random_float64(self, size: int) -> np.ndarray:
        """Uniform in [0,1) built from 53 fresh bits (the IEEE-754 convention)."""
        u = self.random_u64(size) >> np.uint64(11)
        return u.astype(np.float64) * (2.0**-53)

    def random_below(self, size: int, k: int) -> np.ndarray:
        """Uniform on ``{0..k-1}`` by 64-bit modulo -- the deployed-library route.

        ``u64 % k`` carries a modulo bias of at most ``k / 2**64 < 1e-13`` for
        every ``k`` used here, which is 9 orders of magnitude below the Monte
        Carlo null band; the *exact* rejection-based route lives in
        :mod:`bits` where exactness is the thing being measured.
        """
        return (self.random_u64(size) % np.uint64(k)).astype(np.int64)

    # ---------------------------------------------------------------- scalar
    def _refill(self) -> None:
        self._buf = self._words(_BULK_WORDS)
        self._buf_pos = 0

    def _next_word(self) -> int:
        if self._buf_pos >= self._buf.size:
            self._refill()
        w = int(self._buf[self._buf_pos])
        self._buf_pos += 1
        return w

    def next_bit(self) -> int:
        if self._cur_left == 0:
            self._cur = self._next_word()
            self._cur_left = self.word_bits
        self._cur_left -= 1
        return (self._cur >> self._cur_left) & 1

    def next_bits(self, m: int) -> int:
        """``m`` bits, MSB first, as one integer."""
        out = 0
        need = m
        while need > 0:
            if self._cur_left == 0:
                self._cur = self._next_word()
                self._cur_left = self.word_bits
            take = min(need, self._cur_left)
            self._cur_left -= take
            chunk = (self._cur >> self._cur_left) & ((1 << take) - 1)
            out = (out << take) | chunk
            need -= take
        return out


class LCGLow16(BitGen):
    """Numerical-Recipes LCG, emitting only the LOW 16 bits (deliberately weak).

    x_{n+1} = (1664525 x_n + 1013904223) mod 2**32.  The low 16 bits have period
    2**16, so any statistic that consumes more than 65536 of them sees exact
    repetition.  Vectorised by pre-computing the block-jump constants
    A_j = a**j and C_j = c*(a**(j-1)+...+1) once, so a whole block of B values is
    produced with two array ops.
    """

    name = "LCG-low16"
    word_bits = 16
    _A: Final = 1664525
    _C: Final = 1013904223
    _M: Final = 1 << 32
    _BLOCK: Final = 8192

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        self._x = int(seed) % self._M
        a_pows = np.empty(self._BLOCK, dtype=np.uint64)
        c_acc = np.empty(self._BLOCK, dtype=np.uint64)
        a, c = 1, 0
        for j in range(self._BLOCK):
            a = (a * self._A) % self._M
            c = (c * self._A + self._C) % self._M
            a_pows[j] = a
            c_acc[j] = c
        self._a_pows = a_pows
        self._c_acc = c_acc

    def _words(self, size: int) -> np.ndarray:
        out = np.empty(size, dtype=_U32)
        done = 0
        while done < size:
            b = min(self._BLOCK, size - done)
            vals = (self._a_pows[:b] * np.uint64(self._x) + self._c_acc[:b]) & np.uint64(
                0xFFFFFFFF
            )
            out[done : done + b] = (vals & np.uint64(0xFFFF)).astype(_U32)
            self._x = int(vals[b - 1])
            done += b
        return out


class XorShift32(BitGen):
    """Marsaglia xorshift32 with the classic (13, 17, 5) triple.

    The recurrence is a GF(2)-linear map, so a block of B successive states is
    obtained by applying the 32x32 GF(2) matrix M**B.  We pre-compute the
    *leapfrog* matrix once and then advance ``_BLOCK`` independent lanes in
    lockstep, which reproduces the generator's own sequence exactly while using
    only vector ops.
    """

    name = "xorshift32"
    word_bits = 32
    _BLOCK: Final = 4096

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        s = int(seed) & 0xFFFFFFFF
        self._x = s if s != 0 else 0x9E3779B9
        # Column images of the single-step map, as a GF(2) matrix in bit-packed
        # form: col_img[j] = image of the unit vector e_j.
        step = np.array([self._step_scalar(1 << j) for j in range(32)], dtype=np.uint64)
        self._block_mat = self._mat_pow(step, self._BLOCK)
        # lane seeds: the first _BLOCK successive states of the generator
        lanes = np.empty(self._BLOCK, dtype=np.uint64)
        v = self._x
        for j in range(self._BLOCK):
            v = self._step_scalar(v)
            lanes[j] = v
        self._lanes = lanes

    @staticmethod
    def _step_scalar(x: int) -> int:
        x &= 0xFFFFFFFF
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= x >> 17
        x ^= (x << 5) & 0xFFFFFFFF
        return x & 0xFFFFFFFF

    @staticmethod
    def _mat_apply(mat: np.ndarray, vec: np.ndarray) -> np.ndarray:
        """Apply a bit-packed GF(2) matrix (column images) to uint64 vectors."""
        out = np.zeros_like(vec)
        for j in range(32):
            sel = (vec >> np.uint64(j)) & np.uint64(1)
            out ^= np.uint64(mat[j]) * sel
        return out

    @classmethod
    def _mat_pow(cls, mat: np.ndarray, power: int) -> np.ndarray:
        result = np.array([1 << j for j in range(32)], dtype=np.uint64)  # identity
        base = mat.copy()
        p = power
        while p:
            if p & 1:
                result = cls._mat_apply(base, result)
            base = cls._mat_apply(base, base)
            p >>= 1
        return result

    def _words(self, size: int) -> np.ndarray:
        out = np.empty(size, dtype=_U32)
        done = 0
        while done < size:
            b = min(self._BLOCK, size - done)
            out[done : done + b] = self._lanes[:b].astype(_U32)
            self._lanes = self._mat_apply(self._block_mat, self._lanes)
            done += b
        return out


class R250(BitGen):
    """Kirkpatrick-Stoll R250: x_i = x_{i-103} XOR x_{i-250} (32-bit words).

    This is the generator that produced the famous wrong Ising-model answer in
    Ferrenberg, Landau & Wong, PRL 69, 3382 (1992).  Its three-term GF(2)
    recurrence gives it very poor higher-order equidistribution, which is
    exactly the defect a joint-law test should expose.
    """

    name = "R250"
    word_bits = 32
    _P: Final = 250
    _Q: Final = 103

    def __init__(self, seed: int) -> None:
        super().__init__(seed)
        boot = np.random.Generator(np.random.PCG64(int(seed)))
        self._state = boot.integers(0, 1 << 32, size=self._P, dtype=np.uint32)
        # Break the lattice-free bootstrap: enforce linear independence the
        # canonical way (set a triangular pattern of high bits).
        for j in range(32):
            self._state[j] = (self._state[j] | np.uint32(1 << 31)) & np.uint32(
                ~((1 << (31 - j)) - 1) & 0xFFFFFFFF
            )
        self._tail = self._state.copy()

    def _words(self, size: int) -> np.ndarray:
        """Vectorised in blocks of Q=103.

        Within any 103 consecutive outputs, output j reads index ``j-103``,
        which lies strictly before the block start and is therefore already
        final -- so a whole block is one XOR of two slices.  This reproduces the
        scalar recurrence bit-for-bit (asserted in tests/test_prngs.py).
        """
        p, q = self._P, self._Q
        hist = np.empty(p + size, dtype=_U32)
        hist[:p] = self._tail
        for st in range(p, p + size, q):
            en = min(st + q, p + size)
            hist[st:en] = hist[st - q : en - q] ^ hist[st - p : en - p]
        self._tail = hist[-p:].copy()
        return hist[p:]


class _NumPyGen(BitGen):
    """Wrapper for a NumPy BitGenerator (PCG64 / MT19937)."""

    def __init__(self, seed: int, bitgen_cls) -> None:
        super().__init__(seed)
        self._rng = np.random.Generator(bitgen_cls(int(seed)))

    def _words(self, size: int) -> np.ndarray:
        return self._rng.integers(0, 1 << 32, size=size, dtype=np.uint32)

    def random_float64(self, size: int) -> np.ndarray:
        return self._rng.random(size)

    def random_below(self, size: int, k: int) -> np.ndarray:
        return self._rng.integers(0, k, size=size, dtype=np.int64)


class PCG64Gen(_NumPyGen):
    name = "PCG64"
    word_bits = 32

    def __init__(self, seed: int) -> None:
        super().__init__(seed, np.random.PCG64)


class MT19937Gen(_NumPyGen):
    name = "MT19937"
    word_bits = 32

    def __init__(self, seed: int) -> None:
        super().__init__(seed, np.random.MT19937)


class ChaCha20Gen(BitGen):
    """OS CSPRNG (ChaCha20 on Linux), buffered 64 KiB at a time.

    Not reproducible by seed -- that is intrinsic to a true-entropy source and is
    recorded as such in the manifest.  It is the panel's gold-standard arm.
    """

    name = "ChaCha20"
    word_bits = 32
    _CHUNK: Final = 1 << 16

    def _words(self, size: int) -> np.ndarray:
        out = np.empty(size, dtype=_U32)
        done = 0
        while done < size:
            b = min(self._CHUNK // 4, size - done)
            out[done : done + b] = np.frombuffer(os.urandom(4 * b), dtype=_U32)
            done += b
        return out


_REGISTRY = {
    "LCG-low16": LCGLow16,
    "xorshift32": XorShift32,
    "PCG64": PCG64Gen,
    "R250": R250,
    "MT19937": MT19937Gen,
    "ChaCha20": ChaCha20Gen,
}

SCREEN_PRNGS: Final = ("LCG-low16", "xorshift32", "PCG64")
HELDOUT_PRNGS: Final = ("R250", "MT19937", "ChaCha20")
ALL_PRNGS: Final = SCREEN_PRNGS + HELDOUT_PRNGS


def make_prng(name: str, seed: int) -> BitGen:
    if name not in _REGISTRY:
        raise KeyError(f"unknown PRNG {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name](seed)
