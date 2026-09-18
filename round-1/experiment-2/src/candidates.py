#!/usr/bin/env python3
"""The sampler registry: every candidate in BOTH a scalar (bit-counted) and a
vectorised (Monte-Carlo) implementation, plus the exact transition kernel used
by the rational verifier in :mod:`exact`.

STRUCTURAL FACT exploited everywhere
------------------------------------
The stream is the index sequence 1..n in INCREASING order, so an accepted item
is always larger than every held item.  A reservoir kept in sorted order
therefore STAYS sorted under "delete the r-th smallest, then append the new item
at the end".  That makes the sum-mod-k rule O(k) per acceptance and trivially
vectorisable with ``np.take_along_axis``.

Registry
--------
S0     CLASSICAL-R          textbook Algorithm R (the BASELINE)
S1     SUMMODK+REPAIR       the MAIN hypothesis: deterministic eviction rank
                            (sum of reservoir) mod k, randomised only on the
                            fraction s/m of acceptances where a deterministic
                            map provably cannot balance the fibre
S1n    SUMMODK-NOREPAIR     same, never randomised -- negative control AND the
                            zero-eviction-bits upper bound on savings
S2     ALGO-L-float64       Li (1994) Algorithm L exactly as shipped
S2f    ALGO-L-fixed         the corrected rewrite (tracks V = 1-W)
S3     BOTTOM-K             Efraimidis-Spirakis A-Res with uniform weights:
                            one full float PER ITEM -- the bit-costliest baseline
S4     DDG-R                S0 driven by the Knuth-Yao source (candidate 4)
S4d    DEPLOYED-R           S0 driven by a 64-bit-per-decision source
S5a    ROUND-ROBIN          evict slot = (#acceptances) mod k
S5b    FIFO                 evict the oldest held item
S5c    SLOT0                always evict slot 0
S5d    = S1n
S5e    BLOCK-LOCK           streaming systematic: a classical 1-reservoir run
                            over CONSECUTIVE BLOCKS of k items.  First-order
                            inclusion is EXACTLY k/i at every prefix divisible
                            by k, support is only i/k subsets.
S5f    OFFBYONE             accept with probability k/(i+1) instead of k/i
S5g    PAIR-LOCK            classical (k/2)-reservoir over consecutive PAIRS:
                            exact first-order, support C(n/2, k/2)
S6sys  SYSTEMATIC(oracle n) textbook systematic sampling with a random start.
                            NOT a reservoir sampler -- it needs n -- and is
                            labelled as such; it is the guaranteed blind-spot
                            exhibit (support exactly n/k).

S4/S4d are distributionally identical to S0 and therefore appear only in the
bit accounting, never in the Monte Carlo.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Callable

import numpy as np

from bits import _Source
from prngs import BitGen

__all__ = ["CANDIDATES", "Candidate", "get_candidate", "MC_CANDIDATES", "BIT_CANDIDATES"]


# --------------------------------------------------------------------------- #
#  scalar, bit-source-driven implementations (these are what M4 counts)        #
# --------------------------------------------------------------------------- #
def _scalar_slotwise(
    src: _Source,
    n: int,
    k: int,
    *,
    accept_num: Callable[[int], int],
    accept_den: Callable[[int], int],
    choose_slot,
) -> list[int]:
    """Generic 'accept then evict one slot' scalar driver (slot order preserved)."""
    res = list(range(1, k + 1))
    state: dict[str, int] = {"accepts": 0}
    for i in range(k + 1, n + 1):
        with src.op("accept"):
            acc = src.bernoulli(accept_num(i), accept_den(i))
        if acc:
            r = choose_slot(src, res, i, k, state)
            res[r] = i
            state["accepts"] += 1
    return res


def _slot_uniform(src, res, i, k, state) -> int:
    with src.op("evict"):
        return src.uniform_int(k)


def _slot_roundrobin(src, res, i, k, state) -> int:
    return state["accepts"] % k


def _slot_zero(src, res, i, k, state) -> int:
    return 0


def s0_scalar(src: _Source, n: int, k: int) -> list[int]:
    return _scalar_slotwise(
        src, n, k, accept_num=lambda i: k, accept_den=lambda i: i, choose_slot=_slot_uniform
    )


def s5a_scalar(src: _Source, n: int, k: int) -> list[int]:
    return _scalar_slotwise(
        src, n, k, accept_num=lambda i: k, accept_den=lambda i: i, choose_slot=_slot_roundrobin
    )


def s5c_scalar(src: _Source, n: int, k: int) -> list[int]:
    return _scalar_slotwise(
        src, n, k, accept_num=lambda i: k, accept_den=lambda i: i, choose_slot=_slot_zero
    )


def s5f_scalar(src: _Source, n: int, k: int) -> list[int]:
    return _scalar_slotwise(
        src, n, k, accept_num=lambda i: k, accept_den=lambda i: i + 1, choose_slot=_slot_uniform
    )


def s5b_scalar(src: _Source, n: int, k: int) -> list[int]:
    """FIFO: reservoir held in ARRIVAL order, oldest evicted."""
    res = list(range(1, k + 1))
    for i in range(k + 1, n + 1):
        with src.op("accept"):
            acc = src.bernoulli(k, i)
        if acc:
            res.pop(0)
            res.append(i)
    return res


def _summodk_scalar(src: _Source, n: int, k: int, *, repair: bool) -> list[int]:
    """S1 / S1n.  Reservoir kept SORTED; the new item always appends at the end."""
    res = list(range(1, k + 1))
    rowsum = sum(res)
    for i in range(k + 1, n + 1):
        with src.op("accept"):
            acc = src.bernoulli(k, i)
        if not acc:
            continue
        m = i - k          # fibre multiplicity: each (k-1)-subset of [i-1] has i-k preimages
        s = m % k          # deterministic balance is possible only when s == 0
        r = (rowsum - 1) % k
        if repair and s != 0:
            with src.op("repair"):
                if src.bernoulli(s, m):
                    r = src.uniform_int(k)
        rowsum += i - res[r]
        del res[r]
        res.append(i)      # stays sorted, because i exceeds every held item
    return res


def s1_scalar(src: _Source, n: int, k: int) -> list[int]:
    return _summodk_scalar(src, n, k, repair=True)


def s1n_scalar(src: _Source, n: int, k: int) -> list[int]:
    return _summodk_scalar(src, n, k, repair=False)


def _summodk_hybrid_scalar(src: _Source, n: int, k: int) -> list[int]:
    """S1h: deterministic rank when k | (i-k), uniform rank otherwise."""
    res = list(range(1, k + 1))
    rowsum = sum(res)
    for i in range(k + 1, n + 1):
        with src.op("accept"):
            acc = src.bernoulli(k, i)
        if not acc:
            continue
        if (i - k) % k == 0:
            r = (rowsum - 1) % k
        else:
            with src.op("evict"):
                r = src.uniform_int(k)
        rowsum += i - res[r]
        del res[r]
        res.append(i)
    return res


def s3_scalar(src: _Source, n: int, k: int) -> list[int]:
    """Bottom-k / A-Res with uniform weights: keys U_i, keep the k smallest.

    One FULL float per item -- charged to the ``representation`` tag, because
    that is exactly the cost candidate 4 says dominates.
    """
    import heapq

    heap: list[tuple[float, int]] = []
    for i in range(1, n + 1):
        with src.op("representation"):
            key = src.float64()
        if len(heap) < k:
            heapq.heappush(heap, (-key, i))
        elif -key > heap[0][0]:
            heapq.heapreplace(heap, (-key, i))
    return sorted(idx for _, idx in heap)


def _log1mexp(x: float) -> float:
    """Stable log(1 - exp(x)) for x < 0 (the log1mexp identity)."""
    import math

    if x >= 0.0:
        return float("-inf")
    if x < -0.6931471805599453:
        return math.log1p(-math.exp(x))
    return math.log(-math.expm1(x))


def s2_scalar(src: _Source, n: int, k: int, *, variant: str = "shipped") -> list[int]:
    """Algorithm L (Li 1994), in three numerically distinct formulations.

    W is the k-th SMALLEST key seen so far, so the number of skipped items is
    Geometric(W) and  S = floor( log(U) / log(1 - W) ).  Crucially W DECREASES
    along the stream, to W ~ k/n -- it does NOT approach 1 except transiently at
    step 0 for large k.

    variant="shipped"  : the textbook code, S = floor(log(U) / log(1.0 - W)).
                         Its defect is the W -> 0 limit: once W < 2**-53 the
                         expression ``1.0 - W`` rounds to exactly 1.0, the
                         denominator becomes 0.0 and the sampler divides by
                         zero.  Short of that the relative error of the skip
                         rate is ~ 2**-53 / W ~ 2**-53 * n / k.
    variant="vtrack"   : the fix proposed in the hypothesis/plan -- track
                         V = 1-W directly.  Kept and measured because it is the
                         NATURAL fix for the W -> 1 limit, which is the wrong
                         limit; V -> 1 exactly where the problem is, so it
                         cannot help.  Its failure is a result.
    variant="logdomain": the fix that actually works -- accumulate log W so
                         nothing ever underflows, and evaluate log(1-W) with the
                         stable log1mexp identity.

    Algorithm L is also the interesting bit-accounting case: it makes only
    O(k log(n/k)) random DECISIONS in total instead of n, but each decision
    needs a full 53-bit float, so its cost is almost pure ``representation``.
    """
    import math

    if variant not in ("shipped", "vtrack", "logdomain"):
        raise ValueError(f"unknown Algorithm-L variant {variant!r}")
    eps_hi = 1.0 - 2.0**-53
    res = list(range(1, k + 1))
    with src.op("representation"):
        u0 = min(max(src.float64(), 5e-324), eps_hi)
    if variant == "vtrack":
        v = -math.expm1(math.log1p(-u0) / k)          # V = 1 - W
    elif variant == "logdomain":
        logw = math.log1p(-u0) / k
    else:
        w = math.exp(math.log1p(-u0) / k)
    i = k
    guard = 0
    while True:
        guard += 1
        if guard > 4 * n + 1000:
            break
        with src.op("accept"):
            u1 = min(max(src.float64(), 5e-324), eps_hi)
        num = math.log1p(-u1)
        if variant == "vtrack":
            denom = math.log(v) if v > 0.0 else float("-inf")
        elif variant == "logdomain":
            denom = _log1mexp(logw)
        else:
            one_minus_w = 1.0 - w
            denom = math.log(one_minus_w) if one_minus_w > 0.0 else float("-inf")
        skip = math.floor(num / denom) if denom not in (0.0, float("-inf")) else 0.0
        if not math.isfinite(skip) or skip < 0:
            skip = 0.0
        i += int(skip) + 1
        if i > n:
            break
        with src.op("evict"):
            res[src.uniform_int(k)] = i
        with src.op("representation"):
            u2 = min(max(src.float64(), 5e-324), eps_hi)
        if variant == "vtrack":
            v_step = -math.expm1(math.log1p(-u2) / k)
            v = v + v_step - v * v_step
        elif variant == "logdomain":
            logw += math.log1p(-u2) / k
        else:
            w *= math.exp(math.log1p(-u2) / k)
    return sorted(res)


def s2f_scalar(src: _Source, n: int, k: int) -> list[int]:
    return s2_scalar(src, n, k, variant="logdomain")


def s2v_scalar(src: _Source, n: int, k: int) -> list[int]:
    return s2_scalar(src, n, k, variant="vtrack")


def s5e_scalar(src: _Source, n: int, k: int) -> list[int]:
    """BLOCK-LOCK: a classical 1-reservoir over consecutive blocks of k items."""
    res = list(range(1, k + 1))
    blocks = 1
    for i in range(k + 1, n + 1):
        if i % k:
            continue
        blocks += 1
        with src.op("accept"):
            if src.bernoulli(1, blocks):
                res = list(range(i - k + 1, i + 1))
    return res


def s5g_scalar(src: _Source, n: int, k: int) -> list[int]:
    """PAIR-LOCK: a classical (k/2)-reservoir over consecutive PAIRS."""
    if k % 2:
        raise ValueError("PAIR-LOCK requires an even k")
    half = k // 2
    pairs = [(2 * j + 1, 2 * j + 2) for j in range(half)]
    for i in range(k + 2, n + 1, 2):
        pidx = i // 2
        with src.op("accept"):
            acc = src.bernoulli(half, pidx)
        if acc:
            with src.op("evict"):
                r = src.uniform_int(half)
            pairs[r] = (i - 1, i)
    return sorted(x for p in pairs for x in p)


def s6sys_scalar(src: _Source, n: int, k: int) -> list[int]:
    """Systematic sampling with a random start.  Requires n (oracle), by design."""
    if n % k:
        raise ValueError("oracle systematic requires k | n")
    b = n // k
    with src.op("representation"):
        o = src.uniform_int(b)
    return [o + 1 + j * b for j in range(k)]


# --------------------------------------------------------------------------- #
#  vectorised implementations (these are what M5/M6 measure)                   #
# --------------------------------------------------------------------------- #
def _shift_delete_append(sub: np.ndarray, r: np.ndarray, i: int, k: int) -> np.ndarray:
    """Delete column ``r`` of each row and append ``i`` -- keeps rows sorted."""
    cols = np.arange(k)
    take = cols[None, :] + (cols[None, :] >= r[:, None])
    np.minimum(take, k - 1, out=take)
    out = np.take_along_axis(sub, take, axis=1)
    out[:, k - 1] = i
    return out


def _vec_generic(
    gen: BitGen,
    n: int,
    k: int,
    T: int,
    *,
    kind: str,
) -> np.ndarray:
    """Per-item accept/evict samplers, vectorised over trials with compressed rows.

    Only the ACCEPTED rows are touched at each stream position, which turns the
    n*T work into (expected) k*ln(n/k)*T work plus one uniform per row per step.
    """
    res = np.tile(np.arange(1, k + 1, dtype=np.int64), (T, 1))
    rowsum = np.full(T, k * (k + 1) // 2, dtype=np.int64)
    accepts = np.zeros(T, dtype=np.int64)
    for i in range(k + 1, n + 1):
        den = i + 1 if kind == "S5f" else i
        u = gen.random_float64(T)
        idx = np.flatnonzero(u * den < k)
        m_acc = idx.size
        if m_acc == 0:
            continue
        if kind == "S0" or kind == "S5f":
            r = gen.random_below(m_acc, k)
            res[idx, r] = i
        elif kind in ("S1", "S1n", "S1h"):
            r = (rowsum[idx] - 1) % k
            if kind == "S1h" and (i - k) % k != 0:
                r = gen.random_below(m_acc, k)
            if kind == "S1":
                mm = i - k
                ss = mm % k
                if ss:
                    coin = gen.random_float64(m_acc) * mm < ss
                    if coin.any():
                        r_unif = gen.random_below(m_acc, k)
                        r = np.where(coin, r_unif, r)
            evicted = np.take_along_axis(res[idx], r[:, None], 1)[:, 0]
            res[idx] = _shift_delete_append(res[idx], r, i, k)
            rowsum[idx] += i - evicted
        elif kind == "S5a":
            r = accepts[idx] % k
            res[idx, r] = i
        elif kind == "S5b":
            res[idx] = _shift_delete_append(res[idx], np.zeros(m_acc, dtype=np.int64), i, k)
        elif kind == "S5c":
            res[idx, 0] = i
        else:  # pragma: no cover - guarded by the registry
            raise KeyError(kind)
        accepts[idx] += 1
    return res


def s0_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    return _vec_generic(gen, n, k, T, kind="S0")


def s1_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    return _vec_generic(gen, n, k, T, kind="S1")


def s1n_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    return _vec_generic(gen, n, k, T, kind="S1n")


def s5a_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    return _vec_generic(gen, n, k, T, kind="S5a")


def s5b_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    return _vec_generic(gen, n, k, T, kind="S5b")


def s5c_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    return _vec_generic(gen, n, k, T, kind="S5c")


def s5f_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    return _vec_generic(gen, n, k, T, kind="S5f")


def _np_log1mexp(x: np.ndarray) -> np.ndarray:
    """Vectorised stable log(1 - exp(x)) for x < 0."""
    out = np.empty_like(x)
    lo = x < -0.6931471805599453
    with np.errstate(divide="ignore", invalid="ignore"):
        out[lo] = np.log1p(-np.exp(x[lo]))
        out[~lo] = np.log(-np.expm1(x[~lo]))
    return out


def s2_vec(gen: BitGen, n: int, k: int, T: int, *, variant: str = "shipped") -> np.ndarray:
    """Algorithm L, vectorised over trials: ~k*ln(n/k) rounds, not n."""
    eps = 1.0 - 2.0**-53
    res = np.tile(np.arange(1, k + 1, dtype=np.int64), (T, 1))
    pos = np.full(T, k, dtype=np.int64)
    u0 = np.clip(gen.random_float64(T), 5e-324, eps)
    if variant == "vtrack":
        v = -np.expm1(np.log1p(-u0) / k)
    elif variant == "logdomain":
        logw = np.log1p(-u0) / k
    else:
        w = np.exp(np.log1p(-u0) / k)
    alive = np.ones(T, dtype=bool)
    rounds = 0
    while alive.any():
        rounds += 1
        if rounds > 4 * n + 1000:
            break
        act = np.flatnonzero(alive)
        u1 = np.clip(gen.random_float64(act.size), 5e-324, eps)
        with np.errstate(divide="ignore", invalid="ignore"):
            if variant == "vtrack":
                denom = np.log(v[act])
            elif variant == "logdomain":
                denom = _np_log1mexp(logw[act])
            else:
                denom = np.log(1.0 - w[act])
            skip = np.floor(np.log1p(-u1) / denom)
        skip = np.where(np.isfinite(skip) & (skip >= 0), skip, 0.0)
        newpos = pos[act] + skip.astype(np.int64) + 1
        pos[act] = newpos
        stop = newpos > n
        alive[act[stop]] = False
        live = act[~stop]
        if live.size:
            r = gen.random_below(live.size, k)
            res[live, r] = pos[live]
            u2 = np.clip(gen.random_float64(live.size), 5e-324, eps)
            if variant == "vtrack":
                vs = -np.expm1(np.log1p(-u2) / k)
                v[live] = v[live] + vs - v[live] * vs
            elif variant == "logdomain":
                logw[live] += np.log1p(-u2) / k
            else:
                w[live] = w[live] * np.exp(np.log1p(-u2) / k)
    return res


def s2f_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    return s2_vec(gen, n, k, T, variant="logdomain")


def s2v_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    return s2_vec(gen, n, k, T, variant="vtrack")


def s3_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    """Bottom-k: the k smallest of n i.i.d. uniform keys, per trial."""
    out = np.empty((T, k), dtype=np.int64)
    sub = max(1, min(T, int(2e7 // max(n, 1))))
    for lo in range(0, T, sub):
        hi = min(lo + sub, T)
        keys = gen.random_float64((hi - lo) * n).reshape(hi - lo, n)
        part = np.argpartition(keys, k - 1, axis=1)[:, :k]
        out[lo:hi] = np.sort(part, axis=1) + 1
        del keys, part
    return out


def s5e_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    """BLOCK-LOCK, vectorised: a 1-reservoir over the n//k complete blocks."""
    nb = n // k
    blk = np.ones(T, dtype=np.int64)
    for b in range(2, nb + 1):
        u = gen.random_float64(T)
        hit = u * b < 1.0
        blk[hit] = b
    base = (blk - 1) * k
    return base[:, None] + np.arange(1, k + 1, dtype=np.int64)[None, :]


def s5g_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    """PAIR-LOCK, vectorised: a classical (k/2)-reservoir over consecutive pairs."""
    if k % 2:
        raise ValueError("PAIR-LOCK requires an even k")
    half = k // 2
    pr = np.tile(np.arange(1, half + 1, dtype=np.int64), (T, 1))
    for pidx in range(half + 1, n // 2 + 1):
        u = gen.random_float64(T)
        idx = np.flatnonzero(u * pidx < half)
        if idx.size:
            r = gen.random_below(idx.size, half)
            pr[idx, r] = pidx
    out = np.empty((T, k), dtype=np.int64)
    out[:, 0::2] = 2 * pr - 1
    out[:, 1::2] = 2 * pr
    return np.sort(out, axis=1)


def s6sys_vec(gen: BitGen, n: int, k: int, T: int) -> np.ndarray:
    if n % k:
        raise ValueError("oracle systematic requires k | n")
    b = n // k
    o = gen.random_below(T, b)
    return o[:, None] + 1 + np.arange(k, dtype=np.int64)[None, :] * b


# --------------------------------------------------------------------------- #
#  exact transition kernels (consumed by exact.py)                             #
# --------------------------------------------------------------------------- #
def _children_uniform(state, i, k):
    slots = state[0]
    q = Fraction(1, k)
    for r in range(k):
        yield (tuple(slots[:r] + (i,) + slots[r + 1 :]), state[1]), q


def _children_sorted_uniform(state, i, k):
    slots = state[0]
    q = Fraction(1, k)
    for r in range(k):
        yield (slots[:r] + slots[r + 1 :] + (i,), state[1]), q


def _children_summodk(state, i, k, repair: bool):
    slots = state[0]
    m = i - k
    s = m % k
    r_det = (sum(slots) - 1) % k
    det_child = (slots[:r_det] + slots[r_det + 1 :] + (i,), state[1])
    if (not repair) or s == 0:
        yield det_child, Fraction(1)
        return
    lam = Fraction(s, m)
    acc: dict = {}
    acc[det_child] = acc.get(det_child, Fraction(0)) + (1 - lam)
    for r in range(k):
        ch = (slots[:r] + slots[r + 1 :] + (i,), state[1])
        acc[ch] = acc.get(ch, Fraction(0)) + lam * Fraction(1, k)
    for ch, q in acc.items():
        yield ch, q


def _children_hybrid(state, i, k):
    """Deterministic sum-mod-k when k | (i-k); uniform otherwise.

    Isolates WHICH half of the S1 rule is doing the work: if this kernel is
    exactly uniform at every prefix, the deterministic rank rule is provably
    correct on the steps where a deterministic balance is possible at all, and
    the only thing S1 got wrong is the mixture used on the remaining steps.
    """
    slots = state[0]
    if (i - k) % k == 0:
        r = (sum(slots) - 1) % k
        yield (slots[:r] + slots[r + 1 :] + (i,), state[1]), Fraction(1)
        return
    q = Fraction(1, k)
    for r in range(k):
        yield (slots[:r] + slots[r + 1 :] + (i,), state[1]), q


def _children_roundrobin(state, i, k):
    slots, aux = state
    r = aux[0]
    yield (tuple(slots[:r] + (i,) + slots[r + 1 :]), ((r + 1) % k,)), Fraction(1)


def _children_fifo(state, i, k):
    slots = state[0]
    yield (slots[1:] + (i,), state[1]), Fraction(1)


def _children_slot0(state, i, k):
    slots = state[0]
    yield ((i,) + slots[1:], state[1]), Fraction(1)


def _children_blocklock(state, i, k):
    yield (tuple(range(i - k + 1, i + 1)), state[1]), Fraction(1)


def _children_pairlock(state, i, k):
    slots = state[0]
    half = k // 2
    pairs = [slots[2 * j : 2 * j + 2] for j in range(half)]
    q = Fraction(1, half)
    for r in range(half):
        new = pairs[:r] + [(i - 1, i)] + pairs[r + 1 :]
        flat = tuple(sorted(x for p in new for x in p))
        yield (flat, state[1]), q


# --------------------------------------------------------------------------- #
class Candidate:
    """One sampler: scalar bit-counted, vectorised, and exact-kernel views."""

    def __init__(
        self,
        cid: str,
        label: str,
        *,
        scalar,
        vector,
        accept_num=lambda i, k: k,
        accept_den=lambda i, k: i,
        children=None,
        init_aux: tuple = (),
        step_filter=None,
        measurable=None,
        sorted_state: bool = True,
        needs_even_k: bool = False,
        oracle_n: bool = False,
        family: str = "",
        note: str = "",
    ) -> None:
        self.id = cid
        self.label = label
        self.scalar = scalar
        self.vector = vector
        self.accept_num = accept_num
        self.accept_den = accept_den
        self.children = children
        self.init_aux = init_aux
        # step_filter(i,k) -> bool : is item i a decision point at all?
        self.step_filter = step_filter or (lambda i, k: True)
        # measurable(i,k) -> bool : is the prefix-i distribution meant to be uniform?
        self.measurable = measurable or (lambda i, k: True)
        self.sorted_state = sorted_state
        self.needs_even_k = needs_even_k
        self.oracle_n = oracle_n
        self.family = family
        self.note = note


CANDIDATES: dict[str, Candidate] = {
    "S0": Candidate(
        "S0", "CLASSICAL-R", scalar=s0_scalar, vector=s0_vec,
        children=_children_sorted_uniform, family="baseline",
        note="Textbook Algorithm R (Vitter 1985). The BASELINE: provably uniform.",
    ),
    "S1": Candidate(
        "S1", "SUMMODK+REPAIR",
        scalar=s1_scalar, vector=s1_vec,
        children=lambda st, i, k: _children_summodk(st, i, k, True),
        family="main",
        note="MAIN hypothesis: eviction rank = (sum of reservoir - 1) mod k, "
             "randomised only on the fraction s/m of acceptances where "
             "s = (i-k) mod k != 0.",
    ),
    "S1n": Candidate(
        "S1n", "SUMMODK-NOREPAIR",
        scalar=s1n_scalar, vector=s1n_vec,
        children=lambda st, i, k: _children_summodk(st, i, k, False),
        family="control", note="Negative control and the zero-eviction-bits bound.",
    ),
    "S2": Candidate(
        "S2", "ALGO-L-float64", scalar=s2_scalar, vector=s2_vec,
        children=None, family="algol",
        note="Li (1994) Algorithm L exactly as shipped.",
    ),
    "S2f": Candidate(
        "S2f", "ALGO-L-logdomain", scalar=s2f_scalar, vector=s2f_vec,
        children=None, family="algol",
        note="Corrected Algorithm L: accumulates log W and evaluates log(1-W) "
             "with the stable log1mexp identity. Removes the W -> 0 "
             "division-by-zero and the 2**-53/W rate-quantisation floor.",
    ),
    "S2v": Candidate(
        "S2v", "ALGO-L-Vtrack", scalar=s2v_scalar, vector=s2v_vec,
        children=None, family="algol",
        note="The fix proposed in the hypothesis (track V = 1-W). Measured and "
             "kept because it targets the WRONG limit: W -> 1 is transient, the "
             "damaging limit is W -> 0 where V -> 1.",
    ),
    "S3": Candidate(
        "S3", "BOTTOM-K", scalar=s3_scalar, vector=s3_vec,
        children=None, family="baseline",
        note="A-Res / bottom-k with uniform weights: one full float PER ITEM.",
    ),
    "S5a": Candidate(
        "S5a", "ROUND-ROBIN", scalar=s5a_scalar, vector=s5a_vec,
        children=_children_roundrobin, init_aux=(0,), sorted_state=False,
        family="blindspot", note="Evict slot = (#acceptances) mod k.",
    ),
    "S5b": Candidate(
        "S5b", "FIFO", scalar=s5b_scalar, vector=s5b_vec,
        children=_children_fifo, sorted_state=False,
        family="blindspot", note="Evict the oldest held item.",
    ),
    "S5c": Candidate(
        "S5c", "SLOT0", scalar=s5c_scalar, vector=s5c_vec,
        children=_children_slot0, sorted_state=False,
        family="blindspot", note="Always evict slot 0.",
    ),
    "S5e": Candidate(
        "S5e", "BLOCK-LOCK",
        scalar=s5e_scalar, vector=s5e_vec,
        accept_num=lambda i, k: 1, accept_den=lambda i, k: i // k,
        children=_children_blocklock,
        step_filter=lambda i, k: i % k == 0,
        measurable=lambda i, k: i % k == 0,
        family="blindspot",
        note="Streaming systematic: a classical 1-reservoir over consecutive "
             "blocks of k items. Exact first-order inclusion k/i at every "
             "prefix divisible by k; support only i/k subsets.",
    ),
    "S5f": Candidate(
        "S5f", "OFFBYONE", scalar=s5f_scalar, vector=s5f_vec,
        accept_den=lambda i, k: i + 1,
        children=_children_sorted_uniform,
        family="blindspot", note="Accept with probability k/(i+1) instead of k/i.",
    ),
    "S5g": Candidate(
        "S5g", "PAIR-LOCK", scalar=s5g_scalar, vector=s5g_vec,
        accept_num=lambda i, k: k // 2, accept_den=lambda i, k: i // 2,
        children=_children_pairlock,
        step_filter=lambda i, k: i % 2 == 0,
        measurable=lambda i, k: i % 2 == 0,
        needs_even_k=True, family="blindspot",
        note="Classical (k/2)-reservoir over consecutive PAIRS: exact "
             "first-order marginals, support C(n/2, k/2).",
    ),
    "S6sys": Candidate(
        "S6sys", "SYSTEMATIC(oracle n)", scalar=s6sys_scalar, vector=s6sys_vec,
        children=None, oracle_n=True, family="blindspot",
        note="Textbook systematic sampling with a random start. Needs n, so it "
             "is NOT a reservoir sampler; included as the guaranteed "
             "first-order-exact / joint-degenerate exhibit (support n/k).",
    ),
}
CANDIDATES["S1h"] = Candidate(
    "S1h", "SUMMODK-HYBRID",
    scalar=lambda src, n, k: _summodk_hybrid_scalar(src, n, k),
    vector=lambda gen, n, k, T: _vec_generic(gen, n, k, T, kind="S1h"),
    children=_children_hybrid, family="main",
    note="Deterministic sum-mod-k on the steps where k | (i-k), uniform on the "
         "rest. Isolates which half of S1 is correct.",
)
CANDIDATES["S5d"] = CANDIDATES["S1n"]

#: samplers that get Monte-Carlo'd (S4/S4d are distributionally identical to S0)
MC_CANDIDATES = ("S0", "S1", "S1h", "S1n", "S2", "S2f", "S2v", "S3", "S5a", "S5b", "S5c", "S5e", "S5f", "S5g", "S6sys")
#: samplers that get bit-counted; S4/S4d are S0 under different regimes
BIT_CANDIDATES = ("S0", "S1", "S1h", "S1n", "S2", "S2f", "S2v", "S3", "S5e", "S5g")


def get_candidate(cid: str) -> Candidate:
    if cid not in CANDIDATES:
        raise KeyError(f"unknown candidate {cid!r}; known: {sorted(CANDIDATES)}")
    return CANDIDATES[cid]
