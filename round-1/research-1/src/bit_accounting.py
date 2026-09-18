"""Deterministic arithmetic for the research report (no network, no RNG, no LLM).

For every-prefix-uniform k-subset reservoir sampling on a stream of length n:

  Algorithm R, step i (i>k):  accept w.p. k/i            -> H_b(k/i) bits
                              if accepted, evict slot ~ U[k] -> log2(k) bits
  Deterministic-eviction variant (what the hypothesis proposes): accept coin only.

Both realise the same every-prefix-uniform marginals.  Draper & Saad
(arXiv:2505.18879v5, Thm 1.5 / Cor 1.7) make the *realisation* of a chosen
variate sequence cost within eps of its Shannon entropy; it does not change
WHICH distributions the algorithm decides to realise.  So the sums below are the
floors that randomness recycling converges to, per scheme, and their ratio is
the head-room the coupling argument could still claim on top of recycling.

log2 C(n,k) is the entropy of the final k-subset alone -- a lower bound on any
chain, shown for reference.
"""
from math import log2, lgamma, log

def hb(p):
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * log2(p) - (1 - p) * log2(1 - p)

def harmonic(a, b):
    """sum_{i=a..b} 1/i, exact for small ranges, Euler-Maclaurin beyond."""
    if b - a < 2_000_000:
        return sum(1.0 / i for i in range(a, b + 1))
    def H(m):  # H_m
        return (log(m) + 0.5772156649015329 + 1.0 / (2 * m)
                - 1.0 / (12 * m * m) + 1.0 / (120 * m ** 4))
    return H(b) - H(a - 1)

def accept_entropy(n, k, cap=2_000_000):
    """sum_{i=k+1..n} H_b(k/i).  Exact term-by-term up to cap; beyond cap use
    H_b(p) = p*log2(1/p) + p*log2(e) + O(p^2) with p=k/i, integrated exactly."""
    hi = min(n, cap)
    s = sum(hb(k / i) for i in range(k + 1, hi + 1))
    if n > cap:
        # tail: sum_{i=cap+1..n} [ (k/i)log2(i/k) + (k/i)log2(e) ] + O(k^2/i^2)
        # sum (k/i)log2(i) ~ k/ln2 * (ln^2 n - ln^2 cap)/2
        s += (k / log(2)) * ((log(n) ** 2 - log(hi) ** 2) / 2.0)
        s += k * (log2(1.0 / k) + log2(2.718281828459045)) * harmonic(hi + 1, n)
    return s

def evict_entropy(n, k):
    return 0.0 if k <= 1 else k * log2(k) * harmonic(k + 1, n)

def lchoose(n, k):
    return (lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)) / log(2.0)

print(f"{'k':>6} {'n':>12} {'accept_bits':>13} {'evict_bits':>13} {'total':>13} "
      f"{'evict_share':>12} {'log2C(n,k)':>12}")
for k in (2, 4, 10, 32, 100, 1000):
    for n in (10**4, 10**6, 10**9):
        if n <= k: continue
        a, e = accept_entropy(n, k), evict_entropy(n, k)
        print(f"{k:>6} {n:>12} {a:>13.3f} {e:>13.3f} {a+e:>13.3f} "
              f"{e/(a+e):>12.4f} {lchoose(n,k):>12.3f}")

print()
print("k=1 cross-check of the Breitner-comment closed form")
print("  H = sum_{i=2..n} H_b(1/i)  vs  log2(n) + sum_{j=2..n} (1/j)log2(j-1)"
      "  vs  (ln n)^2/(2 ln 2)")
for n in (10**3, 10**5, 10**6):
    direct = sum(hb(1 / i) for i in range(2, n + 1))
    closed = log2(n) + sum((1 / j) * log2(j - 1) for j in range(2, n + 1))
    print(f"  n={n:>9}  direct={direct:.6f}  closed={closed:.6f}  "
          f"asymptote={(log(n)**2)/(2*log(2)):.6f}")
