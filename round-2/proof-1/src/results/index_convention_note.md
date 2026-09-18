# Note for the paper's notation section: the index convention (retires defect D5)

**Convention adopted.** $i$ is the **prefix size**. $R_i$ is the reservoir after $i$ arrivals; the
arriving item is item number $i+1$ (1-indexed), i.e. index $i$ (0-indexed). In Lean, $[i]$ is
`Finset.range i = {0,…,i-1}` and
$$J(i,k)\;=\;\texttt{Finset.powersetCard k (Finset.range i)} .$$
Every divisibility condition in the paper is stated in this convention: the balanced condition is
$k \mid (i-k+1)$.

**Why the note is needed.** The run's own code disagrees with itself:

| file | indexes `i` as | writes the balanced condition as |
|---|---|---|
| `iter_1/gen_art/gen_art_experiment_1/fibermap.py` | the prefix size | `k ∣ (i-k+1)` |
| `iter_1/gen_art/gen_art_experiment_2/candidates.py` | the arriving item | `k ∣ (i-k)` |

**The translation lemma** (`index_convention`, `index_convention_nat` in `Anytime.lean`, both
sorry-free):

```lean
theorem index_convention (i k : ℤ) : k ∣ ((i + 1) - (i - k + 1))
theorem index_convention_nat {i k : ℕ} (hk1 : 1 ≤ k) (hki : k ≤ i) : (i - k + 1) + k = i + 1
```

The difference between the two indices is **exactly $k$**. Hence $(i-k+1)\equiv(i+1)\pmod k$, so
"$k$ divides $i-k+1$" (prefix-size convention) and "$k$ divides $i+1$" (arriving-item convention)
select **the same steps**. `verify_small_cases.py` checks the equivalence exhaustively for
$0\le i<60$, $1\le k<60$ (3 540 cases, 0 failures).

The integer version is stated over $\mathbb Z$ so that natural subtraction cannot truncate; the
$\mathbb N$ version carries $1\le k\le i$ explicitly for the same reason.
