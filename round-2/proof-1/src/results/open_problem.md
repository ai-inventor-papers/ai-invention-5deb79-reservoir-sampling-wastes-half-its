# The open problem the theorems bracket

The formalised results bracket the anytime-uniform reservoir sampler from two sides, and the gap
between them is a clean, sharply stated open question.

**Upper side — what exists.** `theoremC` together with `theoremC_existence` and
`anytime_uniform_step` show that for every $1\le k\le i$ there is a routing plan whose column sums
are the uniform child demand $(i-k+1)/k$, and that *any* such plan keeps every prefix exactly
uniform. `per_step_entropy_min` and `total_constant_in_n` then show that if the eviction coupling
at each step is chosen at a vertex of that polytope — so that its support is a forest and at most
$\binom{i}{k-1}-1$ sources are "split" — the eviction randomness per acceptance is at most
$\min\{\log_2 k,\ \log_2(k)\,k/(i-k+1)\}$ bits, and the **total over the whole stream is bounded by
$\log_2(k)\,k^2$, independently of $n$** (sharp constant $\log_2(k)\,k\,H_k=\Theta(k\log^2 k)$).
So an anytime-uniform sampler whose total eviction randomness does not grow with $n$ **exists**.

**Lower side — what is cheap.** `accept_forced` shows the acceptance coin is pinned to $k/(i+1)$ in
*every* state, so by Knuth–Yao (assumed) the acceptance randomness is a hard floor that no design
can avoid; it is the eviction randomness, and only the eviction randomness, that is negotiable.

**The gap.** The existence result above is non-constructive in the operationally relevant sense: it
says a vertex-supported plan attains the bound, not that one can be *found and sampled from* in
$O(1)$ time and $O(1)$ extra space per arrival. The only known $O(1)$-time, $O(1)$-space
construction — the hybrid rule $H$ — is exactly balanced only on the steps satisfying
$k \mid (i-k+1)$, i.e. on a $\sim 1/k$ fraction of steps (`index_convention` shows this is the same
set as $k\mid(i+1)$), and therefore recovers only a $\sim 1/k$ fraction of the eviction budget the
bound allows. The textbook plan $f_{\mathrm{unif}}$, by contrast, is always feasible but spends the
full $\log_2 k$ bits at every acceptance, i.e. the trivial ceiling.

> **Open problem.** Is there an eviction rule that, at every step $i$, runs in $\mathrm{poly}(k)$
> time and $O(k)$ space, keeps every prefix exactly uniform, and attains the per-step minimum
> eviction entropy — equivalently, attains a total of $O(k\log^2 k)$ bits rather than
> $\Theta(n)$ bits?

**Naming the gap in both directions.**
* *Upper direction.* The bound $\log_2(k)(\binom{i}{k-1}-1)/\binom{i}{k}$ is proved but loose: at
  $k=2$ it exceeds the exactly solved per-step minima by a factor tending to $2$. Closing that
  factor requires knowing the entropy of the *optimal* vertex, not just that a vertex has forest
  support. That is a minimum-entropy-coupling question, and the relevant hardness results (row-sparse
  MEC is strongly NP-hard) mean no polynomial special-case theorem is currently known.
* *Lower direction.* There is no lower bound at all on the eviction randomness of an $O(1)$-space
  sampler. Nothing here rules out the possibility that $O(1)$ space *forces* $\Theta(n)$ eviction
  bits, in which case the hybrid $H$'s $\sim1/k$ recovery would be close to optimal and the open
  problem would have a negative answer. Proving such a space–randomness trade-off — or exhibiting
  the $\mathrm{poly}(k)$-time rule — is the decisive next step.
