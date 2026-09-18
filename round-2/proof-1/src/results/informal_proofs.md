# Paper-ready informal proofs

Every statement below **compiled sorry-free** in `Anytime.lean` (Lean 4 v4.14.0 + Mathlib) and
passes `#print axioms` with only the three standard Lean axioms. The informal proofs are written
to be pasted directly into the paper.

**Convention, fixed once.** $i$ is the *prefix size*: $R_i$ is the reservoir after $i$ arrivals,
and the arriving item is item number $i+1$ (1-indexed) $=$ index $i$ (0-indexed). We write
$[i]=\{0,\dots,i-1\}$ and $J(i,k)$ for the family of $k$-subsets of $[i]$, so $|J(i,k)|=\binom{i}{k}$.
$P_i$ denotes the law of $R_i$ on $J(i,k)$, and *uniform* means $P_i(S)=1/\binom{i}{k}$ for all $S$.

---

## Lemma 0 (index convention). `index_convention`, `index_convention_nat`

**Statement.** For all $i,k\in\mathbb Z$, $\;k \mid (i+1)-(i-k+1)$; equivalently, for
$1\le k\le i$ in $\mathbb N$, $(i-k+1)+k=i+1$.

**Proof.** $(i+1)-(i-k+1)=k$, and $k\mid k$. $\square$

**Why it is stated.** The difference between the prefix-size index $i-k+1$ and the arriving-item
index $i+1$ is exactly $k$, so the balancedness condition $k\mid(i-k+1)$ (prefix-size convention)
and the condition $k\mid(i+1)$ (arriving-item convention) select the *same* steps. The run's own
code uses both conventions in different files, so this one-liner retires the ambiguity; see
`results/index_convention_note.md`.

---

## Lemma 1a (Johnson biregularity). `L0a`, `L0a_real`

**Statement.** For $1\le k\le i$: $\;k\binom{i}{k}=(i-k+1)\binom{i}{k-1}$, and over $\mathbb R$
$$\frac{\binom{i}{k-1}}{\binom{i}{k}}=\frac{k}{i-k+1}.$$

**Proof.** Count the edges of the bipartite *Johnson graph* whose left vertices are $J(i,k)$,
whose right vertices are $J(i,k-1)$, and whose edges join $S$ to $T$ when $T\subset S$. Each
$S\in J(i,k)$ has exactly $k$ children $S\setminus\{x\}$, so the edge count is $k\binom{i}{k}$.
Each $T\in J(i,k-1)$ has exactly $i-(k-1)=i-k+1$ parents $T\cup\{y\}$ with $y\in[i]\setminus T$,
so the edge count is also $(i-k+1)\binom{i}{k-1}$. Dividing by $\binom{i}{k}>0$ gives the ratio.
$\square$

**Formalisation note.** In Lean this is `Nat.choose_succ_right_eq` after writing $k=m+1$; no
factorials are expanded. Biregularity is exactly what makes a transportation plan with uniform
child demand $(i-k+1)/k$ feasible — see Lemma 4.

---

## Lemma 1b (prefix ratio). `L0b`, `L0b_real`

**Statement.** For all $i,k$: $\;(i+1)\binom{i}{k}=(i+1-k)\binom{i+1}{k}$, and for $k\le i$
over $\mathbb R$
$$\frac{\binom{i}{k}}{\binom{i+1}{k}}=\frac{i+1-k}{i+1}.$$

**Proof.** Count pairs $(S,x)$ with $S\in J(i+1,k)$ and $x\in[i+1]\setminus S$: there are
$\binom{i+1}{k}(i+1-k)$ of them. Grouping instead by $x$ first, deleting $x$ from $[i+1]$ leaves a
set of size $i$ and $S$ is any of its $\binom{i}{k}$ $k$-subsets, giving $(i+1)\binom{i}{k}$.
$\square$

**Formalisation note.** One rewrite with `Nat.choose_mul_succ_eq`. Stated in $\mathbb N$, where
the Mathlib lemma lives, with a real-valued corollary in which the cast of $i+1-k$ is discharged
under $k\le i$; all downstream lemmas use only the real corollary.

---

## Lemma 2 (the accept coin is forced). `accept_forced`

**Statement.** Fix $k\le i$ and let $p:J(i,k)\to\mathbb R$ be any state-dependent acceptance
probability. Suppose that for every $S'\in J(i,k)$
$$\frac{1}{\binom{i}{k}}\bigl(1-p(S')\bigr)=\frac{1}{\binom{i+1}{k}} .$$
Then $p(S')=\dfrac{k}{i+1}$ for every $S'\in J(i,k)$ — in particular $p$ cannot depend on $S'$.

**Proof.** The hypothesis is the statement that the only way $R_{i+1}=S'$ with $i\notin S'$ is
that $R_i=S'$ already and the arriving item was rejected; since $R_{i+1}$ is either $R_i$ or $R_i$
with one element swapped, no other history contributes. Hence the linear system relating $P_{i+1}$
to $P_i$ restricted to the item-omitting states is *diagonal*: one unknown per equation. Solving,
$1-p(S')=\binom{i}{k}/\binom{i+1}{k}=(i+1-k)/(i+1)$ by Lemma 1b, so
$p(S')=1-\frac{i+1-k}{i+1}=\frac{k}{i+1}$, independent of $S'$. $\square$

**Why this matters.** This replaces an 11-cell numerical rank check. It is the statement that the
acceptance randomness of an anytime-uniform sampler is *not a design choice*: every exact sampler
spends it, in every state, which is what makes the Knuth–Yao floor of Theorem A unavoidable.

---

## Corollary 2.1 (the acceptance-randomness floor). `acc_floor`

**Statement.** Let $\mathrm{cost}(q)$ be the expected number of fair coin flips used by any
sampler to realise a Bernoulli$(q)$, and let $h(q)$ be the binary entropy in bits. *Assume*
Knuth–Yao: $h(q)\le\mathrm{cost}(q)$ for $0\le q\le 1$. Then for any run over prefixes
$i=k,\dots,n-1$ whose accept probabilities are the forced $k/(i+1)$,
$$\sum_{i=k}^{n-1} h\!\left(\frac{k}{i+1}\right)\;\le\;\sum_{i=k}^{n-1}\mathrm{cost}\!\left(\frac{k}{i+1}\right).$$

**Proof.** Termwise from the assumed Knuth–Yao bound, using $0\le k/(i+1)\le 1$, which holds
because $k\le i+1$ on the summation range. $\square$

**Scope.** Knuth–Yao optimality is an **explicit hypothesis** `hKY`, not an axiom: we do not
formalise the DDG-tree argument. See `results/assumed_not_proved.md`.

---

## Theorem C (any feasible plan preserves uniformity). `theoremC`, `theoremC_reject`, `theoremC_accept`

**Statement.** Fix $1\le k\le i$ and suppose $R_i$ is uniform on $J(i,k)$. Let
$f:J(i,k)\times J(i,k-1)\to\mathbb R$ be any *routing plan* satisfying

* (rows) $\sum_{T\in J(i,k-1)} f(S,T)=1$ for every $S\in J(i,k)$ — $f$ is a probability kernel;
* (columns) $\sum_{S\in J(i,k)} f(S,T)=\dfrac{i-k+1}{k}$ for every $T\in J(i,k-1)$.

Let the arriving item be accepted with probability $k/(i+1)$ (forced, Lemma 2), and on acceptance
let the evicted-to state be $T$ with probability $f(R_i,T)$, so that $R_{i+1}=T\cup\{i\}$. Then
$R_{i+1}$ is uniform on $J(i+1,k)$.

**Proof.** Let $S'\in J(i+1,k)$.

*Reject branch* ($i\notin S'$). Then $S'\subseteq[i]$, so $S'\in J(i,k)$ and the only contributing
history is $R_i=S'$ followed by a rejection:
$$P_{i+1}(S')=\Bigl(1-\tfrac{k}{i+1}\Bigr)\frac{1}{\binom{i}{k}}
=\frac{i+1-k}{i+1}\cdot\frac{1}{\binom{i}{k}}=\frac{1}{\binom{i+1}{k}},$$
the last step being exactly Lemma 1b.

*Accept branch* ($i\in S'$). Put $T:=S'\setminus\{i\}$; then $T\subseteq[i]$ and $|T|=k-1$, so
$T\in J(i,k-1)$. Summing over the previous reservoir,
$$P_{i+1}(S')=\frac{k}{i+1}\sum_{S\in J(i,k)}\frac{1}{\binom{i}{k}}f(S,T)
=\frac{k}{i+1}\cdot\frac{1}{\binom{i}{k}}\cdot\frac{i-k+1}{k}
=\frac{i+1-k}{(i+1)\binom{i}{k}}=\frac{1}{\binom{i+1}{k}},$$
again by Lemma 1b, the factor $k$ cancelling because $k\ge 1$. $\square$

**Two structural observations the formalisation makes precise.**
(i) The row condition is **never used** in the proof: uniformity of the next prefix is a
*column-sum condition only*. The rows are needed solely for $f$ to be a probability kernel.
(ii) No support condition on $f$ is used either, so the theorem is in fact more general than the
"evict one element" phrasing suggests.

**Why this matters.** Every exactness claim in the run was previously "no counterexample found
below $n=18$, $k\le 5$". This makes the flow-repaired sampler, the table sampler, and classical
Algorithm R correct **by construction**, for all $(n,k)$.

---

## Theorem C.1 (the feasible polytope is never empty). `theoremC_existence`, `uniformPlan_row`, `uniformPlan_col`, `card_parents`

**Statement.** For every $1\le k\le i$ the plan
$$f_{\mathrm{unif}}(S,T)=\begin{cases}1/k,& T\subseteq S\\ 0,&\text{otherwise}\end{cases}$$
satisfies both the row and the column condition of Theorem C. Hence a feasible plan always exists,
and Theorem C is never vacuous.

**Proof.** *Rows.* For $S\in J(i,k)$, the $T\in J(i,k-1)$ with $T\subseteq S$ are exactly the
$(k-1)$-subsets of $S$, of which there are $\binom{k}{k-1}=k$; each contributes $1/k$, so the row
sum is $1$. *Columns.* For $T\in J(i,k-1)$, the $S\in J(i,k)$ with $T\subseteq S$ are exactly the
sets $T\cup\{y\}$, $y\in[i]\setminus T$, and $y\mapsto T\cup\{y\}$ is injective on $[i]\setminus T$,
so there are $i-(k-1)=i-k+1$ of them; each contributes $1/k$, giving $(i-k+1)/k$. $\square$

**Remark.** $f_{\mathrm{unif}}$ *is* the textbook Algorithm-R eviction rule ("drop a uniformly
random one of the $k$ current elements"). So Theorem C re-derives the correctness of Algorithm R
as a special case, and it exhibits the textbook rule as an explicit point of the transportation
polytope — a constructive witness, no Birkhoff or Hall needed.

---

## Lemma 3 (forest-support split count). `forest_split_count`

**Statement.** Let a bipartite support have $A$ left vertices (sources) of degrees
$d_1,\dots,d_A\ge 1$ and $E=\sum_j d_j$ edges, with $B$ right vertices, and suppose
$E+1\le A+B$. Then the number of sources of degree $\ge 2$ satisfies
$$\bigl|\{\,j : d_j\ge 2\,\}\bigr| + 1 \;\le\; B .$$

**Proof.** Split the degree sum by whether $d_j\ge 2$:
$$E=\sum_{d_j\ge 2} d_j+\sum_{d_j=1} d_j \;\ge\; 2\,|\mathrm{Split}| + \bigl(A-|\mathrm{Split}|\bigr) = A+|\mathrm{Split}| .$$
Combining with $E+1\le A+B$ gives $A+|\mathrm{Split}|+1\le A+B$, i.e. $|\mathrm{Split}|+1\le B$. $\square$

**The geometric input we do not formalise.** The eviction objective $\sum_S P(S)H(q_S)$ is concave
in the plan, so its minimum over the transportation polytope is attained at a **vertex**; the
support of a vertex of a transportation polytope is a **forest** on the $A+B$ bipartite vertices,
hence has at most $A+B-1$ edges (network-simplex spanning-tree basis; standard). That is precisely
the hypothesis $E+1\le A+B$. We formalise only its arithmetic consequence, which is all the bound
uses. Stated in truncation-free additive form `card + 1 ≤ B` so that natural subtraction cannot bite.

---

## Lemma 4 (max entropy). `entropy_le_log_card`, `entropy_le_logb_card`

**Statement.** Let $q\ge 0$ on a finite set $s$ with $\sum_{a\in s}q(a)=1$. Then
$$H(q):=\sum_{a\in s}-q(a)\log q(a)\;\le\;\log|s| \qquad\text{(nats)},$$
and dividing by $\log 2$, $H_2(q)\le\log_2|s|$ (bits).

**Proof.** Let $t=\{a\in s: q(a)>0\}$; since $-0\log 0=0$, restricting to $t$ changes nothing.
For $a\in t$, apply $\log x\le x-1$ at $x=1/(|s|\,q(a))>0$ and multiply by $q(a)\ge 0$:
$$-q(a)\bigl(\log|s|+\log q(a)\bigr)\;\le\;q(a)\Bigl(\frac{1}{|s|\,q(a)}-1\Bigr)=\frac{1}{|s|}-q(a),$$
i.e. $-q(a)\log q(a)\le q(a)\log|s|+\tfrac1{|s|}-q(a)$. Summing over $a\in t$ and using
$\sum_{a\in t}q(a)=1$ and $|t|\le|s|$,
$$H(q)\;\le\;\log|s|+\frac{|t|}{|s|}-1\;\le\;\log|s| . \qquad\square$$

**Route taken (as required by the plan).** Self-contained elementary $\log x\le x-1$ argument,
**not** Mathlib's measure-theoretic `measureEntropy_le_log_card` (which would drag a `PMF`/`Measure`
into the file) and **not** Jensen via `ConcaveOn.le_map_sum`. The only Mathlib analysis input is
`Real.log_le_sub_one_of_pos`, plus `Real.negMulLog_zero` to discard the zero atoms.

---

## Theorem B, per-acceptance core. `per_step_entropy_bound`, `per_step_entropy_bound_ratio`, `per_step_entropy_bound_ratio_strict`, `per_step_entropy_cap`, `per_step_entropy_min`

**Statement.** Write $A=\binom{i}{k}$, $B=\binom{i}{k-1}$, let the source mass be uniform
($P(S)=1/A$), let each source's eviction coupling $q_S$ be supported on at most $k$ atoms
(so $H(q_S)\le\log_2 k$ by Lemma 4), let $q_S$ be a point mass ($H=0$) at every non-split source,
and let the number of split sources satisfy $|\mathrm{Split}|+1\le B$ (Lemma 3). Then
$$H_{\mathrm{evict}}(i,k):=\sum_S \frac1A H(q_S)\;\le\;\log_2(k)\,\frac{B-1}{A}
\;\le\;\log_2(k)\,\frac{k}{i-k+1},$$
with the second inequality **strict** when $k\ge 2$; and always $H_{\mathrm{evict}}(i,k)\le\log_2 k$.
Hence the usable bound is
$$H_{\mathrm{evict}}(i,k)\;\le\;\min\Bigl\{\log_2 k,\;\log_2(k)\frac{k}{i-k+1}\Bigr\}.$$

**Proof.** Split the sum over $\mathrm{Split}$ and its complement. Non-split terms vanish by
hypothesis. Each split term is at most $\frac1A\log_2 k$ by Lemma 4. Therefore
$H_{\mathrm{evict}}\le\frac{|\mathrm{Split}|}{A}\log_2 k\le\frac{B-1}{A}\log_2 k$ by Lemma 3.
Finally $\frac{B-1}{A}<\frac{B}{A}=\frac{k}{i-k+1}$ by Lemma 1a and $A>0$ (with $\le$ rather than
$<$ when $k=1$, where both sides are $0$ because $\log_2 1=0$). The ceiling
$H_{\mathrm{evict}}\le\log_2 k$ is immediate from $H(q_S)\le\log_2 k$ and $|{\rm sources}|=A$. $\square$

**What the proof does not use (recorded by Lean's unused-binder linter).** `per_step_entropy_bound`
never consumes `hsrc` (that the number of sources equals $\binom{i}{k}$) or `hHnn` (that
$H(q_S)\ge0$): the bound follows from the split/non-split decomposition and Lemma 3 alone. The
hypotheses are kept in the statement because the paper's Theorem B should carry them, but the
inequality is in fact more general. This is the same kind of observation as Theorem C's unused row
condition. See `results/compiler_warnings.md`.

**Sanity checks (both machine-checked as Lean `example`s).**
* At $k=2$ the bound is $\log_2(2)\frac{\binom{i}{1}-1}{\binom{i}{2}}=\frac{2}{i}$ bits, while the
  exactly solved per-step minima reported by the run are $1,\frac13,\frac15,\frac17$ at
  $i=2,4,6,8$, i.e. $\log_2(k)/(i-k+1)=1/(i-1)$. Since $\frac2i\ge\frac1{i-1}$ for every $i\ge2$,
  the bound is **never violated**; the slack is $2(i-1)/i\to 2$, a factor of about two. The bound
  and the solved minima agree in sign and order; there is no discrepancy to report.
* At $k=1000$, $i=k+1$, the bound returns the ceiling $\log_2 1000 = 9.9658$ bits. The fitted law
  it replaces predicted $11.27$ bits per acceptance there — above a ceiling it cannot exceed. The
  proved bound cannot do this, because the $\min$ with $\log_2 k$ is part of the theorem.

---

## Lemma 5 (telescoping). `telescope_bound`

**Statement.** For every $m$, $\displaystyle\sum_{j=0}^{m-1}\frac{1}{(j+1)(j+2)}\le 1$.

**Proof.** $\frac{1}{(j+1)(j+2)}=\frac1{j+1}-\frac1{j+2}$, so the partial sum telescopes to
$1-\frac{1}{m+1}<1$. (Formalised by induction on $m$.) $\square$

---

## Theorem B, total (constant in $n$). `total_constant_in_n`

**Statement.** For every $k\ge1$ and every $n$,
$$\sum_{i=k}^{n-1}\frac{k}{i+1}\cdot\log_2(k)\frac{k}{i-k+1}\;\le\;\log_2(k)\,k^2 .$$

**Proof.** Reindex $i=k+j$, $j=0,\dots,n-k-1$; then $i+1=k+j+1$ and $i-k+1=j+1$, so the $j$-th
term is
$$\frac{\log_2(k)\,k^2}{(k+j+1)(j+1)}\;\le\;\frac{\log_2(k)\,k^2}{(j+2)(j+1)},$$
using $k\ge1$. By Lemma 5 the sum of $\frac{1}{(j+1)(j+2)}$ over any range is at most $1$, and
$\log_2(k)k^2\ge0$, so the total is at most $\log_2(k)k^2$ — a bound **independent of $n$**. $\square$

**Correction to the hypothesis text (reported, not buried).** The hypothesis asserted that the
total is $O(k\log k)$. It is **not** — see the sharp evaluation below, which is $\Theta(k\log^2 k)$.
Only the constant-in-$n$ claim — the load-bearing one — survives, and it survives under either bound.

---

## Theorem B, sharp total. `harm`, `harm_le_harm_add`, `harm_le_self`, `harm_shift`, `sum_inv_prod_le_harm_div`, `total_sharp`, `total_sharp_le_easy`

**Statement.** Let $H_k=\sum_{r=1}^{k}1/r$. For every $k\ge1$ and every $n$,
$$\sum_{i=k}^{n-1}\frac{k}{i+1}\cdot\log_2(k)\frac{k}{i-k+1}\;\le\;\log_2(k)\,k\,H_k
\;\le\;\log_2(k)\,k^2 .$$
Since $H_k=\Theta(\log k)$, the sharp bound is $\Theta(k\log^2 k)$ — strictly better than the
$O(k^2\log k)$ of `total_constant_in_n`, and still **not** the $O(k\log k)$ the hypothesis asserted.

**Proof.** Reindex $i=k+j$ as before; the $j$-th term is $\log_2(k)k^2\cdot\frac{1}{(j+1)(j+1+k)}$.
Partial fractions give
$$\frac{1}{(j+1)(j+1+k)}=\frac1k\Bigl(\frac1{j+1}-\frac1{j+1+k}\Bigr),$$
so, writing $H_m=\sum_{j<m}\frac{1}{j+1}$ and using $\sum_{j<m}\frac{1}{j+1+k}=H_{k+m}-H_k$
(a re-indexing of the harmonic sum),
$$\sum_{j<m}\frac{1}{(j+1)(j+1+k)}=\frac{1}{k}\bigl(H_m-(H_{k+m}-H_k)\bigr)\;\le\;\frac{H_k}{k},$$
because $H_m\le H_{k+m}$ (the harmonic numbers are monotone). Multiplying by $\log_2(k)k^2\ge0$
gives $\log_2(k)\,k\,H_k$. Finally $H_k\le k$ (every term is at most $1$), which recovers the
easy bound $\log_2(k)k^2$ of `total_constant_in_n`. $\square$

**Consistency.** Numerically (`verify_small_cases.py`): at $k=50$ the truncated series is
$1269.64$ and $\log_2(k)kH_k=1269.64$, against the easy bound $\log_2(k)k^2=14109.64$; the
exact-value agreement is to 4 decimal places at $k=2,3,10,50$.

---

## Headline corollary. `anytime_uniform_step`

**Statement.** For every $1\le k\le i$ there exists a routing plan $f$ that is a genuine
probability kernel and for which, starting from a uniform $R_i$, the reservoir $R_{i+1}$ is
exactly uniform on $J(i+1,k)$.

**Proof.** Take $f=f_{\mathrm{unif}}$ from Theorem C.1 and apply Theorem C. $\square$

**Reading.** Together with Lemma 2 this is the full anatomy of one step of an anytime-uniform
sampler: the *accept* coin is forced to $k/(i+1)$ and costs what Knuth–Yao says it costs; the
*eviction* plan is free to range over a non-empty transportation polytope; and uniformity of the
next prefix constrains only the plan's column sums.
