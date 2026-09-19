# Checking if others already solved this sampling idea

## Summary

Prior-art and source-code audit for the claim that reservoir sampling's eviction randomness is unnecessary. VERDICT: GO-WITH-RESCOPE. (Q1) Draper & Saad SODA 2026 (arXiv:2505.18879v5) is a PARTIAL_SCOOP, not a kill: the word 'reservoir' appears 0 times in 476k characters, and their Eq. 1.7/Remark 1.3 make clear they optimise HOW a chosen distribution is realised, not WHICH distributions must be realised -- but their Remark 1.4/Corollary 1.7 DO cover the reservoir's state-dependent sequence, so they must be cited as the realisation layer. Bhattacharya et al. (arXiv:1407.1689) settle k=1 in the eps-error model at Theta(log(n/eps)) and prove bounded-bit exactness impossible; Breitner's comment thread settles exact k=1; Bodini & Durand (GASCom 2026) settle the OFFLINE k-subset case at log2 C(n,k)+O(k). k>=2 streaming is OPEN. (Q2) The balanced fiber map is textbook (normalized matching / de Bruijn SCD) -- claim the application only; exact deterministic balance is IMPOSSIBLE unless k | (i-k+1); and the per-step min-entropy coupling has NO polynomial special-case theorem (Kovacevic Thm 3.8 keeps row-sparse MEC strongly NP-hard). (Q3) The folklore claim is REFUTED: 0 of 15 quotations from 12 sources assert necessity (10 sufficiency-only, 5 descriptive) -- delete the 'overturns a textbook false belief' contribution. Ships a 16-row deployed-implementation table with verbatim eviction lines, RNG granularity and test-suite grades; 6 real bug reports including Apache DataSketches #647, an OPEN joint-distribution-only defect that per-item tests cannot detect. Own deterministic arithmetic (bit_accounting.py) refutes alternate hypothesis #4 for k>=10: the eviction term is 18-34% of total variate entropy at k=10 and 34-59% at k=100, so recycling alone does not subsume the coupling claim -- but the claimed '25-61%' holds only for 10<=k<=100, and the every-prefix constraint costs ~8x more than the final answer's own entropy.

## Research Findings

**Bottom line: GO-WITH-RESCOPE.** The random-bit question for *exact,
every-prefix-uniform, k>=2* reservoir sampling is genuinely open, and the one
serious threat -- Draper & Saad's randomness recycling (SODA 2026) -- is a
PARTIAL_SCOOP that leaves the headline alive while becoming a mandatory
citation. But three of the hypothesis's supporting claims do not survive
contact with the primary sources, and one of them ("every textbook says uniform
eviction is necessary") must be deleted outright.

**Q1. Is the bit question open? Yes -- and Draper & Saad are the tool, not the
result.** Their model covers a sequence of distributions P_1, P_2, ... "subject
to any stochastic process", and Eq. 1.7 makes the scope precise: "P_i cannot
depend on fresh coins to be consumed by the sampler in future rounds, and that
it cannot depend on the internal history of the sampler other than through its
generated outputs X_{<i}" [2]. A reservoir's state *is* a deterministic function
of its own past outputs, so Remark 1.4 and Corollary 1.7 apply to reservoir
sampling directly -- their machinery would drive Algorithm R's cost to
H[X_1,...,X_n] + eps*n + W [2]. What it does **not** do is decide *which*
distributions the algorithm should realise. Remark 1.3 says so explicitly: "it
is essential to consider the surprisals and entropies of the random
distributions (P_1,...,P_n), and not the joint entropy H[X_1,...,X_n] of the
output sequence" [2]. The word "reservoir" appears **zero times** in the 476k
characters of the v5 HTML, "streaming"/"unknown length" zero times, and the
only k-subset-adjacent application is the Fisher-Yates shuffle [1, 2]; the
authors' C library README likewise never mentions reservoir, subset,
without-replacement or shuffle [3]. Verdict: **PARTIAL_SCOOP**. The paper must
cite them as the realisation layer and claim only the coupling/eviction novelty.

This also settles the fight with alternate hypothesis #4 ("the waste is in the
variates, not the coupling"). Our own arithmetic (`bit_accounting.py`,
deterministic, no RNG) sums the Shannon entropy of the variates Algorithm R
*chooses* to draw: the accept coin H_b(k/i) plus log2(k) per acceptance. The
eviction term's share of the total is 6-8% at k=2, 11-22% at k=4, 18-34% at
k=10, 27-48% at k=32, 34-59% at k=100 and 47-78% at k=1000 (over n = 1e4..1e9).
So recycling applied to classical Algorithm R recovers the *representational*
waste but leaves the eviction term entirely intact -- the coupling argument is
not subsumed. Two honest caveats fall out of the same table. First, the
hypothesis's headline band of "25-61%" is only attained for moderate-to-large k;
at k=2-4 the achievable saving is a fifth of that, and the paper must state the
(k, n) regime. Second, even the deterministic-eviction ideal is far above the
entropy of the final answer: at k=10, n=1e6 it costs ~1494 bits versus
log2 C(n,k) = 177.5 bits. The every-prefix constraint, not the eviction rule, is
the dominant cost -- a framing the paper should adopt rather than hide.

Two further prior-art findings matter. Bhattacharya, Issac, Jaiswal & Kumar
prove Theta(log(n/eps)) random bits for *epsilon-error* streaming sampling and,
critically, prove exactness is unattainable with bounded bits at all: "assume
that a uniform sample can be generated using r random bits ... This means that
2^r is divisible by n. This is a contradiction since n is a prime number" [4].
They also state "the lower bound on the number of random bits remains the same
if the sampling algorithm is allowed to store more than one item from the
stream" [4] -- but their sample size is 1 throughout; storing k items to output
1 is not sampling a k-subset, so the k>=2 *output* case is untouched.
ADJACENT, not a scoop. Second, Bodini & Durand solve the **offline** k-subset
problem entropically: "the expected net bit consumption (bits consumed minus
bits recycled) is log2 C(n,k) + O(k)" in O(n) time [10], building on the
random-bit model of Lumbroso [12] and the same group's set-partition work [11]. That is the offline
analogue of our target and must be cited; it makes the streaming/unknown-n
version the actual open cell. Fourteen further scoop-hunt queries (logged in
`research_report.md`) returned nothing combining random bits with k>=2 streaming
sampling. Attribution corrections: Algorithms X, Y, Z are all
Vitter 1985 TOMS (no 1987 paper); Vitter 1984 CACM holds A-D for known N [6, 8];
Li 1994 holds Algorithm L [9]; and the k=1 closed form is due to the comment
thread on Breitner's post, not to Breitner, who left it open [5].

**Q2. The mathematics is known in spirit, and one claim is outright wrong.**
The Boolean lattice's normalized matching property is textbook -- "It is easy to
show that 2^[n] is a rank-symmetric, unimodal normalized matching poset" [13] --
and de Bruijn/Tengbergen/Kruyswijk symmetric chain decompositions come with
three explicit constructions including the parenthesis-matching rule [14]. Claim
the *application*, not the object. Two corrections. (a) An **exactly** balanced
deterministic deletion map cannot exist in general: each (k-1)-subset of [i]
must receive C(i,k)/C(i,k-1) = (i-k+1)/k preimages, an integer only when
k | (i-k+1), so residual randomness is forced at most steps. Wagon & Wilf give
the sharp criterion for the natural sum-mod-k variant -- uniform iff for every
divisor d>1 of m, t mod d > n mod d -- which fails for most (n,t) [15]. (b) The claim that
the per-step minimum-entropy coupling is "small and exactly solvable" has **no
citable theorem behind it**. MEC is NP-hard (Kovacevic et al., Subset-Sum
reduction) and stays strongly NP-hard under row-sparsity: their Theorem 3.8
reduces 3-Partition to a coupling "having exactly one nonzero entry in every
row" [16]. All polynomial results are additive approximations -- 1 bit
(Cicalese et al. [17]), 0.53/1.22 bits (Compton et al. [18]), and a PTAS
polynomial only for constant m and eps [19]. ARIMEC's conditional guarantee
does not apply here [20], and the forest-support structure of optimal couplings
leaves C(mn, m+n-1) candidates [21]. Verdict: BRUTE_FORCE_ONLY, leaning
UNRESOLVED for our specific uniform-marginal Johnson-graph support.

**Q3. The folklore claim is not evidenced -- delete it.** Across 15 verbatim
quotations from 12 sources (Wikipedia, Vitter 1985, Stanford CS168, Weizmann,
Utah, Cornell, cs.stackexchange, StackOverflow, and four blog expositions),
**bucket A (strict necessity) = 0**, bucket B (sufficiency only) = 10, bucket C
(descriptive) = 5. Vitter writes "The candidate it replaces is chosen randomly
from the n candidates" and then "It is easy to see that the resulting set of n
candidates forms a random sample" [7] -- description plus sufficiency, never
necessity. Wikipedia's proof ends "By induction, Algorithm R produces a uniform
random sample of the inputs" [22]. Nobody claims an alternative fails -- not Stanford CS168 [29], Weizmann [24],
Utah [30], Cornell [26], cs.stackexchange [23], StackOverflow [31], nor the
blog expositions [25, 27, 28, 32]. The
"overturns a textbook false belief" contribution must be replaced by the true,
still-interesting statement: *uniform eviction is universally taught and
universally deployed, proofs are sufficiency-only, and the necessity question is
simply never asked.*

The 16-row source audit supports that softer claim strongly. Every deployed
*streaming* sampler evicts uniformly: Spark `(rand.nextDouble() * l).toLong`
[33], rust-random `rng.random_range(..i + 1 + amount)` [39], ClickHouse
`genRandom(total_values)` with visible `% limit` modulo bias [35], go-metrics
`rand.Int63n(s.count)` [48], gstamatelat `sample.set(random.nextInt(sampleSize),
item)` [40], sklearn `j = rng_randint(0, i + 1)` [43]. The priority-queue family
(Flink, Beam, DuckDB) spends a full float64 per element instead [45, 44, 37]. Non-streaming contrast rows -- NumPy [42], Redis [50],
PostgreSQL [51], TensorFlow's shuffle buffer [49] -- and Guava, which ships no
reservoir sampler at all [46], bound how far the folklore reaches.
Algorithm-L skip code uses naive `Math.log(1 - W)` [40] and `log(r) / log(t_w)`
[37], never log1p/expm1 -- direct support for alternate hypothesis #2. Spark's
`GapSampling` is the lone `math.log1p(-f)` user [33].

**Alternate hypothesis #1 needs rescoping, not killing.** It is *not* true that
test suites only check first-order statistics: CPython asserts every ordered
permutation appears [41], sklearn asserts every C(n,k) frozenset appears [43],
commons-rng runs a chi-square over all 10 2-subsets [47]. But all three are
offline samplers on tiny n. Every *streaming* sampler audited is
first-order-only or has no distributional test at all [33, 35, 37, 39, 40, 48].
And there is a real, unpatched joint-only defect in the wild: Apache
DataSketches #647 -- "for items 0 to k-1, they can only ever appear at that
specific index" -- correct marginals, wrong joint [53]. Meanwhile Spark's own
reviewer wrote "the test suite never actually tests for correctness, just basic
input/output sizes" [34], and four of five confirmed fixed bugs (TiKV, Spark,
DuckDB, dataprof) were first-order and none was caught by existing tests
[52, 34, 38, 54]; the sixth confirmed bug, a ClickHouse self-merge aliasing
fault, was likewise found by an automated reviewer, not by tests [36].

**Ranked re-assessment.** (1) Main hypothesis, rescoped to "bit cost of exact
every-prefix-uniform k>=2 sampling, with Draper-Saad as the realisation layer" --
GO. (2) Alternate #1, rescoped to *streaming* samplers, with DataSketches #647
as a real-world target -- GO, strong. (3) Alternate #2 (Algorithm-L float
defect) -- GO, evidence gathered. (4) Alternate #4 -- REFUTED by the arithmetic
above for k >= 10.

## Sources

[1] [Efficient Online Random Sampling via Randomness Recycling (arXiv abstract page)](https://arxiv.org/abs/2505.18879) (Thomas L. Draper, Feras A. Saad; 2026) — Establishes the identity, version history (v1 24 May 2025, v5 7 May 2026) and venue (SODA 2026, pp. 2473-2511) of the primary scoop candidate, plus the abstract's comparison set (Knuth-Yao 1976, Han-Hoshi 1997, Kozen-Soloviev 2022, Shao-Wang 2025) and its stated applications.

> On the practical side, we develop randomness recycling techniques to accelerate a variety of prominent sampling algorithms. We show that randomness recycling enables state-of-the-art runtime performance on the Fisher-Yates shuffle when using a cryptographically secure pseudorandom number generator, and that it reduces the entropy cost of discrete Gaussian sampling.

Locator: Abstract, second paragraph

[2] [Efficient Online Random Sampling via Randomness Recycling (full HTML, v5)](https://arxiv.org/html/2505.18879v5) (Thomas L. Draper, Feras A. Saad; 2026) — The decisive source for the Q1 scoop verdict. Grepped in full (476,303 characters): 'reservoir' 0 hits, 'streaming'/'data stream'/'unknown length' 0 hits. Supplies Eq. 1.7 (the scope of the model), Remark 1.3 (entropies of P_i, not of the output sequence), Remark 1.4 and Corollary 1.7 (state-dependent sequences are covered), Theorem 1.5 and Conjecture 1.6.

> cannot depend on fresh coins to be consumed by the sampler in future rounds, and that it cannot depend on the internal history of the sampler other than through its generated outputs

Locator: Section 1.1, after Eq. 1.7

> A common case for online sampling is generating an output sequence

Locator: Remark 1.4, opening

> We conjecture that the space-entropy combination in Theorem 1.5 is the best possible, up to constant factors and dependence on

Locator: Before Conjecture 1.6

[3] [probsys/randomness-recycling -- reference C implementation (README)](https://raw.githubusercontent.com/probsys/randomness-recycling/master/README.md) (Thomas L. Draper, Feras A. Saad; 2026) — Checked for any reservoir or k-subset example. The README lists inversion, lookup-table, alias, Fast Loaded Dice Roller and Amplified Loaded Dice Roller samplers; grep for reservoir|subset|without replacement|shuffle returns zero hits. Corroborates the NO-reservoir finding at the artifact level. (Default branch is master; the main branch path 404s.)

> These samplers use a technique called randomness recycling to
> reduce the expected amortized entropy consumption over many samples.

Locator: Opening paragraph

[4] [Sampling in Space Restricted Settings](https://arxiv.org/pdf/1407.1689) (Anup Bhattacharya, Davis Issac, Ragesh Jaiswal, Amit Kumar; 2015) — The closest existing bit-complexity result: Theta(log(n/eps)) random bits for maintaining a size-ONE uniform sample over a stream of unknown length, plus the impossibility of exact uniform sampling with bounded bits, plus the Omega(n log n) cost of naive reservoir sampling and the Omega((log n)^2) cost of Vitter's method.

> It is important
> to point out that the lower bound on the number of random bits remains the same if the sampling
> algorithm is allowed to store more than one item from the stream.

Locator: Section 1, contributions

> This means that 2r is divisible by n. This is a contradiction since n is a prime number.

Locator: Section 1, impossibility of exact bounded-bit sampling

> We will discuss a more advanced sampling technique by Vitter that requires Ω((log n)2) random bits in expectation.

Locator: Footnote 3

[5] [Reservoir sampling with few random bits (blog post and comment thread)](https://www.joachim-breitner.de/blog/737-Reservoir_sampling_with_few_random_bits) (Joachim Breitner, Benjamin Price, Christoph Pfister; 2018) — The k=1 precedent. The post (2018-03-25) gives an interval/arithmetic-coding sampler and leaves the entropy question OPEN; comment #1 (Benjamin Price, 2018-04-18) gives the Theta((log n)^2) lower bound and the key insight that the cost is the entropy of the SEQUENCE of kept elements, not of the final answer; comment #2 (Christoph Pfister, 2019-01-31) gives the closed form, the Knuth-Yao H+2 optimum and the H+3 bound for the post's procedure.

> Is this algorithm optimal? (And what does it mean to be optimal?)

Locator: Open questions

> The process described doesn’t just choose one person, but chooses a sequence of people, and saves the last one picked.

Locator: Comment #1, Benjamin Price, 2018-04-18

> Thus, the following result is obtained:

Locator: Comment #2, Christoph Pfister, 2019-01-31, before 'Performance'

[6] [Random Sampling with a Reservoir (ACM TOMS 11(1):37-57) -- UMD mirror](https://www.cs.umd.edu/~samir/498/vitter.pdf) (Jeffrey Scott Vitter; 1985) — Pins the fact that the classical analysis is in VARIATES and CPU TIME, never bits (Table I), and that Algorithms R, X, Y and Z are all in this one 1985 paper. Also gives Vitter's own statement of the Algorithm R eviction rule and of the reservoir invariant.

> Table I. 
> Performance 
> of Algorithms 
> R, X, Y, and Z

Locator: p. 38

> Algorithm R (which is is a reservoir algorithm due to Alan Waterman) works

Locator: Section 2

> The random number 
> generator RANDOM 
> returns a real number in the unit interval.

Locator: Section 2, Algorithm R description

[7] [Random Sampling with a Reservoir (ACM TOMS 11(1):37-57) -- author-hosted mirror](https://www.ittc.ku.edu/~jsv/Papers/Vit85.Reservoir.pdf) (Jeffrey Scott Vitter; 1985) — Author-hosted copy used for the two necessity-bucket quotations from Vitter's Algorithm R description: the eviction rule (bucket C) and the sufficiency assertion that follows it (bucket B). Confirms no necessity claim is made.

> The candidate it replaces is chosen randomly from the n candidates.

Locator: Section 2, Reservoir Algorithms and Algorithm R

> It is easy to see that the resulting set of n candidates forms a random sample of the first t + 1 records.

Locator: Section 2, immediately after the eviction-rule sentence

[8] [Faster methods for random sampling (CACM 27(7):703-718)](https://dl.acm.org/doi/10.1145/358105.893) (Jeffrey Scott Vitter; 1984) — Used to pin the year/venue correction: this 1984 CACM paper, not the 1985 TOMS paper, contains Algorithms A, B, C and D, and addresses the KNOWN-N sequential sampling problem rather than reservoir sampling.

[9] [Reservoir-sampling algorithms of time complexity O(n(1+log(N/n))) (ACM TOMS 20(4):481-493)](https://dl.acm.org/doi/10.1145/198429.198435) (Kim-Hung Li; 1994) — Source of Algorithm L. Abstract page reachable and free-access; the PDF yielded one comparison sentence before dl.acm.org began returning HTTP 403, so the Algorithm L pseudocode line is marked NOT VERIFIED and documented instead from deployed Algorithm-L implementations.

> Reservoir-sampling algorithms of time complexity

Locator: Title

[10] [Entropic Generation of Binary Words (GASCom 2026, EPTCS 445:38-46)](https://arxiv.org/pdf/2606.13157) (Olivier Bodini, Francis Durand; 2026) — Previously uncited and important: the OFFLINE analogue of this paper's target. Solves uniform k-subset generation in the random-bit model at log2 C(n,k) + O(k) net bits in O(n) time via 'random bit recycling', and cites Draper-Saad as the concurrent online counterpart. Makes the streaming/every-prefix cell the actual open problem.

> The uniform generation of k Hamming weight binary words, equivalent to sampling k-subsets from n
> elements, relies on random bits, which can be expensive.

Locator: Abstract

> It is worth noting that the broader paradigm of entropy recycling in random generation is concurrently being explored in some different frameworks

Locator: Our Contribution

[11] [Optimal Random Bit Complexity in Efficient Sampling of Set Partition-Like Structures (LNCS, 2025)](https://doi.org/10.1007/978-3-031-98740-3_29) (Olivier Bodini; 2025) — Located during the scoop hunt as further evidence that random-bit-optimal combinatorial sampling is an active line (Bodini and co-authors). Covers set-partition-like structures, not streaming k-subsets; recorded as landscape, not a scoop.

[12] [Optimal Discrete Uniform Generation from Coin Flips, and Applications](https://doi.org/10.48550/arxiv.1304.1916) (Jeremie Lumbroso; 2013) — The random-bit-model background for generating discrete uniforms at near-entropy cost; cited by Bodini-Durand as one of the 'optimal generation algorithms [10, 12]' their bound relies on. Establishes that the per-draw realisation problem was already solved before recycling.

[13] [Decompositions of the Boolean lattice into rank-symmetric chains (Electron. J. Combin. 23(2) #P2.53)](https://www.combinatorics.org/ojs/index.php/eljc/article/download/v23i2p53/pdf/) (Istvan Tomon; 2016) — Supplies the textbook definition of a normalized matching graph/poset and the statement that the Boolean lattice is one, plus the attribution of symmetric chain decompositions to de Bruijn, Tengbergen & Kruyswijk and of the extension to Griggs. This is the KNOWN_IN_PRINT citation for the paper's 'balanced fiber map'.

> A bipartite graph G = (A, B, E) is a normalized matching graph, if for any X ⊂A we
> have

Locator: Section 1, definitions

> It is easy to show that 2[n] is a rank-symmetric, unimodal normalized matching poset.

Locator: Section 1, after the definitions

[14] [Hypercube problems, Lecture 17: Symmetric chain decomposition (Charles University)](http://ktiml.mff.cuni.cz/~gregor/hypercube/lecture17.pdf) (Petr Gregor, Ondrej Micka; 2017) — Three explicit SCD constructions written out in full -- recursive, greedy lexicographic matching, and the Greene-Kleitman-style parenthesis pairing -- establishing that explicit level-to-level constructions in the Boolean lattice are standard teaching material.

> A symmetric chain decomposition (SCD) is a collection C of disjoint symmetric chains that covers Bn.

Locator: Definition 1

> To get symmetric chain containing X, we simply deﬁne how to “go up” and “go down”
> along the chain.

Locator: Proof (Pairings of 1's and 0's)

[15] [When are subset sums equidistributed modulo m? (Electron. J. Combin. 1(1) #R3)](https://www.combinatorics.org/ojs/index.php/eljc/article/view/v1i1r3) (Stan Wagon, Herbert S. Wilf; 1994) — The sharp characterisation of when the sum-of-elements-mod-m statistic is exactly uniform on t-subsets of [n]. Directly relevant: it shows the natural 'sum mod k' balanced-deletion construction is exactly uniform only on a thin set of (n,t,m), independently confirming that exact deterministic balance is unavailable at most reservoir steps.

> The obvious necessary condition, that $m$ divides ${n\choose t}$, is not sufficient, but a $q$-analogue of that condition is both necessary and sufficient

Locator: Abstract

[16] [On the Entropy of Couplings](https://arxiv.org/html/1303.3235) (Mladen Kovacevic, Ivan Stanojevic, Vojin Senk; 2015) — The original NP-hardness theorem for minimum-entropy coupling (Thm 3.4, Subset-Sum reduction, hard even when one marginal has only two masses) and -- decisively for this paper -- Thm 3.8, which reduces 3-Partition to a ROW-SPARSE coupling problem, showing sparsity alone does not buy tractability.

> Theorem 3.4

Locator: Section III

> Theorem 3.8

Locator: Section III

[17] [Minimum-Entropy Couplings and their Applications](https://arxiv.org/html/1901.07530) (Ferdinando Cicalese, Luisa Gargano, Ugo Vaccaro; 2019) — The classical greedy 1-additive polynomial-time guarantee for m=2 (Theorem 2), plus the separate observation that finding a minimum-SUPPORT coupling is itself NP-hard. Used as the arXiv stand-in because the iris.unisa.it journal PDF returned a Cloudflare 403.

> Theorem 2

Locator: Section III-C

[18] [Minimum-Entropy Coupling Approximation Guarantees Beyond the Majorization Barrier](https://arxiv.org/html/2302.11838) (Spencer Compton, Dmitriy Katz, Benjamin Qi, Kristjan Greenewald, Murat Kocaoglu; 2023) — Confirms the NP-hardness attribution to Kovacevic et al. 2015, states that membership in NP is itself open, notes the LP has n^m variables, and gives the improved additive guarantees of 0.53 bits (m=2) and 1.22 bits (general m).

> It was posed as an open problem in [Kovačević et al., 2015] whether the problem is in NP.

Locator: Section 5, Computing Exactly Optimal Solutions

[19] [Efficient eps-approximate minimum-entropy couplings](https://arxiv.org/html/2509.19598v1) (Spencer Compton; 2025) — The current state of the art: an eps-additive PTAS whose running time is polynomial in n only for CONSTANT numbers of marginals m and constant eps. Establishes that no general exact polynomial algorithm is known, and notes that even exactly computing x log(1/x) in polynomial time is open.

> Computing the minimum-entropy coupling is NP-hard

Locator: Abstract

[20] [Computing Low-Entropy Couplings for Large-Support Distributions (ARIMEC)](https://arxiv.org/html/2405.19540) (Samuel Sokota, Dylan Sam, Christian Schroeder de Witt, Spencer Compton, Jakob Foerster, J. Zico Kolter; 2024) — Shows the field's own framing of the capability gap for large/structured-support coupling, and that ARIMEC's polynomial guarantee is conditional on being able to compute maximum-entropy posterior partitions efficiently -- an assumption that is not established for the Johnson-graph deletion support this paper needs.

> As a result, at the time of writing, there exist no techniques for producing low-entropy couplings of general large-support distributions

Locator: Section 1, Introduction

[21] [An Explicit Description of Extreme Points of the Set of Couplings with Given Marginals](https://ar5iv.labs.arxiv.org/html/2505.12227) (2025) — Shows the optimal coupling's support is always a forest with at most m+n-1 nonzeros, but that the number of such extreme points is bounded only by C(mn, m+n-1) -- so the structural fact alone yields no polynomial algorithm for a fixed sparse support.

[22] [Reservoir sampling (Wikipedia)](https://en.wikipedia.org/wiki/Reservoir_sampling) (2026) — The canonical tutorial source for the folklore claim, and the survey article whose reference list was checked during the scoop hunt. Supplies two necessity-bucket quotations: a descriptive lead (bucket C) and the inductive sufficiency conclusion (bucket B). No necessity assertion anywhere.

> By induction, Algorithm R produces a uniform random sample of the inputs.

Locator: Proof of correctness of Algorithm R

[23] [Returning a random subset with length k of n strings while only storing at most k (cs.stackexchange Q28123)](https://cs.stackexchange.com/questions/28123/returning-a-random-subset-with-length-k-of-n-strings-while-only-storing-at-most) (2014) — Top-voted community explanation of reservoir sampling. Classified bucket B: states the outcome the procedure achieves without any claim that an alternative eviction rule would fail.

[24] [Sublinear Time and Space Algorithms 2024A, Lecture 6: Reservoir Sampling and l0-sampling (Weizmann)](https://www.wisdom.weizmann.ac.il/~robi/teaching/2024a-SublinearAlgorithms/lecture6.pdf) (Robert Krauthgamer; 2024) — Graduate lecture notes. The uniformity statement is a Lemma whose proof is left as an exercise -- sufficiency only, necessity unaddressed.

[25] [Reservoir Sampling (samwho.dev)](https://samwho.dev/reservoir-sampling/) (Sam Rose; 2025) — A widely read interactive exposition. Its only 'must' restates the definition of fairness as the proof target, not a necessity claim about the eviction mechanism -- classified bucket B after strict review.

[26] [Reservoir sampling (Cornell ORIE 6125 course notes)](https://people.orie.cornell.edu/prs233/orie-6125/algorithms/reservoir-sampling.html) — University course notes; describes the eviction step with 'the appropriate probability' and attaches no correctness claim to it -- bucket C.

[27] [Reservoir Sampling (Richard Startin's blog)](https://richardstartin.github.io/posts/reservoir-sampling) (Richard Startin; 2020) — A detailed practitioner exposition that explicitly works through the proof step using the uniform eviction index. Supplies two bucket-B quotations and is the closest any source comes to naming the eviction rule as load-bearing -- while still never claiming necessity.

[28] [Reservoir Sampling (florian.github.io)](https://florian.github.io/reservoir-sampling/) (2019) — Popular exposition covering the k>1 case; justifies the k/i acceptance probability from the uniform-sampling target. Bucket B.

[29] [CS168: The Modern Algorithmic Toolbox, Lecture 13 (Stanford)](https://web.stanford.edu/class/cs168/l/l13.pdf) (Tim Roughgarden, Gregory Valiant; 2024) — States the k>=1 reservoir uniformity claim as an explicit proof target for induction -- the clearest sufficiency-only framing in a major course, and evidence that necessity is not taught either way.

[30] [Models of Computation for Massive Data, L7: Sampling (University of Utah)](https://users.cs.utah.edu/~jeffp/teaching/MCMD/S7.2-sampling.pdf) (Jeff M. Phillips; 2013) — Graduate streaming-algorithms notes; states the inclusion probability descriptively with no claim attached to the choice of evicted slot. Bucket C.

[31] [Reservoir sampling: why is it selected uniformly at random (Stack Overflow Q29131567)](https://stackoverflow.com/questions/29131567/reservoir-sampling-why-is-it-selected-uniformly-at-random) (2015) — A question whose very title asks the 'why uniform' question; the quoted proof excerpt computes the 1/N result for the k=1 rule without arguing any alternative fails. Bucket B.

[32] [Reservoir Sampling: Sample Any Stream Without Seeing Its End (SpaceComplexity blog)](https://spacecomplexity.ai/blog/reservoir-sampling-algorithm) (2026) — A recent (2026) exposition included to test whether modern write-ups assert necessity. It states 'evicts one slot chosen uniformly' as a caption with no correctness claim attached -- bucket C.

[33] [Apache Spark -- SamplingUtils.reservoirSampleAndCount (source)](https://github.com/apache/spark/blob/79f5f281bb69cb2de9f64006180abd753e8ae427/core/src/main/scala/org/apache/spark/util/random/SamplingUtils.scala) — The most widely deployed Algorithm R in data infrastructure. Verbatim eviction line, 53-bit double per element, and the only log1p usage found in the audit (in the unrelated GapSampling path).

[34] [[SPARK-18678][ML] Skewed reservoir sampling in SamplingUtils (pull request)](https://github.com/apache/spark/pull/16129) (Sean Owen; 2016) — A real, fixed, first-order bias bug (replacement probability k/(l-1) instead of k/l) found by code audit, and the source of the audit's single strongest testing quote: the reviewer's observation that the test suite never tested correctness at all.

[35] [ClickHouse -- ReservoirSampler.h (source)](https://github.com/ClickHouse/ClickHouse/blob/16b167a7e4e831ef41374b77402764447e70f24d/src/AggregateFunctions/ReservoirSampler.h) — Algorithm R backing ClickHouse quantile aggregates, with an explicit modulo-bias RNG ('return rng() % limit') and a fixed default seed. No dedicated unit test exists for the sampler in src/AggregateFunctions/tests/.

[36] [ClickHouse issue #101782 -- ReservoirSamplerDeterministic::merge() missing self-merge guard](https://github.com/ClickHouse/ClickHouse/issues/101782) (2026) — A maintainer-confirmed user-visible correctness bug found by an automated review bot rather than the test suite; classified 'other' (memory aliasing) rather than a distributional defect.

[37] [DuckDB -- base_reservoir_sample.cpp (source)](https://github.com/duckdb/duckdb/blob/a7dd104bdaed36d973a7a2bf7bc6d1ba75c81e6b/src/execution/sample/base_reservoir_sample.cpp) — A-Res/Algorithm-L-family weighted reservoir. Direct evidence for alternate hypothesis #2: the skip recurrence is 'log(r) / log(t_w)' with no log1p or expm1 anywhere in the file, and the replacement key carries only ~32 bits of entropy.

[38] [DuckDB issue #18099 -- Reservoir Sampling is broken since DuckDB 1.2.0](https://github.com/duckdb/duckdb/issues/18099) (2025) — A real, user-discovered, fixed first-order bug caused by treating max-order statistics of uniforms as uniform rather than Beta(n,1), compounded by round instead of ceil. Exactly the numerical failure mode alternate hypothesis #2 predicts.

[39] [rust-random/rand -- IteratorRandom::sample / choose_multiple (source)](https://github.com/rust-random/rand/blob/ea71ad8475e5e75cb769220107c418556fbdf19e/src/seq/iterator.rs) — The Rust ecosystem's Algorithm R. Unbiased bounded-integer draw per element; tests are golden-value regressions with no distributional assertion for the k>1 path.

[40] [gstamatelat/random-sampling -- AbstractRandomSampling / LiLSampling (source)](https://github.com/gstamatelat/random-sampling/blob/30f5b6e19a01186c9e184a14bc9b13285eec98ad/src/main/java/gr/james/sampling/AbstractRandomSampling.java) — Shared eviction site for Waterman (R) and LiL (L). Supplies the naive 'Math.log(random1) / Math.log(1 - W)' skip computation and the repository's first-order-only correctness test over up to 4,000,000 repetitions.

[41] [CPython -- Lib/random.py, Random.sample (source)](https://raw.githubusercontent.com/python/cpython/main/Lib/random.py) — Contrast row (offline, n known). Also the strongest test in the audit: test_sample_distribution asserts every ORDERED permutation tuple appears across 10,000 trials at n=5 -- evidence against the unqualified form of alternate hypothesis #1.

[42] [NumPy -- Generator.choice with replace=False (source)](https://raw.githubusercontent.com/numpy/numpy/main/numpy/random/_generator.pyx) — Contrast row: Floyd's algorithm with an open-addressed hash set, requiring pop_size upfront. All replace=False tests are fixed-seed exact-value regressions with no distributional assertion.

[43] [scikit-learn -- sklearn/utils/_random.pyx, sample_without_replacement (source)](https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/sklearn/utils/_random.pyx) — Contains a literal Algorithm R ('reservoir_sampling' method) applied over an index array, and a genuine joint test: check_sample_int_distribution asserts every C(n,k) frozenset appears over 10,000 trials at n=10.

[44] [Apache Beam -- Top.BoundedHeap, used by Sample.FixedSizedSampleFn (source)](https://raw.githubusercontent.com/apache/beam/master/sdks/java/core/src/main/java/org/apache/beam/sdk/transforms/Top.java) — Bottom-k by uniform random key in a bounded min-heap: deterministic eviction of the minimum, one full 32-bit key per element. The most bit-wasteful family audited; tests assert only size and subset membership.

[45] [Apache Flink -- ReservoirSamplerWithoutReplacement (source, release-1.20)](https://raw.githubusercontent.com/apache/flink/release-1.20/flink-java/src/main/java/org/apache/flink/api/java/sampling/ReservoirSamplerWithoutReplacement.java) — A true streaming A-Res sampler spending a full nextDouble() per element as the priority key. Its Kolmogorov-Smirnov test checks the marginal value distribution, not joint inclusion. Package was removed in Flink 2.x with the DataSet API.

[46] [Google Guava -- Streams.java (source, checked for reservoir sampling)](https://raw.githubusercontent.com/google/guava/master/guava/src/com/google/common/collect/Streams.java) — Negative result: Guava ships no reservoir sampler. Zero matches for reservoir|sample across Streams.java, Iterables.java and Iterators.java; its only k-of-n utility is the deterministic Ordering.leastOf top-k selector.

[47] [Apache Commons RNG -- SubsetSamplerUtils / ListSampler (source)](https://raw.githubusercontent.com/apache/commons-rng/master/commons-rng-sampling/src/main/java/org/apache/commons/rng/sampling/SubsetSamplerUtils.java) — Contrast row: a bounded partial Fisher-Yates requiring n upfront; commons-rng ships no Reservoir class. Its ListSamplerTest runs a chi-square goodness-of-fit test over all 10 possible 2-subsets of {0..4} -- a genuine joint test at tiny n.

[48] [rcrowley/go-metrics -- UniformSample (source)](https://raw.githubusercontent.com/rcrowley/go-metrics/master/sample.go) — The Go ecosystem's Algorithm R, with an in-source citation of Vitter 1985. One 63-bit bounded draw per element; tests are fixed-seed golden values plus count/size/range checks.

[49] [TensorFlow -- shuffle_dataset_op.cc, tf.data shuffle buffer (source)](https://raw.githubusercontent.com/tensorflow/tensorflow/master/tensorflow/core/kernels/data/shuffle_dataset_op.cc) — Contrast row: a sliding-window shuffle buffer that emits every element exactly once, not a fixed-size uniform sample. Tests assert multiset equality and per-value repeat counts only.

[50] [Redis -- srandmemberWithCountCommand in src/t_set.c (source)](https://raw.githubusercontent.com/redis/redis/unstable/src/t_set.c) — Contrast row showing why reservoir sampling is unnecessary when O(1) random access exists: Redis uses additive or subtractive rejection sampling into a scratch dict depending on the set/count ratio.

[51] [PostgreSQL -- contrib/tsm_system_rows/tsm_system_rows.c (source)](https://raw.githubusercontent.com/postgres/postgres/master/contrib/tsm_system_rows/tsm_system_rows.c) — Contrast row: a widely deployed 'sampling' feature that spends exactly two RNG draws per scan (start block, relatively-prime stride) and has no per-row randomness at all.

[52] [TiKV issue #19849 -- load-based split reservoir sampling is biased by an off-by-one error](https://github.com/tikv/tikv/issues/19849) (2026) — A real, fixed, first-order bug: sampling from 0..n instead of 0..=n meant observation n was admitted with probability sample_num/n instead of sample_num/(n+1). Found by manual code review; the fix PR added test coverage that had not existed.

[53] [Apache DataSketches issue #647 -- Reservoir sampling improvements](https://github.com/apache/datasketches-java/issues/647) (2025) — The audit's single most valuable finding for alternate hypothesis #1: an open, unpatched, maintainer-authored JOINT-distribution-only defect in reservoir merge. Marginals remain correct while stored positions stay correlated with arrival order, so per-item frequency tests cannot detect it by construction.

[54] [dataprof PR #650 -- make accumulator merge produce the single-pass profile](https://github.com/AndreaBozzo/dataprof/pull/650) (2026) — A real, fixed, first-order merge bug: pooling two reservoirs and subsampling uniformly ignored population sizes, so a 100-row partition took ~52 of 100 slots instead of ~1. Fixed with a hypergeometric draw; found by self-audit, not by tests.

## Verification

Numbered citations resolve to unique listed sources. Passage checks test text occurrence, not claim truth or entailment. Author/year metadata and locators are not independently verified. Details: `research_verification.json`.

- Source [1]: text found — On the practical side, we develop randomness recycling techniques to accelerate a variety of promine
- Source [2]: text found — cannot depend on fresh coins to be consumed by the sampler in future rounds, and that it cannot depe
- Source [2]: text found — A common case for online sampling is generating an output sequence
- Source [2]: text found — We conjecture that the space-entropy combination in Theorem 1.5 is the best possible, up to constant
- Source [3]: text found — These samplers use a technique called randomness recycling to
reduce the expected amortized entropy 
- Source [4]: text found — It is important
to point out that the lower bound on the number of random bits remains the same if t
- Source [4]: text found — This means that 2r is divisible by n. This is a contradiction since n is a prime number.
- Source [4]: text found — We will discuss a more advanced sampling technique by Vitter that requires Ω((log n)2) random bits i
- Source [5]: text found — Is this algorithm optimal? (And what does it mean to be optimal?)
- Source [5]: text found — The process described doesn’t just choose one person, but chooses a sequence of people, and saves th
- Source [5]: text found — Thus, the following result is obtained:
- Source [6]: text found — Table I. 
Performance 
of Algorithms 
R, X, Y, and Z
- Source [6]: text found — Algorithm R (which is is a reservoir algorithm due to Alan Waterman) works
- Source [6]: text found — The random number 
generator RANDOM 
returns a real number in the unit interval.
- Source [7]: text found — The candidate it replaces is chosen randomly from the n candidates.
- Source [7]: text found — It is easy to see that the resulting set of n candidates forms a random sample of the first t + 1 re
- Source [9]: UNVERIFIED — Reservoir-sampling algorithms of time complexity
- Source [10]: text found — The uniform generation of k Hamming weight binary words, equivalent to sampling k-subsets from n
ele
- Source [10]: text found — It is worth noting that the broader paradigm of entropy recycling in random generation is concurrent
- Source [13]: text found — A bipartite graph G = (A, B, E) is a normalized matching graph, if for any X ⊂A we
have
- Source [13]: text found — It is easy to show that 2[n] is a rank-symmetric, unimodal normalized matching poset.
- Source [14]: text found — A symmetric chain decomposition (SCD) is a collection C of disjoint symmetric chains that covers Bn.
- Source [14]: text found — To get symmetric chain containing X, we simply deﬁne how to “go up” and “go down”
along the chain.
- Source [15]: text found — The obvious necessary condition, that $m$ divides ${n\choose t}$, is not sufficient, but a $q$-analo
- Source [16]: text found — Theorem 3.4
- Source [16]: text found — Theorem 3.8
- Source [17]: text found — Theorem 2
- Source [18]: text found — It was posed as an open problem in [Kovačević et al., 2015] whether the problem is in NP.
- Source [19]: text found — Computing the minimum-entropy coupling is NP-hard
- Source [20]: text found — As a result, at the time of writing, there exist no techniques for producing low-entropy couplings o
- Source [22]: text found — By induction, Algorithm R produces a uniform random sample of the inputs.

## Follow-up Questions

- What is the exact minimum of H(S_k, S_{k+1}, ..., S_n) over all couplings whose marginals are uniform on the k-subsets of every prefix -- is the accept-coin term sum_i H_b(k/i) actually a lower bound, or can a coupling that correlates the accept decision with the reservoir contents beat it? Our arithmetic assumes the accept coin is irreducible; nothing found proves it.
- Is minimum-entropy coupling with uniform marginals on consecutive Johnson-graph levels (support = element deletion) polynomial-time solvable, or strongly NP-hard? Kovacevic et al. Thm 3.8 shows row-sparsity alone does not help, but the uniform-marginal, biregular, transitive-automorphism-group structure here is far more special than anything in that reduction.
- Does Draper-Saad recycling applied on top of classical Algorithm R actually attain sum_i [H_b(k/i) + (k/i)log2 k] in practice, and at what value of eps and what auxiliary-state size? Their C library implements the recycling primitives but ships no reservoir example, so this needs to be built and measured before the paper can quote a real end-to-end saving.
- Would a second-order (pairwise co-occurrence) test actually have caught Apache DataSketches #647 at realistic sample sizes, and what is its power relative to the first-order tests that Spark, Flink and gstamatelat ship? That is the concrete, falsifiable experiment alternate hypothesis #1 now has a real target for.

---
*Generated by AI Inventor Pipeline*
