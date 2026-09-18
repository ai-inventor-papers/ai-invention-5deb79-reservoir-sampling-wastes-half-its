#!/usr/bin/env python3
"""Render research_report.md from research_out.json plus the auditable query log."""
from __future__ import annotations
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
d = json.loads((OUT / "research_out.json").read_text(encoding="utf-8"))

QUERY_LOG = [
    ("randomness-efficient sampling without replacement", "general", "5 results (ECCC reports, property-testing review, Stanford seminar page). No relevant results -- all hits use 'randomness-efficient sampler' in the extractor/expander sense."),
    ("random bit complexity of sampling a k-subset", "general", "10 results (StackOverflow, ECCC, UCLA StarAI k-subset gradient estimators). No relevant results."),
    ("derandomized reservoir sampling", "general", "10 results. Only the classical literature (Vitter 1985, Li 1994, Wikipedia, Weizmann lecture notes, distributed/weighted reservoir variants). No derandomisation work found."),
    ("entropy cost of reservoir sampling k items", "general", "10 results, all textbook/PDF noise. No relevant results."),
    ("minimum entropy coupling streaming algorithm", "general", "10 results: the MEC literature (Compton, Sokota, Cicalese, Kocaoglu). No streaming-sampler connection in any of them."),
    ("bit complexity streaming sampling random bits", "scholarly", "10 results. One useful hit: 'Optimal Random Bit Complexity in Efficient Sampling of Set Partition-Like Structures' (LNCS 2025) -- chased separately."),
    ("randomness efficient combination generation low randomness", "scholarly", "10 results, entirely off-topic (OpenAlex citation-count ranking failure). No relevant results."),
    ("symmetric chain decomposition Boolean lattice de Bruijn Tengbergen Kruyswijk", "scholarly", "10 results, all genuine SCD papers. Confirms SCD is a mature, well-populated area."),
    ("normalized matching property Boolean lattice LYM", "scholarly", "10 results on normalized matching in various lattices. Confirms the property is standard."),
    ("minimum entropy coupling NP-hard approximation", "scholarly", "10 results: ISIT 2019/2022 greedy-approximation papers. Confirms the approximation-only landscape."),
    ("Optimal Random Bit Complexity in Efficient Sampling of Set Partition-Like Structures", "general", "8 results. HIGH YIELD: located Bodini & Durand's 'Entropic Generation of Binary Words' (arXiv:2606.13157), the offline k-subset random-bit paper."),
    ("optimal random bit complexity sampling combinatorial structures 2025", "scholarly", "10 results, entirely off-topic. No relevant results."),
    ("random-bit model uniform sampling combinatorial structures Bodini Lumbroso", "general", "7 results: Devroye's 'The Random Bit Model' chapter, Bodini's analytic samplers, random-bit-optimal planar trees. Establishes the random-bit-model school as the adjacent literature."),
    ("Lumbroso optimal discrete uniform generation from coin flips", "general", "7 results, arXiv:1304.1916 located. Background for the per-draw realisation layer."),
    ("entropy optimal sampling binary words given weight random bits", "general", "7 results. Located Draper & Saad's IEEE TIT companion 'Efficient Rejection Sampling in the Entropy-Optimal Range' and re-confirmed Bodini-Durand."),
    ("random bits necessary uniform random subset stream lower bound", "general", "8 results, all generic complexity textbooks/STOC tables of contents. No relevant results."),
    ("anytime uniform sampling unknown length randomness complexity", "general", "7 results, all popular reservoir-sampling expositions. No randomness-complexity treatment."),
    ("pseudorandom generator space-bounded sampling without replacement Nisan", "general", "3 results (Arora-Barak, ECCC). No relevant results -- the INW/Nisan derandomisation line does not treat this problem."),
    ("coupling of prefix marginals uniform k-subset deterministic eviction", "general", "8 results, ALL about LLM KV-cache eviction. No relevant results -- 'eviction' is now dominated by that usage."),
    ("Knuth-Yao streaming sampler reservoir", "general", "6 results. Only Vitter 1985 and unrelated Knuth-Yao Gaussian samplers. No relevant results."),
    ("randomness recycling reservoir sampling k items", "general", "6 results (peteroupc randomfunc page, conference proceedings noise). No relevant results."),
    ("balanced deletion map k-subsets random bits streaming", "general", "6 results, all streaming-with-deletions graph algorithms. No relevant results."),
    ("low randomness combination generation streaming unknown n", "general", "6 results, all textbook/spec noise. No relevant results."),
    ("normalized matching property Boolean lattice definition theorem Sperner theory", "general", "8 results. HIGH YIELD: located Tomon, Electron. J. Combin. 23(2) #P2.53, which states the definition and the Boolean-lattice fact verbatim."),
    ("Greene Kleitman symmetric chain decomposition bracketing parentheses construction", "general", "8 results, including combos.org symmetric-chain Gray codes and the Damiani/D'Antona review of the Greene/Kleitman/Leeb interpretation."),
    ("subset sum modulo k uniform random k-subset distribution", "general", "5 results. HIGH YIELD: located Wagon & Wilf, 'When are subset sums equidistributed modulo m?', Electron. J. Combin. 1(1) #R3."),
    ("equitable edge coloring bipartite graph de Werra balanced", "general", "5 results (JGAA nearly-equitable edge-colouring algorithms, Hilton & de Werra 1982). Relevant as the algorithmic analogue of the balanced-fibre problem but not applied to sampling anywhere."),
    ("biregular bipartite graph balanced orientation each vertex equal in-degree", "general", "2 results, both graph-theory glossaries. No relevant results."),
    ("Vitter 1984 Faster methods for random sampling Communications of the ACM Algorithm A B C D", "general", "8 results. Confirms CACM 27(7) 1984 contains Algorithms A-D and that 'the main result of this paper is the design and analysis of Algorithm D'."),
    ("Li 1994 reservoir-sampling algorithms time complexity Algorithm L pseudocode", "general", "1 result, an unrelated thesis citing Li. No free full text found; ACM DL abstract page used instead."),
]

BIT_TABLE = """\
| k | n | accept bits | eviction bits | total | eviction share | log2 C(n,k) |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1e4 | 127.165 | 16.575 | 143.740 | 0.1153 | 25.575 |
| 2 | 1e6 | 284.221 | 25.785 | 310.006 | 0.0832 | 38.863 |
| 2 | 1e9 | 634.543 | 39.601 | 674.144 | 0.0587 | 58.795 |
| 4 | 1e4 | 217.936 | 61.634 | 279.570 | 0.2205 | 48.565 |
| 4 | 1e6 | 513.627 | 98.475 | 612.102 | 0.1609 | 75.141 |
| 4 | 1e9 | 1186.640 | 153.737 | 1340.377 | 0.1147 | 115.004 |
| 10 | 1e4 | 434.509 | 227.839 | 662.348 | 0.3440 | 111.080 |
| 10 | 1e6 | 1112.857 | 380.818 | 1493.675 | 0.2550 | 177.525 |
| 10 | 1e9 | 2704.074 | 610.288 | 3314.362 | 0.1841 | 277.182 |
| 32 | 1e4 | 997.252 | 916.658 | 1913.910 | 0.4789 | 307.472 |
| 32 | 1e6 | 2920.627 | 1653.477 | 4574.104 | 0.3615 | 520.146 |
| 32 | 1e9 | 7641.586 | 2758.718 | 10400.304 | 0.2653 | 839.052 |
| 100 | 1e4 | 2101.900 | 3056.326 | 5158.225 | 0.5925 | 803.290 |
| 100 | 1e6 | 7354.943 | 6115.902 | 13470.845 | 0.4540 | 1468.385 |
| 100 | 1e9 | 20972.405 | 10705.315 | 31677.719 | 0.3379 | 2464.970 |
| 1000 | 1e4 | 6289.608 | 22942.583 | 29232.191 | 0.7848 | 4683.723 |
| 1000 | 1e6 | 43456.667 | 68836.222 | 112292.889 | 0.6130 | 11401.450 |
| 1000 | 1e9 | 156683.888 | 137677.416 | 294361.304 | 0.4677 | 21367.954 |
"""

K1_TABLE = """\
| n | sum_i H_b(1/i) | log2(n) + sum_j (1/j)log2(j-1) | (ln n)^2 / (2 ln 2) |
|---:|---:|---:|---:|
| 1e3 | 43.150148 | 43.150148 | 34.420600 |
| 1e5 | 110.979854 | 110.979854 | 95.612776 |
| 1e6 | 156.371318 | 156.371318 | 137.682398 |
"""


def esc(x: str) -> str:
    return x.replace("|", "\\|").replace("\n", "<br>")


parts: list[str] = []
A = parts.append

A(f"# {d['title']}\n")
A(f"*Prior-art and source-code audit. All sources fetched {ACCESSED_NOTE}.*\n"
  if False else "*Prior-art and source-code audit. Every quotation below was fetched on 2026-09-18 "
  "from the URL recorded next to it; nothing is reconstructed from memory. Where a source could "
  "not be reached it is marked explicitly rather than paraphrased.*\n")

A("## Executive synthesis\n")
A(d["answer"] + "\n")

A("## 1. Scoop verdicts\n")
for v in d["scoop_verdicts"]:
    A(f"### {v['work']} — **{v['verdict']}**\n")
    A(f"- **Citation:** {v['citation']}")
    A(f"- **URL:** <{v['url']}>")
    A(f"- **Version / accessed:** {v['version_and_access_date']}")
    A(f"- **Section:** {v['section']}\n")
    A("> " + v["deciding_quote"].replace("\n", "\n> ") + "\n")
    A(f"**What it covers.** {v['what_it_covers']}\n")
    A(f"**What it does not cover.** {v['what_it_does_not_cover']}\n")
    A(f"**Implication for our claim.** {v['implication_for_our_claim']}\n")

A("## 2. Attribution map\n")
for a in d["attribution_map"]:
    A(f"### {a['object']} — **{a['status']}**\n")
    A(f"- **Canonical citation:** {a['canonical_citation']}")
    A(f"- **URL:** <{a['url']}>\n")
    A("> " + a["quote"].replace("\n", "\n> ") + "\n")
    A(f"**What we may claim.** {a['what_we_may_claim']}\n")

A("## 3. Deployed-implementation table (16 rows, all read from source)\n")
A("| Library | Version / commit | Algorithm | RNG granularity | log1p/expm1 | Test suite |")
A("|---|---|---|---|---|---|")
for r in d["deployed_implementations"]:
    A(f"| [{esc(r['library'])}]({r['file_url']}) | {esc(r['version_or_commit'])} | "
      f"`{r['algorithm']}` | {esc(r['rng_granularity'])} | {esc(r['uses_log1p_or_expm1'])} | "
      f"{r['test_suite_checks']} |")
A("")
A("### Verbatim eviction lines\n")
for r in d["deployed_implementations"]:
    A(f"**{r['library']}** — `{r['algorithm']}` — <{r['file_url']}>\n")
    A("```\n" + r["eviction_line_verbatim"] + "\n```\n")
    A(f"*{r['notes']}*\n")

A("## 4. Necessity-claim quotations (the folklore audit)\n")
counts = {"A_strict_necessity": 0, "B_sufficiency_only": 0, "C_descriptive": 0}
for q in d["necessity_quotes"]:
    counts[q["bucket"]] = counts.get(q["bucket"], 0) + 1
A(f"**Counts: A (strict necessity asserted) = {counts['A_strict_necessity']}; "
  f"B (sufficiency only) = {counts['B_sufficiency_only']}; "
  f"C (descriptive) = {counts['C_descriptive']}.** "
  f"Total {len(d['necessity_quotes'])} quotations from "
  f"{len({q['source'].split(' -- ')[0] for q in d['necessity_quotes']})} distinct sources.\n")
A("**This is the single most important honest-attribution finding in the audit.** The "
  "hypothesis's claim that 'every tutorial, Wikipedia and StackOverflow states uniform eviction "
  "is necessary' is NOT supported. Not one fetched passage asserts that a non-uniform or "
  "deterministic eviction rule would bias the sample. The claim must be softened to: *uniform "
  "eviction is universally taught and universally deployed, every proof found is "
  "sufficiency-only, and the necessity question is simply never asked.* Do not manufacture "
  "bucket-A quotes to save the contribution; delete the contribution instead.\n")
A("Two sources were UNREACHABLE and contributed nothing: `math.stackexchange.com` and "
  "`quantmemo.com`, both blocked by Cloudflare challenges. One fetched page "
  "(jeremykun.com) contained embedded text purporting to forbid an AI from using its content; "
  "that text was treated as untrusted page data rather than an instruction, and the page was "
  "excluded from the quotation list as a precaution.\n")
for q in d["necessity_quotes"]:
    A(f"**{q['bucket']}** — {q['source']} ({q['date']}) — <{q['url']}>\n")
    A("> " + q["quote"].replace("\n", "\n> ") + "\n")

A("## 5. Real bug reports in deployed samplers\n")
A("| Repo | Defect order | Caught by tests? | Link |")
A("|---|---|---|---|")
for b in d["bug_reports"]:
    A(f"| {b['repo']} | **{b['defect_order']}** | {'yes' if b['caught_by_tests'] else 'no'} | "
      f"<{b['url']}> |")
A("")
for b in d["bug_reports"]:
    A(f"### {b['repo']} — {b['defect_order']}\n")
    A(f"<{b['url']}>\n")
    A(b["summary"] + "\n")

A("## 6. Own arithmetic: where the bits actually go\n")
A("`bit_accounting.py` in this workspace computes, deterministically and with no RNG, network "
  "or LLM call, the Shannon entropy of the variate sequence Algorithm R *chooses to draw* on a "
  "stream of length n: the accept coin H_b(k/i) at each step i>k, plus log2(k) per acceptance "
  "for the uniform eviction. Draper & Saad's recycling makes the *realisation* of a chosen "
  "variate sequence cost within eps of its own entropy, so these sums are the floors that "
  "recycling converges to. The eviction column is exactly what a (near-)deterministic eviction "
  "rule could remove; the final column is the entropy of the answer alone, shown for scale.\n")
A(BIT_TABLE)
A("Two consequences. **(a) Alternate hypothesis #4 is refuted for k >= 10**: recycling on "
  "classical Algorithm R still pays the whole eviction column, which is 18-34% of the total at "
  "k=10 and 34-59% at k=100. It is, however, a fair description at k=2-4, where eviction is only "
  "6-22%. The hypothesis's '25-61%' headline corresponds to roughly 10 <= k <= 100 and must be "
  "stated with its (k, n) regime. **(b) The every-prefix constraint, not the eviction rule, "
  "dominates**: at k=10, n=1e6 even the deterministic-eviction ideal costs ~1494 bits against "
  "log2 C(n,k) = 177.5 bits for the final subset, and Bodini & Durand already achieve "
  "log2 C(n,k) + O(k) offline. The ~8x price of anytime-validity is arguably the stronger "
  "headline.\n")
A("Cross-check of the k=1 closed form quoted in the hypothesis (reproduces exactly, so the "
  "transcription is correct — only the attribution needs fixing):\n")
A(K1_TABLE)

A("## 7. What the literature already settles\n")
for line in d["settled_by_literature"]:
    A(f"- {line}")
A("")

A("## 8. Corrections to the hypothesis\n")
for i, line in enumerate(d["corrections_to_the_hypothesis"], 1):
    A(f"{i}. {line}")
A("")

A("## 9. Scoop-hunt query log (auditable negative)\n")
A(f"{len(QUERY_LOG)} distinct queries were run on 2026-09-18 across general and scholarly "
  "modes. Every query is logged with its outcome, including the empty ones, so the negative "
  "result is reproducible.\n")
A("| # | Query | Mode | Outcome |")
A("|---:|---|---|---|")
for i, (q, mode, res) in enumerate(QUERY_LOG, 1):
    A(f"| {i} | `{esc(q)}` | {mode} | {esc(res)} |")
A("")
A("The Wikipedia 'Reservoir sampling' reference list and the citation neighbourhoods of "
  "arXiv:2505.18879 and arXiv:1407.1689 were also checked. **No source was found anywhere that "
  "combines a random-bit/entropy cost metric with k >= 2 streaming or every-prefix-uniform "
  "sampling.**\n")

A("## 10. Scope cuts and unreached sources\n")
A("- **Li 1994 (Algorithm L) pseudocode: NOT VERIFIED — paywalled.** `dl.acm.org` served the "
  "PDF once (yielding the algorithm-comparison sentence quoted in section 1) and thereafter "
  "returned HTTP 403. The Algorithm-L eviction and skip lines are instead documented from "
  "deployed Algorithm-L implementations (gstamatelat `LiLSampling`, DuckDB "
  "`base_reservoir_sample.cpp`) rather than reconstructed from secondary sources.")
A("- **Cicalese-Gargano-Vaccaro journal PDF (iris.unisa.it): UNREACHABLE** (Cloudflare "
  "challenge, HTTP 403). The arXiv preprint arXiv:1901.07530 was used instead; Theorem 2 is "
  "textually identical.")
A("- **math.stackexchange.com and quantmemo.com: UNREACHABLE** (Cloudflare). Contributed "
  "nothing to the necessity-quotation list.")
A("- **Guava's absence of a reservoir sampler** was established by direct inspection of "
  "`collect/` plus a directory listing of `common/math`; the unauthenticated GitHub code-search "
  "API returned 401, so a stray helper elsewhere in the repo cannot be ruled out with certainty.")
A("- **Redis and PostgreSQL test-suite cells are honestly `unknown`**, not `none`: a "
  "'SRANDMEMBER histogram distribution' Tcl test was located but its assertion body was not "
  "fetched, and no statistical harness beyond the deterministic regression file was found for "
  "`tsm_system_rows`.")
A("- No scope cuts were needed: the deployed-implementation table reached 16 rows (target >= 15) "
  "and the necessity list reached 15 quotations from 12 sources (target >= 8).\n")

A("## 11. Follow-up questions\n")
for q in d["follow_up_questions"]:
    A(f"- {q}")
A("")

A("## Sources\n")
for s in d["sources"]:
    who = ", ".join(s["authors"]) if s.get("authors") else ""
    yr = f" ({s['year']})" if s.get("year") else ""
    head = f"**[{s['index']}]** [{s['title']}]({s['url']})"
    if who:
        head += f" — {who}{yr}"
    elif yr:
        head += f" —{yr}"
    A(head + "  ")
    A(f"{s['summary']}\n")

(OUT / "research_report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
print("wrote research_report.md", (OUT / "research_report.md").stat().st_size, "bytes")
