# Find Flat, Complete on the Graph

### When do code graphs help small coding agents localize bugs? An open 498-task benchmark with competition-schema graphs, and a Gemma 4 study

## Abstract

The Gemma 4 Developer Agent competition equips agents with code-graph tools (`get_code_neighbors`, `get_code_subgraph`, `search_similar_code`). We ask a narrow, practical question: **for bug localization, when should a small on-device agent use the graph instead of plain retrieval?** We release **GraphLoc-498**, a fully reproducible benchmark of 498 real bug-fix tasks mined from 8 Python repositories (the 4 competition repositories plus 4 held-out ones). Each task pairs a GitHub issue written before the fix with function-level gold labels and a code graph in the competition's NetworkX schema. Across 25 localization strategies under equal *read budgets*, we find a clean division of labour:

1. **Graphs are poor finders.** Pure graph expansion from top hits trails simply reading further down a flat ranking at every budget (hit rate 0.84 vs 0.92 at 40 symbols). A dense-led fusion of a code embedder and BM25 finds a gold function in the top 5 for **69.9%** of issues.
2. **Graphs are good completers.** In 235 multi-location fixes, once one edited function is known, a graph breadth-first search recovers **45% more of the remaining edits** than the flat ranking at a 3-symbol budget (0.360 vs 0.249). Structural edges (`contains`, `inherits`) matter: call edges alone are no better than the flat ranking.
3. **256-d embeddings suffice if projected.** Naively truncating a 1536-d embedder to 256-d loses 8 points of Acc@5. A per-repository PCA to 256-d loses nothing.
4. **Gemma-4-E4B, 4-bit, as a listwise reranker** <<GEMMA_ABSTRACT>>

We distil these findings into a tool-use policy, *Find Flat, Complete on the Graph* (FFCG), and give concrete prompting guidance for the competition harness.

## 1. Introduction

Small open models fail at repository-level software engineering less because they cannot write a patch than because they **cannot find where to write it**. Localization errors dominate agent failure analyses on SWE-bench-style tasks (Xia et al., 2024; Chen et al., 2025). The competition responds by shipping every repository as a call/dependency graph with node embeddings, plus graph tools. Yet graph-guided localization is usually evaluated end-to-end, with a frontier model in the loop. That makes it hard to tell *which* part of the graph helps, *when*, and *at what cost in context tokens*. Context tokens are the scarcest resource for a 4B-parameter model under a 12-hour, single-accelerator budget.

We isolate localization and measure it under a **read budget**: the number of symbols whose source an agent must place in context. This lets us compare strategies at equal cost, the comparison an agent designer actually faces. We contribute:

- **GraphLoc-498 (resource).** 498 issue-linked bug-fix tasks with function-level gold labels, graphs in the exact competition schema, and an open mining pipeline that regenerates everything from `git` and the GitHub API. It contains 219 tasks on the competition's repositories (more than the 129 public training tasks) and 279 tasks on held-out repositories for testing generalization.
- **An equal-budget methodology** that separates *finding* the first edit location from *completing* a multi-location fix.
- **Empirical findings** on retrievers, embedding dimensionality, graph propagation, and Gemma 4 reranking, with paired bootstrap confidence intervals.
- **A policy (FFCG) and harness guidance** that turn these findings into tool calls.

## 2. GraphLoc-498

**Mining.** We replay each repository's first-parent history, keeping squash- or merge-commits whose diff touches 1–3 non-test `.py` files (2–120 changed lines) *and* at least one test file. This is the same "code + test" filter as the competition pipeline. For each pull request we follow GitHub's `closingIssuesReferences` (falling back to `fixes #N` references) and keep the task only if the linked **issue predates the pull request** and has a non-trivial body. The issue, not the PR description, becomes the problem statement, which avoids the well-known leakage in which a PR describes its own fix. Each task stores `base_commit`, `patch`, `test_patch`, and issue text, following the competition's `tasks.jsonl` fields.

**Graphs.** For every `base_commit` we parse all non-test source files with Python's `ast` module. Nodes are modules, classes and functions/methods, with ids that are dotted symbol paths (`requests.models.PreparedRequest.prepare_body`), the same convention as the competition graphs, and each node stores its source `text`. Edges are typed `contains`, `calls`, `imports` and `inherits`. Call edges are resolved through local scopes, imports, package re-exports, `self`/`cls`/`super()` dispatch over the class hierarchy, and a unique-name fallback. Graphs are serialized as NetworkX node-link JSON (`directed`, `multigraph`, `nodes[*].id/name/text`, `edges[*].source/target/type/key`), so they drop into the competition's graph tools.

**Gold labels.** Each removed line, and each insertion point, in the gold patch is mapped to the *innermost* function or class enclosing it at `base_commit`. Tasks whose edits touch only module-level code are dropped (19 of 517).

**Statistics.**

<<TABLE1>>

*Table 1. GraphLoc-498 by repository (means unless noted).*

Fixes are local but not trivial. The median patch changes 14 lines, a graph has 563 nodes on average, and **48% of tasks edit more than one symbol**, which motivates the completion analysis. FastAPI yields few tasks (24) because most of its bug reports live in GitHub Discussions rather than Issues, a curation caveat for anyone extending the competition data.

## 3. Methods

**Candidates.** All function and class nodes of the snapshot (≈ 560 per task). An agent "reads" a node by placing its source in context.

**Flat retrievers.** *BM25* over identifier-split symbol ids (up-weighted ×3) plus source. *Dense*: Qodo-Embed-1-1.5B (code-specialised, 1536-d) and bge-base-en-v1.5 (general, 768-d), cosine similarity between the issue text and `id + source`. *Mention*: exact matches of identifiers from the issue, which mimics `grep`. *Fusion*: reciprocal-rank fusion (Cormack et al., 2009) of dense and BM25 weighted 2:1 (`fuse2`). We also tried a variant that up-weighted mentions; it hurt.

**256-d embeddings.** The competition ships 256-d node embeddings. We compare the full embedder, naive truncation to the first 256 dimensions, and a per-snapshot PCA projection to 256 dimensions fitted on node embeddings only (no gold information).

**Graph strategies.** (i) *Personalized PageRank* (PPR; Haveliwala, 2002; cf. HippoRAG, Gutiérrez et al., 2024) seeded with the top-10 flat results and blended with the flat rank. (ii) *1-hop expansion*: top-3 seeds followed by their neighbours, the `get_code_neighbors` analogue. (iii) *Breadth-first search* (BFS) from seeds under a read budget, with neighbours ordered by flat rank. Each runs over three edge sets: `calls`, structural (`contains`+`inherits`), and `all`.

**Equal-budget protocols.** *Finding*: for budget B, is any gold node among the B nodes read? *Completing*: on multi-location tasks, condition on the first gold node in the flat ranking (the agent has found and confirmed one edit site), and measure the fraction of the *remaining* gold nodes recovered within B further reads.

**Gemma 4 reranking.** Gemma-4-E4B-it, one of the competition's base models, loaded with 4-bit NF4 weights on a single 16 GB consumer GPU, receives the issue and the top-20 `fuse2` candidates (symbol id plus the first six non-empty source lines) and must output up to five candidate indices, most likely first, with greedy decoding and no reasoning tokens. In the **graph** condition each candidate additionally lists its callees and callers from the graph. The candidate pool is identical across conditions, so any difference comes from the graph context alone.

**Statistics.** Paired bootstrap over tasks (5,000 resamples, 95% CI).

## 4. Results

### 4.1 Flat retrieval is a strong baseline

<<TABLE2>>

A code-specialised embedder beats BM25 by 10.5 points at Acc@5, and the 2:1 fusion adds another 2.8. All methods are markedly harder on the competition repositories than on the held-out ones (Acc@5 0.571 vs 0.799 for `fuse2`). The likely reason is that `rich` and `httpx` have larger, denser graphs and longer issues. This gap should be expected when transferring results from public benchmarks.

### 4.2 256 dimensions are enough, if you project

| Qodo-Embed-1-1.5B | Acc@1 | Acc@5 | Acc@10 | MRR |
|---|---|---|---|---|
| full 1536-d | 0.363 | 0.671 | 0.743 | 0.496 |
| first 256 dims (truncate) | 0.313 | 0.592 | 0.695 | 0.440 |
| PCA → 256-d | **0.376** | 0.669 | **0.765** | **0.516** |

The same pattern holds for bge-base (truncation 0.584, PCA 0.586, full 0.616 Acc@5). Neither embedder was trained with Matryoshka objectives (Kusupati et al., 2022), so truncation discards information while PCA keeps the useful subspace. **For competitors:** the dimensionality of the provided embeddings is not by itself a bottleneck. How the 256 dimensions were obtained matters more than how many there are.

### 4.3 Graphs are poor finders

![Find vs complete](fig_find_vs_complete.png)

*Figure 1. (a) Probability that any gold node is among the nodes read, at equal budget. (b) Fraction of the remaining gold nodes recovered after one gold node is known (235 multi-location tasks).*

Re-ranking with graph propagation barely moves the needle. PPR over call edges changes Acc@5 from 0.671 to 0.679 and Acc@10 from 0.743 to 0.755. 1-hop expansion of the top-3 seeds *hurts* Acc@5 (0.671 → 0.616–0.629), because neighbours displace good flat candidates. Under an explicit read budget (Fig. 1a), graph BFS from the top three seeds trails the flat ranking at every budget (0.673 vs 0.699 at 5 symbols; 0.835 vs 0.920 at 40). Spending half the budget on 1-hop neighbours is at best neutral (0.876 vs 0.867 at 20).

Why? Hop distances explain it. When the top-1 flat result is wrong (58% of tasks), the nearest gold node is often far away: **36% of tasks have no gold node within 3 call-edges** of the top-1 hit, and 10% even with all edge types. A wrong seed sends graph exploration into the wrong neighbourhood, while the flat ranking keeps sampling independent hypotheses.

### 4.4 Graphs are good completers

The picture reverses once one edit site is known (Fig. 1b). Co-edited symbols cluster: in **69% of multi-location fixes** at least one pair of gold nodes is directly connected (any edge type; 49% via calls alone). BFS from the known site over *all* edges recovers 0.360 of the remaining gold at 3 reads versus 0.249 for the flat ranking (+45% relative), and 0.483 vs 0.357 at 5. Alternating graph and flat picks is best at larger budgets: 0.633 vs 0.510 at 10, 0.755 vs 0.636 at 20. **Edge type matters**: call edges alone tie the flat ranking at 5 reads (0.357 vs 0.357) and fall behind from 10 onward (0.470 vs 0.510; 0.566 vs 0.636 at 20). Sibling methods, overrides and subclass hooks connect through `contains`/`inherits`, not `calls`. Agents that call `get_code_neighbors(node, edge_type="calls")` leave most of the graph's value unused.

### 4.5 An end-to-end policy: Find Flat, Complete on the Graph

Without oracle knowledge of which hit is correct, FFCG reads the top-s flat results and then alternates graph BFS from those anchors (all edges) with further flat results. With s = 3 it improves recall@10 by **+2.6 points** [0.8, 4.4], recall@20 by +2.6 [1.2, 4.2] and all-gold@20 by +2.4 [0.2, 4.6], with no significant change in Acc@5 or Acc@10. The gain over the flat ranking is real but modest *because the anchor is unverified*. The oracle analysis in §4.4 shows the headroom (up to +12 points) that a reliable confirmation signal would unlock, such as a failing reproduction test or the LLM's own judgement.

### 4.6 Gemma 4 as reranker and gate

<<GEMMA_SECTION>>

## 5. Guidance for agents in the competition harness

1. **Find with `search_similar_code` plus lexical search, not with the graph.** Issue the issue text (or a condensed query) to `search_similar_code`, combine it with `grep`/`rg` on identifiers (BM25-like), and read the top 5–10 symbols. Do not spend early turns on `get_code_neighbors`.
2. **Confirm before expanding.** Once a candidate is confirmed (a reproduction test fails inside it, or the model is confident after reading it), switch to `get_code_neighbors(node)` **without** an `edge_type` filter, or `get_code_subgraph` over the confirmed node and its siblings, to find the remaining edit sites.
3. **Prefer structure over calls** when choosing neighbours: siblings in the same class, overridden methods in subclasses, and the containing class's `__init__`.
4. **Keep the reranking step cheap.** A 4-bit E4B listwise rerank of 20 candidates costs about 2k prompt tokens and about 5 s on a consumer GPU. <<GEMMA_GUIDANCE>>

## 6. Related work

**Benchmarks and agents.** SWE-bench (Jimenez et al., 2024) established issue-to-patch evaluation. SWE-agent (Yang et al., 2024) and AutoCodeRover (Zhang et al., 2024) showed that tool interfaces drive agent success. Agentless (Xia et al., 2024) made localization an explicit hierarchical stage (file → class/function → line). **Graph-based localization.** RepoGraph (Ouyang et al., 2025) and CodexGraph (Liu et al., 2024) expose repository graphs to LLMs. LocAgent (Chen et al., 2025) builds heterogeneous code graphs for multi-hop agent search. OrcaLoca (Yu et al., 2025) adds priority-based scheduling and distance-aware context pruning. These works report end-to-end gains with large models. We complement them with an **equal-budget decomposition** showing *where* the graph helps (completion) and where it does not (first-hit finding), for a *small* model. **Graph retrieval.** Personalized PageRank for retrieval-augmented generation (HippoRAG; Gutiérrez et al., 2024) inspired our propagation baseline. **Embeddings.** Matryoshka representation learning (Kusupati et al., 2022) explains why naive truncation of non-MRL embedders is lossy.

## 7. Limitations

Localization accuracy is a proxy: we do not run patches through tests. Gold labels reflect *one* valid fix; alternative fixes may edit other symbols. Our call graph is static and heuristic: dynamic dispatch through untyped attributes is only partly resolved, which likely understates the value of `calls` edges. Our graphs and embeddings are re-created rather than the hidden competition artefacts, although the schema matches. All repositories are mature Python libraries; results may differ for application code. Gemma results use 4-bit bitsandbytes quantization rather than the competition's QAT w4a16 checkpoints.

## Reproducibility

Everything regenerates from public sources: `mine.py` (git + GitHub GraphQL) → `build_all.py` (graphs, gold labels) → `localize.py` (all retrievers, rankings) → `analysis.py`, `completion.py`, `policy.py` (equal-budget, completion and bootstrap analyses) → `gemma_rerank.py`. Total compute was one RTX 5070 Ti (16 GB) for about 3 hours. Code and the GraphLoc-498 release are linked in Project Links.

## References

- Cormack, G., Clarke, C., Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *SIGIR*.
- Chen, Z., et al. (2025). LocAgent: Graph-Guided LLM Agents for Code Localization. *ACL*.
- Gutiérrez, B. J., et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *NeurIPS*.
- Haveliwala, T. (2002). Topic-sensitive PageRank. *WWW*.
- Jimenez, C. E., et al. (2024). SWE-bench: Can Language Models Resolve Real-World GitHub Issues? *ICLR*.
- Kusupati, A., et al. (2022). Matryoshka Representation Learning. *NeurIPS*.
- Liu, X., et al. (2024). CodexGraph: Bridging Large Language Models and Code Repositories via Code Graph Databases. *arXiv:2408.03910*.
- Ouyang, S., et al. (2025). RepoGraph: Enhancing AI Software Engineering with Repository-level Code Graph. *ICLR*.
- Robertson, S., Zaragoza, H. (2009). The Probabilistic Relevance Framework: BM25 and Beyond. *Foundations and Trends in IR*.
- Xia, C. S., et al. (2024). Agentless: Demystifying LLM-based Software Engineering Agents. *arXiv:2407.01489*.
- Yang, J., et al. (2024). SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering. *NeurIPS*.
- Yu, Z., et al. (2025). OrcaLoca: An LLM Agent Framework for Software Issue Localization. *ICML*.
- Zhang, Y., et al. (2024). AutoCodeRover: Autonomous Program Improvement. *ISSTA*.
- Google DeepMind (2026). Gemma 4 model card; Qodo (2025). Qodo-Embed-1 model card; Xiao, S., et al. (2023). C-Pack / bge embeddings.
