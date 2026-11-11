# GraphLoc-498: Find Flat, Complete on the Graph

Code and data for the Kaggle *Gemma 4 Developer Agent Paper Track* submission
"Find Flat, Complete on the Graph: when do code graphs help small coding agents localize bugs?"

## What's here

| Path | Contents |
|---|---|
| `data/graphloc.jsonl` | 498 tasks: `instance_id, repo, base_commit, fix_commit, pr_number, issue_number, problem_statement, patch, test_patch, gold_nodes, gold_files, split, ...` (fields compatible with the competition's `tasks.jsonl`) |
| `data/graphs/<instance_id>.json` | NetworkX node-link code graph at `base_commit`, using the competition schema (`nodes[*].id/name/text`, `edges[*].source/target/type/key`), plus `kind/file/start/end` on nodes |
| `results/` | rankings, per-task metrics, equal-budget, completion, policy and Gemma outputs |
| `src/` | full pipeline (below) |
| `paper/` | writeup and figure |

## Reproduce

```bash
# 1. clone the 8 repos into repos/  (fastapi, rich, requests, httpx, starlette, flask, click, attrs)
# 2. mine issue-linked fix tasks (needs an authenticated `gh` CLI)
python src/mine.py psf/requests repos/requests data/tasks_requests.jsonl 80   # MIN_DATE=2012 for requests
# 3. graphs + gold labels
python src/build_all.py
# 4. retrieval baselines (GPU recommended): BM25, Qodo-Embed-1-1.5B, bge-base, fusion, PPR, 1-hop
python src/localize.py
# 5. analyses
python src/analysis.py      # dataset stats, hop distances, equal-budget finding frontier, co-edit locality
python src/completion.py    # fix-completion given one known site
python src/policy.py        # end-to-end FFCG policy + paired bootstrap
# 6. Gemma 4 listwise rerank (4-bit, 16 GB GPU), plain vs graph-context prompts
python src/gemma_rerank.py google/gemma-4-E4B-it fuse2 plain 20
python src/gemma_rerank.py google/gemma-4-E4B-it fuse2 graph 20
python src/gemma_policy.py plain
```

## Licenses

Pipeline code: Apache-2.0. Task text and code snippets are derived from the upstream repositories
and retain their licenses: fastapi (MIT), rich (MIT), requests (Apache-2.0), httpx (BSD-3),
starlette (BSD-3), flask (BSD-3), click (BSD-3), attrs (MIT). Issue text belongs to its GitHub authors.
