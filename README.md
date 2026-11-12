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
| `paper/` | `writeup_final.md` (the submitted paper), `writeup.md` (template), figure |

## Download

Graphs (`data/graphs/`, 60 MB compressed) and all rankings (`results/rankings.json`) are attached to the
[v1.0 release](https://github.com/bhargavaramgajula007/graphloc-498/releases/tag/v1.0):

```bash
gh release download v1.0 -R bhargavaramgajula007/graphloc-498
tar -xzf graphloc498_graphs.tar.gz -C data && gunzip -c rankings.json.gz > results/rankings.json
```

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
# 7. cross-repo QLoRA (negative result, §4.7): results in results/lora_v1_buggy, results/lora_v2_shuffled
#    HEAD's lora_train.py is v3 (one retrieval-ordered + one shuffled copy, LR 5e-5; `bash run_v3.sh`),
#    result in results/gemma_*_lora-loc_comp2held.jsonl (fold A). The v2 run used the
#    script at commit cdfa5fe (LR 1e-4, 3 orders); v1 differs from v2 only in inserting missed gold at the
#    lowest-ranked slot instead of a random slot.
python src/lora_train.py google/gemma-4-E4B-it competition_repos plain adapters/loc_comp2held 3
ADAPTER=adapters/loc_comp2held SPLIT=heldout_repos python src/gemma_rerank.py google/gemma-4-E4B-it fuse2 plain 20
```

## Licenses

Pipeline code: Apache-2.0. Task text and code snippets are derived from the upstream repositories
and retain their licenses: fastapi (MIT), rich (MIT), requests (Apache-2.0), httpx (BSD-3),
starlette (BSD-3), flask (BSD-3), click (BSD-3), attrs (MIT). Issue text belongs to its GitHub authors.
