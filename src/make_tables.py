"""Fill paper/writeup.md placeholders from results/*.csv -> paper/writeup_final.md"""
import json, os
import numpy as np
import pandas as pd

s = open("paper/writeup.md", encoding="utf-8").read()

# Table 1: dataset stats
st = pd.read_csv("results/dataset_stats.csv")
t1 = ["| Split | Repository | Tasks | Graph nodes | Graph edges | Gold symbols | Gold files | Median Δlines |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in st.iterrows():
    t1.append(f"| {r.split.replace('_', ' ').replace('heldout', 'held-out')} | {r.repo} | {r.tasks} | {r.nodes:.0f} | {r.edges:.0f} | {r.gold:.1f} | {r.files:.1f} | {r.lines:.0f} |")
tasks = [json.loads(l) for l in open("data/graphloc.jsonl", encoding="utf-8")]
t1.append(f"| **all** | 8 repositories | **{len(tasks)}** | {np.mean([t['n_graph_nodes'] for t in tasks]):.0f} | "
          f"{np.mean([t['n_graph_edges'] for t in tasks]):.0f} | {np.mean([len(t['gold_nodes']) for t in tasks]):.1f} | "
          f"{np.mean([len(t['gold_files']) for t in tasks]):.1f} | {np.median([t['n_changed_lines'] for t in tasks]):.0f} |")
s = s.replace("<<TABLE1>>", "\n".join(t1))

# Table 2: main localization results
m = pd.read_csv("results/metrics.csv", index_col=0)
ms = pd.read_csv("results/metrics_split.csv", index_col=[0, 1])
rows = [("mention", "Identifier mention (grep-like)"), ("bm25", "BM25"),
        ("dense:bge-base-en-v1.5", "Dense: bge-base (768-d)"), ("dense:Qodo-Embed-1-1.5B", "Dense: Qodo-Embed-1.5B (1536-d)"),
        ("fuse2", "**Fusion: 2·dense + BM25 (RRF)**"),
        ("dense+ppr0.8:calls", "Dense + PPR (calls)"), ("dense+ppr0.5:all", "Dense + PPR (all edges)"),
        ("dense+hop1:calls", "Dense + 1-hop expansion (calls)"), ("dense+hop1:all", "Dense + 1-hop expansion (all)")]
t2 = ["| Method | Acc@1 | Acc@5 | Acc@10 | All@10 | MRR | File@1 | Acc@5 comp. repos | Acc@5 held-out |",
      "|---|---|---|---|---|---|---|---|---|"]
for k, name in rows:
    r = m.loc[k]
    c = ms.loc[("competition_repos", k), "acc@5"]; h = ms.loc[("heldout_repos", k), "acc@5"]
    t2.append(f"| {name} | {r['acc@1']:.3f} | {r['acc@5']:.3f} | {r['acc@10']:.3f} | {r['all@10']:.3f} | {r['mrr']:.3f} | {r['file@1']:.3f} | {c:.3f} | {h:.3f} |")
t2.append("\n*Table 2. Function-level localization on GraphLoc-498 (Acc@k: any gold symbol in top k; All@10: every gold symbol in top 10).*")
s = s.replace("<<TABLE2>>", "\n".join(t2))

# Gemma
gem = {}
for c in ("plain", "graph"):
    p = f"results/gemma_gemma-4-E4B-it_fuse2_{c}_k20.jsonl"
    if os.path.exists(p):
        gem[c] = pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8")])
if gem:
    open("results/gemma_summary.json", "w").write(json.dumps({c: dict(n=len(d), pool_acc1=d.pool_acc1.mean(), acc1=d.acc1.mean(), acc5=d.acc5.mean(),
        in_pool=d.in_pool.mean(), secs=d.secs.mean(), tokens=d.prompt_tokens.mean()) for c, d in gem.items()}, indent=1))
open("paper/writeup_final.md", "w", encoding="utf-8").write(s)
left = [x for x in ("<<TABLE1>>", "<<TABLE2>>", "<<GEMMA_ABSTRACT>>", "<<GEMMA_SECTION>>", "<<GEMMA_GUIDANCE>>") if x in s]
print("remaining placeholders:", left)
print("words:", len(s.split()))
