"""End-to-end 'Find Flat, Complete on Graph' (FFCG) policy vs flat ranking, on ALL tasks,
without oracle knowledge of which node is gold. Paired bootstrap CIs + per-split breakdown.

FFCG(s): read top-s of the flat ranking, then alternate {graph BFS from those s anchors over all
edges, ordered by (hop, retrieval rank)} and {continue flat}.
"""
import json
from collections import defaultdict
import numpy as np
import networkx as nx
import pandas as pd

tasks = [json.loads(l) for l in open("data/graphloc.jsonl", encoding="utf-8")]
R = json.load(open("results/rankings.json"))
ETS = {"calls", "contains", "inherits", "imports"}
rng = np.random.default_rng(0)


def ffcg(rank, G, kinds, s):
    pos = {n: i for i, n in enumerate(rank)}
    anchors = rank[:s]
    dist = {}
    for a in anchors:
        if a in G:
            for n, d in nx.single_source_shortest_path_length(G, a, cutoff=3).items():
                dist[n] = min(d, dist.get(n, 99))
    gorder = sorted((n for n, d in dist.items() if d > 0 and kinds.get(n) in ("function", "class")),
                    key=lambda n: (dist[n], pos.get(n, 10 ** 6)))
    out = list(anchors)
    gi, fi = iter(gorder), iter(rank[s:])
    while len(out) < 60:
        added = False
        for src in (gi, fi):
            for n in src:
                if n not in out:
                    out.append(n); added = True; break
        if not added:
            break
    return out


def m(rank, gold):
    g = set(gold)
    return {f"acc@{k}": float(bool(g & set(rank[:k]))) for k in (5, 10, 20)} | \
           {f"rec@{k}": len(g & set(rank[:k])) / len(g) for k in (5, 10, 20)} | \
           {f"all@{k}": float(g <= set(rank[:k])) for k in (10, 20)}


rows = []
for base in ("fuse2", "dense:Qodo-Embed-1-1.5B"):
    for t in tasks:
        iid = t["instance_id"]
        g = json.load(open(f"data/graphs/{iid}.json", encoding="utf-8"))
        kinds = {n["id"]: n["kind"] for n in g["nodes"]}
        G = nx.Graph(); G.add_nodes_from(kinds)
        G.add_edges_from((e["source"], e["target"]) for e in g["edges"] if e["type"] in ETS)
        rank = R[base][iid]
        variants = {"flat": rank, "ffcg_s1": ffcg(rank, G, kinds, 1), "ffcg_s2": ffcg(rank, G, kinds, 2),
                    "ffcg_s3": ffcg(rank, G, kinds, 3)}
        for v, r in variants.items():
            rows.append(dict(base=base, policy=v, iid=iid, split=t["split"], multi=len(t["gold_nodes"]) > 1, **m(r, t["gold_nodes"])))

df = pd.DataFrame(rows)
df.to_csv("results/policy_per_task.csv", index=False)
cols = [c for c in df.columns if "@" in c]
print(df.groupby(["base", "policy"])[cols].mean().round(3).to_string())
print("\nby split (fuse2):")
print(df[df.base == "fuse2"].groupby(["split", "policy"])[cols].mean().round(3).to_string())
print("\nmulti-location only (fuse2):")
print(df[(df.base == "fuse2") & df.multi].groupby("policy")[cols].mean().round(3).to_string())

# paired bootstrap: ffcg_s1 - flat
out = []
for base in ("fuse2",):
    d = df[df.base == base].pivot_table(index="iid", columns="policy", values=cols)
    for pol in ("ffcg_s1", "ffcg_s2", "ffcg_s3"):
        for c in cols:
            diff = (d[(c, pol)] - d[(c, "flat")]).values
            bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
            out.append(dict(policy=pol, metric=c, delta=diff.mean(), lo=np.percentile(bs, 2.5), hi=np.percentile(bs, 97.5)))
bt = pd.DataFrame(out).round(3)
bt.to_csv("results/policy_bootstrap.csv", index=False)
print("\nPaired bootstrap (policy - flat), 95% CI:\n", bt.to_string())
