"""Gemma-gated FFCG: use Gemma's reranked list as the flat ranking (its top pick = anchor),
then complete on the graph. Compares against fuse2 flat, Gemma flat, and fuse2-FFCG."""
import json, sys
import numpy as np
import networkx as nx
import pandas as pd
ETS = {"calls", "contains", "inherits", "imports"}


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



cond = sys.argv[1] if len(sys.argv) > 1 else "plain"
tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open("data/graphloc.jsonl", encoding="utf-8")}
R = json.load(open("results/rankings.json"))["fuse2"]
G_ = [json.loads(l) for l in open(f"results/gemma_gemma-4-E4B-it_fuse2_{cond}_k20.jsonl", encoding="utf-8")]
rng = np.random.default_rng(0)
rows = []
for r in G_:
    iid = r["iid"]; t = tasks[iid]
    g = json.load(open(f"data/graphs/{iid}.json", encoding="utf-8"))
    kinds = {n["id"]: n["kind"] for n in g["nodes"]}
    G = nx.Graph(); G.add_nodes_from(kinds)
    G.add_edges_from((e["source"], e["target"]) for e in g["edges"] if e["type"] in ETS)
    flat = R[iid]
    gem = r["reranked"] + [n for n in flat if n not in r["reranked"]]
    V = {"fuse2_flat": flat, "fuse2_ffcg_s3": ffcg(flat, G, kinds, 3), "gemma_flat": gem,
         "gemma_ffcg_s1": ffcg(gem, G, kinds, 1), "gemma_ffcg_s2": ffcg(gem, G, kinds, 2)}
    for v, rk in V.items():
        rows.append(dict(policy=v, iid=iid, split=t["split"], multi=len(t["gold_nodes"]) > 1,
                         acc1=float(rk[0] in set(t["gold_nodes"])), **m(rk, t["gold_nodes"])))
df = pd.DataFrame(rows)
df.to_csv(f"results/gemma_policy_{cond}.csv", index=False)
cols = ["acc1"] + [c for c in df.columns if "@" in c]
print(f"[{cond}] n={df.iid.nunique()}\n", df.groupby("policy")[cols].mean().round(3).to_string())
print("\nmulti-location:\n", df[df.multi].groupby("policy")[cols].mean().round(3).to_string())
d = df.pivot_table(index="iid", columns="policy", values=cols)
out = []
for a, b in [("gemma_flat", "fuse2_flat"), ("gemma_ffcg_s1", "gemma_flat"), ("gemma_ffcg_s1", "fuse2_flat")]:
    for c in cols:
        diff = (d[(c, a)] - d[(c, b)]).values
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(3000)]
        out.append(dict(cmp=f"{a} - {b}", metric=c, delta=round(diff.mean(), 3),
                        lo=round(np.percentile(bs, 2.5), 3), hi=round(np.percentile(bs, 97.5), 3)))
bt = pd.DataFrame(out); bt.to_csv(f"results/gemma_policy_bootstrap_{cond}.csv", index=False)
print(bt.to_string())
