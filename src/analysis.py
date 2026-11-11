"""Graph-structure analyses on GraphLoc.
 1. dataset stats
 2. hop distance from retrieval seeds to gold
 3. equal-budget frontier: graph expansion vs reading deeper in the flat ranking
 4. co-edit locality of multi-node gold sets
"""
import json, os
from collections import Counter, defaultdict
import numpy as np
import networkx as nx
import pandas as pd

tasks = [json.loads(l) for l in open("data/graphloc.jsonl", encoding="utf-8")]
R = json.load(open("results/rankings.json"))
SEED = "fuse2"
os.makedirs("results", exist_ok=True)
ETS = {"calls": {"calls"}, "all": {"calls", "contains", "inherits", "imports"}}


def ugraph(iid, ets):
    g = json.load(open(f"data/graphs/{iid}.json", encoding="utf-8"))
    G = nx.Graph()
    G.add_nodes_from(n["id"] for n in g["nodes"])
    G.add_edges_from((e["source"], e["target"]) for e in g["edges"] if e["type"] in ets)
    kinds = {n["id"]: n["kind"] for n in g["nodes"]}
    return G, kinds


rows, hop_rows, frontier, coedit = [], [], defaultdict(list), []
BUDGETS = [5, 10, 20, 40, 80]
for t in tasks:
    iid, gold = t["instance_id"], set(t["gold_nodes"])
    rows.append(dict(repo=t["repo"], split=t["split"], nodes=t["n_graph_nodes"], edges=t["n_graph_edges"],
                     n_gold=len(gold), n_files=len(t["gold_files"]), lines=t["n_changed_lines"],
                     issue_words=len(t["problem_statement"].split())))
    rank = R[SEED][iid]
    for en, ets in ETS.items():
        G, kinds = ugraph(iid, ets)
        # hop distance top-1 -> nearest gold
        d = nx.single_source_shortest_path_length(G, rank[0], cutoff=6) if rank[0] in G else {}
        hd = min([d[g] for g in gold if g in d], default=99)
        dm = None
        ment = R["mention"].get(iid, [])[:3]
        if ment:
            dd = [nx.single_source_shortest_path_length(G, m, cutoff=6) for m in ment if m in G]
            dm = min([x[g] for x in dd for g in gold if g in x], default=99)
        hop_rows.append(dict(iid=iid, split=t["split"], edges=en, hop_top1=hd, hop_mention=dm))
        # equal-budget frontier
        if en == "calls":
            for B in BUDGETS:
                flat = set(rank[:B])
                # graph: take top-s seeds, BFS by ranked order of neighbours until budget B
                for s in (1, 3):
                    read = list(rank[:s])
                    frontier_q = list(rank[:s])
                    while frontier_q and len(read) < B:
                        u = frontier_q.pop(0)
                        nbrs = sorted((v for v in G[u] if kinds.get(v) in ("function", "class") and v not in read),
                                      key=lambda v: rank.index(v) if v in rank else 10 ** 6)
                        for v in nbrs:
                            if len(read) >= B:
                                break
                            read.append(v); frontier_q.append(v)
                    read += [x for x in rank if x not in read][: max(0, B - len(read))]
                    frontier[(f"graph_bfs_s{s}", B)].append(bool(gold & set(read)))
                # hybrid: half flat, half 1-hop of top seeds
                half = list(rank[: B // 2])
                for u in rank[:3]:
                    for v in sorted(G[u], key=lambda v: rank.index(v) if v in rank else 10 ** 6):
                        if len(half) >= B:
                            break
                        if v not in half and kinds.get(v) in ("function", "class"):
                            half.append(v)
                half += [x for x in rank if x not in half][: max(0, B - len(half))]
                frontier[("hybrid_half", B)].append(bool(gold & set(half)))
                frontier[("flat", B)].append(bool(gold & flat))
            # co-edit locality
            if len(gold) > 1:
                gl = [g for g in gold if g in G]
                pair = [nx.shortest_path_length(G, a, b) if nx.has_path(G, a, b) else 99
                        for i, a in enumerate(gl) for b in gl[i + 1:]]
                Ga, _ = ugraph(iid, ETS["all"])
                pair_all = [nx.shortest_path_length(Ga, a, b) if nx.has_path(Ga, a, b) else 99
                            for i, a in enumerate(gl) for b in gl[i + 1:]]
                coedit.append(dict(iid=iid, n_gold=len(gold), min_calls=min(pair), max_calls=max(pair),
                                   min_all=min(pair_all), max_all=max(pair_all)))

ds = pd.DataFrame(rows)
stats = ds.groupby(["split", "repo"]).agg(tasks=("nodes", "size"), nodes=("nodes", "mean"), edges=("edges", "mean"),
                                          gold=("n_gold", "mean"), files=("n_files", "mean"),
                                          lines=("lines", "median"), issue_words=("issue_words", "median")).round(1)
stats.to_csv("results/dataset_stats.csv"); print(stats.to_string()); print(ds[["nodes", "n_gold", "lines"]].describe().round(1))

hp = pd.DataFrame(hop_rows)
hp.to_csv("results/hops.csv", index=False)
for en in ETS:
    sub = hp[hp.edges == en]
    bins = lambda s: s.clip(upper=4).replace(4, 4).value_counts(normalize=True).sort_index().round(3).to_dict()
    print(f"\nHop distance top1->gold [{en}] (4 = >=4/unreachable):", bins(sub.hop_top1))
    print(f"Hop distance mention->gold [{en}] (n={sub.hop_mention.notna().sum()}):", bins(sub.hop_mention.dropna()))

fr = pd.DataFrame([dict(strategy=k[0], budget=k[1], hit=np.mean(v)) for k, v in frontier.items()])
fr = fr.pivot(index="strategy", columns="budget", values="hit").round(3)
fr.to_csv("results/frontier.csv"); print("\nEqual-budget hit rate (gold in nodes read):\n", fr.to_string())

ce = pd.DataFrame(coedit)
ce.to_csv("results/coedit.csv", index=False)
print(f"\nCo-edit locality, {len(ce)} multi-gold tasks:")
for c in ["min_calls", "min_all", "max_all"]:
    print(c, ce[c].clip(upper=4).value_counts(normalize=True).sort_index().round(3).to_dict())
