"""Fix-completion analysis: given ONE gold node already found (the first gold in the ranking),
how well do strategies recover the REMAINING gold nodes at equal read budget B?
  flat   : continue down the retrieval ranking
  graph  : BFS from the found node over edge set E (neighbours ordered by retrieval rank)
  hybrid : alternate graph / flat picks
"""
import json
from collections import defaultdict
import numpy as np
import networkx as nx
import pandas as pd

tasks = [json.loads(l) for l in open("data/graphloc.jsonl", encoding="utf-8")]
R = json.load(open("results/rankings.json"))["fuse2"]
ETS = {"calls": {"calls"}, "struct": {"contains", "inherits"}, "all": {"calls", "contains", "inherits", "imports"}}
BUDGETS = [3, 5, 10, 20]
res = defaultdict(list)
n_used = 0
for t in tasks:
    gold = set(t["gold_nodes"])
    if len(gold) < 2:
        continue
    iid = t["instance_id"]
    rank = R[iid]
    first = next((n for n in rank if n in gold), None)
    if first is None:
        continue
    n_used += 1
    rest = gold - {first}
    g = json.load(open(f"data/graphs/{iid}.json", encoding="utf-8"))
    kinds = {n["id"]: n["kind"] for n in g["nodes"]}
    pos = {n: i for i, n in enumerate(rank)}
    flat_order = [n for n in rank if n != first]
    for en, ets in ETS.items():
        G = nx.Graph()
        G.add_nodes_from(kinds)
        G.add_edges_from((e["source"], e["target"]) for e in g["edges"] if e["type"] in ets)
        # BFS layers from first, each layer sorted by retrieval rank; skip modules but traverse through them
        dist = nx.single_source_shortest_path_length(G, first, cutoff=4)
        graph_order = sorted((n for n, d in dist.items() if d > 0 and kinds.get(n) in ("function", "class")),
                             key=lambda n: (dist[n], pos.get(n, 10 ** 6)))
        for B in BUDGETS:
            gsel = graph_order[:B]
            gsel += [n for n in flat_order if n not in gsel][: B - len(gsel)]
            hyb, gi, fi = [], iter(graph_order), iter(flat_order)
            while len(hyb) < B:
                for src in (gi, fi):
                    for n in src:
                        if n not in hyb:
                            hyb.append(n); break
                    if len(hyb) >= B:
                        break
                else:
                    pass
                if len(hyb) >= B or (not graph_order and not flat_order):
                    break
            res[("graph:" + en, B)].append(len(rest & set(gsel)) / len(rest))
            res[("hybrid:" + en, B)].append(len(rest & set(hyb)) / len(rest))
    for B in BUDGETS:
        res[("flat", B)].append(len(rest & set(flat_order[:B])) / len(rest))

df = pd.DataFrame([dict(strategy=k[0], budget=k[1], recall=np.mean(v)) for k, v in res.items()])
tab = df.pivot(index="strategy", columns="budget", values="recall").round(3)
tab.to_csv("results/completion.csv")
print(f"Fix-completion recall of remaining gold, {n_used} multi-location tasks:\n", tab.to_string())
