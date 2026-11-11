"""Merge mined tasks, build competition-schema graphs, attach gold localization labels.
Outputs: data/graphloc.jsonl, data/graphs/<instance_id>.json
"""
import glob, json, os, sys
from collections import Counter
sys.path.insert(0, os.path.dirname(__file__))
from graph import build_graph, gold_nodes

IN_DOMAIN = {"fastapi", "rich", "requests", "httpx"}  # the 4 competition repos
os.makedirs("data/graphs", exist_ok=True)
out, stats = [], Counter()
for fn in sorted(glob.glob("data/tasks_*.jsonl")):
    repo_short = fn.split("tasks_")[1][:-6]
    for line in open(fn, encoding="utf-8"):
        t = json.loads(line)
        G = build_graph(f"repos/{repo_short}", t["base_commit"])
        gold, gfiles = gold_nodes(G, t["patch"])
        if not gold:
            stats["no_gold"] += 1
            continue
        t["gold_nodes"], t["gold_files"] = gold, gfiles
        t["split"] = "competition_repos" if repo_short in IN_DOMAIN else "heldout_repos"
        t["n_graph_nodes"] = len(G["nodes"])
        t["n_graph_edges"] = len(G["edges"])
        graph_json = dict(directed=True, multigraph=True, graph={"repo": t["repo"], "commit": t["base_commit"]},
                          nodes=[{k: d[k] for k in ("id", "name", "text", "kind", "file", "start", "end")}
                                 for d in G["nodes"].values()],
                          edges=[dict(source=s, target=tg, type=ty, key=0) for s, tg, ty in G["edges"]])
        with open(f"data/graphs/{t['instance_id']}.json", "w", encoding="utf-8") as f:
            json.dump(graph_json, f)
        out.append(t)
        stats[t["split"]] += 1
        print(t["instance_id"], len(G["nodes"]), gold, flush=True)
with open("data/graphloc.jsonl", "w", encoding="utf-8") as f:
    for t in out:
        f.write(json.dumps(t) + "\n")
print(stats, Counter(t["repo"] for t in out))
