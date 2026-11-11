"""Localization benchmark: rank graph nodes (functions/classes) for each issue.

Methods
  bm25            lexical over symbol id + source
  dense:<model>   embedding cosine (search_similar_code analogue)
  mention         exact identifier mentions in the issue (what an agent greps for)
  fuse            RRF(bm25, dense, mention)
  fuse+ppr:<E>    Personalized PageRank over edge set E seeded by fused top-k (graph reasoning)
  fuse+hop1:<E>   1-hop neighbour expansion (get_code_neighbors analogue)

usage: python src/localize.py [dense_model ...]
writes results/rankings_<method>.json and results/metrics.csv
"""
import hashlib, json, math, os, re, sys
from collections import defaultdict
import numpy as np
import networkx as nx
from rank_bm25 import BM25Okapi

DENSE = sys.argv[1:] or ["Qodo/Qodo-Embed-1-1.5B", "BAAI/bge-base-en-v1.5"]
QUERY_PREFIX = {"Qodo/Qodo-Embed-1-1.5B": "Instruct: Given a GitHub issue, retrieve the code that must be changed to resolve it\nQuery: ",
                "BAAI/bge-base-en-v1.5": "Represent this sentence for searching relevant passages: "}
EDGE_SETS = {"calls": {"calls"}, "struct": {"contains", "inherits"},
             "all": {"calls", "contains", "inherits", "imports"}}
STOP = set("""the and for with that this from have not are was but when what which into
error issue return returns value values none true false self should would could using used
import python version file line test tests example code there their they then than also
""".split())

tasks = [json.loads(l) for l in open("data/graphloc.jsonl", encoding="utf-8")]
os.makedirs("results", exist_ok=True)
os.makedirs("data/emb", exist_ok=True)


def toks(s):
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", s)
    return [w for w in re.split(r"[^A-Za-z0-9]+", s.lower()) if len(w) > 1]


def load_graph(iid):
    g = json.load(open(f"data/graphs/{iid}.json", encoding="utf-8"))
    nodes = {n["id"]: n for n in g["nodes"]}
    return nodes, g["edges"]


_models = {}
_hcache = {}


def embed(model_name, texts, is_query):
    from sentence_transformers import SentenceTransformer
    if model_name not in _models:
        _models[model_name] = SentenceTransformer(model_name, device="cuda",
                                                  model_kwargs={"torch_dtype": "bfloat16"})
        _models[model_name].max_seq_length = 512
    m = _models[model_name]
    if is_query:
        texts = [QUERY_PREFIX.get(model_name, "") + t for t in texts]
        return m.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    # content-hash cache: consecutive snapshots share most symbols
    cache = _hcache.setdefault(model_name, {})
    keys = [hashlib.sha1(t.encode("utf-8", "replace")).hexdigest() for t in texts]
    todo = list(dict.fromkeys(k for k in keys if k not in cache))
    if todo:
        k2t = dict(zip(keys, texts))
        E = m.encode([k2t[k] for k in todo], batch_size=32, normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
        cache.update(zip(todo, E))
    return np.stack([cache[k] for k in keys])


def doc_text(n):
    return f"{n['id']}\n{n['text'][:3000]}"


def rrf(*rankings, k=60, weights=None):
    s = defaultdict(float)
    for i, r in enumerate(rankings):
        w = 1.0 if weights is None else weights[i]
        for rank, nid in enumerate(r):
            s[nid] += w / (k + rank + 1)
    return sorted(s, key=s.get, reverse=True)


def ppr_rerank(seed_rank, edges, cand, etypes, k_seed=10, alpha=0.5, blend=0.5):
    g = nx.Graph()
    g.add_nodes_from(cand)
    g.add_edges_from((e["source"], e["target"]) for e in edges
                     if e["type"] in etypes and e["source"] in cand and e["target"] in cand)
    seeds = seed_rank[:k_seed]
    pers = {n: 1.0 / (i + 1) for i, n in enumerate(seeds)}
    pr = nx.pagerank(g, alpha=alpha, personalization=pers, max_iter=200, tol=1e-8)
    base = {n: 1.0 / (i + 1) for i, n in enumerate(seed_rank[:200])}
    prmax = max(pr.values()) or 1
    score = {n: blend * base.get(n, 0) + (1 - blend) * pr.get(n, 0) / prmax for n in cand}
    return sorted(cand, key=lambda n: (-score[n], seed_rank.index(n) if n in base else 1e9))


def hop1_expand(seed_rank, edges, cand, etypes, k_seed=3):
    adj = defaultdict(list)
    for e in edges:
        if e["type"] in etypes:
            adj[e["source"]].append(e["target"]); adj[e["target"]].append(e["source"])
    out = []
    for s in seed_rank[:k_seed]:
        out.append(s)
    for s in seed_rank[:k_seed]:
        for nb in adj[s]:
            if nb in cand and nb not in out:
                out.append(nb)
    out += [n for n in seed_rank if n not in out]
    return out


def mention_rank(issue, nodes, cand):
    words = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", issue))
    dotted = set(re.findall(r"[A-Za-z_][\w]*(?:\.[A-Za-z_]\w*)+", issue))
    scored = []
    for nid in cand:
        short = nid.rsplit(".", 1)[-1]
        s = 0
        if any(nid.endswith(d) for d in dotted):
            s += 3
        if short in words and short.lower() not in STOP and len(short) > 3:
            s += 1 + (short.count("_") > 0 or any(c.isupper() for c in short[1:]))
        if s:
            scored.append((s, -len(nodes[nid]["text"]), nid))
    return [n for *_, n in sorted(scored, reverse=True)]


def metrics(rank, gold, nodes, gold_files):
    gset = set(gold)
    pos = [i for i, n in enumerate(rank) if n in gset]
    first = pos[0] if pos else 10 ** 9
    files_rank = list(dict.fromkeys(nodes[n]["file"] for n in rank))
    fpos = [i for i, f in enumerate(files_rank) if f in set(gold_files)]
    ffirst = fpos[0] if fpos else 10 ** 9
    return {"acc@1": first < 1, "acc@5": first < 5, "acc@10": first < 10, "acc@20": first < 20,
            "all@10": len([p for p in pos if p < 10]) == len(gset),
            "mrr": 1 / (first + 1) if pos else 0.0,
            "file@1": ffirst < 1, "file@3": ffirst < 3}


def main():
    all_rank = defaultdict(dict)
    per = defaultdict(list)
    for ti, t in enumerate(tasks):
        iid = t["instance_id"]
        nodes, edges = load_graph(iid)
        cand = [n for n, d in nodes.items() if d["kind"] in ("function", "class")]
        cset = set(cand)
        issue = t["problem_statement"][:6000]
        ranks = {}
        # BM25
        bm = BM25Okapi([toks(n) * 3 + toks(nodes[n]["text"][:4000]) for n in cand])
        sc = bm.get_scores(toks(issue))
        ranks["bm25"] = [cand[i] for i in np.argsort(-sc)]
        # dense
        for m in DENSE:
            tag = m.split("/")[-1]
            cache = f"data/emb/{iid}__{tag}.npy"
            if os.path.exists(cache):
                D = np.load(cache)
            else:
                D = embed(m, [doc_text(nodes[n]) for n in cand], False)
                np.save(cache, D)
            q = embed(m, [issue[:2000]], True)[0]
            ranks[f"dense:{tag}"] = [cand[i] for i in np.argsort(-(D @ q))]
            # 256-d truncation ablation (competition ships 256-d node embeddings)
            D2 = D[:, :256] / np.linalg.norm(D[:, :256], axis=1, keepdims=True)
            q2 = q[:256] / np.linalg.norm(q[:256])
            ranks[f"dense256:{tag}"] = [cand[i] for i in np.argsort(-(D2 @ q2))]
            # fair 256-d: PCA fitted on this snapshot's node embeddings
            mu = D.mean(0); U = np.linalg.svd(D - mu, full_matrices=False)[2][:256].T
            P = (D - mu) @ U; P /= np.linalg.norm(P, axis=1, keepdims=True)
            qp = (q - mu) @ U; qp /= np.linalg.norm(qp)
            ranks[f"densepca256:{tag}"] = [cand[i] for i in np.argsort(-(P @ qp))]
        ranks["mention"] = mention_rank(issue, nodes, cand)
        d0 = f"dense:{DENSE[0].split('/')[-1]}"
        ranks["fuse"] = rrf(ranks["bm25"][:200], ranks[d0][:200], ranks["mention"][:50],
                            weights=[1, 1, 2])
        ranks["fuse"] += [n for n in ranks["bm25"] if n not in set(ranks["fuse"])]
        ranks["fuse2"] = rrf(ranks[d0][:300], ranks["bm25"][:300], weights=[2, 1])
        ranks["fuse2"] += [n for n in ranks[d0] if n not in set(ranks["fuse2"])]
        for en, es in EDGE_SETS.items():
            ranks[f"fuse+ppr:{en}"] = ppr_rerank(ranks["fuse"], edges, cset, es)
            ranks[f"fuse+hop1:{en}"] = hop1_expand(ranks["fuse"], edges, cset, es)
            for b in (0.5, 0.8):
                ranks[f"dense+ppr{b}:{en}"] = ppr_rerank(ranks[d0], edges, cset, es, blend=b)
            ranks[f"dense+hop1:{en}"] = hop1_expand(ranks[d0], edges, cset, es)
        for k, r in ranks.items():
            all_rank[k][iid] = r[:100]
            m = metrics(r, t["gold_nodes"], nodes, t["gold_files"])
            m.update(method=k, iid=iid, split=t["split"], repo=t["repo"])
            per[k].append(m)
        print(ti, iid, {k: per[k][-1]["acc@5"] for k in ("bm25", d0, "fuse2", "dense+ppr0.8:calls")},
              flush=True)
    import pandas as pd
    df = pd.DataFrame([m for v in per.values() for m in v])
    df.to_csv("results/per_task.csv", index=False)
    cols = ["acc@1", "acc@5", "acc@10", "acc@20", "all@10", "mrr", "file@1", "file@3"]
    summ = df.groupby(["method"])[cols].mean().round(3)
    summ_split = df.groupby(["split", "method"])[cols].mean().round(3)
    summ.to_csv("results/metrics.csv"); summ_split.to_csv("results/metrics_split.csv")
    print(summ.sort_values("acc@5", ascending=False).to_string())
    json.dump(all_rank, open("results/rankings.json", "w"))


if __name__ == "__main__":
    main()
