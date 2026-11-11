"""Merge the two cross-repo LoRA folds and compare against zero-shot Gemma (same prompt, same pool)."""
import json
import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
load = lambda p: {json.loads(l)["iid"]: json.loads(l) for l in open(p, encoding="utf-8")}
zs = load("results/gemma_gemma-4-E4B-it_fuse2_plain_k20.jsonl")
lora = {}
for f in ("loc_comp2held", "loc_held2comp"):
    try:
        lora.update(load(f"results/gemma_gemma-4-E4B-it_fuse2_plain_k20_lora-{f}.jsonl"))
    except FileNotFoundError:
        print("missing fold", f)
ids = sorted(lora)
with open("results/gemma_lora_crossrepo_merged.jsonl", "w", encoding="utf-8") as fo:
    for i in ids:
        fo.write(json.dumps(lora[i]) + "\n")
rows = []
for i in ids:
    rows.append(dict(iid=i, split=lora[i]["split"], pool=float(zs[i]["pool_acc1"]),
                     zs1=float(zs[i]["acc1"]), zs5=float(zs[i]["acc5"]),
                     lo1=float(lora[i]["acc1"]), lo5=float(lora[i]["acc5"]),
                     parsed=float(len(lora[i]["picks"]) > 0)))
df = pd.DataFrame(rows)


def ci(d):
    bs = [rng.choice(d, len(d)).mean() for _ in range(5000)]
    return round(d.mean(), 3), round(np.percentile(bs, 2.5), 3), round(np.percentile(bs, 97.5), 3)


out = {}
for name, sub in [("all", df)] + [(s, df[df.split == s]) for s in df.split.unique()]:
    out[name] = dict(n=len(sub), retrieval_acc1=round(sub.pool.mean(), 3),
                     zeroshot_acc1=round(sub.zs1.mean(), 3), lora_acc1=round(sub.lo1.mean(), 3),
                     zeroshot_acc5=round(sub.zs5.mean(), 3), lora_acc5=round(sub.lo5.mean(), 3),
                     d_acc1_vs_zeroshot=ci((sub.lo1 - sub.zs1).values),
                     d_acc5_vs_zeroshot=ci((sub.lo5 - sub.zs5).values),
                     d_acc1_vs_retrieval=ci((sub.lo1 - sub.pool).values),
                     parse_rate=round(sub.parsed.mean(), 3))
json.dump(out, open("results/lora_summary.json", "w"), indent=1)
print(json.dumps(out, indent=1))
