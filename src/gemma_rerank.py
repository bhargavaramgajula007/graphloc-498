"""Gemma 4 listwise reranking of top-K localization candidates.

Conditions (same candidate pool, only the prompt differs):
  plain   : candidate = symbol id + signature/docstring head
  graph   : plain + its call-graph context (callers / callees ids)
Pools: 'fuse' (lexical+dense) and 'fuse+ppr:calls' (graph-propagated).

usage: python src/gemma_rerank.py <model_id> <pool_method> <cond> [K]
"""
import json, os, re, sys, time
from collections import defaultdict
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, AutoTokenizer, AutoModelForCausalLM

MODEL, POOL, COND = sys.argv[1], sys.argv[2], sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 20
tag = MODEL.split("/")[-1]
ADAPTER = os.environ.get("ADAPTER")  # optional PEFT LoRA dir
SPLIT = os.environ.get("SPLIT")      # optional: only evaluate this split
OUT = f"results/gemma_{tag}_{POOL.replace(':', '-')}_{COND}_k{K}" +       (f"_lora-{os.path.basename(ADAPTER.rstrip('/'))}" if ADAPTER else "") + ".jsonl"

tasks = [json.loads(l) for l in open("data/graphloc.jsonl", encoding="utf-8")]
rankings = json.load(open(os.environ.get("RANKINGS", "results/rankings.json")))[POOL]
done = set()
if os.path.exists(OUT):
    done = {json.loads(l)["iid"] for l in open(OUT, encoding="utf-8")}

try:
    tok = AutoProcessor.from_pretrained(MODEL)
    from transformers import BitsAndBytesConfig
    q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                        quantization_config=q)
except Exception as e:
    print("falling back to CausalLM:", e)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda")
if ADAPTER:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER)
model.eval()


def head(text, n=6):
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines[:n])[:500]


def build_prompt(t, cands, nodes, edges):
    nbr_out, nbr_in = defaultdict(list), defaultdict(list)
    for e in edges:
        if e["type"] == "calls":
            nbr_out[e["source"]].append(e["target"]); nbr_in[e["target"]].append(e["source"])
    blocks = []
    for i, c in enumerate(cands):
        b = f"[{i}] {c}\n{head(nodes[c]['text'])}"
        if COND == "graph":
            short = lambda xs: ", ".join(x.rsplit(".", 2)[-2] + "." + x.rsplit(".", 1)[-1] for x in xs[:6]) or "-"
            b += f"\n  calls: {short(nbr_out[c])}\n  called by: {short(nbr_in[c])}"
        blocks.append(b)
    return (
        "You are localizing a bug in the Python repository " + t["repo"] + ".\n\n"
        "## Issue\n" + t["problem_statement"][:3000] + "\n\n"
        "## Candidate functions/classes\n" + "\n\n".join(blocks) + "\n\n"
        "Which candidates must be EDITED to resolve the issue? Do not explain. Reply with exactly one "
        "line `ANSWER: i, j, k` listing up to 5 candidate indices, most likely first.")


def parse(text, n):
    m = re.findall(r"ANSWER:\s*([\d,\s\[\]]+)", text)
    src = m[-1] if m else text[-200:]
    out = []
    for x in re.findall(r"\d+", src):
        x = int(x)
        if x < n and x not in out:
            out.append(x)
    return out


with open(OUT, "a", encoding="utf-8") as f:
    for t in tasks[: int(os.environ.get("LIMIT", 10**9))]:
        iid = t["instance_id"]
        if iid in done or (SPLIT and t["split"] != SPLIT):
            continue
        g = json.load(open(f"data/graphs/{iid}.json", encoding="utf-8"))
        nodes = {n["id"]: n for n in g["nodes"]}
        cands = rankings[iid][:K]
        prompt = build_prompt(t, cands, nodes, g["edges"])
        msgs = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                         return_dict=True, return_tensors="pt").to("cuda")
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        text = tok.decode(gen[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        picks = parse(text, len(cands))
        reranked = [cands[i] for i in picks] + [c for j, c in enumerate(cands) if j not in picks]
        gold = set(t["gold_nodes"])
        rec = dict(iid=iid, split=t["split"], picks=picks, reranked=reranked,
                   prompt_tokens=int(inputs["input_ids"].shape[1]), secs=round(time.time() - t0, 1),
                   pool_acc1=cands[0] in gold if cands else False,
                   acc1=bool(reranked and reranked[0] in gold),
                   acc5=any(c in gold for c in reranked[:5]), in_pool=any(c in gold for c in cands),
                   raw=text[-600:])
        f.write(json.dumps(rec) + "\n"); f.flush()
        print(iid, rec["acc1"], rec["acc5"], rec["in_pool"], rec["secs"], "s", flush=True)
