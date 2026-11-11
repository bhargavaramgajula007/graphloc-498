"""QLoRA fine-tune Gemma 4 as a listwise bug localizer (cross-repository).

Train on one split (e.g. competition_repos), evaluate on the other with gemma_rerank.py
ADAPTER=adapters/<name> SPLIT=<other split>. Prompt is byte-identical to gemma_rerank.py.

usage: python src/lora_train.py <model_id> <train_split> <cond> <out_dir> [n_perm]
"""
import json, os, random, sys, time
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
from collections import defaultdict
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL, TRAIN_SPLIT, COND, OUT = sys.argv[1:5]
N_PERM = int(sys.argv[5]) if len(sys.argv) > 5 else 4
K, LR, ACC, MAXLEN = 20, 1e-4, 8, int(os.environ.get("MAXLEN", 2560))
random.seed(0); torch.manual_seed(0)


def head(text, n=6):
    lines = [l for l in text.splitlines() if l.strip()]
    return "\n".join(lines[:n])[:500]


def build_prompt(t, cands, nodes, edges):  # identical to gemma_rerank.py
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


tasks = [json.loads(l) for l in open("data/graphloc.jsonl", encoding="utf-8")]
tasks = [t for t in tasks if t["split"] == TRAIN_SPLIT]
rank = json.load(open("results/rankings.json"))["fuse2"]

proc = AutoProcessor.from_pretrained(MODEL)
tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc

examples = []
for t in tasks:
    iid = t["instance_id"]
    g = json.load(open(f"data/graphs/{iid}.json", encoding="utf-8"))
    nodes = {n["id"]: n for n in g["nodes"]}
    full = rank[iid]
    gold_in_order = [n for n in full if n in set(t["gold_nodes"])] + \
                    [n for n in t["gold_nodes"] if n not in full and n in nodes]
    pool = full[:K]
    missing = [n for n in gold_in_order[:3] if n not in pool]
    for j, n in enumerate(missing):  # inject gold at a RANDOM non-gold slot (tail injection = position leak)
        idx = random.choice([i for i, c in enumerate(pool) if c not in set(t["gold_nodes"])])
        pool[idx] = n
    for p in range(N_PERM):
        cands = list(pool)
        # keep retrieval order (a genuine prior) only when the pool was not modified
        if p > 0 or missing:
            random.shuffle(cands)
        ans = [cands.index(n) for n in gold_in_order if n in cands][:5]
        prompt = build_prompt(t, cands, nodes, g["edges"])
        user = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        ptxt = proc.apply_chat_template(user, add_generation_prompt=True, tokenize=False)
        target = "ANSWER: " + ", ".join(map(str, ans))
        convo = user + [{"role": "assistant", "content": [{"type": "text", "text": target}]}]
        ftxt = proc.apply_chat_template(convo, tokenize=False)
        assert ftxt.startswith(ptxt), "chat template prefix mismatch"
        pids = tok(ptxt, add_special_tokens=False)["input_ids"]
        fids = tok(ftxt, add_special_tokens=False)["input_ids"]
        if len(fids) > MAXLEN:
            continue
        labels = [-100] * len(pids) + fids[len(pids):]
        examples.append((fids, labels))
random.shuffle(examples)
examples = examples[: int(os.environ.get("MAXEX", 10**9))]
print(f"{len(tasks)} tasks -> {len(examples)} examples; example target:",
      tok.decode([x for x in examples[0][1] if x != -100]), flush=True)
print("max len", max(len(e[0]) for e in examples), "mean", sum(len(e[0]) for e in examples) // len(examples))
if os.environ.get("DRY"):
    sys.exit(0)

q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForImageTextToText.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="cuda",
                                                    quantization_config=q, low_cpu_mem_usage=True)
# NOTE: prepare_model_for_kbit_training upcasts all non-quantized params (incl. Gemma 4's large
# per-layer embedding tables) to fp32 -> RAM/VRAM spike. Enable checkpointing directly instead.
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model.config.use_cache = False
cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                 target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
model = get_peft_model(model, cfg)
model.print_trainable_parameters()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.0)
steps = len(examples) // ACC
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10) * max(0.05, 1 - s / max(steps, 1)))
model.train()
t0, run = time.time(), []
for i, (ids, lab) in enumerate(examples):
    x = torch.tensor([ids], device="cuda")
    n_ans = sum(1 for v in lab if v != -100)
    keep = n_ans + 1  # logits only for positions predicting answer tokens (262k vocab -> huge savings)
    out = model(input_ids=x, attention_mask=torch.ones_like(x), logits_to_keep=keep)
    logits = out.logits[0, :-1].float()            # positions L-keep .. L-2 predict tokens L-keep+1 .. L-1
    target = torch.tensor(lab[-(keep - 1):], device="cuda")
    loss = torch.nn.functional.cross_entropy(logits, target, ignore_index=-100)
    (loss / ACC).backward()
    run.append(loss.item())
    if (i + 1) % ACC == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    if (i + 1) % int(os.environ.get("LOGEVERY", 40)) == 0:
        print(f"ex {i+1}/{len(examples)} loss {sum(run[-40:])/40:.4f} lr {sched.get_last_lr()[0]:.2e} "
              f"{time.time()-t0:.0f}s vram {torch.cuda.max_memory_allocated()/2**30:.1f}G", flush=True)
        torch.cuda.reset_peak_memory_stats()
os.makedirs(OUT, exist_ok=True)
model.save_pretrained(OUT)
json.dump(dict(model=MODEL, train_split=TRAIN_SPLIT, cond=COND, n_examples=len(examples), n_perm=N_PERM,
               lr=LR, accum=ACC, r=16, alpha=32, secs=round(time.time() - t0)), open(f"{OUT}/train_meta.json", "w"))
print("saved", OUT, f"{time.time()-t0:.0f}s")
