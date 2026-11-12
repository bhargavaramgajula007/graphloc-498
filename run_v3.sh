#!/usr/bin/env bash
# LoRA v3 (order-preserving): fold A = train on competition repos, evaluate on held-out repos.
# Usage (from repo root, Git Bash):  bash run_v3.sh            # fold A only (~35 min)
#                                     bash run_v3.sh both       # fold A + fold B (~70 min)
set -e
cd "$(dirname "$0")"
source .venv/Scripts/activate
export LOGEVERY=100 LR=5e-5
echo "[$(date -u +%H:%M)] fold A: train"
python src/lora_train.py google/gemma-4-E4B-it competition_repos plain adapters/loc_comp2held 2 > results/lora_comp2held.log 2>&1
echo "[$(date -u +%H:%M)] fold A: eval on held-out repos"
ADAPTER=adapters/loc_comp2held SPLIT=heldout_repos python src/gemma_rerank.py google/gemma-4-E4B-it fuse2 plain 20 > results/lora_eval_comp2held.log 2>&1
if [ "$1" = "both" ]; then
  echo "[$(date -u +%H:%M)] fold B: train"
  python src/lora_train.py google/gemma-4-E4B-it heldout_repos plain adapters/loc_held2comp 2 > results/lora_held2comp.log 2>&1
  echo "[$(date -u +%H:%M)] fold B: eval on competition repos"
  ADAPTER=adapters/loc_held2comp SPLIT=competition_repos python src/gemma_rerank.py google/gemma-4-E4B-it fuse2 plain 20 > results/lora_eval_held2comp.log 2>&1
fi
echo "[$(date -u +%H:%M)] summary"
python src/lora_summary.py
