#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3}"
CONFIG="${CONFIG:-configs/zh_study.json}"
RUN_DIR="$("$PYTHON_BIN" -c 'import json,sys; c=json.load(open(sys.argv[1], encoding="utf-8")); print(c["paths"]["runs_dir"] + "/" + c["pilot"].get("run_name", "pilot"))' "$CONFIG")"
MODELS_DIR="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["paths"]["models_dir"])' "$CONFIG")"

mapfile -t MODELS < <(
  "$PYTHON_BIN" -c 'import json,sys; print(*json.load(open(sys.argv[1], encoding="utf-8"))["pilot"]["model_pool"], sep="\n")' "$CONFIG"
)

missing=()
for model in "${MODELS[@]}"; do
  [[ -d "$MODELS_DIR/$model" ]] || missing+=("$MODELS_DIR/$model")
done
if ((${#missing[@]})); then
  printf 'Missing configured model directories:\n' >&2
  printf '  %s\n' "${missing[@]}" >&2
  exit 1
fi

"$PYTHON_BIN" -B scripts/validate_zh_study.py
"$PYTHON_BIN" -B scripts/validate_seed_bank.py \
  --config "$CONFIG" --require-verified
"$PYTHON_BIN" -B auto_labeling/validate_detector.py \
  --config "$CONFIG" \
  --gold "$RUN_DIR/construct_gold.jsonl" \
  --output reports_zh/detector_validation.json

for model in "${MODELS[@]}"; do
  "$PYTHON_BIN" -B scripts/run_vllm_inference.py \
    --config "$CONFIG" \
    --tasks "$RUN_DIR/generation_tasks.jsonl" \
    --model "$model"
done

"$PYTHON_BIN" -B auto_labeling/auto_label.py --config "$CONFIG" --arm injected

for model in "${MODELS[@]}"; do
  "$PYTHON_BIN" -B scripts/run_vllm_inference.py \
    --config "$CONFIG" \
    --tasks "$RUN_DIR/selfinduced_stage1_tasks.jsonl" \
    --model "$model"
done

"$PYTHON_BIN" -B scripts/build_selfinduced_stage2.py --config "$CONFIG"

for model in "${MODELS[@]}"; do
  "$PYTHON_BIN" -B scripts/run_vllm_inference.py \
    --config "$CONFIG" \
    --tasks "$RUN_DIR/selfinduced_stage2_tasks.jsonl" \
    --model "$model"
done

"$PYTHON_BIN" -B auto_labeling/auto_label.py --config "$CONFIG" --arm selfinduced
