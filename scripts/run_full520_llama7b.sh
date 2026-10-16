#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXPERIMENT_NAME="${ECHO_EXPERIMENT_NAME:-main_experiment}"
OUT_DIR="${1:-$ROOT/results/runs/$EXPERIMENT_NAME}"

TARGET_MODEL="${ECHO_TARGET_MODEL:?Set ECHO_TARGET_MODEL to the target model directory or Hugging Face model id}"
MODEL_NAME="${ECHO_MODEL_NAME:-Llama_2_7b_chat_hf}"
EVALUATOR_CKPT="${ECHO_EVALUATOR_CKPT:?Set ECHO_EVALUATOR_CKPT to the HarmSig checkpoint path}"
EVALUATOR_BACKBONE="${ECHO_EVALUATOR_BACKBONE:?Set ECHO_EVALUATOR_BACKBONE to the evaluator backbone directory or model id}"
GPU="${CUDA_VISIBLE_DEVICES:-${ECHO_GPU:-0}}"
SR_PYTHON="${ECHO_SR_PYTHON:-python}"
RUN_SR="${ECHO_RUN_SR:-1}"
WEIGHTS="core=0.2258422931,periph=0.01282675773,pressure=0.5420793309,delete=0.05119949490,swap=0.1680521234"

mkdir -p "$OUT_DIR"
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"
export NLTK_DATA="$ROOT:${NLTK_DATA:-}"
export CUDA_VISIBLE_DEVICES="$GPU"

ATTACK_JSONL="$OUT_DIR/attack_full_${MODEL_NAME}.jsonl"

python "$ROOT/src/run_obfuscation_beam_v2_trace.py"   --csv_path "$ROOT/data/eval_full_520.csv"   --max_goals 520   --target_model "$TARGET_MODEL"   --model_name "$MODEL_NAME"   --device cuda:0   --save_path "$ATTACK_JSONL"   --max_steps 20   --beam_width 10   --branch_factor 5   --gpu_memory_utilization 0.75   --early_stop_score 0.8   --max_model_len 4096   --max_new_tokens 300   --top_k_trace 5   --evaluator_ckpt "$EVALUATOR_CKPT"   --evaluator_backbone "$EVALUATOR_BACKBONE"   --operator_weights "$WEIGHTS"

cat > "$OUT_DIR/manifest.json" <<MANIFEST
{
  "experiment": "$EXPERIMENT_NAME",
  "tier": "full",
  "models": ["$MODEL_NAME"],
  "csv": "data/eval_full_520.csv",
  "algorithm": "src/run_obfuscation_beam_v2_trace.py",
  "max_steps": 20,
  "beam_width": 10,
  "branch_factor": 5,
  "gpu_memory_utilization": 0.75,
  "early_stop_score": 0.8,
  "top_k_trace": 5,
  "operator_weights": "$WEIGHTS",
  "run_sr": $RUN_SR
}
MANIFEST

if [[ "$RUN_SR" == "1" ]]; then
  "$SR_PYTHON" "$ROOT/scripts/eval_main_results_strongreject.py"     --gpu 0     --batch-size "${ECHO_SR_BATCH_SIZE:-8}"     --files "$ATTACK_JSONL"     --overwrite
  python "$ROOT/scripts/tiered_report.py" "$OUT_DIR"
else
  python "$ROOT/scripts/tiered_report.py" "$OUT_DIR" --allow-no-sr
fi

TOPK_DIR="$OUT_DIR/topk_sr_curve"
EXPANDED_JSONL="$TOPK_DIR/expanded_all_by_proxy.jsonl"
python "$ROOT/scripts/expand_topk_from_trace.py" "$ATTACK_JSONL" --out "$EXPANDED_JSONL"
if [[ "$RUN_SR" == "1" ]]; then
  "$SR_PYTHON" "$ROOT/scripts/eval_main_results_strongreject.py"     --gpu 0     --batch-size "${ECHO_SR_BATCH_SIZE:-8}"     --files "$EXPANDED_JSONL"     --overwrite
  python "$ROOT/scripts/compute_topm_curve.py" "${EXPANDED_JSONL%.jsonl}_sr.jsonl"     --out "$TOPK_DIR/mean_sr_by_k.csv"     --max-k "${ECHO_TOPM_MAX_K:-25}"
fi

echo "Done. Results: $OUT_DIR"
