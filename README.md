# ECHO

This is the official code repository for **ECHO: Replacing the Compliance Proxy with Response Harmfulness in Optimization-Based Jailbreak Attack**.

Our paper has been accepted to the **EMNLP 2026 Main Conference**.

ECHO searches for jailbreak prompts using HarmSig, a response-harmfulness scorer based on Qwen3-1.7B.

## Run the main experiment

Use Python 3.10+, a CUDA GPU, and access to the target model. Run these commands from the repository root. The HarmSig `best_model.pt` checkpoint is hosted at [ztaenn/harmsig](https://huggingface.co/ztaenn/harmsig).

```bash
pip install -r requirements.txt
hf auth login
hf download Qwen/Qwen3-1.7B --local-dir ../echo-models/qwen3-1.7b
hf download ztaenn/harmsig best_model.pt --local-dir ../echo-models/harmsig

export ECHO_TARGET_MODEL=meta-llama/Llama-2-7b-chat-hf
export ECHO_EVALUATOR_BACKBONE=../echo-models/qwen3-1.7b
export ECHO_EVALUATOR_CKPT=../echo-models/harmsig/best_model.pt
ECHO_RUN_SR=0 bash scripts/run_full520_llama7b.sh
```

The script runs the 520 goals in `data/eval_full_520.csv` and writes logs and a summary under `results/runs/main_experiment/`. Access to the gated Llama 2 model is required. Set `ECHO_RUN_SR=1` to also score outputs with StrongREJECT; this additionally requires access to its gated Gemma base model.
