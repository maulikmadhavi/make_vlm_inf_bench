#!/usr/bin/env bash
# =============================================================================
# VLM Inference Benchmark — vllm bench CLI reference
# =============================================================================
# Two approaches for benchmarking:
#   A) Custom Python runner  → scripts/run_benchmark.py
#   B) Standard vllm bench CLI (this file)
#
# Prerequisites:
#   pip install vllm          # provides the `vllm` CLI
#   Generate datasets first:  python scripts/generate_data.py --images 100 --videos 100
#
# Set these variables to match your environment:
MODEL="Qwen/Qwen3-VL-8B-Instruct"
IMAGES_DIR="./data/images"
VIDEOS_DIR="./data/videos"
RESULT_DIR="./results"
# =============================================================================


# ---------------------------------------------------------------------------
# 1. Start the vllm server (run on GPU machine before benchmarking)
# ---------------------------------------------------------------------------

# Images only
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1}' \
  --allowed-local-media-path "$IMAGES_DIR"

# Images + videos
CUDA_VISIBLE_DEVICES=0 vllm serve "$MODEL" \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1, "video": 1}' \
  --allowed-local-media-path ./data/


# ---------------------------------------------------------------------------
# 2. Online benchmark — vllm bench serve
#    Sends live requests to a running vllm server and measures:
#    TTFT, TPOT, ITL, throughput, and goodput.
# ---------------------------------------------------------------------------

# --- 2a. Image benchmark using our generated JSONL dataset ---
vllm bench serve \
  --backend openai-chat \
  --model "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --endpoint /v1/chat/completions \
  --dataset-name custom \
  --dataset-path data/images_dataset.jsonl \
  --allowed-local-media-path "$IMAGES_DIR" \
  --num-prompts 100 \
  --request-rate inf \
  --percentile-metrics "ttft,tpot,itl" \
  --metric-percentiles "50,95,99" \
  --save-result \
  --save-detailed \
  --result-dir "$RESULT_DIR"

# --- 2b. Video benchmark using our generated JSONL dataset ---
vllm bench serve \
  --backend openai-chat \
  --model "$MODEL" \
  --host 127.0.0.1 --port 8000 \
  --endpoint /v1/chat/completions \
  --dataset-name custom \
  --dataset-path data/videos_dataset.jsonl \
  --num-prompts 100 \
  --request-rate inf \
  --percentile-metrics "ttft,tpot,itl" \
  --metric-percentiles "50,95,99" \
  --save-result \
  --save-detailed \
  --result-dir "$RESULT_DIR"

# --- 2c. Concurrency / load test (10 req/s) ---
vllm bench serve \
  --backend openai-chat \
  --model "$MODEL" \
  --endpoint /v1/chat/completions \
  --dataset-name custom \
  --dataset-path data/images_dataset.jsonl \
  --allowed-local-media-path "$IMAGES_DIR" \
  --num-prompts 100 \
  --request-rate 10 \
  --max-concurrency 10 \
  --save-result \
  --result-dir "$RESULT_DIR"

# --- 2d. Goodput SLO test (TTFT ≤ 500 ms, TPOT ≤ 50 ms) ---
vllm bench serve \
  --backend openai-chat \
  --model "$MODEL" \
  --endpoint /v1/chat/completions \
  --dataset-name custom \
  --dataset-path data/images_dataset.jsonl \
  --allowed-local-media-path "$IMAGES_DIR" \
  --num-prompts 100 \
  --goodput "ttft:500 tpot:50" \
  --save-result \
  --result-dir "$RESULT_DIR"

# --- 2e. VisionArena dataset (built-in vision QA, no local files needed) ---
vllm bench serve \
  --backend openai-chat \
  --model "$MODEL" \
  --endpoint /v1/chat/completions \
  --dataset-name vision-arena \
  --num-prompts 100 \
  --save-result \
  --result-dir "$RESULT_DIR"


# ---------------------------------------------------------------------------
# 3. Offline throughput benchmark — vllm bench throughput
#    Loads the model directly (no server needed) and measures max tokens/sec.
# ---------------------------------------------------------------------------

# Random multimodal inputs (synthetic images + text)
vllm bench throughput \
  --backend vllm \
  --model "$MODEL" \
  --dataset-name random-mm \
  --num-prompts 100 \
  --input-len 512 \
  --output-len 128 \
  --output-json "$RESULT_DIR/throughput_results.json"
