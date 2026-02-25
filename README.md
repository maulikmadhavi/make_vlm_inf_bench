# VLM Inference Bench

Benchmarks local VLM models via a running [vllm](https://github.com/vllm-project/vllm) service.
Generates datasets from real public images/videos and supports two benchmarking approaches:
the built-in **custom Python runner** and the standard **vllm bench CLI**.

## Requirements

- Python ≥ 3.10 ([Pixi](https://pixi.sh) manages the environment)
- A running vllm service on a GPU machine
- Free [Pexels API key](https://www.pexels.com/api/) for video downloads

## Quick Start

```bash
# 1. Install dependencies
pixi install

# 2. Configure (copy and edit)
cp .env.example .env

# 3. Generate datasets (downloads 100 images + 100 videos)
pixi run generate-all

# 4. Start vllm on your GPU machine
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1, "video": 1}' \
  --allowed-local-media-path ./data/

# 5. Run benchmarks — choose Approach A or B below
```

---

## Approach A — Custom Python Runner

Provides TTFT, tokens/sec, requests/sec, GPU utilization, and per-request results saved as JSON.

```bash
# Image benchmark
python scripts/run_benchmark.py \
  --dataset data/images_dataset.jsonl \
  --endpoint http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --concurrency 1 \
  --gpu-device 0

# Video benchmark
python scripts/run_benchmark.py --dataset data/videos_dataset.jsonl

# Analyze saved results
python scripts/analyze_results.py results/

# Or via Pixi tasks
pixi run bench-images
pixi run bench-videos
pixi run analyze
```

---

## Approach B — vllm bench CLI

Uses vllm's built-in benchmark tooling (`vllm bench serve` / `vllm bench throughput`).
Reports TTFT, TPOT, ITL percentiles, throughput, and goodput.
See `benchmark_scripts.sh` for the full command reference.

```bash
# Image benchmark against running server
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name custom \
  --dataset-path data/images_dataset.jsonl \
  --allowed-local-media-path ./data/images \
  --num-prompts 100 \
  --percentile-metrics "ttft,tpot,itl" \
  --metric-percentiles "50,95,99" \
  --save-result --save-detailed \
  --result-dir results/

# Video benchmark
vllm bench serve \
  --backend openai-chat \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name custom \
  --dataset-path data/videos_dataset.jsonl \
  --num-prompts 100 \
  --save-result --result-dir results/

# Offline max-throughput test (no server required)
vllm bench throughput \
  --backend vllm \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name random-mm \
  --num-prompts 100 \
  --output-json results/throughput_results.json
```

**Review results:**

```bash
# Aggregate summary table (all JSON files in results/)
python scripts/analyze_vllm_bench.py results/

# Per-request TTFT / TPOT / ITL breakdown (needs --save-detailed)
python scripts/analyze_vllm_bench.py results/ --per-request

# Show all 100 rows
python scripts/analyze_vllm_bench.py results/ --per-request --rows 100

# Or via Pixi
pixi run analyze-vllm
```

**Key flags for `vllm bench serve`:**

| Flag | Description |
|---|---|
| `--request-rate N` | Requests/sec (`inf` = burst all at once) |
| `--max-concurrency N` | Cap in-flight concurrent requests |
| `--goodput "ttft:500 tpot:50"` | SLO targets in ms |
| `--num-warmups N` | Warmup requests before measuring |
| `--save-detailed` | Include per-request data in JSON output |

---

## Approach Comparison

| Feature | Custom Python | vllm bench CLI |
|---|---|---|
| TTFT / tokens/sec / req/sec | Yes | Yes |
| TPOT, ITL percentiles | No | Yes |
| GPU utilization | Yes (pynvml) | No |
| Per-request results | Yes | With `--save-detailed` |
| Goodput / SLO testing | No | Yes |
| Load ramp-up strategies | Basic | Full |
| Offline throughput | No | Yes (`bench throughput`) |

---

## Configuration

Set via environment variables or a `.env` file (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `VLLM_ENDPOINT` | `http://localhost:8000/v1` | vllm service URL |
| `MODEL_NAME` | `Qwen/Qwen3-VL-8B-Instruct` | Model to benchmark |
| `PEXELS_API_KEY` | — | Required for video downloads |
| `DATA_DIR` | `data/` | Downloaded media cache |
| `RESULTS_DIR` | `results/` | Benchmark output directory |

## Project Structure

```
benchmark_scripts.sh  # vllm bench CLI reference commands (Approach B)
scripts/              # Custom Python CLI entry points (Approach A)
src/
  benchmarking/       # vllm client, metrics collector, benchmark runner
  data_generation/    # image/video downloaders, JSONL dataset writer
  utils/              # config, logger, JSON helpers
tests/                # pytest unit tests
```

## Running Tests

```bash
pixi run test
# or directly:
pytest tests/ -v
```
