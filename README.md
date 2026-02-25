# VLM Inference Bench

Benchmarks local VLM models via a running [vllm](https://github.com/vllm-project/vllm) service.
Generates datasets from real public images/videos, fires requests against the OpenAI-compatible
endpoint, and reports TTFT, tokens/sec, requests/sec, and GPU utilization.

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

# 5. Run benchmarks
pixi run bench-images
pixi run bench-videos

# 6. Analyze results
pixi run analyze
```

## Configuration

Set via environment variables or a `.env` file (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `VLLM_ENDPOINT` | `http://localhost:8000/v1` | vllm service URL |
| `MODEL_NAME` | `Qwen/Qwen3-VL-8B-Instruct` | Model to benchmark |
| `PEXELS_API_KEY` | — | Required for video downloads |
| `DATA_DIR` | `data/` | Downloaded media cache |
| `RESULTS_DIR` | `results/` | Benchmark output directory |

## CLI Reference

```bash
# Generate data
python scripts/generate_data.py --images 100 --videos 100

# Run benchmark (all flags optional)
python scripts/run_benchmark.py \
  --dataset data/images_dataset.jsonl \
  --endpoint http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --concurrency 4 \
  --gpu-device 0

# Analyze one or more result files / a directory
python scripts/analyze_results.py results/
```

## Metrics

| Metric | Description |
|---|---|
| **TTFT** | Time to first token (ms) — mean, p50, p95, p99 |
| **Tokens/sec** | Generation throughput per request |
| **Requests/sec** | Overall benchmark throughput |
| **GPU util** | Average compute & memory utilization (via pynvml) |

## Project Structure

```
scripts/          # CLI entry points
src/
  benchmarking/   # vllm client, metrics collector, benchmark runner
  data_generation/# image/video downloaders, JSONL dataset writer
  utils/          # config, logger, JSON helpers
tests/            # pytest unit tests
```

## Running Tests

```bash
pixi run test
# or directly:
pytest tests/ -v
```
