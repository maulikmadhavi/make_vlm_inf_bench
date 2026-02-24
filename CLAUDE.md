# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**vlm_inference_bench** benchmarks local VLM models via a running vllm service. It generates random datasets (100 real images + 100 real videos from public sources), submits them to vllm's OpenAI-compatible endpoint, and reports performance metrics.

Tracked metrics: **Time to First Token (TTFT)**, **tokens/sec**, **requests/sec**, **GPU utilization**.

- **Package Manager**: Pixi (`pixi.toml`)
- **Platform**: win-64, but inference service runs on Linux/GPU (NAS/remote)
- **vllm endpoint**: `http://localhost:8000/v1` (configurable via `VLLM_ENDPOINT` env var)

## Project Structure

```
vlm_inference_bench/
├── data/                              # Generated JSONL datasets (gitignored)
│   ├── images/                        # Downloaded real images cache
│   ├── videos/                        # Downloaded real videos cache
│   ├── images_dataset.jsonl
│   └── videos_dataset.jsonl
├── results/                           # Benchmark outputs (gitignored)
├── scripts/
│   ├── generate_data.py               # CLI: download media + write JSONL
│   ├── run_benchmark.py               # CLI: run benchmark against vllm
│   └── analyze_results.py             # CLI: print summary table
├── src/
│   ├── data_generation/
│   │   ├── dataset_generator.py       # Orchestrates JSONL generation
│   │   ├── image_downloader.py        # Downloads images from COCO/Unsplash
│   │   ├── video_downloader.py        # Downloads videos from Pexels/Pixabay
│   │   └── prompt_generator.py        # Random prompt pool for VLMs
│   ├── benchmarking/
│   │   ├── vllm_client.py             # Streaming OpenAI client wrapper
│   │   ├── metrics_collector.py       # Aggregates TTFT, tps, rps, GPU util
│   │   └── benchmark_runner.py        # Loads JSONL, fires requests, saves results
│   └── utils/
│       ├── config.py                  # Central config (env vars + defaults)
│       ├── logger.py                  # Shared logger
│       └── json_utils.py              # JSONL read/write helpers
├── benchmark_scripts.sh               # Reference vllm CLI commands
├── sample.jsonl                       # Image JSONL example
├── sample_v.jsonl                     # Video JSONL example
└── pixi.toml
```

## JSONL Formats

Image dataset (matches `sample.jsonl`):
```json
{"prompt": "Describe this image in detail.", "image_files": ["/abs/path/image.jpg"]}
```

Video dataset (matches `sample_v.jsonl`):
```json
{"prompt": "Describe in detail what happens in this video.", "video_files": ["/abs/path/video.mp4"]}
```

## Common Commands

```bash
# Generate datasets (downloads 100 images + 100 videos)
python scripts/generate_data.py --images 100 --videos 100

# Start vllm service (run on GPU machine)
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --limit-mm-per-prompt '{"image": 1, "video": 1}' \
  --allowed-local-media-path ./data/

# Run image benchmark
python scripts/run_benchmark.py --dataset data/images_dataset.jsonl

# Run video benchmark
python scripts/run_benchmark.py --dataset data/videos_dataset.jsonl

# Analyze results
python scripts/analyze_results.py results/

# Run tests
pytest tests/ -v
pytest tests/test_data_generation.py -v   # single module
```

## Key Architecture Decisions

- **Streaming requests**: `vllm_client.py` uses SSE streaming to capture TTFT accurately (time from request send until first token chunk arrives).
- **GPU monitoring**: `metrics_collector.py` polls `pynvml` in a background thread during each request; results are averaged per-request then aggregated across runs.
- **Concurrency**: `benchmark_runner.py` sends requests sequentially by default (consistent measurement); pass `--concurrency N` for parallel load.
- **Public media sources**: COCO for images (no API key needed), Pexels API for videos (free API key required, set `PEXELS_API_KEY` env var).
- **Caching**: Downloaded media stored in `data/images/` and `data/videos/` — re-running `generate_data.py` skips already-downloaded files.

## Configuration

Set via environment variables (or `.env` file):
```
VLLM_ENDPOINT=http://localhost:8000/v1
MODEL_NAME=Qwen/Qwen3-VL-8B-Instruct
PEXELS_API_KEY=<your_key>               # Required for video downloads
RESULTS_DIR=results/
DATA_DIR=data/
```

## Dependencies (pixi.toml)

`python`, `openai`, `pynvml`, `requests`, `pillow`, `numpy`, `tqdm`, `pytest`, `tabulate`, `python-dotenv`
