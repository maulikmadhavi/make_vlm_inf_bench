"""CLI: Run VLM inference benchmark against a running vllm service."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking.benchmark_runner import run_benchmark
from src.utils.config import VLLM_ENDPOINT, MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a local vllm service with a JSONL dataset."
    )
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Path to JSONL dataset (images or videos)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save JSON results (default: results/<dataset>_results.json)")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of concurrent requests (default: 1)")
    parser.add_argument("--gpu-device", type=int, default=0,
                        help="CUDA device index to monitor (default: 0)")
    parser.add_argument("--endpoint", type=str, default=VLLM_ENDPOINT,
                        help=f"vllm endpoint URL (default: {VLLM_ENDPOINT})")
    parser.add_argument("--model", type=str, default=MODEL_NAME,
                        help=f"Model name (default: {MODEL_NAME})")
    args = parser.parse_args()

    if not args.dataset.exists():
        print(f"Error: dataset not found: {args.dataset}")
        sys.exit(1)

    run_benchmark(
        dataset_path=args.dataset,
        output_path=args.output,
        concurrency=args.concurrency,
        gpu_device=args.gpu_device,
        endpoint=args.endpoint,
        model=args.model,
    )


if __name__ == "__main__":
    main()
