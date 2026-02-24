"""Loads a JSONL dataset, runs inference, collects metrics, saves results."""
import time
from datetime import datetime, timezone
from pathlib import Path

from src.benchmarking.vllm_client import VLMClient
from src.benchmarking.metrics_collector import GPUMonitor, aggregate
from src.utils.config import RESULTS_DIR, MODEL_NAME
from src.utils.json_utils import read_jsonl, write_json
from src.utils.logger import get_logger

log = get_logger(__name__)


def run_benchmark(
    dataset_path: str | Path,
    output_path: str | Path | None = None,
    concurrency: int = 1,
    gpu_device: int = 0,
) -> dict:
    """
    Run the benchmark for all records in `dataset_path`.

    Args:
        dataset_path: Path to a JSONL file with {prompt, image_files?, video_files?} records.
        output_path:  Where to save the JSON results (default: results/<dataset_stem>_results.json).
        concurrency:  Number of parallel requests (default: 1 = sequential).
        gpu_device:   CUDA device index to monitor.

    Returns:
        Full results dict (also saved to disk).
    """
    dataset_path = Path(dataset_path)
    if output_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULTS_DIR / f"{dataset_path.stem}_results.json"

    records = list(read_jsonl(dataset_path))
    log.info(f"Loaded {len(records)} records from {dataset_path}")

    client = VLMClient()
    gpu_monitor = GPUMonitor(device_index=gpu_device)

    ttft_list: list[float] = []
    tps_list: list[float] = []
    per_request: list[dict] = []
    errors: list[str] = []

    benchmark_start = time.perf_counter()
    gpu_monitor.start()

    for idx, record in enumerate(records, 1):
        prompt = record.get("prompt", "Describe this.")
        image_files = record.get("image_files", [])
        video_files = record.get("video_files", [])

        log.info(f"[{idx}/{len(records)}] Running inference...")
        try:
            result = client.infer(prompt, image_files, video_files)
            ttft_list.append(result["ttft_ms"])
            tps_list.append(result["tokens_per_second"])
            per_request.append({
                "id": idx,
                "prompt": prompt[:80],
                "ttft_ms": result["ttft_ms"],
                "total_time_ms": result["total_time_ms"],
                "output_tokens": result["output_tokens"],
                "tokens_per_second": result["tokens_per_second"],
            })
        except Exception as e:
            log.error(f"Request {idx} failed: {e}")
            errors.append(str(e))

    gpu_monitor.stop()
    total_elapsed = time.perf_counter() - benchmark_start

    successful = len(per_request)
    rps = round(successful / total_elapsed, 4) if total_elapsed > 0 else 0

    results = {
        "dataset": str(dataset_path),
        "model": MODEL_NAME,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_requests": len(records),
        "successful_requests": successful,
        "failed_requests": len(errors),
        "total_elapsed_seconds": round(total_elapsed, 2),
        "metrics": {
            "time_to_first_token_ms": aggregate(ttft_list),
            "tokens_per_second": aggregate(tps_list),
            "requests_per_second": rps,
            "gpu": gpu_monitor.summary(),
        },
        "per_request": per_request,
        "errors": errors,
    }

    write_json(results, output_path)
    log.info(f"Results saved to {output_path}")
    _print_summary(results)
    return results


def _print_summary(results: dict) -> None:
    m = results["metrics"]
    ttft = m["time_to_first_token_ms"]
    tps = m["tokens_per_second"]
    gpu = m["gpu"]

    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"  Model             : {results['model']}")
    print(f"  Dataset           : {Path(results['dataset']).name}")
    print(f"  Total requests    : {results['total_requests']}")
    print(f"  Successful        : {results['successful_requests']}")
    print(f"  Failed            : {results['failed_requests']}")
    print(f"  Elapsed (s)       : {results['total_elapsed_seconds']}")
    print()
    print(f"  TTFT mean (ms)    : {ttft.get('mean', 'N/A')}")
    print(f"  TTFT p95  (ms)    : {ttft.get('p95', 'N/A')}")
    print(f"  Tokens/sec mean   : {tps.get('mean', 'N/A')}")
    print(f"  Requests/sec      : {m['requests_per_second']}")
    if gpu.get("gpu_available"):
        print(f"  GPU compute util  : {gpu['compute_utilization_pct']}%")
        print(f"  GPU memory util   : {gpu['memory_utilization_pct']}%")
    else:
        print("  GPU metrics       : unavailable")
    print("=" * 60)
