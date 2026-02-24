"""CLI: Print a formatted summary table of benchmark results."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.json_utils import read_json
from src.utils.logger import get_logger

log = get_logger(__name__)


def _format_stat(stat: dict, key: str = "mean") -> str:
    v = stat.get(key)
    return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"


def print_results(result_path: Path) -> None:
    data = read_json(result_path)
    m = data["metrics"]
    ttft = m.get("time_to_first_token_ms", {})
    tps = m.get("tokens_per_second", {})
    gpu = m.get("gpu", {})

    print(f"\n{'='*65}")
    print(f"  File    : {result_path.name}")
    print(f"  Model   : {data.get('model', 'N/A')}")
    print(f"  Dataset : {Path(data.get('dataset', '')).name}")
    print(f"  Time    : {data.get('timestamp', 'N/A')}")
    print(f"{'='*65}")
    print(f"  Requests   total={data['total_requests']}  ok={data['successful_requests']}  err={data['failed_requests']}")
    print(f"  Elapsed    {data.get('total_elapsed_seconds', 'N/A')} s")
    print(f"{'-'*65}")
    print(f"  {'Metric':<30} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8}")
    print(f"  {'-'*58}")
    print(f"  {'TTFT (ms)':<30} {_format_stat(ttft,'mean'):>8} {_format_stat(ttft,'p50'):>8} {_format_stat(ttft,'p95'):>8} {_format_stat(ttft,'p99'):>8}")
    print(f"  {'Tokens/sec':<30} {_format_stat(tps,'mean'):>8} {_format_stat(tps,'p50'):>8} {_format_stat(tps,'p95'):>8} {_format_stat(tps,'p99'):>8}")
    print(f"  {'Requests/sec':<30} {m.get('requests_per_second', 'N/A'):>8}")
    print(f"{'-'*65}")
    if gpu.get("gpu_available"):
        print(f"  GPU compute util : {gpu['compute_utilization_pct']}%")
        print(f"  GPU memory util  : {gpu['memory_utilization_pct']}%  ({gpu['memory_used_mb']} / {gpu['memory_total_mb']} MB)")
    else:
        print("  GPU metrics      : unavailable")
    print(f"{'='*65}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze and display VLM benchmark results."
    )
    parser.add_argument("paths", nargs="+", type=Path,
                        help="One or more result JSON files or a results/ directory")
    args = parser.parse_args()

    result_files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            result_files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            result_files.append(p)
        else:
            log.warning(f"Path not found: {p}")

    if not result_files:
        print("No result files found.")
        sys.exit(1)

    for rf in result_files:
        try:
            print_results(rf)
        except Exception as e:
            log.error(f"Failed to parse {rf}: {e}")


if __name__ == "__main__":
    main()
