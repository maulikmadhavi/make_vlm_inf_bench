"""CLI: Print a formatted summary of vllm bench CLI result JSON files.

vllm bench serve --save-result produces a different schema than our custom
runner. This script handles that format, including per-request detail when
--save-detailed was used.

Usage:
    python scripts/analyze_vllm_bench.py results/
    python scripts/analyze_vllm_bench.py results/bench_*.json
    python scripts/analyze_vllm_bench.py results/run.json --per-request
"""
import argparse
import json
import sys
from pathlib import Path


def _ms(v) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) else "N/A"


def _tok(v) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"


def print_summary(data: dict, path: Path) -> None:
    """Print the top-level aggregate metrics vllm bench always writes."""
    W = 68
    print(f"\n{'='*W}")
    print(f"  File    : {path.name}")
    print(f"  Model   : {data.get('model_id', data.get('model', 'N/A'))}")
    print(f"  Backend : {data.get('backend', 'N/A')}")
    print(f"  Date    : {data.get('date', 'N/A')}")
    print(f"{'='*W}")

    completed  = data.get("completed", "N/A")
    total      = data.get("num_prompts", "N/A")
    duration   = data.get("duration", None)
    dur_str    = f"{duration:.1f} s" if isinstance(duration, (int, float)) else "N/A"
    in_tok     = data.get("total_input_tokens", "N/A")
    out_tok    = data.get("total_output_tokens", "N/A")

    print(f"  Requests   completed={completed}/{total}   duration={dur_str}")
    print(f"  Tokens     input={in_tok}   output={out_tok}")
    print(f"{'-'*W}")

    # Throughput
    req_tput  = data.get("request_throughput")
    out_tput  = data.get("output_token_throughput")
    tot_tput  = data.get("total_token_throughput")
    print(f"  {'Throughput':<35} {'Value':>10}")
    print(f"  {'-'*46}")
    print(f"  {'Requests/sec':<35} {_tok(req_tput):>10}")
    print(f"  {'Output tokens/sec':<35} {_tok(out_tput):>10}")
    print(f"  {'Total tokens/sec':<35} {_tok(tot_tput):>10}")
    print(f"{'-'*W}")

    # Latency table
    metrics = [
        ("TTFT (ms)",       "ttft"),
        ("TPOT (ms)",       "tpot"),
        ("ITL  (ms)",       "itl"),
        ("E2E latency (ms)","e2el"),
    ]
    print(f"  {'Metric':<28} {'Mean':>8} {'Median':>8} {'Std':>8} {'P99':>8}")
    print(f"  {'-'*61}")
    for label, key in metrics:
        mean   = data.get(f"mean_{key}_ms")
        median = data.get(f"median_{key}_ms")
        std    = data.get(f"std_{key}_ms")
        p99    = data.get(f"p99_{key}_ms")
        if any(v is not None for v in [mean, median, std, p99]):
            print(f"  {label:<28} {_ms(mean):>8} {_ms(median):>8} {_ms(std):>8} {_ms(p99):>8}")

    # Goodput (only present when --goodput SLOs were set)
    goodput = data.get("goodput")
    if goodput is not None:
        print(f"{'-'*W}")
        print(f"  Goodput (req/s meeting SLO) : {_tok(goodput)}")

    print(f"{'='*W}\n")


def print_per_request(data: dict, n: int = 20) -> None:
    """Print a per-request table if --save-detailed data is present."""
    ttfts  = data.get("ttfts",  [])
    tpots  = data.get("tpots",  [])
    itls   = data.get("itls",   [])
    out_lens = data.get("output_lens", [])
    errors = data.get("errors", [])

    if not ttfts:
        print("  No per-request detail found (re-run with --save-detailed).\n")
        return

    rows = min(n, len(ttfts))
    print(f"  Per-request breakdown (first {rows} of {len(ttfts)}):")
    print(f"  {'#':>4}  {'TTFT ms':>9}  {'TPOT ms':>9}  {'ITL ms':>9}  {'out tok':>8}  {'error'}")
    print(f"  {'-'*62}")
    for i in range(rows):
        ttft_v = ttfts[i] * 1000 if i < len(ttfts) else None
        tpot_v = tpots[i] * 1000 if i < len(tpots) else None
        # itls is a list-of-lists (one list per request)
        itl_mean = None
        if i < len(itls) and isinstance(itls[i], list) and itls[i]:
            itl_mean = sum(itls[i]) / len(itls[i]) * 1000
        elif i < len(itls) and isinstance(itls[i], (int, float)):
            itl_mean = itls[i] * 1000
        out_len = out_lens[i] if i < len(out_lens) else None
        err     = errors[i]   if i < len(errors)   else ""
        print(f"  {i+1:>4}  {_ms(ttft_v):>9}  {_ms(tpot_v):>9}  {_ms(itl_mean):>9}"
              f"  {str(out_len) if out_len is not None else 'N/A':>8}  {err or ''}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze vllm bench CLI result JSON files."
    )
    parser.add_argument(
        "paths", nargs="+", type=Path,
        help="One or more result JSON files or a results/ directory",
    )
    parser.add_argument(
        "--per-request", action="store_true",
        help="Print per-request TTFT/TPOT/ITL table (requires --save-detailed)",
    )
    parser.add_argument(
        "--rows", type=int, default=20,
        help="Number of per-request rows to show (default: 20)",
    )
    args = parser.parse_args()

    files: list[Path] = []
    for p in args.paths:
        if p.is_dir():
            files.extend(sorted(p.glob("*.json")))
        elif p.is_file():
            files.append(p)
        else:
            print(f"Warning: path not found: {p}", file=sys.stderr)

    if not files:
        print("No result files found.")
        sys.exit(1)

    for f in files:
        try:
            data = json.loads(f.read_text())
        except Exception as e:
            print(f"Error reading {f}: {e}", file=sys.stderr)
            continue

        # Skip files that look like our custom-runner format
        if "metrics" in data and "time_to_first_token_ms" in data.get("metrics", {}):
            print(f"Skipping custom-runner result (use analyze_results.py): {f.name}")
            continue

        print_summary(data, f)
        if args.per_request:
            print_per_request(data, n=args.rows)


if __name__ == "__main__":
    main()
