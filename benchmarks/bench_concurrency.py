#!/usr/bin/env python3
"""
OmniVoice Server — HTTP concurrency benchmark.

Sends N parallel requests to a running omnivoice-server instance, stepping N
from 3, 6, 9, ... up to --max-concurrency, and records per-request latency,
throughput, and error rate at each level.

Usage:
    python benchmarks/bench_concurrency.py
    python benchmarks/bench_concurrency.py --url http://localhost:8880 --max-concurrency 42
    python benchmarks/bench_concurrency.py --text "The quick brown fox." --runs 3

Requires: httpx  (already a dependency of omnivoice-server)

Output:
    benchmarks/results/concurrency.csv      raw per-request data
    benchmarks/results/concurrency_report.md  summary table
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_URL = "http://localhost:8080"
DEFAULT_TEXT = (
    "This is a concurrency benchmark for the OmniVoice text-to-speech system. "
    "Each request synthesises this sentence and the harness measures latency "
    "under increasing parallel load."
)
SAMPLE_RATE = 24_000


# ── Result dataclass ─────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    concurrency: int
    run_index: int
    request_index: int
    latency_ms: float
    status_code: int
    audio_bytes: int
    error: str | None = None


# ── Single request ───────────────────────────────────────────────────────────


async def send_request(
    client: httpx.AsyncClient,
    url: str,
    text: str,
    request_index: int,
    concurrency: int,
    run_index: int,
) -> RequestResult:
    """Send one TTS request and return timing + status."""
    payload = {
        "model": "omnivoice",
        "input": text,
        "response_format": "pcm",
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{url}/v1/audio/speech", json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(
            concurrency=concurrency,
            run_index=run_index,
            request_index=request_index,
            latency_ms=round(latency_ms, 1),
            status_code=resp.status_code,
            audio_bytes=len(resp.content),
            error=None if resp.status_code == 200 else resp.text[:200],
        )
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return RequestResult(
            concurrency=concurrency,
            run_index=run_index,
            request_index=request_index,
            latency_ms=round(latency_ms, 1),
            status_code=0,
            audio_bytes=0,
            error=str(e)[:200],
        )


# ── One concurrency level ────────────────────────────────────────────────────


async def run_level(
    client: httpx.AsyncClient,
    url: str,
    text: str,
    concurrency: int,
    run_index: int,
) -> list[RequestResult]:
    """Fire `concurrency` requests in parallel, wait for all, return results."""
    tasks = [
        send_request(client, url, text, i, concurrency, run_index)
        for i in range(concurrency)
    ]
    return await asyncio.gather(*tasks)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL, help="Server base URL")
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=42,
        help="Maximum concurrency level to test (default: 42)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=7,
        help="Step size for concurrency levels: 7, 14, 21... (default: 7)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Repetitions per concurrency level (default: 1)",
    )
    parser.add_argument(
        "--text",
        default=DEFAULT_TEXT,
        help="Text to synthesise",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmarks/results",
        help="Directory for CSV and report outputs",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Per-request timeout in seconds (default: 300)",
    )
    args = parser.parse_args()

    # Build concurrency levels: 3, 6, 9, ... up to max
    levels = list(range(args.step, args.max_concurrency + 1, args.step))
    if not levels or levels[-1] != args.max_concurrency:
        levels.append(args.max_concurrency)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"OmniVoice Server Concurrency Benchmark")
    print(f"  URL:            {args.url}")
    print(f"  Concurrency:    {levels[0]} → {levels[-1]} (step {args.step})")
    print(f"  Runs per level: {args.runs}")
    print(f"  Text:           {args.text[:60]}...")
    print()

    # Warm-up single request
    print("Warm-up request...", end=" ", flush=True)
    try:
        with httpx.Client(timeout=args.timeout) as c:
            r = c.post(
                f"{args.url}/v1/audio/speech",
                json={"model": "omnivoice", "input": "warmup", "response_format": "pcm"},
            )
            print(f"status={r.status_code} bytes={len(r.content)}")
            if r.status_code != 200:
                print(f"  ⚠ Warm-up failed: {r.text[:200]}")
                print("  Is the server running? Start it first.")
                return
    except httpx.ConnectError:
        print(f"\n✗ Cannot connect to {args.url}. Is the server running?")
        return
    print()

    all_results: list[RequestResult] = []

    async def run_all():
        async with httpx.AsyncClient(timeout=args.timeout) as client:
            for level in levels:
                for run_idx in range(args.runs):
                    print(
                        f"  N={level:3d}  run {run_idx + 1}/{args.runs}  "
                        f"({level} parallel requests)...",
                        end=" ",
                        flush=True,
                    )
                    t0 = time.perf_counter()
                    results = await run_level(
                        client, args.url, args.text, level, run_idx
                    )
                    wall_s = time.perf_counter() - t0
                    all_results.extend(results)

                    lats = [r.latency_ms for r in results if r.error is None]
                    errors = sum(1 for r in results if r.error is not None)
                    if lats:
                        print(
                            f"wall={wall_s:.2f}s  "
                            f"mean={statistics.mean(lats):.0f}ms  "
                            f"p95={sorted(lats)[int(len(lats) * 0.95)]:.0f}ms  "
                            f"max={max(lats):.0f}ms  "
                            f"errors={errors}/{level}"
                        )
                    else:
                        print(f"ALL FAILED ({errors} errors)")

    asyncio.run(run_all())

    # ── Write CSV ────────────────────────────────────────────────────────────
    csv_path = output_dir / "concurrency.csv"
    if all_results:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(all_results[0]).keys()))
            writer.writeheader()
            for r in all_results:
                writer.writerow(asdict(r))
        print(f"\n✓ CSV → {csv_path}")

    # ── Write report ─────────────────────────────────────────────────────────
    report_path = output_dir / "concurrency_report.md"
    _write_report(all_results, report_path, args, levels)
    print(f"✓ Report → {report_path}")


def _write_report(
    results: list[RequestResult],
    path: Path,
    args,
    levels: list[int],
) -> None:
    from collections import defaultdict

    # Group by concurrency level
    groups: dict[int, list[RequestResult]] = defaultdict(list)
    for r in results:
        groups[r.concurrency].append(r)

    lines = [
        "# OmniVoice Server Concurrency Benchmark\n\n",
        f"URL: `{args.url}` | Max concurrency: `{args.max_concurrency}`"
        f" | Step: `{args.step}` | Runs per level: `{args.runs}`\n\n",
        "## Latency & Throughput by Concurrency Level\n\n",
        "| N (parallel) | Mean (ms) | p50 (ms) | p95 (ms) | Max (ms) | "
        "Wall (s) | Throughput (req/s) | Errors |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]

    for level in levels:
        rs = groups.get(level, [])
        lats = [r.latency_ms for r in rs if r.error is None]
        errors = sum(1 for r in rs if r.error is not None)
        total = len(rs)

        if not lats:
            lines.append(f"| {level} | — | — | — | — | — | — | {errors}/{total} |\n")
            continue

        lats_sorted = sorted(lats)
        mean_lat = statistics.mean(lats)
        p50 = lats_sorted[int(len(lats_sorted) * 0.50)]
        p95 = lats_sorted[int(len(lats_sorted) * 0.95)]
        max_lat = max(lats)
        # Wall time approximated: total requests / throughput
        # Throughput = successful requests / (max latency in batch)
        wall_s = max_lat / 1000  # approximate per-run wall time
        throughput = len(lats) / wall_s if wall_s > 0 else 0

        lines.append(
            f"| {level} | {mean_lat:.0f} | {p50:.0f} | {p95:.0f} | {max_lat:.0f} | "
            f"{wall_s:.2f} | {throughput:.2f} | {errors}/{total} |\n"
        )

    lines.extend(
        [
            "\n## Interpretation\n\n",
            "- **N** = number of requests fired in parallel\n",
            "- **Mean/p50/p95/Max** = per-request latency within the batch\n",
            "- **Throughput** = requests completed per second (successful only)\n",
            "- **Errors** = failed requests (timeouts, 5xx, connection errors)\n",
            "- If Mean ≈ N × single-request latency, the server is serialising "
            "(no true parallelism)\n",
            "- If Mean stays flat as N grows, the server handles parallel load well\n",
            "\n*Generated by omnivoice-server concurrency benchmark*\n",
        ]
    )

    path.write_text("".join(lines))


if __name__ == "__main__":
    main()
