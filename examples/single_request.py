#!/usr/bin/env python3
"""
Single-request latency probe for omnivoice-server.

Sends one TTS request and prints the server-reported timing breakdown
from the response headers. Check `docker logs omnivoice-server` for the
corresponding [TIMING] log lines.

Usage:
    python single_request.py              # default: http://localhost:8080
    python single_request.py --url http://host:port
    python single_request.py --format wav
    python single_request.py --text "Custom text here"
"""

import argparse
import time
import sys

import httpx

DEFAULT_URL = "http://localhost:8080"
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."
DEFAULT_FORMAT = "wav"


def main() -> int:
    parser = argparse.ArgumentParser(description="Single TTS request — latency probe")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"Server URL (default: {DEFAULT_URL})")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--format", default=DEFAULT_FORMAT, help="Audio format: wav, pcm, mp3, etc.")
    parser.add_argument("--voice", default=None, help="Voice preset (e.g. alloy, echo)")
    parser.add_argument("--output", default=None, help="Save audio to this file")
    args = parser.parse_args()

    endpoint = f"{args.url}/v1/audio/speech"

    payload: dict = {
        "model": "omnivoice",
        "input": args.text,
        "response_format": args.format,
    }
    if args.voice:
        payload["voice"] = args.voice

    print(f"POST {endpoint}")
    print(f"  text:    {args.text!r}")
    print(f"  format:  {args.format}")
    print(f"  voice:   {args.voice or '(default)'}")
    print()

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=60.0) as client:
            resp = client.post(endpoint, json=payload)
    except httpx.ConnectError:
        print(f"ERROR: Cannot connect to {args.url}", file=sys.stderr)
        return 1
    except httpx.TimeoutException:
        print(f"ERROR: Request timed out", file=sys.stderr)
        return 1

    client_wall_ms = (time.perf_counter() - t0) * 1000

    if resp.status_code != 200:
        print(f"ERROR: HTTP {resp.status_code}", file=sys.stderr)
        print(resp.text, file=sys.stderr)
        return 1

    # Extract server-reported timing from headers
    audio_dur = resp.headers.get("X-Audio-Duration-S", "?")
    synth_lat = resp.headers.get("X-Synthesis-Latency-S", "?")

    print("─── Response Headers ───")
    print(f"  X-Audio-Duration-S:      {audio_dur}s")
    print(f"  X-Synthesis-Latency-S:   {synth_lat}s")
    print(f"  Content-Type:            {resp.headers.get('Content-Type')}")
    print(f"  Content-Length:          {len(resp.content)} bytes")
    print()
    print("─── Client-Side Wall Time ───")
    print(f"  Total (connect→last byte): {client_wall_ms:.1f}ms")
    print()
    print("─── Pipeline Timing Breakdown (from docker logs) ───")
    print("  Run: docker logs omnivoice-server --tail 10 2>&1 | grep TIMING")
    print()

    if args.output:
        from pathlib import Path
        Path(args.output).write_bytes(resp.content)
        print(f"✓ Audio saved to {args.output}")
    else:
        print(f"✓ Request succeeded (discarded {len(resp.content)} bytes of audio)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
