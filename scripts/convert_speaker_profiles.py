#!/usr/bin/env python3
"""Convert vLLM-Omni speaker-profile ``.pt`` files into omnivoice-server profiles.

The OmniVoice fine-tune ships speaker profiles as PyTorch ``.pt`` dicts::

    {
      "format": "omnivoice_speaker_profile_v1",
      "name": "yousef",
      "ref_audio_tokens": LongTensor[8, T],   # 8 codebooks (HiggsAudioV2)
      "ref_text": "...",
      "ref_rms": 0.0528,
      "sample_rate": 24000,
      "num_samples": 204480,
      "created_at": 1775549414,
      "speaker_description": "Warm Emirati male voice with local accent.",
    }

omnivoice-server instead expects one directory per voice::

    <profile_dir>/<profile_id>/ref_audio.wav   # mono PCM_16
    <profile_dir>/<profile_id>/meta.json       # {"name", "ref_text", ...}

This script decodes ``ref_audio_tokens`` with the HiggsAudioV2 tokenizer and
writes both files, so the voices are usable as ``clone:<profile_id>``.

Run it **inside** an image that has the OmniVoice stack (e.g.
``inception-tts`` or ``omnivoice-server:ot-v1-internal``), with the model
directory available:

    podman run --rm \\
      -v "$PWD/models/OT_v1:/model:ro" \\
      -v "$PWD/profiles:/out" \\
      --entrypoint python omnivoice-server:ot-v1-internal \\
      /scripts/convert_speaker_profiles.py \\
        --src /model/speaker_profiles --dst /out --prefix adeo_

Note: ``--prefix`` defaults to ``adeo_`` so converted voices are exposed as
``clone:adeo_<name>``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Reuse the server's own WAV writer so output matches exactly what the API returns.
try:
    from omnivoice_server.utils.audio import tensor_to_wav  # type: ignore
except Exception:  # pragma: no cover - fallback when running standalone
    tensor_to_wav = None

import soundfile as sf
import torch


def _decode_tokens(audio_tokenizer, tokens: torch.Tensor) -> torch.Tensor:
    """Decode ``[8, T]`` codebook tokens to a ``[1, N]`` float waveform."""
    codes = tokens.to(torch.long)
    if codes.dim() == 2:
        codes = codes.unsqueeze(0)  # -> [1, 8, T]
    with torch.no_grad():
        audio = audio_tokenizer.decode(codes)
    if hasattr(audio, "audio_values"):  # ModelOutput
        audio = audio.audio_values
    audio = audio.detach().float().cpu()
    if audio.dim() == 3:  # [B, C, N] -> [C, N]
        audio = audio[0]
    return audio


def _write_wav(path: Path, waveform: torch.Tensor, sample_rate: int) -> int:
    """Write mono PCM_16 WAV. Returns the number of samples written."""
    if waveform.dim() == 2 and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # downmix
    waveform = waveform.squeeze(0) if waveform.dim() == 2 else waveform
    waveform = waveform.reshape(1, -1)

    if tensor_to_wav is not None:
        data = tensor_to_wav(waveform, sample_rate=sample_rate)
        path.write_bytes(data)
    else:
        sf.write(str(path), waveform.squeeze(0).numpy(), sample_rate, subtype="PCM_16")
    return waveform.shape[-1]


def convert_one(pt_path: Path, dst_root: Path, prefix: str, tokenizer, force: bool) -> dict:
    """Convert a single ``.pt`` profile. Returns a report entry."""
    raw = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or "ref_audio_tokens" not in raw:
        return {"source": pt_path.name, "status": "skipped", "reason": "not a profile dict"}

    base_name = str(raw.get("name") or pt_path.stem).lower().replace(" ", "_")
    profile_id = base_name if base_name.startswith(prefix) else f"{prefix}{base_name}"
    out_dir = dst_root / profile_id

    if out_dir.exists() and not force:
        return {"profile_id": profile_id, "status": "exists", "source": pt_path.name}

    tokens = raw["ref_audio_tokens"]
    sample_rate = int(raw.get("sample_rate", 24000))
    waveform = _decode_tokens(tokenizer, tokens)

    out_dir.mkdir(parents=True, exist_ok=True)
    num_samples = _write_wav(out_dir / "ref_audio.wav", waveform, sample_rate)

    rms = float(waveform.pow(2).mean().sqrt().item())
    meta = {
        "name": profile_id,
        "ref_text": raw.get("ref_text"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": pt_path.name,
        "source_format": raw.get("format", "omnivoice_speaker_profile_v1"),
        "source_created_at": raw.get("created_at"),
        "ref_rms": rms,
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "speaker_description": raw.get("speaker_description"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")

    return {
        "profile_id": profile_id,
        "status": "converted",
        "source": pt_path.name,
        "duration_s": round(num_samples / sample_rate, 2),
        "sample_rate": sample_rate,
        "rms": round(rms, 5),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path, help="directory containing *.pt speaker profiles")
    ap.add_argument("--dst", required=True, type=Path, help="profile dir to write into")
    ap.add_argument("--model", default="/home/runner/models/OmniVoice", type=Path,
                    help="OmniVoice model dir (contains audio_tokenizer/)")
    ap.add_argument("--prefix", default="adeo_", help="profile-id prefix (default: adeo_)")
    ap.add_argument("--force", action="store_true", help="overwrite existing profiles")
    ap.add_argument("--report", type=Path, default=None, help="report path (default: <dst>/conversion_report.json)")
    args = ap.parse_args()

    pt_files = sorted(p for p in args.src.glob("*.pt"))
    if not pt_files:
        print(f"error: no .pt files in {args.src}", file=sys.stderr)
        return 1

    tokenizer_path = args.model / "audio_tokenizer"
    if not tokenizer_path.is_dir():
        print(f"error: audio_tokenizer not found at {tokenizer_path}", file=sys.stderr)
        return 1

    from omnivoice.models.omnivoice import HiggsAudioV2TokenizerModel

    print(f"loading audio tokenizer from {tokenizer_path} ...")
    tokenizer = HiggsAudioV2TokenizerModel.from_pretrained(str(tokenizer_path), device_map="cpu")

    args.dst.mkdir(parents=True, exist_ok=True)
    entries = []
    for pt in pt_files:
        try:
            entry = convert_one(pt, args.dst, args.prefix, tokenizer, args.force)
        except Exception as exc:  # noqa: BLE001 - report and continue
            entry = {"source": pt.name, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
        entries.append(entry)
        status = entry.get("status")
        pid = entry.get("profile_id", pt.name)
        extra = f" ({entry['duration_s']}s)" if status == "converted" else ""
        print(f"  {pid:22s} {status}{extra}")

    report_path = args.report or (args.dst / "conversion_report.json")
    report_path.write_text(json.dumps({"prefix": args.prefix, "converted": entries}, indent=2, ensure_ascii=False) + "\n")
    ok = sum(1 for e in entries if e.get("status") == "converted")
    print(f"\n{ok}/{len(entries)} converted -> {args.dst}  (report: {report_path})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
