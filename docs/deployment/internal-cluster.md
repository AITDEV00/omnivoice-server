# Internal-Cluster Image (Fine-Tuned OT_v1)

Image: **`omnivoice-server:ot-v1-internal`** — the internal-cluster-compliant build that
serves the **fine-tuned OT_v1** checkpoint with the optimized server + triton/sage
acceleration and the converted clone voices.

Built **from scratch** on the NVIDIA CUDA base image
(`docker.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`) — the *same* base the
deployed `omnivoice-server:latest` was built on. The whole runtime is assembled in
this Dockerfile: Python 3.12 + venv, PyTorch cu128, `omnivoice-triton`, the server,
the fine-tuned weights, the ASR model, and the clone voices.

> **Why not layer on the deployed image?** An earlier version used
> `FROM …/omnivoice-server:latest` + `rm -rf …/OmniVoice` + `COPY models/OT_v1`.
> That leaves the deployed image's baked-in **2.4 GB base checkpoint** as a dead
> layer in the final image (it is deleted but still occupies space), and produced a
> **24.7 GB** image. Building from scratch copies the weights exactly once and
> yields **18.2 GB** — the same size as the deployed image, because only the
> checkpoint changed.

## Layer / push optimization

The Dockerfile is ordered to match the deployed image's layer stack so that a
rebuild only re-pushes the layers that actually changed.

### The bug that made pushes huge

The previous version ran, *after* both model `COPY`s:

```dockerfile
RUN chown -R runner:runner /home/runner/models /home/runner/profiles-baked
```

`chown -R` rewrites every file, so the entire model + ASR tree was duplicated
into a **new 4.9 GB layer** — the 3.27 GB model and 1.62 GB ASR were each stored
twice. Any rebuild re-pushed all 4.9 GB.

**Fix:** use `COPY --chown=runner:runner`, which sets ownership *while writing*
the layer. One layer, correct owner, no duplicate.

| | Before | After |
|---|---|---|
| Image size | 23.1 GB | **18.2 GB** |
| `chown -R` layers | 4.9 GB duplicate | none (only a 20 kB `mkdir`) |
| Model/ASR stored | twice | once |

### Ownership: everything under `/home/runner` is `runner:runner`

The image drops to `USER runner` **before** creating the venv, so the venv,
torch, triton and every pip-installed file are owned by the runtime user —
matching the deployed image, which does the same. Model files and profiles are
`COPY --chown=runner:runner`.

Result: `find /home/runner -not -user runner` returns **0** (the deployed image
has 66 — its whisper model dir is `root:root`).

| Path | Deployed `:latest` | This image |
|---|---|---|
| `/home/runner/venv/**` | `runner:runner` | `runner:runner` ✅ |
| `/home/runner/models/OmniVoice/**` | `root:root` 644 | `runner:runner` 644 |
| `/home/runner/models/whisper-*/**` | `root:root` 644 | `runner:runner` 644 |
| non-`runner` files under `/home/runner` | 66 | **0** |

> **Does `--chown` hurt push dedupe?** No. I tested it: building a layer with the
deployed image's exact plain-`COPY` recipe produced a *different* digest
(`d1e1598ccc…` vs `bfc68c02d88e…`), because tar metadata (mtime/uid/xattr)
differs. Those blobs were never going to dedupe against the deployed image
anyway, so `--chown` costs nothing and gives a cleaner, more correct image.

One consequence of `USER runner`: any `COPY` that a later build step deletes
must also use `--chown=runner:runner`, otherwise the non-root user cannot
remove it. That is why `vendor/omnivoice-triton`, `pyproject.toml` and
`omnivoice_server` are copied with `--chown`.

### Layer order

Expensive layers are pinned at the bottom, volatile ones last:

```
  ... apt, venv, torch (7.0 GB), triton (867 MB), deps, server source
  ────────────────────────────────────────────────────────────────
  COPY --chown model.safetensors   (2.45 GB)   <- only weights
  COPY --chown audio_tokenizer     (806 MB)
  COPY --chown config/tokenizer/…  (tiny)
  COPY --chown whisper-…-turbo     (1.62 GB)
  COPY --chown profiles            (5 MB)      <- only voices
  COPY --chmod entrypoint.sh       (9 kB)
```

### What a rebuild re-pushes

| Change | Re-pushed |
|---|---|
| New clone voice / profile edit | **~5 MB** |
| New `model.safetensors` | ~2.45 GB |
| New `audio_tokenizer` | ~806 MB |
| Server code edit | ~400 KB |
| Entrypoint tweak | ~9 KB |
| New triton source | from the triton layer down (~8 GB) |

Two details make this work:

1. **Model files are `COPY`ed individually**, not as one directory. Editing
   `config.json` invalidates only its 10 kB layer, not the 2.45 GB
   `model.safetensors` layer.
2. **The base layers are byte-identical to the deployed image** (same
   `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`), so those ~14 GB of blobs
   already exist in Harbor and are skipped entirely on push.

Verified: editing a profile changed only the `profiles` layer digest — the
`model.safetensors`, `audio_tokenizer` and `whisper` digests were unchanged.

## What this image contains

| Component | Detail |
|---|---|
| **Base** | `nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` (Ubuntu 24.04 noble, CUDA 12.8, cuDNN 9.7.0) |
| **Python** | 3.12 + venv at `/home/runner/venv` |
| **PyTorch** | `torch 2.11.0+cu128`, `torchaudio 2.11.0+cu128` (pinned `<2.12` — 2.12+ ships cu130) |
| **Triton** | 3.6.0 (pulled in by PyTorch) |
| **omnivoice-triton** | vendored source from `vendor/omnivoice-triton` (kernel fusion + SageAttention) |
| **sageattention** | 1.0.6 (pure-Python, `py3-none-any`; Triton JITs kernels at runtime — **no nvcc needed**) |
| **Fine-tuned weights** | `models/OT_v1` → `/home/runner/models/OmniVoice` (copied once) |
| **ASR** | `models/whisper-large-v3-turbo` → `/home/runner/models/whisper-large-v3-turbo` |
| **Clone voices** | `profiles/` → `/home/runner/profiles-baked` (16 voices, `adeo_` prefix, WAV+JSON only, seeded to `/tmp` at startup) |
| **Entrypoint** | Forces all writes under `/tmp`, seeds profiles, chains the NVIDIA entrypoint |

## GPU architecture coverage

The cu128 PyTorch build carries `sm_70 / sm_75 / sm_80 / sm_86 / sm_90 / sm_100 / sm_120`
kernels, so **one image serves both target GPUs**:

| GPU | Arch | How it runs |
|---|---|---|
| **H200** | `sm_90` (Hopper) | Native `sm_90` kernels |
| **RTX 4090** | `sm_89` (Ada) | `sm_86` binary compatibility (no `sm_89` in the list, no warning) |

SageAttention selects its kernel at runtime — on the RTX 4090 it logs
`SageAttention v1.0+ detected: SM89 kernel`.

## Verification — the model is the fine-tune

| Model | `model.safetensors` sha256 |
|---|---|
| `k2-fsa/OmniVoice` (base, HF) | `730839316de585f4c8298ec0e1712efc10fb19c6fa4e36eb741cb8d51ebcf6aa` |
| Deployed `:latest` | `73083931…` ← **base** |
| **This image** | `4e8b9708fc8525d3113b8f27d71808a00a786773c8a2e12aa6d199cca6f7afdd` ✅ **fine-tuned** |

## Build

There are **two** Dockerfiles. Pick based on how much you need to push.

### Option A — `Dockerfile.internal-patch` (recommended: ~2.5 GB push)

Layers on top of the deployed image, replacing only what actually differs. Because
most of the model tree is byte-identical to the base, this uploads **3 layers
(~2.5 GB)** instead of 21 (~18 GB).

```bash
podman build -t omnivoice-server:ot-v1-patch -f Dockerfile.internal-patch .
```

What differs from `:latest` — verified by sha256:

| File | vs base | Size |
|---|---|---|
| `model.safetensors` | **DIFFERS** (the fine-tune) | 2.45 GB |
| `audio_tokenizer/` | same | 806 MB |
| `tokenizer.json` | same | 11 MB |
| `config.json`, `tokenizer_config.json`, `chat_template.jinja` | same | — |
| `whisper-large-v3-turbo/` | same | 1.62 GB |
| `profiles/` | new | 5 MB |
| `entrypoint.sh` | new | 9 KB |

| | Patch | Full (from scratch) |
|---|---|---|
| Layers to upload | **3** | 21 |
| Bytes to upload | **~2.5 GB** | ~18 GB |
| Local image size | 20.6 GB | 18.2 GB |
| Self-contained | no (needs `:latest` in the registry) | yes |

> The patch image is *larger locally* (20.6 GB) because the base's 3.27 GB
> monolithic model layer remains as dead weight under the override. That is a
> local-disk cost only — the push, and any node pull that already has `:latest`
> cached, transfer just the 3 new layers.

### Option B — `Dockerfile.internal` (self-contained: ~18 GB push)

Builds the whole image from the NVIDIA CUDA base. Uploads everything the first
time, but is independent of the registry's `:latest`.

```bash
cd /home/jyao/ADEO/omnivoice-server

# Prerequisites (build context):
#   models/OT_v1                      fine-tuned checkpoint (3.1 GB)
#   models/whisper-large-v3-turbo     ASR model (1.6 GB)
#   profiles/                         converted voices (scripts/convert_speaker_profiles.py)
#   vendor/omnivoice-triton/          vendored triton source (132 KB)
#   scripts/internal/entrypoint.sh

podman build -t omnivoice-server:ot-v1-internal -f Dockerfile.internal .
```

> The only external dependency is `docker.io/nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04`
> — nothing is pulled from Harbor. For an offline build, `podman pull` it first.
>
> The Dockerfile installs the server's dependencies against a **stub package** first,
> so that slow layer stays cached when only `omnivoice_server/` source changes. Pip
> downloads are cached via `--mount=type=cache,target=/tmp/runner/.cache/pip`.

## Run (full internal-cluster constraints)

```bash
# WSL2: the /etc/cdi spec may be stale after a driver update; bind-mount the
# WSL driver libs instead (see the INCEPTION Makefile). On a normal host with a
# current CDI spec, just use --device nvidia.com/gpu=all.
WSL_DRV_DIR=$(ls -d /usr/lib/wsl/drivers/nv_dispi.inf_amd64_* | grep -v '.ini' | head -1)

podman run -d --name ov-internal \
  --read-only --user 10000:10000 --tmpfs /tmp:rw,size=8g \
  --device /dev/dxg \
  -v /usr/lib/wsl/lib:/usr/lib/wsl/lib:ro \
  -v "$WSL_DRV_DIR:$WSL_DRV_DIR:ro" \
  --env LD_LIBRARY_PATH=/usr/lib/wsl/lib:$WSL_DRV_DIR \
  -p 8080:8080 \
  omnivoice-server:ot-v1-internal
```

## REQ compliance (verified locally on RTX 4090)

| Req | Check | Result |
|---|---|---|
| REQ-1 | Port 8080 | ✅ `EXPOSE 8080`, server on 8080 |
| REQ-2 | `GET /health-check` | ✅ HTTP 200 `{"status":"healthy","ready":true}`, ready in ~20 s |
| REQ-3 | UID 10000 + read-only fs | ✅ `--read-only --user 10000`; `touch /home/runner/x` → *Read-only file system* |
| REQ-4 | Writes to `/tmp` only | ✅ `HOME`/`TMPDIR`/caches all under `/tmp/runner`; `/tmp` writable, no write errors |
| REQ-5 | Single container | ✅ PID 1 = `/home/runner/venv/bin/python3 …/omnivoice-server`, owned by `runner` |
| REQ-6 | OpenAI `/v1` | ✅ `/v1/audio/speech` 200, `/v1/models` 200, `/v1/voices` 200 |
| REQ-7 | Weights baked in | ✅ no host mounts; model in image (hash verified below) |

### Dependency parity with the deployed image

| Package | Deployed `:latest` | This image |
|---|---|---|
| torch | 2.11.0+cu128 | 2.11.0+cu128 |
| triton | 3.6.0 | 3.6.0 |
| sageattention | 1.0.6 | 1.0.6 |
| omnivoice-server | 0.2.4 | 0.2.4 |

## Triton / SageAttention active

```
[omnivoice_server.services.model] Applying omnivoice-triton optimization: mode=hybrid, sage=True
[omnivoice_triton.models.patching] [Partial] Layers [0, 24): 96 RMSNorm, 24 MLP, 24 FusedNormResidual
[omnivoice_triton.models.patching] SageAttention v1.0+ detected: SM89 kernel
[omnivoice_triton.models.patching] SageAttention patched: 24 attention modules
[omnivoice_server.services.model] CUDA Graph wrapper installed (mode=hybrid)
```

## Clone voices (still work)

16 voices are seeded into the writable profile dir at startup. All of them use the
**`adeo_` prefix**, so the clone IDs are `clone:adeo_<name>`:

```
clone:adeo_ahmed  clone:adeo_hamid  clone:adeo_hana    clone:adeo_harry
clone:adeo_heba   clone:adeo_hessa  clone:adeo_jabir   clone:adeo_jane
clone:adeo_khalid clone:adeo_majid  clone:adeo_samia   clone:adeo_sarah
clone:adeo_yousef clone:adeo_zahra  clone:adeo_zoe     clone:adeo_zoe_ar
```

### WAV-only policy (no PyTorch `.pt` in the image)

Profiles are stored in the **open format only**:

```
profiles/<profile_id>/ref_audio.wav   # mono PCM_16, 24 kHz
profiles/<profile_id>/meta.json
```

The vLLM-Omni `.pt` originals in `models/OT_v1/speaker_profiles/` are a
**build-time input**, converted ahead of the build by
`scripts/convert_speaker_profiles.py`. They are never baked into the image —
this is enforced two ways:

1. `Dockerfile.internal` copies the model **file-by-file**, so
   `models/OT_v1/speaker_profiles/*.pt` is simply not selected. (`COPY models/OT_v1`
   would have dragged it in.)
2. `.dockerignore` excludes `**/*.pt`, `**/*.pth`, `**/*.ckpt`, `**/*.bin` as a
   safety net.

Verified: `find / -xdev -name '*.pt' -not -path '*/site-packages/*'` returns **0**,
and `profiles/` contains **0** files that are not `.wav` or `.json`.

Why: `.pt` is an opaque pickle with arbitrary-code-execution risk on load, and the
runtime never needs it — only the decoded WAV does.

### Convert new voices

```bash
podman run --rm \
  -v "$PWD/models/OT_v1:/model:ro" \
  -v "$PWD/profiles:/out" \
  --entrypoint python omnivoice-server:ot-v1-internal \
  /scripts/convert_speaker_profiles.py \
    --src /model/speaker_profiles --dst /out --prefix adeo_
```

`--prefix` defaults to `adeo_`; pass `--force` to overwrite existing profiles.

### Tested

All HTTP 200, valid 24 kHz mono PCM:

| Mode | Request | Result |
|---|---|---|
| `auto` | `{"voice":"auto"}` | ✅ 200, 2.56 s @ 24 kHz |
| `clone:adeo_yousef` | `{"voice":"clone:adeo_yousef"}` | ✅ 200, 2.00 s @ 24 kHz |
| `clone:adeo_ahmed` | `{"voice":"clone:adeo_ahmed"}` | ✅ 200, 1.47 s @ 24 kHz |
| `clone:adeo_zoe_ar` | `{"voice":"clone:adeo_zoe_ar"}` | ✅ 200, 1.21 s @ 24 kHz |
| `design:` | `{"voice":"design:female, low pitch"}` | ✅ 200, 1.72 s @ 24 kHz |
| upload-clone | multipart `text` + `ref_audio` | ✅ 200 (Whisper auto-transcribes) |

```bash
# auto (fine-tuned weights)
curl -s -X POST localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"omnivoice","input":"Hello.","voice":"auto"}' -o auto.wav

# clone (fine-tuned voice)
curl -s -X POST localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"omnivoice","input":"Hello.","voice":"clone:adeo_yousef"}' -o yousef.wav

# design (attributes must come from the supported vocabulary — see /v1/voices)
curl -s -X POST localhost:8080/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"omnivoice","input":"Hello.","voice":"design:female, low pitch"}' -o design.wav

# one-shot clone from an uploaded reference (multipart; field is `text`, not `input`)
curl -s -X POST localhost:8080/v1/audio/speech/clone \
  -F "ref_audio=@ref.wav" -F "text=Cloned speech." -o clone.wav
```

### Design-voice gotchas

- The attribute vocabulary is **fixed**; `design:` values are validated and
  canonicalized. `narration`, `cheerful`, `sad`, `angry`, `happy`, etc. are
  **explicitly rejected** → HTTP 422. Check `GET /v1/voices` (`design:<attributes>`)
  for the supported set.
- There is **no** `/v1/audio/transcriptions` route. Whisper is used *internally*:
  when `/v1/audio/speech/clone` is called without `ref_text`, the server transcribes
  the uploaded `ref_audio` first (visible in the log as a `transformers` Whisper
  language-detection message).

## Tag & push to Harbor

```bash
REGISTRY=registry.adeoaiengine.ecouncil.ae
REPO=openinnovationai/platform/mlops/mlops-serving

podman login --tls-verify=false -u custom-images -p 'CustomImages123' $REGISTRY
podman tag omnivoice-server:ot-v1-internal $REGISTRY/$REPO/omnivoice-server:ot-v1-internal
podman push --tls-verify=false $REGISTRY/$REPO/omnivoice-server:ot-v1-internal
```

## Notes

- **Image size 18.2 GB** — identical to the deployed image (only the checkpoint
  changed). See *Layer / push optimization* above.
- Speaker profiles in `OT_v1/speaker_profiles/*.pt` are vLLM-Omni format; they are
  converted to the server's `ref_audio.wav` + `meta.json` format by
  `scripts/convert_speaker_profiles.py` (run inside the `inception-tts` image).
  The `.pt` files are **never baked into the image** — see *WAV-only policy*.
- `models/`, `profiles/`, and `vendor/` are git-ignored (see `.git/info/exclude`).
- `.dockerignore` excludes `.git`, `docs`, `tests`, `benchmarks`, `samples`,
  `voice_samples`, `examples` and caches — but **must not** exclude `models/`,
  `profiles/`, `vendor/` or `scripts/`, since those are `COPY` sources.
- `HEALTHCHECK` is ignored by podman under the OCI image format (harmless warning).
  Use a Kubernetes `livenessProbe`/`readinessProbe` on `/health-check` instead.
- `"Profile audio NOT FOUND: 'auto' / 'design:…'"` lines in the log are expected
  trace noise — those are not clone profiles.
- When editing the Dockerfile, **keep the volatile layers last** and **never add
  a `RUN chown -R` over a `COPY`ed tree** — both would undo the push savings.
