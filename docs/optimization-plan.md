# omnivoice-server Optimization Plan

## Current Performance (Hybrid+Sage, RTX 5090, num_step=32)

```
Pure GPU (CUDA events, 32 replays):  137 ms   ← kernel execution only
_generate_iterative (wall clock):    146 ms   ← + 9ms Python loop overhead
_decode_and_post_process:             55 ms   ← GPU decode + .cpu() + numpy
_preprocess_all + prepare_inputs:      3 ms   ← tokenizer + tensor prep
─────────────────────────────────────────────
model.generate() total (warm):       158 ms   ← matches triton runner
Server total (warm, cleanup removed): ~160 ms ← + audio encode + HTTP overhead

Cold start (first shape):            339 ms   ← CUDA Graph capture (3s one-time)
Concurrent (max_concurrent=1):       serialized — queue waits add 160ms/req
```

## GPU Path: IDENTICAL to triton runner — DO NOT TOUCH

Verified: server and triton runner use the same `_CUDAGraphForward` class,
same patch ordering (triton → sage → graph), same `generate()` path.
`torch.inference_mode()` already on `generate()`. CUDA Graph captures
`model.forward()` per input shape `(2, 8, seq_len)`. num_step=32 means
32 graph replays per request, NOT 32 graph captures.

---

## Latency Breakdown (warm, steady-state)

### Single request — where the ~160ms goes

| Stage | Time | What happens | CPU? | GPU? |
|-------|------|-------------|------|------|
| `_estimate_target_tokens` | 0.1ms | Rule-based duration estimate (no NN) | ✅ | ❌ |
| `_preprocess_all` | 0.3ms | List normalization, speed/duration logic | ✅ | ❌ |
| `_prepare_inference_inputs` | 1.2ms | Tokenizer encode, tensor cat, `.to(cuda)` | ✅ | ❌ |
| `_generate_iterative` | 105ms | 32× forward() = 32× graph.replay() | ❌ | ✅ (137ms GPU) |
| ↳ Python loop overhead | ~9ms | topk, masked_fill_, copy_ between steps | ✅ | ❌ |
| `_decode_and_post_process` | 55ms | audio_tokenizer.decode + post-process | ✅ | ✅ |
| ↳ audio_tokenizer.decode | ~5ms | GPU decode | ❌ | ✅ |
| ↳ .cpu().numpy() sync | ~0.1ms | GPU→CPU transfer (blocks until decode done) | ✅ | ❌ |
| ↳ remove_silence + fade | ~1ms | numpy operations | ✅ | ❌ |
| ↳ **GPU drain wait** | ~49ms | Implicit sync waiting for prior GPU work | ❌ | ✅ |
| audio_encode (wav) | 0.5ms | soundfile write | ✅ | ❌ |
| HTTP / FastAPI overhead | ~1ms | Request parsing, response | ✅ | ❌ |

### Key finding: the 55ms "decode" is mostly GPU drain

The `_decode_and_post_process` wall clock is 55ms but the actual
`audio_tokenizer.decode()` GPU kernel only takes ~5ms. The remaining ~50ms
is the `.cpu().numpy()` call implicitly synchronizing — it blocks until ALL
previously queued GPU work (the 32 diffusion steps + the decode kernel)
finishes. This is not wasted time — it's the GPU finishing real work.

### 3 concurrent requests — serialization analysis

With `max_concurrent=1`, requests serialize through the semaphore:

| Request | Wall time | model.generate | Queue wait |
|---------|-----------|----------------|------------|
| req 1 | 435ms | 429ms (cold graph capture) | 0ms |
| req 2 | 595ms | 158ms (warm replay) | 428ms |
| req 3 | 751ms | 164ms (warm replay) | 587ms |

The queue waits are purely the serialization cost of max_concurrent=1.
Each subsequent request adds ~160ms of wait. This is **correct behavior** —
see item #4 for why increasing max_concurrent does not help.

---

## Action Items

### 1. ✅ Remove per-request `_cleanup_memory()` — 212ms saved

**Status:** DONE

`gc.collect()` + `torch.cuda.empty_cache()` was 212ms of pure waste per request.
PyTorch's caching allocator already reuses freed blocks in nanoseconds.

### 2. ✅ CPU-side overhead investigation — COMPLETE

**Status:** RESOLVED — no actionable CPU overhead found

The original "266ms CPU overhead" hypothesis was wrong. After instrumenting
all 5 internal stages, the breakdown is:
- GPU work: 137ms (diffusion) + 5ms (decode) = 142ms
- Python loop: 9ms (topk/mask/copy between steps — inherent to algorithm)
- CPU preprocessing: 3ms (tokenizer + tensor prep — already fast)
- GPU sync drain: ~50ms (implicit sync in .cpu() — real GPU work finishing)

There is no CPU bottleneck to parallelize. The pipeline is GPU-bound.
The ~160ms total matches the pure triton runner benchmark (152ms).

### 3. 🟡 Voice clone prompt caching (LRU)

Clone-mode requests re-run `create_voice_clone_prompt()` (ASR + audio encoding)
every time, even for the same reference audio. An LRU cache keyed by
`(ref_audio_path, ref_text)` would save 100-200ms on repeat clones.

**Effort:** ~30 lines. **Only affects clone mode.**

### 4. ❌ Increase `max_concurrent` — REJECTED: GPU contention

**Status:** REJECTED — diffusion model is GPU-compute-bound, not I/O-bound

Initially seemed promising: the 32-step loop has ~9ms Python overhead between
`graph.replay()` calls, suggesting GPU idle time a second request could fill.

**Empirical testing proves otherwise:**

| Metric | Value |
|--------|-------|
| Per-step GPU time | 4.29ms |
| Gap between consecutive steps | 0.00ms |
| GPU utilization | **100%** |

The GPU is **fully saturated** during the diffusion loop. The 9ms of "Python
loop overhead" overlaps with GPU execution — `graph.replay()` returns
immediately (async launch), Python does `topk`/`masked_fill_`/`copy_`, then
the next `replay()` queues immediately. There is no GPU idle gap to fill.

**2-thread contention test (separate static buffers, same GPU):**

```
1 thread, 32 replays:     137.1ms
2 threads, 64 replays:    272.6ms  (per-request: 136.3ms)
Throughput gain:          1.01x (1.0 = no gain, 2.0 = perfect)
```

Two concurrent `graph.replay()` calls fight for the same SMs. Each request
takes the same ~137ms — **zero throughput gain**. The GPU is the bottleneck,
not the CPU or Python loop. This is inherent to diffusion models: each of
the 32 steps is a full transformer forward pass that saturates the GPU.

**Conclusion:** `max_concurrent=1` is optimal for latency. Increasing it
would only help if the model were I/O-bound or had significant CPU-side
bottlenecks between GPU calls — neither is the case here. To increase
throughput, you'd need a second physical GPU.

### 5. 🟡 Pre-allocate CUDA Graph for common shapes

Currently the first request for each text length pays a 3-second graph
capture cost. Pre-warming common shapes at startup would eliminate cold-start
latency for production traffic.

**Effort:** ~15 lines in ModelService (generate a few representative texts at
startup after model load). **Affects cold-start, not steady-state.**

---

## Rejected (do not pursue)

- ~~TF32~~ — model runs in fp16, TF32 is for fp32 matmuls. No-op.
- ~~Reduce num_step 32→16~~ — degrades output quality. 32 is upstream default.
- ~~Add torch.inference_mode()~~ — already on `OmniVoice.generate()`.
- ~~Modify GPU path~~ — identical to triton runner, already optimal.
- ~~Parallelize CPU preprocessing~~ — only 3ms total, not worth the complexity.
- ~~Parallelize decode~~ — the 55ms is GPU drain, not CPU work. Can't parallelize
  a GPU synchronization barrier.
- ~~Overlap decode of req N with generate of req N+1~~ — would require
  restructuring OmniVoice.generate() which is upstream code. Not feasible
  without forking the library.
- ~~Increase max_concurrent~~ — diffusion model saturates GPU at 100% utilization.
  2 concurrent requests = zero throughput gain (1.01x measured). GPU contention
  means both requests take the same time as sequential. Need 2nd physical GPU.
