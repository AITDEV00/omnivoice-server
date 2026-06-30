"""
Loads and holds the OmniVoice model singleton.
Model is loaded once at startup; never reloaded during runtime.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import time
import types
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

import psutil
import torch

if TYPE_CHECKING:
    from omnivoice import OmniVoice

from ..config import Settings

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._model = None
        self._loaded = False

    async def load(self) -> None:
        """Load model in a thread (blocking op, must not block event loop)."""
        loop = asyncio.get_running_loop()
        with ThreadPoolExecutor(max_workers=1) as ex:
            await loop.run_in_executor(ex, self._load_sync)

    def _load_sync(self) -> None:
        from omnivoice import OmniVoice

        ram_before = _get_ram_mb()
        t0 = time.monotonic()

        logger.info(f"Loading model '{self.cfg.model_id}' on {self.cfg.device}...")

        for dtype in self._dtype_candidates():
            try:
                from_pretrained_kwargs = {
                    "device_map": self.cfg.torch_device_map,
                    "dtype": dtype,
                }
                if self.cfg.model_cache_dir is not None:
                    from_pretrained_kwargs["cache_dir"] = str(self.cfg.model_cache_dir)
                model = OmniVoice.from_pretrained(
                    self.cfg.model_id,
                    load_asr=True,
                    asr_model_name=self.cfg.asr_model_id,
                    **from_pretrained_kwargs,
                )

                # Apply omnivoice-triton optimization (after load, before NaN test)
                if self.cfg.enable_triton and self.cfg.device == "cuda":
                    self._apply_triton_optimization(model)

                test = model.generate(text="test", num_step=4)
                if self._has_nan(test):
                    logger.warning(f"dtype={dtype} produced NaN, trying next...")
                    del model
                    gc.collect()
                    continue
                self._model = model
                break
            except Exception as e:
                logger.warning(f"Failed to load with dtype={dtype}: {e}")
                continue

        if self._model is None:
            raise RuntimeError(
                f"Failed to load OmniVoice on device={self.cfg.device}. "
                "Try --device cpu or check GPU/MPS availability."
            )

        elapsed = time.monotonic() - t0
        ram_after = _get_ram_mb()
        logger.info(
            f"Model loaded in {elapsed:.1f}s. "
            f"RAM: {ram_before:.0f}MB -> {ram_after:.0f}MB "
            f"(+{ram_after - ram_before:.0f}MB)"
        )

        # Instrument internal generate() stages for pipeline timing analysis.
        # This wraps the model's internal methods (not the GPU kernels themselves)
        # to measure where CPU-side time is spent inside model.generate().
        self._instrument_generate_stages(self._model)

        self._loaded = True

    def _instrument_generate_stages(self, model) -> None:
        """Wrap OmniVoice.generate() internal methods with timing logs.

        This does NOT modify GPU execution — it only adds perf_counter calls
        around existing CPU-side methods to identify where the 266ms overhead
        between pure GPU (153ms) and server model.generate (419ms) is spent.
        """
        import functools
        import time as _time

        timed_methods = [
            "_preprocess_all",
            "_generate_iterative",
            "_decode_and_post_process",
            "_prepare_inference_inputs",
            "_estimate_target_tokens",
        ]

        for method_name in timed_methods:
            original = getattr(model, method_name, None)
            if original is None:
                continue

            def _make_wrapper(name, orig):
                @functools.wraps(orig)
                def _wrapper(*args, **kwargs):
                    t0 = _time.monotonic()
                    result = orig(*args, **kwargs)
                    elapsed_ms = (_time.monotonic() - t0) * 1000
                    logger.info(
                        f"[TIMING] OmniVoice.{name}={elapsed_ms:.1f}ms"
                    )
                    return result
                return _wrapper

            setattr(model, method_name, _make_wrapper(method_name, original))
            logger.debug(f"Instrumented OmniVoice.{method_name} with timing")

    def _apply_triton_optimization(self, model) -> None:
        """Apply omnivoice-triton kernel fusion + CUDA Graph to the loaded model.

        Patches are applied transparently — model.generate() keeps the same
        signature and behaviour, but the internal forward pass uses fused
        Triton kernels and/or CUDA Graph replay for up to 4.66x speedup.

        Modes:
            - triton:  RMSNorm + SwiGLU + FusedNormResidual kernel fusion
            - faster:  CUDA Graph capture/replay (eliminates kernel launch overhead)
            - hybrid:  Both combined (fastest, ~4.66x)
            - +sage:   SageAttention quantized attention (optional, on by default)
        """
        from omnivoice_triton.models.patching import (
            apply_sage_attention,
            apply_triton_kernels,
            find_patchable_model,
)

        mode = self.cfg.triton_mode
        logger.info(
            f"Applying omnivoice-triton optimization: mode={mode}, "
            f"sage={self.cfg.triton_enable_sage}"
        )

        patchable = find_patchable_model(model)

        # Apply Triton kernel fusion (for triton and hybrid modes)
        if mode in ("triton", "hybrid"):
            count = apply_triton_kernels(patchable, patch_range=(0, 24))
            logger.info(f"Triton kernels applied: {count} patches")

        # Apply SageAttention
        if self.cfg.triton_enable_sage:
            sage_count = apply_sage_attention(patchable, patch_range=(0, 24))
            logger.info(f"SageAttention patched: {sage_count} attention modules")

        # Install CUDA Graph wrapper (for faster and hybrid modes)
        if mode in ("faster", "hybrid"):
            from omnivoice_triton.models.faster_runner import _CUDAGraphForward

            graph_fwd = _CUDAGraphForward(model)
            model.forward = graph_fwd
            logger.info(
                f"CUDA Graph wrapper installed (mode={mode}). "
                "First inference per shape will capture the graph."
            )

    def _dtype_candidates(self) -> list:
        if self.cfg.device in ("cuda", "mps"):
            return [torch.float16, torch.bfloat16, torch.float32]
        return [torch.float32]

    @staticmethod
    def _has_nan(tensors: torch.Tensor | np.ndarray | list | None) -> bool:
        np: types.ModuleType | None
        try:
            import numpy as np
        except Exception:
            np = None

        def contains_nan(x) -> bool:
            if x is None:
                return False
            # Check numpy arrays first to avoid calling torch.isnan on ndarray outputs (issue #17).
            if np is not None and isinstance(x, np.ndarray):
                return bool(np.isnan(x).any())
            if torch.is_tensor(x):
                return bool(torch.isnan(x).any().item())
            if isinstance(x, (list, tuple)):
                return any(contains_nan(i) for i in x)
            return False

        return contains_nan(tensors)

    @property
    def model(self) -> OmniVoice:
        if not self._loaded:
            raise RuntimeError("Model not loaded yet")
        return self._model

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def _get_ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024
