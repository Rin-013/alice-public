"""
TurboQuant KV cache compression — pipeline-wide.

Two cache classes:
  - TurboQuantHybridCache: For hybrid architecture (linear + full attention).
    Only compresses the full-attention layers; linear layers pass through.
  - TurboQuantDynamicCache: Generic DynamicCache for standard transformers.
    Compresses all layers. Used by TTS, STT, and other models.

Both use TurboQuantV3 (MSE-optimal, asymmetric K/V, residual windowing).

Usage:
    from alice.core.optimization.turboquant_cache import create_cache
    cache = create_cache(model.config)  # auto-detects hybrid vs generic
    outputs = model.generate(..., past_key_values=cache, use_cache=True)

Env toggle: ALICE_TURBOQUANT=0 to disable (default on)
"""

import os
import sys
import torch
from typing import Any

# vendor/tq is a junction to vendor/turboquant-pytorch/ (hyphen in dir name
# prevents direct Python import). Add vendor/ to sys.path so `from tq.x import Y` works.
_vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "vendor"))
if os.path.isdir(os.path.join(_vendor_dir, "tq")) and _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

try:
    from tq.compressors_v3 import TurboQuantV3
    TURBOQUANT_AVAILABLE = True
except ImportError as e:
    TURBOQUANT_AVAILABLE = False

try:
    from transformers import DynamicCache
    DYNAMIC_CACHE_AVAILABLE = True
except ImportError:
    DYNAMIC_CACHE_AVAILABLE = False


# Default config — K4/V4 with 128-token residual window
DEFAULT_KEY_BITS = 4
DEFAULT_VALUE_BITS = 4
DEFAULT_RESIDUAL_WINDOW = 128
DEFAULT_PROTECTED_LAYERS = 1  # Protect first and last full-attention layers


# Try to import the repacking utility for Triton kernel compatibility
try:
    from alice.core.optimization.turboquant_attention import _repack_msb_to_lsb
    TRITON_REPACK_AVAILABLE = True
except ImportError:
    TRITON_REPACK_AVAILABLE = False


# ---------------------------------------------------------------------------
# Generic TurboQuantDynamicCache — for standard transformer models
# ---------------------------------------------------------------------------

class TurboQuantDynamicCache(DynamicCache if DYNAMIC_CACHE_AVAILABLE else object):
    """
    Drop-in replacement for DynamicCache with TurboQuant V3 compression.

    Compresses all layers. Works with any standard HF transformer model
    (TTS, STT, future models, etc.).

    Compression kicks in once sequence exceeds residual_window tokens.
    """

    def __init__(
        self,
        n_layers: int = 28,
        head_dim: int = 128,
        key_bits: int = DEFAULT_KEY_BITS,
        value_bits: int = DEFAULT_VALUE_BITS,
        residual_window: int = DEFAULT_RESIDUAL_WINDOW,
        protected_layers: int = DEFAULT_PROTECTED_LAYERS,
    ):
        super().__init__()

        self.tq_n_layers = n_layers
        self.tq_head_dim = head_dim
        self.tq_key_bits = key_bits
        self.tq_value_bits = value_bits
        self.tq_residual_window = residual_window
        self.tq_protected_layers = protected_layers

        self._compressors = {}
        self._compressed_chunks_k = {}
        self._compressed_chunks_v = {}
        self._compressed_token_count = {}

    def _get_compressor(self, layer_idx: int, head_dim: int, device) -> "TurboQuantV3":
        if layer_idx not in self._compressors:
            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.tq_key_bits,
                value_bits=self.tq_value_bits,
                residual_window=0,  # We handle windowing ourselves
                layer_idx=layer_idx,
                n_layers=self.tq_n_layers,
                protected_layers=self.tq_protected_layers,
                seed=42,
                device=str(device),
            )
            self._compressed_chunks_k[layer_idx] = []
            self._compressed_chunks_v[layer_idx] = []
            self._compressed_token_count[layer_idx] = 0

        return self._compressors[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Override update to compress old KV tokens via TurboQuantV3.

        1. Append new tokens via parent DynamicCache
        2. If total seq > residual_window, compress overflow
        3. Decompress all chunks + concat with recent window
        4. Return full KV for attention
        """
        if not TURBOQUANT_AVAILABLE:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)

        # Let parent handle layer creation and append
        out_k, out_v = super().update(key_states, value_states, layer_idx, cache_kwargs)

        B, H, total_seq, D = out_k.shape
        rw = self.tq_residual_window
        comp = self._get_compressor(layer_idx, D, key_states.device)

        if rw > 0 and total_seq > rw:
            overflow = total_seq - rw

            to_compress_k = out_k[:, :, :overflow, :]
            to_compress_v = out_v[:, :, :overflow, :]

            ck, cv = comp.compress_kv(to_compress_k, to_compress_v)
            self._compressed_chunks_k[layer_idx].append(ck)
            self._compressed_chunks_v[layer_idx].append(cv)
            self._compressed_token_count[layer_idx] += overflow

            # Trim the layer's raw cache to residual window only
            layer = self.layers[layer_idx]
            layer.keys = out_k[:, :, overflow:, :]
            layer.values = out_v[:, :, overflow:, :]

        # Rebuild full KV from compressed chunks + raw
        if self._compressed_chunks_k.get(layer_idx):
            parts_k = []
            parts_v = []
            for ck, cv in zip(
                self._compressed_chunks_k[layer_idx],
                self._compressed_chunks_v[layer_idx],
            ):
                dk, dv = comp.decompress_kv(ck, cv)
                parts_k.append(dk.to(key_states.dtype))
                parts_v.append(dv.to(value_states.dtype))

            parts_k.append(self.layers[layer_idx].keys)
            parts_v.append(self.layers[layer_idx].values)

            return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

        return self.layers[layer_idx].keys, self.layers[layer_idx].values

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Total sequence length including compressed tokens."""
        compressed = self._compressed_token_count.get(layer_idx, 0)
        raw = 0
        if layer_idx < len(self.layers) and self.layers[layer_idx].is_initialized:
            raw = self.layers[layer_idx].keys.shape[-2]
        return compressed + raw

    def get_compression_stats(self) -> dict:
        """Return compression stats for logging."""
        if not self._compressed_token_count:
            return {"compressed_tokens": 0, "total_tokens": 0, "layers_compressed": 0}

        layer_idx = 0
        compressed = self._compressed_token_count.get(layer_idx, 0)
        raw = 0
        if layer_idx < len(self.layers) and self.layers[layer_idx].is_initialized:
            raw = self.layers[layer_idx].keys.shape[-2]
        total = compressed + raw

        return {
            "compressed_tokens": compressed,
            "fp16_tokens": raw,
            "total_tokens": total,
            "layers_compressed": len([l for l in range(self.tq_n_layers) if self._compressed_token_count.get(l, 0) > 0]),
            "key_bits": self.tq_key_bits,
            "value_bits": self.tq_value_bits,
            "residual_window": self.tq_residual_window,
        }


# ---------------------------------------------------------------------------
# TurboQuantHybridCache — hybrid architecture (linear + full attention)
# ---------------------------------------------------------------------------

class TurboQuantHybridCache(DynamicCache if DYNAMIC_CACHE_AVAILABLE else object):
    """
    Drop-in replacement for DynamicCache (hybrid architecture) with TurboQuant V3
    KV compression on the full-attention layers only.

    Subclasses modern DynamicCache(config=) which reads config.layer_types to
    set up hybrid linear/full-attention layer objects automatically.
    Linear-attention layers (conv_states, recurrent_states) pass through
    untouched — they use update_conv_state/update_recurrent_state, not update().

    Compression only kicks in once the sequence exceeds residual_window tokens.
    Short conversations run at full fp16 precision with zero overhead.
    """

    def __init__(
        self,
        config,
        key_bits: int = DEFAULT_KEY_BITS,
        value_bits: int = DEFAULT_VALUE_BITS,
        residual_window: int = DEFAULT_RESIDUAL_WINDOW,
        protected_layers: int = DEFAULT_PROTECTED_LAYERS,
        triton_mode: bool = False,
    ):
        super().__init__(config=config)

        self.transformer_layers = [
            i for i, lt in enumerate(config.layer_types)
            if lt == "full_attention"
        ]

        self.tq_key_bits = key_bits
        self.tq_value_bits = value_bits
        self.tq_residual_window = residual_window
        self.tq_protected_layers = protected_layers
        self.triton_mode = triton_mode

        self._compressors = {}
        self._compressed_chunks_k = {}
        self._compressed_chunks_v = {}
        self._compressed_token_count = {}

    def _get_compressor(self, layer_idx: int, head_dim: int, device) -> "TurboQuantV3":
        if layer_idx not in self._compressors:
            fa_position = self.transformer_layers.index(layer_idx)
            n_fa_layers = len(self.transformer_layers)

            self._compressors[layer_idx] = TurboQuantV3(
                head_dim=head_dim,
                key_bits=self.tq_key_bits,
                value_bits=self.tq_value_bits,
                residual_window=0,
                layer_idx=fa_position,
                n_layers=n_fa_layers,
                protected_layers=self.tq_protected_layers,
                seed=42,
                device=str(device),
            )
            self._compressed_chunks_k[layer_idx] = []
            self._compressed_chunks_v[layer_idx] = []
            self._compressed_token_count[layer_idx] = 0

        return self._compressors[layer_idx]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Override update for full-attention layers to compress old KV tokens.

        Linear-attention layers never hit this method (they use
        update_conv_state / update_recurrent_state), so every call here
        is a full-attention layer managed by a DynamicLayer object.
        """
        if not TURBOQUANT_AVAILABLE or layer_idx not in self.transformer_layers:
            return super().update(key_states, value_states, layer_idx, *args, **kwargs)

        if self.triton_mode:
            try:
                import alice.core.optimization.turboquant_attention as _tqa
                _tqa._active_cache = self
            except ImportError:
                pass

        # Let parent DynamicCache append via DynamicLayer.update()
        out_k, out_v = super().update(key_states, value_states, layer_idx, *args, **kwargs)

        B, H, total_seq, D = out_k.shape
        S_new = key_states.shape[2]
        comp = self._get_compressor(layer_idx, D, key_states.device)
        rw = self.tq_residual_window
        layer = self.layers[layer_idx]

        if rw > 0 and total_seq > rw:
            overflow = total_seq - rw

            to_compress_k = out_k[:, :, :overflow, :]
            to_compress_v = out_v[:, :, :overflow, :]

            ck, cv = comp.compress_kv(to_compress_k, to_compress_v)
            self._compressed_chunks_k[layer_idx].append(ck)
            self._compressed_chunks_v[layer_idx].append(cv)
            self._compressed_token_count[layer_idx] += overflow

            layer.keys = out_k[:, :, overflow:, :]
            layer.values = out_v[:, :, overflow:, :]

        if self.triton_mode and S_new == 1 and self._compressed_chunks_k.get(layer_idx):
            return layer.keys, layer.values

        if self._compressed_chunks_k.get(layer_idx):
            parts_k = []
            parts_v = []
            for ck, cv in zip(
                self._compressed_chunks_k[layer_idx],
                self._compressed_chunks_v[layer_idx],
            ):
                dk, dv = comp.decompress_kv(ck, cv)
                parts_k.append(dk.to(key_states.dtype))
                parts_v.append(dv.to(value_states.dtype))

            parts_k.append(layer.keys)
            parts_v.append(layer.values)

            return torch.cat(parts_k, dim=2), torch.cat(parts_v, dim=2)

        return layer.keys, layer.values

    def get_compressed_kv(self, layer_idx: int) -> dict | None:
        """
        Return compressed key data + decompressed values for Triton attention.

        Only available in triton_mode when compressed chunks exist.
        Results are cached and invalidated when new chunks are added.

        Returns dict with:
          - mse_packed: (B, H, N_comp, packed_d) uint8 — repacked LSB-first for Triton
          - norms: (B, H, N_comp) float16 — original vector L2 norms
          - centroids: (n_clusters,) float32 — codebook centroids
          - Pi: (D, D) float32 — rotation matrix
          - mse_bits: int — quantization bits
          - values_fp16: (B, H, N_comp, D) — decompressed values
        Or None if no compressed tokens exist for this layer.
        """
        if not self.triton_mode:
            return None
        if layer_idx not in self.transformer_layers:
            return None
        if not self._compressed_chunks_k.get(layer_idx):
            return None

        n_chunks = len(self._compressed_chunks_k[layer_idx])
        cache_key = f"_tq_cached_{layer_idx}"
        cached = getattr(self, cache_key, None)
        if cached is not None and cached.get("_n_chunks") == n_chunks:
            return cached

        comp = self._compressors[layer_idx]
        key_comp = comp.key_compressor
        val_comp = comp.val_compressor

        all_idx_bytes = []
        all_norms = []
        all_values = []

        for ck, cv in zip(
            self._compressed_chunks_k[layer_idx],
            self._compressed_chunks_v[layer_idx],
        ):
            if ck["compressed"] is not None:
                all_idx_bytes.append(ck["compressed"]["idx_bytes"])
                all_norms.append(ck["compressed"]["vec_norms"])
                dv = val_comp.decompress(cv["compressed"])
                all_values.append(dv)

        if not all_idx_bytes:
            return None

        mse_packed = torch.cat(all_idx_bytes, dim=2)
        norms = torch.cat(all_norms, dim=2)
        values_fp16 = torch.cat(all_values, dim=2)

        if TRITON_REPACK_AVAILABLE:
            mse_packed = _repack_msb_to_lsb(mse_packed, comp.key_bits)

        result = {
            "mse_packed": mse_packed,
            "norms": norms,
            "centroids": key_comp.centroids,
            "Pi": key_comp.Pi,
            "mse_bits": comp.key_bits,
            "values_fp16": values_fp16.to(torch.bfloat16),
            "_n_chunks": n_chunks,
        }
        setattr(self, cache_key, result)
        return result

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        """Total sequence length including compressed tokens."""
        if layer_idx not in self.transformer_layers:
            layer_idx = self.transformer_layers[0]
        compressed = self._compressed_token_count.get(layer_idx, 0)
        raw = self.layers[layer_idx].get_seq_length() if layer_idx < len(self.layers) else 0
        return compressed + raw

    def get_compression_stats(self) -> dict:
        """Return compression stats for logging."""
        if not self._compressed_token_count:
            return {"compressed_tokens": 0, "total_tokens": 0, "layers_compressed": 0}

        layer_idx = self.transformer_layers[0]
        compressed = self._compressed_token_count.get(layer_idx, 0)
        raw = self.layers[layer_idx].get_seq_length() if layer_idx < len(self.layers) else 0
        total = compressed + raw

        return {
            "compressed_tokens": compressed,
            "fp16_tokens": raw,
            "total_tokens": total,
            "layers_compressed": len([l for l in self.transformer_layers if self._compressed_token_count.get(l, 0) > 0]),
            "key_bits": self.tq_key_bits,
            "value_bits": self.tq_value_bits,
            "residual_window": self.tq_residual_window,
        }


# Backward compat alias
TurboQuantCache = TurboQuantHybridCache


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_cache(config=None, model_type: str = "auto", triton_mode: bool = False, **kwargs):
    """
    Factory function — returns the right TurboQuant cache if enabled, else default.

    Args:
        config: Model config (required for hybrid architecture, optional for generic)
        model_type: "hybrid", "generic", or "auto" (auto-detects from config)
        triton_mode: When True, cache skips decompression — attention reads compressed
            data via get_compressed_kv() and uses Triton kernels. Only for hybrid arch.
        **kwargs: Passed to cache constructor (key_bits, value_bits, etc.)
            For generic: n_layers, head_dim required
    """
    enabled = os.environ.get("ALICE_TURBOQUANT", "1") != "0"

    if enabled and not TURBOQUANT_AVAILABLE:
        print("WARNING: ALICE_TURBOQUANT=1 but turboquant-pytorch not found in vendor/")
        return _fallback_cache(config)

    if not enabled:
        return _fallback_cache(config)

    # Auto-detect model type from config. The legacy model-specific DynamicCache
    # was removed from transformers; modern path uses plain DynamicCache(config=)
    # which introspects config.layer_types to build the hybrid cache.
    if model_type == "auto" and config is not None:
        model_cls = getattr(config, "model_type", "")
        if model_cls.startswith("hybrid"):
            model_type = "hybrid"
        else:
            model_type = "generic"

    # TODO: enable after testing TurboQuantHybridCache port to modern DynamicCache
    # if model_type == "hybrid" and DYNAMIC_CACHE_AVAILABLE and config is not None:
    #     return TurboQuantHybridCache(config, triton_mode=triton_mode, **kwargs)

    if model_type == "hybrid" and config is not None and DYNAMIC_CACHE_AVAILABLE:
        return DynamicCache(config=config)

    # Generic path
    return TurboQuantDynamicCache(**kwargs)


def _fallback_cache(config):
    """Return default cache when TQ is disabled.

    Modern transformers uses plain DynamicCache with config passed in —
    DynamicCache introspects config.layer_types to handle hybrid
    linear/full-attention layouts.
    """
    if config is not None and getattr(config, "model_type", "").startswith("hybrid"):
        if DYNAMIC_CACHE_AVAILABLE:
            return DynamicCache(config=config)
    return None
