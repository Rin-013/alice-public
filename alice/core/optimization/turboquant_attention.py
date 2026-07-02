"""
TurboQuant fused attention — Triton kernel integration for HF transformers.

Registers a custom attention function that computes attention scores directly
on TurboQuant-compressed KV cache data using the Triton MSE kernel, avoiding
the decompress-to-fp16 round-trip.

Architecture:
  - Compressed tokens: Triton MSE kernel (turboquant_mse_score)
  - Residual window tokens: Standard matmul (fp16, exact)
  - Scores merged via log-sum-exp (flash-attention style online softmax)

Only activates for full-attention layers in the hybrid architecture.
Linear-attention layers don't use KV cache at all.

Usage:
    Set config._attn_implementation = "turboquant" and pass TurboQuant cache.
    Falls back to eager attention when no compressed data is available.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

log = logging.getLogger("alice.turboquant_attention")

try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    HF_DISPATCH_AVAILABLE = True
except ImportError:
    HF_DISPATCH_AVAILABLE = False

# Import Triton MSE score kernel
try:
    import sys, os
    _vendor_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "vendor"))
    # Add vendor/turboquant to path for the triton_kernels module
    _tq_dir = os.path.join(_vendor_dir, "turboquant")
    if os.path.isdir(_tq_dir) and _tq_dir not in sys.path:
        sys.path.insert(0, _tq_dir)
    from turboquant.triton_kernels import turboquant_mse_score
    TRITON_AVAILABLE = True
except ImportError as e:
    TRITON_AVAILABLE = False
    log.debug(f"Triton MSE kernel not available: {e}")


def _repack_msb_to_lsb(idx_bytes: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Repack MSE indices from MSB-first (compressor format) to LSB-first (Triton kernel format).

    The turboquant-pytorch compressor packs indices MSB-first:
        byte = idx[0] << (bits*(n-1)) | idx[1] << (bits*(n-2)) | ... | idx[n-1]
    The Triton kernel expects LSB-first:
        byte = idx[n-1] << (bits*(n-1)) | ... | idx[1] << bits | idx[0]
    """
    if bits >= 8:
        return idx_bytes  # No packing, no swap needed

    indices_per_byte = 8 // bits
    if indices_per_byte <= 1:
        return idx_bytes

    mask = (1 << bits) - 1
    packed = idx_bytes.long()

    # Unpack all indices from MSB-first format
    msb_shifts = torch.tensor(
        [bits * i for i in range(indices_per_byte - 1, -1, -1)],
        dtype=torch.long, device=idx_bytes.device,
    )
    # (... , n_groups, indices_per_byte)
    unpacked = (packed.unsqueeze(-1) >> msb_shifts) & mask

    # Repack in LSB-first format
    lsb_shifts = torch.tensor(
        [bits * i for i in range(indices_per_byte)],
        dtype=torch.long, device=idx_bytes.device,
    )
    repacked = (unpacked << lsb_shifts).sum(-1).to(torch.uint8)

    return repacked


# Global reference to the active TurboQuant cache — set by cache.update(),
# read by turboquant_attention_forward(). Needed because HF transformers
# consumes past_key_values before calling the attention function and doesn't
# pass it through kwargs.
_active_cache = None


def _repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match query heads (GQA support)."""
    if n_rep == 1:
        return hidden_states
    B, H, S, D = hidden_states.shape
    return hidden_states[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)


def turboquant_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    """
    Custom attention function that uses Triton MSE kernel on compressed keys.

    When compressed KV data is available in the cache, computes:
      1. Triton MSE scores for compressed tokens (fast, ~4 bits per element)
      2. Standard matmul scores for residual window tokens (fp16, exact)
      3. Merges via log-sum-exp for numerically stable softmax across segments

    Falls back to eager attention when no compressed data is available.
    """
    # Access cache via global reference (HF consumes past_key_values before
    # calling attention, so it's not available in kwargs)
    cache = _active_cache

    # Check if we have compressed data available for this layer
    compressed = None
    if (
        TRITON_AVAILABLE
        and cache is not None
        and hasattr(cache, "get_compressed_kv")
        and hasattr(module, "layer_idx")
    ):
        compressed = cache.get_compressed_kv(module.layer_idx)

    B, n_heads, S_q, D = query.shape

    if compressed is None or S_q > 1:
        # No compressed data, or prefill (S_q > 1) — fall back to eager attention.
        # Triton MSE kernel is decode-only (expects single query token).
        return _eager_attention(module, query, key, value, attention_mask, scaling, dropout)

    # === Fused TurboQuant attention (decode only, S_q=1) ===
    n_rep = module.num_key_value_groups

    # Compressed segment data
    mse_packed = compressed["mse_packed"]       # (B, H_kv, N_comp, packed_d) uint8, LSB-first
    norms = compressed["norms"]                 # (B, H_kv, N_comp) float16
    centroids = compressed["centroids"]         # (n_clusters,) float32
    Pi = compressed["Pi"]                       # (D, D) float32
    mse_bits = compressed["mse_bits"]           # int
    N_comp = mse_packed.shape[2]

    # Residual window — these are the key/value states returned by cache.update()
    # They only contain the residual window tokens (recent fp16)
    recent_k = _repeat_kv(key, n_rep)           # (B, n_heads, N_recent, D)
    recent_v = _repeat_kv(value, n_rep)
    N_recent = recent_k.shape[2]

    # Also decompress values for the compressed segment
    values_decompressed = compressed["values_fp16"]  # (B, H_kv, N_comp, D)
    values_decompressed = _repeat_kv(values_decompressed, n_rep)

    # --- Compute scores for compressed tokens via Triton ---
    # Rotate query: q_rot = query @ Pi^T  (one matmul per decode step)
    # query shape: (B, n_heads, S_q, D) — S_q is typically 1 for decode
    q_flat = query.reshape(B * n_heads, S_q, D)
    q_rot = torch.matmul(q_flat.squeeze(1).float(), Pi.T)  # (B*n_heads, D)

    # Expand mse_packed and norms for GQA: repeat KV heads to match query heads
    if n_rep > 1:
        H_kv = mse_packed.shape[1]
        mse_packed_exp = mse_packed[:, :, None, :, :].expand(B, H_kv, n_rep, N_comp, -1)
        mse_packed_exp = mse_packed_exp.reshape(B * n_heads, N_comp, -1)
        norms_exp = norms[:, :, None, :].expand(B, H_kv, n_rep, N_comp)
        norms_exp = norms_exp.reshape(B * n_heads, N_comp)
    else:
        mse_packed_exp = mse_packed.reshape(B * n_heads, N_comp, -1)
        norms_exp = norms.reshape(B * n_heads, N_comp)

    # Triton MSE score: returns (B*n_heads, N_comp) raw logits
    scores_comp = turboquant_mse_score(
        q_rot, mse_packed_exp, norms_exp.float(), centroids, mse_bits
    )  # (BH, N_comp)
    scores_comp = scores_comp * scaling  # apply 1/sqrt(d)
    scores_comp = scores_comp.reshape(B, n_heads, 1, N_comp)

    # --- Compute scores for residual window via standard matmul ---
    scores_recent = torch.matmul(query, recent_k.transpose(2, 3)) * scaling  # (B, n_heads, S_q, N_recent)

    # --- Apply attention mask ---
    # The mask covers the full sequence [compressed | recent]
    # We need to split it accordingly
    if attention_mask is not None:
        N_total = N_comp + N_recent
        if attention_mask.shape[-1] == N_total:
            mask_comp = attention_mask[:, :, :, :N_comp]
            mask_recent = attention_mask[:, :, :, N_comp:]
            scores_comp = scores_comp + mask_comp
            scores_recent = scores_recent + mask_recent
        elif attention_mask.shape[-1] == N_recent:
            # Mask only covers recent tokens (compressed assumed visible)
            scores_recent = scores_recent + attention_mask
        else:
            # Best effort: apply to recent only
            scores_recent = scores_recent + attention_mask

    # --- Merge via log-sum-exp (online softmax) ---
    # Numerically stable softmax across [compressed | recent] segments
    max_comp = scores_comp.amax(dim=-1, keepdim=True)       # (B, H, S_q, 1)
    max_recent = scores_recent.amax(dim=-1, keepdim=True)
    max_all = torch.maximum(max_comp, max_recent)

    exp_comp = torch.exp(scores_comp - max_all)       # (B, H, S_q, N_comp)
    exp_recent = torch.exp(scores_recent - max_all)   # (B, H, S_q, N_recent)

    sum_exp_comp = exp_comp.sum(dim=-1, keepdim=True)
    sum_exp_recent = exp_recent.sum(dim=-1, keepdim=True)
    sum_exp_total = sum_exp_comp + sum_exp_recent

    # Weighted value aggregation
    # Compressed segment: exp_comp @ values_decompressed
    attn_comp = torch.matmul(exp_comp.to(query.dtype), values_decompressed)  # (B, H, S_q, D)
    # Recent segment: exp_recent @ recent_v
    attn_recent = torch.matmul(exp_recent.to(query.dtype), recent_v)          # (B, H, S_q, D)

    attn_output = (attn_comp + attn_recent) / sum_exp_total.to(query.dtype)

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, None


def _eager_attention(module, query, key, value, attention_mask, scaling, dropout):
    """Standard eager attention fallback."""
    n_rep = getattr(module, "num_key_value_groups", 1)
    key_states = _repeat_kv(key, n_rep)
    value_states = _repeat_kv(value, n_rep)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


def register_turboquant_attention():
    """Register turboquant attention in HF's ALL_ATTENTION_FUNCTIONS dispatch."""
    if not HF_DISPATCH_AVAILABLE:
        log.warning("Cannot register turboquant attention: transformers.modeling_utils not available")
        return False

    if not TRITON_AVAILABLE:
        log.warning("Cannot register turboquant attention: Triton MSE kernel not available")
        return False

    ALL_ATTENTION_FUNCTIONS["turboquant"] = turboquant_attention_forward
    log.info("Registered turboquant attention in ALL_ATTENTION_FUNCTIONS")
    return True
