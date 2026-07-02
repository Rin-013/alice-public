"""
CUDA Stream Manager — isolates GPU workloads per system.

Each system (TTS, STT, LLM, emotion) gets its own CUDA stream so compiled
kernels and mixed-precision ops don't interfere across models sharing one GPU.
"""
import torch

_streams: dict[str, torch.cuda.Stream] = {}

def get_stream(name: str) -> torch.cuda.Stream:
    """Get or create a dedicated CUDA stream for the named system."""
    if name not in _streams:
        _streams[name] = torch.cuda.Stream()
    return _streams[name]
