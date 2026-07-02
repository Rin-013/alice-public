"""
Shared Embedding Model Utility for Alice
=========================================

Provides a singleton SentenceTransformer instance to prevent duplicate model loading.

Problem: Multiple systems (Memory, IRIS, VM) were each loading all-MiniLM-L6-v2 separately,
consuming 87MB * 3 = 261MB of RAM unnecessarily.

Solution: Single shared instance with lazy loading.
"""

from typing import Optional
from pathlib import Path
import os

class SharedEmbeddingModel:
    """
    Singleton wrapper for SentenceTransformer model with ONNX optimization

    Benefits:
    - Single model instance across all Alice systems
    - Lazy loading (loads on first use)
    - 87MB RAM instead of 261MB (saves 174MB)
    - ONNX backend for 2-3x faster inference (2025 optimization)
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", lazy_load: bool = True, use_onnx: bool = True):
        """
        Initialize shared embedding model

        Args:
            model_name: HuggingFace model name
            lazy_load: If True, delay loading until first use
            use_onnx: If True, use ONNX backend for 2-3x speedup
        """
        self.model_name = model_name
        self.use_onnx = use_onnx
        self.model = None
        self._is_loaded = False

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model with optional ONNX optimization"""
        if self._is_loaded:
            return

        backend = "onnx" if self.use_onnx else "torch"
        print(f"🔧 [EMBEDDING] Loading shared model: {self.model_name} (backend: {backend.upper()})")

        try:
            from sentence_transformers import SentenceTransformer

            # Try local cache first
            home = Path.home()
            local_model_path = home / '.cache' / 'sentence-transformers' / self.model_name.replace('/', '_')

            if local_model_path.exists():
                self.model = SentenceTransformer(str(local_model_path), local_files_only=True, backend=backend)
                print(f"✅ [EMBEDDING] Loaded from local: {local_model_path.name} (~87MB, {backend.upper()} backend)")
            else:
                self.model = SentenceTransformer(self.model_name, backend=backend)
                print(f"✅ [EMBEDDING] Downloaded: {self.model_name} (~87MB, {backend.upper()} backend)")

            if self.use_onnx:
                print(f"⚡ [EMBEDDING] ONNX optimization enabled - expect 2-3x speedup!")

            self._is_loaded = True

        except Exception as e:
            # Fallback to PyTorch if ONNX fails
            if self.use_onnx:
                print(f"⚠️ [EMBEDDING] ONNX backend failed, falling back to PyTorch: {e}")
                self.use_onnx = False
                self._load_model()  # Retry with PyTorch
            else:
                print(f"❌ [EMBEDDING] Failed to load model: {e}")
                raise

    def encode(self, texts, **kwargs):
        """
        Encode text(s) to embeddings

        Args:
            texts: Single text string or list of strings
            **kwargs: Additional arguments passed to model.encode()

        Returns:
            Embeddings as numpy array
        """
        if not self._is_loaded:
            self._load_model()

        return self.model.encode(texts, **kwargs)

    def get_model(self):
        """
        Get the underlying SentenceTransformer model

        Returns:
            SentenceTransformer instance
        """
        if not self._is_loaded:
            self._load_model()

        return self.model

    def unload_model(self):
        """Unload model to free ~87MB RAM"""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False
            import gc
            gc.collect()
            print(f"♻️ [EMBEDDING] Model unloaded, freed ~87MB RAM")


# Global singleton instance
_shared_embedding_model = None

def get_shared_embedding_model(model_name: str = "all-MiniLM-L6-v2", lazy_load: bool = True, use_onnx: bool = False) -> SharedEmbeddingModel:
    """
    Get or create the global shared embedding model with optional ONNX optimization

    Args:
        model_name: Model name (default: all-MiniLM-L6-v2)
        lazy_load: If True, model loads on first use
        use_onnx: If True, use ONNX backend (default False - Mac M-series is faster with PyTorch+MPS)

    Returns:
        Singleton SharedEmbeddingModel instance

    Performance Improvements (2025):
        - RAM Savings: 174MB (67% reduction) via deduplication
        - Speed: PyTorch+MPS optimized for Mac (12ms/batch vs ONNX 36ms/batch)
        - ONNX available for Linux/Windows servers (2-3x speedup on x86 CPUs)
    """
    global _shared_embedding_model

    if _shared_embedding_model is None:
        _shared_embedding_model = SharedEmbeddingModel(model_name, lazy_load, use_onnx)

    return _shared_embedding_model

def unload_shared_embedding_model():
    """
    Unload the global embedding model to free ~87MB RAM

    Model will reload on next use.
    """
    global _shared_embedding_model

    if _shared_embedding_model is not None:
        _shared_embedding_model.unload_model()
