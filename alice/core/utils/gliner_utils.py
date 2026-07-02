"""
GLiNER-based Fact Extraction Utility for Alice
==============================================

Uses GLiNER (Generalist and Lightweight NER @ NAACL 2024) for
zero-shot entity extraction from conversations.

RAM-optimized with lazy loading and minimal entity labels.
"""

from typing import Dict, List, Optional
import time
from pathlib import Path
import gc

class GLiNERUtility:
    """
    Shared GLiNER model for fact extraction across the Hive

    Design principles:
    - Lazy loading (only load when needed)
    - Minimal entity labels (reduce processing)
    - Memory cleanup after use
    - Fast inference (<100ms target)
    - Reusable singleton pattern

    RAM Optimization:
    - Model loads on first use (lazy)
    - Uses minimal label set (6 categories vs 20+)
    - Can unload model to free ~600MB RAM
    """

    def __init__(self, model_name: str = "urchade/gliner_small-v2.1", lazy_load: bool = True):
        """
        Initialize the GLiNER utility

        Args:
            model_name: HuggingFace model name for GLiNER
            lazy_load: If True, delay model loading until first use
        """
        self.model_name = model_name
        self.model = None
        self._is_loaded = False

        # Minimal entity labels (optimized for RAM)
        # Only 6 categories with 2-3 labels each = 15 total labels
        # (vs previous 20+ labels)
        self.entity_labels = {
            "identity": ["person name", "name"],
            "preferences": ["food", "drink", "hobby"],
            "professional": ["job title", "company"],
            "relationships": ["pet name", "pet"],
            "location": ["city", "location"],
            "interests": ["hobby", "interest"],
        }

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the GLiNER model (called on first use if lazy loading)"""
        if self._is_loaded:
            return

        print(f"🔧 [GLINER] Loading model: {self.model_name}")
        try:
            from gliner import GLiNER
            self.model = GLiNER.from_pretrained(self.model_name)
            self._is_loaded = True
            print(f"✅ [GLINER] Model loaded successfully (~600MB RAM)")
        except Exception as e:
            print(f"❌ [GLINER] Failed to load model: {e}")
            raise

    def unload_model(self):
        """Unload model to free ~600MB RAM"""
        if self.model is not None:
            del self.model
            self.model = None
            self._is_loaded = False
            gc.collect()
            print(f"♻️ [GLINER] Model unloaded, freed ~600MB RAM")

    def extract_facts(self, user_input: str, assistant_response: str) -> List[Dict[str, str]]:
        """
        Extract structured facts from a conversation turn

        Args:
            user_input: What the user said
            assistant_response: What Alice said

        Returns:
            List of facts: [{"key": "name", "value": "Rin", "category": "identity"}, ...]
        """
        # Lazy load model on first use
        if not self._is_loaded:
            self._load_model()

        # Focus on user input (that's where the facts are)
        text = user_input

        # Get all entity labels (reduced to 15 labels for speed)
        all_labels = []
        for category, labels in self.entity_labels.items():
            all_labels.extend(labels)

        # Extract entities
        try:
            entities = self.model.predict_entities(text, all_labels, threshold=0.5)
        except Exception as e:
            print(f"⚠️ [GLINER] Entity extraction failed: {e}")
            return []

        # Convert entities to facts
        facts = []
        for entity in entities:
            # Map entity label to category
            category = self._map_label_to_category(entity['label'])
            key = self._normalize_key(entity['label'])

            fact = {
                "key": key,
                "value": entity['text'],
                "category": category,
                "confidence": entity['score']
            }
            facts.append(fact)

        return facts

    def _map_label_to_category(self, label: str) -> str:
        """Map entity label to fact category"""
        for category, labels in self.entity_labels.items():
            if label in labels:
                return category
        return "general"

    def _normalize_key(self, label: str) -> str:
        """Normalize entity label to a clean key"""
        # Remove spaces, lowercase
        key = label.replace(" ", "_").lower()
        return key


# Global singleton instance (lazy loaded)
_gliner_utility_instance = None

def get_gliner_utility(lazy_load: bool = True) -> GLiNERUtility:
    """
    Get or create the global GLiNER utility instance

    Args:
        lazy_load: If True, model loads on first use (saves RAM at startup)

    Returns:
        Singleton GLiNERUtility instance

    RAM Usage:
        - With lazy_load=True: ~0MB at startup, ~600MB after first use
        - With lazy_load=False: ~600MB at startup
    """
    global _gliner_utility_instance

    if _gliner_utility_instance is None:
        _gliner_utility_instance = GLiNERUtility(lazy_load=lazy_load)

    return _gliner_utility_instance

def unload_gliner():
    """
    Unload the global GLiNER model to free ~600MB RAM

    Useful for low-memory situations. Model will reload on next use.
    """
    global _gliner_utility_instance

    if _gliner_utility_instance is not None:
        _gliner_utility_instance.unload_model()
