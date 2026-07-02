"""
spaCy-based Fact Extraction Utility for Alice
==============================================

Uses spaCy NER for lightweight, fast fact extraction from conversations.

RAM Efficiency: 269MB total (vs GLiNER's 1,610MB = 83% savings)
"""

from typing import Dict, List, Optional
import time
from pathlib import Path

class SpaCyUtility:
    """
    Shared spaCy model for fact extraction across the Hive

    Design principles:
    - Lightweight NER model (~60MB in RAM)
    - Fast inference (<10ms)
    - Battle-tested production NER
    - Reusable singleton pattern

    RAM Optimization:
    - Model: ~60MB (vs GLiNER's ~1,300MB)
    - Total RAM: ~270MB (vs GLiNER's ~1,610MB)
    - 83% RAM savings!
    """

    def __init__(self, model_name: str = "en_core_web_sm", lazy_load: bool = True):
        """
        Initialize the spaCy utility

        Args:
            model_name: spaCy model name
            lazy_load: If True, delay model loading until first use
        """
        self.model_name = model_name
        self.nlp = None
        self._is_loaded = False

        # Entity label mapping (spaCy -> Alice categories)
        self.entity_mapping = {
            "PERSON": "identity",
            "ORG": "professional",
            "GPE": "location",          # Geopolitical entity (cities, countries)
            "LOC": "location",
            "FAC": "location",          # Facilities
            "PRODUCT": "preferences",
            "EVENT": "interests",
            "WORK_OF_ART": "interests",
            "MONEY": "preferences",
            "DATE": "context",
            "TIME": "context",
        }

        if not lazy_load:
            self._load_model()

    def _load_model(self):
        """Load the spaCy model (called on first use if lazy loading)"""
        if self._is_loaded:
            return

        print(f"🔧 [SPACY] Loading model: {self.model_name}")
        try:
            import spacy
            self.nlp = spacy.load(self.model_name)
            self._is_loaded = True
            print(f"✅ [SPACY] Model loaded successfully (~60MB RAM)")
        except Exception as e:
            print(f"❌ [SPACY] Failed to load model: {e}")
            print(f"   Run: python -m spacy download {self.model_name}")
            raise

    def unload_model(self):
        """Unload model to free ~60MB RAM"""
        if self.nlp is not None:
            del self.nlp
            self.nlp = None
            self._is_loaded = False
            import gc
            gc.collect()
            print(f"♻️ [SPACY] Model unloaded, freed ~60MB RAM")

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
        doc = self.nlp(user_input)

        # Extract entities
        facts = []
        for ent in doc.ents:
            # Map spaCy entity type to Alice category
            category = self.entity_mapping.get(ent.label_, "general")

            # Create key from entity label
            key = self._entity_to_key(ent.label_)

            fact = {
                "key": key,
                "value": ent.text,
                "category": category,
                "confidence": 0.85,  # spaCy doesn't provide scores, use default
                "entity_type": ent.label_  # Keep original label for reference
            }
            facts.append(fact)

        return facts

    def _entity_to_key(self, entity_label: str) -> str:
        """Convert spaCy entity label to fact key"""
        key_map = {
            "PERSON": "person_name",
            "ORG": "company",
            "GPE": "location",
            "LOC": "location",
            "FAC": "facility",
            "PRODUCT": "product",
            "EVENT": "event",
            "WORK_OF_ART": "work",
            "MONEY": "money",
            "DATE": "date",
            "TIME": "time",
        }
        return key_map.get(entity_label, entity_label.lower())


# Global singleton instance (lazy loaded)
_spacy_utility_instance = None

def get_spacy_utility(lazy_load: bool = True) -> SpaCyUtility:
    """
    Get or create the global spaCy utility instance

    Args:
        lazy_load: If True, model loads on first use (saves RAM at startup)

    Returns:
        Singleton SpaCyUtility instance

    RAM Usage:
        - With lazy_load=True: ~0MB at startup, ~270MB after first use
        - With lazy_load=False: ~270MB at startup
    """
    global _spacy_utility_instance

    if _spacy_utility_instance is None:
        _spacy_utility_instance = SpaCyUtility(lazy_load=lazy_load)

    return _spacy_utility_instance

def unload_spacy():
    """
    Unload the global spaCy model to free ~60MB RAM

    Useful for low-memory situations. Model will reload on next use.
    """
    global _spacy_utility_instance

    if _spacy_utility_instance is not None:
        _spacy_utility_instance.unload_model()
