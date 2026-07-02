# Copyright 2025 Rin - Alice AI System
"""
Smart Fact Extraction — structured knowledge extraction from conversations.

Rewritten for IRIS. Extracts facts via spaCy (primary), LLM, or regex patterns.
Called from IRIS.add_conversation() after storing the exchange.
"""

import json
import time
import re
import uuid
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

from .types import Memory, MemoryType


class FactType(Enum):
    """Types of facts that can be extracted"""
    PREFERENCE = "preference"
    PERSONAL_INFO = "personal_info"
    RELATIONSHIP = "relationship"
    INTEREST = "interest"
    GOAL = "goal"
    OPINION = "opinion"
    EXPERIENCE = "experience"
    SKILL = "skill"
    HABIT = "habit"
    CONTEXT = "context"


@dataclass
class StructuredFact:
    """A structured fact extracted from conversation"""
    fact_id: str
    fact_type: FactType
    category: str
    key: str
    value: str
    confidence: float
    source_conversation: str
    timestamp: float
    user_id: str
    context: Optional[str] = None
    supersedes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['fact_type'] = self.fact_type.value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructuredFact':
        data['fact_type'] = FactType(data['fact_type'])
        return cls(**data)


@dataclass
class QueryIntent:
    """Understanding of what the user is asking for"""
    intent_type: str
    category: str
    subcategory: Optional[str]
    search_terms: List[str]
    expected_answer_type: str
    confidence: float


class SmartFactExtractor:
    """
    Extracts structured facts from conversations and stores them via IRIS.
    """

    def __init__(self, iris):
        """
        Args:
            iris: IRIS instance (the single memory interface)
        """
        self.iris = iris

        # Try spaCy for extraction
        self.spacy_utility = None
        try:
            from ..utils.spacy_utils import get_spacy_utility
            self.spacy_utility = get_spacy_utility()
        except Exception:
            pass

        # In-memory fact index (category -> key -> fact_id)
        self.structured_facts: Dict[str, StructuredFact] = {}
        self.fact_categories: Dict[str, Dict[str, str]] = {}

        # Regex fallback patterns
        self.extraction_patterns = {
            FactType.PREFERENCE: [
                r"my favorite (.+?) is (.+)",
                r"i (?:like|love|prefer) (.+)",
                r"i'm (?:into|fond of) (.+)",
            ],
            FactType.PERSONAL_INFO: [
                r"i'm (?:a |an )?(.+)",
                r"i (?:work|am) (?:as )?(?:a |an )?(.+)",
                r"my (?:job|occupation|work) is (.+)",
                r"i (?:study|am studying|'m studying) (.+)",
                r"i live in (.+)",
                r"my name is ([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
            ],
            FactType.RELATIONSHIP: [
                r"my (.+?) (?:is|lives|works) (.+)",
                r"i have (?:a |an )?(.+?) (?:named|called) (.+)",
            ],
            FactType.INTEREST: [
                r"i'm (?:interested in|passionate about) (.+)",
                r"i (?:enjoy|love) (.+)",
                r"my hobby is (.+)",
                r"interested in (.+)",
            ]
        }

    def extract_facts_from_conversation(self, user_input: str, alice_response: str,
                                        user_id: str) -> List[StructuredFact]:
        """Extract structured facts from a conversation exchange."""
        facts = []

        # Try spaCy first
        if self.spacy_utility:
            try:
                spacy_facts = self._extract_with_spacy(user_input, alice_response, user_id)
                facts.extend(spacy_facts)
            except Exception:
                pass

        # Fallback to patterns
        if not facts:
            facts = self._extract_with_patterns(user_input, user_id)

        # Store each fact
        stored = []
        for fact in facts:
            result = self._store_structured_fact(fact)
            if result:
                stored.append(result)

        return stored

    def _extract_with_spacy(self, user_input: str, alice_response: str,
                            user_id: str) -> List[StructuredFact]:
        """Extract facts using spaCy utility."""
        raw_facts = self.spacy_utility.extract_facts(user_input, alice_response)

        fact_type_map = {
            "identity": FactType.PERSONAL_INFO,
            "preferences": FactType.PREFERENCE,
            "professional": FactType.PERSONAL_INFO,
            "relationships": FactType.RELATIONSHIP,
            "location": FactType.PERSONAL_INFO,
            "interests": FactType.INTEREST,
            "general": FactType.CONTEXT,
        }

        structured = []
        for raw in raw_facts:
            fact = StructuredFact(
                fact_id=str(uuid.uuid4()),
                fact_type=fact_type_map.get(raw["category"], FactType.CONTEXT),
                category=raw["category"],
                key=raw["key"],
                value=raw["value"],
                confidence=raw.get("confidence", 0.85),
                source_conversation=f"User: {user_input[:50]}...",
                timestamp=time.time(),
                user_id=user_id,
            )
            structured.append(fact)

        return structured

    def _extract_with_patterns(self, user_input: str, user_id: str) -> List[StructuredFact]:
        """Extract facts using regex patterns (fallback)."""
        facts = []
        user_lower = user_input.lower()

        for fact_type, patterns in self.extraction_patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, user_lower):
                    category, key, value = self._categorize_pattern_match(
                        fact_type, match, user_input
                    )
                    if category and key and value:
                        fact = StructuredFact(
                            fact_id=f"pattern_{int(time.time() * 1000)}_{len(facts)}",
                            fact_type=fact_type,
                            category=category,
                            key=key,
                            value=value,
                            confidence=0.7,
                            source_conversation=user_input,
                            timestamp=time.time(),
                            user_id=user_id,
                        )
                        facts.append(fact)

        return facts

    def _categorize_pattern_match(self, fact_type: FactType, match,
                                  original_text: str) -> Tuple[str, str, str]:
        """Categorize a pattern match into (category, key, value)."""
        if fact_type == FactType.PREFERENCE:
            if "favorite" in original_text.lower() and match.lastindex and match.lastindex >= 2:
                category_item = match.group(1).strip()
                value = match.group(2).strip()
                if any(w in category_item for w in ['drink', 'beverage']):
                    return "beverages", "favorite", value
                elif any(w in category_item for w in ['food', 'meal', 'dish']):
                    return "food", "favorite", value
                elif any(w in category_item for w in ['color', 'colour']):
                    return "aesthetics", "favorite_color", value
                else:
                    return "preferences", f"favorite_{category_item}", value
            else:
                return "preferences", "likes", match.group(1).strip()

        elif fact_type == FactType.PERSONAL_INFO:
            if "i'm" in original_text.lower() or "i am" in original_text.lower():
                value = match.group(1).strip()
                job_indicators = [
                    'engineer', 'developer', 'designer', 'manager', 'teacher',
                    'doctor', 'lawyer', 'programmer', 'analyst', 'student',
                    'researcher', 'scientist', 'professor', 'writer', 'artist',
                ]
                if any(ind in value.lower() for ind in job_indicators):
                    return "professional", "occupation", value
                return "personal", "description", value
            elif any(w in original_text.lower() for w in ['work', 'job', 'occupation']):
                return "professional", "occupation", match.group(1).strip()
            elif any(w in original_text.lower() for w in ['study', 'studying']):
                return "education", "current_study", match.group(1).strip()
            elif "live" in original_text.lower():
                return "location", "residence", match.group(1).strip()
            elif "name" in original_text.lower():
                return "identity", "name", match.group(1).strip()

        elif fact_type == FactType.RELATIONSHIP:
            if match.lastindex and match.lastindex >= 2:
                rel_type = match.group(1).strip()
                name = match.group(2).strip()
                if any(w in rel_type for w in ['cat', 'kitten']):
                    return "pets", "cat_name", name
                elif any(w in rel_type for w in ['dog', 'puppy']):
                    return "pets", "dog_name", name
                else:
                    return "relationships", f"{rel_type}_name", name

        elif fact_type == FactType.INTEREST:
            return "interests", "general", match.group(1).strip()

        return None, None, None

    def _store_structured_fact(self, fact: StructuredFact) -> Optional[StructuredFact]:
        """Store a structured fact, handling conflicts with existing facts."""
        # Check for existing fact in same category/key
        existing_id = self.fact_categories.get(fact.category, {}).get(fact.key)

        if existing_id and existing_id in self.structured_facts:
            existing = self.structured_facts[existing_id]
            if fact.confidence >= existing.confidence:
                fact.supersedes = existing_id
                existing.confidence *= 0.5
            else:
                return None

        # Store in memory
        self.structured_facts[fact.fact_id] = fact
        if fact.category not in self.fact_categories:
            self.fact_categories[fact.category] = {}
        self.fact_categories[fact.category][fact.key] = fact.fact_id

        # Store in IRIS long-term
        if self.iris.long_term:
            content = f"[FACT] {fact.category}.{fact.key} = {fact.value}"
            memory = Memory(
                id=str(uuid.uuid4()),
                timestamp=fact.timestamp,
                memory_type=MemoryType.FACT,
                content=content,
                importance=min(0.9, fact.confidence + 0.2),
                user_id=fact.user_id,
                tags=['structured_fact', fact.category, fact.fact_type.value],
            )
            try:
                self.iris.long_term.add_memory(memory)
            except Exception:
                pass

        return fact

    def search_facts_by_intent(self, query: str, user_id: str) -> Optional[str]:
        """Search for facts based on natural language query intent."""
        intent = self._analyze_query_intent(query)
        if not intent:
            return None
        fact = self._find_fact_by_intent(intent, user_id)
        if fact:
            return self._format_fact_response(fact, intent)
        return None

    def _analyze_query_intent(self, query: str) -> Optional[QueryIntent]:
        """Analyze what the user is asking for."""
        q = query.lower()

        if "what" in q and "favorite" in q:
            if any(w in q for w in ['drink', 'beverage']):
                return QueryIntent("retrieve_preference", "beverages", "favorite",
                                   ["drink", "beverage"], "specific_item", 0.9)
            elif any(w in q for w in ['food', 'meal']):
                return QueryIntent("retrieve_preference", "food", "favorite",
                                   ["food", "meal"], "specific_item", 0.9)
            elif "color" in q:
                return QueryIntent("retrieve_preference", "aesthetics", "favorite_color",
                                   ["color"], "specific_item", 0.9)

        elif any(p in q for p in ["what do you do", "what's your job",
                                   "what do i do", "what's my job"]):
            return QueryIntent("retrieve_info", "professional", "occupation",
                               ["job", "work", "occupation"], "specific_item", 0.8)

        elif any(p in q for p in ["where do you live", "where do i live",
                                   "where are you from", "where am i from"]):
            return QueryIntent("retrieve_info", "location", "residence",
                               ["live", "location"], "specific_item", 0.8)

        return None

    def _find_fact_by_intent(self, intent: QueryIntent,
                             user_id: str) -> Optional[StructuredFact]:
        """Find a fact matching the query intent."""
        # Direct lookup
        if intent.category in self.fact_categories:
            if intent.subcategory in self.fact_categories[intent.category]:
                fact_id = self.fact_categories[intent.category][intent.subcategory]
                fact = self.structured_facts.get(fact_id)
                if fact and fact.user_id == user_id and fact.confidence > 0.3:
                    return fact

        # Keyword fallback
        user_facts = sorted(
            [f for f in self.structured_facts.values()
             if f.user_id == user_id and f.confidence > 0.3],
            key=lambda f: f.confidence, reverse=True
        )
        for fact in user_facts:
            if intent.category == fact.category:
                return fact
            if any(t in fact.key.lower() or t in fact.value.lower()
                   for t in intent.search_terms):
                return fact
        return None

    def _format_fact_response(self, fact: StructuredFact, intent: QueryIntent) -> str:
        """Format a fact into a natural response."""
        if intent.intent_type == "retrieve_preference":
            return f"Obviously it's {fact.value}, peasant."
        elif intent.intent_type == "retrieve_info":
            if intent.category == "professional":
                return f"You work as {fact.value}, if I remember correctly."
            elif intent.category == "location":
                return f"You live in {fact.value}."
            return f"You told me {fact.value}."
        return fact.value

    def get_user_fact_summary(self, user_id: str) -> Dict[str, Any]:
        """Get a summary of all facts known about a user."""
        user_facts = [f for f in self.structured_facts.values() if f.user_id == user_id]
        summary = {"total_facts": len(user_facts), "categories": {}}
        for fact in user_facts:
            if fact.category not in summary["categories"]:
                summary["categories"][fact.category] = []
            summary["categories"][fact.category].append({
                "key": fact.key, "value": fact.value, "confidence": fact.confidence
            })
        return summary
