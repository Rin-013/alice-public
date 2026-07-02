# Copyright 2025 Rin - Alice AI System
"""
Short-Term Memory
==================

Session-based memory for current conversation context.
Holds recent exchanges, current emotional state, and session-specific facts.
"""

import time
import uuid
from typing import Dict, List, Any, Optional


class ShortTermMemory:
    """
    Session-based memory for current conversation context.

    Maintains:
    - Recent conversation exchanges
    - Current emotional state
    - Active context/topics
    - Session-specific facts
    """

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self.session_id = str(uuid.uuid4())
        self.session_start = time.time()
        self.current_user: Optional[str] = None

        # Memory stores
        self.conversation_history: List[Dict[str, Any]] = []
        self.context_stack: List[str] = []  # Topics being discussed
        self.emotional_state: Dict[str, Any] = {
            "mood": "neutral",
            "energy_level": 0.5,
            "sass_level": 0.5,
            "engagement": 0.5
        }
        self.session_facts: Dict[str, Any] = {}  # Facts learned this session
        self.active_topics: List[str] = []

    def add_exchange(self,
                     user_message: str,
                     assistant_response: str,
                     emotional_context: Optional[str] = None,
                     chaos_level: float = 0.5):
        """Add user-Alice exchange to conversation history"""
        exchange = {
            "timestamp": time.time(),
            "user_message": user_message,
            "assistant_response": assistant_response,
            "emotional_context": emotional_context,
            "chaos_level": chaos_level,
            "alice_mood": self.emotional_state["mood"]
        }

        self.conversation_history.append(exchange)

        # Maintain size limit
        if len(self.conversation_history) > self.max_entries:
            self.conversation_history.pop(0)

    def update_emotional_state(self,
                               mood: Optional[str] = None,
                               energy_delta: float = 0.0,
                               sass_delta: float = 0.0,
                               engagement_delta: float = 0.0):
        """Update Alice's current emotional state"""
        if mood:
            self.emotional_state["mood"] = mood

        # Apply deltas with bounds checking
        self.emotional_state["energy_level"] = max(0.0, min(1.0,
            self.emotional_state["energy_level"] + energy_delta))
        self.emotional_state["sass_level"] = max(0.0, min(1.0,
            self.emotional_state["sass_level"] + sass_delta))
        self.emotional_state["engagement"] = max(0.0, min(1.0,
            self.emotional_state["engagement"] + engagement_delta))

    def add_context(self, topic: str):
        """Add topic to current context stack"""
        if topic not in self.context_stack:
            self.context_stack.append(topic)

        # Keep recent context
        if len(self.context_stack) > 10:
            self.context_stack.pop(0)

    def learn_fact(self, key: str, value: Any):
        """Learn a fact during this session"""
        self.session_facts[key] = value

    def get_recent_context(self, num_exchanges: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        return self.conversation_history[-num_exchanges:]

    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of current session"""
        return {
            "session_id": self.session_id,
            "duration": time.time() - self.session_start,
            "exchanges": len(self.conversation_history),
            "current_mood": self.emotional_state["mood"],
            "topics_discussed": self.active_topics,
            "facts_learned": len(self.session_facts),
            "current_user": self.current_user
        }

    def clear(self):
        """Clear session memory (for new session)"""
        self.session_id = str(uuid.uuid4())
        self.session_start = time.time()
        self.conversation_history = []
        self.context_stack = []
        self.session_facts = {}
        self.active_topics = []
        self.emotional_state = {
            "mood": "neutral",
            "energy_level": 0.5,
            "sass_level": 0.5,
            "engagement": 0.5
        }

    def set_user(self, user_id: str):
        """Set current user for session"""
        self.current_user = user_id


# For backwards compatibility
__all__ = ['ShortTermMemory']
