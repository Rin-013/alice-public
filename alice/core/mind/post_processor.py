"""
Post-Processor — Mind reads Alice's response and handles bookkeeping.

This is called by Mind automatically after notify_response().
Separated here for clarity, but the logic lives in Mind._do_post_process().

This module provides standalone utilities for post-processing that can
be used independently of Mind (e.g., for testing or manual invocation).
"""

from typing import Dict, List, Optional


def extract_iris_queries(alice_response: str, user_input: str) -> List[str]:
    """
    Heuristic: extract potential IRIS search queries from conversation context.
    Used as fallback when Mind model isn't available.
    """
    queries = []

    # If user asked a question, search for related memories
    if '?' in user_input:
        # Use the user's question as an IRIS query
        queries.append(user_input)

    # If Alice mentioned remembering something
    remember_words = ['remember', 'recall', 'last time', 'before', 'earlier']
    response_lower = alice_response.lower()
    for word in remember_words:
        if word in response_lower:
            queries.append(f"previous conversations about {user_input[:50]}")
            break

    return queries


def estimate_avatar_intent(alice_response: str, current_emotion: str = "neutral") -> str:
    """
    Simple heuristic avatar intent from Alice's response text.
    Used as fallback when Mind model isn't available.
    """
    text = alice_response.lower()

    if any(w in text for w in ['haha', 'lol', 'lmao', '😂', 'funny']):
        return "happy"
    if any(w in text for w in ['hmm', 'interesting', 'wonder', 'think']):
        return "thinking"
    if any(w in text for w in ['!', 'wow', 'amazing', 'awesome', 'omg']):
        return "excited"
    if any(w in text for w in ['whatever', 'sure', 'fine', 'obviously']):
        return "sassy"
    if any(w in text for w in ['sorry', 'sad', 'miss', 'wish']):
        return "sad"

    return current_emotion
