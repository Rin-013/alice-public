"""
LLM Utility Service for Alice
==============================

Provides a shared, lightweight LLM for quick inference tasks across the Hive:
- Fact extraction from conversations
- Text classification
- Entity recognition
- Intent detection
- Quick Q&A

Uses a small, fast model for <100ms inference.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Any, Optional
import json
import re
from pathlib import Path

class LLMUtility:
    """
    Shared lightweight LLM for quick inference tasks

    Design principles:
    - Small model (lightweight params)
    - Fast inference (<100ms target)
    - JSON output format for structured data
    - Reusable across Hive systems
    """

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the utility LLM

        Args:
            model_path: Path to model. If None, uses Alice's main model.
        """
        # For now, reuse Alice's model (we can swap to smaller later)
        if model_path is None:
            model_path = "models/alice-v1.0"

        print(f"🔧 [LLM UTILITY] Loading model from {model_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="mps" if torch.backends.mps.is_available() else "cpu",
            torch_dtype=torch.float16 if torch.backends.mps.is_available() else torch.float32,
            trust_remote_code=True
        )

        print(f"✅ [LLM UTILITY] Loaded ({self.model.num_parameters() / 1e9:.1f}B parameters)")

    def extract_facts(self, user_input: str, assistant_response: str) -> List[Dict[str, str]]:
        """
        Extract structured facts from a conversation turn

        Args:
            user_input: What the user said
            assistant_response: What Alice said

        Returns:
            List of facts: [{"key": "name", "value": "Rin", "category": "identity"}, ...]
        """
        prompt = f"""Extract factual information from this conversation.

User: {user_input}
Assistant: {assistant_response}

Identify facts about the USER (not the assistant). Return ONLY a JSON array of facts.

Format: [{{"key": "name", "value": "John", "category": "identity"}}, ...]

Categories: identity, preferences, professional, relationships, interests, location

Facts (JSON only, no explanation):"""

        # Generate
        response = self._generate(prompt, max_tokens=200, temperature=0.1)

        # Parse JSON
        facts = self._parse_json_response(response)

        return facts if facts else []

    def classify_intent(self, user_input: str) -> str:
        """
        Classify user intent (question, statement, command, etc.)

        Returns:
            Intent label: "question", "statement", "command", "greeting", "other"
        """
        prompt = f"""Classify the user's intent in one word.

User: {user_input}

Intent (one word - question/statement/command/greeting/other):"""

        response = self._generate(prompt, max_tokens=5, temperature=0.0)

        # Extract first word
        intent = response.strip().split()[0].lower()

        return intent if intent in ["question", "statement", "command", "greeting"] else "other"

    def _generate(self, prompt: str, max_tokens: int = 100, temperature: float = 0.3) -> str:
        """
        Generate response using ChatML format

        Args:
            prompt: The prompt
            max_tokens: Max new tokens
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        formatted_prompt = f"""<|im_start|>system
You are a precise fact extraction assistant. Follow instructions exactly.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
"""

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        if torch.backends.mps.is_available():
            inputs = {k: v.to("mps") for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                repetition_penalty=1.1,
                do_sample=temperature > 0,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # Clean up stop tokens
        for stop in ["<|im_end|>", "<|im_start|>", "\n\n"]:
            if stop in response:
                response = response.split(stop)[0]
                break

        return response.strip()

    def _parse_json_response(self, response: str) -> Optional[List[Dict[str, str]]]:
        """
        Parse JSON from LLM response

        Args:
            response: Raw LLM output

        Returns:
            Parsed list of dicts, or None if parsing fails
        """
        try:
            # Try to find JSON array in response
            match = re.search(r'\[.*\]', response, re.DOTALL)
            if match:
                json_str = match.group(0)
                facts = json.loads(json_str)

                # Validate structure
                if isinstance(facts, list):
                    validated = []
                    for fact in facts:
                        if isinstance(fact, dict) and "key" in fact and "value" in fact:
                            validated.append({
                                "key": fact["key"],
                                "value": fact["value"],
                                "category": fact.get("category", "general")
                            })
                    return validated

            return None

        except json.JSONDecodeError:
            print(f"⚠️ [LLM UTILITY] Failed to parse JSON: {response[:100]}")
            return None


# Global singleton instance (lazy loaded)
_llm_utility_instance = None

def get_llm_utility() -> LLMUtility:
    """
    Get or create the global LLM utility instance

    Returns:
        Singleton LLMUtility instance
    """
    global _llm_utility_instance

    if _llm_utility_instance is None:
        _llm_utility_instance = LLMUtility()

    return _llm_utility_instance
