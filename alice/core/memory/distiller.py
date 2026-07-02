"""
Session Distiller
=================

Extracts salient facts from a session's conversation exchanges using a small
LLM. Runs at end_session(), at shutdown — latency doesn't matter there.

Cold and analytical — no personality. The distiller's job is pattern
recognition, not conversation. It watches what Rin said across the session
and extracts the pieces worth remembering permanently. This is the
trained-model replacement for the regex smart_facts extractor (gated off
2026-06-10 after it filled the DB with garbage facts).

Uses Alice's own model — at shutdown it's loaded and idle, so extraction costs
zero extra GPU memory and zero load time. Model choice was probed 2026-06-10:
the smaller base model is too weak (copies example facts as real ones, garbles
names, keeps moods); the larger model was 4/4 clean at temp 0.3. Mind's merged
checkpoint is unusable for this — tuned into Alice's personality, it answers in
her voice.

Usage (from chat.py shutdown, with Alice's model still loaded):

    from alice.core.memory.distiller import SessionDistiller

    memory.end_session(distiller=SessionDistiller(),
                       distill_model=model, distill_tokenizer=tokenizer)

`load_distill_model()` exists for standalone runs (tests, replays) and loads
the same model.
"""

import json
import os
import re
from typing import Optional

DEFAULT_DISTILL_MODEL = "models/alice_main"


def load_distill_model(model_dir: str = None):
    """
    Load the extraction model (quantized on cuda:0, same recipe as Mind's loader).
    Returns (model, tokenizer) or (None, None) if anything is missing —
    callers fall back to the raw session flush.
    """
    model_dir = (model_dir
                 or os.environ.get("ALICE_DISTILL_MODEL")
                 or os.environ.get("ALICE_MODEL")
                 or DEFAULT_DISTILL_MODEL)
    if not os.path.exists(model_dir):
        return None, None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from weight_quantization import QuantizationConfig
        quant_config = QuantizationConfig(
            load_in_4bit=True,
            compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            quantization_config=quant_config,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model.eval()
        return model, tokenizer
    except Exception:
        return None, None


# ── Extraction prompt ──────────────────────────────────────────────────────────
# NO example facts in the prompt: small models copy them out as if they were
# real ("Rin lives in Austin, Texas" entered the output verbatim in testing).
# Short and abstract; grounding comes from the conversation block itself.
_EXTRACT_PROMPT = """\
Below is a conversation between Rin and Alice. Extract durable facts about Rin \
that Rin explicitly stated. A fact must be:
- something Rin actually said in THIS conversation (quote-level grounded, no guessing)
- still true next week (no moods, no jokes, no questions)
- short and self-contained

Output ONLY a JSON array of fact strings. If there are none, output [].

Conversation:
{exchanges}

Facts:\
"""


def _format_exchanges(exchanges: list) -> str:
    """Format raw conversation exchanges into a readable block."""
    lines = []
    for ex in exchanges:
        # ShortTermMemory exchange format: {user_message, assistant_response, ...}
        user_msg = ex.get("user_message", ex.get("content", ""))
        alice_msg = ex.get("assistant_response", "")
        if user_msg:
            lines.append(f"Rin: {user_msg}")
        if alice_msg:
            lines.append(f"Alice: {alice_msg}")
    return "\n".join(lines)


def _parse_json_array(text: str) -> list[str]:
    """
    Robustly extract a JSON array of strings from model output.
    Handles partial output, trailing garbage, minor formatting errors.
    """
    text = text.strip()

    # Find the first [ ... ] block
    start = text.find('[')
    if start == -1:
        return []

    # Find matching close bracket
    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == '[':
            depth += 1
        elif ch == ']':
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        # Model cut off — try closing manually
        text = text[start:] + "]"
        end = len(text)
        start = 0

    candidate = text[start:end]

    try:
        result = json.loads(candidate)
        if isinstance(result, list):
            return [str(x).strip() for x in result if isinstance(x, str) and x.strip()]
    except json.JSONDecodeError:
        pass

    # Fix trailing comma and retry (common LLM output artifact)
    fixed = re.sub(r',\s*]', ']', candidate)
    try:
        result = json.loads(fixed)
        if isinstance(result, list):
            return [str(x).strip() for x in result if isinstance(x, str) and x.strip()]
    except json.JSONDecodeError:
        pass

    # Fallback: extract quoted strings one by one
    return re.findall(r'"([^"]{10,})"', candidate)


class SessionDistiller:
    """
    Extracts memorable facts from a session's conversation exchanges.

    Model-agnostic — pass any mlx model + tokenizer at call time.
    Typically runs after Alice and Ghost have been unloaded to free GPU memory.
    """

    def __init__(self, max_exchanges: int = 40, min_exchange_words: int = 4):
        """
        Args:
            max_exchanges:      Maximum exchanges to send to the model (oldest dropped first)
            min_exchange_words: Skip exchanges shorter than this (filters out noise)
        """
        self.max_exchanges = max_exchanges
        self.min_exchange_words = min_exchange_words

    def distill(self, model, tokenizer, exchanges: list) -> list[str]:
        """
        Run fact extraction over a list of session exchanges.

        Args:
            model:     Loaded transformers CausalLM (see load_distill_model)
            tokenizer: Corresponding tokenizer
            exchanges: List of exchange dicts from ShortTermMemory.conversation_history

        Returns:
            List of fact strings, ready to store in total memory.
            Empty list if nothing extractable or model fails.
        """
        import torch

        # Filter noise and cap length
        filtered = [
            ex for ex in exchanges
            if len((ex.get("user_message", "") + " " + ex.get("assistant_response", "")).split()) >= self.min_exchange_words
        ]
        filtered = filtered[-self.max_exchanges:]  # most recent N

        if not filtered:
            return []

        exchange_text = _format_exchanges(filtered)
        prompt_text = _EXTRACT_PROMPT.format(exchanges=exchange_text)

        # Prefill a short CLOSED think block: an empty one makes the model
        # play it safe ([] every time), an open one makes it think past any
        # token budget. Stating the intent and closing the tag goes straight
        # to the array. With thinking closed, low temp is safe (the greedy
        # repetition loop only happens inside long thinking).
        prompt = (
            f"<|im_start|>system\n"
            f"You extract facts. You output only JSON arrays.\n"
            f"<|im_end|>\n"
            f"<|im_start|>user\n{prompt_text}<|im_end|>\n"
            f"<|im_start|>assistant\n<think>\n"
            f"I will list only facts Rin explicitly stated in this exact "
            f"conversation, skipping moods and jokes.\n</think>\n"
        )

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=300,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            output = tokenizer.decode(
                out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
            )
        except Exception:
            return []

        output = output.split("<|im_end|>")[0].split("<|im_start|>")[0]
        return _parse_json_array(output)

    def distill_into_memory(self, memory_system, model, tokenizer,
                             importance: float = None) -> int:
        """
        All-in-one: distill session exchanges and store facts in total memory.

        Replaces the raw conversation flush in end_session() when a distiller
        is available. Returns the number of facts stored.

        Args:
            memory_system: AliceMemorySystem instance
            model:         Loaded transformers CausalLM
            tokenizer:     Corresponding tokenizer
            importance:    Importance score for distilled facts (higher than
                           raw conversation memories since they've been curated)

        Returns:
            Number of facts stored in total memory.
        """
        from .types import Memory, MemoryType
        from .importance import score_importance

        exchanges = memory_system.short_term.conversation_history
        if not exchanges:
            return 0

        facts = self.distill(model, tokenizer, exchanges)
        if not facts:
            return 0

        stored = 0
        user_id = memory_system.current_user.user_id if memory_system.current_user else None

        for fact in facts:
            if not fact.strip():
                continue
            # Score each fact individually; caller-supplied importance acts as floor
            fact_importance = score_importance(fact, base=importance if importance is not None else 0.60)
            memory = Memory.create(
                content=fact,
                memory_type=MemoryType.FACT,
                user_id=user_id,
                importance=fact_importance,
            )
            try:
                memory_system.long_term.add_memory(memory)  # dedup handles near-duplicates
                stored += 1
            except Exception:
                pass

        return stored


__all__ = ["SessionDistiller", "load_distill_model", "DEFAULT_DISTILL_MODEL"]
