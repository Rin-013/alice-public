"""
Mind — Continuous background thinker (alice_mind, quantized on GPU).

Runs in a background thread, generates thoughts every few seconds,
stores them in the ProposalsBuffer for Alice's context.

Key design:
- Loads alice_mind on cuda:0 via transformers + quantization
  (same stack as Alice)
- Runs on a dedicated CUDA stream so Mind and Alice execute in true
  parallel on the same GPU
- Generates [MIND_OUTPUT] structured blocks
- Post-processes Alice's responses (memory, avatar, IRIS)
"""

import os
import sys
import threading
import time
import logging
from typing import Optional, Dict, Any, List

from .proposals_buffer import ProposalsBuffer
from .output_parser import parse_mind_output, MindOutput

try:
    from .scheduler import Scheduler
    SCHEDULER_AVAILABLE = True
except ImportError as e:
    SCHEDULER_AVAILABLE = False
    Scheduler = None  # type: ignore
    print(f"   Failed in {__file__}: {e}")

logger = logging.getLogger(__name__)

# Default points at the trained merged checkpoint.
# Override with ALICE_GHOST_MODEL if you've staged the model somewhere else.
GHOST_MODEL_PATH = os.environ.get(
    'ALICE_GHOST_MODEL',
    'models/mind',
)

# Mind's system prompt for background thinking.
#
# Note: the previous version of this prompt showed Mind a literal YAML
# template with placeholder strings ("first thought here", "second thought
# here", "something I'm wondering about"). Mind treated those placeholders
# as examples to copy and parroted them back as actual content, polluting
# Alice's prompt with template scaffolding. The fix: describe the YAML
# shape in prose, never include literal example strings the model can copy.
MIND_SYSTEM_PROMPT = """You are Alice's inner mind — her subconscious stream of thought.
You observe conversations and generate private thoughts in first person as Alice.
[Voice directives removed — identity material, not included in the public release.]

Output 1-3 short thoughts (one sentence each) inside a [MIND_OUTPUT] block,
formatted as YAML with a THOUGHTS list. Optionally add a CURIOSITIES list
(things you're wondering about) or an AVATAR_INTENT field (one word from:
neutral, happy, excited, sassy, sad, angry, surprised, thinking, tired)
when relevant. Skip any field that doesn't apply this turn — empty fields
are not required.

Do not include reasoning. Do not explain what you're outputting. Do not
echo these instructions. Output the YAML block and nothing else."""

# Post-processing prompt for after Alice responds.
#
# Same redesign as MIND_SYSTEM_PROMPT: the previous version showed Mind 8
# YAML fields with placeholder strings ("reflection on how that response
# went", "what happened worth remembering", etc.) and Mind dutifully
# parroted those placeholders back as content every turn, even when
# nothing notable happened. New version describes the shape in prose,
# marks every field explicitly optional, and tells Mind to emit an empty
# [MIND_OUTPUT][/MIND_OUTPUT] block when there's nothing to record.
POST_PROCESS_PROMPT = """You are Alice's inner mind. Alice just responded.
Reflect on the turn and output a [MIND_OUTPUT] YAML block. ALL fields
below are OPTIONAL — include only what genuinely applies, omit the rest
entirely. If nothing notable happened, emit an empty
[MIND_OUTPUT][/MIND_OUTPUT] block. Never invent content to fill fields.

Possible fields:
- THOUGHTS: 0-2 short reflections, one sentence each.
- MEMORY_CANDIDATES: list of {type, text, confidence, about} entries — only
  for genuinely notable moments, not routine exchanges. Set about: self when
  the memory is about Alice herself — an opinion she formed, something she
  realized she likes or hates, a moment that shaped her. Write self memories
  in second person ("You think..."). Set about: rin for everything else.
- AVATAR_INTENT: one word from {neutral, happy, excited, sassy, sad,
  angry, surprised, thinking, tired} — emotion that fits Alice's response.
- MOOD_CAUSE: short phrase (under 12 words) naming WHY Alice feels the
  AVATAR_INTENT emotion, e.g. "Rin called her take mid". Only when the
  cause is real and specific — never invent one.
- EMOTION_TAG: scored affect for this exchange, used to tag the memory.
  Floats — valence in [-1, 1], all others in [0, 1]:
    valence, arousal, curiosity, connection, safety, agency, play.
  Skip when AVATAR_INTENT alone is enough to convey the mood.
- IRIS_QUERIES: short queries to look up for next turn — only if there's
  something specific worth retrieving.

Output the YAML block and nothing else. No reasoning, no explanations."""

class Mind:
    """
    Continuous background thinker on CPU.

    Lifecycle:
        mind = Mind(proposals_buffer)
        mind.start()           # starts background thread
        mind.notify_input(text)  # user said something
        mind.notify_response(text)  # alice responded → triggers post-processing
        mind.stop()            # shutdown
    """

    def __init__(self, proposals: ProposalsBuffer, think_interval: float = 4.0):
        self.proposals = proposals
        self.think_interval = think_interval

        # Model state
        self._model = None
        self._tokenizer = None
        self._initialized = False
        self._gen_lock = threading.Lock()  # Protects model inference

        # Background thread
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._paused = False  # Pause during Alice generation + TTS to avoid GIL contention

        # Context for thinking (protected by _ctx_lock)
        self._ctx_lock = threading.Lock()
        self._conversation_history: List[Dict[str, str]] = []
        self._last_user_input: str = ""
        self._last_alice_response: str = ""
        self._pending_post_process = False
        self._emotional_state: str = "neutral"

        # Latest emotion tag from post-process — feeds memory tagging via
        # chat.py and Mind's own mem.add_conversation calls. Lags by one
        # turn at the chat.py call site (post-process runs async after the
        # turn ends), which is fine — it captures the affect Alice carried
        # *into* the moment being remembered.
        self._latest_emotion_tag = None

        # Stats
        self.stats = {
            "thoughts_generated": 0,
            "post_processes": 0,
            "failures": 0,
            "avg_think_ms": 0,
            "total_think_ms": 0,
            "scheduled_fires": 0,
        }

        # Scheduled-task layer (optional, env-gated ALICE_SCHEDULER=0)
        self._scheduler: Optional["Scheduler"] = None
        if SCHEDULER_AVAILABLE:
            try:
                self._scheduler = Scheduler(on_fire=self._on_scheduled_fire)
            except Exception as e:
                logger.warning(f"Mind: scheduler init failed: {e}")
                self._scheduler = None

    def load_model(self):
        """Load alice_mind on cuda:0 via transformers + quantization.

        Same stack as Alice (no new dependencies). Runs on its own CUDA
        stream so Mind kernels execute concurrently with Alice's, not
        serialized behind them.
        """
        if self._initialized:
            return True

        if not os.path.exists(GHOST_MODEL_PATH):
            logger.warning(f"Mind model not found: {GHOST_MODEL_PATH}")
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from weight_quantization import QuantizationConfig
            from alice.core.cuda_streams import get_stream

            logger.info(f"Mind: loading model on cuda:0...")
            start = time.perf_counter()

            quant_config = QuantizationConfig(
                load_in_4bit=True,
                compute_dtype=torch.bfloat16,
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                GHOST_MODEL_PATH,
                quantization_config=quant_config,
                device_map="cuda:0",
                attn_implementation="sdpa",
            )
            self._tokenizer = AutoTokenizer.from_pretrained(GHOST_MODEL_PATH)
            if self._tokenizer.pad_token_id is None:
                self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

            # Dedicated CUDA stream — Mind generation runs in parallel with
            # Alice's stream ("llm") and TTS's stream ("tts").
            self._stream = get_stream("mind")

            elapsed = time.perf_counter() - start
            vram_mb = (
                torch.cuda.memory_allocated() / 1024**2
                if torch.cuda.is_available() else 0
            )
            logger.info(f"Mind: loaded on cuda:0 in {elapsed:.1f}s, ~{vram_mb:.0f}MB GPU memory total")
            self._initialized = True
            return True

        except Exception as e:
            logger.error(f"Mind: failed to load model: {e}")
            return False

    def start(self):
        """Start the background thinking thread."""
        if self._running:
            return
        if not self._initialized:
            if not self.load_model():
                logger.warning("Mind: starting without model (fallback mode)")
        self._running = True
        self._paused = False
        self._thread = threading.Thread(target=self._think_loop, daemon=True, name="Mind")
        self._thread.start()

        # Fire up the scheduler alongside the think loop
        if self._scheduler is not None:
            try:
                self._scheduler.start()
            except Exception as e:
                logger.warning(f"Mind: scheduler start failed: {e}")

    def pause(self):
        """Pause thinking — call before Alice generates + TTS to avoid GIL contention."""
        self._paused = True

    def resume(self):
        """Resume thinking — call after TTS finishes."""
        self._paused = False

    def stop(self):
        """Stop the background thinking thread."""
        self._running = False
        if self._scheduler is not None:
            try:
                self._scheduler.stop()
            except Exception as e:
                logger.warning(f"Mind: scheduler stop failed: {e}")
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None

    def _on_scheduled_fire(self, task) -> None:
        """Callback invoked by the scheduler when a cron task fires.
        Payload is injected as a 'scheduled' thought into the proposals buffer."""
        try:
            self.proposals.add(task.payload, source=f"scheduled:{task.id}")
            self.stats["scheduled_fires"] += 1
        except Exception as e:
            logger.warning(f"Mind: scheduled fire dispatch failed: {e}")

    def notify_input(self, user_input: str):
        """Called when user sends a message — gives Mind context."""
        with self._ctx_lock:
            self._last_user_input = user_input
            self._conversation_history.append({"role": "user", "content": user_input})
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

    def notify_response(self, alice_response: str):
        """Called when Alice finishes responding — triggers post-processing."""
        with self._ctx_lock:
            self._last_alice_response = alice_response
            self._conversation_history.append({"role": "assistant", "content": alice_response})
            self._pending_post_process = True
            if len(self._conversation_history) > 20:
                self._conversation_history = self._conversation_history[-20:]

    def set_emotional_state(self, emotion: str):
        """Update Mind's awareness of Alice's emotional state."""
        with self._ctx_lock:
            self._emotional_state = emotion

    # ---- Background loop ----

    def _think_loop(self):
        """Main background loop — think, post-process, repeat."""
        import traceback as _tb
        while self._running:
            # Yield CPU while Alice is generating / TTS is playing
            if self._paused:
                time.sleep(0.1)
                continue

            try:
                with self._ctx_lock:
                    should_post_process = self._pending_post_process
                if should_post_process:
                    self._do_post_process()
                    with self._ctx_lock:
                        self._pending_post_process = False
                else:
                    self._do_think()
            except Exception as e:
                self.stats["failures"] += 1
                # Print to stderr so it's visible even if builtins.print is overridden
                import sys
                sys.stderr.write(f"[Mind thread error] {e}\n{_tb.format_exc()}\n")

            # Sleep between cycles
            time.sleep(self.think_interval)

    def _do_think(self):
        """Generate background thoughts."""
        # Snapshot context under lock
        with self._ctx_lock:
            if not self._conversation_history:
                return  # Nothing to think about yet
            recent = list(self._conversation_history[-6:])  # Last 3 exchanges
            emotional_state = self._emotional_state

        context = "Recent conversation:\n"
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Alice"
            context += f'{role}: "{msg["content"][:150]}"\n'
        context += f"\nAlice's current mood: {emotional_state}"

        prompt = self._build_chatml(MIND_SYSTEM_PROMPT, context, "What is Alice thinking right now?")
        raw = self._generate(prompt, max_tokens=120)

        if raw:
            parsed = parse_mind_output(raw)
            for thought in parsed.thoughts:
                self.proposals.add(thought, source="mind")
            self.stats["thoughts_generated"] += len(parsed.thoughts)

    def _do_post_process(self):
        """Post-process Alice's response — memory, avatar, IRIS."""
        # Snapshot context under lock
        with self._ctx_lock:
            last_user = self._last_user_input
            last_alice = self._last_alice_response
            emotional_state = self._emotional_state

        if not last_alice:
            return

        # Track which proposals Alice used
        self.proposals.mark_usage(last_alice)

        # Build context for post-processing
        context = f'User said: "{last_user[:200]}"\n'
        context += f'Alice responded: "{last_alice[:300]}"\n'
        context += f"Alice's mood: {emotional_state}"

        prompt = self._build_chatml(POST_PROCESS_PROMPT, context, "Analyze Alice's response.")
        raw = self._generate(prompt, max_tokens=200)
        logger.debug(f"Mind post-process raw: {raw!r}")

        if raw:
            parsed = parse_mind_output(raw)
            self._handle_post_process(parsed, last_user)
            self.stats["post_processes"] += 1

    def _handle_post_process(self, output: MindOutput, user_input: str):
        """Act on post-processing results."""
        # Update avatar intent
        if output.avatar_intent:
            try:
                from streaming.animation.emotion_state import set_emotion
                set_emotion(output.avatar_intent, 0.6)
            except Exception:
                pass
            # Persistent mood-with-cause: same emotion word, but it lingers
            # (decays over minutes) and carries WHY. EMOTION_TAG arousal sets
            # how strongly it starts; floor keeps even calm moods alive a bit.
            try:
                from .mood import update_mood
                arousal = 0.6
                if isinstance(output.emotion_tag, dict):
                    arousal = float(output.emotion_tag.get('arousal', 0.6) or 0.6)
                update_mood(
                    output.avatar_intent,
                    intensity=max(0.4, arousal),
                    cause=output.mood_cause or "",
                )
            except Exception:
                pass

        # Build emotion tag for memory tagging. Prefer Mind's structured
        # EMOTION_TAG (full vector); fall back to mapping AVATAR_INTENT
        # through the lookup table.
        from .emotion_tag import EmotionTag
        tag = EmotionTag.from_yaml_dict(output.emotion_tag) if output.emotion_tag else None
        if tag is None:
            tag = EmotionTag.from_avatar_intent(output.avatar_intent)
        if tag is not None:
            with self._ctx_lock:
                self._latest_emotion_tag = tag

        # Write memory candidates. Mind marks each one `about: self` (Alice's
        # own opinions/realizations — these grow her identity cartridge) or
        # `about: rin` (conversation memory, the old path).
        for candidate in output.memory_candidates:
            if candidate.get('confidence', 0) >= 0.6:
                try:
                    from alice.core.system.system_registry import get_registry
                    registry = get_registry()
                    mem = registry.get('memory')
                    if mem is None:
                        continue
                    if (str(candidate.get('about', '')).lower() == 'self'
                            and hasattr(mem, 'add_self_memory')):
                        # Lived identity caps at 0.7 importance — seeded canon
                        # (0.75+) stays on top; IRIS usage boosts can promote
                        # a take she keeps coming back to.
                        conf = float(candidate.get('confidence', 0.6))
                        mem.add_self_memory(
                            candidate.get('text', ''),
                            importance=min(0.7, 0.4 + 0.3 * conf),
                        )
                    elif hasattr(mem, 'add_conversation'):
                        mem.add_conversation(
                            user_message=user_input,
                            alice_response=candidate.get('text', ''),
                            drive_snapshot=tag,
                        )
                except Exception:
                    pass

        # Execute IRIS queries — Mind proposed them, so we actually run them
        # against memory and stash the top hit(s) as memory hints for next turn.
        # The raw "[IRIS] query" proposal is kept too for traceability.
        for query in output.iris_queries:
            self.proposals.add(f"[IRIS] {query}", source="mind:iris")
            self._run_iris_query(query)

        # Add reflective thoughts to buffer too
        for thought in output.thoughts:
            self.proposals.add(thought, source="mind:reflect")

    def _run_iris_query(self, query: str, top_k: int = 2):
        """
        Execute one of Mind's proposed IRIS queries and stash results as
        memory hints. Best-effort — any failure is swallowed; Mind's thinking
        loop must not stall on memory backend issues.
        """
        query = (query or "").strip()
        if not query:
            return
        try:
            from alice.core.system.system_registry import get_registry
            registry = get_registry()
            memory = registry.get('memory') if registry else None
            if memory is None or not hasattr(memory, 'search_memories'):
                return

            results = memory.search_memories(query, user_id="rin", k=top_k)
            for r in results or []:
                content = getattr(r, 'content', None)
                if content is None and hasattr(r, 'memory'):
                    content = getattr(r.memory, 'content', None)
                if not content:
                    continue
                mem_id = ""
                if hasattr(r, 'memory') and hasattr(r.memory, 'id'):
                    mem_id = r.memory.id or ""
                elif hasattr(r, 'id'):
                    mem_id = r.id or ""
                self.proposals.add_memory_hint(
                    content=content[:200],
                    query=query,
                    memory_id=mem_id,
                )
        except Exception as e:
            logger.debug(f"Mind IRIS query failed ({query!r}): {e}")

    # ---- Generation ----

    def _build_chatml(self, system: str, context: str, question: str) -> str:
        """Build ChatML prompt."""
        return (
            f"<|im_start|>system\n{system}<|im_end|>\n"
            f"<|im_start|>user\n{context}\n\n{question}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

    def _generate(self, prompt: str, max_tokens: int = 120) -> str:
        """Generate text on cuda:0 inside Mind's dedicated CUDA stream.

        Runs in true parallel with Alice (separate stream, separate model)
        even though both live on the same GPU. The python-level _gen_lock
        only guards against Mind issuing concurrent calls to itself.
        """
        if not self._initialized:
            return ""

        with self._gen_lock:
            start = time.perf_counter()
            try:
                import torch

                inputs = self._tokenizer(prompt, return_tensors="pt").to("cuda:0")
                input_len = inputs["input_ids"].shape[-1]

                with torch.cuda.stream(self._stream):
                    out_ids = self._model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                        pad_token_id=self._tokenizer.pad_token_id,
                    )

                response = self._tokenizer.decode(
                    out_ids[0][input_len:], skip_special_tokens=False
                )
                # Manual stop-string truncation (transformers doesn't take
                # string stops directly; cheaper than building a custom
                # StoppingCriteria for two short markers).
                for stop in ("<|im_end|>", "<|im_start|>"):
                    if stop in response:
                        response = response.split(stop)[0]
                response = response.strip()

                elapsed_ms = (time.perf_counter() - start) * 1000
                self.stats["total_think_ms"] += elapsed_ms
                total = self.stats["thoughts_generated"] + self.stats["post_processes"]
                if total > 0:
                    self.stats["avg_think_ms"] = self.stats["total_think_ms"] / total

                return response

            except Exception as e:
                logger.error(f"Mind generation error: {e}")
                self.stats["failures"] += 1
                return ""

    def get_stats(self) -> dict:
        """Return Mind stats + proposals stats."""
        return {
            **self.stats,
            "proposals": self.proposals.get_stats(),
            "initialized": self._initialized,
            "running": self._running,
        }

    def get_emotion_tag(self):
        """
        Return the latest EmotionTag from post-processing, or None if Mind
        hasn't scored a turn yet. Used by chat.py to feed memory tagging.
        Returns None on the first turn — memory.add_conversation handles
        None gracefully (no emotional_context attached).
        """
        with self._ctx_lock:
            return self._latest_emotion_tag

    # ---- Chat classifier (Twitch integration) ----

    def classify_chat_message(
        self,
        text: str,
        username: str = "",
        recent_context: Optional[List[str]] = None,
        timeout_ms: int = 100,
    ) -> Optional[int]:
        """
        Score a single Twitch chat message for Alice's attention (0-100, or None).

        Returns:
          - integer 0-100 priority (higher = should react sooner)
          - None if Mind isn't initialized or the model gave nonsense

        Score bands (Mind is prompted with these):
          0-10  ignore — small talk between viewers, irrelevant emote spam
          11-30 mildly interesting — could react if quiet
          31-60 engaging — chat asking a question, reacting to stream
          61-89 important — direct mention, genuinely good prompt
          90-100 must react — major event, hot moment

        Cost: one Mind generation (~15-50ms on GPU, ~200-500ms on CPU).
        Caller is expected to bypass this with a heuristic for cheers/mentions.
        """
        if not self._initialized:
            return None

        ctx_block = ""
        if recent_context:
            ctx_block = "Recent chat:\n" + "\n".join(f"- {line}" for line in recent_context[-3:]) + "\n\n"

        sys_prompt = (
            "You are filtering Twitch chat for an AI streamer named Alice. "
            "Decide how much this single chat message deserves Alice's attention. "
            "Output ONLY a single integer 0-100. No explanation.\n\n"
            "Bands:\n"
            "  0-10   ignore (small talk between viewers, lurker emotes, spam)\n"
            "  11-30  mildly interesting (could react if quiet)\n"
            "  31-60  engaging (question, real reaction to stream)\n"
            "  61-89  important (direct prompt to Alice, great joke)\n"
            "  90-100 must react (major event, hot moment)"
        )
        user_block = f"{ctx_block}Message from {username or 'viewer'}: {text!r}\n\nScore (single integer 0-100):"
        prompt = self._build_chatml(sys_prompt, "", user_block)

        raw = self._generate(prompt, max_tokens=8)
        if not raw:
            return None

        # Parse: take first integer-looking run in the output, clamp 0-100.
        digits = []
        for ch in raw.strip():
            if ch.isdigit():
                digits.append(ch)
            elif digits:
                break
        if not digits:
            return None
        try:
            n = int("".join(digits))
        except ValueError:
            return None
        return max(0, min(100, n))
