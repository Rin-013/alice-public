#!/usr/bin/env python3
"""
Clean Alice Chat - Zero debug output.
Uses CUDA + quantization on GPU.

Debug mode: ALICE_DEBUG=1 python chat.py
"""
import sys
import os
import multiprocessing

from alice.core.system.env import load_env
load_env()

# Reduce CUDA memory fragmentation when models + TTS share the GPU
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

# v4 Living Mind: enable with ALICE_MIND=1 (default on)
MIND_ENABLED = os.environ.get('ALICE_MIND', '1') != '0'
ANIMATION_ENABLED = os.environ.get('ALICE_ANIMATION', '1') != '0'

# Force UTF-8 on Windows (emoji in print statements crash cp1252)
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")

import warnings
import logging
import random
import threading
import asyncio
import time
import queue as _queue_mod

# Silence transformers verbosity BEFORE any model import. set_verbosity_error()
# alone is too late — transformers checks the env var at import time, so loaders
# (STT model, EmotionBERT, sentence-transformers) print "Some weights of X were not
# initialized…" / "attention_mask is not set…" / etc. before user code runs.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
import transformers as _transformers
_transformers.logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# FILE LOGGING — everything goes to logs/chat.log, nothing spams the console
# ============================================================================
_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(_LOG_DIR, exist_ok=True)
_LOG_PATH = os.path.join(_LOG_DIR, "chat.log")

# Set up root logger to file
_file_handler = logging.FileHandler(_LOG_PATH, encoding="utf-8", mode="w")  # overwrite each session
_file_handler.setFormatter(logging.Formatter(
    "%(asctime)s.%(msecs)03d [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
))
_file_handler.setLevel(logging.DEBUG)

_chat_logger = logging.getLogger("alice")
_chat_logger.setLevel(logging.DEBUG)
_chat_logger.addHandler(_file_handler)
_chat_logger.propagate = False

# Also capture root logger (catches transformers, torch, etc.)
logging.root.addHandler(_file_handler)
logging.root.setLevel(logging.DEBUG)

def log(msg, level=logging.INFO):
    """Log to file. Use this throughout chat.py."""
    _chat_logger.log(level, msg)

# Debug mode - set ALICE_DEBUG=1 to see prompts
DEBUG_MODE = os.environ.get('ALICE_DEBUG', '0') == '1'
TTS_ENABLED = os.environ.get('ALICE_TTS', '1') != '0'
STT_ENABLED = os.environ.get('ALICE_STT', '1') != '0'
TWITCH_ENABLED = os.environ.get('ALICE_TWITCH', '1') != '0'
AUTONOMOUS_ENABLED = os.environ.get('ALICE_AUTONOMOUS', '0') == '1'
GROWTH_ENABLED = os.environ.get('ALICE_GROWTH', '0') == '1'
IDLE_TURN_SECONDS = float(os.environ.get('ALICE_IDLE_TURN_SECONDS', '60'))


GENERATION_CONFIG = {
    "max_tokens": 50,            # Concise conversational responses
    "temp_min": 0.65,            # Temperature range (wavering)
    "temp_max": 0.75,            # Higher to let sass come through
    "top_p": 0.92,               # Wider vocabulary for personality
    "min_p": 0.04,               # Filter unlikely tokens (reduces nonsense)
    "top_k": 50,                 # Slightly wider for variety
}

# Conversation history cap (in user+assistant pairs). Without this,
# `conversation_history` grew unbounded — by turn 20 the prompt carried
# 40+ messages, blowing up token count linearly per session. 6 pairs
# (12 msgs) keeps short-term coherence without late-session blow-up.
# Override with ALICE_HISTORY_PAIRS env var.
HISTORY_CAP_PAIRS = int(os.environ.get("ALICE_HISTORY_PAIRS", "6"))

# Direction-tag injection: every training example steers Alice with
# <alice_direction>...</alice_direction> prepended to the user message;
# without it she's out-of-distribution and collapses to her loudest
# register. ALICE_DIRECTION=0 disables for A/B comparison.
DIRECTION_TAGS = os.environ.get("ALICE_DIRECTION", "1") != "0"

# Fairy input guard: run every incoming message through the injection
# pattern library (fairy.protect is_input=True) before it touches Mind,
# memory, or the prompt. Blocked turns get Alice's canned comeback and
# never enter history. ALICE_INPUT_GUARD=0 disables.
INPUT_GUARD = os.environ.get("ALICE_INPUT_GUARD", "1") != "0"

# Speculative voice turns (docs/plans/VOICE_LATENCY.md Phase 2): start
# get_response at the STT soft endpoint (~224ms of silence) instead of the
# hard endpoint (~384ms), overlapping prefill + early decode with the rest
# of the endpoint wait. Console output and TTS stay gated until the hard
# endpoint confirms; if Rin keeps talking the turn is aborted with zero
# side effects. Default OFF until live-tested with a mic.
SPECULATIVE = os.environ.get("ALICE_STT_SPECULATIVE", "0") == "1"
# ============================================================================

# (Removed EMOTION_TO_MOTION — emotions now come from AliceEmotionalState,
#  not post-hoc EmotionBERT classification on Alice's output.)

# --- Module-level state (set during init, used by get_response) ---
model = None
tokenizer = None
mind = None  # v4 Mind (CPU background thinker)
fairy = None
memory = None
script_integration = None
tts = None
speech = None  # SpeechPipeline wrapping `tts` — fire-and-forget speech queue
stt = None
twitch_runtime = None
_voice_queue = _queue_mod.Queue()
_real_print = print
_allow_print = True
conversation_history = []

# One-shot signal — set when fairy filtered Alice's last response.
# Consumed (and cleared) on the NEXT turn's prompt build so Alice
# can riff on the filter event. Format: TOS category string or None.
_last_filter_event = None
# Tool-call results from Alice's previous turn. Injected into NEXT turn's
# system prompt so she can react. Cleared after injection.
_pending_tool_results: list = []
USER_ID = "rin"
_last_exchange = {"user": None, "alice": None}
# perf_counter of Rin's last spoken word for the current turn (None for
# keyboard/Twitch/autonomous turns). Set by _get_input_with_voice from
# stt.last_speech_end_ts; read at first TTS submit to log VOICE_E2E —
# the true last-word→first-audio latency we're driving toward 400ms.
_voice_turn_speech_end = None


class _SpecCtrl:
    """Gate + abort pair for one speculative turn (ALICE_STT_SPECULATIVE).

    get_response holds all externally visible output (console prints, TTS
    submits) on `wait_visible()` until either `gate` (hard endpoint confirmed
    the utterance) or `abort` (Rin kept talking / input superseded) is set.
    In practice the gate resolves ~120ms after the turn starts, long before
    the first token decodes — the wait almost never actually blocks.
    """

    def __init__(self):
        self.gate = threading.Event()
        self.abort = threading.Event()
        self._resolved = threading.Event()  # set when either of the above is

    def confirm(self):
        self.gate.set()
        self._resolved.set()

    def cancel(self):
        self.abort.set()
        self._resolved.set()

    def wait_visible(self, timeout: float = 30.0) -> bool:
        """True = confirmed, speak/print freely. False = aborted, discard.
        A resolution timeout is treated as abort — better a silently dropped
        turn than a thread wedged holding the model."""
        if not self._resolved.wait(timeout=timeout):
            log("SPEC turn gate timed out — treating as abort", logging.WARNING)
            self.abort.set()
        return self.gate.is_set() and not self.abort.is_set()


class _SpecManager:
    """Owns the (at most one) in-flight speculative turn.

    STT thread:  start(text) at soft endpoint, cancel() if speech resumes.
    Main thread: try_adopt(text) when the hard endpoint delivers — matching
    text opens the gate and returns the turn to join; mismatch cancels.
    wait_idle() joins cancelled threads before a normal turn starts so two
    generate() calls never overlap on the GPU.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._turn = None          # {"text", "ctrl", "thread"}
        self._zombies = []         # cancelled turns not yet joined

    def start(self, text: str) -> None:
        ctrl = _SpecCtrl()
        turn = {"text": text, "ctrl": ctrl}

        def _run():
            try:
                asyncio.run(get_response(text, spec=ctrl))
            except Exception:
                import traceback
                log(f"SPEC turn crashed:\n{traceback.format_exc()}", logging.ERROR)

        th = threading.Thread(target=_run, daemon=True, name="SpecTurn")
        turn["thread"] = th
        with self._lock:
            if self._turn is not None:  # shouldn't happen — defensive
                self._cancel_locked()
            self._turn = turn
        log(f"SPEC start (soft endpoint): {text!r}")
        th.start()

    def cancel(self) -> None:
        """Called from the STT thread when speech resumes — must not block."""
        with self._lock:
            self._cancel_locked()

    def _cancel_locked(self) -> None:
        if self._turn is None:
            return
        self._turn["ctrl"].cancel()
        self._zombies.append(self._turn)
        log("SPEC cancel — speculative turn discarded")
        self._turn = None

    def try_adopt(self, text: str):
        """Main thread, at hard endpoint. Returns the in-flight turn (gate
        opened, caller joins it) if `text` matches; else None and any
        in-flight turn is cancelled."""
        with self._lock:
            turn = self._turn
            if turn is None:
                return None
            if turn["text"] != text or turn["ctrl"].abort.is_set():
                self._cancel_locked()
                return None
            self._turn = None  # ownership moves to the caller
        turn["ctrl"].confirm()
        log("SPEC confirm — adopting in-flight turn")
        return turn

    def wait_idle(self, timeout: float = 5.0) -> None:
        """Join cancelled turns so their generate() has fully unwound before
        a new turn touches the model. Abort stops generation within ~1 token,
        so this should never block measurably."""
        with self._lock:
            zombies, self._zombies = self._zombies, []
        for z in zombies:
            z["thread"].join(timeout=timeout)
            if z["thread"].is_alive():
                log("SPEC zombie thread still alive after join timeout", logging.WARNING)


_spec = _SpecManager() if SPECULATIVE else None


class SuppressOutput:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')
    def __exit__(self, *args):
        sys.stdout.close()
        sys.stderr.close()
        sys.stdout = self._stdout
        sys.stderr = self._stderr


def _controlled_print(*args, **kwargs):
    if _allow_print:
        _real_print(*args, **kwargs)


def safe_print(*args, **kwargs):
    _real_print(*args, **kwargs)


def get_temperature():
    """Get temperature that wavers within configured range"""
    return random.uniform(GENERATION_CONFIG["temp_min"], GENERATION_CONFIG["temp_max"])


# Markdown emphasis (*word*, **word**, _word_) survives Alice's training and
# leaks into TTS, where the talker either reads "asterisk" literally or chokes.
# Stage directions in asterisks (*sigh*, *checks notes*) should also be silent.
# Strip those + any other char that the talker can't pronounce. Console output
# keeps the original — we only sanitize the TTS feed.
import re as _re_tts
_TTS_EMPHASIS_RE = _re_tts.compile(r"\*+([^*]+)\*+")  # *word* or **word**
_TTS_UNDERSCORE_EMPHASIS_RE = _re_tts.compile(r"(?<!\w)_([^_]+)_(?!\w)")  # _word_ but not snake_case
_TTS_STRIP_CHARS_RE = _re_tts.compile(r"[*_`~|<>{}\[\]\\]")  # leftover markdown / structural
_TTS_EMOJI_RE = _re_tts.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental
    "\U00002600-\U000026FF"  # misc symbols
    "]+",
    flags=_re_tts.UNICODE,
)


def _sanitize_for_tts(text: str) -> str:
    """Strip markdown emphasis, structural chars, and emoji before TTS."""
    if not text:
        return text
    # *word* / **word** → word (keep the inner content, drop the asterisks)
    text = _TTS_EMPHASIS_RE.sub(r"\1", text)
    text = _TTS_UNDERSCORE_EMPHASIS_RE.sub(r"\1", text)
    # Any leftover structural chars
    text = _TTS_STRIP_CHARS_RE.sub("", text)
    # Emoji
    text = _TTS_EMOJI_RE.sub("", text)
    return text


async def get_response(user_input: str, spec: "_SpecCtrl | None" = None) -> str:
    """Run one full turn. `spec` (ALICE_STT_SPECULATIVE) marks a speculative
    turn started at the STT soft endpoint: all console output and TTS submits
    block on spec.wait_visible() until the hard endpoint confirms (gate) or
    Rin keeps talking (abort). An aborted turn discards everything — no
    history, no memory, no tools, no audio."""
    global conversation_history, _last_exchange
    t_turn_start = time.perf_counter()

    log(f"=== TURN START{' (SPEC)' if spec else ''}: {user_input!r} ===")

    # Fairy input guard — injection/jailbreak/extraction check BEFORE the
    # input reaches Mind, memory search, or the prompt. A blocked turn
    # returns Alice's in-character comeback and is never written to
    # history, so the attack text can't poison later context either.
    if INPUT_GUARD and fairy:
        guard_result = fairy.protect(user_input, is_input=True)
        if guard_result.blocked:
            log(f"FAIRY BLOCKED INPUT — reason: {guard_result.reason}, "
                f"threat_level: {guard_result.threat_level:.2f}")
            if spec is not None and not spec.wait_visible():
                return ""
            comeback = guard_result.safe_response or "Nice try."
            # Caller already printed the "Alice: " prefix before calling us.
            safe_print(comeback, end="", flush=True)
            if speech:
                speech.submit(_sanitize_for_tts(comeback), "neutral")
            return comeback

    # IRIS recall telemetry: open a turn buffer. Thread-local so search.py
    # can append candidates without plumbing turn_id through every signature.
    from alice.core.memory import telemetry
    turn_id = telemetry.start_turn(USER_ID, user_input)
    log(f"Telemetry turn opened: {turn_id}")

    # v4 Mind: notify of new input (Mind thinks continuously in background)
    if mind:
        mind.notify_input(user_input)
        log(f"Mind notified of input, proposals: {mind.proposals.get_stats()}")

    # Emotional state — EmotionBERT + state-machine path removed. Mind now
    # owns emotion: it emits AVATAR_INTENT in its post-process YAML, and
    # mind.py:_handle_post_process pushes that directly to streaming/animation.
    # `alice_emotion` here is a placeholder for downstream callers (TTS
    # ignores it, animation is driven by Mind).
    alice_emotion = "neutral"
    alice_intensity = 0.0

    # Step 1: Gather context
    context_data = {}

    # Memory search
    if memory:
        try:
            t0 = time.perf_counter()
            import io, contextlib
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                relevant = memory.search_memories(user_input, USER_ID, k=5)
            if relevant:
                context_data["memory_context"] = {"relevant_memories": relevant}
            log(f"Memory search: {len(relevant) if relevant else 0} results in {(time.perf_counter()-t0)*1000:.0f}ms")
        except Exception as e:
            log(f"Memory search failed: {e}", logging.WARNING)

    # Step 2: Integrations (emotion_integration removed — Mind drives emotion)
    t0 = time.perf_counter()
    await script_integration.process_pre_llm(user_input, USER_ID, context_data)
    system_prompt = context_data.get("script_prompt", "")
    log(f"Integrations done in {(time.perf_counter()-t0)*1000:.0f}ms, prompt={len(system_prompt)} chars")

    if not system_prompt:
        log("WARNING: empty system prompt, returning '...'", logging.WARNING)
        return "..."

    # Output style preamble removed — was driven by EmotionBERT-derived
    # alice_emotion. With Mind owning emotion, output style is governed
    # by Alice's text and the personality block in the template.

    # v4 Mind: inject proposals. Capped low (n=2) — buffer holds 20 with
    # TTL=120s, but the prompt only needs the freshest 2 thoughts. Larger
    # values bloated the prompt without proportionate quality benefit.
    if mind:
        proposals_text = mind.proposals.get_context_string(n=2)
        if proposals_text:
            system_prompt = system_prompt.rstrip() + "\n\n" + proposals_text
            log(f"Mind proposals injected: {proposals_text!r}")

        # P4: memory hints — facts Mind pre-fetched from IRIS via its own
        # curiosity. Rendered as "You recalled: …" so Alice sees them as
        # her own recollection rather than Mind's suggestion.
        hints_text = mind.proposals.get_memory_hints_string(n=2)
        if hints_text:
            system_prompt = system_prompt.rstrip() + "\n\n" + hints_text
            log(f"Mind memory hints injected: {hints_text!r}")

    # Tool results from previous turn — Alice fired a tool, the result
    # came back. Inject as bracketed lines so she can react in her voice.
    global _pending_tool_results
    if _pending_tool_results:
        result_lines = [r.to_alice_context() for r in _pending_tool_results]
        result_block = "Tool results from your last turn:\n" + "\n".join(result_lines)
        system_prompt = system_prompt.rstrip() + "\n\n" + result_block
        log(f"Tool results injected: {len(_pending_tool_results)}")
        _pending_tool_results = []

    # If fairy filtered Alice's last response, inject a one-shot note
    # so she knows what happened. The bit: viewers find it funny when
    # she gets filtered, so let her riff on it instead of pretending
    # nothing happened.
    global _last_filter_event
    if _last_filter_event:
        filter_note = (
            f"[FAIRY FILTERED YOU LAST TURN — category: {_last_filter_event}]\n"
            f"You tried to say something that fairy cut off mid-sentence "
            f"(viewers heard 'filter' and then silence). You can react to "
            f"this naturally — chat thinks it's funny when you get filtered. "
            f"Don't repeat the violating thing, just acknowledge the bit if "
            f"it feels right."
        )
        system_prompt = system_prompt.rstrip() + "\n\n" + filter_note
        log(f"Fairy filter event injected: category={_last_filter_event}")
        _last_filter_event = None  # one-shot — consume

    # Build ChatML. The current user turn gets the direction tag (matching
    # training format); history stays untagged, and the clean user_input
    # is what lands in history + memory below.
    tagged_input = user_input
    if DIRECTION_TAGS:
        try:
            from alice.core.mind.direction import wrap_user_input
            tagged_input = wrap_user_input(user_input)
            log(f"Direction tag: {tagged_input.split(chr(10))[0]}")
        except Exception as e:
            log(f"Direction tag failed: {e}", logging.WARNING)

    chatml = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    for msg in conversation_history:
        chatml += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    chatml += f"<|im_start|>user\n{tagged_input}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n"

    log(f"ChatML built: {len(chatml)} chars, {len(conversation_history)} history msgs")

    if DEBUG_MODE:
        safe_print("\n" + "="*50)
        safe_print("SYSTEM PROMPT:")
        safe_print("-"*50)
        safe_print(system_prompt)
        safe_print("="*50 + "\n")

    # Pause Mind during Alice generation. Both on cuda:0 (separate streams),
    # but a single GPU's SMs serialize when both saturate them — measured
    # concurrent slower than sequential. See parallel_inference_dead_end memory.
    if mind:
        mind.pause()
        log("Mind paused")

    # Tokenize + generate
    temp = get_temperature()
    t0 = time.perf_counter()
    input_ids = tokenizer(chatml, return_tensors="pt").input_ids.to(model.device)
    log(f"Tokenized: {input_ids.shape[1]} tokens, temp={temp:.3f}, tokenize={( time.perf_counter()-t0)*1000:.0f}ms")
    # Per-turn prompt-size telemetry — used to baseline + verify the
    # prompt-bloat reduction work (plans/polished-kindling-marble.md).
    # Remove or downgrade to DEBUG once that work lands and is verified.
    log(f"PROMPT_SIZE: system_prompt={len(system_prompt)}ch, "
        f"history_msgs={len(conversation_history)}, "
        f"chatml={len(chatml)}ch, tokens={input_ids.shape[1]}")

    from transformers import TextIteratorStreamer
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    # KV cache compression (default on, ALICE_TURBOQUANT=0 to disable)
    # triton_mode=True skips decompression — attention reads compressed data
    # directly via Triton kernels for attention speedup
    from alice.core.optimization.turboquant_cache import create_cache
    tq_triton = getattr(model.config, '_attn_implementation', None) == 'turboquant'
    tq_cache = create_cache(model.config, triton_mode=tq_triton)

    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=GENERATION_CONFIG["max_tokens"],
        temperature=temp,
        top_p=GENERATION_CONFIG["top_p"],
        min_p=GENERATION_CONFIG["min_p"],
        top_k=GENERATION_CONFIG["top_k"],
        do_sample=True,
        streamer=streamer,
        use_cache=True,
        past_key_values=tq_cache,
    )

    # Speculative turn: let an abort (Rin kept talking) stop generation
    # within one decode step instead of running max_tokens to completion.
    if spec is not None:
        from transformers import StoppingCriteria, StoppingCriteriaList

        class _AbortCriteria(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return spec.abort.is_set()

        generate_kwargs["stopping_criteria"] = StoppingCriteriaList([_AbortCriteria()])

    t_gen_start = time.perf_counter()

    # Run LLM on its own CUDA stream to isolate from TTS/STT compiled kernels
    import torch as _torch
    from alice.core.cuda_streams import get_stream
    _llm_stream = get_stream("llm")

    @_torch._dynamo.disable
    def _generate_on_stream(**kwargs):
        # Suppress per-turn transformers warnings ("attention_mask is not set",
        # "Both `max_new_tokens` and `max_length` set", custom logits processor,
        # etc.) — they fire on every generate() and aren't actionable.
        with _torch.cuda.stream(_llm_stream), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.generate(**kwargs)
        # Record event so TTS can wait on LLM completion without global sync
        _llm_done_event.record(_llm_stream)

    _llm_done_event = _torch.cuda.Event()
    gen_thread = threading.Thread(target=_generate_on_stream, kwargs=generate_kwargs)
    gen_thread.start()
    log("LLM generation started (dedicated CUDA stream)")

    response = ""
    stream_filter = fairy.create_streaming_filter()
    _in_think_block = False
    token_count = 0

    # Tool-call parser sits BEFORE fairy: pulls <tool_call>{...}</tool_call>
    # blocks out so the JSON never reaches TTS or chat. Only present when
    # Twitch is connected — no point parsing if there's no dispatcher.
    tool_parser = None
    pending_tool_calls: list = []
    if twitch_runtime and twitch_runtime.connected:
        from streaming.twitch.tool_call_parser import StreamingToolCallParser
        tool_parser = StreamingToolCallParser()

    # Clause chunker (Path B) — accumulates visible tokens, emits at clause
    # boundaries per a growing schedule (60 → 140 → 240 chars). Each emit
    # becomes one SpeechPipeline.submit so audio starts mid-LLM-generation.
    # Tradeoff: every chunk gets its own end-of-utterance cadence from
    # TTS model; growing schedule + boundary-only cuts minimize audible
    # seams. The earlier "one-shot full response" approach gave the cleanest
    # prosody but paid full LLM gen time as silence.
    from alice.core.voice.clause_chunker import ClauseChunker
    # ALICE_TTS_SCHEDULE="60,140,240" overrides the chunk-length schedule —
    # widen (e.g. "140,240,400") to trade TTFA for fewer chunk seams.
    _sched_env = os.environ.get("ALICE_TTS_SCHEDULE", "").strip()
    if speech and _sched_env:
        chunker = ClauseChunker(schedule=tuple(int(x) for x in _sched_env.split(",")))
    else:
        # First chunk 60→40 chars (2026-06-11 latency pass). Bench showed
        # typical responses are 28-67 chars — at 60 the first chunk usually
        # never fired mid-stream and audio waited for FULL generation. At 40
        # (~12 tokens ≈ 650ms at 18 tok/s) most turns start speaking early.
        # Cost: one more early prosody seam — ear-veto via
        # ALICE_TTS_SCHEDULE=60,140,240 if it sounds choppy.
        chunker = ClauseChunker(schedule=(40, 140, 240, 240)) if speech else None
    t_tts_submit: "float | None" = None

    def _submit_chunk(chunk: str) -> None:
        nonlocal t_tts_submit
        if not chunk or not speech:
            return
        if spec is not None and not spec.wait_visible():
            return  # aborted speculative turn — nothing may reach the speakers
        tts_chunk = _sanitize_for_tts(chunk)
        if not tts_chunk:
            return
        if t_tts_submit is None:
            t_tts_submit = time.perf_counter()
            log(f"TTS first-chunk submit ({len(tts_chunk)} chars): {tts_chunk[:60]!r}")
            if _voice_turn_speech_end is not None:
                # First audio follows this by the TTS first-chunk gen time
                # (~200-300ms) — see 'first chunk at' in the subprocess log.
                log(f"VOICE_E2E: last-word→first-TTS-submit "
                    f"{(t_tts_submit - _voice_turn_speech_end)*1000:.0f}ms")
        speech.submit(tts_chunk, alice_emotion)

    def _emit(text: str):
        """Run text through fairy, print to console, and feed the clause
        chunker. Boundary-aligned chunks are submitted to TTS immediately
        so audio starts mid-generation (Path B)."""
        if not text:
            return ""
        if spec is not None and not spec.wait_visible():
            return ""  # aborted speculative turn — print nothing
        filtered = stream_filter(text)
        safe_print(filtered, end="", flush=True)
        if chunker and filtered:
            for ready in chunker.feed(filtered):
                _submit_chunk(ready)
        return filtered

    # Stream tokens
    t_first_token: "float | None" = None
    for token_text in streamer:
        token_count += 1
        if t_first_token is None:
            t_first_token = time.perf_counter()
            log(f"LLM first token: {(t_first_token - t_gen_start)*1000:.0f}ms "
                f"after gen start (prefill + 1 decode step)")

        if "<|im_end|>" in token_text or "<|im_start|>" in token_text:
            token_text = token_text.split("<|im_end|>")[0].split("<|im_start|>")[0]
            if token_text and not _in_think_block:
                if tool_parser:
                    visible, calls = tool_parser.feed(token_text)
                    pending_tool_calls.extend(calls)
                else:
                    visible = token_text
                response += _emit(visible)
            break

        if "<think>" in token_text:
            _in_think_block = True
            log(f"Think block started at token {token_count}")
            continue
        if "</think>" in token_text:
            _in_think_block = False
            log(f"Think block ended at token {token_count}")
            continue
        if _in_think_block:
            continue

        if tool_parser:
            visible, calls = tool_parser.feed(token_text)
            pending_tool_calls.extend(calls)
        else:
            visible = token_text
        response += _emit(visible)

        if "<|im_end|>" in response or "<|im_start|>" in response:
            response = response.split("<|im_end|>")[0].split("<|im_start|>")[0]
            log("Backup end-token check triggered")
            break

    gen_thread.join()

    # Aborted speculative turn: Rin kept talking past the soft endpoint.
    # Nothing was printed or spoken (gate never opened); leave no trace —
    # no history, no memory, no tools, no Twitch. The fuller utterance
    # will arrive as a fresh turn at the next endpoint.
    if spec is not None and spec.abort.is_set():
        log(f"SPEC turn aborted after {token_count} tokens — discarded")
        if mind:
            mind.resume()
        return ""

    # Flush parser's held buffer first — might have visible tail or unclosed tag.
    if tool_parser:
        held_visible, more_calls = tool_parser.flush()
        pending_tool_calls.extend(more_calls)
        if held_visible:
            response += _emit(held_visible)

    # Flush any tokens still held in the streaming filter's holdback
    # buffer (default 3-token delay). Without this, the last 3 tokens
    # of every response would be silently dropped — including from TTS.
    final_tail = stream_filter.flush()
    if final_tail:
        safe_print(final_tail, end="", flush=True)
        response += final_tail
        if chunker:
            for ready in chunker.feed(final_tail):
                _submit_chunk(ready)

    t_gen_end = time.perf_counter()
    safe_print()

    response = response.strip()
    log(f"LLM done: {token_count} tokens, {len(response)} chars, {(t_gen_end-t_gen_start)*1000:.0f}ms")
    if t_first_token is not None and token_count > 1 and t_gen_end > t_first_token:
        log(f"LLM decode rate: {(token_count - 1) / (t_gen_end - t_first_token):.1f} tok/s "
            f"({token_count - 1} tokens after first)")
    log(f"Response: {response!r}")

    # Flush whatever's still buffered in the clause chunker — partial last
    # sentence with no terminator, or the whole response if it was too short
    # to ever cross a schedule threshold. This is the final speech submit
    # for the turn; t_tts_submit may already be set from a mid-stream chunk.
    if chunker:
        _submit_chunk(chunker.flush())

    # Dispatch any tool calls Alice emitted, stash results for next turn.
    if pending_tool_calls and twitch_runtime:
        from streaming.twitch.dispatcher import ToolResult, ToolStatus
        for call in pending_tool_calls:
            if call.parse_error:
                _pending_tool_results.append(ToolResult(
                    tool_name=call.tool or "?",
                    status=ToolStatus.VALIDATION_ERROR,
                    message=f"your tool call had a problem: {call.parse_error}",
                    error=call.parse_error,
                ))
                log(f"Tool call parse error: {call.parse_error}", logging.WARNING)
                continue
            t0 = time.perf_counter()
            result = twitch_runtime.dispatch_tool(call.tool, call.args)
            log(f"Tool {call.tool} → {result.status.value} ({(time.perf_counter()-t0)*1000:.0f}ms): {result.message}")
            _pending_tool_results.append(result)

    # Send Alice's final speech to Twitch chat (visible text only — tool calls
    # were already pulled out by the parser).
    if twitch_runtime and twitch_runtime.connected and response:
        twitch_runtime.send_chat(response)

    # Record fairy filter event for the next turn's system prompt.
    # See _last_filter_event injection in the prompt-build section.
    tripped_this_turn = bool(getattr(stream_filter, "was_filtered", False))
    if tripped_this_turn:
        _last_filter_event = stream_filter.violation_category
        log(f"FAIRY FILTERED Alice — category: {stream_filter.violation_category}")

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response})
    # Cap to last HISTORY_CAP_PAIRS pairs (= 2 * pairs messages).
    if len(conversation_history) > HISTORY_CAP_PAIRS * 2:
        conversation_history = conversation_history[-(HISTORY_CAP_PAIRS * 2):]

    if memory:
        try:
            t0 = time.perf_counter()
            # Mind's last post-process produced an emotion tag (one-turn lag,
            # since post-process runs async after the previous turn). Pass it
            # so the memory is tagged with the affect Alice carried into this
            # moment. None is fine — memory handles untagged writes.
            tag = mind.get_emotion_tag() if mind else None
            memory.add_conversation(
                user_message=user_input,
                alice_response=response,
                drive_snapshot=tag,
            )
            log(f"Memory saved in {(time.perf_counter()-t0)*1000:.0f}ms")
        except Exception as e:
            log(f"Memory save failed: {e}", logging.WARNING)

    # Free fragmented CUDA blocks. Runs in parallel with TTS generation now
    # (subprocess has its own CUDA context, this only touches Alice's main
    # process allocator). No longer on the critical path before audio.
    import torch as _torch_gc
    _torch_gc.cuda.empty_cache()

    # IRIS feedback: boost memories Alice actually referenced.
    # Phase 4: cosine-based usefulness is authoritative. Word-overlap is logged
    # alongside for one week so we can diff before/after on the same turns.
    used_ids: list[str] = []
    usefulness_scores: dict[str, float] = {}
    overlap_scores: dict[str, float] = {}
    if memory and context_data.get("memory_context"):
        stop_words = {"the","a","an","is","was","are","were","be","been","being","have","has","had",
                       "do","does","did","will","would","could","should","can","may","might","shall",
                       "i","you","he","she","it","we","they","me","him","her","us","them","my","your",
                       "his","its","our","their","this","that","and","but","or","not","no","so","if",
                       "in","on","at","to","for","of","with","by","from","as","into","about","than"}
        retrieved = context_data["memory_context"].get("relevant_memories", [])
        response_words = set(response.lower().split())
        from alice.core.memory import usefulness as _use

        boosted = 0
        for result in retrieved:
            mid = result.memory.id
            content = result.memory.content or ""

            # Legacy word-overlap — logged for A/B comparison, not used for routing.
            mem_words = set(content.lower().split()) - stop_words
            overlap = (len(mem_words & response_words) / len(mem_words)) if mem_words else 0.0
            overlap_scores[mid] = round(overlap, 4)

            # Cosine usefulness — the real signal. None means "couldn't score"
            # (short response, empty content); treat as unknown, don't update.
            cos_score = _use.score_usefulness(content, response)
            if cos_score is not None:
                usefulness_scores[mid] = cos_score
                try:
                    lt = getattr(memory, 'long_term', None)
                    if lt is not None and hasattr(lt, 'record_usefulness'):
                        lt.record_usefulness(mid, cos_score)
                except Exception as e:
                    log(f"usefulness EMA update failed for {mid}: {e}", logging.WARNING)

                if cos_score >= _use.COSINE_USED_THRESHOLD:
                    used_ids.append(mid)
                    try:
                        memory.record_memory_used(mid)
                        boosted += 1
                    except Exception as e:
                        log(f"IRIS boost failed for {mid}: {e}", logging.WARNING)
        if boosted:
            log(f"IRIS feedback: boosted {boosted}/{len(retrieved)} retrieved memories")

    # Telemetry: stash post-response signals and flush the turn buffer.
    # Fails open — a broken log sink never affects the turn.
    try:
        telemetry.record_usage(
            used=used_ids,
            usefulness=usefulness_scores,
            word_overlap=overlap_scores,
            response_len=len(response),
        )

        # Bandit reward update: use mean usefulness across this turn's picked
        # memories as the reward signal for the arm the bandit picked. Skip
        # very short responses (<30 chars) — too noisy to label. No picks
        # this turn means no signal to attribute, also skip.
        #
        # Reads arm+features from the bandit's OWN thread-local (not
        # telemetry's) so learning still works with ALICE_TELEMETRY=0.
        try:
            from alice.core.memory.iris import strategy_bandit as _sb
            # Always consume — stale pending must not cross turns even if
            # someone flipped the flag mid-session.
            pending = _sb.consume_pending()
            if (
                _sb.is_enabled()
                and pending is not None
                and len(response) >= 30
                and usefulness_scores
            ):
                reward = sum(usefulness_scores.values()) / max(1, len(usefulness_scores))
                _sb.get_bandit().update_reward(
                    arm_id=int(pending["arm_id"]),
                    features=list(pending["features"]),
                    reward=float(reward),
                )
        except Exception as e:
            log(f"Bandit reward update failed: {e}", logging.DEBUG)

        telemetry.finalize_turn(
            latency_ms=(time.perf_counter() - t_turn_start) * 1000.0
        )
    except Exception as e:
        log(f"Telemetry finalize failed: {e}", logging.WARNING)

    if GROWTH_ENABLED:
        try:
            from alice.core.growth import capture_experience
            from alice.core.mind.mood import get_mood
            emotion_tag = mind.get_emotion_tag() if mind else None
            mood_state = get_mood()
            capture_experience(
                user_input, response,
                emotion_tag=emotion_tag,
                mood_state=mood_state,
                usefulness_scores=usefulness_scores,
                context_messages=conversation_history[-10:],
                memory_context=context_data.get("memory_context"),
            )
        except Exception as e:
            log(f"Growth capture failed: {e}", logging.DEBUG)

    # Post-response emotion update removed — was running EmotionBERT on
    # Alice's own output and pushing to animation. Mind now owns this:
    # mind.py:_handle_post_process reads AVATAR_INTENT from its YAML
    # output and calls streaming.animation.set_emotion() directly.

    # End-of-turn sync: wait for TTS generation + playback to finish so the
    # next turn doesn't start mid-utterance AND Mind doesn't resume onto
    # cuda:0 while TTS is still generating (single-GPU SM contention —
    # see parallel_inference_dead_end memory). Everything above this point
    # ran in parallel with TTS generation.
    if speech and t_tts_submit is not None:
        speech.wait_playback_done()
        log(f"TTS end-to-end (submit→drained): {(time.perf_counter()-t_tts_submit)*1000:.0f}ms")

    # v4 Mind: notify + resume.
    if mind:
        mind.set_emotional_state(alice_emotion)
        mind.notify_response(response)
        mind.resume()
        log(f"Mind resumed, stats={mind.get_stats()}")

    _last_exchange["user"] = user_input
    _last_exchange["alice"] = response

    log(f"=== TURN END: {(time.perf_counter()-t_turn_start)*1000:.0f}ms total ===")
    return response


def _get_input_with_voice(prompt: str) -> str:
    """
    Get next input from keyboard / voice / Twitch chat / autonomous tick.
    Rin wins (kb + voice). Twitch chat is checked when Rin is silent.
    Autonomous turn fires after IDLE_TURN_SECONDS of total silence.

    Fast path: if no STT, no Twitch, no autonomy, just block on input().
    """
    global _voice_turn_speech_end
    _voice_turn_speech_end = None  # only voice turns carry a speech-end stamp

    if not stt and not twitch_runtime and not AUTONOMOUS_ENABLED:
        return input(prompt).strip()

    kb_queue = _queue_mod.Queue()

    def _kb_reader():
        try:
            text = input(prompt)
            kb_queue.put(text.strip())
        except (EOFError, KeyboardInterrupt):
            kb_queue.put(None)

    kb_thread = threading.Thread(target=_kb_reader, daemon=True)
    kb_thread.start()

    started_at = time.time()

    while True:
        # 1. Keyboard (Rin)
        try:
            kb_text = kb_queue.get_nowait()
            if kb_text is None:
                raise EOFError
            if kb_text:
                return kb_text
        except _queue_mod.Empty:
            pass

        # 2. Voice (Rin)
        if stt:
            try:
                voice_text = _voice_queue.get_nowait()
                if voice_text:
                    _voice_turn_speech_end = getattr(stt, "last_speech_end_ts", None)
                    safe_print(f"\r\033[KYou (voice): {voice_text}")
                    return voice_text
            except _queue_mod.Empty:
                pass

        # 3. Twitch chat (only when Rin silent for >1s — Rin still wins races)
        if twitch_runtime and time.time() - started_at > 1.0:
            chat_msg = twitch_runtime.poll_chat_input(timeout=0.0)
            if chat_msg:
                who = chat_msg.display_name or chat_msg.username
                safe_print(f"\r\033[KChat ({who}): {chat_msg.text}")
                return f"[chat: {who}] {chat_msg.text}"

        # 4. Autonomous tick (idle for IDLE_TURN_SECONDS, ALICE_AUTONOMOUS=1 to enable)
        if AUTONOMOUS_ENABLED and time.time() - started_at > IDLE_TURN_SECONDS:
            try:
                from alice.core.mind.idle_turn import next_autonomous_input
                result = next_autonomous_input(mind=mind, twitch_runtime=twitch_runtime)
            except Exception as e:
                log(f"autonomous turn picker failed: {e}", logging.WARNING)
                result = None
            if result:
                mode, framed = result
                log(f"autonomous turn fired: mode={mode}")
                safe_print(f"\r\033[K[autonomous: {mode}]")
                return framed
            # Reset window so we don't spin trying again every frame.
            started_at = time.time()

        # 10ms poll — at 50ms this added up to 50ms (avg 25) of dead time
        # between STT delivering a transcript and the turn starting.
        time.sleep(0.01)


def init():
    """Load all models and systems. Only runs in the main process."""
    global model, tokenizer, mind, fairy, memory
    global script_integration, tts, speech, stt, twitch_runtime, _allow_print

    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    warnings.filterwarnings('ignore')

    # Suppress noisy library loggers (but keep our alice logger + file handler alive)
    for name in list(logging.root.manager.loggerDict):
        if not name.startswith("alice"):
            logging.getLogger(name).setLevel(logging.WARNING)

    log("=== ALICE SESSION START ===")
    _real_print("Loading Alice...", end="", flush=True)

    # Load CUDA model with quantization
    t0 = time.perf_counter()
    log("Loading Alice LLM...")
    with SuppressOutput():
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from weight_quantization import QuantizationConfig

        MODEL_PATH = os.environ.get('ALICE_MODEL', 'models/alice_main')
        quant_config = QuantizationConfig(
            load_in_4bit=True,
            compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            quantization_config=quant_config,
            device_map="cuda:0",
        )
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if GROWTH_ENABLED:
        adapter_path = os.environ.get("ALICE_GROWTH_ADAPTER", "models/growth/adapters/active")
        if os.path.isdir(adapter_path):
            try:
                from adapter_library import AdapterModel
                model = AdapterModel.from_pretrained(model, adapter_path)
                log(f"Growth adapter loaded from {adapter_path}")
                try:
                    import json as _json
                    _gs = _json.load(open('alice/data/growth/state.json'))
                    _real_print(f" [growth L{_gs.get('level', '?')}]...", end="", flush=True)
                except Exception:
                    _real_print(" [growth]...", end="", flush=True)
            except Exception as e:
                log(f"Growth adapter load failed: {e}", logging.WARNING)

    import torch
    gpu_mem_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
    log(f"Alice LLM loaded: device={model.device}, {gpu_mem_mb:.0f}MB GPU memory, {(time.perf_counter()-t0)*1000:.0f}ms")
    _real_print(f" model loaded (device={model.device})...", end="", flush=True)
    if gpu_mem_mb:
        _real_print(f" [{gpu_mem_mb:.0f}MB GPU memory]...", end="", flush=True)

    # Fused attention: register Triton kernel path
    if os.environ.get('ALICE_TURBOQUANT', '1') != '0':
        try:
            from alice.core.optimization.turboquant_attention import register_turboquant_attention
            if register_turboquant_attention():
                model.config._attn_implementation = "turboquant"
                log("Fused attention enabled")
                _real_print(" [fused-attn]...", end="", flush=True)
            else:
                log("Fused attention unavailable, using decompress path")
        except Exception as e:
            log(f"Fused attention failed: {e}", logging.WARNING)

    # v4 Mind: load background thinker for continuous background thinking
    if MIND_ENABLED:
        t0 = time.perf_counter()
        log("Loading Mind...")
        try:
            import torch as _torch_mind
            from alice.core.mind import Mind, ProposalsBuffer
            mem_before = _torch_mind.cuda.memory_allocated() if _torch_mind.cuda.is_available() else 0
            proposals_buffer = ProposalsBuffer(max_proposals=20, ttl_seconds=120.0)
            mind = Mind(proposals_buffer, think_interval=4.0)
            if mind.load_model():
                mem_after = _torch_mind.cuda.memory_allocated() if _torch_mind.cuda.is_available() else 0
                mind_mb = (mem_after - mem_before) / 1024**2
                log(f"Mind loaded: ~{mind_mb:.0f}MB GPU memory (own CUDA stream), {(time.perf_counter()-t0)*1000:.0f}ms")
                _real_print(f" mind(cuda:0, ~{mind_mb:.0f}MB)...", end="", flush=True)
            else:
                log("Mind loaded in fallback mode (no model)")
                _real_print(" mind(fallback, no model)...", end="", flush=True)
        except Exception as e:
            log(f"Mind failed: {e}", logging.ERROR)
            _real_print(f" mind failed ({e})...", end="", flush=True)
            mind = None
    else:
        log("Mind disabled, no background thinker")

    # Load Alice systems
    t0 = time.perf_counter()
    log("Loading systems (registry, fairy, memory, emotion, scripts)...")
    with SuppressOutput():
        from alice.core.system.system_initializer import initialize_all_systems
        from alice.core.system.system_registry import get_registry
        registry = initialize_all_systems()

        fairy = registry.get('fairy')
        memory = registry.get('memory')

        # Initialize memory session - REQUIRED for storage/retrieval
        if memory:
            memory.start_session("rin", "Rin")

        # EmotionDetectionIntegration removed — Mind owns emotion now.
        from alice.core.scripting.script_integration import ScriptIntegration
        script_integration = ScriptIntegration(None)

    log(f"Systems loaded in {(time.perf_counter()-t0)*1000:.0f}ms")
    log(f"  fairy={'yes' if fairy else 'no'}, memory={'yes' if memory else 'no'}")

    # Twitch — connect IRC + register tool dispatcher. Falls back to None
    # if env vars missing or ALICE_TWITCH=0; chat.py treats None as "offline".
    if TWITCH_ENABLED:
        t0 = time.perf_counter()
        log("Connecting Twitch...")
        try:
            from streaming.twitch.runtime import start_twitch
            twitch_runtime = start_twitch(fairy=fairy, mind=mind)
            if twitch_runtime:
                log(f"Twitch connected as {twitch_runtime.client.auth.bot_username} in {(time.perf_counter()-t0)*1000:.0f}ms")
                _real_print(f" twitch({twitch_runtime.client.auth.bot_username})...", end="", flush=True)
            else:
                log("Twitch not configured (missing env vars or ALICE_TWITCH=0)")
        except Exception as e:
            log(f"Twitch failed to start: {e}", logging.ERROR)
            _real_print(f" twitch failed ({e})...", end="", flush=True)

    # Load STT BEFORE TTS — TTS uses torch.compile with reduce-overhead mode
    # (CUDA graphs). Loading models AFTER TTS invalidates captured graphs and
    # degrades RTF from 0.29x to 1.4x. STT must load first.
    if STT_ENABLED:
        t0 = time.perf_counter()
        log("Loading STT...")
        try:
            from alice.core.voice.stt import AliceSTT
            stt = AliceSTT(
                on_transcription=lambda text: _voice_queue.put(text),
                # Speculative turns (ALICE_STT_SPECULATIVE=1): start composing
                # at the soft endpoint, cancel if Rin keeps talking.
                on_tentative=(_spec.start if _spec else None),
                on_tentative_cancelled=(_spec.cancel if _spec else None),
            )
            stt_mem_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
            log(f"STT loaded: {stt_mem_mb:.0f}MB GPU memory, {(time.perf_counter()-t0)*1000:.0f}ms")
            _real_print(" stt loaded...", end="", flush=True)
        except Exception as e:
            log(f"STT failed: {e}", logging.ERROR)
            _real_print(f" stt failed ({e})...", end="", flush=True)

    # Load TTS LAST — torch.compile reduce-overhead mode captures CUDA graphs
    # during warmup. All other GPU models must be loaded before this point.
    if TTS_ENABLED:
        t0 = time.perf_counter()
        tts_mode = os.environ.get('ALICE_TTS_MODE', 'subprocess')  # subprocess or inprocess
        if tts_mode == 'subprocess':
            log("Loading TTS (subprocess)...")
            _real_print(" tts loading (subprocess)...", end="", flush=True)
            try:
                from alice.core.voice.tts_worker import TTSWorker
                from alice.core.voice.speech_pipeline import SpeechPipeline
                tts = TTSWorker(device="cuda:0")
                speech = SpeechPipeline(tts)
                log(f"TTS worker ready: compiled={tts.compiled}, {(time.perf_counter()-t0)*1000:.0f}ms")
                _real_print(f" tts ready (compiled={tts.compiled})...", end="", flush=True)
            except Exception as e:
                log(f"TTS subprocess failed: {e}", logging.ERROR)
                _real_print(f" tts failed ({e})...", end="", flush=True)
        else:
            log("Loading TTS (in-process)...")
            _real_print(" tts loading...", end="", flush=True)
            try:
                with SuppressOutput():
                    from alice.core.voice.tts import AliceTTS
                    from alice.core.voice.speech_pipeline import SpeechPipeline
                    tts = AliceTTS()
                speech = SpeechPipeline(tts)
                tts_mem_mb = torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0
                log(f"TTS loaded in-process: {tts_mem_mb:.0f}MB GPU memory, compiled={tts._compiled}, {(time.perf_counter()-t0)*1000:.0f}ms")
                _real_print(f" tts ready...", end="", flush=True)
            except Exception as e:
                log(f"TTS failed: {e}", logging.ERROR)
                _real_print(f" tts failed ({e})...", end="", flush=True)

    _real_print(" ready!")

    # Quick memory status check (suppress debug output)
    if memory:
        try:
            import io
            import contextlib
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                test_search = memory.search_memories("test", "rin", k=1)
            memory_count = len(test_search) if test_search else 0
            _real_print(f"💾 Memory: {memory_count} memories available\n")
        except:
            _real_print("💾 Memory: Fresh start (no prior memories)\n")
    else:
        _real_print("⚠️ Memory: Not available\n")

    import builtins
    builtins.print = _controlled_print
    _allow_print = False


def main():
    mode = "voice + text" if stt else "text only"
    safe_print(f"Type 'quit' to exit, 'clear' to reset. ({mode})\n")

    if mind:
        mind.start()
        log("Mind background thread started")
        safe_print("Mind: background thinking active (cuda:0, own stream)\n")

    if stt:
        stt.start()
        log("STT listener started")

    animation_proc = None
    if ANIMATION_ENABLED:
        try:
            import subprocess
            animation_proc = subprocess.Popen(
                [sys.executable, "streaming/animation/motion_engine.py"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            log(f"Animation engine started (pid={animation_proc.pid})")
            safe_print("Animation: motion engine started\n")
        except Exception as e:
            log(f"Animation engine failed to start: {e}", logging.WARNING)

    log("=== CHAT LOOP START ===")

    while True:
        try:
            user_input = _get_input_with_voice("You: ")
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                log("User quit")
                if _spec:
                    _spec.cancel()
                safe_print("Bye!")
                break
            if user_input.lower() == 'clear':
                if _spec:
                    _spec.cancel()
                    _spec.wait_idle()
                conversation_history.clear()
                log("Conversation cleared")
                safe_print("(cleared)\n")
                continue

            if stt:
                stt.pause()

            safe_print("Alice: ", end="", flush=True)
            # Speculative path: if a turn for this exact utterance is already
            # in flight (started at the STT soft endpoint), open its gate and
            # join it instead of starting over. Mismatch (keyboard/Twitch
            # input, or a different utterance) cancels the in-flight turn.
            adopted = _spec.try_adopt(user_input) if _spec else None
            if adopted is not None:
                adopted["thread"].join(timeout=300)
                if adopted["thread"].is_alive():
                    log("SPEC adopted turn join timed out", logging.ERROR)
            else:
                if _spec:
                    _spec.wait_idle()  # let any cancelled turn fully unwind
                asyncio.run(get_response(user_input))
            safe_print()

            if stt:
                stt.resume()
                safe_print("You: ", end="", flush=True)

        except KeyboardInterrupt:
            log("KeyboardInterrupt")
            safe_print("\nBye!")
            break
        except EOFError:
            log("EOFError")
            safe_print("\n[EOFError — exiting]")
            break
        except Exception as e:
            import traceback as _tb
            err = _tb.format_exc()
            log(f"MAIN LOOP ERROR: {e}\n{err}", logging.ERROR)
            safe_print(f"\n[ERROR in main loop: {e}]")
            safe_print(err)
            continue

    log("=== CHAT LOOP END ===")
    if twitch_runtime:
        twitch_runtime.stop()
        log("Twitch runtime stopped")
    if speech:
        speech.shutdown()
        log("Speech pipeline stopped")
    if tts:
        tts.shutdown()
        log("TTS worker stopped")
    if mind:
        mind.stop()
        log("Mind stopped")
    if stt:
        stt.stop()
        log("STT stopped")

    # end_session was never called before 2026-06-10 — sessions started but
    # never ended, so session memories never flushed to long-term and the
    # whole post-session pipeline (distiller, divergence, curator) was dark.
    # Distilled flush reuses Alice's own model, still loaded and idle here —
    # zero extra memory/load (ALICE_DISTILL=0 skips straight to the raw
    # flush); fail-open to raw on any error.
    if memory:
        try:
            distiller = None
            if os.environ.get("ALICE_DISTILL", "1") != "0" and model is not None:
                from alice.core.memory.distiller import SessionDistiller
                distiller = SessionDistiller()
            summary = memory.end_session(
                distiller=distiller, distill_model=model, distill_tokenizer=tokenizer
            )
            log(f"Session ended: {summary}")
            safe_print(f"(memory: session flushed — "
                       f"{summary.get('distilled_facts', summary.get('session_flushed', 0))} "
                       f"{summary.get('session_flush_mode', 'raw')})")
        except Exception as e:
            log(f"end_session failed: {e}", logging.ERROR)
    if GROWTH_ENABLED:
        try:
            from alice.core.growth import get_growth_state
            state = get_growth_state()
            if state["xp"] >= state["xp_next"]:
                safe_print(f"(growth: level {state['level'] + 1} ready — "
                           f"run `python training/growth/consolidate.py` to level up)")
        except Exception as e:
            log(f"Growth level check failed: {e}", logging.DEBUG)
    if animation_proc and animation_proc.poll() is None:
        animation_proc.terminate()
        animation_proc.wait(timeout=5)
        log("Animation engine stopped")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    try:
        init()
        main()
    except KeyboardInterrupt:
        log("KeyboardInterrupt (top-level)")
        _real_print("\nBye!")
    except Exception as e:
        import traceback
        msg = f"FATAL CRASH: {e}\n{traceback.format_exc()}"
        log(msg, logging.CRITICAL)
        _real_print(f"\n\n{msg}")
        _real_print(f"\nFull log: {_LOG_PATH}")
        sys.exit(1)
    except BaseException as e:
        import traceback
        msg = f"EXIT: {type(e).__name__}: {e}\n{traceback.format_exc()}"
        log(msg, logging.CRITICAL)
        _real_print(f"\n\n{msg}")
        _real_print(f"\nFull log: {_LOG_PATH}")
        raise
    finally:
        log("=== SESSION END ===")
