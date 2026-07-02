"""
AliceTTS - Streaming text-to-speech with Alice's x-vector voice.

Single-generation dual-track streaming:
  Full response text → one stream_generate_pcm() call with non_streaming_mode=False
  → dual-track architecture interleaves text/audio tokens → PCM chunks yielded
  as they arrive → played via OutputStream.

non_streaming_mode=False uses the dual-track architecture where text tokens are
fed incrementally during generation, so first audio arrives after just a few
characters (~100-300ms) instead of prefilling ALL text first (1.5s+).

Near-greedy sampling locks voice identity.

Uses a streaming fork of the TTS model for fast codebook generation.

Debug: ALICE_DEBUG=1 to see detailed TTS logs.
"""
import os
import re
import subprocess
import time
import traceback

import numpy as np

# Persist torch.compile / triton kernel cache across restarts.
# Must be set BEFORE importing torch.
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "torch_compile")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE_DIR, "triton"))

import torch

DEBUG = os.environ.get("ALICE_DEBUG", "0") == "1"
_log_file = None

def _log(msg):
    if not DEBUG:
        return
    global _log_file
    if _log_file is None:
        _log_file = open(os.path.join(os.path.dirname(__file__), "tts_debug.log"), "w")
    _log_file.write(f"[{time.time():.3f}] {msg}\n")
    _log_file.flush()

# Near-zero temps + top_k 1 = effectively greedy/deterministic.
# Too-cold temps cause metallic artifacts; too-high rep_penalty causes choppy audio
TALKER_TEMPERATURE = 0.7
SUBTALKER_TEMPERATURE = 0.6   # match production (tts_subprocess.py, tuned 2026-05-26)
TOP_K = 50
REPETITION_PENALTY = 1.05
# Last-N window for the repetition penalty — full-history penalty is the
# Known stutter mode on long utterances. Mirrors tts_subprocess.py.
REPETITION_PENALTY_WINDOW = int(os.environ.get("ALICE_TTS_REP_WINDOW", "64"))
VOICE_SEED = 42

EMOTION_TO_INSTRUCT = {
    "neutral": "Speak in a casual, friendly conversational tone",
    "happy": "Speak with warmth and a cheerful, upbeat tone",
    "excited": "Speak with excitement and high energy",
    "sassy": "Speak in a playful, sassy, slightly teasing tone",
    "sad": "Speak in a soft, melancholic tone",
    "angry": "Speak with sharp, irritated energy",
    "thinking": "Speak in a casual, friendly conversational tone",  # same as neutral — TTS model breaks on "calm"/"thoughtful"
    "surprised": "Speak with genuine surprise and wonder",
    "tired": "Speak softly at a normal speaking rate",
}

# Multiple EOS IDs — workaround for upstream TTS model issue
# ~0.5% of inferences miss primary EOS (2150), adding 2157 catches them
EOS_TOKEN_IDS = [2150, 2157]


def _max_tokens_for_text(text: str) -> int:
    """12Hz codec = 12.5 frames/sec. 14 tokens/word ~= 1.1s/word with room for pauses."""
    return max(len(text.split()) * 14, 25)


def _max_frames_for_text(text: str) -> int:
    """For streaming path (stream_generate_pcm). Same logic as _max_tokens_for_text."""
    return _max_tokens_for_text(text)

# Sentence splitting regex — split on . ! ? followed by space or end of string.
# Keeps the punctuation with the sentence.
_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')

VCVARSALL_PATH = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"

def _ensure_msvc_env():
    """Set up MSVC environment variables for torch.compile (cl.exe needs INCLUDE/LIB/PATH)."""
    if os.environ.get("_ALICE_MSVC_READY"):
        _log("MSVC env already set")
        return True
    if not os.path.exists(VCVARSALL_PATH):
        _log(f"vcvarsall.bat not found at {VCVARSALL_PATH}")
        return False
    try:
        _log("Running vcvarsall.bat x64...")
        result = subprocess.run(
            f'cmd /c "\"{VCVARSALL_PATH}\" x64 && set"',
            capture_output=True, text=True, shell=True, timeout=30,
        )
        if result.returncode != 0:
            _log(f"vcvarsall failed: returncode={result.returncode}")
            return False
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ[key] = val
        os.environ["_ALICE_MSVC_READY"] = "1"
        _log("MSVC env loaded")
        return True
    except Exception as e:
        _log(f"MSVC env error: {e}")
        return False

_DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "tts-model")
MODEL_DIR = os.environ.get("ALICE_TTS_MODEL_DIR", _DEFAULT_MODEL_DIR)
XVECTOR_PATH = os.path.join(MODEL_DIR, "alice_xvector.pt")


def _patch_audio_encoder_attention():
    """Monkey-patch audio encoder to use optimized attention on target GPU architecture.

    Flash Attn 2 imports OK but fails at runtime on newer GPU architectures.
    Optimized attention works as a drop-in replacement for the varlen function.
    """
    try:
        if not torch.cuda.is_available():
            return
        props = torch.cuda.get_device_properties(0)
        if props.major < 12:
            return  # Flash Attn works fine on older architectures

        import tts_engine.core.tokenizer_25hz.vq.audio_encoder as we
        try:
            from optimized_attention import optimized_attn_varlen

            def _sage_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, max_seqlen_k, dropout_p=0.0, **kw):
                return optimized_attn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k,
                                       max_seqlen_q, max_seqlen_k)

            we.flash_attn_varlen_func = _sage_wrapper
            _log("Patched audio encoder: Flash Attn -> optimized attention")
        except ImportError:
            we.flash_attn_varlen_func = None
            _log("No optimized attention available, audio encoder using slow fallback")
    except Exception as e:
        _log(f"Audio encoder attention patch skipped: {e}")


class AliceTTS:
    """Streaming TTS with Alice's cloned voice."""

    def __init__(self, model_dir: str = None, xvector_path: str = None, device: str = "cuda:0"):
        model_dir = model_dir or MODEL_DIR
        xvector_path = xvector_path or XVECTOR_PATH

        _log(f"init: model_dir={model_dir}")
        _log(f"init: xvector_path={xvector_path}")
        _log(f"init: device={device}")

        t0 = time.time()
        from tts_engine import TTSModel

        _log("Loading TTS model from_pretrained...")
        self.model = TTSModel.from_pretrained(
            model_dir,
            device_map=device,
            dtype=torch.bfloat16,
        )

        _patch_audio_encoder_attention()

        # Detect model variant up front — drives quantization, decoder routing,
        # ref_code, and TurboQuant decisions. The smaller base model is too
        # small to absorb weight quantization or KV-cache compression without
        # audible quality loss.
        _mtype = getattr(self.model.model.config, "tts_model_type", None)
        self._is_custom_voice = (_mtype == "custom_voice")
        _log(f"tts_model_type={_mtype}, is_custom_voice={self._is_custom_voice}")

        # Weight quantization — int8 default for both variants.
        # The earlier "int8 destroys base" claim was from stacking int8 with
        # TurboQuant + decode_padded; with our current per-variant routing
        # (no TurboQuant, chunked_decode, no ref_code on base) int8 is clean.
        # int4 uses Triton which fails on some GPU architectures.
        # Set ALICE_TTS_QUANTIZE=int4|int8|0 to override.
        _default_quant = "int8"
        _quant_mode = os.environ.get("ALICE_TTS_QUANTIZE", _default_quant).lower()
        if _quant_mode != "0":
            try:
                from weight_quantization import quantize_
                t_q = time.time()
                if _quant_mode == "int4":
                    from weight_quantization import Int4WeightOnlyConfig
                    _log("Applying int4 weight quantization...")
                    quantize_(self.model.model, Int4WeightOnlyConfig(
                        group_size=32, int4_packing_format='plain', version=1,
                    ))
                    _log(f"int4 quantization done in {time.time()-t_q:.1f}s")
                    print("  [TTS] int4 quantized")
                else:
                    from weight_quantization import Int8WeightOnlyConfig
                    _log("Applying int8 weight quantization...")
                    quantize_(self.model.model, Int8WeightOnlyConfig())
                    _log(f"int8 quantization done in {time.time()-t_q:.1f}s")
                    print("  [TTS] int8 quantized")
            except Exception as e:
                _log(f"quantization failed, using bf16: {e}")
                print(f"  [TTS] quantization failed ({e}), using bf16")
        else:
            _log("TTS loaded in bf16 (quantization disabled)")

        _log(f"Model loaded in {time.time()-t0:.1f}s")

        # Enable fast codebook generation (bypasses HuggingFace GenerationMixin overhead)
        _log("Enabling fast codebook gen...")
        self.model.model.talker.enable_fast_codebook_gen(True)

        # Enable torch.compile if available (requires C++ compiler + MSVC env)
        # Skip with ALICE_TTS_NO_COMPILE=1
        self._compiled = False
        if os.environ.get("ALICE_TTS_NO_COMPILE") == "1":
            print("  [TTS] torch.compile disabled (ALICE_TTS_NO_COMPILE=1)")
        elif _ensure_msvc_env():
            try:
                _log("Enabling streaming optimizations (torch.compile)...")
                t0 = time.time()
                self.model.model.enable_streaming_optimizations(
                    use_compile=True,
                    use_cuda_graphs=False,  # CUDA graphs conflict with dynamic KV cache sizes
                    compile_codebook_predictor=True,
                    compile_talker=True,
                )
                self._compiled = True
                _log(f"torch.compile enabled in {time.time()-t0:.1f}s")
                print("  [TTS] torch.compile enabled (first generation will be slow due to compilation)")
            except Exception as e:
                _log(f"torch.compile failed: {e}\n{traceback.format_exc()}")
                print(f"  [TTS] torch.compile failed ({e}), using uncompiled")
        else:
            print("  [TTS] MSVC not found, torch.compile disabled (running ~7x slower)")

        _log(f"Loading x-vector from {xvector_path}...")
        xvec = torch.load(xvector_path, weights_only=True, map_location=device)
        _log(f"X-vector shape: {xvec.shape}, dtype: {xvec.dtype}")

        # Streaming-decoder routing depends on the model variant.
        # Determined empirically:
        #
        #   custom_voice variant:
        #     - Uses speech_tokenizer.decode_padded (use_optimized_decode=True)
        #     - That path's conv receptive field bleeds zero-context into the
        #       first real frame, producing a ~30ms Nyquist burst at sample
        #       ~2000 of the first chunk. Feeding real codec frames as pad
        #       context fixes it.
        #     - So: use_optimized_decode=True, ref_code=<precomputed>
        #
        #   base variant:
        #     - decode_padded produces muddy audio for this decoder; must use
        #       chunked_decode (use_optimized_decode=False), which is what
        #       generate() uses internally.
        #     - chunked_decode doesn't have the zero-pad conv issue and in
        #       fact mis-renders if ref_code is prepended (the prepend confuses
        #       the variable-length conv).
        #     - So: use_optimized_decode=False, ref_code=None
        # chunked_decode for BOTH variants — decode_padded garbles articulation
        # on both (subtler on custom_voice; ear-confirmed). ALICE_TTS_PADDED_DECODE=1
        # restores the old path. Mirrors tts_subprocess.py.
        self._use_optimized_decode = (
            self._is_custom_voice and os.environ.get("ALICE_TTS_PADDED_DECODE") == "1"
        )
        _log(f"use_optimized_decode={self._use_optimized_decode}")

        ref_code = None
        if self._use_optimized_decode:
            # Loads precomputed codec tensor (alice_ref_code.pt, ~26-65KB).
            # Falls back to encoding ALICE_TTS_ICL_REF_AUDIO on the fly.
            # Regenerate with alice/tests/precompute_ref_code.py.
            default_pt = os.path.join(model_dir, "alice_ref_code.pt")
            if os.path.exists(default_pt):
                try:
                    ref_code = torch.load(default_pt, weights_only=True, map_location=device)
                    _log(f"decoder pad-ctx: loaded {default_pt}, shape={tuple(ref_code.shape)}")
                except Exception as e:
                    _log(f"decoder pad-ctx: failed to load {default_pt}: {e}")
                    ref_code = None

            if ref_code is None:
                ref_audio_path = os.environ.get("ALICE_TTS_ICL_REF_AUDIO")
                if ref_audio_path and os.path.exists(ref_audio_path):
                    try:
                        import soundfile as sf
                        wav_np, sr = sf.read(ref_audio_path, always_2d=False)
                        if wav_np.ndim > 1:
                            wav_np = wav_np.mean(axis=-1)
                        wav_np = wav_np.astype("float32")
                        enc = self.model.model.speech_tokenizer.encode([wav_np], sr=sr)
                        ref_code = enc.audio_codes[0]
                        _log(f"decoder pad-ctx: encoded {ref_audio_path}, shape={tuple(ref_code.shape)}")
                    except Exception as e:
                        _log(f"decoder pad-ctx: failed to encode {ref_audio_path}: {e}")
                        ref_code = None
                else:
                    _log("decoder pad-ctx: no precomputed .pt and no ALICE_TTS_ICL_REF_AUDIO — first-chunk click will return")
        else:
            _log("decoder pad-ctx: skipped (base variant uses chunked_decode, ref_code prepend is harmful)")

        self.voice_clone_prompt = {
            "ref_code": [ref_code],
            "ref_spk_embedding": [xvec.unsqueeze(0) if xvec.dim() == 1 else xvec],
            "x_vector_only_mode": [True],
            "icl_mode": [False],
        }

        import sounddevice as sd
        self._sd = sd
        _log(f"sounddevice default output: {sd.query_devices(kind='output')['name']}")

        # Streaming state
        self._text_buffer = ""
        self._emotion = "neutral"
        self._stop_playback = False

        # Warmup: torch.compile JIT-compiles separate kernels for each distinct
        # tensor shape. We run 3 warmups of increasing length so that short,
        # medium, and long inputs all hit cached kernels at runtime.
        if self._compiled:
            self._warmup()

        _log("init complete")

    # ------------------------------------------------------------------
    # Warmup: pre-compile torch kernels for short/medium/long inputs
    # ------------------------------------------------------------------

    def _warmup(self):
        """Run silent generations to trigger torch.compile for common bucket sizes.

        With bucket-padded tensors, we only need two passes: one short (hits
        the small prefill + trailing bucket) and one longer (hits the next
        bucket up). All real inputs that fall within these buckets will reuse
        the compiled kernels.
        """
        warmup_texts = [
            "Hello.",                                                    # short (~10 tokens)
            "This is a longer warmup sentence to make sure the model "   # long (~50 tokens)
            "compiles kernels for multi-sentence responses so there is "
            "no delay when speaking longer passages during conversation.",
        ]
        for i, text in enumerate(warmup_texts):
            t0 = time.time()
            _log(f"warmup {i+1}/{len(warmup_texts)}: {text[:40]!r}...")
            for _ in self._stream_generate(text, "neutral"):
                pass  # discard audio
            _log(f"warmup {i+1} done in {time.time()-t0:.1f}s")
        print(f"  [TTS] warmup complete ({len(warmup_texts)} passes)")

    # ------------------------------------------------------------------
    # Internal: tokenize + build instruct for a given text/emotion
    # ------------------------------------------------------------------

    def _prepare_inputs(self, text: str, emotion: str):
        """Tokenize text and instruct, return (input_ids, instruct_ids)."""
        instruct = EMOTION_TO_INSTRUCT.get(emotion, EMOTION_TO_INSTRUCT["neutral"])
        _log(f"prepare: text={text[:60]!r}... emotion={emotion} instruct={instruct[:40]!r}")

        input_ids = self.model._tokenize_texts([self.model._build_assistant_text(text)])
        instruct_ids = self.model._tokenize_texts([self.model._build_instruct_text(instruct)])
        return input_ids, instruct_ids

    # ------------------------------------------------------------------
    # Core: stream PCM chunks for full text (single generation)
    # ------------------------------------------------------------------

    def _stream_generate(self, text: str, emotion: str):
        """
        Single-generation streaming: yields (pcm_numpy, sample_rate) chunks.

        The entire text is generated in one call to stream_generate_pcm(),
        keeping a single KV cache context for voice consistency. PCM chunks
        are yielded every 8 codec frames (~300-500ms into generation).
        """
        word_count = len(text.split())

        # Short text: force neutral emotion to prevent "thinking"/"tired" from
        # making the model drag out a single word into 30+ seconds of audio.
        if word_count < 3:
            _log(f"stream_generate: short text ({word_count} words), forcing neutral emotion")
            emotion = "neutral"

        input_ids, instruct_ids = self._prepare_inputs(text, emotion)

        _log(f"stream_generate: starting stream_generate_pcm for {len(text)} chars")
        t0 = time.time()
        chunk_count = 0

        # Run TTS on its own CUDA stream to isolate from LLM/STT compiled kernels
        from alice.core.cuda_streams import get_stream
        tts_stream = get_stream("tts")

        # Create TurboQuant cache if enabled (fresh per generation).
        # Skip for base-type models — KV-cache compression destroys
        # speaker conditioning on the smaller talker.
        tq_cache = None
        _tq_default = "1" if self._is_custom_voice else "0"
        if os.environ.get("ALICE_TURBOQUANT", _tq_default) != "0":
            try:
                from alice.core.optimization.turboquant_cache import TurboQuantDynamicCache, TURBOQUANT_AVAILABLE
                if TURBOQUANT_AVAILABLE:
                    tq_cache = TurboQuantDynamicCache(
                        n_layers=28, head_dim=128,
                        key_bits=4, value_bits=2,
                        residual_window=128, protected_layers=4,
                    )
                    _log("TurboQuant cache created for TTS (K4/V2, rw=128, prot=4)")
            except Exception as e:
                _log(f"TurboQuant TTS cache failed: {e}")

        # Fixed seed for consistent voice timbre across generations
        torch.manual_seed(VOICE_SEED)
        torch.cuda.manual_seed(VOICE_SEED)

        with torch.cuda.stream(tts_stream):
            for pcm_chunk, sr in self.model.model.stream_generate_pcm(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                voice_clone_prompt=self.voice_clone_prompt,
                languages=["English"],
                non_streaming_mode=False,
                temperature=TALKER_TEMPERATURE,
                subtalker_temperature=SUBTALKER_TEMPERATURE,
                top_k=TOP_K,
                repetition_penalty=REPETITION_PENALTY,
                repetition_penalty_window=REPETITION_PENALTY_WINDOW,
                eos_token_id=EOS_TOKEN_IDS,
                emit_every_frames=8,
                decode_window_frames=80,
                overlap_samples=0,  # vendor's crossfade blends non-aligned samples → click at each boundary
                max_frames=_max_frames_for_text(text),
                use_optimized_decode=self._use_optimized_decode,
                past_key_values=tq_cache,
            ):
                if self._stop_playback:
                    _log("stream_generate: stop requested, breaking")
                    break

                chunk_count += 1
                if chunk_count == 1:
                    _log(f"stream_generate: first chunk at {time.time()-t0:.3f}s, sr={sr}")

                yield pcm_chunk, sr

        elapsed = time.time() - t0
        _log(f"stream_generate: done. {chunk_count} chunks in {elapsed:.3f}s")

    # ------------------------------------------------------------------
    # Legacy: generate full wav for a single chunk (kept for debugging)
    # ------------------------------------------------------------------

    def _generate_wav(self, text: str, emotion: str):
        """Generate audio for a text chunk. Returns (wav_numpy, sample_rate).

        Kept for backward compatibility / debugging. Primary path is _stream_generate().
        """
        word_count = len(text.split())

        # Short text: force neutral emotion to prevent "thinking"/"tired" from
        # making the model drag out a single word into 30+ seconds of audio.
        if word_count < 3:
            _log(f"generate: short text ({word_count} words), forcing neutral emotion")
            emotion = "neutral"

        input_ids, instruct_ids = self._prepare_inputs(text, emotion)

        torch.manual_seed(VOICE_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(VOICE_SEED)

        t0 = time.time()
        talker_codes_list, _ = self.model.model.generate(
            input_ids=input_ids,
            instruct_ids=instruct_ids,
            languages=["English"],
            voice_clone_prompt=self.voice_clone_prompt,
            non_streaming_mode=True,
            temperature=TALKER_TEMPERATURE,
            subtalker_temperature=SUBTALKER_TEMPERATURE,
            max_new_tokens=_max_tokens_for_text(text),
            top_k=TOP_K,
            repetition_penalty=REPETITION_PENALTY,
            eos_token_id=EOS_TOKEN_IDS,
        )
        gen_time = time.time() - t0
        _log(f"  model.generate in {gen_time:.3f}s")

        t0 = time.time()
        wavs, sr = self.model.model.speech_tokenizer.decode(
            [{"audio_codes": c} for c in talker_codes_list]
        )
        _log(f"  decode in {time.time()-t0:.3f}s")

        wav = wavs[0]
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()
        if wav.ndim > 1:
            wav = wav.squeeze()

        # Pad 150ms silence so the tail doesn't get clipped
        wav = np.concatenate([wav, np.zeros(int(sr * 0.15), dtype=wav.dtype)])

        duration = len(wav) / sr
        rtf = gen_time / duration if duration > 0 else float("inf")
        _log(f"  audio={duration:.2f}s, RTF={rtf:.3f}x")
        return wav, sr

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences. Keeps punctuation attached."""
        parts = _SENTENCE_RE.split(text.strip())
        # Filter empty strings, merge very short fragments into previous
        sentences = []
        for p in parts:
            p = p.strip()
            if not p:
                continue
            # If this fragment is very short (< 3 words) and we have a previous
            # sentence, merge it to avoid tiny TTS generations
            if sentences and len(p.split()) < 3:
                sentences[-1] = sentences[-1] + " " + p
            else:
                sentences.append(p)
        return sentences if sentences else [text.strip()]

    # ------------------------------------------------------------------
    # Streaming API: start_stream → feed_text → end_stream
    # ------------------------------------------------------------------

    def start_stream(self, emotion: str = "neutral"):
        """
        Begin streaming mode. Call feed_text() with each token, then end_stream().

        Text accumulates until end_stream() is called, then generated in a single
        stream_generate_pcm() call with dual-track streaming (non_streaming_mode=False).

        Emotion is determined BEFORE generation by the emotional state system,
        not detected post-hoc from Alice's text.
        """
        _log(f"start_stream: emotion={emotion}")
        self._emotion = emotion
        self._text_buffer = ""
        self._stop_playback = False
        _log("start_stream: ready to accumulate text")

    def feed_text(self, token: str):
        """Feed a streamed text token. Just accumulates — no generation yet."""
        self._text_buffer += token

    def end_stream(self):
        """
        Generate and play audio for the accumulated text.

        Uses single-generation dual-track streaming: one stream_generate_pcm()
        call with non_streaming_mode=False for the full text. The dual-track
        architecture feeds text tokens incrementally during generation, giving
        fast first audio (~100-300ms) with consistent voice (single KV cache).

        Generation and playback are decoupled: a background thread plays chunks
        from a queue while the main thread continues generating. This prevents
        blocking stream.write() calls from stalling GPU generation.
        """
        import threading
        import queue
        from streaming.animation.emotion_state import set_speaking

        text = self._text_buffer.strip()
        self._text_buffer = ""

        if not text:
            _log("end_stream: empty text, skipping")
            return

        _log(f"end_stream: {len(text)} chars, emotion={self._emotion}")
        import sys as _sys
        _sys.stderr.write(f"[TTS] Speaking: {text!r}\n")

        set_speaking(True)
        total_samples = 0
        sr = 24000
        t0 = time.perf_counter()

        # Timing instrumentation
        _chunk_times = []
        _chunk_sizes = []
        _write_times = []
        _raw_chunks = []

        # Queue-based playback: generation fills queue, playback thread drains it.
        # Pre-buffer PRE_BUFFER chunks before starting playback to build runway
        # against RTF > 1.0x (generation slower than real-time).
        PRE_BUFFER = 3  # ~1.9s of audio runway before playback starts
        audio_queue = queue.Queue(maxsize=32)
        _SENTINEL = None  # signals end of generation
        _playback_started = threading.Event()

        def _playback_thread():
            """Drain audio_queue → sounddevice OutputStream."""
            stream = None
            try:
                # Wait for pre-buffer to fill (or generation to end early)
                _playback_started.wait()

                while True:
                    item = audio_queue.get()
                    if item is _SENTINEL:
                        break
                    pcm_f32, chunk_sr = item
                    if stream is None:
                        stream = self._sd.OutputStream(
                            samplerate=chunk_sr, channels=1, dtype="float32",
                            blocksize=chunk_sr,  # 1s buffer — reduces write blocking
                        )
                        stream.start()
                    t_w0 = time.perf_counter()
                    stream.write(pcm_f32)
                    _write_times.append(time.perf_counter() - t_w0)
            except Exception as e:
                _log(f"playback thread error: {e}\n{traceback.format_exc()}")
            finally:
                if stream is not None:
                    # Drain remaining buffered audio
                    try:
                        remaining = total_samples / sr - (time.perf_counter() - t0)
                        if remaining > 0:
                            time.sleep(remaining + 0.05)
                    except Exception:
                        pass
                    stream.stop()
                    stream.close()

        player = threading.Thread(target=_playback_thread, daemon=True, name="TTS-Playback")
        player.start()

        try:
            for pcm_chunk, sr in self._stream_generate(text, self._emotion):
                if self._stop_playback:
                    break

                t_chunk = time.perf_counter() - t0
                _chunk_times.append(t_chunk)
                _chunk_sizes.append(len(pcm_chunk))
                _raw_chunks.append(pcm_chunk.copy())
                total_samples += len(pcm_chunk)

                if len(_chunk_times) == 1:
                    _log(f"end_stream: first audio at {t_chunk:.3f}s")

                pcm_f32 = pcm_chunk.astype(np.float32).reshape(-1, 1)
                audio_queue.put((pcm_f32, sr))

                # Start playback after pre-buffer fills (or on last chunk if short)
                if not _playback_started.is_set() and len(_chunk_times) >= PRE_BUFFER:
                    _playback_started.set()

        except Exception as e:
            _log(f"end_stream: FAILED: {e}\n{traceback.format_exc()}")
        finally:
            # If generation ended before pre-buffer filled, start playback now
            _playback_started.set()
            audio_queue.put(_SENTINEL)
            player.join(timeout=30)
            set_speaking(False)

        duration = total_samples / sr if sr > 0 else 0
        elapsed = time.perf_counter() - t0
        _log(f"end_stream: done. {duration:.2f}s audio, {elapsed:.2f}s wall")

        # Print timing report (uses sys.stderr to bypass builtins.print override)
        import sys
        _p = sys.stderr.write
        _p(f"\n[TTS] {len(_chunk_times)} chunks, {duration:.1f}s audio, {elapsed:.1f}s wall, first@{_chunk_times[0]*1000:.0f}ms\n" if _chunk_times else "\n[TTS] no chunks\n")
        if _chunk_times:
            gaps = []
            for i in range(1, len(_chunk_times)):
                gap = (_chunk_times[i] - _chunk_times[i-1]) * 1000
                if gap > 80:
                    gaps.append((i, gap))
            if gaps:
                _p(f"[TTS] WARNING: {len(gaps)} generation gaps > 80ms:\n")
                for idx, gap in gaps:
                    _p(f"  chunk {idx}: gen_gap={gap:.0f}ms samples={_chunk_sizes[idx]}\n")
            else:
                _p(f"[TTS] OK: no generation gaps > 80ms (max={max((_chunk_times[i]-_chunk_times[i-1])*1000 for i in range(1,len(_chunk_times))):.0f}ms)\n")

        # Save WAV for inspection
        if _raw_chunks and total_samples > 0:
            try:
                import wave as _wave
                _wav_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "alice", "tests", "tts_last_playback.wav")
                os.makedirs(os.path.dirname(_wav_path), exist_ok=True)
                _all = np.concatenate(_raw_chunks)
                _int16 = (np.clip(_all, -1, 1) * 32767).astype(np.int16)
                with _wave.open(_wav_path, 'w') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sr)
                    wf.writeframes(_int16.tobytes())
                _p(f"[TTS] wav saved: alice/tests/tts_last_playback.wav\n")
            except Exception as _e:
                _p(f"[TTS] wav save failed: {_e}\n")

    # ------------------------------------------------------------------
    # One-shot API (non-streaming convenience, used by warmup)
    # ------------------------------------------------------------------

    def speak(self, text: str, emotion: str = "neutral"):
        """Generate and play speech for a complete text (blocks until done)."""
        _log(f"speak: text={text[:60]!r} emotion={emotion}")
        self.start_stream(emotion)
        self.feed_text(text)
        self.end_stream()
        _log("speak: done")
