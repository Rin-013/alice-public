"""
AliceSTT - Always-on voice input using a speech-to-text model + Silero VAD.

Uses a HuggingFace transformers STT model with torch.compile on encoder for
fast transcription on CUDA. Silero VAD detects speech boundaries.

Pipeline: Mic (sounddevice) → Silero VAD (onset+endpoint) → STT model → text callback
"""
import logging
import os
import subprocess
import threading
import time
import warnings
import numpy as np
from typing import Callable, Optional

logger = logging.getLogger(__name__)

# Persist torch.compile / triton kernel cache across restarts.
# Must be set BEFORE importing torch.
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "torch_compile")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE_DIR, "triton"))

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import sounddevice as sd
    import torch
    from stt_library import STTModel
    from stt_library import STTProcessor
    STT_AVAILABLE = True
except ImportError as e:
    STT_AVAILABLE = False
    print(f"   Failed in {__file__}: {e}")

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 512  # Silero VAD minimum: 512 samples (~32ms at 16kHz)
CHUNK_MS = CHUNK_SAMPLES * 1000.0 / SAMPLE_RATE  # 32ms per VAD chunk

# Hard endpoint — trailing silence that confirms the utterance is over. This
# is the latency FLOOR for every voice turn: Alice cannot start responding
# sooner than this after Rin's last word. 750ms → 400ms (2026-06-11 latency
# pass). Raise it if Alice starts talking over mid-sentence thinking pauses.
ENDPOINT_MS = int(os.environ.get("ALICE_STT_ENDPOINT_MS", "400"))
SILENCE_CHUNKS_THRESHOLD = max(2, round(ENDPOINT_MS / CHUNK_MS))

# Soft endpoint — transcribe EAGERLY at this much silence so the text is
# already in hand when the hard endpoint confirms (STT ~40ms hides
# entirely inside the remaining wait). If speech resumes before the hard
# endpoint, the eager transcript is discarded and buffering continues.
SOFT_ENDPOINT_MS = int(os.environ.get("ALICE_STT_SOFT_ENDPOINT_MS", "224"))
SOFT_SILENCE_CHUNKS = max(1, round(SOFT_ENDPOINT_MS / CHUNK_MS))

MIN_SPEECH_CHUNKS = 5  # ~160ms minimum speech — catches "hmm"/"yeah", filters clicks

VCVARSALL_PATH = r"C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvarsall.bat"


def _ensure_msvc_env():
    """Set up MSVC environment for torch.compile."""
    if os.environ.get("_ALICE_MSVC_READY"):
        return True
    if not os.path.exists(VCVARSALL_PATH):
        return False
    try:
        result = subprocess.run(
            f'cmd /c "\"{VCVARSALL_PATH}\" x64 && set"',
            capture_output=True, text=True, shell=True, timeout=30,
        )
        if result.returncode != 0:
            return False
        for line in result.stdout.splitlines():
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ[key] = val
        os.environ["_ALICE_MSVC_READY"] = "1"
        return True
    except Exception:
        return False


class AliceSTT:
    """
    Always-on speech-to-text using an STT model + Silero VAD.

    Fast transcription on CUDA (after warmup). Low GPU memory footprint.
    Silero VAD: <1ms per chunk on CPU.
    """

    def __init__(
        self,
        on_transcription: Callable[[str], None],
        model_name: str = "stt-model",  # HuggingFace model identifier
        device: str = "cuda:0",
        on_tentative: Optional[Callable[[str], None]] = None,
        on_tentative_cancelled: Optional[Callable[[], None]] = None,
    ):
        if not STT_AVAILABLE:
            raise RuntimeError("sounddevice, torch, or transformers not available")

        self._on_transcription = on_transcription
        # Speculative endpointing (docs/plans/VOICE_LATENCY.md Phase 2):
        # on_tentative fires at the SOFT endpoint with the eager transcript so
        # the caller can start composing a response ~160ms before the hard
        # endpoint commits. If speech resumes first, on_tentative_cancelled
        # fires and the tentative text must be discarded. When the hard
        # endpoint confirms, on_transcription delivers the IDENTICAL string
        # (the eager transcript is reused, not re-run) — callers match on it.
        self._on_tentative = on_tentative
        self._on_tentative_cancelled = on_tentative_cancelled
        self._device = device
        self._running = False
        self._paused = False
        self._thread: Optional[threading.Thread] = None
        # perf_counter of the last VAD speech chunk in the utterance most
        # recently delivered via on_transcription — i.e. when Rin's last word
        # ended. chat.py reads this to log true last-word→first-audio latency.
        self.last_speech_end_ts: Optional[float] = None

        # Load Silero VAD (tiny, runs on CPU — <1ms per chunk)
        self._vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )

        # Load STT model (transformers, bf16 on CUDA — matches TTS/LLM precision)
        self._processor = STTProcessor.from_pretrained(model_name)
        self._stt_model = STTModel.from_pretrained(
            model_name, dtype=torch.bfloat16,
        ).to(device)
        self._stt_model.config.forced_decoder_ids = None
        self._stt_model.generation_config.forced_decoder_ids = None
        # Align max_length with our max_new_tokens=128 at call site. Setting to
        # None backfilled with transformers default of 20; setting to a real
        # value matching the call kwargs stops the dual-set warning.
        self._stt_model.generation_config.max_length = 128
        # Mute transformers' generation warnings (max_length, attention_mask,
        # custom logits processor noise that fires every transcription).
        try:
            import transformers
            transformers.logging.set_verbosity_error()
        except Exception:
            pass

        # torch.compile DISABLED on STT — the STT model is small and fast
        # uncompiled, and its compiled kernels conflict with TTS's compiled
        # kernels via torch._dynamo's shared code cache, causing TTS recompilation
        # that degrades RTF. Not worth the marginal savings.
        self._compiled = False
        print("  [STT] torch.compile disabled (prevents TTS kernel cache conflicts)")

    def start(self) -> None:
        """Start the STT listener in a background thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def pause(self) -> None:
        """Pause listening (e.g. while Alice is speaking to avoid echo)."""
        self._paused = True

    def resume(self) -> None:
        """Resume listening after pause."""
        self._paused = False
        self._vad_model.reset_states()

    def _listen_loop(self) -> None:
        """Main mic loop: capture audio, detect speech, transcribe.

        Two-stage endpointing: at SOFT_SILENCE_CHUNKS of trailing silence the
        buffer is transcribed eagerly (result held, not delivered); at
        SILENCE_CHUNKS_THRESHOLD the held text is delivered instantly. Speech
        resuming between the two discards the eager transcript. STT model's
        ~40ms thus never sits on the post-endpoint critical path.
        """
        buffer = []
        speaking = False
        silence_count = 0
        speech_count = 0
        eager_text: Optional[str] = None  # transcript held from soft endpoint
        tentative_live = False  # on_tentative fired, not yet confirmed/cancelled
        last_speech_ts = 0.0

        def _cancel_tentative():
            nonlocal tentative_live
            if tentative_live:
                tentative_live = False
                if self._on_tentative_cancelled:
                    try:
                        self._on_tentative_cancelled()
                    except Exception:
                        logger.warning("on_tentative_cancelled raised", exc_info=True)

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="float32",
            blocksize=CHUNK_SAMPLES,
        ) as stream:
            while self._running:
                chunk, overflowed = stream.read(CHUNK_SAMPLES)
                if self._paused:
                    if speaking:
                        buffer.clear()
                        speaking = False
                        silence_count = 0
                        speech_count = 0
                        eager_text = None
                        _cancel_tentative()
                        self._vad_model.reset_states()
                    continue

                chunk_1d = chunk.squeeze()
                chunk_tensor = torch.from_numpy(chunk_1d)
                speech_prob = self._vad_model(chunk_tensor, SAMPLE_RATE).item()

                if speech_prob > 0.5:
                    speaking = True
                    silence_count = 0
                    speech_count += 1
                    eager_text = None  # speech resumed — held transcript is stale
                    _cancel_tentative()
                    last_speech_ts = time.perf_counter()
                    buffer.append(chunk_1d.copy())
                elif speaking:
                    silence_count += 1
                    buffer.append(chunk_1d.copy())

                    if (
                        silence_count == SOFT_SILENCE_CHUNKS
                        and silence_count < SILENCE_CHUNKS_THRESHOLD
                        and speech_count >= MIN_SPEECH_CHUNKS
                    ):
                        eager_text = self._transcribe(np.concatenate(buffer))
                        if eager_text and self._on_tentative:
                            self.last_speech_end_ts = last_speech_ts
                            tentative_live = True
                            try:
                                self._on_tentative(eager_text)
                            except Exception:
                                tentative_live = False
                                logger.warning("on_tentative raised", exc_info=True)

                    if silence_count >= SILENCE_CHUNKS_THRESHOLD:
                        if speech_count >= MIN_SPEECH_CHUNKS:
                            text = eager_text
                            if text is None:
                                text = self._transcribe(np.concatenate(buffer))
                            if text and not self._paused:
                                self.last_speech_end_ts = last_speech_ts
                                logger.info(f"STT transcribed: {text!r}")
                                self._on_transcription(text)
                            elif not text:
                                logger.debug("STT empty transcription (VAD tripped on non-speech)")

                        buffer.clear()
                        speaking = False
                        silence_count = 0
                        speech_count = 0
                        eager_text = None
                        tentative_live = False  # confirmed via on_transcription
                        self._vad_model.reset_states()

    def _transcribe(self, audio: np.ndarray) -> Optional[str]:
        """Run STT model on captured audio. Returns the text ('' if nothing
        intelligible, None on error) — delivery is the listen loop's job."""
        try:
            input_features = self._processor(
                audio, sampling_rate=SAMPLE_RATE, return_tensors="pt",
            ).input_features.to(self._device, dtype=torch.bfloat16)

            # TurboQuant cache disabled on STT — TurboQuantDynamicCache is
            # missing `is_updated` (set by transformers' modern generation loop)
            # and crashes with "local variable 'is_updated' referenced before
            # assignment" every transcription. The STT model is small enough that
            # KV compression saves negligible memory — not worth the breakage.
            from alice.core.cuda_streams import get_stream
            # Suppress transformers' per-call generation warnings (attention_mask,
            # max_length/max_new_tokens dual-set, custom logits processor noise).
            # They go through warnings.warn(), not transformers.logging — module-
            # level filterwarnings gets reset by other libraries, so scope it here.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with torch.cuda.stream(get_stream("stt")):
                    with torch.no_grad():
                        predicted_ids = self._stt_model.generate(
                            input_features=input_features,
                            max_new_tokens=128,
                        )

            return self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        except Exception as e:
            logger.warning(f"STT transcribe failed: {e}", exc_info=True)
            return None
