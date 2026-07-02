"""
TTS Subprocess — Standalone TTS model server running in .venv_tts.

Communicates via stdin/stdout JSON lines. Audio playback happens entirely
inside this process (sounddevice + VB-Cable), so no PCM crosses the boundary.

Run directly:  .venv_tts/Scripts/python alice/core/voice/tts_subprocess.py
Or spawned by TTSWorker in tts_worker.py.
"""
import json
import os
import sys
import time
import traceback

# Persist torch.compile / triton kernel cache across restarts.
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".cache", "torch_compile")
os.makedirs(_CACHE_DIR, exist_ok=True)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", _CACHE_DIR)
os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(_CACHE_DIR, "triton"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Required for torch.use_deterministic_algorithms(True) to work with cuBLAS
# on CUDA ≥10.2 — without it, cuBLAS GEMMs raise RuntimeError under the
# deterministic flag. ":4096:8" is the recommended setting for low memory
# overhead while keeping GEMMs deterministic.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

# Suppress library noise
import warnings
warnings.filterwarnings('ignore')
import logging
logging.disable(logging.CRITICAL)
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['_ALICE_TTS_WORKER'] = '1'

import numpy as np
import subprocess as _subprocess


def _log(msg):
    sys.stderr.write(f"[TTS-subprocess] {msg}\n")
    sys.stderr.flush()


def _send(obj):
    """Write JSON line to stdout."""
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _recv():
    """Read JSON line from stdin. Returns None on EOF."""
    line = sys.stdin.readline()
    if not line:
        return None
    return json.loads(line.strip())


# ---------- Paths ----------
_base = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get(
    "ALICE_TTS_MODEL_DIR",
    os.path.join(_base, "..", "..", "..", "models", "tts-model"),
)
# Default xvector. Base variant ships `alice_xvector_native.pt` (preferred there);
# custom_voice variant only has `alice_xvector.pt`. Try native first, fall back.
def _default_xvector() -> str:
    native = os.path.join(MODEL_DIR, "alice_xvector_native.pt")
    if os.path.exists(native):
        return native
    return os.path.join(MODEL_DIR, "alice_xvector.pt")

XVECTOR_PATH = os.environ.get("ALICE_XVECTOR_PATH", _default_xvector())

# Voice settings.
# Per-emotion instruct conditioning was removed (changed the talker's prompt
# every turn → voice drift). BUT we still pass a SINGLE FIXED instruct string
# every call — removing instruct entirely caused the talker to fail clean EOS
# detection (known upstream issue: runs to max_frames, audible as stuttering /
# getting stuck on phonemes like "oh"). The fixed instruct gives the talker
# the in-distribution conditioning it was trained to terminate cleanly with;
# the constancy across calls preserves voice consistency.
# Emotional inflection now comes from Alice's text (punctuation, word choice).
# The `emotion` arg on the speak API is accepted for caller compatibility
# but ignored — every call uses FIXED_INSTRUCT.
# 2026-06-10 probe (probe_tts_quality.py): "Speak at a brisk natural pace in
# a casual, friendly tone" measured better on ALL metrics — pace 12.7 vs 11.7
# ch/s median (long texts 13-18, in the natural band), 0/12 EOS-miss (even
# fixed the short-text drag-out), tightest tails. BUT it shifts timbre
# slightly (f0 +5%, centroid +11% — brighter/faster). Ears decide: compare
# alice/tests/tts_samples/quality_win64_brisk_*.wav vs quality_win64_*.wav,
# then flip via ALICE_TTS_INSTRUCT or by editing the default here.
FIXED_INSTRUCT = os.environ.get(
    "ALICE_TTS_INSTRUCT",
    "Speak in a casual, friendly conversational tone",
)
# DECOUPLED: talker chooses content tokens (which codec word), subtalker
# chooses acoustic detail (pitch, breath, prosody) across the other 7 codebook
# layers. Lowering ONLY the subtalker constrains pitch/timbre variance across
# calls (voice drift between identical-input runs) without
# making word-level sampling sound robotic.
#
# Subtalker pacing trade-off (2026-05-26): subtalker_temp=0.4 locked voice
# consistency but produced over-articulated, ~70% pace delivery (~8.7
# chars/sec vs 13-15 natural). Bumped to 0.6 — gives prosody enough room
# to find a natural cadence while still suppressing turn-to-turn drift.
# If drift returns at 0.6, the move is reference-audio re-extraction, not
# dropping the temp back down.
TALKER_TEMPERATURE = 0.7
SUBTALKER_TEMPERATURE = 0.6
TOP_K = 50
REPETITION_PENALTY = 1.05
# Penalize only the last N talker tokens instead of the full history.
# Full-history penalty is a known stutter/looping mode: the usable
# first-codebook vocab is ~2048 tokens, so on long utterances most common
# tokens end up permanently penalized and sampling drifts onto rare tokens
# (phoneme repeats, missed EOS). 64 frames ≈ 5s — enough to do the penalty's
# real job (suppress local repeats) without starving the talker.
# ALICE_TTS_REP_WINDOW=0 restores full-history behavior.
REPETITION_PENALTY_WINDOW = int(os.environ.get("ALICE_TTS_REP_WINDOW", "64"))
VOICE_SEED = 42
EOS_TOKEN_IDS = [2150, 2157]


def _max_tokens_for_text(text: str) -> int:
    """12Hz codec = 12.5 frames/sec. 14 tokens/word ~= 1.1s/word with room for pauses."""
    return max(len(text.split()) * 14, 25)


def _max_frames_for_text(text: str) -> int:
    """For streaming path. Same logic."""
    return _max_tokens_for_text(text)


def _ensure_msvc_env():
    vcvarsall = r"C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvarsall.bat"
    if os.environ.get("_ALICE_MSVC_READY"):
        return True
    if not os.path.exists(vcvarsall):
        return False
    try:
        result = _subprocess.run(
            f'cmd /c "\"{vcvarsall}\" x64 && set"',
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


def _find_vb_cable():
    """Find VB-Cable WASAPI device for VTS lip sync."""
    try:
        import sounddevice as sd
        for i, d in enumerate(sd.query_devices()):
            if "CABLE Input" in d["name"] and d["max_output_channels"] > 0:
                if d["max_output_channels"] == 2:
                    return i, int(d["default_samplerate"])
        for i, d in enumerate(sd.query_devices()):
            if "CABLE Input" in d["name"] and d["max_output_channels"] > 0:
                return i, int(d["default_samplerate"])
    except Exception:
        pass
    return None, None


def _resample(pcm, from_sr, to_sr):
    if from_sr == to_sr:
        return pcm
    ratio = to_sr / from_sr
    new_len = int(len(pcm) * ratio)
    indices = np.arange(new_len) / ratio
    idx = np.minimum(indices.astype(np.int32), len(pcm) - 1)
    return pcm[idx]


def _patch_audio_encoder_attention():
    """Monkey-patch audio encoder to use optimized attention on target GPU architecture."""
    try:
        import torch
        if not torch.cuda.is_available():
            return
        props = torch.cuda.get_device_properties(0)
        if props.major < 12:
            return

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


def main():
    import torch

    # ---------- Load model ----------
    _log("Loading TTS model...")
    t0 = time.time()

    from tts_engine import TTSModel

    model = TTSModel.from_pretrained(
        MODEL_DIR,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    _patch_audio_encoder_attention()

    # Per-variant routing: "custom_voice" vs "base".
    # Drives quantization default + decoder path + ref_code. The base model
    # is too small to absorb int8 weight quantization or `decode_padded` without
    # quality loss.
    _mtype = getattr(model.model.config, "tts_model_type", None)
    is_custom_voice = (_mtype == "custom_voice")
    _log(f"tts_model_type={_mtype}, is_custom_voice={is_custom_voice}")

    # int8 default for both variants — confirmed clean on base once we stopped
    # stacking it with TurboQuant + decode_padded (the original "int8 destroys
    # base" claim was from that combo). Override with ALICE_TTS_QUANTIZE.
    _default_quant = "int8"
    _quant_mode = os.environ.get("ALICE_TTS_QUANTIZE", _default_quant).lower()
    if _quant_mode != "0":
        try:
            from weight_quantization import quantize_
            if _quant_mode == "int4":
                from weight_quantization import Int4WeightOnlyConfig
                quantize_(model.model, Int4WeightOnlyConfig(
                    group_size=32, int4_packing_format='plain', version=1,
                ))
                _log("int4 quantized")
            else:
                from weight_quantization import Int8WeightOnlyConfig
                quantize_(model.model, Int8WeightOnlyConfig())
                _log("int8 quantized")
        except Exception as e:
            _log(f"quantization failed, using bf16: {e}")
    else:
        _log("TTS loaded in bf16")
    model.model.talker.enable_fast_codebook_gen(True)

    # torch.compile
    compiled = False
    if _ensure_msvc_env():
        try:
            model.model.enable_streaming_optimizations(
                use_compile=True,
                use_cuda_graphs=False,
                compile_codebook_predictor=True,
                compile_talker=True,
            )
            compiled = True
            _log("torch.compile enabled")
        except Exception as e:
            _log(f"torch.compile failed: {e}")
    else:
        _log("MSVC not found, torch.compile disabled")

    # Load x-vector
    _log(f"xvector: {XVECTOR_PATH}")
    xvec = torch.load(XVECTOR_PATH, weights_only=True, map_location="cuda:0")

    # Decode routing — chunked_decode for BOTH variants.
    # decode_padded garbles articulation on both variants; chunked_decode handles
    # variable-length windows natively: no zero-pad → no Nyquist burst → no
    # ref_code prepend needed either (the prepend corrupts chunked output).
    # Set ALICE_TTS_PADDED_DECODE=1 to restore the old padded path.
    use_optimized_decode = (
        is_custom_voice and os.environ.get("ALICE_TTS_PADDED_DECODE") == "1"
    )
    ref_code = None
    if use_optimized_decode:
        _ref_pt = os.path.join(MODEL_DIR, "alice_ref_code.pt")
        if os.path.exists(_ref_pt):
            try:
                ref_code = torch.load(_ref_pt, weights_only=True, map_location="cuda:0")
                _log(f"decoder pad-ctx: loaded {_ref_pt}, shape={tuple(ref_code.shape)}")
            except Exception as e:
                _log(f"decoder pad-ctx: failed to load {_ref_pt}: {e}")
        else:
            _log(f"decoder pad-ctx: {_ref_pt} not found — first-chunk Nyquist burst will return")

    voice_clone_prompt = {
        "ref_code": [ref_code],
        "ref_spk_embedding": [xvec.unsqueeze(0) if xvec.dim() == 1 else xvec],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }

    _log(f"Model loaded in {time.time()-t0:.1f}s, compiled={compiled}")

    # ---------- Warmup ----------
    if compiled:
        import sounddevice as sd  # noqa — verify it loads
        warmup_texts = [
            "Hello.",
            "This is a longer warmup sentence to make sure the model "
            "compiles kernels for multi-sentence responses so there is "
            "no delay when speaking longer passages during conversation.",
        ]
        for i, text in enumerate(warmup_texts):
            t0 = time.time()
            input_ids = model._tokenize_texts([model._build_assistant_text(text)])
            instruct_ids = model._tokenize_texts([model._build_instruct_text(FIXED_INSTRUCT)])
            for _ in model.model.stream_generate_pcm(
                input_ids=input_ids,
                instruct_ids=instruct_ids,
                voice_clone_prompt=voice_clone_prompt,
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
                overlap_samples=0,  # vendor crossfade blends non-aligned samples → click at each boundary
                use_optimized_decode=use_optimized_decode,
                max_frames=_max_frames_for_text(text),
            ):
                pass
            _log(f"Warmup {i+1}/{len(warmup_texts)} done in {time.time()-t0:.1f}s")

    # ---------- Persistent playback infrastructure ----------
    # The playback thread lives for the entire subprocess lifetime. Every
    # `speak` enqueues audio chunks and returns as soon as GENERATION
    # completes — playback runs in the background. This lets the next
    # sentence's generation pipeline with the previous sentence's playback,
    # which is the whole point of sentence-streaming TTS. Without this,
    # speak() blocks on playback and there's no real concurrency.
    import sounddevice as sd
    import threading
    import queue

    audio_queue: "queue.Queue" = queue.Queue()
    drain_event = threading.Event()
    # Identity sentinels: string comparison broke when tag was a numpy array
    # (item[0] for audio chunks is the PCM array — `array == "str"` returns
    # element-wise array → ValueError on truthiness).
    DRAIN_MARKER = object()
    SHUTDOWN_MARKER = object()
    playback_state = {
        "stream": None,
        "vb_stream": None,
        "current_sr": None,
        "samples_buffered": 0,
        "t_first_write": None,
    }
    vb_device, vb_sr = _find_vb_cable()
    if vb_device is not None:
        _log(f"VB-Cable found (device {vb_device}, {vb_sr}Hz)")

    def _playback_loop():
        while True:
            try:
                item = audio_queue.get()
                if item is None:
                    continue
                # Identity sentinel checks first; audio chunks are tuples of
                # (numpy-array, int sample-rate) and won't match either.
                if item is SHUTDOWN_MARKER:
                    break
                if item is DRAIN_MARKER:
                    # PortAudio's device-side buffer holds writes for up to
                    # ~`blocksize / sr` seconds past when `stream.write()`
                    # returns. With `blocksize=sr` (1s blocks, see stream open
                    # below), our prior `samples_buffered / sr` math
                    # under-counted by nearly a full second — so drain returned
                    # while audio was still in the DAC queue, and that tail
                    # leaked into the START of the next speak()'s playback.
                    # `stream.stop()` waits for PortAudio's buffer to drain
                    # naturally before stopping — accurate, no math required.
                    # Restart immediately so the next batch can write.
                    stream = playback_state["stream"]
                    if stream is not None:
                        try:
                            stream.stop()
                            stream.start()
                        except Exception as e:
                            _log(f"drain stop/start failed: {e}")
                    # Reset for next batch
                    playback_state["samples_buffered"] = 0
                    playback_state["t_first_write"] = None
                    drain_event.set()
                    continue
                # Audio chunk
                pcm_f32, sr = item
                stream = playback_state["stream"]
                if stream is None or playback_state["current_sr"] != sr:
                    if stream is not None:
                        try:
                            stream.stop(); stream.close()
                        except Exception:
                            pass
                    if playback_state["vb_stream"] is not None:
                        try:
                            playback_state["vb_stream"].stop()
                            playback_state["vb_stream"].close()
                        except Exception:
                            pass
                        playback_state["vb_stream"] = None
                    stream = sd.OutputStream(
                        samplerate=sr, channels=1, dtype="float32", blocksize=sr,
                    )
                    stream.start()
                    playback_state["stream"] = stream
                    playback_state["current_sr"] = sr
                    if vb_device is not None:
                        vb_stream = sd.OutputStream(
                            samplerate=vb_sr, channels=1, dtype="float32",
                            blocksize=vb_sr, device=vb_device,
                        )
                        vb_stream.start()
                        playback_state["vb_stream"] = vb_stream
                if playback_state["t_first_write"] is None:
                    playback_state["t_first_write"] = time.perf_counter()
                stream.write(pcm_f32)
                playback_state["samples_buffered"] += len(pcm_f32)
                vb_stream = playback_state["vb_stream"]
                if vb_stream is not None:
                    try:
                        vb_data = _resample(pcm_f32, sr, vb_sr)
                        vb_stream.write(vb_data)
                    except Exception:
                        pass
            except Exception as e:
                # Never let one bad item kill the thread — log and continue.
                _log(f"Playback loop iteration error: {e}\n{traceback.format_exc()}")
                continue

    playback_thread = threading.Thread(target=_playback_loop, daemon=True)
    playback_thread.start()

    # ---------- Signal ready ----------
    _send({"status": "ready", "compiled": compiled})
    _log("Ready, waiting for commands...")

    # ---------- Command loop ----------
    while True:
        try:
            cmd = _recv()
            if cmd is None:
                _log("stdin closed, exiting")
                break

            if cmd.get("cmd") == "shutdown":
                _log("Shutdown requested")
                audio_queue.put(SHUTDOWN_MARKER)
                break

            if cmd.get("cmd") == "drain":
                drain_event.clear()
                audio_queue.put(DRAIN_MARKER)
                drain_event.wait(timeout=180)
                _send({"status": "done"})
                continue

            if cmd.get("cmd") == "speak":
                text = cmd["text"].strip()
                if not text:
                    _send({"status": "done"})
                    continue

                t0 = time.perf_counter()
                # `emotion` arg accepted for API compatibility, ignored —
                # we always pass FIXED_INSTRUCT (see top of file).

                # Fixed seed for consistent voice timbre
                torch.manual_seed(VOICE_SEED)
                torch.cuda.manual_seed(VOICE_SEED)

                # Tokenize
                input_ids = model._tokenize_texts([model._build_assistant_text(text)])
                instruct_ids = model._tokenize_texts([model._build_instruct_text(FIXED_INSTRUCT)])

                chunk_count = 0
                total_samples_this_speak = 0
                sr_out = 24000
                _all_chunks = []
                # Tail silence-trim: the talker emits ~7-14 trailing prosody
                # frames (quiet breath/decay codec tokens) before EOS, which
                # decode to a tail vowel ("bye" → "bye-e"). We keep a rolling
                # ~500ms window of audio out of the playback queue so we can
                # trim it after generation completes. Earlier audio streams
                # normally — TTFB unchanged.
                TAIL_BUFFER_MS = 500
                tail_buffer = np.zeros(0, dtype=np.float32)
                try:
                    for pcm_chunk, sr in model.model.stream_generate_pcm(
                        input_ids=input_ids,
                        instruct_ids=instruct_ids,
                        voice_clone_prompt=voice_clone_prompt,
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
                        overlap_samples=0,  # vendor crossfade blends non-aligned samples → click at boundaries
                        use_optimized_decode=use_optimized_decode,
                        max_frames=_max_frames_for_text(text),
                    ):
                        chunk_count += 1
                        total_samples_this_speak += len(pcm_chunk)
                        sr_out = sr
                        _all_chunks.append(pcm_chunk.copy())
                        # Append to rolling tail; flush everything except the
                        # last TAIL_BUFFER_MS to playback immediately.
                        tail_samples = int((TAIL_BUFFER_MS / 1000.0) * sr)
                        combined = np.concatenate([tail_buffer, pcm_chunk.astype(np.float32)])
                        if len(combined) > tail_samples:
                            send = combined[:-tail_samples]
                            tail_buffer = combined[-tail_samples:]
                            audio_queue.put((send.reshape(-1, 1), sr))
                        else:
                            tail_buffer = combined
                except Exception as e:
                    _log(f"Generation error: {e}\n{traceback.format_exc()}")

                # Trim trailing low-RMS samples from the held tail. Walk
                # backwards in 20ms windows; stop at the first window above
                # the silence threshold. Anything past that is decay/breath
                # the talker emitted before EOS — not actual phonation.
                if len(tail_buffer) > 0:
                    SILENCE_RMS = 0.01     # ~-40 dBFS; conservative, won't clip real speech
                    WINDOW_MS = 20
                    win = int((WINDOW_MS / 1000.0) * sr_out)
                    if win > 0 and len(tail_buffer) > win:
                        # Start a couple windows in from the end so we don't
                        # mis-trim a normal-energy final phoneme that happens
                        # to ramp down briefly.
                        last_speech_end = len(tail_buffer)
                        i = len(tail_buffer) - win
                        while i > 0:
                            block = tail_buffer[i:i + win]
                            rms = float(np.sqrt(np.mean(block * block)))
                            if rms >= SILENCE_RMS:
                                last_speech_end = i + win
                                break
                            i -= win
                        # Tiny 5ms linear fade-out so the cut isn't a click.
                        fade_samples = min(int(0.005 * sr_out), last_speech_end)
                        if fade_samples > 0:
                            ramp = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
                            tail_buffer[last_speech_end - fade_samples:last_speech_end] *= ramp
                        trimmed = tail_buffer[:last_speech_end]
                        if len(trimmed) > 0:
                            audio_queue.put((trimmed.reshape(-1, 1), sr_out))
                        trimmed_ms = (len(tail_buffer) - last_speech_end) / sr_out * 1000.0
                        if trimmed_ms > 5:
                            _log(f"Tail trim: removed {trimmed_ms:.0f}ms of trailing decay")
                    else:
                        audio_queue.put((tail_buffer.reshape(-1, 1), sr_out))

                gen_elapsed = time.perf_counter() - t0
                duration = total_samples_this_speak / sr_out
                gen_rtf = gen_elapsed / max(duration, 0.001)
                _log(f"Generated {chunk_count} chunks, {duration:.1f}s audio in {gen_elapsed:.1f}s (gen-RTF={gen_rtf:.2f}x); playback runs in background")

                # Save WAV for inspection (this speak's generated audio only)
                if _all_chunks:
                    try:
                        import wave as _wave
                        _wav_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "alice", "tests")
                        os.makedirs(_wav_dir, exist_ok=True)
                        _wav_path = os.path.join(_wav_dir, "tts_last_playback.wav")
                        _all = np.concatenate(_all_chunks)
                        _int16 = (np.clip(_all, -1, 1) * 32767).astype(np.int16)
                        with _wave.open(_wav_path, 'w') as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(sr_out)
                            wf.writeframes(_int16.tobytes())
                    except Exception:
                        pass

                # Return as soon as generation completes — playback is async.
                # The next speak() can fire immediately and overlap its
                # generation with this speak's still-ongoing playback.
                _send({"status": "done", "duration": duration, "wall": gen_elapsed})

        except Exception as e:
            _log(f"Command loop error: {e}\n{traceback.format_exc()}")
            _send({"status": "error", "msg": str(e)})

    _log("Subprocess exiting")


if __name__ == "__main__":
    main()
