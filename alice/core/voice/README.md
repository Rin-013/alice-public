# Voice System — TTS + STT

> **Status**: Production.

## Overview

Alice's voice system handles text-to-speech and speech-to-text. TTS runs in an isolated subprocess so it can maintain its own dependency versions while the rest of Alice runs on different library versions.

## Architecture

```
alice/core/voice/
├── clause_chunker.py    # text-side splitter: buffers tokens, emits at clause boundaries
├── speech_pipeline.py   # queue + worker thread; submit() non-blocking, wait_playback_done() syncs
├── tts.py               # in-process TTS (benches/probes; production uses subprocess)
├── tts_worker.py        # spawns the TTS subprocess, JSON-line stdin/stdout bridge
├── tts_subprocess.py    # runs inside isolated venv, owns the audio output stream
└── stt.py               # STT + VAD
```

## Real-time streaming pipeline

LLM tokens reach the speakers via a three-stage pipeline. Audio starts playing while the LLM is still generating.

```
chat.py LLM streaming loop
   │ _emit(token)
   ▼
ClauseChunker.feed(token)
   │   accumulates buffer, emits on clause boundaries
   ▼
SpeechPipeline.submit(chunk)
   │   non-blocking; enqueues into worker thread
   ▼
TTSWorker.speak(chunk) → subprocess → speakers
```

## STT

- VAD detects speech on audio chunks
- Two-stage endpointing: soft endpoint (eager transcribe), hard endpoint (deliver)
- Speech resuming between the two discards the eager transcript
- Result fires `on_transcription` callback

## Usage

Production (subprocess + real-time pipeline):

```python
from alice.core.voice.tts_worker import TTSWorker
from alice.core.voice.speech_pipeline import SpeechPipeline
from alice.core.voice.clause_chunker import ClauseChunker

tts = TTSWorker(device="cuda:0")
speech = SpeechPipeline(tts)
chunker = ClauseChunker()

for token in llm_token_stream:
    for chunk in chunker.feed(token):
        speech.submit(chunk)
final = chunker.flush()
if final:
    speech.submit(final)

speech.wait_playback_done()
speech.shutdown(); tts.shutdown()
```

## Env toggles

```
ALICE_TTS=0                  disable TTS entirely
ALICE_TTS_MODEL_DIR=...      override model directory
ALICE_TTS_NO_COMPILE=1       skip torch.compile
ALICE_DEBUG=1                write tts_debug.log
```
