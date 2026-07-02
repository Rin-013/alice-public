# Alice

Alice is a locally-run AI companion and streamer: a multi-model system with persistent memory, a background "inner mind," real-time voice, a safety filter, and an animated avatar — all running on a single consumer GPU.

A note on authorship: I designed the systems; Claude Code did most of the implementation. My Python was very limited when this started, and rather than wait until I knew everything, I built the thing and learned along the way. A year in, the main things I've learned are the architecture of modern transformer-based AI systems and how to iterate on a design until it survives contact with reality. This repository is the result of that year — building, testing, and making thousands of mistakes.

The early versions tried to fake cognition with regex and keyword matching. That failed, repeatedly, and the failures shaped the core design rule of the project: if it's a cognitive function — emotion, memory relevance, engagement — a model does it, not a heuristic. Alice settled into her current shape a few months ago: several small models, each with one job, coordinated by a handful of orchestration systems.

## Architecture

```
BACKGROUND:  Mind (smaller LLM) ─── emotion tags, curiosities, proposals ───► Buffer
                │ pauses during generation + speech                      │
                ▼                                                        ▼
FOREGROUND:  Alice (main LLM) ◄── context: emotion + memory + proposals ──► Fairy (safety)
                │                                                        │
                ▼                                                        ▼
             ClauseChunker ──► TTS ──► Speakers ──► Avatar
```

Mind runs continuously in the background — tagging emotions, pre-fetching memories, generating proposals. Alice sees what Mind produced and uses it or ignores it. Mind thinks; Alice decides.

## Stack

| Component | Footprint | Runtime |
|---|---|---|
| Alice (front LLM) | ~3.0 GB | GPU |
| Mind (background LLM) | ~0.8 GB | GPU |
| Voice (TTS) | ~2.8 GB | GPU subprocess |
| Hearing (STT) | ~0.14 GB | GPU |
| Memory (Iris) | — | CPU |
| Safety (Fairy) | — | CPU |
| Animation | VTube Studio | WebSocket |

Everything shares a single consumer GPU, with each model on its own dedicated CUDA stream.

## Key systems

**Iris** — Alice's memory. Three tiers (short-term list, session FAISS, long-term SQLite + FAISS), a LinUCB bandit that learns which retrieval strategies work, an identity cartridge that seeds who she is, and a distiller that extracts facts at end-of-session.

**Fairy** — Safety filter. TOS compliance + prompt-injection guard. Runs on every input and every streamed output chunk.

**Direction tags** — Mind writes a register tag that steers Alice's tone for a turn without scripting her words.

**Voice pipeline** — Streaming clause chunker feeds TTS as the LLM generates, targeting sub-second latency from the last spoken word to first audio.

**Growth** — Experiential learning from watching content. Behind a feature flag, still experimental.

## License

Source viewable for educational purposes. No commercial use, redistribution, or derivative works without explicit permission. Model weights, training data, voice assets, and identity materials (prompts and personality definitions) are not included.

See [LICENSE](LICENSE) for full terms.
