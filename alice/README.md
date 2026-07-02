# alice/

Alice's runtime package. Everything that runs when you `python chat.py`.

## Layout

```
alice/
├── core/               # All active systems
│   ├── mind/           # Background thinker (emotion tags, mood, proposals)
│   ├── memory/         # IRIS — recall, identity cartridge, depth layer, distiller
│   ├── fairy/          # TOS + security filter (4 modules)
│   ├── voice/          # TTS + STT + clause chunker + speech pipeline
│   ├── growth/         # Experiential learning (behind ALICE_GROWTH=1)
│   ├── system/         # Registry, DI, env, initializer
│   ├── optimization/   # Custom attention + cache
│   ├── scripting/      # Templates (base_chat.txt) + StateStorage
│   ├── utils/          # Embeddings, GLiNER, LLM, spaCy
│   ├── cuda_streams.py # Per-system CUDA stream isolation
│   └── config.py       # Config
└── data/               # Runtime data (databases, mood state, logs)
```

Entry point is `chat.py` in project root. See [`CLAUDE.md`](../CLAUDE.md) for the full development guide.
