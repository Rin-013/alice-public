# Alice Core Systems

> v4 Living Mind architecture — June 2026

## Active Modules

| Module | Path | Purpose |
|---|---|---|
| **Mind** | `mind/` | Background thinker — emotion tags, mood, direction tags, proposals, scheduler, inner voices |
| **Memory (IRIS)** | `memory/` | 3-tier recall, identity cartridge, depth layer (akashic + index), distiller, knowledge base, LinUCB bandit |
| **Fairy** | `fairy/` | TOS + security filter — 4 pure modules, input guard wired in chat.py |
| **Voice** | `voice/` | TTS (subprocess) + STT + ClauseChunker + SpeechPipeline |
| **Growth** | `growth/` | Experiential learning — XP, replay buffer, safety (behind `ALICE_GROWTH=1`) |
| System | `system/` | Registry, DI, env loader, system initializer |
| Optimization | `optimization/` | Custom attention + cache |
| Scripting | `scripting/` | `base_chat.txt` template + StateStorage |
| Utils | `utils/` | MiniLM embeddings, GLiNER NER, lightweight LLM, spaCy |

Support files: `cuda_streams.py` (per-system CUDA stream isolation), `config.py`.

## Architecture

All systems use **dependency injection** via `SystemRegistry`:

```python
from alice.core.system import get_registry
fairy = get_registry().get('fairy')   # never import singletons directly
```

See [`CLAUDE.md`](../../CLAUDE.md) for development rules.
