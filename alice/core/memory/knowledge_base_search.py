"""
KnowledgeBase — static YAML knowledge files, embedded + FAISS-searchable.

Lives at `alice/core/memory/knowledge_base/` (the YAML files themselves).
This module wraps them in a queryable index so Alice can look things up
through IRIS rather than reading them raw.

Each YAML file is a list of `{name, facts: [...]}` entries. We embed each
entry (name + joined facts) as one doc, keep the source filename for
attribution, and serve cosine-nearest matches.

Lazy: index builds on first `.search()` call. Re-uses the shared MiniLM
embedding model so we don't load a second copy.

Usage via IRIS:
    iris = get_iris()
    hits = iris.search_knowledge_base("furina hydro fontaine", k=3)
    for h in hits:
        print(h["name"], "—", h["source"])
        print(h["facts"][:200])

Or directly:
    from alice.core.memory.knowledge_base_search import get_knowledge_base
    kb = get_knowledge_base()
    hits = kb.search("hololive myth members", k=5)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger("alice.memory.kb")

KB_DIR = Path(__file__).resolve().parent / "knowledge_base"


@dataclass
class KnowledgeEntry:
    name: str
    facts: str           # joined fact list, "\n- " separated
    source: str          # filename without extension
    fact_list: List[str] = field(default_factory=list)


def _parse_yaml_file(path: Path) -> List[KnowledgeEntry]:
    """Parse one YAML file into KnowledgeEntry list. Tolerant of malformed."""
    try:
        with path.open() as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logger.warning(f"kb: failed to parse {path.name}: {e}")
        return []
    if not isinstance(data, list):
        return []
    source = path.stem
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        facts = item.get("facts", [])
        if not name or not isinstance(facts, list):
            continue
        fact_strs = [str(f).strip() for f in facts if str(f).strip()]
        if not fact_strs:
            continue
        out.append(KnowledgeEntry(
            name=name,
            facts="\n- ".join(fact_strs),
            source=source,
            fact_list=fact_strs,
        ))
    return out


class KnowledgeBase:
    """
    Embedded, FAISS-indexed lookup over the YAML knowledge files.
    """

    def __init__(self, kb_dir: Path = KB_DIR):
        self.kb_dir = kb_dir
        self._entries: List[KnowledgeEntry] = []
        self._index = None
        self._embeddings = None
        self._embed_model = None
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.kb_dir.exists():
            logger.warning(f"kb: directory missing: {self.kb_dir}")
            self._loaded = True
            return
        for yml in sorted(self.kb_dir.glob("*.yaml")):
            self._entries.extend(_parse_yaml_file(yml))
        logger.info(f"kb: loaded {len(self._entries)} entries from {len(list(self.kb_dir.glob('*.yaml')))} files")
        self._loaded = True

    def _ensure_indexed(self) -> bool:
        """Build FAISS index lazily. Returns True if index is ready."""
        self._ensure_loaded()
        if self._index is not None:
            return True
        if not self._entries:
            return False
        try:
            from alice.core.utils.embedding_utils import get_shared_embedding_model
            import numpy as np
            import faiss
        except ImportError as e:
            logger.warning(f"kb: faiss/embeddings unavailable, search disabled: {e}")
            return False

        model = get_shared_embedding_model()
        if model is None:
            logger.warning("kb: shared embedding model unavailable")
            return False
        self._embed_model = model

        texts = [f"{e.name}\n{e.facts}" for e in self._entries]
        try:
            vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        except TypeError:
            # Older sentence-transformers without normalize_embeddings kwarg
            vecs = model.encode(texts, show_progress_bar=False)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            vecs = vecs / norms

        vecs = np.asarray(vecs, dtype="float32")
        dim = vecs.shape[1]
        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)
        self._embeddings = vecs
        logger.info(f"kb: indexed {len(self._entries)} entries, dim={dim}")
        return True

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Return top-k entries matching `query`. Each result is:
            {name, source, facts, fact_list, score}
        Empty list if KB is unavailable.
        """
        if not self._ensure_indexed():
            return []
        import numpy as np
        try:
            qv = self._embed_model.encode([query], normalize_embeddings=True, show_progress_bar=False)
        except TypeError:
            qv = self._embed_model.encode([query], show_progress_bar=False)
            qv = qv / max(float(np.linalg.norm(qv)), 1e-9)
        qv = np.asarray(qv, dtype="float32").reshape(1, -1)
        scores, idxs = self._index.search(qv, min(k, len(self._entries)))
        out = []
        for score, idx in zip(scores[0].tolist(), idxs[0].tolist()):
            if idx < 0:
                continue
            e = self._entries[idx]
            out.append({
                "name": e.name,
                "source": e.source,
                "facts": e.facts,
                "fact_list": e.fact_list,
                "score": float(score),
            })
        return out

    @property
    def size(self) -> int:
        self._ensure_loaded()
        return len(self._entries)


_singleton: Optional[KnowledgeBase] = None


def get_knowledge_base() -> KnowledgeBase:
    global _singleton
    if _singleton is None:
        _singleton = KnowledgeBase()
    return _singleton


__all__ = ["KnowledgeBase", "KnowledgeEntry", "get_knowledge_base", "KB_DIR"]
