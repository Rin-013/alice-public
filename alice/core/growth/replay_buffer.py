import json
import math
import random
from pathlib import Path
from datetime import datetime, timezone


class PrioritizedReplayBuffer:
    """Priority-weighted sampling over experiences.jsonl."""

    def __init__(self, experience_file=None):
        # type: (str | Path | None) -> None
        if experience_file is None:
            base = Path(__file__).resolve().parents[2]
            self.file_path = base / "data" / "growth" / "experiences.jsonl"
        else:
            self.file_path = Path(experience_file)

    def _load_all(self):
        # type: () -> list[dict]
        records = []
        try:
            with open(self.file_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict):
                            records.append(record)
                    except (json.JSONDecodeError, TypeError):
                        # skip malformed lines
                        continue
        except FileNotFoundError:
            pass
        return records

    def recompute_priority(self, record, now=None):
        # type: (dict, float | None) -> float
        # emotional intensity
        emo = record["emotion"]
        valence_abs = abs(emo["valence"])
        arousal = emo["arousal"]
        emotional_intensity = (valence_abs + arousal) / 2.0

        # salience & surprise with defaults
        salience = record.get("salience", 0.0)
        surprise = record.get("surprise", 0.0)

        base_priority = 0.3 * emotional_intensity + 0.25 * salience + 0.25 * surprise

        # recency decay
        timestamp_str = record["timestamp"]
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            # assume UTC if naïve
            dt = dt.replace(tzinfo=timezone.utc)
        record_timestamp = dt.timestamp()

        if now is None:
            now = datetime.now(timezone.utc).timestamp()

        age_hours = max(0.0, (now - record_timestamp) / 3600.0)
        recency = 0.5 ** (age_hours / 48.0)

        priority = base_priority + 0.2 * recency
        return max(0.01, min(1.0, priority))

    def sample(self, n=128, identity_ratio=0.2):
        # type: (int, float) -> list[dict]
        all_records = self._load_all()
        if not all_records:
            return []

        anchors = [r for r in all_records if r.get("identity_anchor")]
        rest = [r for r in all_records if not r.get("identity_anchor")]

        n_anchor = min(int(n * identity_ratio), len(anchors))
        n_rest = n - n_anchor

        # weighted sampling without replacement helper
        def weighted_sample(pool, priorities, k):
            if k >= len(pool):
                return pool[:]
            # copy to avoid mutating original
            pool = list(pool)
            priorities = list(priorities)
            result = []
            for _ in range(k):
                total = sum(priorities)
                if total <= 0:
                    # fallback uniform
                    idx = random.randrange(len(pool))
                    result.append(pool.pop(idx))
                    priorities.pop(idx)
                    continue
                r = random.random() * total
                cumulative = 0.0
                chosen = None
                for i, w in enumerate(priorities):
                    cumulative += w
                    if r < cumulative:
                        chosen = pool[i]
                        del pool[i], priorities[i]
                        break
                if chosen is not None:
                    result.append(chosen)
            return result

        # anchor sampling
        if anchors:
            anchor_prios = [self.recompute_priority(a) for a in anchors]
            sampled_anchors = weighted_sample(anchors, anchor_prios, n_anchor)
        else:
            sampled_anchors = []

        # rest sampling
        if rest:
            rest_prios = [self.recompute_priority(r) for r in rest]
            sampled_rest = weighted_sample(rest, rest_prios, n_rest)
        else:
            sampled_rest = []

        result = sampled_anchors + sampled_rest
        random.shuffle(result)
        return result

    def stats(self):
        # type: () -> dict
        records = self._load_all()
        if not records:
            return {
                "total": 0,
                "anchors": 0,
                "mean_priority": 0.0,
                "oldest": "",
                "newest": ""
            }

        total = len(records)
        anchors = sum(1 for r in records if r.get("identity_anchor"))
        priorities = [self.recompute_priority(r) for r in records]
        mean_priority = sum(priorities) / total if total else 0.0

        # timestamps are ISO strings – directly min/max works
        timestamps = [r["timestamp"] for r in records]
        oldest = min(timestamps)
        newest = max(timestamps)

        return {
            "total": total,
            "anchors": anchors,
            "mean_priority": mean_priority,
            "oldest": oldest,
            "newest": newest
        }
