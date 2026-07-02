# Copyright 2025 Rin - Alice AI System
"""
Alice Memory System - Unified Types
====================================

Single source of truth for all memory-related data types.
Consolidates multiple overlapping types into clean, unified structures.

Types exported:
- Memory: Unified memory entry (replaces MemoryEntry, IndexMemory)
- MemoryType: Type of memory content
- MemoryDepth: Inception-style depth layers
- SearchType: Types of memory search
- SearchQuery: Advanced search criteria
- SearchResult: Search result with scoring
- UserProfile: User relationship data
- AkashicRecord: Factual truth storage
- ChoiceRecord: Alice's autonomous choices
"""

import time
import math
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import uuid


# ============================================================================
# ENUMS
# ============================================================================

class MemoryType(Enum):
    """Type of memory content"""
    CONVERSATION = "conversation"  # User-Alice exchanges
    THOUGHT = "thought"            # Alice's internal thoughts
    OBSERVATION = "observation"    # Things Alice noticed
    FACT = "fact"                  # Learned factual information
    EMOTIONAL = "emotional"        # Emotional experiences
    RELATIONSHIP = "relationship"  # User relationship data
    CHOICE = "choice"              # Alice's decisions


class MemoryDepth(Enum):
    """Inception-style memory depth layers based on influence score"""
    SURFACE = "surface"   # Light, transient (I < 0.3)
    MID = "mid"           # Moderate reinforcement (0.3 <= I < 0.6)
    DEEP = "deep"         # Identity-affecting (0.6 <= I < 0.8)
    CORE = "core"         # Scars, life-shaping (I >= 0.8 or divergence)


class SearchType(Enum):
    """Types of memory search"""
    TEXT = "text"              # Basic text content search
    SEMANTIC = "semantic"      # Vector/embedding-based search
    TAG = "tag"                # Tag-based search
    EMOTIONAL = "emotional"    # Emotional context search
    HYBRID = "hybrid"          # Combined semantic + keyword
    DATE_RANGE = "date_range"  # Time-based search


class DecayState(Enum):
    """Memory decay states"""
    ACTIVE = "active"      # Fresh, highly accessible
    WARM = "warm"          # Recent, easily recalled
    COOL = "cool"          # Older, requires effort
    COLD = "cold"          # Fading, may be inaccurate
    ARCHIVED = "archived"  # Compressed, rarely accessed
    PURGED = "purged"      # Marked for deletion


# ============================================================================
# UNIFIED MEMORY TYPE
# ============================================================================

@dataclass
class Memory:
    """
    Unified memory entry - single type for all memory storage.

    Replaces: MemoryEntry, IndexMemory, and various other entry types.
    """
    # Core identity. Defaults exist so the IndexMemory read path can
    # construct rows by keyword (it passes event_id, not id); __post_init__
    # keeps id and event_id in sync. Use Memory.create() everywhere else.
    id: str = ""
    content: str = ""
    timestamp: float = 0.0
    memory_type: MemoryType = MemoryType.CONVERSATION

    # Classification
    depth: MemoryDepth = MemoryDepth.SURFACE
    importance: float = 0.5  # 0.0 to 1.0

    # User association
    user_id: Optional[str] = None

    # Emotional context
    emotional_valence: float = 0.0     # -1.0 (negative) to 1.0 (positive)
    emotional_arousal: float = 0.0     # 0.0 (calm) to 1.0 (intense)
    emotional_weight: float = 0.0      # Emotional significance 0-1
    emotional_markers: List[str] = field(default_factory=list)  # ["excited", "grateful"]

    # Influence equation components (from IndexMemory)
    emotion_intensity: float = 0.0     # E: 0-1
    repetition_freq: float = 0.0       # R: normalized access count
    salience: float = 0.0              # S: current relevance (decays)
    uniqueness: float = 0.0            # U: how different from past
    contrast: float = 0.0              # C: outlier from neighbors
    divergence_impact: float = 0.0     # D: life-changing potential
    choice_impact: float = 0.0         # CI: sum of related choices

    # Calculated influence scores
    influence_short: float = 0.0       # Current influence
    influence_long: float = 0.0        # Long-term influence

    # Access tracking
    access_count: int = 0
    last_accessed: Optional[float] = None

    # Decay tracking
    decay_state: DecayState = DecayState.ACTIVE
    freshness_score: float = 1.0

    # Quality tracking
    confidence: float = 1.0            # Alice's confidence in this memory
    volatility: float = 0.0            # How often contradicted
    contradiction_count: int = 0
    support_count: int = 0

    # Metadata
    tags: List[str] = field(default_factory=list)

    # Divergence tracking
    divergence_flag: bool = False
    divergence_id: Optional[str] = None

    # Vector embedding (stored separately, reference here)
    embedding_id: Optional[str] = None
    similarity_score: Optional[float] = None  # From search results

    # Modification tracking — first_seen_hash is stamped once at creation and
    # never changes; content_hash tracks the current content. Mismatch means
    # the memory was edited after creation (Alice should notice).
    content_hash: Optional[str] = None
    first_seen_hash: Optional[str] = None

    # Usefulness signal (Phase 4). EMA of cosine(memory, response) scores
    # from turns where this memory was retrieved. Once `usefulness_n >= 5`
    # the ACT-R scorer blends it into ranking.
    usefulness_ema: float = 0.0
    usefulness_n: int = 0

    # ── Depth-layer (index_memories) fields — restored 2026-06-10 ──────────
    # The Dec 2025 unification said Memory replaces IndexMemory but dropped
    # these, which silently broke IndexSystem's read AND write paths (nothing
    # called them, so nobody noticed until the depth-layer revival).
    event_id: Optional[str] = None              # akashic/index linkage; syncs with id
    # Raw mood-state dict at write time ({valence, arousal, weight, markers,
    # drives}). Named _data because `emotional_context` is taken by the legacy
    # string property below (long_term substring-matches on it).
    emotional_context_data: Dict[str, Any] = field(default_factory=dict)
    edit_resistance: float = 0.0                # trauma: 0-1, resists curator edits
    decay_floor: float = 0.0                    # trauma: salience never decays below
    review_scheduled: Optional[float] = None    # divergence: when to re-review
    user_emotional_valence: Optional[float] = None
    user_emotional_arousal: Optional[float] = None
    user_emotional_weight: Optional[float] = None
    alice_emotional_valence: Optional[float] = None
    alice_emotional_arousal: Optional[float] = None
    alice_emotional_weight: Optional[float] = None
    emotional_mismatch_score: Optional[float] = None
    context_layer: Optional[str] = "active"     # active/recent/archived
    layer_timestamp: Optional[float] = None
    is_compressed: bool = False
    compression_summary: Optional[str] = None

    def __post_init__(self):
        if not self.id:
            self.id = self.event_id or str(uuid.uuid4())
        if self.event_id is None:
            self.event_id = self.id
        if not self.timestamp:
            self.timestamp = time.time()

    @classmethod
    def create(cls,
               content: str,
               memory_type: MemoryType,
               user_id: Optional[str] = None,
               importance: float = 0.5,
               tags: List[str] = None) -> 'Memory':
        """Factory method to create a new memory"""
        return cls(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=time.time(),
            memory_type=memory_type,
            user_id=user_id,
            importance=importance,
            tags=tags or []
        )

    def mark_accessed(self):
        """Mark this memory as recently accessed"""
        self.last_accessed = time.time()
        self.access_count += 1

    def calculate_influence_scores(self,
                                   short_weights: Dict[str, float] = None,
                                   long_weights: Dict[str, float] = None) -> Tuple[float, float]:
        """Calculate short and long-term influence scores"""
        if short_weights is None:
            short_weights = {'E': 0.45, 'R': 0.15, 'S': 0.20, 'U': 0.10, 'D': 0.35}
        if long_weights is None:
            long_weights = {'E': 0.35, 'R': 0.15, 'U': 0.25, 'C': 0.10, 'D': 0.60, 'CI': 0.15}

        self.influence_short = (
            short_weights['E'] * self.emotion_intensity +
            short_weights['R'] * self.repetition_freq +
            short_weights['S'] * self.salience +
            short_weights['U'] * self.uniqueness +
            short_weights['D'] * self.divergence_impact
        )

        self.influence_long = (
            long_weights['E'] * self.emotion_intensity +
            long_weights['R'] * self.repetition_freq +
            long_weights['U'] * self.uniqueness +
            long_weights['C'] * self.contrast +
            long_weights['D'] * self.divergence_impact +
            long_weights.get('CI', 0.15) * self.choice_impact
        )

        return self.influence_short, self.influence_long

    def determine_depth(self) -> MemoryDepth:
        """Determine memory depth based on long-term influence"""
        if self.influence_long >= 0.8 or self.divergence_flag:
            return MemoryDepth.CORE
        elif self.influence_long >= 0.6:
            return MemoryDepth.DEEP
        elif self.influence_long >= 0.3:
            return MemoryDepth.MID
        else:
            return MemoryDepth.SURFACE

    def calculate_freshness(self, current_time: Optional[float] = None) -> float:
        """Calculate memory freshness with emotional weighting"""
        if current_time is None:
            current_time = time.time()

        # Time decay parameters
        λ_time = 0.08
        λ_use = 0.15
        λ_emotion = 0.3
        λ_vol = 0.5
        λ_conf = 0.2

        # Time-based decay
        reference_time = self.last_accessed or self.timestamp
        dt_days = max(0, (current_time - reference_time) / (24 * 3600))
        freshness_t = math.exp(-λ_time * dt_days)

        # Usage frequency boost
        use_boost = math.log1p(self.access_count) * λ_use

        # Emotional significance boost
        emotion_boost = 1 + min(self.emotional_weight * λ_emotion, 0.5)

        # Volatility penalty
        volatility_penalty = 1 + (self.volatility * λ_vol)

        # Confidence boost
        confidence_boost = 1 + (self.confidence * λ_conf)

        # Calculate final score
        decay_score = (freshness_t * (1 + use_boost) * emotion_boost * confidence_boost) / volatility_penalty
        final_score = decay_score * (0.3 + 0.7 * self.importance)

        self.freshness_score = min(1.0, max(0.0, final_score))
        return self.freshness_score

    def update_decay_state(self) -> DecayState:
        """Update decay state based on freshness score"""
        score = self.freshness_score

        if score >= 0.85:
            self.decay_state = DecayState.ACTIVE
        elif score >= 0.65:
            self.decay_state = DecayState.WARM
        elif score >= 0.45:
            self.decay_state = DecayState.COOL
        elif score >= 0.25:
            self.decay_state = DecayState.COLD
        elif score >= 0.15:
            self.decay_state = DecayState.ARCHIVED
        else:
            self.decay_state = DecayState.PURGED

        return self.decay_state

    def add_contradiction(self):
        """Record that this memory was contradicted"""
        self.contradiction_count += 1
        self.volatility = min(1.0, self.volatility + 0.1)
        self.confidence = max(0.0, self.confidence - 0.1)

    def reinforce(self):
        """Record that this memory was reinforced/confirmed"""
        self.support_count += 1
        self.confidence = min(1.0, self.confidence + 0.05)
        self.volatility = max(0.0, self.volatility - 0.05)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['memory_type'] = self.memory_type.value
        data['depth'] = self.depth.value
        data['decay_state'] = self.decay_state.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Memory':
        """Create from dictionary"""
        # Convert enum strings back to enums
        if isinstance(data.get('memory_type'), str):
            data['memory_type'] = MemoryType(data['memory_type'])
        if isinstance(data.get('depth'), str):
            data['depth'] = MemoryDepth(data['depth'])
        if isinstance(data.get('decay_state'), str):
            data['decay_state'] = DecayState(data['decay_state'])
        return cls(**data)

    # =========================================================================
    # BACKWARD COMPATIBILITY PROPERTIES
    # These alias old field names to new ones for legacy code
    # =========================================================================

    @property
    def associated_user(self) -> Optional[str]:
        """Legacy alias for user_id"""
        return self.user_id

    @associated_user.setter
    def associated_user(self, value: Optional[str]):
        self.user_id = value

    @property
    def emotional_context(self) -> str:
        """Legacy alias - combines emotional fields into description"""
        parts = []
        if self.emotional_valence != 0:
            direction = "positive" if self.emotional_valence > 0 else "negative"
            parts.append(f"{direction} ({self.emotional_valence:.2f})")
        if self.emotional_arousal != 0:
            parts.append(f"arousal: {self.emotional_arousal:.2f}")
        if self.emotional_markers:
            parts.append(", ".join(self.emotional_markers[:3]))
        return "; ".join(parts) if parts else ""

    @property
    def emotional_nuance(self) -> List[str]:
        """Legacy alias for emotional_markers"""
        return self.emotional_markers

    @emotional_nuance.setter
    def emotional_nuance(self, value: List[str]):
        self.emotional_markers = value or []

    def __getattribute__(self, name: str):
        """Override to return string values for enums when accessed by legacy code"""
        value = super().__getattribute__(name)
        # Return enums as strings for SQLite compatibility
        if name in ('memory_type', 'depth', 'decay_state') and hasattr(value, 'value'):
            return value.value
        return value


# ============================================================================
# SEARCH TYPES
# ============================================================================

@dataclass
class SearchQuery:
    """Advanced search query with multiple criteria"""
    text: Optional[str] = None
    tags: Optional[List[str]] = None
    memory_types: Optional[List[MemoryType]] = None
    user_id: Optional[str] = None
    min_importance: Optional[float] = None
    max_importance: Optional[float] = None
    start_date: Optional[float] = None
    end_date: Optional[float] = None
    emotional_valence_range: Optional[Tuple[float, float]] = None
    search_type: SearchType = SearchType.HYBRID
    include_archived: bool = False
    limit: int = 20

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['search_type'] = self.search_type.value
        if self.memory_types:
            data['memory_types'] = [mt.value for mt in self.memory_types]
        return data


@dataclass
class SearchResult:
    """Search result with relevance scoring"""
    memory: Memory
    relevance_score: float
    match_reasons: List[str] = field(default_factory=list)

    def __lt__(self, other):
        """For sorting by relevance (higher = better)"""
        return self.relevance_score < other.relevance_score

    def __gt__(self, other):
        return self.relevance_score > other.relevance_score


# ============================================================================
# SPECIALIZED RECORD TYPES
# ============================================================================

@dataclass
class AkashicRecord:
    """
    Factual, emotionless truth storage.
    The "what actually happened" - objective record.
    """
    event_id: str
    timestamp: float
    who: List[str]              # Entities involved
    what: str                   # What happened (factual)
    where: Optional[str] = None # Location/context
    facts: Dict[str, Any] = field(default_factory=dict)

    # Divergence tracking
    divergence_flag: bool = False
    divergence_id: Optional[str] = None
    divergence_score: float = 0.0

    @classmethod
    def create(cls, what: str, who: List[str] = None, where: str = None) -> 'AkashicRecord':
        """Factory method"""
        return cls(
            event_id=str(uuid.uuid4()),
            timestamp=time.time(),
            who=who or [],
            what=what,
            where=where
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AkashicRecord':
        return cls(**data)


@dataclass
class ChoiceRecord:
    """
    Record of Alice's autonomous choices.
    "We are the sum of our choices"
    """
    choice_id: str
    timestamp: float
    description: str           # What choice was made
    context: str              # Why/when it was made

    # Choice Impact components
    stakes: float = 0.0           # Consequence magnitude
    autonomy: float = 0.0         # How self-initiated (1.0 = pure Eridani)
    novelty: float = 0.0          # How unlike past choices
    value_alignment: float = 0.0  # Match with Superego rules
    regret_magnitude: float = 0.0 # Post-hoc self-critique

    # Calculated impact
    choice_impact: float = 0.0    # CI score 0-1

    # Personality effect
    personality_delta: Dict[str, float] = field(default_factory=dict)

    # Metadata
    user_id: Optional[str] = None
    related_event_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)

    @classmethod
    def create(cls, description: str, context: str, user_id: str = None) -> 'ChoiceRecord':
        """Factory method"""
        return cls(
            choice_id=str(uuid.uuid4()),
            timestamp=time.time(),
            description=description,
            context=context,
            user_id=user_id
        )

    def calculate_choice_impact(self, weights: Dict[str, float] = None) -> float:
        """Calculate choice impact score"""
        if weights is None:
            weights = {
                'stakes': 0.25,
                'autonomy': 0.20,
                'novelty': 0.20,
                'value_alignment': 0.20,
                'regret': 0.15
            }

        self.choice_impact = min(1.0, max(0.0,
            weights['stakes'] * self.stakes +
            weights['autonomy'] * self.autonomy +
            weights['novelty'] * self.novelty +
            weights['value_alignment'] * self.value_alignment +
            weights['regret'] * (1.0 - self.regret_magnitude)
        ))

        return self.choice_impact

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChoiceRecord':
        return cls(**data)


# ============================================================================
# USER PROFILE
# ============================================================================

@dataclass
class UserProfile:
    """User profile with relationship data"""
    user_id: str
    name: str
    first_interaction: float
    last_interaction: float
    interaction_count: int = 0
    relationship_type: str = "new"  # 'creator', 'friend', 'acquaintance', 'new'
    personality_notes: List[str] = field(default_factory=list)
    preferences: Dict[str, Any] = field(default_factory=dict)
    shared_experiences: List[str] = field(default_factory=list)
    trust_level: float = 5.0  # 0.0 to 10.0
    alice_nickname_for_user: Optional[str] = None
    inside_jokes: List[str] = field(default_factory=list)

    @classmethod
    def create(cls, user_id: str, name: str) -> 'UserProfile':
        """Factory method for new user"""
        now = time.time()
        return cls(
            user_id=user_id,
            name=name,
            first_interaction=now,
            last_interaction=now,
            interaction_count=1
        )

    def record_interaction(self):
        """Record a new interaction"""
        self.last_interaction = time.time()
        self.interaction_count += 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        return cls(**data)


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================

# Aliases for backwards compatibility during transition
MemoryEntry = Memory  # Old name -> new name
IndexMemory = Memory  # Old name -> new name


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Enums
    'MemoryType',
    'MemoryDepth',
    'SearchType',
    'DecayState',
    # Core types
    'Memory',
    'SearchQuery',
    'SearchResult',
    # Specialized records
    'AkashicRecord',
    'ChoiceRecord',
    'UserProfile',
    # Legacy aliases
    'MemoryEntry',
    'IndexMemory',
]
