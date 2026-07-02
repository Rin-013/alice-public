#!/usr/bin/env python3
"""
Oracle - Alice's Context Watcher Service
Entity linking, salience scoring, and context prefetching for ultra-low latency

Named after the Oracle from The Matrix - sees all, knows all, predicts all.
Oracle watches every conversation, identifies important entities and relationships,
and prefetches relevant context before Alice even needs it:

- Entity detection: People, places, concepts, relationships
- Salience scoring: What's important in current context
- Context packs: Pre-assembled relevant information bundles
- Memory activation: Wake up relevant memories before they're needed
- Predictive loading: Anticipate what Alice will need to know

"I'd offer you a cookie, but you're not here for the cookie." - Oracle's omniscience
"""

import json
import time
import re
import sqlite3
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# Alice's frustration system for authentic reactions
try:
    from .alice_frustration import alice_curse, alice_random_curse, alice_recovery
except ImportError:
    def alice_curse(level="mild", context=None): return "damn"
    def alice_random_curse(intensity="mild"): return "shit"
    def alice_recovery(): return "*clears throat*"

@dataclass
class Entity:
    """A detected entity in conversation"""
    id: str                    # Unique entity identifier
    name: str                  # Entity name/label
    entity_type: str          # person, place, concept, event, object, etc.
    aliases: List[str]        # Alternative names/references
    
    # Context information
    first_mentioned: float    # When first seen
    last_mentioned: float     # Most recent mention
    mention_count: int        # How many times referenced
    salience_score: float     # Current importance (0-1)
    
    # Relationship information
    connected_entities: List[str]    # Other entities connected to this one
    relationship_types: Dict[str, str]  # entity_id -> relationship_type
    
    # Memory associations
    associated_memories: List[str]   # Memory IDs related to this entity
    emotional_valence: float         # How Alice feels about this entity (-1 to 1)
    
    # Metadata
    attributes: Dict[str, Any]       # Additional properties
    confidence: float                # How confident we are this is correct
    last_updated: float              # Last update timestamp

@dataclass
class ContextPack:
    """Pre-assembled context bundle for fast retrieval"""
    pack_id: str                     # Unique pack identifier
    primary_entities: List[str]      # Main entities this pack covers
    context_type: str               # conversation, technical, personal, etc.
    
    # Context content
    key_facts: List[str]            # Important facts to remember
    recent_interactions: List[Dict] # Recent relevant interactions
    relationship_summary: str      # How entities relate to each other
    alice_knowledge: List[str]      # What Alice knows about these entities
    
    # Activation information
    activation_score: float         # How relevant right now (0-1)
    last_accessed: float           # When last used
    access_count: int              # How many times accessed
    
    # Predictive information
    likely_questions: List[str]     # Questions user might ask
    suggested_responses: List[str]  # Potential Alice responses
    conversation_branches: List[str] # Likely conversation directions
    
    # Metadata
    created_time: float
    updated_time: float
    expires_time: Optional[float]   # When this context becomes stale

@dataclass
class SalienceEvent:
    """Event that affects entity salience scoring"""
    timestamp: float
    entity_id: str
    event_type: str        # mention, question, elaboration, connection, etc.
    salience_delta: float  # How much this affects salience
    context: str           # What happened
    decay_rate: float     # How fast this event's impact decays

class EntityDetector:
    """Detects and classifies entities in text"""
    
    def __init__(self):
        # Entity detection patterns
        self.entity_patterns = {
            "person": [
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b(?=\s+(?:said|told|asked|mentioned|thinks|believes))',
                r'\b(my|your|his|her|their)\s+(friend|mom|dad|brother|sister|boss|teacher|girlfriend|boyfriend)\b',
                r'\b([A-Z][a-z]+)\s+(?:is|was)\s+(?:a|an|the)\s+(?:person|guy|girl|man|woman)\b',
                r'@([a-zA-Z0-9_]+)',  # Social media handles
            ],
            
            "place": [
                r'\b(at|in|to|from|near)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:University|College|School|Hospital|Airport|Station)\b',
                r'\b(New York|Los Angeles|Chicago|Boston|Seattle|Portland|Austin|Denver|Atlanta)\b',
                r'\b([A-Z][a-z]+)\s+(?:City|State|County|Country)\b',
            ],
            
            "concept": [
                r'\b(artificial intelligence|machine learning|quantum computing|blockchain|cryptocurrency)\b',
                r'\b(programming|coding|software|hardware|technology|science|physics|chemistry|biology)\b',
                r'\b(philosophy|psychology|sociology|economics|politics|religion|spirituality)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:theory|concept|idea|principle|method|approach)\b',
            ],
            
            "event": [
                r'\b(the|that)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:conference|meeting|event|party|wedding|funeral)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:2019|2020|2021|2022|2023|2024|2025)\b',
                r'\b(?:when|during|after|before)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:happened|occurred)\b',
            ],
            
            "object": [
                r'\b(my|your|his|her|their|the|a|an)\s+(computer|laptop|phone|car|book|game|movie|song|album)\b',
                r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:version|model|edition|release)\b',
                r'\b(iPhone|Android|Windows|MacOS|Linux|PlayStation|Xbox|Nintendo)\b',
            ]
        }
        
        # Known entity aliases
        self.entity_aliases = {
            # Programming languages
            "python": ["py", "python3", "cpython"],
            "javascript": ["js", "node", "nodejs", "es6", "es2015"],
            "c++": ["cpp", "cxx", "c plus plus"],
            
            # Companies/Platforms  
            "google": ["alphabet", "search engine"],
            "microsoft": ["msft", "redmond"],
            "apple": ["cupertino", "mac", "ios"],
            "facebook": ["meta", "fb", "zuckerberg"],
            
            # Common names
            "alice": ["ai", "assistant", "bot"],
            "user": ["human", "person", "you"]
        }
        
        print("🔮 Oracle entity detector initialized")
        print(f"   Entity patterns: {sum(len(patterns) for patterns in self.entity_patterns.values())}")
        print(f"   Known aliases: {len(self.entity_aliases)}")
    
    def detect_entities(self, text: str, context: Dict[str, Any] = None) -> List[Entity]:
        """Detect entities in text and return Entity objects"""
        if context is None:
            context = {}
        
        detected = {}
        text_normalized = text.lower()
        
        # Apply patterns for each entity type
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract entity name (usually from first capture group)
                    if match.groups():
                        entity_name = match.group(1) if match.group(1) else match.group(0)
                    else:
                        entity_name = match.group(0)
                    
                    # Clean and normalize entity name
                    entity_name = entity_name.strip().lower()
                    if len(entity_name) < 2 or entity_name in ["the", "and", "or", "but", "is", "was", "are", "were"]:
                        continue
                    
                    # Create entity ID
                    entity_id = f"{entity_type}_{entity_name.replace(' ', '_')}"
                    
                    # Check for aliases
                    canonical_name = entity_name
                    for canonical, aliases in self.entity_aliases.items():
                        if entity_name in aliases or entity_name == canonical:
                            canonical_name = canonical
                            entity_id = f"{entity_type}_{canonical.replace(' ', '_')}"
                            break
                    
                    # Create or update entity
                    if entity_id in detected:
                        detected[entity_id].mention_count += 1
                        detected[entity_id].last_mentioned = time.time()
                    else:
                        detected[entity_id] = Entity(
                            id=entity_id,
                            name=canonical_name,
                            entity_type=entity_type,
                            aliases=self.entity_aliases.get(canonical_name, [entity_name]),
                            first_mentioned=time.time(),
                            last_mentioned=time.time(),
                            mention_count=1,
                            salience_score=0.5,  # Initial salience
                            connected_entities=[],
                            relationship_types={},
                            associated_memories=[],
                            emotional_valence=0.0,
                            attributes={},
                            confidence=0.8,  # Pattern-based detection confidence
                            last_updated=time.time()
                        )
        
        # Enhance entities with context
        entities = list(detected.values())
        self._enhance_entities_with_context(entities, text, context)
        
        return entities
    
    def _enhance_entities_with_context(self, entities: List[Entity], text: str, context: Dict[str, Any]):
        """Add contextual information to detected entities"""
        text_lower = text.lower()
        
        for entity in entities:
            # Detect emotional context
            entity_mentions = [mention for mention in [entity.name] + entity.aliases if mention in text_lower]
            
            for mention in entity_mentions:
                mention_pos = text_lower.find(mention)
                if mention_pos >= 0:
                    # Look at words around the mention
                    start = max(0, mention_pos - 50)
                    end = min(len(text_lower), mention_pos + len(mention) + 50)
                    context_window = text_lower[start:end]
                    
                    # Emotional valence detection
                    positive_words = ["love", "like", "great", "awesome", "amazing", "wonderful", "fantastic", "cool"]
                    negative_words = ["hate", "dislike", "terrible", "awful", "horrible", "sucks", "annoying", "frustrating"]
                    
                    positive_count = sum(1 for word in positive_words if word in context_window)
                    negative_count = sum(1 for word in negative_words if word in context_window)
                    
                    if positive_count > negative_count:
                        entity.emotional_valence = min(1.0, entity.emotional_valence + 0.2)
                    elif negative_count > positive_count:
                        entity.emotional_valence = max(-1.0, entity.emotional_valence - 0.2)
            
            # Add attributes based on entity type
            if entity.entity_type == "person":
                entity.attributes["is_human"] = True
                entity.attributes["relationship_depth"] = "mentioned"
            elif entity.entity_type == "place":
                entity.attributes["location_type"] = "physical"
            elif entity.entity_type == "concept":
                entity.attributes["complexity"] = "moderate"
                entity.attributes["domain"] = self._detect_domain(entity.name)

    def _detect_domain(self, entity_name: str) -> str:
        """Detect the domain/field an entity belongs to"""
        domain_keywords = {
            "technology": ["computer", "software", "programming", "ai", "algorithm", "code"],
            "science": ["physics", "chemistry", "biology", "quantum", "theory", "experiment"],
            "entertainment": ["movie", "music", "game", "show", "book", "art"],
            "personal": ["friend", "family", "relationship", "emotion", "feeling"],
            "business": ["company", "work", "job", "career", "money", "finance"]
        }
        
        entity_lower = entity_name.lower()
        for domain, keywords in domain_keywords.items():
            if any(keyword in entity_lower for keyword in keywords):
                return domain
        
        return "general"

class SalienceScorer:
    """Calculates and tracks entity salience scores"""
    
    def __init__(self):
        self.salience_events: List[SalienceEvent] = []
        self.base_decay_rate = 0.95  # How fast salience decays over time
        self.recency_weight = 0.3    # How much recent mentions matter
        self.frequency_weight = 0.4   # How much total mentions matter
        self.context_weight = 0.3    # How much current context matters
        
        print("🔮 Oracle salience scorer initialized")
    
    def calculate_salience(self, entity: Entity, current_context: Dict[str, Any] = None) -> float:
        """Calculate current salience score for an entity"""
        if current_context is None:
            current_context = {}
        
        current_time = time.time()
        
        # Recency component - how recently was this entity mentioned
        time_since_mention = current_time - entity.last_mentioned
        recency_score = max(0.0, 1.0 - (time_since_mention / 3600))  # Decay over 1 hour
        
        # Frequency component - how often is this entity mentioned
        frequency_score = min(1.0, entity.mention_count / 10.0)  # Cap at 10 mentions
        
        # Context component - how relevant is this entity to current conversation
        context_score = self._calculate_context_relevance(entity, current_context)
        
        # Weighted combination
        salience = (
            recency_score * self.recency_weight +
            frequency_score * self.frequency_weight +
            context_score * self.context_weight
        )
        
        # Apply decay from time passage
        time_decay = self.base_decay_rate ** (time_since_mention / 3600)
        salience *= time_decay
        
        # Boost for entities mentioned multiple times recently
        recent_events = [e for e in self.salience_events 
                        if e.entity_id == entity.id and current_time - e.timestamp < 600]  # 10 minutes
        if len(recent_events) > 1:
            salience *= 1.2  # 20% boost for recent repeated mentions
        
        entity.salience_score = max(0.0, min(1.0, salience))
        return entity.salience_score
    
    def _calculate_context_relevance(self, entity: Entity, context: Dict[str, Any]) -> float:
        """Calculate how relevant entity is to current context"""
        relevance = 0.5  # Base relevance
        
        # Check if entity relates to current conversation topic
        current_topic = context.get("topic", "").lower()
        if current_topic:
            if entity.name in current_topic or any(alias in current_topic for alias in entity.aliases):
                relevance += 0.3
            
            # Domain matching
            entity_domain = entity.attributes.get("domain", "general")
            if entity_domain in current_topic:
                relevance += 0.2
        
        # Check if user is asking about this entity
        user_questions = context.get("recent_questions", [])
        for question in user_questions:
            if entity.name in question.lower() or any(alias in question.lower() for alias in entity.aliases):
                relevance += 0.4
                break
        
        # Check emotional context alignment
        conversation_mood = context.get("mood", "neutral")
        if conversation_mood == "positive" and entity.emotional_valence > 0:
            relevance += 0.1
        elif conversation_mood == "negative" and entity.emotional_valence < 0:
            relevance += 0.1
        
        return min(1.0, relevance)
    
    def record_salience_event(self, entity_id: str, event_type: str, context: str, salience_delta: float = 0.1):
        """Record an event that affects entity salience"""
        event = SalienceEvent(
            timestamp=time.time(),
            entity_id=entity_id,
            event_type=event_type,
            salience_delta=salience_delta,
            context=context,
            decay_rate=0.98  # How fast this event's impact decays
        )
        
        self.salience_events.append(event)
        
        # Keep only recent events (last 24 hours)
        cutoff_time = time.time() - 24*3600
        self.salience_events = [e for e in self.salience_events if e.timestamp > cutoff_time]
    
    def get_top_salient_entities(self, entities: List[Entity], limit: int = 10) -> List[Entity]:
        """Get most salient entities, sorted by salience score"""
        # Update salience scores
        for entity in entities:
            self.calculate_salience(entity)
        
        # Sort by salience score
        sorted_entities = sorted(entities, key=lambda e: e.salience_score, reverse=True)
        return sorted_entities[:limit]

class ContextPackManager:
    """Manages pre-assembled context packs for fast retrieval"""
    
    def __init__(self):
        self.context_packs: Dict[str, ContextPack] = {}
        self.pack_templates = {
            "person_introduction": {
                "key_facts": ["name", "relationship", "recent_interactions"],
                "likely_questions": ["How do you know them?", "What are they like?", "When did you last talk?"],
                "context_type": "personal"
            },
            "technical_discussion": {
                "key_facts": ["technology_overview", "current_trends", "alice_knowledge_level"],
                "likely_questions": ["How does it work?", "What are the advantages?", "Is it difficult to learn?"],
                "context_type": "technical"
            },
            "entertainment_topic": {
                "key_facts": ["genre", "popularity", "alice_opinion", "related_content"],
                "likely_questions": ["Have you seen/played/heard it?", "What did you think?", "Any recommendations?"],
                "context_type": "entertainment"
            }
        }
        
        print("🔮 Oracle context pack manager initialized")
    
    def create_context_pack(self, primary_entities: List[Entity], context_type: str = "general") -> ContextPack:
        """Create a context pack for given entities"""
        pack_id = f"pack_{int(time.time())}_{len(self.context_packs)}"
        
        # Determine pack template
        template = self.pack_templates.get(context_type, {})
        
        # Extract key facts from entities
        key_facts = []
        for entity in primary_entities:
            facts = [
                f"{entity.name} is a {entity.entity_type}",
                f"Mentioned {entity.mention_count} times",
                f"Emotional valence: {entity.emotional_valence:.2f}"
            ]
            if entity.attributes:
                for key, value in entity.attributes.items():
                    facts.append(f"{key}: {value}")
            key_facts.extend(facts)
        
        # Generate relationship summary
        relationships = []
        for entity in primary_entities:
            if entity.connected_entities:
                for connected_id in entity.connected_entities:
                    rel_type = entity.relationship_types.get(connected_id, "related to")
                    relationships.append(f"{entity.name} {rel_type} {connected_id}")
        
        relationship_summary = "; ".join(relationships) if relationships else "No known relationships"
        
        # Generate likely questions based on entity types
        likely_questions = template.get("likely_questions", [])
        for entity in primary_entities:
            if entity.entity_type == "person":
                likely_questions.extend([
                    f"Tell me about {entity.name}",
                    f"How do you know {entity.name}?",
                    f"What's {entity.name} like?"
                ])
            elif entity.entity_type == "concept":
                likely_questions.extend([
                    f"Explain {entity.name}",
                    f"How does {entity.name} work?",
                    f"What's your opinion on {entity.name}?"
                ])
        
        # Generate Alice knowledge summary
        alice_knowledge = []
        for entity in primary_entities:
            if entity.emotional_valence > 0.2:
                alice_knowledge.append(f"Alice has positive feelings about {entity.name}")
            elif entity.emotional_valence < -0.2:
                alice_knowledge.append(f"Alice has concerns about {entity.name}")
            
            if entity.mention_count > 3:
                alice_knowledge.append(f"Alice is very familiar with {entity.name}")
        
        # Create context pack
        pack = ContextPack(
            pack_id=pack_id,
            primary_entities=[e.id for e in primary_entities],
            context_type=context_type,
            key_facts=key_facts,
            recent_interactions=[],  # Would be filled from conversation history
            relationship_summary=relationship_summary,
            alice_knowledge=alice_knowledge,
            activation_score=sum(e.salience_score for e in primary_entities) / len(primary_entities),
            last_accessed=0,
            access_count=0,
            likely_questions=list(set(likely_questions)),  # Remove duplicates
            suggested_responses=[],  # Would be generated based on Alice's personality
            conversation_branches=[],  # Would be predicted based on context
            created_time=time.time(),
            updated_time=time.time(),
            expires_time=None
        )
        
        self.context_packs[pack_id] = pack
        return pack
    
    def get_relevant_packs(self, entities: List[Entity], limit: int = 5) -> List[ContextPack]:
        """Get context packs relevant to current entities"""
        entity_ids = set(e.id for e in entities)
        relevant_packs = []
        
        for pack in self.context_packs.values():
            # Check overlap with current entities
            pack_entity_ids = set(pack.primary_entities)
            overlap = len(entity_ids & pack_entity_ids)
            
            if overlap > 0:
                # Calculate relevance score
                relevance = overlap / len(pack_entity_ids)
                relevance *= pack.activation_score
                
                # Boost recently accessed packs
                time_since_access = time.time() - pack.last_accessed
                if time_since_access < 3600:  # Last hour
                    relevance *= 1.2
                
                relevant_packs.append((pack, relevance))
        
        # Sort by relevance and return top packs
        relevant_packs.sort(key=lambda x: x[1], reverse=True)
        return [pack for pack, _ in relevant_packs[:limit]]
    
    def access_pack(self, pack_id: str) -> Optional[ContextPack]:
        """Access a context pack and update access statistics"""
        if pack_id not in self.context_packs:
            return None
        
        pack = self.context_packs[pack_id]
        pack.last_accessed = time.time()
        pack.access_count += 1
        
        return pack
    
    def cleanup_expired_packs(self):
        """Remove expired or stale context packs"""
        current_time = time.time()
        to_remove = []
        
        for pack_id, pack in self.context_packs.items():
            # Remove if explicitly expired
            if pack.expires_time and current_time > pack.expires_time:
                to_remove.append(pack_id)
                continue
            
            # Remove if not accessed in 24 hours and low activation
            time_since_access = current_time - pack.last_accessed
            if time_since_access > 24*3600 and pack.activation_score < 0.3:
                to_remove.append(pack_id)
                continue
        
        for pack_id in to_remove:
            del self.context_packs[pack_id]
        
        if to_remove:
            print(f"🔮 Oracle cleaned up {len(to_remove)} expired context packs")

class Oracle:
    """
    Alice's omniscient context watcher service
    
    Sees all, knows all, predicts what Alice will need before she needs it
    """
    
    def __init__(self, db_path: str = "oracle_knowledge.db"):
        self.db_path = Path(db_path)
        
        # Initialize components
        self.entity_detector = EntityDetector()
        self.salience_scorer = SalienceScorer()
        self.context_manager = ContextPackManager()
        
        # Entity and context storage
        self.entities: Dict[str, Entity] = {}
        self.conversation_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.predictions_made = 0
        self.cache_hits = 0
        self.context_packs_created = 0
        
        # Initialize database
        self._init_database()
        
        # Load existing entities
        self._load_entities()
        
        print("🔮👁️ Oracle Context Watcher initialized")
        print(f"   Known entities: {len(self.entities)}")
        print(f"   Active context packs: {len(self.context_manager.context_packs)}")
        print("   Watching all conversations with omniscient awareness")
    
    def _init_database(self):
        """Initialize SQLite database for persistent storage"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS entities (
                        id TEXT PRIMARY KEY,
                        name TEXT,
                        entity_type TEXT,
                        aliases TEXT,
                        first_mentioned REAL,
                        last_mentioned REAL,
                        mention_count INTEGER,
                        salience_score REAL,
                        connected_entities TEXT,
                        relationship_types TEXT,
                        associated_memories TEXT,
                        emotional_valence REAL,
                        attributes TEXT,
                        confidence REAL,
                        last_updated REAL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp REAL,
                        speaker TEXT,
                        content TEXT,
                        detected_entities TEXT,
                        context_used TEXT
                    )
                """)
                
                conn.commit()
                
        except Exception as e:
            print(f"🔮 Warning: Database init error: {e}")
    
    def _load_entities(self):
        """Load entities from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT * FROM entities")
                for row in cursor.fetchall():
                    entity = self._row_to_entity(row)
                    self.entities[entity.id] = entity
                    
        except Exception as e:
            print(f"🔮 {alice_curse('annoyed')} Error loading entities: {e}")
    
    def observe_conversation(self, speaker: str, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Oracle observes a piece of conversation and extracts context
        
        This is the main entry point - call this for every message
        """
        if context is None:
            context = {}
        
        observation_start = time.time()
        
        # Detect entities in the content
        detected_entities = self.entity_detector.detect_entities(content, context)
        
        # Update or create entities
        updated_entities = []
        for entity in detected_entities:
            if entity.id in self.entities:
                # Update existing entity
                existing = self.entities[entity.id]
                existing.mention_count += 1
                existing.last_mentioned = entity.last_mentioned
                existing.salience_score = self.salience_scorer.calculate_salience(existing, context)
                updated_entities.append(existing)
            else:
                # New entity
                entity.salience_score = self.salience_scorer.calculate_salience(entity, context)
                self.entities[entity.id] = entity
                updated_entities.append(entity)
        
        # Record salience events
        for entity in updated_entities:
            event_type = "mention" if entity.mention_count == 1 else "repeat_mention"
            self.salience_scorer.record_salience_event(
                entity.id, 
                event_type, 
                f"{speaker}: {content[:50]}...",
                0.1 if event_type == "mention" else 0.05
            )
        
        # Get most salient entities
        top_entities = self.salience_scorer.get_top_salient_entities(
            list(self.entities.values()), 
            limit=10
        )
        
        # Create or update context packs
        if len(updated_entities) > 0:
            # Determine context type
            context_type = self._classify_conversation_context(content, updated_entities)
            
            # Create context pack for this conversation segment
            context_pack = self.context_manager.create_context_pack(updated_entities, context_type)
            self.context_packs_created += 1
        
        # Get relevant existing context packs
        relevant_packs = self.context_manager.get_relevant_packs(top_entities, limit=3)
        
        # Record conversation
        conversation_entry = {
            "timestamp": observation_start,
            "speaker": speaker,
            "content": content,
            "detected_entities": [e.id for e in detected_entities],
            "context_used": [pack.pack_id for pack in relevant_packs]
        }
        self.conversation_history.append(conversation_entry)
        
        # Save to database
        self._save_conversation(conversation_entry)
        for entity in updated_entities:
            self._save_entity(entity)
        
        # Generate predictions for Alice
        predictions = self._generate_predictions(top_entities, content, context)
        self.predictions_made += 1
        
        observation_time = (time.time() - observation_start) * 1000
        
        # Return observation results
        return {
            "observation_time_ms": observation_time,
            "entities_detected": len(detected_entities),
            "entities_updated": len(updated_entities),
            "top_salient_entities": [{"id": e.id, "name": e.name, "salience": e.salience_score} for e in top_entities[:5]],
            "relevant_context_packs": len(relevant_packs),
            "context_pack_created": len(updated_entities) > 0,
            "predictions": predictions,
            "cache_performance": {
                "cache_hits": self.cache_hits,
                "total_requests": self.predictions_made,
                "hit_rate": self.cache_hits / max(1, self.predictions_made)
            }
        }
    
    def _classify_conversation_context(self, content: str, entities: List[Entity]) -> str:
        """Classify the type of conversation context"""
        content_lower = content.lower()
        
        # Check for question patterns
        if any(word in content_lower for word in ["what", "how", "why", "when", "where", "who"]):
            return "question"
        
        # Check entity types
        entity_types = [e.entity_type for e in entities]
        if "person" in entity_types:
            return "personal"
        elif "concept" in entity_types:
            return "technical"
        elif "place" in entity_types:
            return "travel"
        
        # Check for emotional content
        if any(word in content_lower for word in ["love", "hate", "excited", "frustrated", "happy", "sad"]):
            return "emotional"
        
        return "general"
    
    def _generate_predictions(self, entities: List[Entity], content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions about what Alice might need"""
        predictions = {
            "likely_followup_questions": [],
            "suggested_alice_responses": [],
            "context_to_prefetch": [],
            "memory_activation_suggestions": []
        }
        
        # Analyze content for likely follow-ups
        content_lower = content.lower()
        
        # Question predictions
        if "what is" in content_lower:
            for entity in entities:
                if entity.entity_type == "concept":
                    predictions["likely_followup_questions"].extend([
                        f"How does {entity.name} work?",
                        f"Why is {entity.name} important?",
                        f"Can you give an example of {entity.name}?"
                    ])
        
        if "tell me about" in content_lower:
            for entity in entities:
                predictions["likely_followup_questions"].extend([
                    f"What's interesting about {entity.name}?",
                    f"Have you used {entity.name}?",
                    f"What do you think of {entity.name}?"
                ])
        
        # Alice response suggestions based on entities
        for entity in entities:
            if entity.emotional_valence > 0.3:
                predictions["suggested_alice_responses"].append(f"Express enthusiasm about {entity.name}")
            elif entity.emotional_valence < -0.3:
                predictions["suggested_alice_responses"].append(f"Express concerns about {entity.name}")
            
            if entity.mention_count > 5:
                predictions["suggested_alice_responses"].append(f"Reference previous discussions about {entity.name}")
        
        # Context prefetching suggestions
        for entity in entities:
            if entity.salience_score > 0.7:
                predictions["context_to_prefetch"].append(f"Deep context for {entity.name}")
                predictions["memory_activation_suggestions"].append(f"Activate memories related to {entity.name}")
        
        return predictions
    
    def get_context_for_response(self, entities_mentioned: List[str]) -> Dict[str, Any]:
        """Get assembled context for Alice's response generation"""
        relevant_entities = []
        for entity_id in entities_mentioned:
            if entity_id in self.entities:
                relevant_entities.append(self.entities[entity_id])
        
        # Get relevant context packs
        relevant_packs = self.context_manager.get_relevant_packs(relevant_entities, limit=3)
        
        # Assemble context bundle
        context_bundle = {
            "entities": {e.id: asdict(e) for e in relevant_entities},
            "context_packs": [asdict(pack) for pack in relevant_packs],
            "salience_ranking": sorted(relevant_entities, key=lambda e: e.salience_score, reverse=True),
            "relationship_map": self._build_relationship_map(relevant_entities),
            "emotional_context": {e.id: e.emotional_valence for e in relevant_entities},
            "alice_knowledge_level": {e.id: self._assess_alice_knowledge(e) for e in relevant_entities}
        }
        
        # Update access statistics
        for pack in relevant_packs:
            self.context_manager.access_pack(pack.pack_id)
            self.cache_hits += 1
        
        return context_bundle
    
    def _build_relationship_map(self, entities: List[Entity]) -> Dict[str, Dict[str, str]]:
        """Build map of relationships between entities"""
        relationship_map = {}
        
        for entity in entities:
            relationships = {}
            for connected_id in entity.connected_entities:
                if connected_id in self.entities:
                    rel_type = entity.relationship_types.get(connected_id, "related")
                    relationships[connected_id] = rel_type
            
            if relationships:
                relationship_map[entity.id] = relationships
        
        return relationship_map
    
    def _assess_alice_knowledge(self, entity: Entity) -> str:
        """Assess Alice's knowledge level about an entity"""
        if entity.mention_count > 10:
            return "expert"
        elif entity.mention_count > 5:
            return "familiar"
        elif entity.mention_count > 1:
            return "basic"
        else:
            return "minimal"
    
    def _save_entity(self, entity: Entity):
        """Save entity to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO entities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entity.id,
                    entity.name,
                    entity.entity_type,
                    json.dumps(entity.aliases),
                    entity.first_mentioned,
                    entity.last_mentioned,
                    entity.mention_count,
                    entity.salience_score,
                    json.dumps(entity.connected_entities),
                    json.dumps(entity.relationship_types),
                    json.dumps(entity.associated_memories),
                    entity.emotional_valence,
                    json.dumps(entity.attributes),
                    entity.confidence,
                    entity.last_updated
                ))
                conn.commit()
        except Exception as e:
            print(f"🔮 {alice_curse('frustrated')} Entity save error: {e}")
    
    def _save_conversation(self, conversation: Dict[str, Any]):
        """Save conversation entry to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO conversations (timestamp, speaker, content, detected_entities, context_used)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    conversation["timestamp"],
                    conversation["speaker"],
                    conversation["content"],
                    json.dumps(conversation["detected_entities"]),
                    json.dumps(conversation["context_used"])
                ))
                conn.commit()
        except Exception as e:
            print(f"🔮 {alice_curse('annoyed')} Conversation save error: {e}")
    
    def _row_to_entity(self, row) -> Entity:
        """Convert database row to Entity object"""
        return Entity(
            id=row[0],
            name=row[1],
            entity_type=row[2],
            aliases=json.loads(row[3] or "[]"),
            first_mentioned=row[4],
            last_mentioned=row[5],
            mention_count=row[6],
            salience_score=row[7],
            connected_entities=json.loads(row[8] or "[]"),
            relationship_types=json.loads(row[9] or "{}"),
            associated_memories=json.loads(row[10] or "[]"),
            emotional_valence=row[11],
            attributes=json.loads(row[12] or "{}"),
            confidence=row[13],
            last_updated=row[14]
        )
    
    def get_oracle_stats(self) -> Dict[str, Any]:
        """Get Oracle performance and knowledge statistics"""
        return {
            "total_entities": len(self.entities),
            "conversations_observed": len(self.conversation_history),
            "context_packs_created": self.context_packs_created,
            "predictions_made": self.predictions_made,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": self.cache_hits / max(1, self.predictions_made),
            "entity_type_distribution": self._get_entity_type_distribution(),
            "top_salient_entities": self._get_top_entities_summary(10),
            "recent_activity": len([c for c in self.conversation_history if time.time() - c["timestamp"] < 3600]),
            "average_entities_per_conversation": len(self.entities) / max(1, len(self.conversation_history))
        }
    
    def _get_entity_type_distribution(self) -> Dict[str, int]:
        """Get distribution of entity types"""
        distribution = {}
        for entity in self.entities.values():
            entity_type = entity.entity_type
            distribution[entity_type] = distribution.get(entity_type, 0) + 1
        return distribution
    
    def _get_top_entities_summary(self, limit: int) -> List[Dict[str, Any]]:
        """Get summary of top entities by salience"""
        top_entities = sorted(self.entities.values(), key=lambda e: e.salience_score, reverse=True)[:limit]
        return [
            {
                "name": entity.name,
                "type": entity.entity_type,
                "salience": entity.salience_score,
                "mentions": entity.mention_count,
                "emotional_valence": entity.emotional_valence
            }
            for entity in top_entities
        ]
    
    def cleanup_stale_data(self):
        """Clean up stale entities and context packs"""
        current_time = time.time()
        stale_entities = []
        
        # Mark entities as stale if not mentioned in 30 days and low salience
        for entity_id, entity in self.entities.items():
            time_since_mention = current_time - entity.last_mentioned
            if time_since_mention > 30*24*3600 and entity.salience_score < 0.1:
                stale_entities.append(entity_id)
        
        # Remove stale entities
        for entity_id in stale_entities:
            del self.entities[entity_id]
        
        # Clean up context packs
        self.context_manager.cleanup_expired_packs()
        
        if stale_entities:
            print(f"🔮 Oracle cleaned up {len(stale_entities)} stale entities")

# Global Oracle instance
oracle = None

def initialize_oracle(db_path: str = "oracle_knowledge.db") -> Oracle:
    """Initialize global Oracle instance"""
    global oracle
    if oracle is None:
        oracle = Oracle(db_path)
    return oracle

def observe_message(speaker: str, content: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Oracle observes a message and extracts context"""
    if oracle is None:
        initialize_oracle()
    return oracle.observe_conversation(speaker, content, context)

def get_response_context(entities: List[str]) -> Dict[str, Any]:
    """Get context for Alice's response"""
    if oracle is None:
        initialize_oracle()
    return oracle.get_context_for_response(entities)

def get_oracle_stats() -> Dict[str, Any]:
    """Get Oracle statistics"""
    if oracle is None:
        return {"error": "Oracle not initialized"}
    return oracle.get_oracle_stats()

if __name__ == "__main__":
    print("🔮👁️ Testing Oracle Context Watcher")
    print("=" * 60)
    
    # Initialize Oracle
    oracle_system = Oracle("test_oracle.db")
    
    # Simulate conversation
    print("\n🔮 Simulating conversation observation:")
    
    test_messages = [
        ("User", "Hi Alice! I've been learning about machine learning recently."),
        ("Alice", "That's awesome! Machine learning is fascinating. What specifically interests you?"),
        ("User", "I'm particularly interested in neural networks and deep learning. My friend John recommended I start with Python."),
        ("Alice", "John has good taste! Python is perfect for ML. Have you looked at TensorFlow or PyTorch?"),
        ("User", "Not yet, but I've heard TensorFlow is good for beginners. What do you think?"),
        ("Alice", "TensorFlow is solid, but honestly PyTorch feels more intuitive to me. What's your background?"),
        ("User", "I work at Google, actually. We use TensorFlow a lot there."),
        ("Alice", "Oh damn, that's cool! Google's ML work is incredible. You must see some amazing stuff.")
    ]
    
    for speaker, message in test_messages:
        print(f"\n--- {speaker}: \"{message[:50]}{'...' if len(message) > 50 else ''}\" ---")
        
        result = oracle_system.observe_conversation(speaker, message)
        
        print(f"🔮 Observation time: {result['observation_time_ms']:.1f}ms")
        print(f"   Entities detected: {result['entities_detected']}")
        print(f"   Context packs: {result['relevant_context_packs']}")
        
        if result['top_salient_entities']:
            print("   Top entities:")
            for entity in result['top_salient_entities']:
                print(f"     - {entity['name']} ({entity['salience']:.3f})")
        
        if result['predictions']['likely_followup_questions']:
            print("   Predictions:", result['predictions']['likely_followup_questions'][0])
    
    # Test context retrieval
    print(f"\n🔮 Testing context retrieval:")
    entities_of_interest = ["concept_machine_learning", "person_john", "concept_tensorflow"]
    context = oracle_system.get_context_for_response(entities_of_interest)
    
    print(f"   Entities in context: {len(context['entities'])}")
    print(f"   Context packs available: {len(context['context_packs'])}")
    print(f"   Relationship map: {len(context['relationship_map'])} connections")
    
    # Show Oracle statistics
    print(f"\n📊 Oracle Knowledge Base:")
    stats = oracle_system.get_oracle_stats()
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for subkey, subvalue in value.items():
                print(f"     {subkey}: {subvalue}")
        elif isinstance(value, list):
            print(f"   {key}: {len(value)} items")
        elif isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    print(f"\n🔮✅ Oracle ready! Alice's omniscient context watcher is online.")
    print("I see all conversations, know all entities, predict all needs! 👁️")