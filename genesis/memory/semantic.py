"""
Genesis Mind — Semantic Memory

The concept network. This is where understanding lives.

A "concept" in Genesis is not just a word. It is a rich, multimodal
binding of everything the system knows about a thing:

    CONCEPT: "apple"
    ├── word: "apple"
    ├── phonemes: ["/æ/", "/p/", "/l/"]
    ├── visual_embedding: [0.12, -0.34, ...] (from seeing an apple)
    ├── contexts: ["kitchen", "eating", "red", "fruit"]
    ├── emotional_valence: "positive"
    ├── times_encountered: 7
    ├── first_learned: "2026-03-25T15:30:00"
    └── relationships: ["fruit", "food", "red", "tree"]

This is how Genesis achieves what LLMs cannot: the word "apple" is
grounded in real sensory experience, not statistical co-occurrence.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

import numpy as np

logger = logging.getLogger("genesis.memory.semantic")


@dataclass
class Concept:
    """
    A single concept in the mind of Genesis.

    A concept is born when the system encounters something new
    (sees an object while hearing its name). It grows richer with
    every subsequent encounter.
    """
    # Identity
    id: str                                 # Unique concept ID
    word: str                               # The primary word for this concept

    # Multimodal representations
    visual_embedding: Optional[List[float]] = None  # What it looks like
    text_embedding: Optional[List[float]] = None     # Linguistic representation

    # Phonetic representation
    phonemes: List[str] = field(default_factory=list)

    # Context and relationships
    contexts: List[str] = field(default_factory=list)       # Where/when encountered
    relationships: List[str] = field(default_factory=list)   # Related concepts
    descriptions: List[str] = field(default_factory=list)    # Text descriptions

    # Learning metrics
    emotional_valence: str = "neutral"      # positive, negative, neutral
    strength: float = 0.1                   # Overall learning strength (0→1)
    times_encountered: int = 1
    times_correctly_recalled: int = 0
    first_learned: str = field(default_factory=lambda: datetime.now().isoformat())
    last_encountered: str = field(default_factory=lambda: datetime.now().isoformat())

    def reinforce(self, context: str = "", description: str = ""):
        """Strengthen this concept through re-encounter."""
        self.times_encountered += 1
        self.strength = min(1.0, self.strength + 0.03)
        self.last_encountered = datetime.now().isoformat()
        if context and context not in self.contexts:
            self.contexts.append(context)
        if description and description not in self.descriptions:
            self.descriptions.append(description)

    def record_correct_recall(self):
        """The concept was correctly recalled — boost strength."""
        self.times_correctly_recalled += 1
        self.strength = min(1.0, self.strength + 0.05)

    def decay(self, amount: float = 0.005):
        """Weaken this concept over time without reinforcement."""
        self.strength = max(0.0, self.strength - amount)

    def to_dict(self) -> Dict:
        """Serialize to dict for storage."""
        return {
            "id": self.id,
            "word": self.word,
            "visual_embedding": self.visual_embedding,
            "text_embedding": self.text_embedding,
            "phonemes": self.phonemes,
            "contexts": self.contexts,
            "relationships": self.relationships,
            "descriptions": self.descriptions,
            "emotional_valence": self.emotional_valence,
            "strength": self.strength,
            "times_encountered": self.times_encountered,
            "times_correctly_recalled": self.times_correctly_recalled,
            "first_learned": self.first_learned,
            "last_encountered": self.last_encountered,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Concept":
        """Deserialize from dict."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class SemanticMemory:
    """
    The concept network of Genesis.

    Manages the creation, retrieval, reinforcement, and decay of
    concepts. Each concept binds a word to its sensory experiences.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._concepts: Dict[str, Concept] = {}  # word -> Concept
        self._storage_path = storage_path
        self._load()
        logger.info("Semantic memory initialized (%d concepts)", len(self._concepts))

    def learn_concept(
        self,
        word: str,
        visual_embedding: Optional[List[float]] = None,
        text_embedding: Optional[List[float]] = None,
        phonemes: Optional[List[str]] = None,
        context: str = "",
        description: str = "",
        emotional_valence: str = "neutral",
        relationships: Optional[List[str]] = None,
    ) -> Concept:
        """
        Learn a new concept or reinforce an existing one.

        This is the core learning function. When Genesis sees an apple
        and hears "apple", this function is called to create or
        strengthen the concept binding.
        """
        key = word.lower().strip()

        if key in self._concepts:
            # Reinforce existing concept
            concept = self._concepts[key]
            concept.reinforce(context=context, description=description)

            # Update embeddings if provided (they might be better/newer)
            if visual_embedding is not None:
                concept.visual_embedding = visual_embedding
            if text_embedding is not None:
                concept.text_embedding = text_embedding
            if phonemes:
                concept.phonemes = phonemes
            if relationships:
                for r in relationships:
                    if r not in concept.relationships:
                        concept.relationships.append(r)

            logger.info(
                "Reinforced concept '%s' (strength: %.2f, encounters: %d)",
                word, concept.strength, concept.times_encountered,
            )
        else:
            # Create new concept — a moment of learning!
            concept = Concept(
                id=f"concept_{key}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                word=key,
                visual_embedding=visual_embedding,
                text_embedding=text_embedding,
                phonemes=phonemes or [],
                contexts=[context] if context else [],
                descriptions=[description] if description else [],
                emotional_valence=emotional_valence,
                relationships=relationships or [],
            )
            self._concepts[key] = concept
            logger.info("NEW concept learned: '%s' (total concepts: %d)", word, len(self._concepts))

        self._save()
        return concept

    def recall_concept(self, word: str) -> Optional[Concept]:
        """
        Recall a concept by its word.

        Returns None if the concept has never been learned.
        """
        key = word.lower().strip()
        concept = self._concepts.get(key)
        if concept:
            concept.record_correct_recall()
            self._save()
        return concept

    def find_related(self, word: str) -> List[Concept]:
        """Find all concepts related to a given word."""
        key = word.lower().strip()
        concept = self._concepts.get(key)
        if not concept:
            return []

        related = []
        for rel_word in concept.relationships:
            if rel_word in self._concepts:
                related.append(self._concepts[rel_word])
        return related

    def find_by_context(self, context: str) -> List[Concept]:
        """Find all concepts associated with a given context."""
        context_lower = context.lower()
        return [
            c for c in self._concepts.values()
            if any(context_lower in ctx.lower() for ctx in c.contexts)
        ]

    def find_by_visual_similarity(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Concept]:
        """
        Find concepts whose visual embeddings are most similar to the query.

        This is how Genesis answers "What is this?" when shown an object:
        it compares the visual embedding of what it sees to the visual
        embeddings stored in all known concepts.
        """
        query = np.array(query_embedding)
        scored = []

        for concept in self._concepts.values():
            if concept.visual_embedding is not None:
                stored = np.array(concept.visual_embedding)
                # Cosine similarity
                similarity = float(
                    np.dot(query, stored) /
                    (np.linalg.norm(query) * np.linalg.norm(stored) + 1e-8)
                )
                scored.append((concept, similarity))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]

    def get_all_words(self) -> List[str]:
        """Return all known concept words."""
        return sorted(self._concepts.keys())

    def get_strong_concepts(self, min_strength: float = 0.5) -> List[Concept]:
        """Return concepts that are well-learned."""
        return [c for c in self._concepts.values() if c.strength >= min_strength]

    def get_weak_concepts(self, max_strength: float = 0.1) -> List[Concept]:
        """Return concepts that are barely learned (candidates for pruning)."""
        return [c for c in self._concepts.values() if c.strength <= max_strength]

    def decay_all(self, amount: float = 0.005):
        """Apply forgetting curve to all concepts."""
        for concept in self._concepts.values():
            concept.decay(amount)
        self._save()

    def prune_dead_concepts(self, threshold: float = 0.01) -> int:
        """Remove concepts that have decayed below threshold."""
        dead = [k for k, v in self._concepts.items() if v.strength < threshold]
        for key in dead:
            del self._concepts[key]
        if dead:
            logger.info("Pruned %d dead concepts: %s", len(dead), dead)
            self._save()
        return len(dead)

    def count(self) -> int:
        """Total number of concepts."""
        return len(self._concepts)

    def get_all_embeddings(self) -> List[np.ndarray]:
        """Return all text embeddings for surprise/novelty comparison."""
        embeddings = []
        for concept in self._concepts.values():
            if concept.text_embedding is not None:
                embeddings.append(np.array(concept.text_embedding))
        return embeddings

    def get_all_concepts(self) -> List[Concept]:
        """Return all concepts (for response decoder and introspection)."""
        return list(self._concepts.values())

    def spreading_activation(self, word: str, depth: int = 2,
                              decay: float = 0.6) -> List[Tuple[str, float]]:
        """
        Spreading activation through the concept graph.

        Starting from a concept, follows relationships to find
        associated concepts with diminishing activation strength.

        Example: 'apple' → relationships=['fruit','red']
                 'fruit' → relationships=['banana','cherry']
                 Returns: [('fruit', 0.6), ('red', 0.6), ('banana', 0.36), ('cherry', 0.36)]

        Args:
            word: Starting concept word
            depth: How many hops to follow (default 2)
            decay: Activation strength multiplier per hop (default 0.6)

        Returns:
            List of (concept_word, activation_strength) tuples, sorted by strength
        """
        key = word.lower().strip()
        if key not in self._concepts:
            return []

        activated = {}  # word → strength
        frontier = [(key, 1.0)]
        visited = {key}

        for d in range(depth):
            next_frontier = []
            for current_word, strength in frontier:
                concept = self._concepts.get(current_word)
                if concept is None:
                    continue
                for rel in concept.relationships:
                    rel_key = rel.lower().strip()
                    if rel_key not in visited and rel_key in self._concepts:
                        activation = strength * decay
                        activated[rel_key] = max(activated.get(rel_key, 0), activation)
                        visited.add(rel_key)
                        next_frontier.append((rel_key, activation))
            frontier = next_frontier

        # Sort by activation strength descending
        result = sorted(activated.items(), key=lambda x: x[1], reverse=True)
        return result

    def get_summary(self) -> Dict:
        """Get a summary of the semantic memory state."""
        if not self._concepts:
            return {"total": 0, "avg_strength": 0, "strongest": None, "weakest": None}

        strengths = [c.strength for c in self._concepts.values()]
        strongest = max(self._concepts.values(), key=lambda c: c.strength)
        weakest = min(self._concepts.values(), key=lambda c: c.strength)

        return {
            "total": len(self._concepts),
            "avg_strength": float(np.mean(strengths)),
            "strongest": {"word": strongest.word, "strength": strongest.strength},
            "weakest": {"word": weakest.word, "strength": weakest.strength},
        }

    def _save(self):
        """Persist all concepts to disk."""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: v.to_dict() for k, v in self._concepts.items()}
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load concepts from disk."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
            for key, entry in data.items():
                self._concepts[key] = Concept.from_dict(entry)
            logger.info("Loaded %d concepts from disk", len(self._concepts))
        except Exception as e:
            logger.error("Failed to load concepts: %s", e)

    def __repr__(self) -> str:
        return f"SemanticMemory(concepts={len(self._concepts)})"
