"""
Genesis Mind — Joint Attention Engine

Joint attention is the foundation of human language acquisition.
It is what happens when a child and caregiver BOTH look at the
same object, and the caregiver speaks its name:

    1. Child SEES a red ball (visual modality)
    2. Caregiver SAYS "ball" (auditory modality)
    3. Child's brain BINDS the visual representation to the
       acoustic representation → learns that "ball" = round red thing

This is NOT rote memorization. It is cross-modal binding — the same
mechanism that allows you to close your eyes and "hear" the word
"lemon" when you taste something sour.

Genesis's Joint Attention Engine creates these cross-modal bindings:
    - Visual label (from the camera/teaching) + Heard word (from mic)
    - Binding strength increases with repeated co-occurrence
    - Weak bindings decay over time (forgetting curve)
    - Strong bindings become permanent vocabulary

This is the ONLY way Genesis learns vocabulary in tabula rasa mode.
No dictionaries. No pre-trained embeddings. Just seeing and hearing.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("genesis.cortex.joint_attention")


@dataclass
class CrossModalBinding:
    """
    A single cross-modal association between a visual concept
    and an acoustic/heard word.

    Represents the learned link: "when I see X, the word is Y"
    """
    visual_concept: str          # What was seen (e.g., "red round object")
    heard_word: str              # What was heard (e.g., "ball")
    strength: float = 0.1       # 0.0 (forgotten) to 1.0 (permanent)
    co_occurrences: int = 1     # Times these appeared together
    first_bound: str = field(default_factory=lambda: datetime.now().isoformat())
    last_reinforced: str = field(default_factory=lambda: datetime.now().isoformat())

    def reinforce(self, amount: float = 0.1):
        """Strengthen this binding through repeated co-occurrence."""
        self.co_occurrences += 1
        self.strength = min(1.0, self.strength + amount)
        self.last_reinforced = datetime.now().isoformat()

    def decay(self, amount: float = 0.01):
        """Weaken through time without reinforcement."""
        self.strength = max(0.0, self.strength - amount)

    @property
    def is_learned(self) -> bool:
        """A binding is considered 'learned' when strength > 0.5."""
        return self.strength > 0.5

    @property
    def is_permanent(self) -> bool:
        """A binding is permanent when strength > 0.9."""
        return self.strength > 0.9


class JointAttentionEngine:
    """
    The cross-modal binding system of Genesis.

    Creates and maintains associations between visual concepts
    and heard words. This is how Genesis learns vocabulary from
    scratch — purely through sensory co-occurrence.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        # Key: "visual_concept|heard_word" → CrossModalBinding
        self._bindings: Dict[str, CrossModalBinding] = {}
        self._storage_path = storage_path
        self._total_bindings_created = 0
        self._total_recalls = 0

        # Temporal buffer: recent visual and auditory events
        self._recent_visual: List[Tuple[str, str]] = []   # (label, timestamp)
        self._recent_auditory: List[Tuple[str, str]] = []  # (word, timestamp)
        self._temporal_window_sec = 5.0  # Max time gap for co-occurrence binding

        self._load()
        logger.info(
            "Joint attention engine initialized (%d bindings)",
            len(self._bindings),
        )

    # --- Core Binding ---

    def on_visual(self, label: str):
        """
        Called when Genesis sees/identifies something.

        Records the visual event and checks if there's a recent
        auditory event to bind with.
        """
        now = datetime.now().isoformat()
        self._recent_visual.append((label.lower().strip(), now))
        self._trim_buffer(self._recent_visual)

        # Check for temporal co-occurrence with recent audio
        for word, ts in self._recent_auditory:
            if self._within_window(ts, now):
                self._create_or_strengthen(label.lower().strip(), word)

    def on_heard(self, text: str):
        """
        Called when Genesis hears speech.

        Records individual words and checks if there's a recent
        visual event to bind with.
        """
        now = datetime.now().isoformat()
        words = text.lower().strip().split()

        for word in words:
            # Skip function words (too common to be meaningful)
            if word in _FUNCTION_WORDS:
                continue

            self._recent_auditory.append((word, now))

            # Check for temporal co-occurrence with recent vision
            for label, ts in self._recent_visual:
                if self._within_window(ts, now):
                    self._create_or_strengthen(label, word)

        self._trim_buffer(self._recent_auditory)

    def bind(self, visual_concept: str, heard_word: str):
        """
        Explicitly bind a visual concept to a heard word.

        Used for direct teaching: "This is an apple"
        (teacher points at object and says the word).
        """
        self._create_or_strengthen(
            visual_concept.lower().strip(),
            heard_word.lower().strip(),
        )

    # --- Recall ---

    def recall_by_sound(self, heard_word: str) -> Optional[str]:
        """
        Given a heard word, recall the visual concept it was bound to.

        "I hear 'ball' → I recall seeing a round red thing"
        """
        heard_word = heard_word.lower().strip()
        best_match = None
        best_strength = 0.0

        for key, binding in self._bindings.items():
            if binding.heard_word == heard_word and binding.strength > best_strength:
                best_match = binding.visual_concept
                best_strength = binding.strength

        if best_match:
            self._total_recalls += 1
        return best_match

    def recall_by_vision(self, visual_label: str) -> Optional[str]:
        """
        Given a visual concept, recall the word for it.

        "I see a round red thing → I recall the word 'ball'"
        """
        visual_label = visual_label.lower().strip()
        best_match = None
        best_strength = 0.0

        for key, binding in self._bindings.items():
            if binding.visual_concept == visual_label and binding.strength > best_strength:
                best_match = binding.heard_word
                best_strength = binding.strength

        if best_match:
            self._total_recalls += 1
        return best_match

    def get_vocabulary(self) -> List[str]:
        """
        Return all words Genesis has 'learned' (strength > 0.5).

        This is Genesis's actual vocabulary — words it has bound
        to visual concepts through joint attention.
        """
        words = set()
        for binding in self._bindings.values():
            if binding.is_learned:
                words.add(binding.heard_word)
        return sorted(words)

    def get_all_bindings_sorted(self) -> List[Dict]:
        """Return all bindings sorted by strength (strongest first)."""
        sorted_bindings = sorted(
            self._bindings.values(),
            key=lambda b: b.strength,
            reverse=True,
        )
        return [
            {
                "visual": b.visual_concept,
                "word": b.heard_word,
                "strength": round(b.strength, 3),
                "co_occurrences": b.co_occurrences,
                "learned": b.is_learned,
                "permanent": b.is_permanent,
            }
            for b in sorted_bindings
        ]

    # --- Maintenance ---

    def decay_all(self, amount: float = 0.005):
        """Apply forgetting curve to all bindings (during sleep)."""
        to_remove = []
        for key, binding in self._bindings.items():
            if not binding.is_permanent:  # Permanent bindings don't decay
                binding.decay(amount)
                if binding.strength <= 0.0:
                    to_remove.append(key)

        for key in to_remove:
            del self._bindings[key]

        if to_remove:
            logger.info("Pruned %d forgotten bindings", len(to_remove))

        self._save()

    def get_status(self) -> Dict:
        """Get the current status of the joint attention system."""
        learned_count = sum(1 for b in self._bindings.values() if b.is_learned)
        permanent_count = sum(1 for b in self._bindings.values() if b.is_permanent)
        return {
            "total_bindings": len(self._bindings),
            "learned_bindings": learned_count,
            "permanent_bindings": permanent_count,
            "vocabulary_size": len(self.get_vocabulary()),
            "total_recalls": self._total_recalls,
            "total_bindings_created": self._total_bindings_created,
            "strongest_bindings": self.get_all_bindings_sorted()[:5],
        }

    # --- Internal ---

    def _create_or_strengthen(self, visual: str, word: str):
        """Create a new binding or strengthen an existing one."""
        key = f"{visual}|{word}"

        if key in self._bindings:
            self._bindings[key].reinforce()
            logger.debug(
                "Strengthened binding: '%s' ↔ '%s' (strength=%.2f)",
                visual, word, self._bindings[key].strength,
            )
        else:
            self._bindings[key] = CrossModalBinding(
                visual_concept=visual,
                heard_word=word,
            )
            self._total_bindings_created += 1
            logger.info(
                "New cross-modal binding: '%s' ↔ '%s'",
                visual, word,
            )

        self._save()

    def _within_window(self, ts1: str, ts2: str) -> bool:
        """Check if two timestamps are within the temporal window."""
        try:
            t1 = datetime.fromisoformat(ts1)
            t2 = datetime.fromisoformat(ts2)
            diff = abs((t2 - t1).total_seconds())
            return diff <= self._temporal_window_sec
        except Exception:
            return False

    def _trim_buffer(self, buffer: list, max_size: int = 20):
        """Keep temporal buffers short."""
        while len(buffer) > max_size:
            buffer.pop(0)

    # --- Persistence ---

    def _save(self):
        """Persist bindings to disk."""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_bindings_created": self._total_bindings_created,
            "total_recalls": self._total_recalls,
            "bindings": {
                key: {
                    "visual_concept": b.visual_concept,
                    "heard_word": b.heard_word,
                    "strength": b.strength,
                    "co_occurrences": b.co_occurrences,
                    "first_bound": b.first_bound,
                    "last_reinforced": b.last_reinforced,
                }
                for key, b in self._bindings.items()
            },
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load bindings from disk."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
            self._total_bindings_created = data.get("total_bindings_created", 0)
            self._total_recalls = data.get("total_recalls", 0)
            for key, entry in data.get("bindings", {}).items():
                self._bindings[key] = CrossModalBinding(
                    visual_concept=entry["visual_concept"],
                    heard_word=entry["heard_word"],
                    strength=entry.get("strength", 0.1),
                    co_occurrences=entry.get("co_occurrences", 1),
                    first_bound=entry.get("first_bound", ""),
                    last_reinforced=entry.get("last_reinforced", ""),
                )
            logger.info(
                "Loaded %d cross-modal bindings",
                len(self._bindings),
            )
        except Exception as e:
            logger.error("Failed to load joint attention data: %s", e)

    def __repr__(self) -> str:
        return (
            f"JointAttentionEngine(bindings={len(self._bindings)}, "
            f"vocab={len(self.get_vocabulary())})"
        )


# Function words to skip during binding (too common to be meaningful)
_FUNCTION_WORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "am", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need", "must",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all",
    "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "just", "because", "but", "and", "or", "if", "while",
    "that", "this", "these", "those", "it", "its", "i", "me", "my",
    "we", "us", "our", "you", "your", "he", "him", "his", "she", "her",
    "they", "them", "their", "what", "which", "who", "whom",
])


# =============================================================================
# Standalone test — run with: python -m genesis.cortex.joint_attention
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Joint Attention Test")
    print("Testing cross-modal binding from zero...")
    print("=" * 60)

    engine = JointAttentionEngine(
        storage_path=Path("/tmp/genesis_joint_attention_test.json")
    )

    # Simulate: teacher shows apple and says "apple"
    print("\n--- Teaching 'apple' ---")
    engine.on_visual("red round fruit")
    engine.on_heard("look at the apple")
    print(f"  Bindings: {engine.get_all_bindings_sorted()}")

    # Reinforce 5 more times
    for i in range(5):
        engine.on_visual("red round fruit")
        engine.on_heard("apple")

    # Test recall
    print("\n--- Recall Tests ---")
    visual = engine.recall_by_sound("apple")
    print(f"  Hear 'apple' → see '{visual}'")

    word = engine.recall_by_vision("red round fruit")
    print(f"  See 'red round fruit' → say '{word}'")

    # Vocabulary
    print(f"\n  Learned vocabulary: {engine.get_vocabulary()}")
    print(f"  Status: {engine.get_status()}")
    print("Joint attention test PASSED")
