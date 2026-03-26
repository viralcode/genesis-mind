"""
Genesis Mind — Acoustic Babbling Engine

A human infant does not begin by speaking words. They begin by
producing random vocalizations — babbling. Through a reinforcement
loop (babble → hear self → receive social feedback → strengthen
or weaken), infants gradually converge on the phonemes of their
native language.

This module gives Genesis the same developmental pathway:

    1. RANDOM BABBLING:  Generate random consonant-vowel sequences
                         ("ba", "da", "ma", "guh", "pah")
    2. SELF-MONITORING:  Hear the sound it just made (proprioceptive
                         auditory feedback)
    3. REINFORCEMENT:    If the human responds positively (smiling,
                         repeating, speaking), dopamine fires and the
                         specific phoneme sequence is strengthened
    4. CONVERGENCE:      Over time, the babbling repertoire shifts
                         from random noise toward recognizable speech
                         sounds

The babbling engine is drive-linked: when social drives are high,
Genesis babbles more frequently — just like a real infant seeking
attention through vocalization.
"""

import json
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("genesis.senses.babbling")


# =============================================================================
# Phoneme Inventory — The raw building blocks of sound
# =============================================================================

# Consonant phonemes (onset)
CONSONANTS = [
    "b", "d", "g", "m", "n", "p", "t", "k",
    "f", "s", "h", "w", "l", "r", "y",
]

# Vowel phonemes (nucleus)
VOWELS = [
    "ah", "eh", "ee", "oh", "oo",
    "uh", "ay", "ow", "ih",
]

# Mapping from phoneme symbols to speakable approximations for TTS
PHONEME_TO_SPEECH = {
    "b": "buh",  "d": "duh",  "g": "guh",  "m": "muh",
    "n": "nuh",  "p": "puh",  "t": "tuh",  "k": "kuh",
    "f": "fuh",  "s": "suh",  "h": "huh",  "w": "wuh",
    "l": "luh",  "r": "ruh",  "y": "yuh",
    "ah": "ah",  "eh": "eh",  "ee": "ee",  "oh": "oh",
    "oo": "oo",  "uh": "uh",  "ay": "ay",  "ow": "ow",
    "ih": "ih",
}


@dataclass
class VocalUnit:
    """
    A single learned vocalization in the repertoire.

    Represents a phoneme sequence that Genesis has produced and
    optionally received feedback on. Stronger units are babbled
    more frequently (positive reinforcement).
    """
    phonemes: List[str]              # e.g., ["b", "ah"] → "bah"
    speakable: str                   # e.g., "bah"
    strength: float = 0.1           # 0.0 (forgotten) to 1.0 (deeply reinforced)
    times_produced: int = 1         # How many times Genesis has said this
    times_reinforced: int = 0       # How many times human responded positively
    first_produced: str = field(default_factory=lambda: datetime.now().isoformat())
    last_produced: str = field(default_factory=lambda: datetime.now().isoformat())
    associated_concept: Optional[str] = None  # Bound visual concept (if any)

    def reinforce(self, amount: float = 0.1):
        """Strengthen this vocalization (positive feedback received)."""
        self.times_reinforced += 1
        self.strength = min(1.0, self.strength + amount)

    def weaken(self, amount: float = 0.02):
        """Weaken through lack of reinforcement (forgetting curve)."""
        self.strength = max(0.0, self.strength - amount)

    def produce(self):
        """Mark this unit as having been produced again."""
        self.times_produced += 1
        self.last_produced = datetime.now().isoformat()


class BabblingEngine:
    """
    The vocal motor system of Genesis.

    Generates random phoneme sequences (babbling), maintains a
    repertoire of reinforced vocalizations, and provides the
    interface for drive-linked speech production.
    """

    def __init__(self, storage_path: Optional[Path] = None,
                 max_repertoire_size: int = 200):
        self._repertoire: Dict[str, VocalUnit] = {}
        self._storage_path = storage_path
        self._max_repertoire_size = max_repertoire_size
        self._total_babbles = 0
        self._total_reinforcements = 0
        self._last_babble: Optional[str] = None
        self._last_babble_time: Optional[str] = None
        self._load()

        logger.info(
            "Babbling engine initialized (repertoire=%d units)",
            len(self._repertoire),
        )

    # --- Core Babbling ---

    def babble(self, syllable_count: int = None) -> Tuple[str, List[str]]:
        """
        Produce a babble — a random or semi-random phoneme sequence.

        If the repertoire has reinforced units, Genesis preferentially
        produces those (exploitation). Otherwise it explores randomly
        (exploration).

        Returns:
            (speakable_text, phoneme_list) — e.g., ("bah dah", ["b", "ah", "d", "ah"])
        """
        if syllable_count is None:
            syllable_count = random.randint(1, 3)

        # Exploration vs Exploitation
        if self._repertoire and random.random() < self._exploitation_probability():
            # Exploit: produce a reinforced vocalization
            unit = self._sample_reinforced()
            unit.produce()
            self._record_babble(unit.speakable)
            return unit.speakable, unit.phonemes
        else:
            # Explore: generate random syllables
            phonemes = []
            syllables = []
            for _ in range(syllable_count):
                c = random.choice(CONSONANTS)
                v = random.choice(VOWELS)
                phonemes.extend([c, v])
                syllables.append(PHONEME_TO_SPEECH.get(c, c) + PHONEME_TO_SPEECH.get(v, v))

            speakable = " ".join(syllables)

            # Add to repertoire as a weak, exploratory unit
            key = "-".join(phonemes)
            if key not in self._repertoire:
                self._repertoire[key] = VocalUnit(
                    phonemes=phonemes,
                    speakable=speakable,
                )
            else:
                self._repertoire[key].produce()

            self._record_babble(speakable)
            self._prune_repertoire()
            return speakable, phonemes

    def babble_for_concept(self, concept: str) -> Tuple[str, List[str]]:
        """
        Attempt to vocalize a specific concept.

        If Genesis has a reinforced vocalization associated with this
        concept, use it. Otherwise, babble randomly (the infant
        doesn't know the word yet).
        """
        # Search for a bound vocalization
        for key, unit in self._repertoire.items():
            if unit.associated_concept == concept and unit.strength > 0.2:
                unit.produce()
                self._record_babble(unit.speakable)
                return unit.speakable, unit.phonemes

        # No binding found — random babble
        return self.babble(syllable_count=random.randint(1, 2))

    # --- Reinforcement Learning ---

    def reinforce_last(self, amount: float = 0.15):
        """
        Reinforce the last babble that was produced.

        Called when the human responds positively after Genesis babbles.
        This is the core RL mechanism — positive social feedback
        strengthens the vocalization.
        """
        if self._last_babble is None:
            return

        for unit in self._repertoire.values():
            if unit.speakable == self._last_babble:
                unit.reinforce(amount)
                self._total_reinforcements += 1
                logger.info(
                    "Reinforced babble '%s' (strength=%.2f, reinforcements=%d)",
                    unit.speakable, unit.strength, unit.times_reinforced,
                )
                self._save()
                return

    def bind_to_concept(self, concept: str):
        """
        Bind the last babble to a visual/semantic concept.

        This is the cross-modal binding: Genesis babbles "bah",
        the teacher shows an apple and says "apple" → Genesis
        associates "bah" with "apple". Over time, Genesis will
        try to say "bah" when it sees an apple.
        """
        if self._last_babble is None:
            return

        for unit in self._repertoire.values():
            if unit.speakable == self._last_babble:
                unit.associated_concept = concept
                unit.reinforce(0.1)
                logger.info(
                    "Bound babble '%s' → concept '%s'",
                    unit.speakable, concept,
                )
                self._save()
                return

    def weaken_all(self, amount: float = 0.005):
        """Apply forgetting curve to all vocalizations (during sleep)."""
        for unit in self._repertoire.values():
            unit.weaken(amount)
        # Prune forgotten units
        self._repertoire = {
            k: v for k, v in self._repertoire.items() if v.strength > 0.01
        }
        self._save()

    # --- Drive-Linked Vocalization ---

    def should_babble(self, social_drive: float, curiosity_drive: float,
                      phase: int) -> bool:
        """
        Determine if Genesis should spontaneously babble based on
        its current drive state.

        Higher social need = more babbling (seeking attention).
        Higher curiosity = more exploration of new sounds.
        Early phases babble more frequently.
        """
        base_probability = 0.05  # 5% chance per tick
        drive_boost = (social_drive * 0.3) + (curiosity_drive * 0.1)
        phase_boost = max(0, (3 - phase)) * 0.1  # Younger = more babbling

        probability = base_probability + drive_boost + phase_boost
        return random.random() < probability

    # --- Queries ---

    def get_strongest_vocalizations(self, n: int = 10) -> List[Dict]:
        """Return the top-N strongest vocalizations in the repertoire."""
        sorted_units = sorted(
            self._repertoire.values(),
            key=lambda u: u.strength,
            reverse=True,
        )
        return [
            {
                "speakable": u.speakable,
                "phonemes": u.phonemes,
                "strength": round(u.strength, 3),
                "times_produced": u.times_produced,
                "times_reinforced": u.times_reinforced,
                "associated_concept": u.associated_concept,
            }
            for u in sorted_units[:n]
        ]

    def get_status(self) -> Dict:
        """Get the current state of the babbling engine."""
        return {
            "repertoire_size": len(self._repertoire),
            "total_babbles": self._total_babbles,
            "total_reinforcements": self._total_reinforcements,
            "last_babble": self._last_babble,
            "last_babble_time": self._last_babble_time,
            "strongest": self.get_strongest_vocalizations(5),
            "exploitation_probability": round(self._exploitation_probability(), 3),
        }

    # --- Internal ---

    def _exploitation_probability(self) -> float:
        """
        How likely Genesis is to exploit (use reinforced babbles)
        vs explore (random babbling).

        Starts at pure exploration. As reinforcements accumulate,
        shifts toward exploitation — just like real infant babbling.
        """
        if self._total_reinforcements == 0:
            return 0.0
        # Sigmoid-like: converges to 0.8 over ~50 reinforcements
        return min(0.8, self._total_reinforcements / (self._total_reinforcements + 20))

    def _sample_reinforced(self) -> VocalUnit:
        """Sample a vocalization weighted by reinforcement strength."""
        units = list(self._repertoire.values())
        weights = [u.strength for u in units]
        total = sum(weights)
        if total == 0:
            return random.choice(units)
        probs = [w / total for w in weights]
        return random.choices(units, weights=probs, k=1)[0]

    def _record_babble(self, speakable: str):
        """Record metadata about the last babble."""
        self._last_babble = speakable
        self._last_babble_time = datetime.now().isoformat()
        self._total_babbles += 1

    def _prune_repertoire(self):
        """Keep repertoire under max size by pruning weakest units."""
        if len(self._repertoire) > self._max_repertoire_size:
            sorted_keys = sorted(
                self._repertoire.keys(),
                key=lambda k: self._repertoire[k].strength,
            )
            # Remove weakest 10%
            to_remove = sorted_keys[:len(sorted_keys) // 10]
            for key in to_remove:
                del self._repertoire[key]

    # --- Persistence ---

    def _save(self):
        """Persist the vocal repertoire to disk."""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "total_babbles": self._total_babbles,
            "total_reinforcements": self._total_reinforcements,
            "repertoire": {
                key: {
                    "phonemes": u.phonemes,
                    "speakable": u.speakable,
                    "strength": u.strength,
                    "times_produced": u.times_produced,
                    "times_reinforced": u.times_reinforced,
                    "first_produced": u.first_produced,
                    "last_produced": u.last_produced,
                    "associated_concept": u.associated_concept,
                }
                for key, u in self._repertoire.items()
            },
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load the vocal repertoire from disk."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
            self._total_babbles = data.get("total_babbles", 0)
            self._total_reinforcements = data.get("total_reinforcements", 0)
            for key, entry in data.get("repertoire", {}).items():
                self._repertoire[key] = VocalUnit(
                    phonemes=entry["phonemes"],
                    speakable=entry["speakable"],
                    strength=entry.get("strength", 0.1),
                    times_produced=entry.get("times_produced", 1),
                    times_reinforced=entry.get("times_reinforced", 0),
                    first_produced=entry.get("first_produced", ""),
                    last_produced=entry.get("last_produced", ""),
                    associated_concept=entry.get("associated_concept"),
                )
            logger.info(
                "Loaded vocal repertoire: %d units, %d total babbles",
                len(self._repertoire), self._total_babbles,
            )
        except Exception as e:
            logger.error("Failed to load vocal repertoire: %s", e)

    def __repr__(self) -> str:
        return (
            f"BabblingEngine(repertoire={len(self._repertoire)}, "
            f"babbles={self._total_babbles}, reinforcements={self._total_reinforcements})"
        )


# =============================================================================
# Standalone test — run with: python -m genesis.senses.babbling
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Babbling Engine Test")
    print("Testing acoustic babbling from zero...")
    print("=" * 60)

    engine = BabblingEngine(storage_path=Path("/tmp/genesis_babbling_test.json"))

    # Phase 1: Random babbling
    print("\n--- Random Babbling (5 attempts) ---")
    for i in range(5):
        text, phonemes = engine.babble()
        print(f"  Babble {i+1}: '{text}' (phonemes: {phonemes})")

    # Phase 2: Reinforce a babble
    print("\n--- Reinforcing last babble ---")
    engine.reinforce_last(0.3)
    engine.reinforce_last(0.3)
    engine.reinforce_last(0.3)

    # Phase 3: Exploitation (should prefer reinforced babble)
    print("\n--- After reinforcement (5 attempts) ---")
    for i in range(5):
        text, phonemes = engine.babble()
        print(f"  Babble {i+1}: '{text}' (phonemes: {phonemes})")

    # Phase 4: Bind to concept
    print("\n--- Binding to concept 'mama' ---")
    engine.bind_to_concept("mama")

    # Phase 5: Try to vocalize concept
    print("\n--- Vocalizing 'mama' ---")
    text, phonemes = engine.babble_for_concept("mama")
    print(f"  Result: '{text}' (phonemes: {phonemes})")

    print(f"\nStatus: {engine.get_status()}")
    print("Babbling test PASSED")
