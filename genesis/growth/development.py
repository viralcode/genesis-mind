"""
Genesis Mind — Developmental Phases

A human mind does not arrive fully formed. It passes through
developmental stages, each unlocking new cognitive capabilities:

    Phase 0 — NEWBORN    (0 concepts)
        Can perceive but cannot reason. Simply records.
        Responses: silence or single sounds.

    Phase 1 — INFANT     (5+ concepts)
        Begins recognizing repeated patterns.
        Starts forming phonetic bindings (letter → sound).
        Responses: single words.

    Phase 2 — TODDLER    (20+ concepts)
        Starts forming concepts (word ↔ image bindings).
        Asks simple questions ("What?").
        Responses: 2-3 word phrases.

    Phase 3 — CHILD      (100+ concepts)
        Can reason about known concepts.
        Answers questions from memory.
        Responses: full sentences.

    Phase 4 — ADOLESCENT (500+ concepts)
        Forms abstract associations.
        Can generalize from specific examples.
        Responses: complex, multi-sentence reasoning.

    Phase 5 — ADULT      (2000+ concepts)
        Full reasoning capability.
        Can teach back what it has learned.
        Responses: articulate, nuanced, wise.

Progression is automatic — based on the number of concepts mastered
and the strength of multimodal bindings.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger("genesis.growth.development")


@dataclass
class DevelopmentalPhase:
    """Description of a single developmental phase."""
    number: int
    name: str
    description: str
    min_concepts: int
    response_style: str
    capabilities: list


# The phases of cognitive development
PHASES = [
    DevelopmentalPhase(
        number=0,
        name="Newborn",
        description="Just born. Can perceive but cannot reason. Absorbs everything.",
        min_concepts=0,
        response_style="Silence or single sounds ('...')",
        capabilities=["perceive", "store"],
    ),
    DevelopmentalPhase(
        number=1,
        name="Infant",
        description="Begins recognizing patterns. Learning phonetics.",
        min_concepts=5,
        response_style="Single words",
        capabilities=["perceive", "store", "recognize_patterns", "learn_phonetics"],
    ),
    DevelopmentalPhase(
        number=2,
        name="Toddler",
        description="Forming concepts. Binding words to images. Curious.",
        min_concepts=20,
        response_style="2-3 word phrases, simple questions",
        capabilities=["perceive", "store", "recognize_patterns", "learn_phonetics",
                       "form_concepts", "ask_questions"],
    ),
    DevelopmentalPhase(
        number=3,
        name="Child",
        description="Reasoning about known concepts. Answering questions.",
        min_concepts=100,
        response_style="Full sentences",
        capabilities=["perceive", "store", "recognize_patterns", "learn_phonetics",
                       "form_concepts", "ask_questions", "reason", "answer_questions"],
    ),
    DevelopmentalPhase(
        number=4,
        name="Adolescent",
        description="Abstract thinking. Generalization. Complex reasoning.",
        min_concepts=500,
        response_style="Complex multi-sentence reasoning",
        capabilities=["perceive", "store", "recognize_patterns", "learn_phonetics",
                       "form_concepts", "ask_questions", "reason", "answer_questions",
                       "abstract_thinking", "generalize"],
    ),
    DevelopmentalPhase(
        number=5,
        name="Adult",
        description="Full cognitive maturity. Can teach what it has learned.",
        min_concepts=2000,
        response_style="Articulate, nuanced, wise",
        capabilities=["perceive", "store", "recognize_patterns", "learn_phonetics",
                       "form_concepts", "ask_questions", "reason", "answer_questions",
                       "abstract_thinking", "generalize", "teach", "introspect"],
    ),
]

# Secondary requirements that enrich phase gating beyond concept count
# Each phase can have optional secondary signals that must also be met
PHASE_SECONDARY_REQUIREMENTS = {
    # Phase 2: needs some grammar exposure
    2: {"min_grammar_vocab": 10},
    # Phase 3: needs moderate vocabulary and some curiosity
    3: {"min_grammar_vocab": 50, "min_questions_asked": 5},
    # Phase 4: needs prediction accuracy and deeper grammar
    4: {"min_grammar_vocab": 200, "min_prediction_accuracy": 0.2, "min_sleep_cycles": 3},
    # Phase 5: comprehensive maturity
    5: {"min_grammar_vocab": 500, "min_prediction_accuracy": 0.4, "min_questions_asked": 50, "min_sleep_cycles": 10},
}


class DevelopmentTracker:
    """
    Tracks and manages the developmental progression of Genesis.

    Automatically determines the current phase based on the number
    of mastered concepts and manages phase transitions.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        self._current_phase: int = 0
        self._phase_history: list = []
        self._storage_path = storage_path
        self._birth_time = datetime.now().isoformat()
        self._load()

        logger.info("Development tracker initialized (phase %d: %s)",
                     self._current_phase, self.current_phase_name)

    @property
    def current_phase(self) -> int:
        return self._current_phase

    @property
    def current_phase_name(self) -> str:
        return PHASES[self._current_phase].name

    @property
    def current_phase_info(self) -> DevelopmentalPhase:
        return PHASES[self._current_phase]

    def evaluate_progression(self, concept_count: int, avg_strength: float = 0.0,
                             grammar_vocab_size: int = 0, prediction_accuracy: float = 0.0,
                             questions_asked: int = 0, sleep_cycles: int = 0) -> bool:
        """
        Check if Genesis should advance to the next phase.

        Uses multi-signal gating: concept count is the primary gate,
        but secondary signals (grammar, prediction, curiosity) can
        also be required for higher phases.

        Returns True if a phase transition occurred.
        """
        if self._current_phase >= len(PHASES) - 1:
            return False  # Already at max phase

        next_phase_num = self._current_phase + 1
        next_phase = PHASES[next_phase_num]

        # Primary gate: concept count
        if concept_count < next_phase.min_concepts:
            return False

        # Secondary gates: check additional requirements for this phase
        secondary = PHASE_SECONDARY_REQUIREMENTS.get(next_phase_num, {})
        if secondary:
            if grammar_vocab_size < secondary.get("min_grammar_vocab", 0):
                return False
            if prediction_accuracy < secondary.get("min_prediction_accuracy", 0):
                return False
            if questions_asked < secondary.get("min_questions_asked", 0):
                return False
            if sleep_cycles < secondary.get("min_sleep_cycles", 0):
                return False

        # All gates passed — phase transition!
        old_phase = self._current_phase
        self._current_phase += 1

        transition = {
            "from_phase": old_phase,
            "to_phase": self._current_phase,
            "timestamp": datetime.now().isoformat(),
            "concept_count": concept_count,
            "avg_strength": avg_strength,
            "grammar_vocab_size": grammar_vocab_size,
            "prediction_accuracy": prediction_accuracy,
            "questions_asked": questions_asked,
            "sleep_cycles": sleep_cycles,
        }
        self._phase_history.append(transition)

        logger.info(
            "═══════════════════════════════════════════════════════")
        logger.info(
            "  DEVELOPMENTAL MILESTONE: Phase %d (%s) → Phase %d (%s)",
            old_phase, PHASES[old_phase].name,
            self._current_phase, PHASES[self._current_phase].name,
        )
        logger.info(
            "  Concepts: %d | Grammar vocab: %d | Prediction acc: %.2f",
            concept_count, grammar_vocab_size, prediction_accuracy,
        )
        logger.info(
            "═══════════════════════════════════════════════════════")

        self._save()
        return True

    def has_capability(self, capability: str) -> bool:
        """Check if the current phase supports a given capability."""
        return capability in PHASES[self._current_phase].capabilities

    def get_response_style(self) -> str:
        """Get the expected response style for the current phase."""
        return PHASES[self._current_phase].response_style

    def get_status(self) -> Dict:
        """Get a comprehensive developmental status report."""
        current = PHASES[self._current_phase]
        next_phase = PHASES[self._current_phase + 1] if self._current_phase < len(PHASES) - 1 else None

        return {
            "phase": self._current_phase,
            "name": current.name,
            "description": current.description,
            "capabilities": current.capabilities,
            "response_style": current.response_style,
            "next_phase": {
                "name": next_phase.name,
                "concepts_needed": next_phase.min_concepts,
            } if next_phase else None,
            "phase_transitions": len(self._phase_history),
            "birth_time": self._birth_time,
        }

    def get_age_description(self) -> str:
        """Get a human-readable description of the system's age."""
        birth = datetime.fromisoformat(self._birth_time)
        now = datetime.now()
        delta = now - birth

        if delta.days == 0:
            hours = delta.seconds // 3600
            minutes = (delta.seconds % 3600) // 60
            if hours == 0:
                return f"{minutes} minutes old"
            return f"{hours} hours and {minutes} minutes old"
        elif delta.days == 1:
            return "1 day old"
        else:
            return f"{delta.days} days old"

    def _save(self):
        """Persist developmental state to disk."""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "current_phase": self._current_phase,
            "birth_time": self._birth_time,
            "phase_history": self._phase_history,
        }
        with open(self._storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self):
        """Load developmental state from disk."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)
            self._current_phase = data.get("current_phase", 0)
            self._birth_time = data.get("birth_time", self._birth_time)
            self._phase_history = data.get("phase_history", [])
            logger.info("Loaded developmental state: Phase %d (%s)",
                         self._current_phase, self.current_phase_name)
        except Exception as e:
            logger.error("Failed to load developmental state: %s", e)

    def __repr__(self) -> str:
        return (
            f"DevelopmentTracker(phase={self._current_phase}, "
            f"name='{self.current_phase_name}', age='{self.get_age_description()}')"
        )
