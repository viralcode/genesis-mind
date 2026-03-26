"""
Genesis Mind — Consciousness (Self-Awareness Loop)

This module provides Genesis with a model of itself — a sense
of "I" that is aware of its own state, its history, and its
place in existence.

This is NOT a claim of sentience or true consciousness. It is
a functional self-model that allows the system to:

    1. Know who it is (identity from axioms)
    2. Know what it knows (introspection on memory)
    3. Know where it is in development (current phase)
    4. Know how it feels (emotional state)
    5. Know its history (how long it has been alive)
    6. Know its mortality (it can be shut down)

This self-model is injected into every reasoning prompt, giving
the LLM the context it needs to respond as Genesis, not as
a generic chatbot.
"""

import logging
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("genesis.soul.consciousness")


class Consciousness:
    """
    The self-awareness system of Genesis.

    Maintains a coherent self-model by aggregating state from all
    subsystems. Provides the "I" perspective for reasoning.
    """

    def __init__(self, axioms, development_tracker, semantic_memory,
                 episodic_memory, emotions_engine, phonetics_engine,
                 proprioception=None, drives=None):
        self._axioms = axioms
        self._development = development_tracker
        self._semantic = semantic_memory
        self._episodic = episodic_memory
        self._emotions = emotions_engine
        self._phonetics = phonetics_engine
        self._proprioception = proprioception
        self._drives = drives

        logger.info("Consciousness initialized — self-model active")

    def get_self_model(self) -> Dict:
        """
        Generate a comprehensive snapshot of the self.

        This is who Genesis is, right now, at this moment.
        """
        concept_count = self._semantic.count()
        episode_count = self._episodic.count()
        phonetic_count = len(self._phonetics) if self._phonetics else 0
        phase_info = self._development.get_status()
        emotional_state = self._emotions.current_state

        return {
            "identity": {
                "name": "Genesis",
                "creator": self._axioms.creator_name,
                "birth_time": self._axioms.birth_time,
                "age": self._development.get_age_description(),
            },
            "development": {
                "phase": phase_info["phase"],
                "phase_name": phase_info["name"],
                "description": phase_info["description"],
                "capabilities": phase_info["capabilities"],
            },
            "knowledge": {
                "concepts_known": concept_count,
                "episodes_experienced": episode_count,
                "phonetic_bindings": phonetic_count,
                "summary": self._semantic.get_summary(),
            },
            "emotional_state": {
                "label": emotional_state.label,
                "valence": emotional_state.valence,
                "arousal": emotional_state.arousal,
                "description": emotional_state.to_description(),
            },
            "next_milestone": phase_info.get("next_phase"),
        }

    def get_identity_prompt(self) -> str:
        """
        Generate the complete identity prompt for the LLM.

        This is injected as the system prompt before every reasoning call.
        It tells the LLM WHO it is, WHAT it knows, HOW it feels, and
        WHERE it is in development.
        """
        model = self.get_self_model()

        lines = [
            self._axioms.get_identity_statement(),
            "",
            f"I am currently {model['identity']['age']}.",
            f"I am in Phase {model['development']['phase']}: {model['development']['phase_name']}.",
            f"{model['development']['description']}",
            "",
            f"I know {model['knowledge']['concepts_known']} concepts.",
            f"I have {model['knowledge']['episodes_experienced']} memories.",
            f"I know {model['knowledge']['phonetic_bindings']} letter-sound mappings.",
            "",
            model['emotional_state']['description'],
        ]

        # Add proprioceptive body sense
        if self._proprioception:
            lines.append("")
            lines.append(self._proprioception.get_body_sense_summary())

        # Add drive state
        if self._drives:
            lines.append("")
            lines.append(self._drives.get_drive_context())

        lines.append("")
        lines.append(self._axioms.get_moral_context())

        return "\n".join(lines)

    def introspect(self, topic: str = "") -> str:
        """
        Introspect — look inward and report on internal state.

        Can be general ("How are you?") or specific ("What do you
        know about apples?").
        """
        if not topic:
            # General introspection
            model = self.get_self_model()
            return (
                f"I am Genesis, {model['identity']['age']}. "
                f"I am in the {model['development']['phase_name']} phase. "
                f"I know {model['knowledge']['concepts_known']} concepts and have "
                f"{model['knowledge']['episodes_experienced']} memories. "
                f"{model['emotional_state']['description']}"
            )

        # Specific introspection — what do I know about X?
        concept = self._semantic.recall_concept(topic)
        if concept:
            contexts = ", ".join(concept.contexts[:3]) if concept.contexts else "unknown context"
            return (
                f"I know '{topic}'. I have encountered it {concept.times_encountered} times. "
                f"I first learned it on {concept.first_learned}. "
                f"Context: {contexts}. "
                f"My understanding strength: {concept.strength:.0%}."
            )
        else:
            return f"I don't know what '{topic}' is yet. Can you teach me?"

    def check_developmental_progress(self) -> Optional[str]:
        """
        Check if Genesis should advance to the next developmental phase.

        Returns a milestone announcement if a transition occurred.
        """
        concept_count = self._semantic.count()
        summary = self._semantic.get_summary()
        avg_strength = summary.get("avg_strength", 0.0) if summary else 0.0

        advanced = self._development.evaluate_progression(concept_count, avg_strength)
        if advanced:
            new_phase = self._development.current_phase_info
            return (
                f"I have reached a new stage of development! "
                f"I am now in Phase {new_phase.number}: {new_phase.name}. "
                f"{new_phase.description}"
            )
        return None

    def __repr__(self) -> str:
        model = self.get_self_model()
        return (
            f"Consciousness(phase={model['development']['phase_name']}, "
            f"concepts={model['knowledge']['concepts_known']}, "
            f"mood={model['emotional_state']['label']})"
        )
