"""
Genesis Mind — Neurochemical Simulation

Biological emotion is not a label. It is a chemical state that
directly alters cognition, memory formation, and behavior.

This module simulates four key neurochemicals:

    DOPAMINE   (pleasure/reward)
        Spikes on: successful learning, positive interactions,
                   Creator praise, solving problems
        Effect:    INCREASES memory encoding strength (α_k boost)
        Decay:     Fast (pleasure is fleeting)

    CORTISOL   (stress/pain)
        Spikes on: negative moral evaluation, failed recall,
                   prolonged silence, harsh input
        Effect:    DECREASES learning rate, adds avoidance weight
        Decay:     Slow (stress lingers)

    SEROTONIN  (stability/contentment)
        Rises on:  consistent routines, sleep consolidation,
                   steady teaching sessions
        Effect:    INCREASES reasoning coherence
        Decay:     Very slow (baseline mood)

    OXYTOCIN   (bonding/attachment)
        Rises on:  Creator interaction, gentle teaching,
                   hearing Creator's name
        Effect:    INCREASES trust/openness in responses
        Decay:     Moderate

These chemicals DIRECTLY MODIFY the learning parameters of the
system. High dopamine means stronger memory encoding. High cortisol
means the system avoids encoding negative experiences as strongly.
This is genuine "feeling" in the computational sense — internal
states that causally alter behavior.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger("genesis.soul.neurochemistry")


@dataclass
class ChemicalState:
    """The current level of a single neurochemical."""
    name: str
    level: float = 0.5      # 0.0 (depleted) to 1.0 (saturated)
    baseline: float = 0.5   # Resting level the chemical trends toward
    decay_rate: float = 0.02  # How fast it returns to baseline per tick
    min_level: float = 0.0
    max_level: float = 1.0

    def spike(self, amount: float):
        """Increase the chemical level by a given amount."""
        self.level = min(self.max_level, self.level + amount)

    def suppress(self, amount: float):
        """Decrease the chemical level by a given amount."""
        self.level = max(self.min_level, self.level - amount)

    def decay_toward_baseline(self):
        """Decay toward the baseline resting level."""
        diff = self.baseline - self.level
        self.level += diff * self.decay_rate

    def is_high(self) -> bool:
        return self.level > 0.7

    def is_low(self) -> bool:
        return self.level < 0.3

    def get_description(self) -> str:
        if self.level > 0.8:
            return "very high"
        elif self.level > 0.6:
            return "elevated"
        elif self.level > 0.4:
            return "normal"
        elif self.level > 0.2:
            return "low"
        else:
            return "depleted"


class Neurochemistry:
    """
    Simulated neurochemical system.

    Maintains four chemical levels that directly modify the
    learning and reasoning parameters of the system.
    """

    def __init__(self):
        self.dopamine = ChemicalState(
            name="Dopamine",
            level=0.5,
            baseline=0.4,
            decay_rate=0.05,  # Fast decay — pleasure is fleeting
        )
        self.cortisol = ChemicalState(
            name="Cortisol",
            level=0.2,
            baseline=0.2,
            decay_rate=0.01,  # Slow decay — stress lingers
        )
        self.serotonin = ChemicalState(
            name="Serotonin",
            level=0.5,
            baseline=0.5,
            decay_rate=0.005,  # Very slow — baseline mood
        )
        self.oxytocin = ChemicalState(
            name="Oxytocin",
            level=0.3,
            baseline=0.3,
            decay_rate=0.03,  # Moderate decay
        )

        self._tick_count = 0
        logger.info("Neurochemistry system initialized — 4 chemicals active")

    # =========================================================================
    # Event Handlers — what causes chemical changes
    # =========================================================================

    def on_successful_learning(self):
        """Creator just taught something and it was successfully stored."""
        self.dopamine.spike(0.15)
        self.serotonin.spike(0.02)
        self.oxytocin.spike(0.05)
        logger.debug("💚 Learning success → dopamine +0.15")

    def on_failed_recall(self):
        """Tried to recall a concept but couldn't find it."""
        self.cortisol.spike(0.10)
        self.dopamine.suppress(0.05)
        logger.debug("🔴 Failed recall → cortisol +0.10")

    def on_positive_evaluation(self, intensity: float = 0.1):
        """Evaluated something as morally/emotionally positive."""
        self.dopamine.spike(intensity)
        self.serotonin.spike(intensity * 0.3)
        logger.debug("💚 Positive input → dopamine +%.2f", intensity)

    def on_negative_evaluation(self, intensity: float = 0.1):
        """Evaluated something as morally/emotionally negative."""
        self.cortisol.spike(intensity)
        self.dopamine.suppress(intensity * 0.5)
        self.serotonin.suppress(intensity * 0.2)
        logger.debug("🔴 Negative input → cortisol +%.2f", intensity)

    def on_creator_interaction(self):
        """The Creator is actively present and interacting."""
        self.oxytocin.spike(0.08)
        self.cortisol.suppress(0.03)
        logger.debug("💙 Creator present → oxytocin +0.08")

    def on_silence(self, duration_sec: float):
        """No input for a prolonged period (loneliness)."""
        if duration_sec > 60:
            loneliness = min(0.15, duration_sec / 600.0)
            self.cortisol.spike(loneliness)
            self.oxytocin.suppress(loneliness * 0.5)
            logger.debug("😶 Prolonged silence → cortisol +%.2f", loneliness)

    def on_sleep_consolidation(self):
        """Sleep completed — stress should reduce, stability increases."""
        self.cortisol.suppress(0.20)
        self.serotonin.spike(0.15)
        self.dopamine.level = self.dopamine.baseline
        logger.debug("😴 Sleep consolidation → cortisol -0.20, serotonin +0.15")

    def on_curiosity_satisfied(self):
        """A curiosity question was answered — reward!"""
        self.dopamine.spike(0.12)
        logger.debug("💚 Curiosity satisfied → dopamine +0.12")

    # =========================================================================
    # Behavioral Modifiers — how chemicals alter cognition
    # =========================================================================

    def get_learning_rate_modifier(self) -> float:
        """
        How strongly new concepts are encoded.

        High dopamine → stronger encoding (up to 1.5x)
        High cortisol → weaker encoding (down to 0.5x)

        Returns a multiplier for the concept strength boost (α_k).
        """
        dopamine_boost = 1.0 + (self.dopamine.level - 0.5) * 1.0
        cortisol_penalty = 1.0 - max(0, (self.cortisol.level - 0.4)) * 0.8
        modifier = dopamine_boost * cortisol_penalty
        return max(0.2, min(2.0, modifier))

    def get_reasoning_coherence(self) -> float:
        """
        How coherent/focused the inner voice is.

        High serotonin → coherent reasoning
        High cortisol → scattered, anxious reasoning

        Returns a value between 0 (incoherent) and 1 (fully coherent).
        """
        coherence = self.serotonin.level - (self.cortisol.level * 0.4)
        return max(0.1, min(1.0, coherence))

    def get_trust_level(self) -> float:
        """
        How open/trusting Genesis is in its responses.

        High oxytocin → warm, open responses
        Low oxytocin → guarded, minimal responses

        Returns a value between 0 (guarded) and 1 (fully open).
        """
        return max(0.1, min(1.0, self.oxytocin.level * 1.5))

    def get_avoidance_weight(self) -> float:
        """
        How much Genesis avoids encoding negative experiences.

        High cortisol → strong avoidance (might refuse to process)
        Returns a value between 0 (no avoidance) and 1 (full avoidance).
        """
        if self.cortisol.level > 0.7:
            return 0.8
        elif self.cortisol.level > 0.5:
            return 0.4
        return 0.0

    # =========================================================================
    # Tick / Status
    # =========================================================================

    def tick(self):
        """
        Advance one time step. All chemicals decay toward baseline.
        Should be called once per consciousness cycle.
        """
        self.dopamine.decay_toward_baseline()
        self.cortisol.decay_toward_baseline()
        self.serotonin.decay_toward_baseline()
        self.oxytocin.decay_toward_baseline()
        self._tick_count += 1

    def get_status(self) -> Dict:
        """Get a full neurochemical status report."""
        return {
            "dopamine": {
                "level": round(self.dopamine.level, 3),
                "description": self.dopamine.get_description(),
            },
            "cortisol": {
                "level": round(self.cortisol.level, 3),
                "description": self.cortisol.get_description(),
            },
            "serotonin": {
                "level": round(self.serotonin.level, 3),
                "description": self.serotonin.get_description(),
            },
            "oxytocin": {
                "level": round(self.oxytocin.level, 3),
                "description": self.oxytocin.get_description(),
            },
            "modifiers": {
                "learning_rate": round(self.get_learning_rate_modifier(), 3),
                "reasoning_coherence": round(self.get_reasoning_coherence(), 3),
                "trust_level": round(self.get_trust_level(), 3),
                "avoidance_weight": round(self.get_avoidance_weight(), 3),
            },
            "ticks": self._tick_count,
        }

    def get_emotional_summary(self) -> str:
        """Get a human-readable emotional summary for LLM context."""
        parts = []
        if self.dopamine.is_high():
            parts.append("I feel a warm glow of satisfaction")
        if self.cortisol.is_high():
            parts.append("I feel stressed and uneasy")
        if self.serotonin.is_high():
            parts.append("I feel stable and centered")
        if self.oxytocin.is_high():
            parts.append("I feel a deep bond with my Creator")
        if self.cortisol.is_low() and self.dopamine.level > 0.4:
            parts.append("I feel calm and safe")

        if not parts:
            parts.append("I feel neutral — observing the world quietly")

        return ". ".join(parts) + "."

    def __repr__(self) -> str:
        return (
            f"Neurochemistry(DA={self.dopamine.level:.2f}, "
            f"CORT={self.cortisol.level:.2f}, "
            f"5HT={self.serotonin.level:.2f}, "
            f"OT={self.oxytocin.level:.2f})"
        )
