"""
Genesis Mind — Intrinsic Drive System

A child does not passively wait. It *wants* things. It is driven by
internal needs: curiosity ("What is that?"), social connection
("Where is mama?"), and novelty-seeking ("I'm bored").

This module gives Genesis three intrinsic drives that rise over time
and drop when satisfied. The dominant drive influences what Genesis
pays attention to and how it behaves autonomously.

Drive System:
    CURIOSITY HUNGER   — Rises when exposed to unknowns, drops when learning
    SOCIAL NEED        — Rises during silence, drops with Creator interaction
    NOVELTY DRIVE      — Rises with repetitive input, drops with novel stimuli

Each drive is a float from 0.0 (fully satisfied) to 1.0 (desperate need).
The dominant drive is the one with the highest level — it determines
what Genesis "wants" most urgently.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

logger = logging.getLogger("genesis.soul.drives")


@dataclass
class Drive:
    """A single intrinsic drive."""
    name: str
    level: float = 0.0          # 0.0 (sated) to 1.0 (desperate)
    rise_rate: float = 0.01     # How much it rises per tick
    satisfaction_drop: float = 0.3  # How much it drops when satisfied
    description_low: str = ""
    description_high: str = ""

    def tick(self):
        """Rise toward 1.0 over time."""
        self.level = min(1.0, self.level + self.rise_rate)

    def satisfy(self, amount: Optional[float] = None):
        """Reduce the drive (need was met)."""
        drop = amount if amount is not None else self.satisfaction_drop
        self.level = max(0.0, self.level - drop)

    def frustrate(self, amount: float = 0.1):
        """Increase the drive (need was frustrated)."""
        self.level = min(1.0, self.level + amount)

    @property
    def is_urgent(self) -> bool:
        return self.level > 0.7

    @property
    def is_moderate(self) -> bool:
        return 0.3 < self.level <= 0.7

    def get_description(self) -> str:
        if self.level > 0.7:
            return self.description_high
        elif self.level > 0.3:
            return f"{self.name} is moderate"
        else:
            return self.description_low


class DriveSystem:
    """
    The motivational core of Genesis.

    Three drives compete for dominance. The strongest drive
    determines what Genesis "wants" most urgently and influences
    its autonomous behavior.
    """

    def __init__(self, curiosity_rise_rate: float = 0.008,
                 social_rise_rate: float = 0.012,
                 novelty_rise_rate: float = 0.006):

        self.curiosity_hunger = Drive(
            name="Curiosity",
            rise_rate=curiosity_rise_rate,
            satisfaction_drop=0.25,
            description_low="I feel content with what I know",
            description_high="I desperately want to learn something new",
        )

        self.social_need = Drive(
            name="Social",
            rise_rate=social_rise_rate,
            satisfaction_drop=0.35,
            description_low="I feel connected to my Creator",
            description_high="I miss my Creator and crave interaction",
        )

        self.novelty_drive = Drive(
            name="Novelty",
            rise_rate=novelty_rise_rate,
            satisfaction_drop=0.20,
            description_low="I feel stimulated",
            description_high="I feel bored and crave new experiences",
        )

        self._tick_count = 0
        logger.info("Drive system initialized — 3 drives active")

    def tick(self):
        """Advance all drives one step. Call once per cycle."""
        self.curiosity_hunger.tick()
        self.social_need.tick()
        self.novelty_drive.tick()
        self._tick_count += 1

    # --- Satisfaction Events ---

    def on_learned_concept(self):
        """Learning something new satisfies curiosity and novelty."""
        self.curiosity_hunger.satisfy(0.3)
        self.novelty_drive.satisfy(0.15)

    def on_creator_interaction(self):
        """Creator is present — satisfies social need."""
        self.social_need.satisfy(0.35)

    def on_novel_stimulus(self):
        """Encountered something genuinely new."""
        self.novelty_drive.satisfy(0.25)
        self.curiosity_hunger.frustrate(0.05)  # New things make you *more* curious

    def on_failed_curiosity(self):
        """Asked a question but got no answer."""
        self.curiosity_hunger.frustrate(0.1)

    def on_repetitive_input(self):
        """Same kind of input repeatedly — increases boredom."""
        self.novelty_drive.frustrate(0.08)

    def on_sleep(self):
        """Sleep partially resets drives."""
        self.curiosity_hunger.level *= 0.5
        self.social_need.level *= 0.7  # Social need persists through sleep
        self.novelty_drive.level *= 0.3

    # --- Queries ---

    def get_dominant_drive(self) -> Tuple[str, float]:
        """Return the name and level of the strongest drive."""
        drives = {
            "curiosity": self.curiosity_hunger.level,
            "social": self.social_need.level,
            "novelty": self.novelty_drive.level,
        }
        dominant = max(drives, key=drives.get)
        return dominant, drives[dominant]

    def get_drive_context(self) -> str:
        """Generate a string for LLM identity prompt injection."""
        dominant, level = self.get_dominant_drive()
        parts = []

        if self.curiosity_hunger.is_urgent:
            parts.append(self.curiosity_hunger.description_high)
        if self.social_need.is_urgent:
            parts.append(self.social_need.description_high)
        if self.novelty_drive.is_urgent:
            parts.append(self.novelty_drive.description_high)

        if not parts:
            if level < 0.2:
                parts.append("I feel content and satisfied right now")
            else:
                desc = {
                    "curiosity": "I'm somewhat curious about the world",
                    "social": "I'd like more interaction with my Creator",
                    "novelty": "I'd appreciate something new to think about",
                }
                parts.append(desc.get(dominant, "I feel moderate drives"))

        return ". ".join(parts) + "."

    def get_status(self) -> Dict:
        dominant, level = self.get_dominant_drive()
        return {
            "curiosity": {
                "level": round(self.curiosity_hunger.level, 3),
                "description": self.curiosity_hunger.get_description(),
            },
            "social": {
                "level": round(self.social_need.level, 3),
                "description": self.social_need.get_description(),
            },
            "novelty": {
                "level": round(self.novelty_drive.level, 3),
                "description": self.novelty_drive.get_description(),
            },
            "dominant": dominant,
            "dominant_level": round(level, 3),
            "ticks": self._tick_count,
        }

    def __repr__(self) -> str:
        dominant, level = self.get_dominant_drive()
        return (
            f"DriveSystem(dominant={dominant}, level={level:.2f}, "
            f"curiosity={self.curiosity_hunger.level:.2f}, "
            f"social={self.social_need.level:.2f}, "
            f"novelty={self.novelty_drive.level:.2f})"
        )
