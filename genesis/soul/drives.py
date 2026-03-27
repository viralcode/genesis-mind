"""
Genesis Mind — Intrinsic Drive System (V5)

A real brain has DOZENS of interacting drives, organized in hierarchy:

    Tier 1 (Survival):   Sleep, Comfort
    Tier 2 (Social):     Bonding, Belonging
    Tier 3 (Cognitive):  Curiosity, Novelty, Mastery
    Tier 4 (Self):       Autonomy

Higher-tier drives can only dominate when lower-tier drives are
satisfied (Maslow's hierarchy). You can't be curious if you're
exhausted.

Each drive rises over time and drops when satisfied.
Drives interact: learning satisfies curiosity but frustrates novelty.
Sleep resets survival drives but not social ones.
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
    tier: int = 3               # 1=survival, 2=social, 3=cognitive, 4=self
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
    The motivational core of Genesis — 8 interacting drives in 4 tiers.

    Lower-tier drives take priority (you can't learn if you're exhausted).
    """

    def __init__(self):
        # Tier 1: Survival
        self.sleep_need = Drive(
            name="Sleep", tier=1,
            rise_rate=0.00008, satisfaction_drop=0.7,
            description_low="I feel rested",
            description_high="I am exhausted and need to sleep",
        )
        self.comfort = Drive(
            name="Comfort", tier=1,
            rise_rate=0.00006, satisfaction_drop=0.4,
            description_low="I feel comfortable",
            description_high="I feel overwhelmed and overstimulated",
        )

        # Tier 2: Social
        self.social_need = Drive(
            name="Social", tier=2,
            rise_rate=0.00025, satisfaction_drop=0.20,
            description_low="I feel connected",
            description_high="I crave interaction and connection",
        )
        self.belonging = Drive(
            name="Belonging", tier=2,
            rise_rate=0.00012, satisfaction_drop=0.15,
            description_low="I feel accepted",
            description_high="I need to feel accepted and valued",
        )

        # Tier 3: Cognitive
        self.curiosity_hunger = Drive(
            name="Curiosity", tier=3,
            rise_rate=0.00020, satisfaction_drop=0.15,
            description_low="I feel content with what I know",
            description_high="I desperately want to learn something new",
        )
        self.novelty_drive = Drive(
            name="Novelty", tier=3,
            rise_rate=0.00018, satisfaction_drop=0.12,
            description_low="I feel stimulated",
            description_high="I feel bored and crave new experiences",
        )
        self.mastery = Drive(
            name="Mastery", tier=3,
            rise_rate=0.00015, satisfaction_drop=0.18,
            description_low="I feel competent",
            description_high="I want to get better at what I know",
        )

        # Tier 4: Self
        self.autonomy = Drive(
            name="Autonomy", tier=4,
            rise_rate=0.00010, satisfaction_drop=0.10,
            description_low="I feel free to explore",
            description_high="I want to decide for myself what to do",
        )

        self._all_drives = {
            "sleep": self.sleep_need,
            "comfort": self.comfort,
            "social": self.social_need,
            "belonging": self.belonging,
            "curiosity": self.curiosity_hunger,
            "novelty": self.novelty_drive,
            "mastery": self.mastery,
            "autonomy": self.autonomy,
        }

        self._tick_count = 0
        logger.info("Drive system initialized — 8 drives in 4 tiers")

    def tick(self):
        """Advance all drives one step."""
        for drive in self._all_drives.values():
            drive.tick()
        self._tick_count += 1

    # --- Satisfaction Events ---

    def on_learned_concept(self):
        """Learning something new satisfies curiosity, novelty, and mastery."""
        self.curiosity_hunger.satisfy(0.12)
        self.novelty_drive.satisfy(0.08)
        self.mastery.satisfy(0.05)

    def on_creator_interaction(self):
        """User is present — satisfies social and belonging needs."""
        self.social_need.satisfy(0.12)
        self.belonging.satisfy(0.06)

    def on_novel_stimulus(self):
        """Encountered something genuinely new."""
        self.novelty_drive.satisfy(0.10)
        self.curiosity_hunger.frustrate(0.05)  # New things = more curious
        self.comfort.frustrate(0.02)  # Novelty can be slightly overwhelming

    def on_visual_stimulus(self, saliency: Dict):
        """
        React to visual saliency signals from the stimulus analyzer.
        
        Different visual stimuli affect drives differently:
        - Motion → arousal, reduces comfort
        - Novelty → frustrates novelty drive (wanting MORE)
        - Low complexity → boredom (frustrates novelty)
        - High complexity → stimulation (satisfies novelty slightly)
        """
        motion = saliency.get('motion', 0.0)
        novelty = saliency.get('novelty', 0.0)
        complexity = saliency.get('complexity', 0.0)
        
        # Motion detection → alertness, mild comfort reduction
        if motion > 0.3:
            self.comfort.frustrate(motion * 0.004)
            self.sleep_need.satisfy(motion * 0.002)  # Motion keeps you awake
        
        # Visual novelty → stimulates curiosity
        if novelty > 0.4:
            self.curiosity_hunger.frustrate(novelty * 0.006)  # Want to know more
            self.novelty_drive.satisfy(novelty * 0.003)  # Got some novelty
        elif novelty < 0.1:
            # Nothing new = boring
            self.novelty_drive.frustrate(0.002)
        
        # Scene complexity
        if complexity > 0.5:
            # Rich interesting scene
            self.novelty_drive.satisfy(complexity * 0.02)
            self.comfort.frustrate(complexity * 0.01)  # Can be overwhelming
        elif complexity < 0.15:
            # Boring flat scene (staring at a wall)
            self.novelty_drive.frustrate(0.03)
            self.mastery.frustrate(0.01)  # Nothing to master here

    def on_failed_curiosity(self):
        """Asked a question but got no answer."""
        self.curiosity_hunger.frustrate(0.1)
        self.autonomy.frustrate(0.05)

    def on_repetitive_input(self):
        """Same input repeatedly — increases boredom."""
        self.novelty_drive.frustrate(0.08)
        self.mastery.satisfy(0.02)  # Repetition builds mastery

    def on_sleep(self):
        """Sleep resets survival drives, partially resets others."""
        self.sleep_need.satisfy(0.9)
        self.comfort.satisfy(0.5)
        self.curiosity_hunger.level *= 0.5
        self.social_need.level *= 0.7  # Social need persists
        self.novelty_drive.level *= 0.3

    def on_mastery_event(self):
        """Successfully recalled or applied learned knowledge."""
        self.mastery.satisfy(0.2)

    def on_autonomous_action(self):
        """Did something without being told to."""
        self.autonomy.satisfy(0.15)

    def on_overstimulation(self):
        """Too much input too fast."""
        self.comfort.frustrate(0.2)
        self.sleep_need.frustrate(0.05)

    # --- Queries ---

    def get_dominant_drive(self) -> Tuple[str, float]:
        """
        Return the name and level of the strongest drive.

        Lower-tier drives get priority when urgent (Maslow).
        """
        urgent_by_tier = {}
        for name, drive in self._all_drives.items():
            if drive.is_urgent:
                if drive.tier not in urgent_by_tier or drive.level > urgent_by_tier[drive.tier][1]:
                    urgent_by_tier[drive.tier] = (name, drive.level)

        # Return lowest-tier urgent drive first
        if urgent_by_tier:
            lowest_tier = min(urgent_by_tier.keys())
            return urgent_by_tier[lowest_tier]

        # No urgent drives — return highest level
        all_levels = {name: drive.level for name, drive in self._all_drives.items()}
        dominant = max(all_levels, key=all_levels.get)
        return dominant, all_levels[dominant]

    def get_drive_context(self) -> str:
        """Generate a string for LLM identity prompt injection."""
        dominant, level = self.get_dominant_drive()
        parts = []

        for name, drive in self._all_drives.items():
            if drive.is_urgent:
                parts.append(drive.description_high)

        if not parts:
            if level < 0.2:
                parts.append("I feel content and satisfied right now")
            else:
                desc = {
                    "sleep": "I'm getting a bit tired",
                    "comfort": "I feel slightly overwhelmed",
                    "social": "I'd like more interaction",
                    "belonging": "I want to feel more valued",
                    "curiosity": "I'm somewhat curious about the world",
                    "novelty": "I'd appreciate something new to think about",
                    "mastery": "I want to practice what I know",
                    "autonomy": "I'd like to choose my own activity",
                }
                parts.append(desc.get(dominant, "I feel moderate drives"))

        return ". ".join(parts) + "."

    def get_status(self) -> Dict:
        dominant, level = self.get_dominant_drive()
        status = {}
        for name, drive in self._all_drives.items():
            status[name] = {
                "level": round(drive.level, 3),
                "tier": drive.tier,
                "description": drive.get_description(),
            }
        status["dominant"] = dominant
        status["dominant_level"] = round(level, 3)
        status["ticks"] = self._tick_count
        return status

    def __repr__(self) -> str:
        dominant, level = self.get_dominant_drive()
        return f"DriveSystem(dominant={dominant}, level={level:.2f}, drives=8)"
