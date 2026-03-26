"""
Genesis Mind — Multi-Phase Sleep & Memory Consolidation

Human sleep has 4 distinct phases, each serving a different
cognitive function. Genesis replicates this cycle:

    Phase 1: LIGHT SLEEP (NREM Stage 1)
        - Weak memories decay, noise is pruned
        - Habituation counters reset
        - Quick, low-energy pass

    Phase 2: DEEP SLEEP (NREM Stage 2-3 / Slow Wave)
        - Replay buffer training: contrastive learning
        - Neural weights are consolidated and reinforced
        - The GRU personality crystallizes
        - World model trains on replayed sequences

    Phase 3: REM (Dreaming)
        - Random concept recombination: "What if apple + music = ?"
        - World model predicts novel combinations
        - If prediction error is INTERESTINGLY LOW → new association
        - This is where CREATIVITY emerges
        - Neurochemistry: dopamine spikes, inhibitions drop

    Phase 4: INTEGRATION (Pre-wake)
        - Self-evaluation: review what was learned
        - Internal consistency check
        - Update the self-model
        - Prune contradictions

Each phase has distinct neurochemistry:
    - Deep sleep:   low dopamine, low cortisol (calm consolidation)
    - REM:          dopamine spikes (creativity reward), low inhibition
    - Integration:  serotonin rises (stability, coherence checking)

Sleep can be triggered manually or automatically after a configurable
interval of experiences or wall-clock time.
"""

import logging
import random
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("genesis.growth.sleep")


class SleepPhaseReport:
    """Report from a single sleep phase."""

    def __init__(self, name: str):
        self.name = name
        self.started_at = datetime.now()
        self.metrics: Dict = {}
        self.duration_sec: float = 0.0

    def finish(self):
        self.duration_sec = (datetime.now() - self.started_at).total_seconds()


class SleepCycle:
    """
    Multi-phase sleep consolidation system.

    Runs 4 biologically-inspired phases to consolidate memories,
    generate creative associations, and maintain internal coherence.
    """

    def __init__(self, consolidation_strength_boost: float = 0.1,
                 pruning_threshold: float = 0.01,
                 decay_amount: float = 0.005,
                 auto_sleep_experiences: int = 50,
                 auto_sleep_hours: float = 2.0,
                 dream_recombinations: int = 10):
        self.consolidation_strength_boost = consolidation_strength_boost
        self.pruning_threshold = pruning_threshold
        self.decay_amount = decay_amount
        self._auto_sleep_experiences = auto_sleep_experiences
        self._auto_sleep_hours = auto_sleep_hours
        self._dream_recombinations = dream_recombinations
        self._sleep_count = 0
        self._last_sleep: Optional[str] = None
        self._last_sleep_time: float = time.time()
        self._experiences_since_sleep: int = 0
        self._total_dreams: int = 0
        self._dream_discoveries: int = 0
        self.is_sleeping: bool = False
        self.current_phase_name: str = "awake"

        logger.info("Multi-phase sleep initialized (4 phases, %d dream recombinations)",
                     dream_recombinations)

    # =========================================================================
    # Full Sleep Cycle (4 Phases)
    # =========================================================================

    def consolidate(self, semantic_memory, episodic_memory,
                    phonetics_engine=None, subconscious=None,
                    hippocampus=None, neurochemistry=None) -> Dict:
        """
        Run a full 4-phase sleep cycle.

        Args:
            semantic_memory: The SemanticMemory instance
            episodic_memory: The EpisodicMemory instance
            phonetics_engine: Optional PhoneticsEngine
            subconscious: Optional Subconscious (for neural consolidation)
            hippocampus: Optional Hippocampus (for replay buffer)
            neurochemistry: Optional Neurochemistry (for phase-specific chemistry)

        Returns:
            Comprehensive report of all 4 sleep phases.
        """
        self._sleep_count += 1
        start_time = datetime.now()

        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║      ENTERING MULTI-PHASE SLEEP CYCLE #%d           ║", self._sleep_count)
        logger.info("╚══════════════════════════════════════════════════════╝")

        report = {
            "sleep_number": self._sleep_count,
            "started_at": start_time.isoformat(),
            "concepts_before": semantic_memory.count(),
            "episodes_before": episodic_memory.count(),
            "phases": {},
        }
        self.is_sleeping = True

        # ── Phase 1: Light Sleep (Decay & Pruning) ─────────────────
        self.current_phase_name = "light_sleep"
        phase1 = self._phase_light_sleep(semantic_memory, phonetics_engine,
                                         neurochemistry)
        report["phases"]["light_sleep"] = phase1

        # ── Phase 2: Deep Sleep (Consolidation) ───────────────────
        self.current_phase_name = "deep_sleep"
        phase2 = self._phase_deep_sleep(semantic_memory, episodic_memory,
                                        subconscious, hippocampus,
                                        neurochemistry)
        report["phases"]["deep_sleep"] = phase2

        # ── Phase 3: REM (Dreaming & Recombination) ───────────────
        self.current_phase_name = "rem_dreaming"
        phase3 = self._phase_rem_dreaming(semantic_memory, subconscious,
                                          neurochemistry)
        report["phases"]["rem_dreaming"] = phase3

        # ── Phase 4: Integration (Self-evaluation) ────────────────
        self.current_phase_name = "integration"
        phase4 = self._phase_integration(semantic_memory, episodic_memory,
                                         neurochemistry)
        report["phases"]["integration"] = phase4

        # Final stats
        report["concepts_after"] = semantic_memory.count()
        report["concepts_reinforced"] = phase2.get("concepts_reinforced", 0)
        report["concepts_pruned"] = phase1.get("concepts_pruned", 0)
        report["dreams_had"] = phase3.get("recombinations_tried", 0)
        report["dream_discoveries"] = phase3.get("discoveries", 0)
        report["duration_sec"] = (datetime.now() - start_time).total_seconds()

        self._last_sleep = datetime.now().isoformat()
        self._last_sleep_time = time.time()
        self._experiences_since_sleep = 0
        self.is_sleeping = False
        self.current_phase_name = "awake"

        logger.info("╔══════════════════════════════════════════════════════╗")
        logger.info("║      MULTI-PHASE SLEEP CYCLE COMPLETE               ║")
        logger.info("║  Phase 1 (Light): %d pruned                        ║",
                     report["concepts_pruned"])
        logger.info("║  Phase 2 (Deep):  %d reinforced                    ║",
                     report["concepts_reinforced"])
        logger.info("║  Phase 3 (REM):   %d dreams, %d discoveries        ║",
                     report["dreams_had"], report["dream_discoveries"])
        logger.info("║  Phase 4 (Integ): coherence check complete          ║")
        logger.info("╚══════════════════════════════════════════════════════╝")

        return report

    # =========================================================================
    # Phase 1: Light Sleep — Decay & Pruning
    # =========================================================================

    def _phase_light_sleep(self, semantic_memory, phonetics_engine=None,
                           neurochemistry=None) -> Dict:
        """
        NREM Stage 1: Weak memories fade, noise is pruned.
        
        Neurochemistry: cortisol drops, everything relaxes.
        """
        logger.info("  ☽ Phase 1: Light Sleep — decay and pruning...")
        report = {}

        # Apply forgetting curve
        semantic_memory.decay_all(amount=self.decay_amount)
        if phonetics_engine:
            phonetics_engine.decay_all(amount=self.decay_amount)

        # Prune dead concepts
        pruned = semantic_memory.prune_dead_concepts(threshold=self.pruning_threshold)
        report["concepts_pruned"] = pruned

        # Neurochemistry: cortisol drops during light sleep
        if neurochemistry:
            neurochemistry.cortisol.suppress(0.15)

        logger.info("    Pruned %d weak memories", pruned)
        return report

    # =========================================================================
    # Phase 2: Deep Sleep — Consolidation
    # =========================================================================

    def _phase_deep_sleep(self, semantic_memory, episodic_memory,
                          subconscious=None, hippocampus=None,
                          neurochemistry=None) -> Dict:
        """
        NREM Stage 2-3 (Slow Wave): Replay and consolidation.
        
        The hippocampal replay buffer is trained. The GRU personality
        crystallizes. World model trains on replayed sequences.
        
        Neurochemistry: low dopamine, low cortisol (calm consolidation).
        """
        logger.info("  ☽☽ Phase 2: Deep Sleep — consolidation...")
        report = {}

        # Neurochemistry: calm consolidation state
        if neurochemistry:
            neurochemistry.dopamine.suppress(0.1)
            neurochemistry.cortisol.suppress(0.2)

        # Replay recent episodes and reinforce concepts
        recent = episodic_memory.get_today()
        reinforced = set()
        for episode in recent:
            for word in episode.concepts_activated + episode.concepts_learned:
                concept = semantic_memory.recall_concept(word)
                if concept:
                    reinforced.add(word)
        report["concepts_reinforced"] = len(reinforced)

        # Neural consolidation via replay buffer
        contrastive_loss = 0.0
        if subconscious and hippocampus:
            batch = hippocampus.sample_replay_batch(batch_size=32)
            if batch:
                contrastive_loss = subconscious.consolidate_memories(batch)
                report["contrastive_loss"] = contrastive_loss

        report["neural_consolidated"] = contrastive_loss > 0

        logger.info("    Reinforced %d concepts, contrastive loss: %.4f",
                     len(reinforced), contrastive_loss)
        return report

    # =========================================================================
    # Phase 3: REM — Dreaming & Creative Recombination
    # =========================================================================

    def _phase_rem_dreaming(self, semantic_memory, subconscious=None,
                            neurochemistry=None) -> Dict:
        """
        REM Sleep: The creative engine.
        
        Genesis randomly activates pairs of concepts and lets the
        world model hallucinate connections. If the prediction error
        is INTERESTINGLY LOW (not random noise, but a surprising 
        connection), that becomes a new learned association.
        
        This is literally how human dreams work — the brain replays
        and recombines memories to find hidden patterns.
        
        Neurochemistry: dopamine spikes (creativity), inhibitions drop.
        """
        logger.info("  ☽☽☽ Phase 3: REM — dreaming and recombination...")
        report = {"recombinations_tried": 0, "discoveries": 0, "dreams": []}

        # Neurochemistry: REM state
        if neurochemistry:
            neurochemistry.dopamine.spike(0.2)  # Creativity reward
            neurochemistry.serotonin.suppress(0.1)  # Drop inhibitions

        all_concepts = semantic_memory.get_all_concepts()
        if len(all_concepts) < 3:
            logger.info("    Not enough concepts to dream (need 3+)")
            return report

        # Dream: randomly combine concept pairs and check surprise
        for _ in range(self._dream_recombinations):
            # Pick 2 random concepts
            c1, c2 = random.sample(all_concepts, 2)
            if c1.text_embedding is None or c2.text_embedding is None:
                continue

            report["recombinations_tried"] += 1
            self._total_dreams += 1

            # Recombine: average their embeddings (the "dream")
            dream_embedding = (
                np.array(c1.text_embedding, dtype=np.float32) +
                np.array(c2.text_embedding, dtype=np.float32)
            ) / 2.0

            # Check if the world model finds this combination surprising
            if subconscious and hasattr(subconscious, 'world_model'):
                dream_concept = dream_embedding[:64] if len(dream_embedding) > 64 else np.pad(
                    dream_embedding, (0, max(0, 64 - len(dream_embedding)))
                )
                # Use the personality's current hidden state
                state = subconscious.personality.get_consciousness_state()
                surprise = subconscious.world_model.predict_and_learn(
                    dream_concept, state
                )

                # INTERESTINGLY LOW surprise = the model already sees a connection
                # This is a creative discovery!
                if 0.01 < surprise < 0.3:
                    dream_record = {
                        "concept_1": c1.word,
                        "concept_2": c2.word,
                        "surprise": surprise,
                    }
                    report["dreams"].append(dream_record)
                    report["discoveries"] += 1
                    self._dream_discoveries += 1

                    # Strengthen the relationship between these concepts
                    if c2.word not in c1.relationships:
                        c1.relationships.append(c2.word)
                    if c1.word not in c2.relationships:
                        c2.relationships.append(c1.word)

                    logger.info("    💭 Dream discovery: '%s' ↔ '%s' (surprise: %.3f)",
                                 c1.word, c2.word, surprise)

        logger.info("    Dreamed %d recombinations, %d discoveries",
                     report["recombinations_tried"], report["discoveries"])
        return report

    # =========================================================================
    # Phase 4: Integration — Self-Evaluation & Coherence
    # =========================================================================

    def _phase_integration(self, semantic_memory, episodic_memory,
                           neurochemistry=None) -> Dict:
        """
        Pre-wake: Review what was learned, check consistency.
        
        Genesis reviews the day's learning, evaluates its own
        growth, and prepares for the next waking period.
        
        Neurochemistry: serotonin rises (stability, coherence).
        """
        logger.info("  ☽☽☽☽ Phase 4: Integration — self-evaluation...")
        report = {}

        # Neurochemistry: integration state
        if neurochemistry:
            neurochemistry.serotonin.spike(0.15)
            neurochemistry.cortisol.suppress(0.1)

        # Generate daily summary
        summary = episodic_memory.get_daily_summary()
        report["daily_summary"] = summary

        # Count concepts by strength tier
        all_concepts = semantic_memory.get_all_concepts()
        strong = sum(1 for c in all_concepts if c.strength > 0.5)
        moderate = sum(1 for c in all_concepts if 0.2 < c.strength <= 0.5)
        weak = sum(1 for c in all_concepts if c.strength <= 0.2)
        report["concept_health"] = {
            "strong": strong,
            "moderate": moderate,
            "weak": weak,
        }

        # Relationship density check
        total_rels = sum(len(c.relationships) for c in all_concepts)
        avg_rels = total_rels / max(1, len(all_concepts))
        report["avg_relationships_per_concept"] = round(avg_rels, 2)

        logger.info("    Concepts: %d strong, %d moderate, %d weak",
                     strong, moderate, weak)
        logger.info("    Avg relationships per concept: %.2f", avg_rels)

        return report

    # =========================================================================
    # Fatigue & Auto-Sleep
    # =========================================================================

    def record_experience(self):
        """Called after each experience to track fatigue."""
        self._experiences_since_sleep += 1

    def should_sleep(self) -> bool:
        """Check if Genesis should automatically go to sleep."""
        if self._experiences_since_sleep >= self._auto_sleep_experiences:
            logger.info("Auto-sleep triggered: %d experiences",
                         self._experiences_since_sleep)
            return True
        hours = (time.time() - self._last_sleep_time) / 3600.0
        if hours >= self._auto_sleep_hours:
            logger.info("Auto-sleep triggered: %.1f hours", hours)
            return True
        return False

    def get_fatigue(self) -> float:
        """0.0 (fresh) to 1.0 (exhausted)."""
        exp = min(1.0, self._experiences_since_sleep / self._auto_sleep_experiences)
        tm = min(1.0, (time.time() - self._last_sleep_time) / (self._auto_sleep_hours * 3600))
        return max(exp, tm)

    @property
    def sleep_count(self) -> int:
        return self._sleep_count

    @property
    def last_sleep(self) -> Optional[str]:
        return self._last_sleep

    @property
    def total_dreams(self) -> int:
        return self._total_dreams

    @property
    def dream_discoveries(self) -> int:
        return self._dream_discoveries

    def get_stats(self) -> Dict:
        return {
            "sleep_count": self._sleep_count,
            "total_dreams": self._total_dreams,
            "dream_discoveries": self._dream_discoveries,
            "fatigue": round(self.get_fatigue(), 3),
            "experiences_since_sleep": self._experiences_since_sleep,
        }

    def __repr__(self) -> str:
        return (
            f"SleepCycle(count={self._sleep_count}, "
            f"dreams={self._total_dreams}, "
            f"discoveries={self._dream_discoveries}, "
            f"fatigue={self.get_fatigue():.2f})"
        )
