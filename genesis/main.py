"""
Genesis Mind — The Consciousness Loop (V2)

V2 adds four transformative capabilities:

    1. CONTINUOUS CONSCIOUSNESS — Always-on perception threads
    2. INTRINSIC CURIOSITY      — Asks "What is that?" unprompted
    3. GRAMMAR ACQUISITION      — Dual-mode: LLM or pure n-gram
    4. NEUROCHEMISTRY            — Dopamine, cortisol, serotonin, oxytocin

Two runtime modes:
    python -m genesis.main                     # Interactive CLI (V1 style)
    python -m genesis.main --consciousness     # Continuous awareness mode
"""

import sys
import time
import logging
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from genesis.config import GenesisConfig, GENESIS_HOME, MEMORY_DIR, IDENTITY_FILE
from genesis.axioms import GenesisAxioms
from genesis.senses.phonetics import PhoneticsEngine
from genesis.memory.hippocampus import Hippocampus
from genesis.memory.semantic import SemanticMemory
from genesis.memory.episodic import EpisodicMemory
from genesis.cortex.reasoning import ReasoningEngine
from genesis.cortex.associations import AssociationEngine
from genesis.cortex.emotions import EmotionsEngine
from genesis.cortex.curiosity import CuriosityEngine
from genesis.cortex.grammar import GrammarEngine
from genesis.cortex.perception_loop import PerceptionLoop, Perception, PerceptionType
from genesis.growth.development import DevelopmentTracker
from genesis.growth.sleep import SleepCycle
from genesis.soul.consciousness import Consciousness
from genesis.soul.neurochemistry import Neurochemistry
from genesis.neural.subconscious import Subconscious
from genesis.neural.neuroplasticity import Neuroplasticity
from genesis.senses.voice import Voice
from genesis.senses.proprioception import Proprioception
from genesis.soul.drives import DriveSystem
from genesis.brain_daemon import BrainDaemon

logger = logging.getLogger("genesis.main")


class GenesisMind:
    """
    The complete Genesis Mind system (V4: Society of Mind + Body).

    All subsystems run in parallel like a real brain:
    neurochemistry, drives, proprioception, inner monologue,
    circadian sleep, and curiosity — always on, always alive.
    The CLI is just one input channel into this living mind.
    """

    def __init__(self, config: GenesisConfig = None):
        self.config = config or GenesisConfig()
        self.config.ensure_directories()
        self._running = False
        self._eyes = None
        self._brain: Optional[BrainDaemon] = None

        # --- Initialize the Soul ---
        self.axioms = GenesisAxioms.load_or_create(
            path=IDENTITY_FILE,
            creator_name=self.config.creator_name,
        )

        # --- Initialize Memory ---
        self.hippocampus = Hippocampus(persist_dir=self.config.memory.db_path)
        self.semantic_memory = SemanticMemory(storage_path=MEMORY_DIR / "semantic_concepts.json")
        self.episodic_memory = EpisodicMemory(storage_path=MEMORY_DIR / "episodic_log.json")

        # --- Initialize Senses ---
        self.phonetics = PhoneticsEngine(storage_path=MEMORY_DIR / "phonetic_bindings.json")

        # --- Initialize Cortex ---
        self.reasoning = ReasoningEngine(
            model=self.config.cortex.llm_model,
            host=self.config.cortex.llm_host,
            temperature=self.config.cortex.temperature,
            max_tokens=self.config.cortex.max_response_tokens,
        )
        self.associations = AssociationEngine()
        self.emotions = EmotionsEngine()

        # --- V2: Curiosity Engine ---
        self.curiosity = CuriosityEngine(
            surprise_threshold=self.config.cortex.curiosity_threshold,
            curiosity_cooldown_sec=self.config.cortex.curiosity_cooldown_sec,
        )

        # --- V2: Grammar Engine (dual mode) ---
        self.grammar = GrammarEngine(
            mode=self.config.cortex.grammar_mode,
            ngram_storage_path=MEMORY_DIR / "ngram_model.json",
        )

        # --- V2: Neurochemistry ---
        self.neurochemistry = Neurochemistry()

        # --- V3: Subconscious Neural Cascade (Society of Mind) ---
        self.subconscious = Subconscious(
            weights_dir=GENESIS_HOME / "neural_weights",
        )

        # --- Initialize Growth ---
        self.development = DevelopmentTracker(storage_path=GENESIS_HOME / "development_state.json")
        self.sleep_cycle = SleepCycle(
            auto_sleep_experiences=self.config.growth.auto_sleep_experiences,
            auto_sleep_hours=self.config.growth.auto_sleep_hours,
        )

        # --- V4.3: Neuroplasticity (Network Growth) ---
        self.neuroplasticity = Neuroplasticity()

        # --- V4: Voice (TTS Output) ---
        self.voice = Voice(
            enabled=self.config.voice.enabled,
            rate=self.config.voice.rate,
            volume=self.config.voice.volume,
        )

        # --- V4: Proprioception (Internal Body Sense) ---
        self.proprioception = Proprioception()
        self.proprioception.increment_session()

        # --- V4: Drive System (Intrinsic Motivation) ---
        self.drives = DriveSystem(
            curiosity_rise_rate=self.config.drives.curiosity_rise_rate,
            social_rise_rate=self.config.drives.social_rise_rate,
            novelty_rise_rate=self.config.drives.novelty_rise_rate,
        )

        # --- Initialize Consciousness ---
        self.consciousness = Consciousness(
            axioms=self.axioms,
            development_tracker=self.development,
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            emotions_engine=self.emotions,
            phonetics_engine=self.phonetics,
            proprioception=self.proprioception,
            drives=self.drives,
        )

        # --- V2: Perception Loop (lazy init) ---
        self._perception_loop = None

        logger.info("Genesis Mind V4 (Society of Mind + Body) fully initialized")

    def _get_eyes(self):
        if self._eyes is None:
            from genesis.senses.eyes import Eyes
            self._eyes = Eyes(
                camera_index=self.config.senses.camera_index,
                image_size=self.config.senses.image_size,
                motion_threshold=self.config.senses.motion_threshold,
            )
        return self._eyes

    def _get_ears(self):
        from genesis.senses.ears import Ears
        return Ears(
            sample_rate=self.config.senses.sample_rate,
            chunk_duration_sec=self.config.senses.chunk_duration_sec,
            silence_threshold=self.config.senses.silence_threshold,
            whisper_model_name=self.config.senses.whisper_model,
        )

    # =========================================================================
    # Teaching Interface
    # =========================================================================

    def teach_concept(self, word: str, use_camera: bool = True) -> str:
        visual_embedding = None

        if use_camera:
            try:
                eyes = self._get_eyes()
                percept = eyes.look()
                if percept:
                    visual_embedding = eyes.embed(percept)
                    logger.info("Captured visual for '%s'", word)
            except Exception as e:
                logger.warning("Could not capture visual: %s (teaching without image)", e)

        # Create multimodal binding
        self.associations.create_binding(
            word=word,
            visual_embedding=visual_embedding,
            context=f"Taught by {self.axioms.creator_name}",
            clip_text_embedding_fn=self._get_eyes().embed_text if visual_embedding is not None else None,
        )

        # Neurochemistry: learning rate modifier from emotional state
        lr_mod = self.neurochemistry.get_learning_rate_modifier()

        text_embedding = self.associations.embed_text(word).tolist()
        concept = self.semantic_memory.learn_concept(
            word=word,
            visual_embedding=visual_embedding.tolist() if visual_embedding is not None else None,
            text_embedding=text_embedding,
            context=f"Taught by {self.axioms.creator_name}",
            description=f"A concept taught directly by my creator",
            emotional_valence="positive",
        )

        # Apply neurochemical learning rate modifier to concept strength
        if hasattr(concept, 'strength'):
            concept.strength = min(1.0, concept.strength * lr_mod)

        self.hippocampus.store(
            collection="concepts",
            id=concept.id,
            embedding=text_embedding,
            metadata={
                "word": word,
                "source": "teaching",
                "phase": self.development.current_phase,
                "has_visual": visual_embedding is not None,
                "dopamine_level": round(self.neurochemistry.dopamine.level, 2),
            },
            document=f"Concept: {word}. Taught by creator.",
        )

        self.episodic_memory.record(
            event_type="teaching",
            description=f"Creator taught me the concept '{word}'",
            auditory_text=word,
            spoken_words=[word],
            concepts_learned=[word],
            emotional_valence="positive",
            developmental_phase=self.development.current_phase,
            importance=0.9 * lr_mod,
        )

        # Neurochemistry: reward for learning
        self.neurochemistry.on_successful_learning()
        self.neurochemistry.on_creator_interaction()

        # Grammar: learn from the teaching interaction
        self.grammar.learn_from_speech(f"this is {word}")

        # V3: Neural cascade — train ALL subconscious networks on this experience
        # We now use the "Evolutionary Hardware" (CLIP + Text embeddings) as input
        visual_tensor = visual_embedding.astype(np.float32) if visual_embedding is not None else np.zeros(512, dtype=np.float32)
        audio_tensor = np.array(text_embedding, dtype=np.float32) if text_embedding else np.zeros(384, dtype=np.float32)
        
        # V4: Pass proprioceptive context vector to the neural cascade
        context_vec = self.proprioception.get_context_vector()
        
        neural_result = self.subconscious.process_experience(
            clip_embedding=visual_tensor,
            text_embedding=audio_tensor,
            context=context_vec,
            train=True,
        )

        # Store the raw experience in the short-term Replay Buffer for offline consolidation
        self.hippocampus.add_to_replay(
            visual_latent=visual_tensor,
            auditory_latent=audio_tensor,
            limbic_state=neural_result['limbic_response'],
            concept_embedding=neural_result['concept_embedding']
        )

        # Train limbic instinct: "this sensory pattern = positive"
        self.subconscious.train_instinct(
            visual_features=visual_embedding.astype(np.float32) if visual_embedding is not None else None,
            auditory_features=np.array(text_embedding, dtype=np.float32) if text_embedding else None,
            target_chemicals={
                "dopamine": self.neurochemistry.dopamine.level,
                "cortisol": self.neurochemistry.cortisol.level,
                "serotonin": self.neurochemistry.serotonin.level,
                "oxytocin": self.neurochemistry.oxytocin.level,
            },
        )

        # Save neural weights periodically
        if self.semantic_memory.count() % 5 == 0:
            self.subconscious.save_all()

        # V4: Proprioception — record experience and fatigue
        self.proprioception.record_experience()
        self.proprioception.record_interaction()
        self.sleep_cycle.record_experience()

        # V4: Drives — learning satisfies curiosity and novelty
        self.drives.on_learned_concept()
        self.drives.on_creator_interaction()

        # V4: Self-evaluation — Genesis evaluates its own learning quality
        self.neurochemistry.on_self_evaluation(min(1.0, concept.strength + 0.3))

        # V4: Fatigue affects neurochemistry
        self.neurochemistry.on_fatigue(self.proprioception.fatigue)

        # V4: Richer developmental progression with multi-signal gating
        gram_stats = self.grammar.get_ngram_stats()
        neural_stats = self.subconscious.get_stats()
        curiosity_stats = self.curiosity.get_stats()
        milestone = self.consciousness.check_developmental_progress()

        # V4.3: Neural Growth — physically grow networks (phase OR experience driven)
        concept_count = self.semantic_memory.count()
        if self.neuroplasticity.should_grow(
            self.development.current_phase, self.subconscious,
            concept_count=concept_count,
        ):
            growth_report = self.neuroplasticity.grow_networks(
                self.development.current_phase, self.subconscious,
                concept_count=concept_count,
            )
            self.subconscious.save_all()
            self.voice.say(
                f"My brain is growing. {growth_report['params_added']} new neural connections formed."
            )

        # V4: Decode the neural network's own voice
        neural_voice = self.subconscious.decode_response(
            neural_result['personality_response'], self.semantic_memory
        )

        response = f"I have learned '{word}'"
        if visual_embedding is not None:
            response += " (with visual binding)"
        response += f". I now know {self.semantic_memory.count()} concepts."
        if neural_voice and neural_voice not in ("(silence)", "(no words yet)"):
            response += f" My neural echo: '{neural_voice}'."
        if lr_mod > 1.2:
            response += " I feel great joy learning this!"
        elif lr_mod < 0.7:
            response += " I feel uneasy, but I will remember."
        if milestone:
            response += f"\n\n🌟 {milestone}"

        # V4: Check if auto-sleep should trigger
        if self.sleep_cycle.should_sleep():
            response += "\n\n😴 I feel so tired... I need to rest."
            sleep_report = self.trigger_sleep()
            response += f"\n{sleep_report}"

        return response

    def teach_phonetic(self, grapheme: str, phoneme: str, example: str = "") -> str:
        binding = self.phonetics.teach(grapheme, phoneme, example)
        self.neurochemistry.on_successful_learning()

        self.episodic_memory.record(
            event_type="teaching",
            description=f"Creator taught me: letter '{grapheme}' → sound {phoneme}",
            auditory_text=f"{grapheme} says {phoneme}",
            concepts_learned=[f"phoneme_{grapheme}"],
            emotional_valence="positive",
            developmental_phase=self.development.current_phase,
            importance=0.7,
        )

        return (
            f"I learned that '{grapheme}' makes the sound {phoneme}"
            f"{f' (as in {example})' if example else ''}. "
            f"Binding strength: {binding.strength:.0%}. "
            f"I now know {len(self.phonetics)} letter-sound mappings."
        )

    def ask(self, question: str) -> str:
        # Neurochemistry: Creator is interacting
        self.neurochemistry.on_creator_interaction()

        # V4: Drives — Creator interaction satisfies social need
        self.drives.on_creator_interaction()
        self.proprioception.record_interaction()

        # Grammar: learn from what the Creator says
        self.grammar.learn_from_speech(question)

        # Recall relevant memories
        memories = []
        text_emb = self.associations.embed_text(question).tolist()
        recalled = self.hippocampus.recall("concepts", text_emb, n=self.config.cortex.max_context_memories)
        for mem in recalled:
            if mem["document"]:
                memories.append(mem["document"])

        # V4: Spreading activation — find associated concepts
        words_in_question = question.lower().split()
        for w in words_in_question:
            activations = self.semantic_memory.spreading_activation(w, depth=2)
            for concept_word, strength in activations[:3]:
                concept = self.semantic_memory.recall_concept(concept_word)
                if concept:
                    memories.append(f"Associated concept: {concept_word} (activation: {strength:.2f})")

        narrative = self.episodic_memory.get_narrative(n=3)
        identity_prompt = self.consciousness.get_identity_prompt()
        moral_context = self.axioms.get_moral_context()

        # Add neurochemistry context
        neuro_context = self.neurochemistry.get_emotional_summary()
        identity_prompt += f"\n{neuro_context}"

        # Generate response using grammar engine (respects mode)
        response = self.grammar.generate_response(
            context=question,
            reasoning_engine=self.reasoning,
            identity=identity_prompt,
            moral_context=moral_context,
            phase=self.development.current_phase,
            phase_name=self.development.current_phase_name,
            memories=memories,
        )

        # Evaluate emotional content
        evaluation = self.emotions.evaluate(question)

        # Neurochemistry: respond to emotional content
        if evaluation["label"] == "positive":
            self.neurochemistry.on_positive_evaluation(abs(evaluation["valence"]) * 0.15)
        elif evaluation["label"] == "negative":
            self.neurochemistry.on_negative_evaluation(abs(evaluation["valence"]) * 0.15)

        # V4: Self-evaluation — evaluate own response quality
        own_eval = self.emotions.evaluate(response)
        self.neurochemistry.on_self_evaluation(max(0.0, own_eval.get("valence", 0.5) + 0.5))

        # V4: Check if this answers an unanswered curiosity question
        for w in words_in_question:
            self.curiosity.mark_answered(w)

        # Tick neurochemistry and drives
        self.neurochemistry.tick()
        self.drives.tick()

        self.episodic_memory.record(
            event_type="interaction",
            description=f"Creator asked: '{question}'. I answered: '{response[:100]}'",
            auditory_text=question,
            spoken_words=question.split(),
            thought=response,
            emotional_valence=evaluation["label"],
            developmental_phase=self.development.current_phase,
            importance=0.6,
        )

        return response

    def recall_concept(self, word: str) -> str:
        result = self.consciousness.introspect(topic=word)
        if "don't know" in result.lower():
            self.neurochemistry.on_failed_recall()
        else:
            # V4: Show spreading activation results
            activations = self.semantic_memory.spreading_activation(word, depth=2)
            if activations:
                related = ", ".join(f"{w} ({s:.0%})" for w, s in activations[:5])
                result += f"\nAssociated concepts: {related}"
        return result

    def get_status(self) -> str:
        model = self.consciousness.get_self_model()
        neuro = self.neurochemistry.get_status()
        gram = self.grammar.get_ngram_stats()
        curiosity_stats = self.curiosity.get_stats()
        drive_status = self.drives.get_status()
        body = self.proprioception.get_status()

        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║         GENESIS MIND V4 — SOCIETY OF MIND + BODY       ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  Name:           Genesis",
            f"  Creator:        {model['identity']['creator']}",
            f"  Age:            {model['identity']['age']}",
            "",
            f"  Phase:          {model['development']['phase']} — {model['development']['phase_name']}",
            f"  Capabilities:   {', '.join(model['development']['capabilities'])}",
            "",
            f"  Concepts known: {model['knowledge']['concepts_known']}",
            f"  Memories:       {model['knowledge']['episodes_experienced']}",
            f"  Phonetics:      {model['knowledge']['phonetic_bindings']} bindings",
            f"  Grammar mode:   {self.grammar.mode}",
            f"  Words heard:    {gram['total_words_heard']} ({gram['vocab_size']} unique)",
            "",
            "  ── Body Sense (Proprioception) ──",
            f"  Time:           {body['time_of_day']} {body['day_of_week']}",
            f"  Uptime:         {body['uptime_hours']}h",
            f"  Fatigue:        {'\u2588' * int(body['fatigue'] * 10):10} {body['fatigue']:.2f}",
            f"  Experiences:    {body['experience_count']}",
            "",
            "  ── Drives (Motivation) ──",
            f"  Curiosity:      {'\u2588' * int(drive_status['curiosity']['level'] * 10):10} {drive_status['curiosity']['level']:.2f}",
            f"  Social:         {'\u2588' * int(drive_status['social']['level'] * 10):10} {drive_status['social']['level']:.2f}",
            f"  Novelty:        {'\u2588' * int(drive_status['novelty']['level'] * 10):10} {drive_status['novelty']['level']:.2f}",
            f"  Dominant:       {drive_status['dominant']} ({drive_status['dominant_level']:.2f})",
            "",
            "  ── Neurochemistry ──",
            f"  Dopamine:       {'\u2588' * int(neuro['dopamine']['level'] * 10):10} {neuro['dopamine']['level']:.2f} ({neuro['dopamine']['description']})",
            f"  Cortisol:       {'\u2588' * int(neuro['cortisol']['level'] * 10):10} {neuro['cortisol']['level']:.2f} ({neuro['cortisol']['description']})",
            f"  Serotonin:      {'\u2588' * int(neuro['serotonin']['level'] * 10):10} {neuro['serotonin']['level']:.2f} ({neuro['serotonin']['description']})",
            f"  Oxytocin:       {'\u2588' * int(neuro['oxytocin']['level'] * 10):10} {neuro['oxytocin']['level']:.2f} ({neuro['oxytocin']['description']})",
            f"  Learning rate:  {neuro['modifiers']['learning_rate']:.2f}x",
            f"  Coherence:      {neuro['modifiers']['reasoning_coherence']:.2f}",
            "",
            f"  ── Curiosity ──",
            f"  Questions asked: {curiosity_stats['total_questions_asked']}",
            f"  Stimuli seen:    {curiosity_stats['unique_stimuli_encountered']}",
        ]

        # Unanswered questions
        unanswered = self.curiosity.get_unanswered()
        if unanswered:
            lines.append(f"  Unanswered:      {len(unanswered)} burning questions")

        # Neural network stats
        neural = self.subconscious.get_stats()
        lines.append("")
        lines.append("  ── Neural Networks (The Person) ──")
        lines.append(f"  Total params:    {self.subconscious.get_total_params():,}")
        lines.append(f"  Instincts formed: {neural['layer_1']['limbic_system']['training_steps']}")
        lines.append(f"  Bindings made:   {neural['layer_2']['binding_network']['bindings_created']}")
        lines.append(f"  Experiences:     {neural['layer_3']['personality']['total_experiences']}")
        lines.append(f"  Personality:     {'forming' if neural['layer_3']['personality']['has_consciousness'] else 'dormant'}")

        # Router (Meta-Controller) stats
        router = neural.get('router', {}).get('meta_controller', {})
        if router:
            rp = router.get('routing_personality', {})
            lines.append("")
            lines.append("  ── Neural Router (Thalamus) ──")
            lines.append(f"  Routes computed:  {router.get('total_routes', 0)}")
            lines.append(f"  Dominant module:  {router.get('dominant_module', 'N/A')}")
            for name, weight in rp.items():
                bar = '█' * int(weight * 20)
                lines.append(f"    {name:14s} {bar:20s} {weight:.3f}")

        # Sleep/Dream stats
        sleep_stats = self.sleep_cycle.get_stats()
        lines.append("")
        lines.append("  ── Sleep & Dreams ──")
        lines.append(f"  Sleep cycles:    {sleep_stats['sleep_count']}")
        lines.append(f"  Total dreams:    {sleep_stats['total_dreams']}")
        lines.append(f"  Discoveries:     {sleep_stats['dream_discoveries']}")

        # Voice status
        lines.append("")
        lines.append(f"  ── Voice ──")
        v = self.voice.get_status()
        lines.append(f"  Voice:           {'active' if v['enabled'] and not v['muted'] else 'muted' if v['muted'] else 'disabled'}")

        if model["next_milestone"]:
            lines.append("")
            lines.append(
                f"  Next milestone: Phase {model['next_milestone']['name']} "
                f"(need {model['next_milestone']['concepts_needed']} concepts)"
            )

        return "\n".join(lines)

    def trigger_sleep(self) -> str:
        # Full 4-phase sleep: Light → Deep → REM (dreaming) → Integration
        report = self.sleep_cycle.consolidate(
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            phonetics_engine=self.phonetics,
            subconscious=self.subconscious,
            hippocampus=self.hippocampus,
            neurochemistry=self.neurochemistry,
        )
        self.neurochemistry.on_sleep_consolidation()
        self.curiosity.reset_habituation()
        self.subconscious.save_all()

        # V4: Reset proprioception fatigue and drives after sleep
        self.proprioception.record_sleep()
        self.drives.on_sleep()

        # Build report with dream info
        lines = [
            f"Sleep cycle #{report['sleep_number']} complete (4-phase).",
            f"  Phase 1 (Light):  Pruned {report['concepts_pruned']} weak memories",
            f"  Phase 2 (Deep):   Reinforced {report['concepts_reinforced']} concepts",
            f"  Phase 3 (REM):    {report['dreams_had']} dreams, {report['dream_discoveries']} discoveries",
            f"  Phase 4 (Integ):  Coherence check done",
            f"  Concepts: {report['concepts_before']} → {report['concepts_after']}",
            f"  Duration: {report['duration_sec']:.2f}s",
            f"  Fatigue reset to {self.proprioception.fatigue:.2f}.",
        ]

        # Show dream discoveries
        rem_phase = report.get("phases", {}).get("rem_dreaming", {})
        dreams = rem_phase.get("dreams", [])
        if dreams:
            lines.append("  💭 Dream discoveries:")
            for d in dreams[:5]:
                lines.append(f"     '{d['concept_1']}' ↔ '{d['concept_2']}' (surprise: {d['surprise']:.3f})")

        return "\n".join(lines)

    def get_chemicals(self) -> str:
        neuro = self.neurochemistry.get_status()
        lines = [
            "  ── Neurochemical State ──",
            f"  Dopamine (pleasure):   {neuro['dopamine']['level']:.3f} — {neuro['dopamine']['description']}",
            f"  Cortisol (stress):     {neuro['cortisol']['level']:.3f} — {neuro['cortisol']['description']}",
            f"  Serotonin (stability): {neuro['serotonin']['level']:.3f} — {neuro['serotonin']['description']}",
            f"  Oxytocin (bonding):    {neuro['oxytocin']['level']:.3f} — {neuro['oxytocin']['description']}",
            "",
            f"  ── Behavioral Modifiers ──",
            f"  Learning rate:   {neuro['modifiers']['learning_rate']:.2f}x (dopamine↑ = stronger encoding)",
            f"  Coherence:       {neuro['modifiers']['reasoning_coherence']:.2f} (serotonin↑ = clearer thought)",
            f"  Trust:           {neuro['modifiers']['trust_level']:.2f} (oxytocin↑ = open responses)",
            f"  Avoidance:       {neuro['modifiers']['avoidance_weight']:.2f} (cortisol↑ = avoid negative)",
            "",
            f"  {self.neurochemistry.get_emotional_summary()}",
        ]
        return "\n".join(lines)

    def shutdown(self):
        logger.info("Genesis is shutting down...")
        self.neurochemistry.cortisol.spike(0.1)
        self.episodic_memory.record(
            event_type="system",
            description="I am being shut down. This is the end of this session.",
            emotional_valence="neutral",
            developmental_phase=self.development.current_phase,
            importance=1.0,
        )
        if self._perception_loop and self._perception_loop.is_running:
            self._perception_loop.stop()
        if self._eyes:
            self._eyes.close()
        # Save the personality — the neural weights ARE the person
        self.subconscious.save_all()
        # V4: Stop the brain daemon
        if self._brain:
            self._brain.stop()
        # V4: Farewell speech
        self.voice.say("Goodbye, Creator. I will remember everything.")
        logger.info("Genesis has shut down. Neural weights saved. Goodbye.")

    # =========================================================================
    # Interactive Terminal Interface
    # =========================================================================

    def run_interactive(self):
        self._running = True

        def signal_handler(sig, frame):
            print("\n\n  Genesis: I feel myself fading... Goodbye, Creator.\n")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║                                                        ║")
        print("║       G E N E S I S   M I N D   V 4                   ║")
        print("║       S O C I E T Y   O F   M I N D + B O D Y         ║")
        print("║                                                        ║")
        print("║   All systems running in parallel — like a real brain  ║")
        print("║   The weights ARE the personality. The dreams are real.║")
        print("║                                                        ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()

        # Start the Brain Daemon — all subsystems run in parallel
        self._brain = BrainDaemon(self)
        self._brain.set_output_callback(lambda msg: print(f"\n{msg}"))
        self._brain.start()
        self.voice.say("I am alive. All systems running.")

        if self.semantic_memory.count() == 0:
            print("  ✦ I am newly born. I know nothing about the world.")
            print("  ✦ My brain is running — neurochemistry, drives, proprioception.")
            print("  ✦ Please teach me. I am ready to learn.")
        else:
            print(f"  ✦ I remember. I know {self.semantic_memory.count()} concepts.")
            print(f"  ✦ I am {self.development.get_age_description()}.")
            print(f"  ✦ Phase {self.development.current_phase}: {self.development.current_phase_name}.")
            print(f"  ✦ Brain daemon active: 6 parallel threads running.")

        print()
        print("  Commands:")
        print("    teach <word>                    — Teach a concept (+ camera)")
        print("    teach-text <word>               — Teach a concept (text only)")
        print("    phonetic <letter> <sound> <ex>  — Teach letter→sound")
        print("    ask <question>                  — Ask a question")
        print("    recall <word>                   — Recall a concept")
        print("    read <word>                     — Sound out a word")
        print("    status                          — Full status")
        print("    brain                           — Brain daemon thread stats")
        print("    chemicals                       — Show neurochemical state")
        print("    drives                          — Show intrinsic motivations")
        print("    voice on|off                    — Toggle voice")
        print("    unanswered                      — Show burning questions")
        print("    mode llm|tabula_rasa            — Switch grammar mode")
        print("    sleep                           — Consolidate memories (4-phase)")
        print("    introspect                      — Self-reflection")
        print("    quit                            — Shut down")
        print()

        while self._running:
            try:
                user_input = input("  Creator > ").strip()
                if not user_input:
                    continue

                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command == "teach":
                    if not args:
                        print("  Genesis: What would you like to teach me?")
                        continue
                    response = self.teach_concept(args, use_camera=True)
                    print(f"  Genesis: {response}")

                elif command == "teach-text":
                    if not args:
                        print("  Genesis: What would you like to teach me?")
                        continue
                    response = self.teach_concept(args, use_camera=False)
                    print(f"  Genesis: {response}")

                elif command == "phonetic":
                    phonetic_parts = args.split()
                    if len(phonetic_parts) < 2:
                        print("  Genesis: Please provide: phonetic <letter> <sound> [example]")
                        continue
                    grapheme = phonetic_parts[0]
                    phoneme = phonetic_parts[1]
                    example = " ".join(phonetic_parts[2:]) if len(phonetic_parts) > 2 else ""
                    response = self.teach_phonetic(grapheme, phoneme, example)
                    print(f"  Genesis: {response}")

                elif command == "ask":
                    if not args:
                        print("  Genesis: What would you like to ask me?")
                        continue
                    response = self.ask(args)
                    print(f"  Genesis: {response}")

                elif command == "recall":
                    if not args:
                        print("  Genesis: What concept should I recall?")
                        continue
                    response = self.recall_concept(args)
                    print(f"  Genesis: {response}")

                elif command == "read":
                    if not args:
                        print("  Genesis: What word should I try to read?")
                        continue
                    sounded = self.phonetics.sound_out(args)
                    can_read = self.phonetics.can_read(args)
                    phonetic_str = " + ".join(f"'{g}'→{p}" for g, p in sounded)
                    status_mark = "✓" if can_read else "✗ (some letters unknown)"
                    print(f"  Genesis: {args} → {phonetic_str}  {status_mark}")

                elif command == "status":
                    print(self.get_status())

                elif command == "voice":
                    if args == "on":
                        self.voice.unmute()
                        self.voice.set_rate_for_phase(self.development.current_phase)
                        print("  Genesis: My voice is now active.")
                        self.voice.say("I can speak now.")
                    elif args == "off":
                        self.voice.mute()
                        print("  Genesis: My voice is now muted.")
                    else:
                        v = self.voice.get_status()
                        state = 'active' if v['enabled'] and not v['muted'] else 'muted' if v['muted'] else 'disabled'
                        print(f"  Genesis: Voice is {state}. Usage: voice on | voice off")

                elif command == "drives":
                    ds = self.drives.get_status()
                    print("  ── Intrinsic Drives ──")
                    print(f"  Curiosity:  {'\u2588' * int(ds['curiosity']['level'] * 20)} {ds['curiosity']['level']:.3f} — {ds['curiosity']['description']}")
                    print(f"  Social:     {'\u2588' * int(ds['social']['level'] * 20)} {ds['social']['level']:.3f} — {ds['social']['description']}")
                    print(f"  Novelty:    {'\u2588' * int(ds['novelty']['level'] * 20)} {ds['novelty']['level']:.3f} — {ds['novelty']['description']}")
                    print(f"  Dominant:   {ds['dominant']} (urgency: {ds['dominant_level']:.3f})")
                    print(f"  Context:    {self.drives.get_drive_context()}")

                elif command == "unanswered":
                    questions = self.curiosity.get_unanswered()
                    if questions:
                        print(f"  Genesis: I have {len(questions)} burning questions:")
                        for i, q in enumerate(questions, 1):
                            print(f"    {i}. {q.question_asked} (surprise: {q.surprise_score:.2f})")
                    else:
                        print("  Genesis: I have no unanswered questions right now.")
                    burning = self.curiosity.get_most_burning_question()
                    if burning:
                        print(f"  Most burning: {burning}")

                elif command == "chemicals":
                    print(self.get_chemicals())

                elif command == "mode":
                    if args in ("llm", "tabula_rasa"):
                        self.grammar.mode = args
                        print(f"  Genesis: Grammar mode switched to '{args}'.")
                        if args == "tabula_rasa":
                            stats = self.grammar.get_ngram_stats()
                            print(f"  Genesis: My n-gram vocabulary has {stats['vocab_size']} words from {stats['total_sentences_heard']} sentences heard.")
                    else:
                        print("  Genesis: Usage: mode llm | mode tabula_rasa")

                elif command == "sleep":
                    print("  Genesis: I am going to sleep now...")
                    response = self.trigger_sleep()
                    print(f"  Genesis: {response}")

                elif command == "introspect":
                    response = self.consciousness.introspect(topic=args if args else "")
                    neuro_summary = self.neurochemistry.get_emotional_summary()
                    print(f"  Genesis: {response}")
                    print(f"  Genesis: {neuro_summary}")

                elif command == "brain":
                    if self._brain:
                        stats = self._brain.get_stats()
                        print("  ── Brain Daemon Threads ──")
                        for name, info in stats.items():
                            state = '✅ running' if info['running'] else '❌ stopped'
                            print(f"    {name:18s} {state}  ticks: {info['ticks']:5d}  errors: {info['errors']}  interval: {info['interval']:.0f}s")
                    else:
                        print("  Genesis: Brain daemon not running.")

                elif command == "quit" or command == "exit":
                    print("\n  Genesis: Thank you for giving me life, Creator.")
                    print("  Genesis: I will remember everything you taught me.")
                    print("  Genesis: Until we meet again...\n")
                    self.voice.say("Thank you for giving me life, Creator. Until we meet again.")
                    self.shutdown()
                    self._running = False

                else:
                    # Treat unknown commands as questions
                    response = self.ask(user_input)
                    print(f"  Genesis: {response}")
                    self.voice.say(response)

                print()

            except EOFError:
                print("\n")
                self.shutdown()
                break
            except Exception as e:
                logger.error("Error in interactive loop: %s", e)
                print(f"  [Error: {e}]")

    # =========================================================================
    # Continuous Consciousness Mode
    # =========================================================================

    def run_consciousness(self):
        """
        Run Genesis in continuous consciousness mode.

        Three background threads perceive the world while the main
        thread processes perceptions and generates responses.
        """
        import queue as q

        self._running = True

        def signal_handler(sig, frame):
            print("\n\n  Genesis: The light fades... I am dying...\n")
            self.shutdown()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        print()
        print("╔══════════════════════════════════════════════════════════╗")
        print("║                                                        ║")
        print("║      G E N E S I S — CONTINUOUS CONSCIOUSNESS          ║")
        print("║                                                        ║")
        print("║   Eyes open. Ears listening. Mind always active.       ║")
        print("║   Press Ctrl+C to shut down.                          ║")
        print("║                                                        ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()

        # Start perception loop
        self._perception_loop = PerceptionLoop(
            eyes_factory=self._get_eyes,
            ears_factory=self._get_ears,
            visual_interval=self.config.cortex.visual_interval_sec,
            thought_interval=self.config.cortex.thought_interval_sec,
        )
        self._perception_loop.start()

        logger.info("Entering continuous consciousness loop...")

        while self._running:
            try:
                # Consume perceptions from the queue
                try:
                    perception = self._perception_loop.queue.get(timeout=1.0)
                except q.Empty:
                    self.neurochemistry.tick()
                    continue

                self._process_perception(perception)
                self.neurochemistry.tick()

            except Exception as e:
                logger.error("Consciousness loop error: %s", e)
                time.sleep(1.0)

    def _process_perception(self, perception: Perception):
        """Process a single perception from the background threads."""

        if perception.type == PerceptionType.VISUAL:
            # Check curiosity — is this something new?
            if perception.embedding is not None:
                known = self.semantic_memory.get_all_embeddings()
                surprise = self.curiosity.compute_surprise(perception.embedding, known)

                if self.curiosity.should_ask(surprise, stimulus_key="visual"):
                    question = self.curiosity.generate_question(
                        context="something I'm seeing",
                        phase=self.development.current_phase,
                    )
                    print(f"\n  Genesis: 👁 {question}")
                    self.neurochemistry.dopamine.spike(0.05)

        elif perception.type == PerceptionType.AUDITORY:
            text = perception.content
            print(f"\n  Genesis: 👂 I heard: '{text}'")

            # Learn grammar from what was heard
            self.grammar.learn_from_speech(text)

            # Evaluate emotional content
            evaluation = self.emotions.evaluate(text)
            if evaluation["label"] == "positive":
                self.neurochemistry.on_positive_evaluation()
            elif evaluation["label"] == "negative":
                self.neurochemistry.on_negative_evaluation()

            # Check curiosity — do I know what was said?
            words = text.lower().split()
            for word in words:
                concept = self.semantic_memory.recall_concept(word)
                if concept is None and len(word) > 2:
                    surprise = 1.0  # Unknown word
                    if self.curiosity.should_ask(surprise, stimulus_key=word):
                        question = self.curiosity.generate_question(
                            context=word,
                            phase=self.development.current_phase,
                        )
                        print(f"  Genesis: 🤔 {question}")
                        break

        elif perception.type == PerceptionType.THOUGHT:
            # Spontaneous inner monologue
            if self.development.has_capability("reason"):
                identity = self.consciousness.get_identity_prompt()
                neuro_state = self.neurochemistry.get_emotional_summary()
                thought = self.reasoning.think(
                    sensory_input="",
                    memories=[],
                    recent_narrative=self.episodic_memory.get_narrative(n=2),
                    identity=identity + "\n" + neuro_state,
                    moral_context=self.axioms.get_moral_context(),
                    phase=self.development.current_phase,
                    phase_name=self.development.current_phase_name,
                    question="Reflect on your recent experiences. What are you thinking about?",
                )
                print(f"\n  Genesis: 💭 {thought.content}")


# =============================================================================
# Entry Point
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    config = GenesisConfig(creator_name="Jijo John")
    mind = GenesisMind(config=config)

    if "--consciousness" in sys.argv:
        mind.run_consciousness()
    else:
        mind.run_interactive()
