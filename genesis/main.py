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

# V5: Brain Realism Systems
from genesis.memory.working_memory import WorkingMemory
from genesis.cortex.attention import AttentionSystem
from genesis.cortex.emotional_state import PersistentEmotionalState
from genesis.cortex.theory_of_mind import TheoryOfMind
from genesis.cortex.metacognition import Metacognition
from genesis.cortex.play import PlayBehavior
from genesis.senses.motor import SimulatedMotor
from genesis.senses.babbling import BabblingEngine
from genesis.cortex.joint_attention import JointAttentionEngine
from genesis.neural.sensorimotor import SensorimotorLoop
from genesis.neural.acoustic_word_memory import AcousticWordMemory

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
            storage_path=GENESIS_HOME / "neural_weights" / "reasoner.pt",
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
            mode="tabula_rasa",
            ngram_storage_path=MEMORY_DIR / "ngram_model.json",
        )

        # --- V2: Neurochemistry ---
        self.neurochemistry = Neurochemistry()

        # --- V3: Subconscious Neural Cascade (Society of Mind) ---
        self.subconscious = Subconscious(
            weights_dir=GENESIS_HOME / "neural_weights",
        )

        # --- V7: Sensorimotor Loop (Pure Neural Acoustic Pipeline) ---
        self.sensorimotor = SensorimotorLoop(
            weights_dir=GENESIS_HOME / "acoustic_weights",
            sample_rate=self.config.acoustic.sample_rate,
            n_mels=self.config.acoustic.n_mels,
            latent_dim=self.config.acoustic.latent_dim,
            codebook_size=self.config.acoustic.codebook_size,
            lm_layers=self.config.acoustic.lm_layers,
            lm_heads=self.config.acoustic.lm_heads,
            lm_embd=self.config.acoustic.lm_embd,
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

        # --- V6: Babbling Engine (Acoustic Language Acquisition) ---
        self.babbling_engine = BabblingEngine(
            storage_path=MEMORY_DIR / "vocal_repertoire.json",
        )

        # --- V6: Joint Attention Engine (Cross-Modal Binding) ---
        self.joint_attention = JointAttentionEngine(
            storage_path=MEMORY_DIR / "joint_attention.json",
        )

        # --- V9: Acoustic Word Memory (DTW-based word recognition) ---
        self.acoustic_word_memory = AcousticWordMemory(
            storage_path=MEMORY_DIR / "acoustic_word_memory.json",
        )

        # Wire babbling and sensorimotor into voice for neural vocalization
        self.voice.set_babbling_engine(self.babbling_engine)
        self.voice.set_sensorimotor(self.sensorimotor)
        self.voice.set_acoustic_memory(self.acoustic_word_memory)
        self.voice.set_phase(self.development.current_phase)

        # --- V4: Proprioception (Internal Body Sense) ---
        self.proprioception = Proprioception()
        self.proprioception.increment_session()

        # --- V4: Drive System (8 drives, 4 Maslow tiers) ---
        self.drives = DriveSystem()

        # --- V5: Brain Realism Systems ---
        self.working_memory = WorkingMemory(capacity=7)
        self.attention = AttentionSystem()
        self.emotional_state = PersistentEmotionalState()
        self.theory_of_mind = TheoryOfMind()
        self.metacognition = Metacognition()
        self.play = PlayBehavior()
        self.motor = SimulatedMotor()

        # Enable ToM at Phase 3+
        if self.development.current_phase >= 3:
            self.theory_of_mind.enable()

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
            from genesis.neural.visual_cortex import VisualCortex
            cortex = VisualCortex(
                latent_dim=64,
                input_size=self.config.senses.image_size[0],
                storage_path=GENESIS_HOME / "neural_weights" / "visual_cortex.pt",
            )
            self._eyes = Eyes(
                camera_index=self.config.senses.camera_index,
                image_size=self.config.senses.image_size,
                motion_threshold=self.config.senses.motion_threshold,
                visual_cortex=cortex,
            )
        return self._eyes

    def _get_ears(self):
        if not hasattr(self, '_ears_instance') or self._ears_instance is None:
            from genesis.senses.ears import Ears
            self._ears_instance = Ears(
                sample_rate=self.config.senses.sample_rate,
                chunk_duration_sec=self.config.senses.chunk_duration_sec,
                silence_threshold=self.config.senses.silence_threshold,
            )
        return self._ears_instance

    # =========================================================================
    # Teaching Interface
    # =========================================================================

    def teach_concept(self, word: str, use_camera: bool = True) -> str:
        visual_embedding = None

        # V5: Attention filter — is this worth processing deeply?
        novelty = 1.0 if not self.semantic_memory.recall_concept(word) else 0.2
        attention_result = self.attention.compute_salience(
            stimulus_key=word,
            novelty=novelty,
            emotional_intensity=self.emotional_state.get_emotional_intensity(),
            drive_states=self.drives.get_status(),
        )

        if use_camera:
            try:
                eyes = self._get_eyes()
                percept = eyes.look()
                if percept:
                    visual_embedding = eyes.embed(percept)
                    logger.info("Captured visual for '%s'", word)
            except Exception as e:
                logger.warning("Could not capture visual: %s (teaching without image)", e)

        # V9: Capture mic audio → VQ tokens → store as acoustic exemplar
        # This is how the concept's "sound" is learned
        try:
            ears = self._get_ears()
            audio_percept = ears.listen_once(duration_sec=2.0)
            if audio_percept and audio_percept.raw_audio is not None:
                vq_tokens = self.sensorimotor.hear(audio_percept.raw_audio)
                if vq_tokens and len(vq_tokens) >= 3:
                    self.acoustic_word_memory.store_exemplar(
                        word=word,
                        vq_tokens=vq_tokens,
                        timestamp=datetime.now().isoformat(),
                    )
                    logger.info("Stored acoustic exemplar for '%s' (%d tokens)", word, len(vq_tokens))
        except Exception as e:
            logger.debug("Could not capture audio for '%s': %s", word, e)

        # Create multimodal binding
        self.associations.create_binding(
            word=word,
            visual_embedding=visual_embedding,
            context=f"Taught by {self.axioms.creator_name}",
        )

        # Neurochemistry: learning rate modifier from chemical state
        encoding_strength = self.neurochemistry.get_memory_encoding_strength()

        text_embedding = self.associations.embed_text(word).tolist()
        concept = self.semantic_memory.learn_concept(
            word=word,
            visual_embedding=visual_embedding.tolist() if visual_embedding is not None else None,
            text_embedding=text_embedding,
            context=f"Taught by {self.axioms.creator_name}",
            description=f"A concept I was taught",
            emotional_valence="+0.70",
        )

        # V5: Apply neurochemical encoding strength AND attention depth
        if hasattr(concept, 'strength'):
            depth_modifier = 1.0 if attention_result.processing_depth == "deep" else 0.6
            concept.strength = min(1.0, concept.strength * encoding_strength * depth_modifier)

        # V5: Working memory — attend to this concept
        self.working_memory.attend(
            key=word,
            content=concept,
            embedding=np.array(text_embedding) if text_embedding else None,
            salience=attention_result.salience,
            emotional_weight=self.emotional_state.get_emotional_intensity(),
        )

        # V5: Emotional response to learning
        self.emotional_state.on_experience(
            valence=0.3,  # Positive: being taught is pleasant
            arousal=0.2,
            novelty=novelty,
        )

        # V5: Metacognition — track this learning event
        self.metacognition.on_learn(word, success=True)

        # V5: Theory of Mind — observe what user is teaching
        self.theory_of_mind.observe_interaction(word, topic=word, sentiment=0.3)

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
                "attention_depth": attention_result.processing_depth,
            },
            document=f"Concept: {word}. Learned through teaching.",
        )

        self.episodic_memory.record(
            event_type="teaching",
            description=f"I was taught the concept '{word}'",
            auditory_text=word,
            spoken_words=[word],
            concepts_learned=[word],
            emotional_valence="+0.70",
            developmental_phase=self.development.current_phase,
            importance=0.9 * encoding_strength,
        )

        # Neurochemistry: reward for learning
        self.neurochemistry.on_successful_learning()
        self.neurochemistry.on_creator_interaction()

        # Grammar: learn from the teaching interaction
        self.grammar.learn_from_speech(f"this is {word}")

        # V6: Joint Attention — bind the taught concept to the word
        self.joint_attention.bind(word, word)

        # V3: Neural cascade — train ALL subconscious networks on this experience
        visual_tensor = visual_embedding.astype(np.float32) if visual_embedding is not None else np.zeros(64, dtype=np.float32)
        audio_tensor = np.array(text_embedding, dtype=np.float32) if text_embedding else np.zeros(64, dtype=np.float32)
        
        # V4: Pass proprioceptive context vector to the neural cascade
        context_vec = self.proprioception.get_context_vector()
        
        neural_result = self.subconscious.process_experience(
            visual_embedding=visual_tensor,
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
            self.voice.say("")

        # V4: Decode the neural network's own voice
        neural_voice = self.subconscious.decode_response(
            neural_result['personality_response'], self.semantic_memory
        )

        # V9: Neural expression — no hardcoded English
        parts = []
        # Show what was learned (data, not narration)
        parts.append(f"📥 {word}")
        if visual_embedding is not None:
            parts.append("👁")
        parts.append(f"[{self.semantic_memory.count()}]")
        # Neural echo from decoded thought vector
        if neural_voice and neural_voice not in ("(silence)", "(no words yet)"):
            parts.append(f"🧠 '{neural_voice}'")
        # Emotional indicator from neurochemistry (no English)
        if encoding_strength > 1.2:
            parts.append("😊")
        elif encoding_strength < 0.7:
            parts.append("😟")
        if milestone:
            parts.append(f"\n🌟 {milestone}")

        # Auto-sleep check
        if self.sleep_cycle.should_sleep():
            parts.append("\n😴")
            sleep_report = self.trigger_sleep()
            parts.append(f"\n{sleep_report}")

        return " ".join(parts)

    def teach_phonetic(self, grapheme: str, phoneme: str, example: str = "") -> str:
        binding = self.phonetics.teach(grapheme, phoneme, example)
        self.neurochemistry.on_successful_learning()

        self.episodic_memory.record(
            event_type="teaching",
            description=f"Creator taught me: letter '{grapheme}' → sound {phoneme}",
            auditory_text=f"{grapheme} says {phoneme}",
            concepts_learned=[f"phoneme_{grapheme}"],
            emotional_valence="+0.70",
            developmental_phase=self.development.current_phase,
            importance=0.7,
        )

        return (
            f"📥 '{grapheme}' → {phoneme}"
            f"{f' ({example})' if example else ''} "
            f"[{binding.strength:.0%}] "
            f"[{len(self.phonetics)} mappings]"
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

        # V8: Neural reasoning — no LLM, use attention-based pattern matching
        context_vec = self.consciousness.get_state_vector() if hasattr(self, 'consciousness') else None
        
        # Get memory embeddings for the neural reasoner
        memory_embeddings = []
        for mem in recalled:
            if mem.get("embedding"):
                memory_embeddings.append(np.array(mem["embedding"], dtype=np.float32))

        # Text embedding for the question
        text_emb_arr = np.array(text_emb, dtype=np.float32)
        
        thought = self.reasoning.think(
            auditory_embedding=text_emb_arr,
            context_vector=context_vec,
            memory_embeddings=memory_embeddings if memory_embeddings else None,
            phase=self.development.current_phase,
        )

        # Decode thought vector through ResponseDecoder
        if thought.raw_embedding is not None:
            decoded = self.subconscious.decode_response(
                thought.raw_embedding, self.semantic_memory
            )
            if decoded:
                thought.content = decoded

        # V7: Use neural babbling for response if no concepts decoded
        if not thought.content:
            response = self.grammar.generate_response(
                context=question,
                reasoning_engine=self.reasoning,
                phase=self.development.current_phase,
                phase_name=self.development.current_phase_name,
                memories=memories,
                babbling_engine=self.babbling_engine,
                joint_attention=self.joint_attention,
            )
        else:
            response = thought.content

        # V7: Generate neural audio response through acoustic pipeline
        try:
            waveform, neural_tokens = self.sensorimotor.generate_spontaneous(
                max_tokens=40, temperature=0.85,
            )
            if len(waveform) > 1600:  # At least 0.1s of audio
                self.sensorimotor.vocoder.play(waveform)
                logger.debug("Neural vocalization: %d tokens → %d samples",
                             len(neural_tokens), len(waveform))
        except Exception as e:
            logger.error("Neural audio generation failed: %s", e)

        # Evaluate emotional content via neurochemistry (no keyword matching)
        evaluation = self.emotions.evaluate(question)

        # Neurochemistry: respond based on valence
        valence = evaluation.get("valence", 0.0)
        if valence > 0.2:
            self.neurochemistry.on_positive_evaluation(abs(valence) * 0.15)
        elif valence < -0.2:
            self.neurochemistry.on_negative_evaluation(abs(valence) * 0.15)

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
            description=f"Q: '{question}' → R: '{response[:100]}'",
            auditory_text=question,
            spoken_words=question.split(),
            thought=response,
            emotional_valence=f"{valence:+.2f}",
            developmental_phase=self.development.current_phase,
            importance=0.6,
        )

        return response

    def recall_concept(self, word: str) -> str:
        result = self.consciousness.introspect(topic=word)
        if not result or "don't know" in result.lower():
            self.neurochemistry.on_failed_recall()
            return f"❓ {word}"
        else:
            activations = self.semantic_memory.spreading_activation(word, depth=2)
            if activations:
                related = ", ".join(f"{w} ({s:.0%})" for w, s in activations[:5])
                result += f"\n🔗 {related}"
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
            "║       GENESIS MIND V5 — BIOLOGICALLY REALISTIC        ║",
            "╚══════════════════════════════════════════════════════╝",
            "",
            f"  Name:           Genesis",
            f"  Guardian:       {model['identity']['creator']}",
            f"  Age:            {model['identity']['age']}",
            "",
            f"  Phase:          {model['development']['phase']} — {model['development']['phase_name']}",
            f"  LLM:            {'ACTIVE' if model['development']['phase'] >= 3 else 'DORMANT (Phase 3+)'}",
            f"  Capabilities:   {', '.join(model['development']['capabilities'])}",
            "",
            f"  Concepts known: {model['knowledge']['concepts_known']}",
            f"  Retrievable:    {len(self.semantic_memory.get_retrievable_concepts())}",
            f"  Fading:         {len(self.semantic_memory.get_fading_concepts())}",
            f"  Memories:       {model['knowledge']['episodes_experienced']}",
            f"  Phonetics:      {model['knowledge']['phonetic_bindings']} bindings",
            f"  Grammar mode:   {self.grammar.mode}",
            f"  Words heard:    {gram['total_words_heard']} ({gram['vocab_size']} unique)",
            "",
            "  ── Body Sense (Proprioception) ──",
            f"  Time:           {body['time_of_day']} {body['day_of_week']}",
            f"  Uptime:         {body['uptime_hours']}h",
            f"  Fatigue:        {'█' * int(body['fatigue'] * 10):10} {body['fatigue']:.2f}",
            f"  Experiences:    {body['experience_count']}",
            "",
            "  ── Working Memory (7±2 slots) ──",
            f"  Buffer:         {self.working_memory.get_stats()['utilization']}",
            f"  Total processed:{self.working_memory.get_stats()['total_processed']}",
            f"  Forgotten:      {self.working_memory.get_stats()['forgotten']}",
            "",
            "  ── Drives (8 drives, 4 Maslow tiers) ──",
        ]

        # Show all 8 drives grouped by tier
        tier_names = {1: "Survival", 2: "Social", 3: "Cognitive", 4: "Self"}
        for tier in [1, 2, 3, 4]:
            tier_drives = [(n, d) for n, d in drive_status.items()
                           if isinstance(d, dict) and d.get('tier') == tier]
            for name, info in tier_drives:
                level = info['level']
                bar = '█' * int(level * 10)
                lines.append(f"  {name:14s} {bar:10} {level:.2f}")
        lines.append(f"  Dominant:       {drive_status['dominant']} ({drive_status['dominant_level']:.2f})")

        lines.extend([
            "",
            "  ── Emotional State (8 dimensions) ──",
        ])
        emo = self.emotional_state.get_status()
        for dim, val in emo['dimensions'].items():
            bar = '█' * int(abs(val) * 10)
            sign = "+" if val >= 0 else "-"
            lines.append(f"  {dim:14s} {sign}{bar:10} {val:+.3f}")
        lines.append(f"  Dominant:       {emo['dominant']}")
        lines.append(f"  Valence:        {emo['valence']:+.3f}  Arousal: {emo['arousal']:.3f}")

        lines.extend([
            "",
            "  ── Neurochemistry ──",
            f"  Dopamine:       {'█' * int(neuro['dopamine']['level'] * 10):10} {neuro['dopamine']['level']:.2f} ({neuro['dopamine']['description']})",
            f"  Cortisol:       {'█' * int(neuro['cortisol']['level'] * 10):10} {neuro['cortisol']['level']:.2f} ({neuro['cortisol']['description']})",
            f"  Serotonin:      {'█' * int(neuro['serotonin']['level'] * 10):10} {neuro['serotonin']['level']:.2f} ({neuro['serotonin']['description']})",
            f"  Oxytocin:       {'█' * int(neuro['oxytocin']['level'] * 10):10} {neuro['oxytocin']['level']:.2f} ({neuro['oxytocin']['description']})",
            f"  Memory encoding:{neuro['modifiers']['learning_rate']:.2f}x",
            f"  Coherence:      {neuro['modifiers']['reasoning_coherence']:.2f}",
            "",
            "  ── Attention ──",
        ])
        att = self.attention.get_stats()
        lines.extend([
            f"  Stimuli:        {att['total_stimuli']}",
            f"  Deep processed: {att['deep_processed']} ({att['deep_pct']})",
            f"  Ignored:        {att['ignored']}",
            f"  Unique tracked: {att['unique_stimuli_tracked']}",
            "",
            f"  ── Curiosity ──",
            f"  Questions asked: {curiosity_stats['total_questions_asked']}",
            f"  Stimuli seen:    {curiosity_stats['unique_stimuli_encountered']}",
        ])

        # Metacognition
        meta = self.metacognition.get_stats()
        lines.extend([
            "",
            "  ── Metacognition (Self-Monitoring) ──",
            f"  Concepts tracked: {meta['concepts_tracked']}",
            f"  Avg confidence:   {meta['avg_confidence']:.0%}",
            f"  Knowledge gaps:   {meta['knowledge_gaps']}",
            f"  Strategy:         {meta['strategy']}",
        ])

        # Theory of Mind
        tom = self.theory_of_mind.get_status()
        lines.extend([
            "",
            "  ── Theory of Mind ──",
            f"  Active:          {tom['active']}",
        ])
        if tom['active']:
            lines.extend([
                f"  User interactions: {tom['user_interactions']}",
                f"  Topics taught:     {tom['topics_taught']}",
                f"  User sentiment:    {tom['user_sentiment']}",
                f"  User patience:     {tom['user_patience']}",
            ])

        # Play
        play_stats = self.play.get_stats()
        lines.extend([
            "",
            "  ── Play & Exploration ──",
            f"  Play sessions:    {play_stats['play_sessions']}",
            f"  Discoveries:      {play_stats['discoveries']}",
            f"  Favorite:         {play_stats['favorite_concept'] or 'none yet'}",
        ])

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
            description="shutdown",
            emotional_valence="0.0",
            developmental_phase=self.development.current_phase,
            importance=1.0,
        )
        if self._perception_loop and self._perception_loop.is_running:
            self._perception_loop.stop()
        if self._eyes:
            self._eyes.close()
        # Save the personality — the neural weights ARE the person
        self.subconscious.save_all()
        # V7: Save the acoustic neural weights
        self.sensorimotor.save_all()
        # V4: Stop the brain daemon
        if self._brain:
            self._brain.stop()
        # Farewell: neural vocalization only (no hardcoded English)
        self.voice.say("")
        logger.info("Genesis has shut down. Neural weights saved. Goodbye.")

    # =========================================================================
    # Interactive Terminal Interface
    # =========================================================================

    def run_interactive(self):
        self._running = True

        def signal_handler(sig, frame):
            print("\n")
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
        # Startup: neural vocalization
        self.voice.say("")

        if self.semantic_memory.count() == 0:
            print("  ✦ Phase 0 — tabula rasa")
            print(f"  ✦ 0 concepts | 0 phonemes")
        else:
            print(f"  ✦ {self.semantic_memory.count()} concepts | Phase {self.development.current_phase}")
            print(f"  ✦ {self.development.get_age_description()}")

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
        print("    voice on|off                    -- Toggle voice")
        print("    unanswered                      -- Show burning questions")
        print("    mode llm|tabula_rasa            -- Switch grammar mode")
        print("    babble                          -- Trigger a babble")
        print("    bindings                        -- Show learned cross-modal bindings")
        print("    vocab                           -- Show learned vocabulary")
        print("    neural-speak                    -- Generate and play neural audio")
        print("    neural-stats                    -- Show acoustic pipeline stats")
        print("    sleep                           -- Consolidate memories (4-phase)")
        print("    introspect                      -- Self-reflection")
        print("    quit                            -- Shut down")
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
                        print("  [usage: teach <word>]")
                        continue
                    response = self.teach_concept(args, use_camera=True)
                    print(f"  Genesis: {response}")

                elif command == "teach-text":
                    if not args:
                        print("  [usage: teach-text <word>]")
                        continue
                    response = self.teach_concept(args, use_camera=False)
                    print(f"  Genesis: {response}")

                elif command == "phonetic":
                    phonetic_parts = args.split()
                    if len(phonetic_parts) < 2:
                        print("  [usage: phonetic <letter> <sound> [example]]")
                        continue
                    grapheme = phonetic_parts[0]
                    phoneme = phonetic_parts[1]
                    example = " ".join(phonetic_parts[2:]) if len(phonetic_parts) > 2 else ""
                    response = self.teach_phonetic(grapheme, phoneme, example)
                    print(f"  Genesis: {response}")

                elif command == "ask":
                    if not args:
                        print("  [usage: ask <question>]")
                        continue
                    response = self.ask(args)
                    print(f"  Genesis: {response}")

                elif command == "recall":
                    if not args:
                        print("  [usage: recall <word>]")
                        continue
                    response = self.recall_concept(args)
                    print(f"  Genesis: {response}")

                elif command == "read":
                    if not args:
                        print("  [usage: read <word>]")
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
                        print("  [voice: on]")
                    elif args == "off":
                        self.voice.mute()
                        print("  [voice: off]")
                    else:
                        v = self.voice.get_status()
                        state = 'active' if v['enabled'] and not v['muted'] else 'muted' if v['muted'] else 'disabled'
                        print(f"  [voice: {state}]")

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
                        print(f"  ❓ {len(questions)} questions:")
                        for i, q in enumerate(questions, 1):
                            print(f"    {i}. {q.question_asked} (surprise: {q.surprise_score:.2f})")
                    else:
                        print("  ❓ 0")
                    burning = self.curiosity.get_most_burning_question()
                    if burning:
                        print(f"  🔥 {burning}")

                elif command == "chemicals":
                    print(self.get_chemicals())

                elif command == "mode":
                    if args in ("llm", "tabula_rasa"):
                        self.grammar.mode = args
                        print(f"  [grammar: {args}]")
                        if args == "tabula_rasa":
                            stats = self.grammar.get_ngram_stats()
                            print(f"  [vocab: {stats['vocab_size']} | heard: {stats['total_sentences_heard']}]")
                    else:
                        print("  [usage: mode llm | mode tabula_rasa]")

                elif command == "sleep":
                    print("  😴")
                    response = self.trigger_sleep()
                    print(f"  {response}")

                elif command == "introspect":
                    response = self.consciousness.introspect(topic=args if args else "")
                    neuro_summary = self.neurochemistry.get_emotional_summary()
                    print(f"  🧠 {response}")
                    print(f"  ⚔️ {neuro_summary}")

                elif command == "brain":
                    if self._brain:
                        stats = self._brain.get_stats()
                        print("  ── Brain Daemon Threads ──")
                        for name, info in stats.items():
                            state = '✅ running' if info['running'] else '❌ stopped'
                            print(f"    {name:18s} {state}  ticks: {info['ticks']:5d}  errors: {info['errors']}  interval: {info['interval']:.0f}s")
                    else:
                        print("  [brain daemon not running]")

                elif command == "babble":
                    text, phonemes = self.babbling_engine.babble()
                    print(f"  Genesis: {text}")
                    print(f"  (phonemes: {phonemes})")

                elif command == "bindings":
                    bindings = self.joint_attention.get_all_bindings_sorted()
                    if not bindings:
                        print("  [0 bindings]")
                    else:
                        print(f"  -- Cross-Modal Bindings ({len(bindings)}) --")
                        for b in bindings[:20]:
                            status = 'LEARNED' if b['learned'] else 'weak'
                            print(f"    '{b['visual']}' <-> '{b['word']}' strength={b['strength']:.3f} x{b['co_occurrences']} [{status}]")

                elif command == "vocab":
                    vocab = self.joint_attention.get_vocabulary()
                    ngram_stats = self.grammar.get_ngram_stats()
                    babble_status = self.babbling_engine.get_status()
                    print(f"  -- Language Acquisition Status --")
                    print(f"    Grammar mode:       {self.grammar.mode}")
                    print(f"    Learned vocabulary: {len(vocab)} words {vocab[:20]}")
                    print(f"    N-gram vocab:       {ngram_stats['vocab_size']} words")
                    print(f"    Sentences heard:    {ngram_stats['total_sentences_heard']}")
                    print(f"    Babble repertoire:  {babble_status['repertoire_size']} units")
                    print(f"    Total babbles:      {babble_status['total_babbles']}")
                    print(f"    Reinforcements:     {babble_status['total_reinforcements']}")
                    strongest = babble_status.get('strongest', [])
                    if strongest:
                        print(f"    Strongest babbles:")
                        for s in strongest[:5]:
                            concept = f" -> '{s['associated_concept']}' " if s['associated_concept'] else ""
                            print(f"      '{s['speakable']}'{concept} strength={s['strength']:.3f}")

                elif command == "neural-speak":
                    print("  🎤")
                    waveform, tokens = self.sensorimotor.generate_spontaneous(
                        max_tokens=30, temperature=0.9,
                    )
                    print(f"  [{len(tokens)} tokens → {len(waveform)} samples ({len(waveform)/16000:.2f}s)]")
                    self.sensorimotor.vocoder.play(waveform)

                elif command == "neural-stats":
                    stats = self.sensorimotor.get_stats()
                    print("  ── Pure Neural Acoustic Pipeline ──")
                    print(f"    Total interactions: {stats['total_interactions']}")
                    print(f"    Total params:      {stats['total_params']:,}")
                    ac = stats['auditory_cortex']
                    print(f"    Auditory Cortex:   {ac['params']:,} params, {ac['frames_processed']} frames")
                    vq = stats['vq_codebook']
                    print(f"    VQ Codebook:       {vq['codebook_size']} entries, {vq['active_codes']} active ({vq['codebook_utilization']*100:.1f}% util)")
                    ab = stats['acoustic_brain']
                    print(f"    Acoustic LM:       {ab['params']:,} params, {ab['total_sequences_heard']} seqs heard, loss={ab['avg_loss']:.4f}")
                    vo = stats['vocoder']
                    print(f"    Neural Vocoder:    {vo['params']:,} params, {vo['total_syntheses']} syntheses")

                elif command == "quit" or command == "exit":
                    print()
                    self.voice.say("")
                    self.shutdown()
                    self._running = False

                else:
                    # Treat unknown commands as questions
                    response = self.ask(user_input)
                    print(f"  Genesis: {response}")
                    self.voice.say("")

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
            print("\n")
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
                    print(f"\n  Genesis: 👁 ❓")
                    self.neurochemistry.dopamine.spike(0.05)

        elif perception.type == PerceptionType.AUDITORY:
            text = perception.content

            # V7: Feed raw audio to acoustic neural pipeline (trains from every sound)
            if perception.raw_audio is not None:
                try:
                    acoustic_tokens = self.sensorimotor.hear(perception.raw_audio)
                    logger.debug("Acoustic pipeline processed %d tokens from audio", len(acoustic_tokens))
                except Exception as e:
                    logger.error("Sensorimotor hear failed: %s", e)

            if not text:
                return  # Pure audio event, no text content to process

            print(f"\n  Genesis: 👂 '{text}'")

            # Learn grammar from what was heard
            self.grammar.learn_from_speech(text)

            # Evaluate emotional content via neurochemistry
            evaluation = self.emotions.evaluate(text)
            eval_valence = evaluation.get("valence", 0.0)
            if eval_valence > 0.2:
                self.neurochemistry.on_positive_evaluation()
            elif eval_valence < -0.2:
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
                        print(f"  Genesis: 🤔 ❓")
                        break

        elif perception.type == PerceptionType.THOUGHT:
            # Spontaneous inner monologue
            if self.development.has_capability("reason"):
                # V8: Neural reasoning — no LLM
                context_vec = self.consciousness.get_state_vector() if hasattr(self, 'consciousness') else None
                thought = self.reasoning.think(
                    context_vector=context_vec,
                    phase=self.development.current_phase,
                )
                if thought.raw_embedding is not None:
                    decoded = self.subconscious.decode_response(
                        thought.raw_embedding, self.semantic_memory
                    )
                    if decoded:
                        thought.content = decoded
                if thought.content:
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
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    config = GenesisConfig(creator_name="Jijo John")
    mind = GenesisMind(config=config)

    if "--consciousness" in sys.argv:
        mind.run_consciousness()
    else:
        # Start Web Dashboard
        try:
            from genesis.dashboard.server import DashboardServer
            dashboard = DashboardServer(mind, port=5050)
            dashboard.start()
            print("\n  [Dashboard] Neural interface live at: http://localhost:5050\n")
        except Exception as e:
            print(f"\n  [Dashboard] Failed to start: {e}\n")

        mind.run_interactive()
