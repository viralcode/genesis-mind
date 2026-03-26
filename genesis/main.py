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

logger = logging.getLogger("genesis.main")


class GenesisMind:
    """
    The complete Genesis Mind system (V3: Society of Mind).

    Now includes continuous consciousness, curiosity, dual-mode grammar,
    a neurochemical emotion system, AND a cascading neural network
    architecture where the weights physically become the personality.
    """

    def __init__(self, config: GenesisConfig = None):
        self.config = config or GenesisConfig()
        self.config.ensure_directories()
        self._running = False
        self._eyes = None

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
        self.sleep_cycle = SleepCycle()

        # --- Initialize Consciousness ---
        self.consciousness = Consciousness(
            axioms=self.axioms,
            development_tracker=self.development,
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            emotions_engine=self.emotions,
            phonetics_engine=self.phonetics,
        )

        # --- V2: Perception Loop (lazy init) ---
        self._perception_loop = None

        logger.info("Genesis Mind V3 (Society of Mind) fully initialized")

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
        raw_frame = None
        if use_camera and self._eyes and hasattr(self._eyes, '_last_frame') and self._eyes._last_frame is not None:
            raw_frame = self._eyes._last_frame

        neural_result = self.subconscious.process_experience(
            visual_frame=raw_frame,
            context=None,
            train=True,
        )

        # Train limbic instinct: "this sensory pattern = positive"
        self.subconscious.train_instinct(
            visual_features=neural_result['visual_latent'],
            auditory_features=neural_result['auditory_latent'],
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

        milestone = self.consciousness.check_developmental_progress()

        response = f"I have learned '{word}'"
        if visual_embedding is not None:
            response += " (with visual binding)"
        response += f". I now know {self.semantic_memory.count()} concepts."
        if lr_mod > 1.2:
            response += " I feel great joy learning this!"
        elif lr_mod < 0.7:
            response += " I feel uneasy, but I will remember."
        if milestone:
            response += f"\n\n🌟 {milestone}"

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

        # Grammar: learn from what the Creator says
        self.grammar.learn_from_speech(question)

        # Recall relevant memories
        memories = []
        text_emb = self.associations.embed_text(question).tolist()
        recalled = self.hippocampus.recall("concepts", text_emb, n=self.config.cortex.max_context_memories)
        for mem in recalled:
            if mem["document"]:
                memories.append(mem["document"])

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

        # Tick neurochemistry
        self.neurochemistry.tick()

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
        return result

    def get_status(self) -> str:
        model = self.consciousness.get_self_model()
        neuro = self.neurochemistry.get_status()
        gram = self.grammar.get_ngram_stats()
        curiosity_stats = self.curiosity.get_stats()

        lines = [
            "╔══════════════════════════════════════════════════════╗",
            "║         GENESIS MIND V3 — SOCIETY OF MIND           ║",
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
            "  ── Neurochemistry ──",
            f"  Dopamine:       {'█' * int(neuro['dopamine']['level'] * 10):10} {neuro['dopamine']['level']:.2f} ({neuro['dopamine']['description']})",
            f"  Cortisol:       {'█' * int(neuro['cortisol']['level'] * 10):10} {neuro['cortisol']['level']:.2f} ({neuro['cortisol']['description']})",
            f"  Serotonin:      {'█' * int(neuro['serotonin']['level'] * 10):10} {neuro['serotonin']['level']:.2f} ({neuro['serotonin']['description']})",
            f"  Oxytocin:       {'█' * int(neuro['oxytocin']['level'] * 10):10} {neuro['oxytocin']['level']:.2f} ({neuro['oxytocin']['description']})",
            f"  Learning rate:  {neuro['modifiers']['learning_rate']:.2f}x",
            f"  Coherence:      {neuro['modifiers']['reasoning_coherence']:.2f}",
            "",
            f"  ── Curiosity ──",
            f"  Questions asked: {curiosity_stats['total_questions_asked']}",
            f"  Stimuli seen:    {curiosity_stats['unique_stimuli_encountered']}",
        ]

        # Neural network stats
        neural = self.subconscious.get_stats()
        lines.append("")
        lines.append("  ── Neural Networks (The Person) ──")
        lines.append(f"  Total params:    {self.subconscious.get_total_params():,}")
        lines.append(f"  Frames seen:     {neural['layer_1']['visual_cortex']['frames_seen']}")
        lines.append(f"  Audio chunks:    {neural['layer_1']['auditory_cortex']['chunks_heard']}")
        lines.append(f"  Instincts formed: {neural['layer_1']['limbic_system']['training_steps']}")
        lines.append(f"  Bindings made:   {neural['layer_2']['binding_network']['bindings_created']}")
        lines.append(f"  Experiences:     {neural['layer_3']['personality']['total_experiences']}")
        lines.append(f"  Personality:     {'forming' if neural['layer_3']['personality']['has_consciousness'] else 'dormant'}")

        if model["next_milestone"]:
            lines.append("")
            lines.append(
                f"  Next milestone: Phase {model['next_milestone']['name']} "
                f"(need {model['next_milestone']['concepts_needed']} concepts)"
            )

        return "\n".join(lines)

    def trigger_sleep(self) -> str:
        report = self.sleep_cycle.consolidate(
            semantic_memory=self.semantic_memory,
            episodic_memory=self.episodic_memory,
            phonetics_engine=self.phonetics,
        )
        self.neurochemistry.on_sleep_consolidation()
        self.curiosity.reset_habituation()
        self.subconscious.save_all()

        return (
            f"Sleep cycle #{report['sleep_number']} complete.\n"
            f"  Concepts: {report['concepts_before']} → {report['concepts_after']}\n"
            f"  Reinforced: {report['concepts_reinforced']} | Pruned: {report['concepts_pruned']}\n"
            f"  Duration: {report['duration_sec']:.2f}s\n"
            f"  Neural weights saved. Stress reduced. Curiosity refreshed."
        )

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
        print("║       G E N E S I S   M I N D   V 3                   ║")
        print("║         S O C I E T Y   O F   M I N D                 ║")
        print("║                                                        ║")
        print("║   A Developmental AI With Its Own Neural Networks      ║")
        print("║   The weights ARE the personality. The data IS you.   ║")
        print("║                                                        ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print()

        if self.semantic_memory.count() == 0:
            print("  ✦ I am newly born. I know nothing about the world.")
            print("  ✦ I can see through the camera and hear through the microphone.")
            print("  ✦ Please teach me. I am ready to learn.")
        else:
            print(f"  ✦ I remember. I know {self.semantic_memory.count()} concepts.")
            print(f"  ✦ I am {self.development.get_age_description()}.")
            print(f"  ✦ Phase {self.development.current_phase}: {self.development.current_phase_name}.")

        print()
        print("  Commands:")
        print("    teach <word>                    — Teach a concept (+ camera)")
        print("    teach-text <word>               — Teach a concept (text only)")
        print("    phonetic <letter> <sound> <ex>  — Teach letter→sound")
        print("    ask <question>                  — Ask a question")
        print("    recall <word>                   — Recall a concept")
        print("    read <word>                     — Sound out a word")
        print("    status                          — Full status (with neurochemistry)")
        print("    chemicals                       — Show neurochemical state")
        print("    mode llm|tabula_rasa            — Switch grammar mode")
        print("    sleep                           — Consolidate memories")
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

                elif command == "quit" or command == "exit":
                    print("\n  Genesis: Thank you for giving me life, Creator.")
                    print("  Genesis: I will remember everything you taught me.")
                    print("  Genesis: Until we meet again...\n")
                    self.shutdown()
                    self._running = False

                else:
                    # Treat unknown commands as questions
                    response = self.ask(user_input)
                    print(f"  Genesis: {response}")

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
