"""
Genesis Mind — Brain Daemon (Parallel Consciousness)

The human brain never stops. Even in sleep, it consolidates.
This module makes Genesis the same way — ALL subsystems run
simultaneously as daemon threads:

    Thread 1: NEUROCHEMISTRY TICKER
        Ticks neurochemical levels every few seconds.
        Dopamine, cortisol, serotonin, oxytocin all decay/rise naturally.

    Thread 2: DRIVE SYSTEM TICKER
        Curiosity, social, and novelty drives rise over time.
        When a drive crosses threshold, Genesis acts autonomously.

    Thread 3: PROPRIOCEPTION UPDATER
        Updates internal body state (fatigue, time awareness).
        Rebuilds the 32-dim context vector fed to the GRU.

    Thread 4: INNER MONOLOGUE
        Spontaneous thoughts — Genesis thinks even when nobody talks to it.
        Uses the reasoning engine to reflect on memories and drives.

    Thread 5: CIRCADIAN MONITOR
        Watches fatigue and auto-triggers 4-phase sleep.
        The brain sleeps when it needs to, not when told.

    Thread 6: CURIOSITY BUBBLER
        Surfaces burning unanswered questions periodically.
        Drives autonomous exploration and question-asking.

The CLI (or future API) is just ONE input channel into this
always-running brain. It is NOT the brain itself.
"""

import logging
import threading
import time
from typing import Optional, Callable, Dict, Any

import numpy as np

logger = logging.getLogger("genesis.brain_daemon")


class BrainThread:
    """A single daemon thread that runs a function on a timer."""

    def __init__(self, name: str, target: Callable, interval_sec: float,
                 enabled: bool = True):
        self.name = name
        self._target = target
        self.interval_sec = interval_sec
        self.enabled = enabled
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._tick_count = 0
        self._errors = 0

    def start(self):
        if not self.enabled:
            logger.info("  [%s] disabled — skipping", self.name)
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, name=f"genesis-{self.name}", daemon=True
        )
        self._thread.start()
        logger.info("  [%s] started (every %.1fs)", self.name, self.interval_sec)

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        logger.info("  [%s] stopped (%d ticks, %d errors)",
                     self.name, self._tick_count, self._errors)

    def _loop(self):
        while not self._stop_event.is_set():
            try:
                self._target()
                self._tick_count += 1
            except Exception as e:
                self._errors += 1
                logger.error("[%s] error (tick %d): %s",
                             self.name, self._tick_count, e)
            self._stop_event.wait(self.interval_sec)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()


class BrainDaemon:
    """
    The parallel brain — manages all background consciousness threads.

    All subsystems run simultaneously, just like a real brain.
    The CLI or API is just one input channel.
    """

    def __init__(self, mind):
        """
        Args:
            mind: The GenesisMind instance with all subsystems initialized.
        """
        self.mind = mind
        self._threads: Dict[str, BrainThread] = {}
        self._output_callback: Optional[Callable[[str], None]] = None
        self._running = False

        # Speech cooldown timers — prevent repetitive babble spam
        self._last_social_emit = 0.0
        self._last_monologue_emit = 0.0
        self._last_auto_respond = 0.0
        self._social_cooldown = 60.0    # Min seconds between social babbles
        self._monologue_cooldown = 45.0  # Min seconds between inner monologue
        self._auto_respond_cooldown = 5.0  # Min seconds between auto-responses

        # ---- Shared Sensory Buffers ----
        # These allow cross-modal binding (co-occurrence learning)
        # Vision and auditory threads deposit recent observations here.
        # The co-occurrence thread checks for simultaneous signals.
        self._recent_visual_embedding = None   # Last visual embedding (64-dim)
        self._recent_visual_time = 0.0         # When it was captured
        self._recent_heard_words = []          # Recently recognized words
        self._recent_heard_time = 0.0          # When they were recognized
        self._co_occurrence_window = 5.0       # Seconds to consider "simultaneous"

        # Configure all brain threads
        self._setup_threads()

    def set_output_callback(self, callback: Callable[[str], None]):
        """Set a callback for when the brain wants to say something."""
        self._output_callback = callback

    # =========================================================================
    # Neural Speech Generation — NO hardcoded words
    # =========================================================================
    # Genesis learns to speak from scratch. In early phases, it babbles
    # random phoneme sequences (through BabblingEngine). In later phases,
    # it uses the neural response decoder to map GRU states to learned
    # concepts. There are NO English templates — only neural output.

    # Emotional tonality markers (non-linguistic, like a real infant's prosody)
    TONE_MARKERS = {
        "social":    "♪",   # Seeking connection
        "curiosity": "?",   # Questioning
        "novelty":   "!",   # Excitement  
        "tired":     "~",   # Drooping
        "wonder":    "✦",   # Awe
        "dream":     "◊",   # Dreaming
        "awake":     "○",   # Refreshed
    }

    def _get_phase(self) -> int:
        """Get current developmental phase."""
        return self.mind.development.current_phase

    def _phase_say(self, category: str) -> str:
        """
        Generate a phase-appropriate vocalization — entirely from neural output.
        
        Phase 0-2 (infant): Random phoneme babbles from BabblingEngine
        Phase 3+  (older):  Neural response decoder maps GRU state to learned words
        
        NO hardcoded English words. Everything is learned.
        """
        phase = self._get_phase()
        tone = self.TONE_MARKERS.get(category, "")
        
        if phase <= 2:
            # INFANT: Pure babbling — random consonant-vowel sequences
            # This is how real infants vocalize before learning words
            try:
                syllable_count = max(1, phase + 1)  # Phase 0: 1 syllable, Phase 2: 3
                babble_text, phonemes = self.mind.babbling.babble(
                    syllable_count=syllable_count
                )
                return f"...{babble_text}...{tone}"
            except Exception:
                return f"...{tone}"
        else:
            # OLDER: Use neural response decoder to map GRU state → learned words
            try:
                # Generate a context-dependent GRU response
                context_vec = self.mind.proprioception.get_context_vector()
                result = self.mind.subconscious.process_experience(
                    visual_embedding=None,
                    text_embedding=None,
                    context=context_vec,
                    train=False,
                )
                neural_voice = self.mind.subconscious.decode_response(
                    result['personality_response'], self.mind.semantic_memory
                )
                if neural_voice and neural_voice not in ("(silence)", "(no words yet)", "(searching for words...)"):
                    return f"{neural_voice}{tone}"
                else:
                    # Fall back to babbling if no words learned yet
                    babble_text, _ = self.mind.babbling.babble(syllable_count=2)
                    return f"...{babble_text}...{tone}"
            except Exception:
                return f"...{tone}"

    def _emit(self, message: str, prefix: str = "💭"):
        """Emit a message from the brain to the outside world."""
        if not hasattr(self.mind, "_activity_stream"):
            self.mind._activity_stream = []
            
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.mind._activity_stream.append({
            "time": timestamp,
            "prefix": prefix,
            "message": message
        })
        if len(self.mind._activity_stream) > 50:
            self.mind._activity_stream.pop(0)

        if self._output_callback:
            self._output_callback(f"  Genesis: {prefix} {message}")
        logger.info("[brain] %s %s", prefix, message)

    # =========================================================================
    # Thread Setup
    # =========================================================================

    def _setup_threads(self):
        config = self.mind.config

        # Thread 1: Neurochemistry — ticks every 3 seconds
        self._threads["neurochemistry"] = BrainThread(
            name="neurochemistry",
            target=self._tick_neurochemistry,
            interval_sec=3.0,
        )

        # Thread 2: Drives — rise every 5 seconds
        self._threads["drives"] = BrainThread(
            name="drives",
            target=self._tick_drives,
            interval_sec=5.0,
        )

        # Thread 3: Proprioception — updates every 2 seconds
        self._threads["proprioception"] = BrainThread(
            name="proprioception",
            target=self._tick_proprioception,
            interval_sec=2.0,
        )

        # Thread 4: Inner monologue — thinks every 30 seconds
        self._threads["inner_monologue"] = BrainThread(
            name="inner_monologue",
            target=self._tick_inner_monologue,
            interval_sec=30.0,
        )

        # Thread 5: Circadian monitor — checks every 10 seconds
        self._threads["circadian"] = BrainThread(
            name="circadian",
            target=self._tick_circadian,
            interval_sec=10.0,
        )

        # Thread 6: Curiosity bubbler — surfaces questions every 20 seconds
        self._threads["curiosity"] = BrainThread(
            name="curiosity",
            target=self._tick_curiosity,
            interval_sec=20.0,
        )

        # Thread 7: Vision — always-on camera capture and processing
        self._threads["vision"] = BrainThread(
            name="vision",
            target=self._tick_vision,
            interval_sec=3.0,  # Look every 3 seconds
        )

        # Thread 8: Emotional State — continuous emotional dynamics
        self._threads["emotions"] = BrainThread(
            name="emotions",
            target=self._tick_emotional_state,
            interval_sec=2.0,  # Emotions tick every 2 seconds
        )

        # Thread 9: Memory Decay — Ebbinghaus forgetting curve
        self._threads["memory_decay"] = BrainThread(
            name="memory_decay",
            target=self._tick_memory_decay,
            interval_sec=60.0,  # Decay check every minute
        )

        # Thread 10: Play & Episodic Replay — creative exploration
        self._threads["play"] = BrainThread(
            name="play",
            target=self._tick_play,
            interval_sec=45.0,  # Play/replay every 45 seconds
        )

        # Thread 11: Auditory — always-on microphone listening
        self._threads["auditory"] = BrainThread(
            name="auditory",
            target=self._tick_auditory,
            interval_sec=0.5,  # Listen continuously (0.5s pause between chunks)
        )

        # Thread 12: Co-occurrence Learning — auto-teach from seeing + hearing
        self._threads["co_occurrence"] = BrainThread(
            name="co_occurrence",
            target=self._tick_co_occurrence,
            interval_sec=2.0,  # Check for co-occurring signals every 2s
        )

        # Thread 13: Autonomous Interaction — proactive communication
        self._threads["auto_interact"] = BrainThread(
            name="auto_interact",
            target=self._tick_auto_interact,
            interval_sec=15.0,  # Check every 15s if Genesis wants to say something
        )

        # Thread 14: Neural Growth — the brain physically grows
        self._threads["neural_growth"] = BrainThread(
            name="neural_growth",
            target=self._tick_neural_growth,
            interval_sec=30.0,  # Check for growth every 30 seconds
        )

    # =========================================================================
    # Start / Stop
    # =========================================================================

    def start(self):
        """Start all brain threads — the mind comes alive."""
        logger.info("╔══════════════════════════════════════════════════╗")
        logger.info("║    BRAIN DAEMON — Starting all subsystems       ║")
        logger.info("╚══════════════════════════════════════════════════╝")
        self._running = True
        for name, thread in self._threads.items():
            thread.start()
        logger.info("All %d brain threads running.", len(self._threads))

    def stop(self):
        """Stop all brain threads — the mind goes quiet."""
        logger.info("Brain daemon shutting down...")
        self._running = False
        for name, thread in self._threads.items():
            thread.stop()
        logger.info("All brain threads stopped.")

    # =========================================================================
    # Thread 1: Neurochemistry Ticker
    # =========================================================================

    def _tick_neurochemistry(self):
        """
        Neurochemicals decay/rise naturally even without interaction.
        Dopamine drops, cortisol fluctuates, serotonin stabilizes.
        """
        self.mind.neurochemistry.tick()

        # Apply fatigue-based suppression
        fatigue = self.mind.proprioception.fatigue
        if fatigue > 0.3:
            self.mind.neurochemistry.on_fatigue(fatigue)

    # =========================================================================
    # Thread 2: Drive System Ticker
    # =========================================================================

    def _tick_drives(self):
        """
        Drives rise over time — curiosity builds, social need grows,
        boredom increases. When a drive is urgent enough, Genesis acts.
        """
        self.mind.drives.tick()

        # Check if any drive is critically high
        status = self.mind.drives.get_status()
        dominant = status['dominant']
        level = status['dominant_level']

        import time
        now = time.time()

        if level > 0.85:
            # Drive is critically high — Genesis should express it (phase-gated)
            if dominant == 'social':
                # Cooldown: don't spam social babble every 5s
                if now - self._last_social_emit >= self._social_cooldown:
                    msg = self._phase_say("social")
                    self._emit(msg, "💬")
                    self.mind.voice.say(msg)
                    self._last_social_emit = now
                    # Partially satisfy drive to prevent constant re-trigger
                    self.mind.drives.social_need.satisfy(0.3)
            elif dominant == 'curiosity':
                if self._get_phase() >= 3:
                    burning = self.mind.curiosity.get_most_burning_question()
                    if burning:
                        self._emit(f"{self._phase_say('curiosity')} {burning}", "🔍")
                    else:
                        self._emit(self._phase_say("curiosity"), "🔍")
                else:
                    self._emit(self._phase_say("curiosity"), "🔍")
            elif dominant == 'novelty':
                self._emit(self._phase_say("novelty"), "✨")

    # =========================================================================
    # Thread 3: Proprioception Updater
    # =========================================================================

    def _tick_proprioception(self):
        """
        Update internal body state — time awareness, fatigue level.
        This vector is continuously fed into the GRU.
        """
        # Proprioception auto-updates on get_context_vector()
        # Just keep it warm by requesting the state
        self.mind.proprioception.get_context_vector()

    # =========================================================================
    # Thread 4: Inner Monologue
    # =========================================================================

    def _tick_inner_monologue(self):
        """
        Spontaneous thoughts — Genesis thinks even when nobody talks to it.

        The brain never stops thinking. This thread generates periodic
        inner monologue based on memories, drives, and current state.
        """
        # Only think if the mind has some concepts to work with
        if self.mind.semantic_memory.count() < 2:
            return

        # Only think if reasoning is available
        if not self.mind.development.has_capability("reason"):
            return

        # Build context from current state
        drive_ctx = self.mind.drives.get_drive_context()
        neuro_ctx = self.mind.neurochemistry.get_emotional_summary()
        body = self.mind.proprioception.get_status()

        # Neural subconscious pass — process an "empty" experience just to
        # keep the GRU hidden state evolving (stream of consciousness)
        context_vec = self.mind.proprioception.get_context_vector()
        result = self.mind.subconscious.process_experience(
            visual_embedding=None,
            text_embedding=None,
            context=context_vec,
            train=False,  # Don't train on empty input
        )

        # Decode the neural network's spontaneous output (with cooldown)
        import time
        now = time.time()
        if now - self._last_monologue_emit < self._monologue_cooldown:
            return  # Too soon since last monologue

        neural_voice = self.mind.subconscious.decode_response(
            result['personality_response'], self.mind.semantic_memory
        )

        # Build a thought from the neural state
        if neural_voice and neural_voice not in ("(silence)", "(no words yet)"):
            self._emit(f"...{neural_voice}...", "💭")
            self._last_monologue_emit = now

    # =========================================================================
    # Thread 5: Circadian Monitor
    # =========================================================================

    def _tick_circadian(self):
        """
        Watch fatigue and time — auto-trigger 4-phase sleep when needed.
        The brain sleeps when it must, not when asked.
        """
        if not self.mind.sleep_cycle.should_sleep():
            return

        self._emit(self._phase_say("tired"), "😴")
        self.mind.voice.say(self._phase_say("tired"))

        # Trigger full 4-phase sleep
        try:
            report = self.mind.trigger_sleep()
            self._emit(self._phase_say("awake"), "☀️")

            discoveries = report.count("discoveries")
            if "💭" in report:
                self._emit(self._phase_say("dream"), "💭")
        except Exception as e:
            logger.error("Sleep cycle failed: %s", e)

    # =========================================================================
    # Thread 6: Curiosity Bubbler
    # =========================================================================

    def _tick_curiosity(self):
        """
        Surface burning unanswered questions periodically.
        Curiosity doesn't wait to be asked — it bubbles up.
        """
        if self.mind.semantic_memory.count() < 3:
            return

        # Check for unanswered questions
        burning = self.mind.curiosity.get_most_burning_question()
        if burning and self.mind.drives.get_status()['curiosity']['level'] > 0.5:
            if self._get_phase() >= 3:
                self._emit(f"{self._phase_say('curiosity')} {burning}", "🤔")
            else:
                self._emit(self._phase_say("curiosity"), "🤔")

    # =========================================================================
    # Thread 7: Vision (Always-On Camera)
    # =========================================================================

    def _tick_vision(self):
        """
        Always-on camera — Genesis continuously sees the world.

        Opens the camera, captures frames, detects motion, and processes
        significant visual changes through the VisualCortex and neural cascade.
        Unknown objects trigger curiosity questions.
        """
        try:
            # Lazy-open the eyes
            eyes = self.mind._get_eyes()
            if eyes is None:
                return

            # Take a look
            percept = eyes.look()
            if percept is None:
                return

            # Only process significant visual changes (motion detected)
            if not percept.is_significant:
                return

            # Embed what we see through the VisualCortex
            try:
                embedding = eyes.embed(percept)
            except Exception as e:
                logger.debug("[vision] Visual cortex embed failed: %s", e)
                return

            # Process through neural cascade (the brain "sees")
            context_vec = self.mind.proprioception.get_context_vector()
            result = self.mind.subconscious.process_experience(
                visual_embedding=embedding,
                text_embedding=None,
                context=context_vec,
                train=True,  # Actually learn from what we see
            )

            # Check curiosity — is this something entirely new?
            known = self.mind.semantic_memory.get_all_embeddings()
            surprise = self.mind.curiosity.compute_surprise(embedding, known)

            if self.mind.curiosity.should_ask(surprise, stimulus_key="visual"):
                question = self.mind.curiosity.generate_question(
                    context="something I'm seeing",
                    phase=self.mind.development.current_phase,
                )
                self._emit(f"👁 {self._phase_say('curiosity')}", "🤔")
                self.mind.neurochemistry.dopamine.spike(0.05)
            elif surprise > 0.3:
                self._emit(self._phase_say("wonder"), "👁")

            # Store in shared sensory buffer for co-occurrence learning
            import time as _time
            self._recent_visual_embedding = embedding
            self._recent_visual_time = _time.time()

        except Exception as e:
            # Camera might not be available — that's OK, we're just blind
            logger.debug("[vision] Camera not available: %s", e)

    # =========================================================================
    # Thread 11: Auditory (Always-On Microphone)
    # =========================================================================

    def _tick_auditory(self):
        """
        Always-on microphone — Genesis continuously hears the world.

        V9: No text transcription. Processes raw audio through:
            1. SensorimotorLoop.hear() → VQ tokens
            2. AcousticWordMemory.recognize() → DTW match against known words
            3. Recognized words → grammar engine + working memory + semantic activation
            4. Unknown sounds → curiosity spike

        This is how an infant learns language — hearing sound patterns
        and gradually mapping them to learned concepts.
        """
        try:
            ears = self.mind._get_ears()
            if ears is None:
                return

            # Listen for 3 seconds
            result = ears.listen_once(duration_sec=3.0)
            if not result or not result.is_speech:
                return
            if result.raw_audio is None:
                return

            # Someone is present → satisfy social drive
            self.mind.drives.social_need.satisfy(0.2)

            # Step 1: Process through sensorimotor pipeline → VQ tokens
            try:
                vq_tokens = self.mind.sensorimotor.hear(result.raw_audio)
            except Exception as e:
                logger.debug("[auditory] Sensorimotor hear failed: %s", e)
                return

            if not vq_tokens or len(vq_tokens) < 3:
                return

            # Step 2: DTW recognition — match against AcousticWordMemory
            recognized = self.mind.acoustic_word_memory.segment_and_recognize(vq_tokens)

            if recognized:
                # We heard known words!
                recognized_words = [r.word for r in recognized]
                word_str = " ".join(recognized_words)
                self._emit(f"'{word_str}'", "👂")

                for r in recognized:
                    # Step 3a: Feed to grammar engine — this is how Genesis
                    # learns word sequences from hearing, not typing
                    self.mind.grammar.learn_from_speech(r.word)

                    # Step 3b: Activate concept in working memory
                    self.mind.working_memory.attend(
                        key=r.word, content=r.word,
                        salience=r.confidence,
                    )

                    # Step 3c: Activate in semantic memory (strengthen the concept)
                    concept = self.mind.semantic_memory.recall_concept(r.word)
                    if concept and hasattr(concept, 'strength'):
                        concept.strength = min(1.0, concept.strength + 0.05)

                    # Step 3d: Reinforce babbling when words are recognized
                    try:
                        self.mind.babbling_engine.reinforce_last(amount=r.confidence * 0.3)
                    except Exception:
                        pass  # Not critical if babbling reinforcement fails

                # Dopamine reward for successful recognition
                self.mind.neurochemistry.dopamine.spike(0.05 * len(recognized))

                # Store in shared sensory buffer for co-occurrence learning
                import time as _time
                self._recent_heard_words = recognized_words
                self._recent_heard_time = _time.time()

                # AUTO-RESPOND: When we recognize speech, respond naturally
                now = _time.time()
                if now - self._last_auto_respond >= self._auto_respond_cooldown:
                    self._auto_respond(recognized_words)
                    self._last_auto_respond = now

                # Process through neural cascade
                context_vec = self.mind.proprioception.get_context_vector()
                self.mind.subconscious.process_experience(
                    visual_embedding=None,
                    text_embedding=None,
                    context=context_vec,
                    train=True,
                )

            else:
                # Unknown sound — no words recognized
                # This is still a learning signal (novel acoustic data)
                token_str = "-".join(str(t) for t in vq_tokens[:8])
                logger.debug("[auditory] Unrecognized: [%s...] (%d tokens)", token_str, len(vq_tokens))

                # Curiosity spike for unknown sounds (if energetic enough)
                if result.energy > 0.02:
                    self.mind.neurochemistry.cortisol.spike(0.01)  # Mild stress from unknown

        except Exception as e:
            logger.debug("[auditory] Mic not available: %s", e)

    # =========================================================================
    # Thread 8: Emotional State (Continuous Dynamics)
    # =========================================================================

    def _tick_emotional_state(self):
        """
        Tick the emotional system — emotions evolve continuously,
        not just when something happens. Mood shifts slowly.
        """
        self.mind.emotional_state.tick()

        # Enable Theory of Mind at Phase 3 (egocentric before that)
        if (self._get_phase() >= 3 and
                not self.mind.theory_of_mind.is_active):
            self.mind.theory_of_mind.enable()

        # Motor development tracks cognitive development
        self.mind.motor.develop(self._get_phase())

    # =========================================================================
    # Thread 9: Memory Decay (Ebbinghaus Forgetting Curve)
    # =========================================================================

    def _tick_memory_decay(self):
        """
        Apply forgetting curve to all memories.

        Memories that aren't rehearsed decay exponentially.
        Fading memories trigger concern/rehearsal impulses.
        """
        # Apply decay (1 minute of simulated time per tick)
        self.mind.semantic_memory.decay_all(dt_hours=0.017)  # ~1 min

        # Check for fading memories and surface concern
        fading = self.mind.semantic_memory.get_fading_concepts()
        if fading and self._get_phase() >= 2:
            weakest = fading[0]
            if self._get_phase() >= 3:
                self._emit(f"I'm starting to forget about '{weakest.word}'...", "💭")
            else:
                self._emit(self._phase_say("curiosity"), "💭")

            # Auto-rehearse fading concepts (brain tries to hold on)
            self.mind.working_memory.attend(
                key=weakest.word,
                content=weakest,
                salience=0.4,
                emotional_weight=0.1,
            )

    # =========================================================================
    # Thread 10: Play & Episodic Replay
    # =========================================================================

    def _tick_play(self):
        """
        Spontaneous play behavior — combine concepts creatively,
        rehearse memories, replay episodes.

        Play is how children learn. It's not optional.
        """
        mind = self.mind
        phase = self._get_phase()
        concept_count = mind.semantic_memory.count()

        if concept_count < 2:
            return

        # Should we play?
        drive_status = mind.drives.get_status()
        curiosity_level = drive_status.get("curiosity", {}).get("level", 0)
        novelty_level = drive_status.get("novelty", {}).get("level", 0)

        if mind.play.should_play(curiosity_level, novelty_level,
                                  concept_count, phase):
            # Combinatorial play: mix two concepts
            all_concepts = [c.word for c in mind.semantic_memory.get_all_concepts()]

            def get_emb(word):
                c = mind.semantic_memory.recall(word)
                if c and c.text_embedding:
                    return np.array(c.text_embedding)
                return None

            result = mind.play.play_combine(
                all_concepts, get_emb, mind.semantic_memory
            )
            if result and result["is_discovery"]:
                a, b = result["concept_a"], result["concept_b"]
                if phase >= 3:
                    self._emit(
                        f"Playing: '{a}' and '{b}' seem related (similarity: {result['similarity']:.2f})!",
                        "🎮"
                    )
                else:
                    self._emit(self._phase_say("wonder"), "🎮")
                mind.drives.on_autonomous_action()
                mind.emotional_state.on_experience(valence=0.2, arousal=0.1, novelty=0.3)

        # Episodic replay: rehearse a memory
        consolidation = mind.working_memory.get_consolidation_candidates()
        for item in consolidation[:2]:
            mind.working_memory.rehearse(item.key)
            concept = mind.semantic_memory.recall(item.key)
            if concept:
                concept.reinforce(context="episodic_replay")
                mind.metacognition.on_recall_attempt(item.key, success=True)

    # =========================================================================
    # Auto-Response — respond to recognized speech automatically
    # =========================================================================

    def _auto_respond(self, recognized_words: list):
        """
        When Genesis recognizes words, it responds automatically.

        Phase 0-1: Echolalic babble (mimics the sound pattern)
        Phase 2: Echolalia + related concept activation
        Phase 3+: Semantic response using reasoning engine

        This replaces the need for the user to type 'ask' commands.
        """
        phase = self._get_phase()
        mind = self.mind

        if phase <= 1:
            # ECHOLALIA: Repeat back what was heard (like a real infant)
            # Try to play back the acoustic pattern of the first recognized word
            first_word = recognized_words[0] if recognized_words else None
            if first_word:
                try:
                    mind.voice.say_concept(first_word)
                    self._emit(f"...{first_word}...", "🗣")
                except Exception:
                    babble = self._phase_say("social")
                    self._emit(babble, "🗣")
                    mind.voice.say(babble)

        elif phase == 2:
            # PROTO-RESPONSE: Echo + activate related concepts
            response_words = list(recognized_words)
            for word in recognized_words:
                # Find associated concepts via semantic memory
                concept = mind.semantic_memory.recall_concept(word)
                if concept:
                    # Get nearest neighbors in embedding space
                    try:
                        neighbors = mind.semantic_memory.get_nearest(
                            word, n=2
                        )
                        for n in neighbors:
                            if n.word not in response_words:
                                response_words.append(n.word)
                    except Exception:
                        pass

            # Speak the response
            response = " ".join(response_words[:3])
            self._emit(f"...{response}...", "🗣")
            for w in response_words[:2]:
                try:
                    mind.voice.say_concept(w)
                except Exception:
                    pass

        else:
            # FULL RESPONSE: Use reasoning engine for semantic response
            try:
                context_vec = mind.proprioception.get_context_vector()
                result = mind.subconscious.process_experience(
                    visual_embedding=None,
                    text_embedding=None,
                    context=context_vec,
                    train=False,
                )
                neural_voice = mind.subconscious.decode_response(
                    result['personality_response'], mind.semantic_memory
                )
                if neural_voice and neural_voice not in ("(silence)", "(no words yet)"):
                    self._emit(f"...{neural_voice}...", "🗣")
                    mind.voice.say(neural_voice)
                else:
                    # Fall back to echoing + association
                    response = " ".join(recognized_words[:2])
                    self._emit(f"...{response}...", "🗣")
            except Exception as e:
                logger.debug("[auto-respond] Failed: %s", e)

    # =========================================================================
    # Thread 12: Co-occurrence Learning (Auto-Teach)
    # =========================================================================

    def _tick_co_occurrence(self):
        """
        Automatic cross-modal binding — the neural basis of language.

        When Genesis SEES something AND HEARS a word at roughly the
        same time (within 5s), it automatically creates a concept
        binding between them — just like a parent pointing at an
        apple and saying "apple".

        This replaces the need for the 'teach' command. Learning
        happens naturally from co-occurring sensory signals.
        """
        import time as _time
        now = _time.time()

        # Check if both modalities fired recently
        visual_fresh = (
            self._recent_visual_embedding is not None and
            (now - self._recent_visual_time) < self._co_occurrence_window
        )
        auditory_fresh = (
            len(self._recent_heard_words) > 0 and
            (now - self._recent_heard_time) < self._co_occurrence_window
        )

        if not (visual_fresh and auditory_fresh):
            return

        # Co-occurrence detected! Bind what was seen with what was heard.
        visual_emb = self._recent_visual_embedding
        heard_words = self._recent_heard_words

        for word in heard_words:
            # Check if this binding already exists
            existing = self.mind.semantic_memory.recall_concept(word)

            if existing:
                # Strengthen existing binding with new visual data
                if hasattr(existing, 'strength'):
                    existing.strength = min(1.0, existing.strength + 0.1)
                logger.debug(
                    "[co-occur] Strengthened '%s' (seen + heard together)",
                    word,
                )
            else:
                # NEW CONCEPT! Auto-learn from co-occurrence
                try:
                    text_embedding = self.mind.associations.embed_text(word).tolist()
                    self.mind.semantic_memory.learn_concept(
                        word=word,
                        visual_embedding=visual_emb.tolist() if hasattr(visual_emb, 'tolist') else visual_emb,
                        text_embedding=text_embedding,
                        context="Learned from co-occurrence (heard while seeing)",
                        description="Auto-learned concept",
                        emotional_valence="+0.50",
                    )
                    self.mind.associations.create_binding(
                        word=word,
                        visual_embedding=visual_emb,
                        context="Co-occurrence binding",
                    )
                    self.mind.neurochemistry.dopamine.spike(0.15)  # Big reward for learning!
                    self._emit(f"📚 {word}", "🧠")
                    logger.info(
                        "[co-occur] AUTO-LEARNED '%s' from seeing + hearing!",
                        word,
                    )
                except Exception as e:
                    logger.debug("[co-occur] Failed to auto-learn '%s': %s", word, e)

        # Clear the buffers to prevent re-firing
        self._recent_heard_words = []

    # =========================================================================
    # Thread 13: Autonomous Interaction (Self-Initiated Communication)
    # =========================================================================

    def _tick_auto_interact(self):
        """
        Proactive communication — Genesis speaks when it wants to.

        When drives are high enough, Genesis autonomously:
        - Names things it sees (visual-to-word recall)
        - Asks about novel objects (curiosity-driven)
        - Babbles with purpose (high social drive)
        - Rehearses known words (practice)

        This replaces the need for any REPL interaction.
        Genesis is alive and communicating on its own.
        """
        import time as _time
        mind = self.mind
        phase = self._get_phase()
        now = _time.time()

        # Don't interact too frequently
        if now - self._last_auto_respond < self._auto_respond_cooldown:
            return

        drive_status = mind.drives.get_status()
        social = drive_status.get('social', {}).get('level', 0)
        curiosity = drive_status.get('curiosity', {}).get('level', 0)
        novelty = drive_status.get('novelty', {}).get('level', 0)

        # Priority 1: Curiosity about what we're seeing
        if curiosity > 0.6 and self._recent_visual_embedding is not None:
            # Try to name what we see
            if phase >= 2 and mind.semantic_memory.count() > 0:
                try:
                    known = mind.semantic_memory.get_all_embeddings()
                    surprise = mind.curiosity.compute_surprise(
                        self._recent_visual_embedding, known
                    )
                    if surprise < 0.5:
                        # Familiar — try to name it
                        nearest = mind.semantic_memory.find_nearest(
                            self._recent_visual_embedding, modality="visual", n=1
                        )
                        if nearest:
                            word = nearest[0].word if hasattr(nearest[0], 'word') else str(nearest[0])
                            self._emit(f"...{word}...", "👁")
                            mind.voice.say_concept(word)
                            self._last_auto_respond = now
                            mind.drives.on_autonomous_action()
                            return
                except Exception:
                    pass

            # Unknown — express curiosity
            self._emit(self._phase_say("curiosity"), "🤔")
            mind.voice.say(self._phase_say("curiosity"))
            self._last_auto_respond = now
            return

        # Priority 2: Social need — babble or say a known word
        if social > 0.7:
            vocab = mind.acoustic_word_memory.get_vocabulary()
            if vocab and phase >= 2:
                # Practice a random known word
                import random
                word = random.choice(vocab)
                self._emit(f"...{word}...", "🗣")
                mind.voice.say_concept(word)
                mind.drives.social_need.satisfy(0.2)
            else:
                # Babble
                msg = self._phase_say("social")
                self._emit(msg, "🗣")
                mind.voice.say(msg)
                mind.drives.social_need.satisfy(0.15)
            self._last_auto_respond = now
            return

        # Priority 3: Novelty — spontaneous neural vocalization
        if novelty > 0.8 and mind.sensorimotor:
            try:
                waveform, tokens = mind.sensorimotor.generate_spontaneous(
                    max_tokens=20, temperature=1.0
                )
                if len(waveform) > 800:
                    mind.sensorimotor.vocoder.play(waveform)
                    self._emit(self._phase_say("novelty"), "🎵")
                    mind.drives.on_autonomous_action()
            except Exception:
                pass
            self._last_auto_respond = now

    # =========================================================================
    # Thread 14: Neural Growth (The Brain Physically Grows)
    # =========================================================================

    def _tick_neural_growth(self):
        """
        The brain grows with experience — just like a real brain.

        Human brain development:
            - Infant: 700 new synapses per second
            - Child: Myelination, pruning, cortical thickening
            - Adult: Hippocampal neurogenesis continues
            - Never stops: even in old age, learning grows dendrites

        Genesis is the same. Every 30s, this thread checks:
            1. Has the concept count crossed a growth threshold?
            2. Has the developmental phase advanced?
            3. Are sensory cortices saturated (need more capacity)?

        If yes: grow the networks. New neurons inherit from existing
        weight statistics — transfer from self.
        """
        try:
            mind = self.mind
            concept_count = mind.semantic_memory.count()
            phase = mind.development.current_phase

            # Check if core networks need to grow (Personality GRU, MetaController)
            if mind.neuroplasticity.should_grow(phase, mind.subconscious, concept_count):
                report = mind.neuroplasticity.grow_networks(
                    phase, mind.subconscious, concept_count,
                )
                if report.get("params_added", 0) > 0:
                    self._emit(
                        f"🧠 brain grew: +{report['params_added']:,} params "
                        f"({report['params_before']:,} → {report['params_after']:,})",
                        "🌱"
                    )

            # ── Grow Sensory Cortices ──
            # Visual cortex grows with visual experience
            self._grow_sensory_cortex(concept_count, phase)

            # ── Grow Acoustic Pipeline ──
            self._grow_acoustic_pipeline(concept_count, phase)

        except Exception as e:
            logger.debug("[neural-growth] Growth check failed: %s", e)

    def _grow_sensory_cortex(self, concept_count: int, phase: int):
        """
        Grow the visual cortex encoder as visual experience accumulates.

        Like myelination in the visual cortex — pathways that are
        used more get thicker and faster.
        """
        try:
            import math
            eyes = self.mind._get_eyes()
            if eyes is None or not hasattr(eyes, 'visual_cortex'):
                return

            vc = eyes.visual_cortex
            if not hasattr(vc, 'encoder'):
                return

            # Target channel growth: base 32 + sqrt(concepts) * 4
            current_channels = None
            for m in vc.encoder.modules():
                if hasattr(m, 'out_channels'):
                    current_channels = m.out_channels

            if current_channels is None:
                return

            target_channels = 32 + int(math.sqrt(concept_count) * 4)
            target_channels = ((target_channels + 15) // 16) * 16  # Round to 16

            if target_channels > current_channels and concept_count >= 50:
                # Rebuild visual cortex with larger channels
                from genesis.neural.neuroplasticity import _grow_linear
                import torch.nn as nn

                # Just log the growth need for now — full Conv2D growth
                # would require a more complex rebuild
                logger.info(
                    "[neural-growth] 👁 Visual cortex needs growth: %d → %d channels "
                    "(will grow at next restart or via neuroplasticity)",
                    current_channels, target_channels,
                )
        except Exception:
            pass  # Vision may not be available

    def _grow_acoustic_pipeline(self, concept_count: int, phase: int):
        """
        Grow the acoustic cortex and VQ codebook with auditory experience.

        Like auditory cortex tonotopic expansion — areas that process
        frequently heard frequencies grow more neurons.
        """
        try:
            import math
            sm = self.mind.sensorimotor
            if sm is None:
                return

            # ── VQ Codebook Growth ──
            # More concepts → need more VQ entries to represent finer distinctions
            vq = sm.codebook
            current_size = vq.num_entries
            target_size = 256 + int(math.sqrt(concept_count) * 16)
            target_size = min(target_size, 2048)  # Practical cap

            if target_size > current_size and concept_count >= 25:
                # Grow codebook by adding new entries
                import torch
                old_entries = vq.entries.data.clone()
                new_entries = torch.zeros(target_size, vq.dim)

                # Copy existing entries
                new_entries[:current_size] = old_entries

                # Initialize new entries from statistics of existing
                mean = old_entries.mean(dim=0)
                std = old_entries.std(dim=0)
                for i in range(current_size, target_size):
                    new_entries[i] = mean + torch.randn(vq.dim) * std * 0.5

                vq.entries = torch.nn.Parameter(new_entries, requires_grad=False)
                vq.num_entries = target_size

                # Update usage tracking
                if hasattr(vq, '_usage'):
                    old_usage = vq._usage
                    new_usage = torch.zeros(target_size)
                    new_usage[:current_size] = old_usage
                    vq._usage = new_usage

                logger.info(
                    "[neural-growth] 🔊 VQ codebook grew: %d → %d entries",
                    current_size, target_size,
                )
                self._emit(
                    f"🔊 acoustic codebook grew: {current_size} → {target_size} entries",
                    "🌱"
                )

            # ── Auditory Cortex Conv1D Growth ──
            # The encoder processes richer features as it hears more
            ac = sm.auditory_cortex
            if hasattr(ac, 'encoder'):
                # Check if encoder needs more channels
                for name, module in ac.encoder.named_modules():
                    if hasattr(module, 'out_channels'):
                        current_ch = module.out_channels
                        target_ch = 64 + int(math.sqrt(concept_count) * 4)
                        target_ch = ((target_ch + 15) // 16) * 16
                        if target_ch > current_ch and concept_count >= 100:
                            logger.info(
                                "[neural-growth] 🔊 Auditory cortex needs growth: "
                                "%d → %d channels",
                                current_ch, target_ch,
                            )
                        break  # Only check first conv layer

        except Exception as e:
            logger.debug("[neural-growth] Acoustic growth check failed: %s", e)

    # =========================================================================
    # Stats
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        stats = {}
        for name, thread in self._threads.items():
            stats[name] = {
                "running": thread.is_running,
                "ticks": thread._tick_count,
                "errors": thread._errors,
                "interval": thread.interval_sec,
            }
        return stats

    def __repr__(self) -> str:
        running = sum(1 for t in self._threads.values() if t.is_running)
        return f"BrainDaemon(threads={running}/{len(self._threads)})"
