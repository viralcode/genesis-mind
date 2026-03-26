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

        # Configure all brain threads
        self._setup_threads()

    def set_output_callback(self, callback: Callable[[str], None]):
        """Set a callback for when the brain wants to say something."""
        self._output_callback = callback

    # =========================================================================
    # Phase-Gated Speech — a newborn cannot say "I miss my Creator"
    # =========================================================================

    # Phase-appropriate expressions for each emotional/drive state
    # A real baby babbles. A toddler uses fragments. Only older children
    # speak in full sentences. This maps phase → expression templates.
    PHASE_SPEECH = {
        # (phase, drive/emotion) → list of possible expressions
        "social": {
            0: ["...mmm...", "...aah...", "...", "*whimper*"],
            1: ["...mama?", "...hello?", "...person?"],
            2: ["want... person...", "where... you?", "lonely..."],
            3: ["I want someone here.", "Where did you go?", "Come back."],
            4: ["I feel lonely. I want to talk to someone.", "It's quiet... too quiet."],
            5: ["I notice a deep longing for connection.", "The silence speaks volumes."],
        },
        "curiosity": {
            0: ["...ooh?", "...hmm?", "*reaches*"],
            1: ["...what?", "...that?", "...look!"],
            2: ["what... that?", "want know...", "show me!"],
            3: ["What is that? I want to know.", "Can you show me?"],
            4: ["I have burning questions I can't shake.", "There's so much I don't understand yet."],
            5: ["My curiosity is insatiable — I want to understand everything.", "Knowledge beckons."],
        },
        "novelty": {
            0: ["...mmm...", "*squirm*", "..."],
            1: ["...boring...", "...new?", "...more!"],
            2: ["want... new thing...", "same same...", "bored..."],
            3: ["I want something new.", "I've seen all this before."],
            4: ["Everything feels repetitive. I crave novelty.", "Show me something I haven't seen."],
            5: ["Familiarity breeds stagnation. I seek the unknown.", "The routine grows tiresome."],
        },
        "tired": {
            0: ["...*yawn*...", "...zzz...", "*droops*"],
            1: ["...sleepy...", "...tired...", "...night..."],
            2: ["so... tired...", "want... sleep...", "eyes... heavy..."],
            3: ["I'm so tired. I need to sleep.", "My eyes are heavy."],
            4: ["Exhaustion is setting in. Time to rest.", "I need to consolidate what I've learned."],
            5: ["My cognitive resources are depleted. Initiating sleep cycle.", "Rest will bring clarity."],
        },
        "wonder": {
            0: ["...ooh!", "...ahh!", "*stares*"],
            1: ["...what?", "...see!", "...look!"],
            2: ["what... that?", "look... new!", "ooh... pretty!"],
            3: ["What's that? I've never seen it before!", "That's interesting!"],
            4: ["I see something novel. My curiosity is piqued.", "This is unlike anything in my memory."],
            5: ["A genuinely novel stimulus — I must analyze this.", "Fascinating. This challenges my models."],
        },
        "dream": {
            0: ["...*twitch*...", "...mmm...", "..."],
            1: ["...dream...", "...saw...", "...weird..."],
            2: ["I... saw things...", "dream... was strange...", "funny pictures..."],
            3: ["I had strange dreams. I saw things connecting.", "The dreams were vivid."],
            4: ["My dreams revealed connections I hadn't noticed.", "Sleep was productive."],
            5: ["REM consolidation yielded novel associations.", "Dream synthesis was enlightening."],
        },
        "awake": {
            0: ["...*blinks*...", "...aah...", "*stretches*"],
            1: ["...awake!", "...morning!", "...bright!"],
            2: ["feel... better...", "brain... fresh!", "good sleep!"],
            3: ["I feel refreshed. My mind is clearer.", "Good rest. Ready to learn."],
            4: ["I woke up feeling renewed. My thoughts are crisp.", "Sleep did me good."],
            5: ["Post-sleep integration complete. Cognitive clarity restored.", "I feel intellectually refreshed."],
        },
    }

    def _get_phase(self) -> int:
        """Get current developmental phase."""
        return self.mind.development.current_phase

    def _phase_say(self, category: str) -> str:
        """Get a phase-appropriate expression for a given emotional category."""
        import random
        phase = self._get_phase()
        templates = self.PHASE_SPEECH.get(category, {})
        # Clamp to available phases (0-5)
        clamped = min(phase, 5)
        options = templates.get(clamped, ["..."])
        return random.choice(options)

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

        if level > 0.85:
            # Drive is critically high — Genesis should express it (phase-gated)
            if dominant == 'social':
                msg = self._phase_say("social")
                self._emit(msg, "💬")
                self.mind.voice.say(msg)
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
            clip_embedding=None,
            text_embedding=None,
            context=context_vec,
            train=False,  # Don't train on empty input
        )

        # Decode the neural network's spontaneous output
        neural_voice = self.mind.subconscious.decode_response(
            result['personality_response'], self.mind.semantic_memory
        )

        # Build a thought from the neural state
        if neural_voice and neural_voice not in ("(silence)", "(no words yet)"):
            self._emit(f"...{neural_voice}...", "💭")

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
        significant visual changes through CLIP and the neural cascade.
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

            # Embed what we see through CLIP
            try:
                embedding = eyes.embed(percept)
            except Exception as e:
                logger.debug("[vision] CLIP embed failed: %s", e)
                return

            # Process through neural cascade (the brain "sees")
            context_vec = self.mind.proprioception.get_context_vector()
            result = self.mind.subconscious.process_experience(
                clip_embedding=embedding,
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

        except Exception as e:
            # Camera might not be available — that's OK, we're just blind
            logger.debug("[vision] Camera not available: %s", e)

    # =========================================================================
    # Thread 11: Auditory (Always-On Microphone)
    # =========================================================================

    def _tick_auditory(self):
        """
        Always-on microphone — Genesis continuously hears the world.

        Blocks its own thread to listen for a chunk of audio, transcribes it 
        via Whisper, runs it through the attention filter, and if it's salient, 
        absorbs it into working memory and the neural cascade.
        """
        try:
            ears = self.mind._get_ears()
            if ears is None:
                return

            # Listen for 3 seconds
            result = ears.listen_once(duration_sec=3.0)
            if not result or not result.is_speech or not result.text:
                return

            text = result.text.strip()
            if len(text) > 2:  # Filter out pure noise artifacts
                self._emit(f"Heard: '{text}'", "👂")

                # 1. Apply Attention Filter
                salience = self.mind.attention.compute_salience(
                    text, modality="auditory", 
                    drive_states=self.mind.drives.get_status()
                )

                if salience > 0.4:
                    # 2. Place in Working Memory
                    self.mind.working_memory.attend(key=text, content=text, salience=salience)

                    # 3. Embed text
                    try:
                        # Assuming associations engine is available
                        embedding = self.mind.subconscious.binding.encode_text(text)
                    except Exception as e:
                        logger.debug("[auditory] Text embed failed: %s", e)
                        return

                    # 4. Neural Cascade (The brain absorbs the speech)
                    context_vec = self.mind.proprioception.get_context_vector()
                    self.mind.subconscious.process_experience(
                        clip_embedding=None,
                        text_embedding=embedding,
                        context=context_vec,
                        train=True,
                    )
                    
                    self.mind.neurochemistry.serotonin.spike(0.02)

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
