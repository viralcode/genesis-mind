"""
Genesis Mind — Continuous Background Consciousness (Perception Loop)

In V1, Genesis was command-driven: it only perceived the world
when you typed 'teach' or 'ask'. A real mind is always ON.

This module provides three parallel threads that run continuously:

    ┌─────────────────────────────────────────────────────────┐
    │                  PERCEPTION LOOP                        │
    │                                                         │
    │   Thread 1: VISUAL    — Camera every 2s, motion detect  │
    │   Thread 2: AUDITORY  — Microphone, AuditoryCortex      │
    │   Thread 3: THOUGHT   — Spontaneous inner monologue     │
    │                                                         │
    │   All threads → PerceptionQueue → Main Consciousness    │
    └─────────────────────────────────────────────────────────┘

The main loop consumes the queue and processes each perception
through memory recall, emotional evaluation, and curiosity scoring.
"""

import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Any
from enum import Enum

logger = logging.getLogger("genesis.cortex.perception_loop")


class PerceptionType(Enum):
    VISUAL = "visual"
    AUDITORY = "auditory"
    THOUGHT = "thought"
    CREATOR_INPUT = "creator_input"


@dataclass
class Perception:
    """A single unit of perceived experience from any modality."""
    type: PerceptionType
    content: Any               # Raw content (frame, text, thought)
    embedding: Optional[Any] = None  # Vector embedding if available
    raw_audio: Optional[Any] = None  # V7: Raw audio waveform for acoustic pipeline
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    motion_score: float = 0.0  # For visual perceptions
    is_speech: bool = False    # For auditory perceptions
    surprise: float = 0.0      # Novelty score (set by curiosity engine)


class PerceptionLoop:
    """
    Continuous background consciousness.

    Runs three async threads that constantly feed perceptions
    into a shared queue. The main consciousness loop consumes
    this queue and processes each perception.
    """

    def __init__(self, eyes_factory, ears_factory,
                 visual_interval: float = 2.0,
                 thought_interval: float = 10.0,
                 max_queue_size: int = 100):
        self._eyes_factory = eyes_factory
        self._ears_factory = ears_factory
        self._visual_interval = visual_interval
        self._thought_interval = thought_interval

        self._queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._threads = []

        # Track what we've seen for motion filtering
        self._last_visual_embedding = None
        self._perception_count = 0

        logger.info("Perception loop initialized (visual=%.1fs, thought=%.1fs)",
                     visual_interval, thought_interval)

    @property
    def queue(self) -> queue.Queue:
        return self._queue

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def perception_count(self) -> int:
        return self._perception_count

    def start(self):
        """Start all perception threads."""
        if self._running:
            logger.warning("Perception loop already running")
            return

        self._running = True

        # Thread 1: Visual perception
        t_visual = threading.Thread(
            target=self._visual_loop,
            name="genesis-eyes",
            daemon=True,
        )

        # Thread 2: Auditory perception
        t_audio = threading.Thread(
            target=self._auditory_loop,
            name="genesis-ears",
            daemon=True,
        )

        # Thread 3: Spontaneous thought
        t_thought = threading.Thread(
            target=self._thought_loop,
            name="genesis-thought",
            daemon=True,
        )

        self._threads = [t_visual, t_audio, t_thought]
        for t in self._threads:
            t.start()

        logger.info("═══════════════════════════════════════════")
        logger.info("  CONSCIOUSNESS ACTIVATED — 3 threads live")
        logger.info("═══════════════════════════════════════════")

    def stop(self):
        """Stop all perception threads."""
        self._running = False
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()
        logger.info("Perception loop stopped")

    def inject(self, perception: Perception):
        """Manually inject a perception (e.g., Creator typed something)."""
        try:
            self._queue.put_nowait(perception)
            self._perception_count += 1
        except queue.Full:
            logger.warning("Perception queue full, dropping oldest")
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(perception)
            except queue.Empty:
                pass

    def _enqueue(self, perception: Perception):
        """Internal: add a perception to the queue."""
        try:
            self._queue.put_nowait(perception)
            self._perception_count += 1
        except queue.Full:
            pass  # Drop if full — consciousness can't process everything

    # =========================================================================
    # Thread 1: Visual Perception
    # =========================================================================
    def _visual_loop(self):
        """Continuously observe the world through the camera."""
        eyes = None
        try:
            eyes = self._eyes_factory()
        except Exception as e:
            logger.warning("Could not initialize eyes: %s (visual thread disabled)", e)
            return

        logger.info("👁  Visual thread started")

        while self._running:
            try:
                percept = eyes.look()
                if percept and percept.is_significant:
                    # Only enqueue if something meaningful changed
                    try:
                        embedding = eyes.embed(percept)
                        self._enqueue(Perception(
                            type=PerceptionType.VISUAL,
                            content=percept,
                            embedding=embedding,
                            motion_score=percept.motion_score,
                        ))
                    except Exception as e:
                        logger.debug("Visual embedding failed: %s", e)

                time.sleep(self._visual_interval)

            except Exception as e:
                logger.error("Visual loop error: %s", e)
                time.sleep(5.0)  # Back off on error

        if eyes:
            eyes.close()

    # =========================================================================
    # Thread 2: Auditory Perception
    # =========================================================================
    def _auditory_loop(self):
        """Continuously listen through the microphone."""
        try:
            ears = self._ears_factory()
        except Exception as e:
            logger.warning("Could not initialize ears: %s (auditory thread disabled)", e)
            return

        logger.info("👂 Auditory thread started")

        while self._running:
            try:
                # Listen for a chunk of audio
                result = ears.listen_once()
                if result is None:
                    continue

                # V7: Always pass raw audio for acoustic neural training
                raw_audio = getattr(result, 'raw_audio', None)
                text = getattr(result, 'text', '') or ''
                is_speech = getattr(result, 'is_speech', False)

                if is_speech and text.strip() and len(text.strip()) > 2:
                    self._enqueue(Perception(
                        type=PerceptionType.AUDITORY,
                        content=text.strip(),
                        raw_audio=raw_audio,
                        is_speech=True,
                    ))
                    logger.info("👂 Heard: '%s'", text[:80])
                elif raw_audio is not None and is_speech:
                    # Even without text, pass raw audio for neural training
                    self._enqueue(Perception(
                        type=PerceptionType.AUDITORY,
                        content='',
                        raw_audio=raw_audio,
                        is_speech=True,
                    ))

            except Exception as e:
                logger.error("Auditory loop error: %s", e)
                time.sleep(2.0)

    # =========================================================================
    # Thread 3: Spontaneous Thought
    # =========================================================================
    def _thought_loop(self):
        """
        Generate spontaneous inner thoughts periodically.

        This simulates the default mode network — the brain's activity
        when not focused on external tasks. It reviews recent perceptions
        and generates reflective thoughts.
        """
        logger.info("💭 Thought thread started")

        while self._running:
            try:
                time.sleep(self._thought_interval)
                if not self._running:
                    break

                # Generate a spontaneous thought request
                self._enqueue(Perception(
                    type=PerceptionType.THOUGHT,
                    content="spontaneous_reflection",
                ))

            except Exception as e:
                logger.error("Thought loop error: %s", e)
                time.sleep(5.0)

    def __repr__(self) -> str:
        status = "running" if self._running else "stopped"
        return f"PerceptionLoop(status={status}, perceptions={self._perception_count})"
