"""
Genesis Mind — Voice (Neural Vocalization)

Genesis speaks through its NeuralVocoder — a from-scratch audio synthesis
system. There is NO pretrained TTS engine (pyttsx3 removed).

The voice system works in two modes:
    1. Babble mode (Phase 0-2): Random phoneme sequences through the
       babbling engine, played via the NeuralVocoder.
    2. Neural mode (Phase 3+): The acoustic language model generates
       token sequences, which the vocoder synthesizes into audio.

This module is a thin interface that the rest of the system calls.
The actual synthesis is done by the sensorimotor loop.
"""

import logging
import threading
from typing import List, Optional

from genesis.senses.babbling import BabblingEngine

logger = logging.getLogger("genesis.senses.voice")


class Voice:
    """
    Genesis's mouth — neural vocalization only.

    No pyttsx3. No pretrained TTS. Voice output comes from either
    the babbling engine or the neural vocoder.
    """

    def __init__(self, enabled: bool = True, rate: int = 150, volume: float = 0.9):
        self._enabled = enabled
        self._rate = rate
        self._volume = volume
        self._muted = False
        self._lock = threading.Lock()
        self._current_phase = 0
        self._babbling_engine: Optional[BabblingEngine] = None
        self._sensorimotor = None  # Injected from GenesisMind
        self._acoustic_memory = None  # Injected from GenesisMind

        logger.info("Voice initialized (neural-only, rate=%d, volume=%.1f)", rate, volume)

    def set_babbling_engine(self, engine: BabblingEngine):
        """Connect the babbling engine for phase-gated vocalization."""
        self._babbling_engine = engine

    def set_sensorimotor(self, sensorimotor):
        """Connect the sensorimotor loop for neural audio synthesis."""
        self._sensorimotor = sensorimotor

    def set_acoustic_memory(self, memory):
        """Connect acoustic word memory for concept-based speech."""
        self._acoustic_memory = memory

    def set_phase(self, phase: int):
        """Update the developmental phase for voice gating."""
        self._current_phase = phase

    def say(self, text: str):
        """
        Neural vocalization.

        Phase 0-1: Redirects to babbling (random phoneme sequences).
        Phase 2+: Attempts neural vocoder synthesis. Falls back to babble.

        This is a no-op for text content — Genesis doesn't have a
        pretrained TTS engine. It can only produce neural audio.
        """
        if not self._enabled or self._muted:
            return
        if not text or not text.strip():
            return

        # All phases: attempt neural babble
        if self._current_phase <= 1:
            self.babble_random()
        elif self._sensorimotor:
            # Try proto-speech: reproduce known word acoustic patterns
            thread = threading.Thread(
                target=self._neural_speak, daemon=True
            )
            thread.start()
        else:
            self.babble_random()

    def say_concept(self, word: str):
        """
        Speak a specific learned concept by playing its stored
        acoustic VQ tokens through the vocoder.

        This is proto-speech — reproducing a previously heard
        sound pattern for a known concept.
        """
        if not self._enabled or self._muted:
            return
        if not self._acoustic_memory or not self._sensorimotor:
            self.babble_random()
            return

        tokens = self._acoustic_memory.get_exemplar_tokens(word)
        if tokens:
            thread = threading.Thread(
                target=self._play_tokens, args=(tokens,), daemon=True
            )
            thread.start()
        else:
            self.babble_random()

    def _play_tokens(self, tokens: list):
        """Play VQ token sequence through the vocoder."""
        with self._lock:
            try:
                if self._sensorimotor:
                    waveform = self._sensorimotor.speak(tokens)
                    if len(waveform) > 800:
                        self._sensorimotor.vocoder.play(waveform)
            except Exception as e:
                logger.debug("Token playback failed: %s", e)

    def _neural_speak(self):
        """Generate and play neural audio through the sensorimotor loop."""
        with self._lock:
            try:
                if self._sensorimotor:
                    waveform, tokens = self._sensorimotor.generate_spontaneous(
                        max_tokens=40, temperature=0.9
                    )
                    if len(waveform) > 800:
                        self._sensorimotor.vocoder.play(waveform)
            except Exception as e:
                logger.debug("Neural vocalization failed: %s", e)

    def babble_random(self):
        """
        Generate and vocalize a random babble.

        Uses the BabblingEngine to produce random phoneme sequences.
        Returns the babble text for display.
        """
        if not self._enabled or self._muted:
            return ""

        if self._babbling_engine:
            text, phonemes = self._babbling_engine.babble()
            # Vocalize through neural vocoder if available
            if self._sensorimotor:
                thread = threading.Thread(
                    target=self._neural_speak, daemon=True
                )
                thread.start()
            return text
        return ""

    def mute(self):
        """Mute the voice."""
        self._muted = True
        logger.info("Voice muted")

    def unmute(self):
        """Unmute the voice."""
        self._muted = False
        logger.info("Voice unmuted")

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def is_muted(self) -> bool:
        return self._muted

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "muted": self._muted,
            "rate": self._rate,
            "volume": self._volume,
            "engine": "neural_vocoder",
            "has_sensorimotor": self._sensorimotor is not None,
        }

    def __repr__(self) -> str:
        state = "muted" if self._muted else ("active" if self._enabled else "disabled")
        return f"Voice(state={state}, neural_only=True)"
