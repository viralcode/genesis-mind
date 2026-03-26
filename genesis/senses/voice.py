"""
Genesis Mind — Voice (Text-to-Speech Output)

A mind without a mouth is incomplete. Genesis can perceive (eyes, ears)
but until now could not speak back. This module gives Genesis a voice
using pyttsx3 — a fully offline, CPU-native TTS engine.

The voice adapts to the developmental phase:
    - Newborn:    Slow, simple utterances
    - Child:      Normal pace
    - Adult:      Natural, fluid speech

The voice runs on a background thread so it never blocks the
consciousness loop.
"""

import logging
import random
import threading
from typing import List, Optional

from genesis.senses.babbling import BabblingEngine, PHONEME_TO_SPEECH

logger = logging.getLogger("genesis.senses.voice")


class Voice:
    """
    Genesis's mouth — text-to-speech output.

    Uses pyttsx3 for fully offline speech synthesis.
    Runs on a background thread to avoid blocking.
    """

    def __init__(self, enabled: bool = True, rate: int = 150, volume: float = 0.9):
        self._enabled = enabled
        self._rate = rate
        self._volume = volume
        self._muted = False
        self._engine = None
        self._lock = threading.Lock()
        self._current_phase = 0  # Developmental phase gate
        self._babbling_engine: Optional[BabblingEngine] = None

        if self._enabled:
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
                self._engine.setProperty('rate', self._rate)
                self._engine.setProperty('volume', self._volume)
                logger.info("Voice initialized (rate=%d, volume=%.1f)", rate, volume)
            except Exception as e:
                logger.warning("Could not initialize TTS engine: %s (voice disabled)", e)
                self._engine = None
                self._enabled = False
        else:
            logger.info("Voice disabled by configuration")

    def set_babbling_engine(self, engine: BabblingEngine):
        """Connect the babbling engine for phase-gated vocalization."""
        self._babbling_engine = engine

    def set_phase(self, phase: int):
        """Update the developmental phase for voice gating."""
        self._current_phase = phase

    def say(self, text: str):
        """
        Speak the given text aloud.

        PHASE-GATED:
        - Phase 0-1: Blocked. Redirects to babbling.
        - Phase 2: Can speak single learned words only.
        - Phase 3+: Full speech unlocked.

        Non-blocking — runs on a background thread.
        """
        if not self._enabled or self._muted or self._engine is None:
            return
        if not text or not text.strip():
            return

        # Phase gate: early phases can only babble
        if self._current_phase <= 1:
            # Redirect to babbling
            self.babble_random()
            return
        elif self._current_phase == 2:
            # Limit to short phrases (max 3 words)
            words = text.split()
            text = " ".join(words[:3])

        thread = threading.Thread(target=self._speak, args=(text,), daemon=True)
        thread.start()

    def speak_phonemes(self, phonemes: List[str]):
        """
        Speak a sequence of phonemes using TTS approximation.

        Converts IPA-like symbols to speakable text and vocalizes them.
        This is how Genesis "babbles" — producing raw sounds without words.
        """
        if not self._enabled or self._muted or self._engine is None:
            return

        speakable_parts = []
        for p in phonemes:
            speakable_parts.append(PHONEME_TO_SPEECH.get(p, p))

        text = " ".join(speakable_parts)
        thread = threading.Thread(target=self._speak, args=(text,), daemon=True)
        thread.start()

    def babble_random(self):
        """
        Generate and speak a random babble.

        Uses the BabblingEngine if available, otherwise produces
        a simple random consonant-vowel pair.
        """
        if not self._enabled or self._muted or self._engine is None:
            return

        if self._babbling_engine:
            text, phonemes = self._babbling_engine.babble()
        else:
            # Fallback: simple random CV babble
            from genesis.senses.babbling import CONSONANTS, VOWELS
            c = random.choice(CONSONANTS)
            v = random.choice(VOWELS)
            text = PHONEME_TO_SPEECH.get(c, c) + PHONEME_TO_SPEECH.get(v, v)

        thread = threading.Thread(target=self._speak, args=(text,), daemon=True)
        thread.start()
        return text

    def _speak(self, text: str):
        """Internal blocking speech — runs on background thread."""
        with self._lock:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                logger.warning("TTS error: %s", e)

    def set_rate(self, rate: int):
        """Set speech rate (words per minute). Default ~150."""
        self._rate = rate
        if self._engine:
            self._engine.setProperty('rate', rate)

    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, volume))
        if self._engine:
            self._engine.setProperty('volume', self._volume)

    def set_rate_for_phase(self, phase: int):
        """Adjust speech rate based on developmental phase."""
        phase_rates = {
            0: 100,   # Newborn: very slow
            1: 120,   # Infant: slow
            2: 135,   # Toddler: moderate
            3: 150,   # Child: normal
            4: 165,   # Adolescent: slightly fast
            5: 175,   # Adult: natural fluency
        }
        rate = phase_rates.get(phase, 150)
        self.set_rate(rate)

    def mute(self):
        """Mute the voice (speech calls become no-ops)."""
        self._muted = True
        logger.info("Voice muted")

    def unmute(self):
        """Unmute the voice."""
        self._muted = False
        logger.info("Voice unmuted")

    @property
    def is_enabled(self) -> bool:
        return self._enabled and self._engine is not None

    @property
    def is_muted(self) -> bool:
        return self._muted

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "muted": self._muted,
            "rate": self._rate,
            "volume": self._volume,
            "engine_active": self._engine is not None,
        }

    def __repr__(self) -> str:
        state = "muted" if self._muted else ("active" if self._enabled else "disabled")
        return f"Voice(state={state}, rate={self._rate})"
