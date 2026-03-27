"""
Genesis Mind — The Ears (V8: No Pretrained Models)

The auditory perception system. Opens the microphone, captures audio,
and processes it through the from-scratch auditory cortex.

V8 CHANGE: Removed OpenAI Whisper (39M pretrained params).
Audio is now processed as raw mel spectrograms → VQ codebook → acoustic
tokens. Genesis does NOT transcribe audio to text. It hears raw sound
patterns and learns to associate them with concepts through experience.

This is how a real infant hears — raw acoustic streams, not words.
"""

import time
import logging
import threading
import queue
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Callable

import numpy as np

logger = logging.getLogger("genesis.senses.ears")


@dataclass
class AuditoryPercept:
    """
    A single moment of hearing.

    Contains raw audio data and acoustic tokens (NOT text).
    Text transcription is gone — Genesis hears raw sound patterns.
    """
    raw_audio: Optional[np.ndarray] = None  # Raw audio waveform
    mel_spectrogram: Optional[np.ndarray] = None  # Mel spectrogram
    acoustic_tokens: Optional[List[int]] = None   # VQ codebook indices
    acoustic_embedding: Optional[np.ndarray] = None  # 64-dim embedding
    sample_rate: int = 16000
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    duration_sec: float = 0.0
    energy: float = 0.0
    is_speech: bool = False
    # text field kept for backward compat with CLI, but populated only
    # through acoustic pattern matching, not transcription
    text: str = ""
    words: List[str] = field(default_factory=list)


class Ears:
    """
    The auditory perception system of Genesis.

    V8: No Whisper. Processes raw audio as mel spectrograms and acoustic
    tokens. Genesis hears sound patterns, not English words.
    """

    def __init__(self, sample_rate: int = 16000, chunk_duration_sec: float = 3.0,
                 silence_threshold: float = 0.001, auditory_cortex=None,
                 **kwargs):  # Accept and ignore whisper_model_name for compat
        self.sample_rate = sample_rate
        self.chunk_duration_sec = chunk_duration_sec
        self.silence_threshold = silence_threshold
        self._auditory_cortex = auditory_cortex

        self._listening = False
        self._audio_queue: queue.Queue = queue.Queue()
        self._listen_thread: Optional[threading.Thread] = None

        logger.info(
            "Ears initialized (rate=%dHz, chunk=%.1fs, from-scratch)",
            sample_rate, chunk_duration_sec,
        )

    def set_auditory_cortex(self, cortex):
        """Set the auditory cortex reference (for late binding)."""
        self._auditory_cortex = cortex

    def _compute_energy(self, audio: np.ndarray) -> float:
        """Compute the RMS energy of an audio chunk (overflow-safe)."""
        samples = audio.astype(np.float64)
        samples = np.clip(samples, -1e9, 1e9)  # guard against corrupt values
        rms = np.sqrt(np.mean(samples ** 2))
        return float(rms) if np.isfinite(rms) else 0.0

    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute a mel spectrogram from raw audio — NO pretrained models.
        
        This is pure signal processing (FFT + mel filterbank), not learned.
        """
        import torch
        import torchaudio.transforms as T

        waveform = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)
        
        mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=64,
        )
        mel = mel_transform(waveform)
        
        # Log scale for better dynamic range
        mel = torch.log(mel + 1e-9)
        
        return mel.squeeze(0).numpy()

    def listen_once(self, duration_sec: float = None) -> Optional[AuditoryPercept]:
        """
        Listen for a fixed duration and return what was heard.
        
        Returns raw audio + mel spectrogram + acoustic tokens.
        NO text transcription — Genesis hears patterns, not words.
        """
        import sounddevice as sd

        duration = duration_sec or self.chunk_duration_sec
        logger.debug("Listening for %.1f seconds...", duration)

        try:
            audio = sd.rec(
                int(duration * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            audio = audio.flatten()

            energy = self._compute_energy(audio)
            is_speech = energy > self.silence_threshold

            percept = AuditoryPercept(
                raw_audio=audio,
                sample_rate=self.sample_rate,
                duration_sec=duration,
                energy=energy,
                is_speech=is_speech,
            )

            if is_speech:
                # Compute mel spectrogram (pure signal processing, not pretrained)
                try:
                    percept.mel_spectrogram = self._compute_mel_spectrogram(audio)
                except Exception as e:
                    logger.debug("Mel computation failed: %s", e)

                # If auditory cortex is available, get acoustic embedding
                if self._auditory_cortex is not None:
                    try:
                        result = self._auditory_cortex.process_audio(audio)
                        if result:
                            percept.acoustic_embedding = result.get("embedding")
                            percept.acoustic_tokens = result.get("tokens")
                    except Exception as e:
                        logger.debug("Auditory cortex processing failed: %s", e)

                # For backward compatibility with CLI: represent acoustic
                # content as a simple energy/pattern descriptor
                if percept.acoustic_tokens:
                    token_str = "-".join(str(t) for t in percept.acoustic_tokens[:8])
                    percept.text = f"[acoustic:{token_str}]"
                elif is_speech:
                    percept.text = f"[sound:e={energy:.3f}]"

            return percept

        except Exception as e:
            logger.error("Failed to capture audio: %s", e)
            return None

    def start_continuous_listening(self, callback: Callable[[AuditoryPercept], None]):
        """Start listening continuously in a background thread."""
        if self._listening:
            return

        self._listening = True

        def _listen_loop():
            logger.info("Continuous listening started")
            while self._listening:
                percept = self.listen_once()
                if percept and percept.is_speech:
                    callback(percept)

        self._listen_thread = threading.Thread(target=_listen_loop, daemon=True)
        self._listen_thread.start()

    def stop_continuous_listening(self):
        """Stop the continuous listening thread."""
        self._listening = False
        if self._listen_thread:
            self._listen_thread.join(timeout=5.0)
            self._listen_thread = None
            logger.info("Continuous listening stopped")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stop_continuous_listening()


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Ears Test (V8: No Whisper)")
    print("=" * 60)

    with Ears() as ears:
        for i in range(3):
            print(f"\n--- Listening ({i+1}/3) ---")
            percept = ears.listen_once(duration_sec=3.0)
            if percept:
                print(f"  Energy: {percept.energy:.4f}")
                print(f"  Speech detected: {percept.is_speech}")
                if percept.mel_spectrogram is not None:
                    print(f"  Mel shape: {percept.mel_spectrogram.shape}")
                if percept.acoustic_tokens:
                    print(f"  Acoustic tokens: {percept.acoustic_tokens[:8]}")
