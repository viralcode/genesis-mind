"""
Genesis Mind — Sensorimotor Loop

The sensorimotor loop is the master pipeline that ties together
ALL acoustic processing into a single coherent cycle:

    HEAR → ENCODE → QUANTIZE → THINK → SYNTHESIZE → SPEAK → SELF-MONITOR

This is the neural equivalent of:
    Ear → Auditory Cortex → Broca's Area → Motor Cortex → Vocal Tract → Ear

The loop runs continuously when Genesis is in acoustic mode.
Every piece of heard audio trains the models. Every generated
response is re-heard (proprioceptive feedback).

There is NO TEXT anywhere in this pipeline.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from genesis.neural.auditory_cortex import AuditoryCortex
from genesis.neural.vq_codebook import VQCodebook
from genesis.neural.acoustic_lm import AcousticBrain, BOS_TOKEN
from genesis.neural.neural_vocoder import NeuralVocoder

logger = logging.getLogger("genesis.neural.sensorimotor")


class SensorimotorLoop:
    """
    The full acoustic neural pipeline — hear, think, speak.
    
    Components (all from scratch, zero pretrained models):
    - Ear input    → AuditoryCortex (Conv1D mel encoder, ~50K params)
    - Perception   → VQ Codebook (EMA-based, 256 entries)
    - Cognition    → AcousticBrain (Transformer on audio tokens)
    - Output       → NeuralVocoder (Griffin-Lim synthesis)
    """

    def __init__(self, weights_dir: Path,
                 sample_rate: int = 16000,
                 n_mels: int = 80,
                 latent_dim: int = 64,
                 codebook_size: int = 256,
                 lm_layers: int = 4,
                 lm_heads: int = 4,
                 lm_embd: int = 128):

        self.weights_dir = weights_dir
        weights_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

        # --- HEARING: Raw Audio → Neural Representation ---
        self.auditory_cortex = AuditoryCortex(
            sample_rate=sample_rate, n_mels=n_mels,
            latent_dim=latent_dim, lr=0.001,
        )

        # --- PERCEPTION: Continuous → Discrete Tokens ---
        self.vq_codebook = VQCodebook(
            codebook_size=codebook_size, latent_dim=latent_dim,
        )

        # --- THINKING: Token Sequence → Response Tokens ---
        self.acoustic_brain = AcousticBrain(
            n_embd=lm_embd, n_head=lm_heads, n_layer=lm_layers,
            lr=0.0003,
        )

        # --- SPEAKING: Tokens → Waveform ---
        self.vocoder = NeuralVocoder(
            latent_dim=latent_dim, n_mels=n_mels,
            sample_rate=sample_rate, lr=0.001,
        )

        # Context buffer: recent heard tokens
        self._heard_buffer: List[int] = []
        self._max_context = 200

        # State tracking
        self._total_interactions = 0
        self._last_heard_tokens: Optional[List[int]] = None
        self._last_response_tokens: Optional[List[int]] = None
        self._last_output_waveform: Optional[np.ndarray] = None

        # Load weights
        self._load_all()

        total_params = (
            self.auditory_cortex.get_params() +
            self.vq_codebook.codebook_size * self.vq_codebook.latent_dim +
            self.acoustic_brain.get_params() +
            self.vocoder.get_params()
        )
        logger.info("═" * 55)
        logger.info("  SENSORIMOTOR LOOP INITIALIZED — %d total params", total_params)
        logger.info("  Hearing:    Auditory Cortex (%d params)", self.auditory_cortex.get_params())
        logger.info("  Perception: VQ Codebook (%d entries × %d dim)", codebook_size, latent_dim)
        logger.info("  Thinking:   Acoustic LM (%d params)", self.acoustic_brain.get_params())
        logger.info("  Speaking:   Neural Vocoder (%d params + Griffin-Lim)", self.vocoder.get_params())
        logger.info("  NO TEXT. NO PRE-TRAINING. PURE NEURAL AUDIO.")
        logger.info("═" * 55)

    # --- CORE LOOP ---

    def hear(self, waveform: np.ndarray) -> List[int]:
        """
        Process heard audio through the full perception pipeline.
        
        Raw audio → Mel → Encoder → VQ tokens → stored in context
        → Acoustic LM trained on these tokens
        
        Args:
            waveform: numpy array of 16kHz mono audio
            
        Returns:
            tokens: list of discrete VQ token IDs
        """
        # Step 1: Auditory cortex encodes to latent
        latent = self.auditory_cortex.hear(waveform)  # (1, D, T)

        # Step 2: VQ quantizes to discrete tokens
        self.vq_codebook.train()
        z_q, token_ids, vq_loss = self.vq_codebook(latent)  # (1, D, T), (1, T), scalar

        tokens = token_ids[0].tolist()

        # Step 3: Train the Acoustic LM on these tokens
        lm_loss = self.acoustic_brain.learn_from_tokens(tokens)

        # Step 4: Update context buffer
        self._heard_buffer.extend(tokens)
        if len(self._heard_buffer) > self._max_context:
            self._heard_buffer = self._heard_buffer[-self._max_context:]

        self._last_heard_tokens = tokens

        logger.debug(
            "Heard %d tokens (VQ loss=%.4f, LM loss=%.4f)",
            len(tokens), vq_loss.item(), lm_loss,
        )

        return tokens

    def think(self, max_response_tokens: int = 50,
              temperature: float = 0.8) -> List[int]:
        """
        Generate a response — the acoustic equivalent of "thinking
        what to say".
        
        Uses the Acoustic LM to predict a sequence of audio tokens
        conditioned on recently heard tokens.
        
        Returns:
            response_tokens: list of discrete VQ token IDs
        """
        context = self._heard_buffer[-100:] if self._heard_buffer else None

        response = self.acoustic_brain.generate_response(
            context_tokens=context,
            max_tokens=max_response_tokens,
            temperature=temperature,
        )

        self._last_response_tokens = response
        return response

    def speak(self, token_ids: List[int]) -> np.ndarray:
        """
        Convert token IDs to audible speech via the neural vocoder.
        
        Args:
            token_ids: list of VQ token IDs
            
        Returns:
            waveform: numpy array of audio samples
        """
        if not token_ids:
            return np.zeros(1600, dtype=np.float32)  # 0.1s silence

        ids_tensor = torch.tensor([token_ids], dtype=torch.long)

        # Tokens → codebook embeddings
        embeddings = self.vq_codebook.tokens_to_embeddings(ids_tensor)  # (1, D, T)

        # Embeddings → waveform
        waveform = self.vocoder.synthesize_from_embeddings(embeddings)

        self._last_output_waveform = waveform
        return waveform

    def speak_and_play(self, token_ids: List[int]) -> np.ndarray:
        """Synthesize and play through speakers."""
        waveform = self.speak(token_ids)
        self.vocoder.play(waveform)
        return waveform

    def respond(self, heard_waveform: np.ndarray,
                temperature: float = 0.8) -> Tuple[np.ndarray, List[int]]:
        """
        Full response cycle: hear → think → speak.
        
        This is the complete sensorimotor loop:
        1. Process input audio
        2. Generate response tokens
        3. Synthesize output audio
        4. Self-monitor (re-encode output for proprioception)
        
        Returns:
            (output_waveform, response_tokens)
        """
        # HEAR
        heard_tokens = self.hear(heard_waveform)

        # THINK
        response_tokens = self.think(temperature=temperature)

        # SPEAK
        output_waveform = self.speak(response_tokens)

        # SELF-MONITOR: re-encode own output for proprioceptive feedback
        if len(output_waveform) > 1600:  # At least 0.1s
            self._self_monitor(output_waveform)

        self._total_interactions += 1

        return output_waveform, response_tokens

    def generate_spontaneous(self, max_tokens: int = 30,
                             temperature: float = 1.0) -> Tuple[np.ndarray, List[int]]:
        """
        Spontaneous vocalization — babble without hearing anything first.
        
        Used when drives are high but no input is available.
        """
        tokens = self.acoustic_brain.generate_response(
            context_tokens=None, max_tokens=max_tokens,
            temperature=temperature,
        )
        waveform = self.speak(tokens)
        self._total_interactions += 1
        return waveform, tokens

    # --- INTERNAL ---

    def _self_monitor(self, waveform: np.ndarray):
        """
        Proprioceptive feedback: re-encode own speech.
        
        This is how the brain monitors its own motor output —
        hearing yourself speak and using that to improve.
        """
        latent = self.auditory_cortex.hear(waveform)
        self.vq_codebook.eval()
        with torch.no_grad():
            z_q, token_ids, _ = self.vq_codebook(latent)

        # Train the vocoder's mel reconstructor
        mel_target = self.auditory_cortex.mel_filter(
            torch.tensor(waveform, dtype=torch.float32)
        ).unsqueeze(0)

        # Only train if sizes are compatible
        try:
            self.vocoder.train_reconstruction(z_q, mel_target)
        except Exception:
            pass  # Size mismatch during early training

    # --- QUERIES ---

    def get_stats(self) -> Dict:
        """Get comprehensive stats across all acoustic components."""
        return {
            "total_interactions": self._total_interactions,
            "auditory_cortex": self.auditory_cortex.get_stats(),
            "vq_codebook": self.vq_codebook.get_stats(),
            "acoustic_brain": self.acoustic_brain.get_stats(),
            "vocoder": self.vocoder.get_stats(),
            "context_buffer_size": len(self._heard_buffer),
            "total_params": (
                self.auditory_cortex.get_params() +
                self.vq_codebook.codebook_size * self.vq_codebook.latent_dim +
                self.acoustic_brain.get_params() +
                self.vocoder.get_params()
            ),
        }

    # --- PERSISTENCE ---

    def save_all(self):
        """Save all acoustic neural weights."""
        self.auditory_cortex.save_weights(self.weights_dir / "auditory_cortex.pt")
        self.vq_codebook.save_weights(self.weights_dir / "vq_codebook.pt")
        self.acoustic_brain.save_weights(self.weights_dir / "acoustic_lm.pt")
        self.vocoder.save_weights(self.weights_dir / "neural_vocoder.pt")
        logger.info("All acoustic weights saved to %s", self.weights_dir)

    def _load_all(self):
        """Load all acoustic neural weights."""
        self.auditory_cortex.load_weights(self.weights_dir / "auditory_cortex.pt")
        self.vq_codebook.load_weights(self.weights_dir / "vq_codebook.pt")
        self.acoustic_brain.load_weights(self.weights_dir / "acoustic_lm.pt")
        self.vocoder.load_weights(self.weights_dir / "neural_vocoder.pt")

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"SensorimotorLoop(params={stats['total_params']:,}, "
            f"interactions={stats['total_interactions']})"
        )


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Sensorimotor Loop Test")
    print("Testing the full hear → think → speak cycle...")
    print("=" * 60)

    loop = SensorimotorLoop(
        weights_dir=Path("/tmp/genesis_acoustic_test"),
    )

    # Create synthetic audio input (1s of 440Hz sine wave = "someone speaking")
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    input_audio = 0.3 * np.sin(2 * np.pi * 440 * t)

    # --- Test 1: Hear ---
    print("\n--- Test 1: Hear ---")
    tokens = loop.hear(input_audio)
    print(f"  Input:  {len(input_audio)} samples (1.0s)")
    print(f"  Output: {len(tokens)} tokens: {tokens[:10]}...")

    # --- Test 2: Think ---
    print("\n--- Test 2: Think ---")
    response = loop.think(max_response_tokens=20)
    print(f"  Response: {len(response)} tokens: {response[:10]}...")

    # --- Test 3: Speak ---
    print("\n--- Test 3: Speak ---")
    waveform = loop.speak(response)
    print(f"  Waveform: {len(waveform)} samples ({len(waveform)/16000:.2f}s)")

    # --- Test 4: Full loop ---
    print("\n--- Test 4: Full respond() cycle ---")
    output, resp_tokens = loop.respond(input_audio)
    print(f"  Heard → Thought → Spoke")
    print(f"  Output: {len(output)} samples ({len(output)/16000:.2f}s)")
    print(f"  Tokens: {resp_tokens[:10]}...")

    # --- Test 5: Spontaneous babble ---
    print("\n--- Test 5: Spontaneous vocalization ---")
    babble_wav, babble_tokens = loop.generate_spontaneous(max_tokens=15)
    print(f"  Babble: {len(babble_tokens)} tokens → {len(babble_wav)} samples")

    # Stats
    print(f"\n  Stats: {loop.get_stats()}")
    print("\nSensorimotor Loop test PASSED")
