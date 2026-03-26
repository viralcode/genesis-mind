"""
Genesis Mind — Auditory Cortex

The biological auditory cortex converts pressure waves (sound)
into neural representations through a series of processing stages:

    Cochlea → Tonotopic mapping → Primary Auditory Cortex (A1)
    → Belt/Parabelt regions → Wernicke's area

This module replicates this hierarchy digitally:

    1. COCHLEAR FILTER:  Raw waveform → Mel Spectrogram (80 bands)
                         (like the basilar membrane's frequency
                          decomposition)
    2. A1 ENCODER:       Mel frames → Conv1D stack → 64-dim latent
                         (learns spectro-temporal patterns)
    3. TEMPORAL POOL:    Frame-level features → sequence summary

All weights are learned from scratch. No pre-trained embeddings.
Genesis starts deaf and learns to hear.
"""

import logging
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("genesis.neural.auditory_cortex")


# =============================================================================
# Mel Spectrogram — The Digital Cochlea
# =============================================================================

class MelFilterBank:
    """
    Compute Mel spectrograms from raw waveforms using pure PyTorch.
    No torchaudio dependency — uses hand-built mel filter banks.
    
    This is the "cochlea" — it decomposes sound into frequency bands,
    just like the basilar membrane in the inner ear.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 512,
                 hop_length: int = 160, n_mels: int = 80,
                 f_min: float = 0.0, f_max: float = 8000.0):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Build the mel filter bank matrix
        self.mel_fb = self._build_mel_filterbank(
            n_fft, n_mels, sample_rate, f_min, f_max
        )
        # Hann window for STFT
        self.window = torch.hann_window(n_fft)

    def _hz_to_mel(self, f: float) -> float:
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def _mel_to_hz(self, m: float) -> float:
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _build_mel_filterbank(self, n_fft, n_mels, sr, f_min, f_max):
        """Build triangular mel filterbank matrix."""
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mels = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz = torch.tensor([self._mel_to_hz(m.item()) for m in mels])
        bins = torch.floor((n_fft + 1) * hz / sr).long()

        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            low, center, high = bins[i], bins[i + 1], bins[i + 2]
            for j in range(low, center):
                if center > low:
                    fb[i, j] = (j - low).float() / (center - low).float()
            for j in range(center, high):
                if high > center:
                    fb[i, j] = (high - j).float() / (high - center).float()
        return fb

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Convert raw waveform to mel spectrogram.
        
        Args:
            waveform: (samples,) or (1, samples) tensor, 16kHz mono
            
        Returns:
            mel: (n_mels, time_frames) log-mel spectrogram
        """
        if waveform.dim() == 2:
            waveform = waveform.squeeze(0)

        # Pad if too short
        if waveform.shape[0] < self.n_fft:
            waveform = F.pad(waveform, (0, self.n_fft - waveform.shape[0]))

        # STFT
        window = self.window.to(waveform.device)
        spec = torch.stft(
            waveform, self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
            return_complex=True,
            center=True,
            pad_mode='reflect',
        )

        # Power spectrogram
        power = spec.abs().pow(2)  # (n_fft//2+1, time)

        # Apply mel filter bank
        mel_fb = self.mel_fb.to(power.device)
        mel = torch.matmul(mel_fb, power)  # (n_mels, time)

        # Log compression (like biological loudness perception)
        mel = torch.log(mel.clamp(min=1e-9))

        return mel


# =============================================================================
# A1 Encoder — Primary Auditory Cortex
# =============================================================================

class A1Encoder(nn.Module):
    """
    Primary Auditory Cortex encoder.
    
    Processes mel spectrograms through a stack of 1D convolutions
    to extract spectro-temporal patterns. Each layer captures
    progressively more complex audio features:
    
        Layer 1: Onset/offset detection (like A1 neurons)
        Layer 2: Frequency modulation patterns
        Layer 3: Spectro-temporal motifs (syllable-scale)
        Layer 4: Compression to latent dim
    """

    def __init__(self, n_mels: int = 80, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            # Layer 1: Broad spectral features (80 → 128)
            nn.Conv1d(n_mels, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.GELU(),

            # Layer 2: Temporal patterns (128 → 128, stride 2 for downsampling)
            nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            # Layer 3: Abstraction (128 → 64)
            nn.Conv1d(128, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.GELU(),

            # Layer 4: Latent projection (64 → latent_dim)
            nn.Conv1d(64, latent_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for all conv layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Encode mel spectrogram to latent representations.
        
        Args:
            mel: (batch, n_mels, time) or (n_mels, time)
            
        Returns:
            latent: (batch, latent_dim, time') where time' = time // 4
        """
        if mel.dim() == 2:
            mel = mel.unsqueeze(0)  # Add batch dim

        return self.encoder(mel)


# =============================================================================
# AuditoryCortex — Full Hearing Pipeline
# =============================================================================

class AuditoryCortex:
    """
    The complete auditory processing pipeline.
    
    Combines the cochlear filter (mel spectrogram) with the
    neural encoder (A1) to form the full hearing system.
    
    Input:  Raw audio waveform (16kHz mono)
    Output: Sequence of 64-dim latent vectors (one per ~40ms frame)
    """

    def __init__(self, sample_rate: int = 16000, n_mels: int = 80,
                 latent_dim: int = 64, lr: float = 0.001):
        self.sample_rate = sample_rate
        self.latent_dim = latent_dim
        self.mel_filter = MelFilterBank(
            sample_rate=sample_rate, n_mels=n_mels,
        )
        self.encoder = A1Encoder(n_mels=n_mels, latent_dim=latent_dim)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr)

        self._frames_processed = 0
        self._total_loss = 0.0

        logger.info(
            "Auditory Cortex initialized (%d params)",
            sum(p.numel() for p in self.encoder.parameters()),
        )

    def hear(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Process raw audio into neural representations.
        
        Args:
            waveform: numpy array of audio samples (16kHz mono)
            
        Returns:
            latent_sequence: (1, latent_dim, time_frames) tensor
        """
        self.encoder.eval()
        with torch.no_grad():
            wav_tensor = torch.tensor(waveform, dtype=torch.float32)
            mel = self.mel_filter(wav_tensor)  # (n_mels, time)
            latent = self.encoder(mel)  # (1, latent_dim, time')
        
        self._frames_processed += latent.shape[-1]
        return latent

    def train_contrastive(self, anchor_wav: np.ndarray,
                          positive_wav: np.ndarray,
                          negative_wav: np.ndarray) -> float:
        """
        Train the encoder via contrastive learning.
        
        Anchor + Positive should be similar sounds → close embeddings.
        Anchor + Negative should be different sounds → far embeddings.
        
        This is how the auditory cortex learns to discriminate
        between different sounds without any labels.
        """
        self.encoder.train()

        anchor = self._encode_for_training(anchor_wav)
        positive = self._encode_for_training(positive_wav)
        negative = self._encode_for_training(negative_wav)

        # Mean pool over time to get single vectors
        a = anchor.mean(dim=-1)  # (1, latent_dim)
        p = positive.mean(dim=-1)
        n = negative.mean(dim=-1)

        # Triplet margin loss
        loss = F.triplet_margin_loss(a, p, n, margin=1.0)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._total_loss += loss.item()
        return loss.item()

    def _encode_for_training(self, waveform: np.ndarray) -> torch.Tensor:
        """Encode waveform with gradients enabled for training."""
        wav_tensor = torch.tensor(waveform, dtype=torch.float32)
        mel = self.mel_filter(wav_tensor)
        return self.encoder(mel)

    def get_params(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters())

    def get_stats(self) -> dict:
        return {
            "params": self.get_params(),
            "frames_processed": self._frames_processed,
            "avg_loss": self._total_loss / max(1, self._frames_processed),
        }

    def save_weights(self, path):
        torch.save(self.encoder.state_dict(), path)

    def load_weights(self, path):
        try:
            self.encoder.load_state_dict(torch.load(path, weights_only=True))
            logger.info("Loaded auditory cortex weights from %s", path)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Could not load auditory cortex weights: %s", e)


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Auditory Cortex Test")
    print("=" * 60)

    cortex = AuditoryCortex(sample_rate=16000, n_mels=80, latent_dim=64)

    # Generate synthetic audio (1 second of a 440Hz sine wave)
    t = np.linspace(0, 1.0, 16000, dtype=np.float32)
    sine_wave = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Test hearing
    print("\n--- Hearing Test ---")
    latent = cortex.hear(sine_wave)
    print(f"  Input:  {len(sine_wave)} samples (1.0s @ 16kHz)")
    print(f"  Mel:    {cortex.mel_filter(torch.tensor(sine_wave)).shape}")
    print(f"  Output: {latent.shape} (batch, latent_dim, time_frames)")
    print(f"  Params: {cortex.get_params():,}")

    # Test contrastive training
    print("\n--- Contrastive Training Test ---")
    noise = np.random.randn(16000).astype(np.float32) * 0.1
    sine_similar = 0.5 * np.sin(2 * np.pi * 445 * t).astype(np.float32)

    for i in range(5):
        loss = cortex.train_contrastive(sine_wave, sine_similar, noise)
        print(f"  Epoch {i+1}: loss={loss:.4f}")

    print(f"\n  Stats: {cortex.get_stats()}")
    print("Auditory cortex test PASSED")
