"""
Genesis Mind — Neural Vocoder

The vocal tract of Genesis. Converts discrete acoustic tokens back
into audible waveforms — like how the motor cortex drives the
larynx, tongue, and lips to produce speech.

Pipeline:
    Token IDs → VQ Codebook vectors → Learned upsampler → Mel frames
    → Griffin-Lim algorithm → Raw waveform → Speaker output

Griffin-Lim is a classical (non-learned) algorithm that reconstructs
waveforms from magnitude spectrograms. It works by iteratively
refining the phase estimate. No training required — Genesis can
produce sound from day one.

The learned upsampler converts the compressed VQ representations
(~25 frames/sec) back to mel-spectrogram resolution (~100 frames/sec).
"""

import logging
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("genesis.neural.neural_vocoder")


class MelReconstructor(nn.Module):
    """
    Learned upsampler: VQ embeddings → Mel spectrogram frames.
    
    Takes the compressed codebook vectors and projects them back
    into mel-spectrogram space, upsampling temporally.
    """

    def __init__(self, latent_dim: int = 64, n_mels: int = 80,
                 upsample_factor: int = 4):
        super().__init__()
        self.upsample_factor = upsample_factor

        self.decoder = nn.Sequential(
            # Project from latent to mel space
            nn.ConvTranspose1d(latent_dim, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, n_mels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct mel spectrogram from codebook vectors.
        
        Args:
            z: (batch, latent_dim, time) codebook embeddings
            
        Returns:
            mel: (batch, n_mels, time * upsample_factor) reconstructed mel
        """
        return self.decoder(z)


class GriffinLim:
    """
    Griffin-Lim algorithm — phase reconstruction from magnitude spectrogram.
    
    This is a classical signal processing algorithm (no neural network).
    Given a magnitude spectrogram |STFT|, it estimates the phase
    through iterative STFT → ISTFT → STFT cycles.
    
    The result is an audible waveform. Quality is decent (robotic
    but intelligible) — good enough for a developing AI that is
    learning to speak.
    """

    def __init__(self, n_fft: int = 512, hop_length: int = 160,
                 n_iters: int = 32, sample_rate: int = 16000,
                 n_mels: int = 80):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_iters = n_iters
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.window = torch.hann_window(n_fft)

        # Build inverse mel filter matrix for mel → linear spectrogram
        self.inv_mel_fb = self._build_inv_mel_filterbank(
            n_fft, n_mels, sample_rate
        )

    def _hz_to_mel(self, f):
        return 2595.0 * math.log10(1.0 + f / 700.0)

    def _mel_to_hz(self, m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    def _build_inv_mel_filterbank(self, n_fft, n_mels, sr):
        """Build pseudo-inverse of mel filterbank for mel→linear conversion."""
        mel_min = self._hz_to_mel(0.0)
        mel_max = self._hz_to_mel(sr / 2)
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

        # Pseudo-inverse: (n_fft//2+1, n_mels)
        return torch.pinverse(fb)

    def synthesize(self, mel: torch.Tensor) -> np.ndarray:
        """
        Convert log-mel spectrogram to waveform using Griffin-Lim.
        
        Args:
            mel: (n_mels, time) or (1, n_mels, time) log-mel spectrogram
            
        Returns:
            waveform: numpy array of audio samples
        """
        if mel.dim() == 3:
            mel = mel.squeeze(0)

        # Exp to undo log compression
        mel_linear = torch.exp(mel)

        # Mel → linear spectrogram via pseudo-inverse
        inv_fb = self.inv_mel_fb.to(mel.device)
        magnitude = torch.clamp(inv_fb @ mel_linear, min=0.0)  # (n_fft//2+1, time)

        # Griffin-Lim: iterative phase estimation
        window = self.window.to(mel.device)
        angles = torch.randn_like(magnitude) * 2 * math.pi
        complex_spec = magnitude * torch.exp(1j * angles)

        for _ in range(self.n_iters):
            # ISTFT
            waveform = torch.istft(
                complex_spec, self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
            )
            # STFT to get new phase estimate
            new_spec = torch.stft(
                waveform, self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                return_complex=True,
                center=True,
                pad_mode='reflect',
            )
            # Keep original magnitude, use new phase
            angles = new_spec.angle()
            complex_spec = magnitude * torch.exp(1j * angles)

        # Final ISTFT
        waveform = torch.istft(
            complex_spec, self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=window,
        )

        # Normalize
        waveform = waveform / (waveform.abs().max() + 1e-8)

        return waveform.detach().cpu().numpy()


class NeuralVocoder:
    """
    The complete speech synthesis pipeline.
    
    Token IDs → Codebook embeddings → Mel reconstruction → Griffin-Lim → Audio
    
    This is Genesis's "vocal tract" — it converts neural representations
    into physical sound waves.
    """

    def __init__(self, latent_dim: int = 64, n_mels: int = 80,
                 sample_rate: int = 16000, lr: float = 0.001):
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.sample_rate = sample_rate

        self.mel_reconstructor = MelReconstructor(
            latent_dim=latent_dim, n_mels=n_mels,
        )
        self.griffin_lim = GriffinLim(
            n_mels=n_mels, sample_rate=sample_rate,
        )
        self.optimizer = torch.optim.Adam(
            self.mel_reconstructor.parameters(), lr=lr,
        )

        self._total_syntheses = 0

        logger.info(
            "Neural Vocoder initialized (%d params + Griffin-Lim)",
            sum(p.numel() for p in self.mel_reconstructor.parameters()),
        )

    def synthesize_from_embeddings(self, embeddings: torch.Tensor) -> np.ndarray:
        """
        Convert codebook embeddings to audio waveform.
        
        Args:
            embeddings: (1, latent_dim, time) tensor from VQ codebook
            
        Returns:
            waveform: numpy array of 16kHz audio
        """
        self.mel_reconstructor.eval()
        with torch.no_grad():
            mel = self.mel_reconstructor(embeddings)  # (1, n_mels, time*4)

        waveform = self.griffin_lim.synthesize(mel)
        self._total_syntheses += 1
        return waveform

    def train_reconstruction(self, embeddings: torch.Tensor,
                             target_mel: torch.Tensor) -> float:
        """
        Train the mel reconstructor to match original mel spectrograms.
        
        This is called during the self-monitoring loop: Genesis speaks,
        re-hears itself, and trains the vocoder to better reconstruct
        the original mel spectrogram.
        """
        self.mel_reconstructor.train()
        predicted_mel = self.mel_reconstructor(embeddings)

        # Match sizes (target might be different length)
        T = min(predicted_mel.shape[-1], target_mel.shape[-1])
        loss = F.mse_loss(predicted_mel[:, :, :T], target_mel[:, :, :T])

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def play(self, waveform: np.ndarray):
        """
        Play waveform through speakers.
        
        Amplifies the output to ensure audibility (neural vocoder output
        is often very quiet during early training).
        """
        try:
            import sounddevice as sd
            # Stop any currently-playing stream first (prevents AUHAL -50 error)
            try:
                sd.stop()
            except Exception:
                pass
            
            # Amplify — untrained vocoder produces very quiet output
            waveform = np.ascontiguousarray(waveform, dtype=np.float32)
            peak = np.abs(waveform).max()
            if peak > 1e-6:
                # Normalize to 80% of max volume
                waveform = waveform * (0.8 / peak)
            else:
                logger.debug("Waveform is silent (peak=%.2e), skipping playback", peak)
                return
            
            # Clamp to [-1, 1] to avoid audio driver errors
            waveform = np.clip(waveform, -1.0, 1.0)
            
            logger.debug(
                "Playing %d samples (%.2fs, peak=%.3f)",
                len(waveform), len(waveform) / self.sample_rate, np.abs(waveform).max()
            )
            sd.play(waveform, self.sample_rate, blocking=False)
        except ImportError:
            logger.warning("sounddevice not available — cannot play audio")
        except Exception as e:
            logger.debug("Audio playback issue: %s", e)

    def get_params(self) -> int:
        return sum(p.numel() for p in self.mel_reconstructor.parameters())

    def get_stats(self) -> dict:
        return {
            "params": self.get_params(),
            "total_syntheses": self._total_syntheses,
            "sample_rate": self.sample_rate,
        }

    def save_weights(self, path):
        torch.save(self.mel_reconstructor.state_dict(), path)

    def load_weights(self, path):
        try:
            self.mel_reconstructor.load_state_dict(
                torch.load(path, weights_only=True)
            )
            logger.info("Loaded vocoder weights from %s", path)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Could not load vocoder weights: %s", e)


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — Neural Vocoder Test")
    print("=" * 60)

    vocoder = NeuralVocoder(latent_dim=64, n_mels=80, sample_rate=16000)

    # Simulate VQ embeddings (as if from codebook)
    embeddings = torch.randn(1, 64, 25)  # ~1s of encoded audio

    print("\n--- Synthesis Test ---")
    waveform = vocoder.synthesize_from_embeddings(embeddings)
    print(f"  Input embeddings: {embeddings.shape}")
    print(f"  Output waveform:  {waveform.shape} ({len(waveform)/16000:.2f}s)")
    print(f"  Waveform range:   [{waveform.min():.3f}, {waveform.max():.3f}]")
    print(f"  Params: {vocoder.get_params():,}")

    # Test play (will fail gracefully if no audio device)
    print("\n--- Playback Test ---")
    vocoder.play(waveform)
    print("  (attempting playback...)")

    print(f"\n  Stats: {vocoder.get_stats()}")
    print("Neural Vocoder test PASSED")
