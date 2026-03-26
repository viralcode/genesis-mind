"""
Genesis Mind — Auditory Cortex (Layer 1: Subconscious)

A 1D convolutional autoencoder that learns to HEAR from scratch.

Unlike Whisper (pre-trained on 680K hours of audio), this network
starts with random weights and learns to compress raw audio
spectrograms through reconstruction. It develops its own internal
auditory representations based solely on what THIS Genesis has heard.

Architecture:
    Encoder: Conv1d(64,32,5) → Conv1d(32,16,5) → FC(adaptive, 32)
    Decoder: FC(32, adaptive) → ConvT(16,32,5) → ConvT(32,64,5)

    Input:  Mel spectrogram (64 mel bands × T time frames)
    Latent: 32-dimensional vector
    Output: Reconstructed spectrogram

    Params: ~32K (trains on CPU in <5ms per step)

Training:
    Loss = MSE(input_spectrogram, reconstruction)
    Trained on every audio chunk heard through the microphone.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("genesis.neural.auditory_cortex")


class AuditoryEncoder(nn.Module):
    """Compresses a mel spectrogram into a 32-dim latent vector."""

    def __init__(self, n_mels: int = 64, latent_dim: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mels = n_mels

        self.conv1 = nn.Conv1d(n_mels, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv1d(32, 16, kernel_size=5, stride=2, padding=2)
        self.relu = nn.ReLU(inplace=True)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(8)
        self.fc = nn.Linear(16 * 8, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, n_mels, T)
        x = self.relu(self.conv1(x))         # (B, 32, T/2)
        x = self.relu(self.conv2(x))         # (B, 16, T/4)
        x = self.adaptive_pool(x)            # (B, 16, 8)
        x = x.view(x.size(0), -1)           # (B, 128)
        x = self.fc(x)                       # (B, 32)
        return x


class AuditoryDecoder(nn.Module):
    """Reconstructs a mel spectrogram from a 32-dim latent vector."""

    def __init__(self, n_mels: int = 64, latent_dim: int = 32, output_frames: int = 32):
        super().__init__()
        self.output_frames = output_frames

        self.fc = nn.Linear(latent_dim, 16 * 8)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose1d(16, 32, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.deconv2 = nn.ConvTranspose1d(32, n_mels, kernel_size=5, stride=2, padding=2, output_padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc(z))            # (B, 128)
        x = x.view(x.size(0), 16, 8)        # (B, 16, 8)
        x = self.relu(self.deconv1(x))       # (B, 32, 16)
        x = torch.sigmoid(self.deconv2(x))   # (B, 64, 32)
        # Interpolate to match output_frames
        x = nn.functional.interpolate(x, size=self.output_frames, mode='linear', align_corners=False)
        return x


class AuditoryCortex(nn.Module):
    """
    The complete auditory autoencoder — Genesis's learned ears.

    Trains on every audio chunk. The encoder's output IS Genesis's
    auditory perception. The decoder ensures the latent space
    captures meaningful acoustic structure.
    """

    def __init__(self, n_mels: int = 64, latent_dim: int = 32,
                 lr: float = 0.001, target_frames: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.n_mels = n_mels
        self.target_frames = target_frames

        self.encoder = AuditoryEncoder(n_mels, latent_dim)
        self.decoder = AuditoryDecoder(n_mels, latent_dim, target_frames)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self._chunks_heard = 0
        self._total_loss = 0.0

        total = sum(p.numel() for p in self.parameters())
        logger.info("Auditory cortex initialized (%d parameters, latent_dim=%d)", total, latent_dim)

    def compute_mel_spectrogram(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        Convert raw audio to a mel spectrogram.

        This is a simple, dependency-free implementation using
        FFT + mel filterbank. No librosa required.
        """
        # Short-time Fourier transform
        n_fft = 1024
        hop_length = 512
        n_frames = 1 + (len(audio) - n_fft) // hop_length

        if n_frames < 1:
            # Pad short audio
            audio = np.pad(audio, (0, n_fft - len(audio)))
            n_frames = 1

        # Windowed FFT
        window = np.hanning(n_fft)
        stft = np.zeros((n_fft // 2 + 1, n_frames))
        for i in range(n_frames):
            start = i * hop_length
            frame = audio[start:start + n_fft] * window
            spectrum = np.abs(np.fft.rfft(frame))
            stft[:, i] = spectrum

        # Mel filterbank
        mel_filters = self._mel_filterbank(sr, n_fft, self.n_mels)
        mel_spec = np.dot(mel_filters, stft)

        # Log scale
        mel_spec = np.log1p(mel_spec)

        # Normalize to [0, 1]
        max_val = mel_spec.max()
        if max_val > 0:
            mel_spec = mel_spec / max_val

        return mel_spec

    def _mel_filterbank(self, sr: int, n_fft: int, n_mels: int) -> np.ndarray:
        """Create a mel filterbank matrix."""
        low_freq = 0
        high_freq = sr / 2

        # Mel scale conversion
        def hz_to_mel(f):
            return 2595 * np.log10(1 + f / 700)

        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        mel_low = hz_to_mel(low_freq)
        mel_high = hz_to_mel(high_freq)
        mel_points = np.linspace(mel_low, mel_high, n_mels + 2)
        hz_points = mel_to_hz(mel_points)

        bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
        n_bins = n_fft // 2 + 1

        filters = np.zeros((n_mels, n_bins))
        for i in range(n_mels):
            for j in range(bin_points[i], min(bin_points[i + 1], n_bins)):
                if bin_points[i + 1] > bin_points[i]:
                    filters[i, j] = (j - bin_points[i]) / (bin_points[i + 1] - bin_points[i])
            for j in range(bin_points[i + 1], min(bin_points[i + 2], n_bins)):
                if bin_points[i + 2] > bin_points[i + 1]:
                    filters[i, j] = (bin_points[i + 2] - j) / (bin_points[i + 2] - bin_points[i + 1])

        return filters

    def encode(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Perceive audio — compress it into Genesis's auditory latent space."""
        mel = self.compute_mel_spectrogram(audio, sr)
        tensor = self._preprocess_mel(mel)
        with torch.no_grad():
            latent = self.encoder(tensor)
        return latent.squeeze(0).numpy()

    def train_on_audio(self, audio: np.ndarray, sr: int = 16000) -> float:
        """
        Hear audio AND learn from it.

        Every audio chunk physically reshapes the auditory cortex weights.
        """
        mel = self.compute_mel_spectrogram(audio, sr)
        tensor = self._preprocess_mel(mel)

        latent = self.encoder(tensor)
        reconstruction = self.decoder(latent)

        # Match the time dimension
        target = nn.functional.interpolate(tensor, size=self.target_frames, mode='linear', align_corners=False)
        loss = self.criterion(reconstruction, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._chunks_heard += 1
        self._total_loss += loss.item()

        if self._chunks_heard % 50 == 0:
            avg = self._total_loss / self._chunks_heard
            logger.info("Auditory cortex: %d chunks heard, avg loss: %.4f", self._chunks_heard, avg)

        return loss.item()

    def _preprocess_mel(self, mel: np.ndarray) -> torch.Tensor:
        """Convert mel spectrogram to tensor (1, n_mels, T)."""
        tensor = torch.from_numpy(mel.astype(np.float32)).unsqueeze(0)  # (1, n_mels, T)
        return tensor

    def save_weights(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'chunks_heard': self._chunks_heard,
            'total_loss': self._total_loss,
        }, path)
        logger.info("Auditory cortex saved (%d chunks)", self._chunks_heard)

    def load_weights(self, path: Path):
        if path.exists():
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.load_state_dict(checkpoint['state_dict'])
            self._chunks_heard = checkpoint.get('chunks_heard', 0)
            self._total_loss = checkpoint.get('total_loss', 0.0)
            logger.info("Auditory cortex loaded (%d chunks previously heard)", self._chunks_heard)

    def get_stats(self) -> dict:
        return {
            "chunks_heard": self._chunks_heard,
            "avg_loss": self._total_loss / max(1, self._chunks_heard),
            "params": sum(p.numel() for p in self.parameters()),
            "latent_dim": self.latent_dim,
        }
