"""
Genesis Mind — Visual Cortex (Layer 1: Subconscious)

A convolutional autoencoder that learns to SEE from scratch.

Unlike CLIP (which was pre-trained on 400M image-text pairs),
this network starts with random weights and learns to compress
raw camera frames through reconstruction. Over time it develops
its own internal visual representations — unique to THIS Genesis
instance based on what IT has seen.

Architecture:
    Encoder: Conv2d(1,16,3) → Conv2d(16,32,3) → FC(32*14*14, 64)
    Decoder: FC(64, 32*14*14) → ConvT(32,16,3) → ConvT(16,1,3)

    Input:  64×64 grayscale frame
    Latent: 64-dimensional vector
    Output: 64×64 reconstructed frame

    Params: ~64K (trains on CPU in <10ms per step)

Training:
    Loss = MSE(input, reconstruction)
    Optimizer = Adam(lr=0.001)
    Trained on every single frame the camera captures.
    The weights literally encode "how Genesis sees the world."
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("genesis.neural.visual_cortex")


class VisualEncoder(nn.Module):
    """Compresses a 64×64 grayscale image into a 64-dim latent vector."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        self.latent_dim = latent_dim

        # Conv layers: 1×64×64 → 16×30×30 → 32×13×13
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # Flatten 32×13×13 = 5408 → 64
        self.fc = nn.Linear(32 * 13 * 13, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))   # (B, 16, 30, 30)
        x = self.relu(self.conv2(x))   # (B, 32, 13, 13)
        x = x.view(x.size(0), -1)      # (B, 5408)
        x = self.fc(x)                 # (B, 64)
        return x


class VisualDecoder(nn.Module):
    """Reconstructs a 64×64 grayscale image from a 64-dim latent vector."""

    def __init__(self, latent_dim: int = 64):
        super().__init__()

        self.fc = nn.Linear(latent_dim, 32 * 13 * 13)
        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(16, 1, kernel_size=5, stride=2, padding=1, output_padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc(z))                # (B, 5408)
        x = x.view(x.size(0), 32, 13, 13)       # (B, 32, 13, 13)
        x = self.relu(self.deconv1(x))           # (B, 16, ~27, ~27)
        x = torch.sigmoid(self.deconv2(x))       # (B, 1, ~55, ~55)
        # Interpolate to exact 64×64
        x = nn.functional.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return x


class VisualCortex(nn.Module):
    """
    The complete visual autoencoder — Genesis's learned eyes.

    Trains on every frame. The encoder's output IS Genesis's
    visual perception. The decoder ensures the latent space
    contains meaningful information (reconstruction objective).
    """

    def __init__(self, latent_dim: int = 64, lr: float = 0.001):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = VisualEncoder(latent_dim)
        self.decoder = VisualDecoder(latent_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self._frames_seen = 0
        self._total_loss = 0.0

        # Count params
        total = sum(p.numel() for p in self.parameters())
        logger.info("Visual cortex initialized (%d parameters, latent_dim=%d)", total, latent_dim)

    def encode(self, frame: np.ndarray) -> np.ndarray:
        """
        Perceive a frame — compress it into Genesis's visual latent space.

        Args:
            frame: (H, W) or (H, W, 3) numpy array, uint8 or float32

        Returns:
            64-dim numpy array — Genesis's visual perception
        """
        tensor = self._preprocess(frame)
        with torch.no_grad():
            latent = self.encoder(tensor)
        return latent.squeeze(0).numpy()

    def train_on_frame(self, frame: np.ndarray) -> float:
        """
        See a frame AND learn from it.

        This is the core training loop — every frame physically
        reshapes the weights of Genesis's visual cortex.

        Returns the reconstruction loss.
        """
        tensor = self._preprocess(frame)

        # Forward pass
        latent = self.encoder(tensor)
        reconstruction = self.decoder(latent)
        loss = self.criterion(reconstruction, tensor)

        # Backward pass — weights change
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._frames_seen += 1
        self._total_loss += loss.item()

        if self._frames_seen % 100 == 0:
            avg_loss = self._total_loss / self._frames_seen
            logger.info("Visual cortex: %d frames seen, avg loss: %.4f",
                        self._frames_seen, avg_loss)

        return loss.item()

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """Convert a raw frame to a normalized 1×1×64×64 tensor."""
        if frame.ndim == 3:
            # Convert to grayscale
            gray = np.mean(frame, axis=2)
        else:
            gray = frame.copy()

        # Resize to 64×64
        from PIL import Image
        img = Image.fromarray(gray.astype(np.uint8))
        img = img.resize((64, 64), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 64, 64)
        return tensor

    def save_weights(self, path: Path):
        """Save the visual cortex — save this mind's way of seeing."""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'frames_seen': self._frames_seen,
            'total_loss': self._total_loss,
        }, path)
        logger.info("Visual cortex saved (%d frames encoded)", self._frames_seen)

    def load_weights(self, path: Path):
        """Restore the visual cortex — remember how to see."""
        if path.exists():
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.load_state_dict(checkpoint['state_dict'])
            self._frames_seen = checkpoint.get('frames_seen', 0)
            self._total_loss = checkpoint.get('total_loss', 0.0)
            logger.info("Visual cortex loaded (%d frames previously seen)", self._frames_seen)

    def get_stats(self) -> dict:
        return {
            "frames_seen": self._frames_seen,
            "avg_loss": self._total_loss / max(1, self._frames_seen),
            "params": sum(p.numel() for p in self.parameters()),
            "latent_dim": self.latent_dim,
        }
