"""
Genesis Mind — Visual Cortex (From-Scratch Vision)

Replaces CLIP (150M+ pretrained params) with a tiny convolutional
autoencoder that learns visual representations from raw pixels.

Architecture:
    Encoder: 3 conv layers (3→16→32→64 channels) + flatten → 64-dim
    Decoder: Mirror (for self-supervised reconstruction loss)

Training: Every frame Genesis sees is used for reconstruction learning.
The encoder learns to compress visual input into a 64-dim latent vector.
Initially random — Genesis is literally blind. Gradually learns to
distinguish shapes, colors, and objects through experience.

Total params: ~50K (vs CLIP's 150M). All learned from scratch.
"""

import logging
import json
from pathlib import Path

from genesis.neural.device import DEVICE, to_device, try_compile, get_autocast_context
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("genesis.neural.visual_cortex")


class VisualEncoder(nn.Module):
    """
    Convolutional encoder: raw pixels → 64-dim latent vector.
    
    3 conv layers with increasing channels, BatchNorm, and ReLU.
    Final flatten + linear projects to 64-dim embedding.
    """
    
    def __init__(self, latent_dim: int = 64, input_size: int = 64):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_size = input_size
        
        # Conv layers: 3→16→32→64 channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)  # 64→32
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)  # 32→16
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)  # 16→8
        self.bn3 = nn.BatchNorm2d(64)
        
        # Flatten → linear → 64-dim latent
        self._flat_size = 64 * (input_size // 8) * (input_size // 8)
        self.fc = nn.Linear(self._flat_size, latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image → 64-dim vector."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)


class VisualDecoder(nn.Module):
    """
    Transposed convolutional decoder: 64-dim → reconstructed image.
    
    Mirror of the encoder. Used for self-supervised reconstruction loss.
    """
    
    def __init__(self, latent_dim: int = 64, input_size: int = 64):
        super().__init__()
        self.input_size = input_size
        self._spatial = input_size // 8  # 8 for 64x64
        self._flat_size = 64 * self._spatial * self._spatial
        
        self.fc = nn.Linear(latent_dim, self._flat_size)
        
        # Transposed conv layers: 64→32→16→3
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.deconv2 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.deconv3 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode 64-dim vector → reconstructed image."""
        x = self.fc(z)
        x = x.view(x.size(0), 64, self._spatial, self._spatial)
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = torch.sigmoid(self.deconv3(x))  # Output [0, 1] pixel values
        return x


class VisualCortex:
    """
    Genesis's visual system — learns to see from raw pixel experience.
    
    Unlike CLIP (which comes pre-trained on billions of image-text pairs),
    this starts with random weights. Every frame Genesis sees is used to
    train the autoencoder via reconstruction loss. Over time, the encoder
    learns meaningful visual features.
    
    The 64-dim latent vector is Genesis's internal visual representation.
    Two similar-looking objects will have similar latent vectors — but
    only after Genesis has seen enough to learn.
    """
    
    def __init__(self, latent_dim: int = 64, input_size: int = 64,
                 learning_rate: float = 1e-3, 
                 storage_path: Optional[Path] = None):
        self.latent_dim = latent_dim
        self.input_size = input_size
        self._storage_path = storage_path
        
        self.encoder = to_device(VisualEncoder(latent_dim, input_size))
        self.decoder = to_device(VisualDecoder(latent_dim, input_size))
        
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate,
        )
        
        self._train_steps = 0
        self._total_loss = 0.0
        self._frames_seen = 0
        
        # ═══ FRAME ACCUMULATION BUFFER ═══
        # Accumulate 4 frames before batched training for stabler gradients
        self._frame_buffer = []
        self._frame_batch_size = 4
        
        # Load saved weights BEFORE torch.compile() — compile wraps keys
        self._load()
        
        # torch.compile() for free speedup (must be after weight loading)
        self.encoder = try_compile(self.encoder, 'VisualEncoder')
        self.decoder = try_compile(self.decoder, 'VisualDecoder')
        
        total_params = sum(
            p.numel() for p in list(self.encoder.parameters()) + list(self.decoder.parameters())
        )
        logger.info(
            "Visual cortex initialized (latent=%d, params=%d, trained=%d steps, batch=%d)",
            latent_dim, total_params, self._train_steps, self._frame_batch_size,
        )
    
    def see(self, image: np.ndarray, train: bool = True) -> np.ndarray:
        """
        Process a visual frame — encode it and optionally learn from it.
        
        When training, frames are accumulated into a batch of 4 before
        a single batched backward pass. This gives stabler gradients
        and better GPU utilization than single-frame updates.
        
        Returns:
            64-dim visual embedding as numpy array
        """
        tensor = self._preprocess(image)
        
        if not train:
            self.encoder.eval()
            with torch.no_grad():
                embedding = self.encoder(tensor)
            return embedding.squeeze(0).cpu().numpy()
        
        # Always return embedding immediately (encode is cheap)
        self.encoder.eval()
        with torch.no_grad():
            embedding = self.encoder(tensor)
        result = embedding.squeeze(0).cpu().numpy()
        
        # Accumulate frame for batched training
        self._frame_buffer.append(tensor)
        if len(self._frame_buffer) >= self._frame_batch_size:
            self._train_batch()
        
        return result
    
    def _train_batch(self):
        """Train on accumulated frame batch — AMP enabled."""
        if not self._frame_buffer:
            return
        
        self.encoder.train()
        self.decoder.train()
        
        # Stack frames into a batch
        batch = torch.cat(self._frame_buffer, dim=0)  # (N, 3, H, W)
        self._frame_buffer.clear()
        
        # Forward pass with AMP autocast
        with get_autocast_context():
            embeddings = self.encoder(batch)
            reconstructions = self.decoder(embeddings)
            loss = F.mse_loss(reconstructions, batch)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            max_norm=1.0,
        )
        self.optimizer.step()
        
        batch_size = batch.shape[0]
        self._train_steps += batch_size
        self._total_loss += loss.item() * batch_size
        self._frames_seen += batch_size
        
        # Periodic logging
        if self._train_steps % 100 < batch_size:
            avg_loss = self._total_loss / max(1, self._train_steps)
            logger.info(
                "Visual cortex: %d frames, avg_loss=%.4f (batch=%d, AMP)",
                self._train_steps, avg_loss, batch_size,
            )
            self._save()
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert raw image → model input tensor.
        
        Resizes to input_size × input_size, normalizes to [0, 1],
        converts HWC → CHW, adds batch dim.
        """
        from PIL import Image
        
        if isinstance(image, np.ndarray):
            pil = Image.fromarray(image.astype(np.uint8))
        else:
            pil = image
        
        # Resize to target size
        pil = pil.resize((self.input_size, self.input_size), Image.BILINEAR)
        
        # Convert to float tensor [0, 1], CHW format
        arr = np.array(pil, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)  # Grayscale → RGB
        arr = arr[:, :, :3]  # Ensure 3 channels
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
        
        return tensor
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two visual embeddings."""
        e1 = np.array(emb1, dtype=np.float32).flatten()
        e2 = np.array(emb2, dtype=np.float32).flatten()
        n1, n2 = np.linalg.norm(e1), np.linalg.norm(e2)
        if n1 > 0 and n2 > 0:
            return float(np.dot(e1, e2) / (n1 * n2))
        return 0.0
    
    def get_stats(self) -> dict:
        return {
            "train_steps": self._train_steps,
            "frames_seen": self._frames_seen,
            "avg_loss": round(self._total_loss / max(1, self._train_steps), 6),
            "latent_dim": self.latent_dim,
            "total_params": sum(
                p.numel() for p in list(self.encoder.parameters()) + list(self.decoder.parameters())
            ),
        }
    
    def _save(self):
        """Save encoder/decoder weights to disk."""
        if self._storage_path is None:
            return
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "train_steps": self._train_steps,
            "total_loss": self._total_loss,
            "frames_seen": self._frames_seen,
        }, self._storage_path)
    
    def _load(self):
        """Load saved weights from disk."""
        if self._storage_path is None or not self._storage_path.exists():
            return
        try:
            checkpoint = torch.load(self._storage_path, map_location="cpu", weights_only=False)
            self.encoder.load_state_dict(checkpoint["encoder"])
            self.decoder.load_state_dict(checkpoint["decoder"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self._train_steps = checkpoint.get("train_steps", 0)
            self._total_loss = checkpoint.get("total_loss", 0.0)
            self._frames_seen = checkpoint.get("frames_seen", 0)
            logger.info("Visual cortex loaded (%d training steps)", self._train_steps)
        except Exception as e:
            logger.error("Failed to load visual cortex: %s", e)
    
    def save(self):
        """Public save method for shutdown."""
        self._save()
    
    def __repr__(self) -> str:
        return (
            f"VisualCortex(latent={self.latent_dim}, "
            f"steps={self._train_steps}, frames={self._frames_seen})"
        )
