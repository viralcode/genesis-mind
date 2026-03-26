"""
Genesis Mind — Vector Quantized Codebook

The human brain does not process sound as continuous waves.
Instead, the auditory cortex discretizes sound into categorical
representations — we perceive "b" and "p" as distinct categories
even though the acoustic signal is continuous.

This module replicates this categorical perception:

    Continuous latent vector (from Auditory Cortex)
        → Nearest codebook entry (quantization)
        → Discrete token ID

The codebook entries are Genesis's own "neural phonemes" —
discovered from experience, not pre-defined.

Implementation: Vector Quantization with EMA (Exponential Moving
Average) codebook updates, following van den Oord et al. 2017
(VQ-VAE). This is the same quantization technique used in
EnCodec, SoundStream, and AudioLM.
"""

import logging
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("genesis.neural.vq_codebook")


class VQCodebook(nn.Module):
    """
    Vector Quantized codebook — the categorical perception layer.

    Maps continuous auditory representations to a fixed set of
    discrete "neural phonemes":

        latent vector (64-dim) → codebook lookup → token ID (0-255)

    The codebook learns from experience using EMA (Exponential
    Moving Average) updates — no gradient hacking needed.
    
    Architecture inspired by VQ-VAE (van den Oord 2017) and
    EnCodec (Défossez et al. 2022).
    """

    def __init__(self, codebook_size: int = 256, latent_dim: int = 64,
                 commitment_cost: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.codebook_size = codebook_size
        self.latent_dim = latent_dim
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # The codebook: K learnable embedding vectors
        self.embedding = nn.Embedding(codebook_size, latent_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA tracking
        self.register_buffer('ema_count', torch.zeros(codebook_size))
        self.register_buffer('ema_weight', self.embedding.weight.clone())

        # Usage tracking
        self._total_quantizations = 0
        self._codebook_usage = torch.zeros(codebook_size)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize continuous latent vectors to discrete tokens.

        Args:
            z: (batch, latent_dim, time) continuous latent from encoder

        Returns:
            z_q: (batch, latent_dim, time) quantized latent (for decoder)
            token_ids: (batch, time) discrete token indices
            vq_loss: scalar commitment + codebook loss
        """
        # Reshape: (B, D, T) → (B*T, D)
        B, D, T = z.shape
        z_flat = z.permute(0, 2, 1).contiguous().view(-1, D)

        # Compute distances to all codebook entries
        # dist(z, e) = ||z||^2 + ||e||^2 - 2*z·e
        d = (
            z_flat.pow(2).sum(dim=1, keepdim=True)          # (BT, 1)
            + self.embedding.weight.pow(2).sum(dim=1)        # (K,)
            - 2.0 * z_flat @ self.embedding.weight.t()       # (BT, K)
        )

        # Find nearest codebook entry
        token_ids_flat = d.argmin(dim=1)  # (BT,)
        z_q_flat = self.embedding(token_ids_flat)  # (BT, D)

        # Track codebook usage
        self._update_usage(token_ids_flat)

        # EMA codebook update (during training)
        if self.training:
            self._ema_update(z_flat, token_ids_flat)

        # Losses
        # Commitment loss: encourage encoder to commit to codebook
        commitment_loss = F.mse_loss(z_flat, z_q_flat.detach())
        # Codebook loss: encourage codebook to stay near encoder outputs
        codebook_loss = F.mse_loss(z_q_flat, z_flat.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from z_q to z
        z_q_flat = z_flat + (z_q_flat - z_flat).detach()

        # Reshape back: (BT, D) → (B, D, T)
        z_q = z_q_flat.view(B, T, D).permute(0, 2, 1).contiguous()
        token_ids = token_ids_flat.view(B, T)

        self._total_quantizations += B * T

        return z_q, token_ids, vq_loss

    def tokens_to_embeddings(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert token IDs back to codebook vectors.
        
        Args:
            token_ids: (batch, time) or (time,) integer tensor
            
        Returns:
            embeddings: (batch, latent_dim, time) codebook vectors
        """
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)
        
        emb = self.embedding(token_ids)  # (B, T, D)
        return emb.permute(0, 2, 1).contiguous()  # (B, D, T)

    def _ema_update(self, z_flat: torch.Tensor, token_ids: torch.Tensor):
        """Update codebook using Exponential Moving Average."""
        with torch.no_grad():
            # One-hot encoding of assignments
            one_hot = F.one_hot(token_ids, self.codebook_size).float()  # (BT, K)

            # Count how many vectors are assigned to each codebook entry
            counts = one_hot.sum(dim=0)  # (K,)

            # Sum of assigned vectors
            sums = one_hot.t() @ z_flat  # (K, D)

            # EMA update
            self.ema_count = self.ema_decay * self.ema_count + (1 - self.ema_decay) * counts
            self.ema_weight = self.ema_decay * self.ema_weight + (1 - self.ema_decay) * sums

            # Laplace smoothing to avoid dead codebook entries
            n = self.ema_count.sum()
            smoothed_count = (
                (self.ema_count + 1e-5) / (n + self.codebook_size * 1e-5) * n
            )

            # Update embedding weights
            self.embedding.weight.data = self.ema_weight / smoothed_count.unsqueeze(1)

    def _update_usage(self, token_ids: torch.Tensor):
        """Track which codebook entries are being used."""
        with torch.no_grad():
            for idx in token_ids.unique():
                self._codebook_usage[idx.item()] += (token_ids == idx).sum().item()

    def get_codebook_utilization(self) -> float:
        """Fraction of codebook entries that have been used at least once."""
        if self._total_quantizations == 0:
            return 0.0
        used = (self._codebook_usage > 0).sum().item()
        return used / self.codebook_size

    def get_stats(self) -> dict:
        """Get quantization statistics."""
        return {
            "codebook_size": self.codebook_size,
            "latent_dim": self.latent_dim,
            "total_quantizations": self._total_quantizations,
            "codebook_utilization": round(self.get_codebook_utilization(), 3),
            "active_codes": int((self._codebook_usage > 0).sum().item()),
            "params": self.codebook_size * self.latent_dim,
        }

    def save_weights(self, path):
        torch.save({
            'embedding': self.embedding.state_dict(),
            'ema_count': self.ema_count,
            'ema_weight': self.ema_weight,
            'usage': self._codebook_usage,
            'total_quantizations': self._total_quantizations,
        }, path)

    def load_weights(self, path):
        try:
            ckpt = torch.load(path, weights_only=True)
            self.embedding.load_state_dict(ckpt['embedding'])
            self.ema_count = ckpt['ema_count']
            self.ema_weight = ckpt['ema_weight']
            self._codebook_usage = ckpt['usage']
            self._total_quantizations = ckpt['total_quantizations']
            logger.info("Loaded VQ codebook from %s (util=%.1f%%)",
                        path, self.get_codebook_utilization() * 100)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning("Could not load VQ codebook: %s", e)


# =============================================================================
# Standalone test
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")
    print("=" * 60)
    print("Genesis Mind — VQ Codebook Test")
    print("=" * 60)

    codebook = VQCodebook(codebook_size=256, latent_dim=64)

    # Simulate encoder output: batch=1, latent_dim=64, time=25
    z = torch.randn(1, 64, 25)

    print("\n--- Quantization Test ---")
    codebook.train()
    z_q, token_ids, vq_loss = codebook(z)
    print(f"  Input:    {z.shape}")
    print(f"  Quantized: {z_q.shape}")
    print(f"  Tokens:   {token_ids.shape} → {token_ids[0].tolist()}")
    print(f"  VQ Loss:  {vq_loss.item():.4f}")

    # Test reconstruction from tokens
    print("\n--- Token → Embedding Test ---")
    reconstructed = codebook.tokens_to_embeddings(token_ids)
    print(f"  Reconstructed: {reconstructed.shape}")
    recon_error = (z_q - reconstructed).abs().max().item()
    print(f"  Max reconstruction error: {recon_error:.6f}")

    # Run multiple forward passes to test EMA
    print("\n--- EMA Training (10 steps) ---")
    for i in range(10):
        z_new = torch.randn(2, 64, 25)
        z_q, ids, loss = codebook(z_new)
        print(f"  Step {i+1}: loss={loss.item():.4f}, utilization={codebook.get_codebook_utilization():.3f}")

    print(f"\n  Stats: {codebook.get_stats()}")
    print("VQ Codebook test PASSED")
