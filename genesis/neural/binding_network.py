"""
Genesis Mind — Binding Network (Layer 2: Associative Bridge)

In the human brain, the superior temporal sulcus and angular gyrus
bind visual and auditory signals into unified percepts. When you
see a dog AND hear "dog" at the same time, these regions fuse
them into a single cross-modal concept.

This module is a small MLP that learns to create unified concept
embeddings from separate visual and auditory latent vectors.

    Input:  visual_latent (64-dim) ⊕ auditory_latent (32-dim) = 96-dim
    Output: unified concept embedding (64-dim)

The key insight: this network learns which visual features
correlate with which auditory features. Over time, it discovers
that the visual pattern of "round red object" co-occurs with
the auditory pattern of the word "apple" — and binds them.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("genesis.neural.binding_network")


class CrossModalBinder(nn.Module):
    """
    Fuses visual and auditory latent vectors into a unified concept.

    Architecture: 3-layer MLP with residual connection.

    The residual connection ensures that early in training (when
    the weights are random), the output still preserves input
    information. As training progresses, the learned transformation
    increasingly dominates.
    """

    def __init__(self, visual_dim: int = 64, auditory_dim: int = 32,
                 output_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        input_dim = visual_dim + auditory_dim

        self.transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Residual projection (input_dim → output_dim)
        self.residual = nn.Linear(input_dim, output_dim)

        # Learnable gate: how much to trust the transform vs raw input
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, visual: torch.Tensor, auditory: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([visual, auditory], dim=-1)

        transformed = self.transform(combined)
        residual = self.residual(combined)
        gate = self.gate(combined)

        # Gated fusion: early training → mostly residual; late → mostly transformed
        output = gate * transformed + (1 - gate) * residual

        # L2 normalize to keep embeddings on the unit sphere
        output = output / (output.norm(dim=-1, keepdim=True) + 1e-8)
        return output


class BindingNetwork:
    """
    The associative bridge between perception and concepts.

    Takes visual and auditory features from Layer 1 and produces
    unified concept embeddings. Trained via contrastive learning:
    visual+audio pairs from the SAME concept should produce
    similar embeddings; different concepts should produce
    dissimilar embeddings.
    """

    def __init__(self, visual_dim: int = 64, auditory_dim: int = 32,
                 output_dim: int = 64, lr: float = 0.001):
        self.visual_dim = visual_dim
        self.auditory_dim = auditory_dim
        self.output_dim = output_dim

        self.network = CrossModalBinder(visual_dim, auditory_dim, output_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self._bindings_created = 0
        self._training_steps = 0
        self._total_loss = 0.0

        total = sum(p.numel() for p in self.network.parameters())
        logger.info("Binding network initialized (%d parameters)", total)

    def bind(self, visual_features: Optional[np.ndarray] = None,
             auditory_features: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Create a unified concept embedding from visual + auditory features.

        Either or both modalities can be present. Missing modalities
        are filled with zeros (the network learns to handle partial input).
        """
        v_tensor, a_tensor = self._prepare_tensors(visual_features, auditory_features)

        with torch.no_grad():
            unified = self.network(v_tensor, a_tensor)

        self._bindings_created += 1
        return unified.squeeze(0).numpy()

    def train_binding(self, visual_features: Optional[np.ndarray],
                      auditory_features: Optional[np.ndarray],
                      target_embedding: Optional[np.ndarray] = None,
                      negative_embedding: Optional[np.ndarray] = None) -> float:
        """
        Train the binding network on a visual-audio pair.

        If target_embedding is provided, uses MSE loss to match it.
        If negative_embedding is also provided, uses contrastive loss
        to push the output away from the negative.
        """
        v_tensor, a_tensor = self._prepare_tensors(visual_features, auditory_features)
        unified = self.network(v_tensor, a_tensor)

        if target_embedding is not None:
            target = torch.from_numpy(np.array(target_embedding, dtype=np.float32))
            target = target.unsqueeze(0) if target.dim() == 1 else target
            target = target / (target.norm(dim=-1, keepdim=True) + 1e-8)

            # Positive loss: push toward target
            positive_loss = 1.0 - nn.functional.cosine_similarity(unified, target, dim=-1).mean()

            total_loss = positive_loss

            # Negative loss: push away from negative
            if negative_embedding is not None:
                neg = torch.from_numpy(np.array(negative_embedding, dtype=np.float32)).unsqueeze(0)
                neg = neg / (neg.norm(dim=-1, keepdim=True) + 1e-8)
                negative_sim = nn.functional.cosine_similarity(unified, neg, dim=-1).mean()
                # We want negative_sim to be low, so penalize high similarity
                negative_loss = torch.clamp(negative_sim - 0.1, min=0.0)
                total_loss = total_loss + 0.5 * negative_loss
        else:
            # Self-supervised: consistency loss
            # If we permute the input slightly, the output should be similar
            noise_v = v_tensor + torch.randn_like(v_tensor) * 0.05
            noise_a = a_tensor + torch.randn_like(a_tensor) * 0.05
            unified_noisy = self.network(noise_v, noise_a)
            total_loss = 1.0 - nn.functional.cosine_similarity(unified, unified_noisy, dim=-1).mean()

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        self._training_steps += 1
        self._total_loss += total_loss.item()

        return total_loss.item()

    def _prepare_tensors(self, visual: Optional[np.ndarray],
                         auditory: Optional[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        if visual is None:
            v = np.zeros(self.visual_dim, dtype=np.float32)
        else:
            v = np.array(visual, dtype=np.float32).flatten()[:self.visual_dim]
            if len(v) < self.visual_dim:
                v = np.pad(v, (0, self.visual_dim - len(v)))

        if auditory is None:
            a = np.zeros(self.auditory_dim, dtype=np.float32)
        else:
            a = np.array(auditory, dtype=np.float32).flatten()[:self.auditory_dim]
            if len(a) < self.auditory_dim:
                a = np.pad(a, (0, self.auditory_dim - len(a)))

        return (torch.from_numpy(v).unsqueeze(0),
                torch.from_numpy(a).unsqueeze(0))

    def save_weights(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.network.state_dict(),
            'bindings': self._bindings_created,
            'steps': self._training_steps,
            'loss': self._total_loss,
        }, path)
        logger.info("Binding network saved (%d bindings)", self._bindings_created)

    def load_weights(self, path: Path):
        if path.exists():
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.network.load_state_dict(checkpoint['state_dict'])
            self._bindings_created = checkpoint.get('bindings', 0)
            self._training_steps = checkpoint.get('steps', 0)
            self._total_loss = checkpoint.get('loss', 0.0)
            logger.info("Binding network loaded (%d prior bindings)", self._bindings_created)

    def get_stats(self) -> dict:
        return {
            "bindings_created": self._bindings_created,
            "training_steps": self._training_steps,
            "avg_loss": self._total_loss / max(1, self._training_steps),
            "params": sum(p.numel() for p in self.network.parameters()),
        }
