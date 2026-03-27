"""
Genesis Mind — Binding Network (Layer 2: Associative Bridge)

In the human brain, the superior temporal sulcus and angular gyrus
bind visual and auditory signals into unified percepts. When you
see a dog AND hear "dog" at the same time, these regions fuse
them into a single cross-modal concept.

This module is a small MLP that learns to create unified concept
embeddings from separate visual and auditory latent vectors.

    Input:  visual_latent (64-dim) ⊕ auditory_latent (64-dim) = 128-dim
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
    Fuses visual and auditory latent vectors into a unified concept using a Dual Encoder.
    
    This architecture enables true contrastive learning (InfoNCE) where we can pull
    matching pairs together and push mismatched pairs apart in a shared latent space.
    """

    def __init__(self, visual_dim: int = 64, auditory_dim: int = 64,
                 output_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        self.v_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.a_proj = nn.Sequential(
            nn.Linear(auditory_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, visual: torch.Tensor, auditory: torch.Tensor) -> torch.Tensor:
        """Returns the fused concept (mean of L2 normalized projections)."""
        v_emb, a_emb = self.get_projections(visual, auditory)
        fused = (v_emb + a_emb) / 2.0
        return fused / (fused.norm(dim=-1, keepdim=True) + 1e-8)

    def get_projections(self, visual: torch.Tensor, auditory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        v_emb = self.v_proj(visual)
        a_emb = self.a_proj(auditory)
        v_emb = v_emb / (v_emb.norm(dim=-1, keepdim=True) + 1e-8)
        a_emb = a_emb / (a_emb.norm(dim=-1, keepdim=True) + 1e-8)
        return v_emb, a_emb


class BindingNetwork:
    """
    The associative bridge between perception and concepts.

    Takes visual and auditory features from Layer 1 and produces
    unified concept embeddings. Trained via contrastive learning:
    visual+audio pairs from the SAME concept should produce
    similar embeddings; different concepts should produce
    dissimilar embeddings.
    """

    def __init__(self, visual_dim: int = 64, auditory_dim: int = 64,
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

    def train_binding_batch(self, visual_features_list: list, auditory_features_list: list) -> float:
        """
        Self-supervised Contrastive Learning (InfoNCE) across a batch.
        visual and auditory features are aligned (v[i] belongs with a[i]).
        """
        if len(visual_features_list) < 2:
            return 0.0
            
        v_tensors, a_tensors = [], []
        for v, a in zip(visual_features_list, auditory_features_list):
            vt, at = self._prepare_tensors(v, a)
            v_tensors.append(vt.squeeze(0))
            a_tensors.append(at.squeeze(0))
            
        v_batch = torch.stack(v_tensors)
        a_batch = torch.stack(a_tensors)
        
        v_emb, a_emb = self.network.get_projections(v_batch, a_batch)
        
        # InfoNCE Loss
        logit_scale = self.network.logit_scale.exp()
        logits_per_visual = logit_scale * v_emb @ a_emb.t()
        logits_per_auditory = logits_per_visual.t()
        
        batch_size = v_batch.size(0)
        labels = torch.arange(batch_size, device=v_batch.device)
        
        loss_v = nn.functional.cross_entropy(logits_per_visual, labels)
        loss_a = nn.functional.cross_entropy(logits_per_auditory, labels)
        total_loss = (loss_v + loss_a) / 2.0
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        self._training_steps += 1
        self._total_loss += total_loss.item()
        
        return total_loss.item()

    def train_binding(self, visual_features: Optional[np.ndarray],
                      auditory_features: Optional[np.ndarray],
                      target_embedding: Optional[np.ndarray] = None,
                      negative_embedding: Optional[np.ndarray] = None) -> float:
        return 0.0

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
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                self.network.load_state_dict(checkpoint['state_dict'])
                self._bindings_created = checkpoint.get('bindings', 0)
                self._training_steps = checkpoint.get('steps', 0)
                self._total_loss = checkpoint.get('loss', 0.0)
                logger.info("Binding network loaded (%d prior bindings)", self._bindings_created)
            except RuntimeError as e:
                logger.warning("Binding weights incompatible (architecture changed), reinitializing: %s", e)
                path.unlink(missing_ok=True)

    def get_stats(self) -> dict:
        return {
            "bindings_created": self._bindings_created,
            "training_steps": self._training_steps,
            "avg_loss": self._total_loss / max(1, self._training_steps),
            "params": sum(p.numel() for p in self.network.parameters()),
        }
