"""
Genesis Mind — Limbic System (Layer 1: Subconscious)

The limbic system is the seat of instinct and emotion in biology.
Before the prefrontal cortex has time to reason about a stimulus,
the amygdala has already fired a chemical response.

This module is a small MLP that learns to map raw sensory features
directly to neurochemical responses — BEFORE conscious processing.

    Input:  Concatenated visual (64-dim) + auditory (64-dim) features = 128-dim
    Output: 4-dim neurochemical response (dopamine, cortisol, serotonin, oxytocin)

Training:
    Supervised by the axiom-based moral evaluation system.
    When the conscious mind evaluates something as "positive",
    the limbic system is trained to produce the same response
    for that sensory pattern in the future — WITHOUT needing
    the conscious evaluation. This is how instinct forms.

    Over time, Genesis will "feel" before it "thinks."
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from genesis.training_utils import safe_backward, WarmupScheduler, init_weights

from genesis.neural.device import DEVICE, get_state_dict_safe, strip_compile_prefix, to_device

logger = logging.getLogger("genesis.neural.limbic_system")


class LimbicNetwork(nn.Module):
    """
    Maps raw sensory features → neurochemical response.

    A 3-layer MLP that learns instinctual emotional reactions.
    Trained by the conscious evaluation system so that over time,
    the subconscious "catches up" and can react instinctively.
    """

    def __init__(self, input_dim: int = 96, hidden_dim: int = 64):
        super().__init__()

        # 4 output channels: dopamine, cortisol, serotonin, oxytocin
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4),
            nn.Sigmoid(),  # Outputs in [0, 1]
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


class LimbicSystem:
    """
    The emotional and instinctual core.

    Takes raw sensory features (or concepts) and maps them to
    four primary neurochemicals:
        - Dopamine (Reward/Pleasure)
        - Cortisol (Stress/Pain)
        - Serotonin (Stability/Confidence)
        - Oxytocin (Bonding/Trust)

    This network learns to react *before* conscious thought happens.
    If it sees a snake, it spikes cortisol instantly.
    """

    def __init__(self, visual_dim: int = 64, auditory_dim: int = 64,
                 hidden_dim: int = 64, lr: float = 0.0005):
        self.visual_dim = visual_dim
        self.auditory_dim = auditory_dim
        input_dim = visual_dim + auditory_dim

        self.network = to_device(LimbicNetwork(input_dim=input_dim))
        init_weights(self.network)  # Xavier/zeros initialization
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.scheduler = WarmupScheduler(self.optimizer, warmup_steps=50, mode="cosine")
        self.criterion = nn.MSELoss()

        self._reactions = 0
        self._training_steps = 0
        self._total_loss = 0.0
        self._last_grad_norm = 0.0

        total = sum(p.numel() for p in self.network.parameters())
        logger.info("Limbic system initialized (%d parameters, device=%s)", total, DEVICE)

    def react(self, visual_features: Optional[np.ndarray] = None,
              auditory_features: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Generate an instinctual neurochemical response to sensory input.

        This is INSTANT — no reasoning, no memory lookup, just raw
        pattern → emotion mapping based on learned experience.

        Returns:
            Dict with dopamine, cortisol, serotonin, oxytocin levels (0-1)
        """
        features = self._make_features(visual_features, auditory_features)
        tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            response = self.network(tensor).squeeze(0).cpu().numpy()

        self._reactions += 1

        return {
            "dopamine": float(response[0]),
            "cortisol": float(response[1]),
            "serotonin": float(response[2]),
            "oxytocin": float(response[3]),
        }

    def train_instinct(self, visual_features: Optional[np.ndarray],
                       auditory_features: Optional[np.ndarray],
                       target_chemicals: Dict[str, float]) -> float:
        """
        Train the limbic system to associate this sensory pattern
        with the given neurochemical response.

        Called AFTER the conscious mind has evaluated the experience.
        Over time, this teaches the subconscious to react correctly
        before the conscious mind even processes the input.

        This is how instinct forms:
            1. Baby touches hot stove → conscious mind evaluates "pain"
            2. Limbic system trained: visual_pattern(stove) → cortisol
            3. Next time: limbic system fires cortisol INSTANTLY
               before conscious mind even recognizes "stove"
        """
        features = self._make_features(visual_features, auditory_features)
        input_tensor = torch.from_numpy(features).unsqueeze(0).to(DEVICE)

        target = torch.tensor([[
            target_chemicals.get("dopamine", 0.5),
            target_chemicals.get("cortisol", 0.2),
            target_chemicals.get("serotonin", 0.5),
            target_chemicals.get("oxytocin", 0.3),
        ]], dtype=torch.float32, device=DEVICE)

        prediction = self.network(input_tensor)
        loss = self.criterion(prediction, target)

        grad_norm = safe_backward(
            loss, self.optimizer,
            self.network.parameters(),
            max_norm=1.0,
            scheduler=self.scheduler,
        )
        self._last_grad_norm = grad_norm

        self._training_steps += 1
        self._total_loss += loss.item()

        return loss.item()

    def _make_features(self, visual: Optional[np.ndarray],
                       auditory: Optional[np.ndarray]) -> np.ndarray:
        """Concatenate visual and auditory features into a single vector."""
        if visual is None:
            visual = np.zeros(self.visual_dim, dtype=np.float32)
        else:
            visual = np.array(visual, dtype=np.float32).flatten()[:self.visual_dim]
            if len(visual) < self.visual_dim:
                visual = np.pad(visual, (0, self.visual_dim - len(visual)))

        if auditory is None:
            auditory = np.zeros(self.auditory_dim, dtype=np.float32)
        else:
            auditory = np.array(auditory, dtype=np.float32).flatten()[:self.auditory_dim]
            if len(auditory) < self.auditory_dim:
                auditory = np.pad(auditory, (0, self.auditory_dim - len(auditory)))

        return np.concatenate([visual, auditory])

    def save_weights(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': get_state_dict_safe(self.network),
            'reactions': self._reactions,
            'training_steps': self._training_steps,
            'total_loss': self._total_loss,
        }, path)
        logger.info("Limbic system saved (%d reactions, %d training steps)",
                     self._reactions, self._training_steps)

    def load_weights(self, path: Path):
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                self.network.load_state_dict(strip_compile_prefix(checkpoint['state_dict']))
                self._reactions = checkpoint.get('reactions', 0)
                self._training_steps = checkpoint.get('training_steps', 0)
                self._total_loss = checkpoint.get('total_loss', 0.0)
                logger.info("Limbic system loaded (%d prior reactions)", self._reactions)
            except RuntimeError as e:
                logger.warning("Limbic weights incompatible (architecture changed), reinitializing: %s", e)
                path.unlink(missing_ok=True)

    def get_stats(self) -> dict:
        return {
            "reactions": self._reactions,
            "training_steps": self._training_steps,
            "avg_loss": self._total_loss / max(1, self._training_steps),
            "params": sum(p.numel() for p in self.network.parameters()),
            "last_grad_norm": round(self._last_grad_norm, 6),
            "current_lr": round(self.scheduler.get_lr(), 8),
        }
