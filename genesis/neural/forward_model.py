"""
Genesis Mind — Forward Model (Predictive Coding)

Instead of just compressing frames (like an autoencoder), Genesis
needs to build an internal model of physics and time. It must learn
how the world changes from state to state.

This module is a World Model that takes the current concept state
and an internal context, and tries to predict the NEXT concept state.
When it fails to predict correctly, that surprise becomes a strong
learning signal (and triggers curiosity and cortisol/dopamine).

    Predictive Coding (JEPA-inspired):
    concept(t+1)_predicted = ForwardModel(concept(t), hidden_state)
    Loss = MSE(concept(t+1)_predicted, concept(t+1)_actual)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from genesis.neural.device import DEVICE, get_state_dict_safe, strip_compile_prefix, to_device

logger = logging.getLogger("genesis.neural.forward_model")


class ForwardPredictor(nn.Module):
    """
    Predicts the future state in concept-space.
    """

    def __init__(self, concept_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        
        # Takes the current concept (64) + current hidden conscious state (128)
        self.net = nn.Sequential(
            nn.Linear(concept_dim + hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, concept_dim)
        )

    def forward(self, current_concept: torch.Tensor, conscious_state: torch.Tensor):
        combined = torch.cat([current_concept, conscious_state], dim=-1)
        predicted_next = self.net(combined)
        predicted_next = predicted_next / (predicted_next.norm(dim=-1, keepdim=True) + 1e-8)
        return predicted_next


class WorldModel:
    """
    The internal simulator of Genesis. 
    Predicts what forms the next sensory input will take.
    """

    def __init__(self, concept_dim: int = 64, hidden_dim: int = 128, lr: float = 0.001):
        self.concept_dim = concept_dim
        self.hidden_dim = hidden_dim

        self.network = to_device(ForwardPredictor(concept_dim, hidden_dim))
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.last_concept: Optional[torch.Tensor] = None
        self.last_state: Optional[torch.Tensor] = None

        self._predictions_made = 0
        self._total_loss = 0.0

        total = sum(p.numel() for p in self.network.parameters())
        logger.info("Forward World Model initialized (%d parameters, device=%s)", total, DEVICE)

    def predict_and_learn(self, current_concept: np.ndarray, current_state: np.ndarray) -> float:
        """
        Takes the freshly perceived concept, compares it against what we MIGHT
        have predicted a moment ago, computes loss, and then makes a NEW prediction
        for the future.
        """
        curr_c_tensor = torch.from_numpy(current_concept).float().unsqueeze(0).to(DEVICE)
        curr_s_tensor = torch.from_numpy(current_state).float().unsqueeze(0).to(DEVICE)

        surprise = 0.0

        # If we made a prediction in the past, evaluate how wrong we were
        if self.last_concept is not None and self.last_state is not None:
            predicted_concept = self.network(self.last_concept, self.last_state)
            
            # Loss is MSE between prediction and actual outcome
            loss = self.criterion(predicted_concept, curr_c_tensor)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            surprise = loss.item()
            self._total_loss += surprise
            self._predictions_made += 1

        # Store today's observation to be evaluated tomorrow
        self.last_concept = curr_c_tensor.detach()
        self.last_state = curr_s_tensor.detach()

        return surprise

    def save_weights(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': get_state_dict_safe(self.network),
            'predictions': self._predictions_made,
            'loss': self._total_loss,
            # Architecture dimensions — critical for correct reload
            'hidden_dim': self.hidden_dim,
            'concept_dim': self.concept_dim,
        }, path)
        logger.info("World model saved (%d predictions)", self._predictions_made)

    def load_weights(self, path: Path):
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)

                # Reconstruct at saved dimensions if they differ
                saved_hidden = checkpoint.get('hidden_dim', self.hidden_dim)
                saved_concept = checkpoint.get('concept_dim', self.concept_dim)

                if saved_hidden != self.hidden_dim:
                    logger.info(
                        "Resizing world model to match checkpoint: hidden %d→%d",
                        self.hidden_dim, saved_hidden,
                    )
                    self.hidden_dim = saved_hidden
                    self.concept_dim = saved_concept
                    self.network = to_device(ForwardPredictor(saved_concept, saved_hidden))
                    self.optimizer = optim.Adam(self.network.parameters(), lr=0.001)

                self.network.load_state_dict(strip_compile_prefix(checkpoint['state_dict']))
                self._predictions_made = checkpoint.get('predictions', 0)
                self._total_loss = checkpoint.get('loss', 0.0)
                logger.info("World model loaded (%d prior predictions, hidden=%d)",
                            self._predictions_made, self.hidden_dim)
            except RuntimeError as e:
                logger.warning("World model weights incompatible, reinitializing: %s", e)

    def get_stats(self) -> dict:
        return {
            "predictions_made": self._predictions_made,
            "avg_surprise": self._total_loss / max(1, self._predictions_made),
            "params": sum(p.numel() for p in self.network.parameters()),
        }
