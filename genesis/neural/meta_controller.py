"""
Genesis Mind — Meta-Controller (Neural Router)

The brain's thalamus doesn't think — it ROUTES. It decides which
cortical regions receive which signals, and how strongly.

This meta-controller is the thalamus of Genesis. Instead of a
hardcoded pipeline (Limbic → Binding → Personality → WorldModel),
the meta-controller learns to DYNAMICALLY WEIGHT each sub-network's
contribution based on the current input.

Architecture:
    Input: concatenated sensory features (visual + auditory)
    Output: 4 routing weights (one per sub-network), softmax-normalized

    routing_weights = MetaRouter(sensory_input)
    
    final_output = Σ (routing_weights[i] * sub_network[i](input))

This means:
    - For emotional stimuli → limbic weight ↑, binding weight ↓
    - For novel visual input → binding weight ↑, world model weight ↑
    - For familiar concepts → personality weight ↑ (rely on memory)
    - For abstract language → limbic weight ↓, personality weight ↑

The routing pattern IS part of the personality. Different minds
route differently — that's what makes individuals unique.

Training:
    The router is trained via the world model's surprise signal.
    Low surprise (good prediction) → reward current routing.
    High surprise (bad prediction) → adjust routing weights.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from genesis.neural.device import DEVICE, to_device

logger = logging.getLogger("genesis.neural.meta_controller")


class RouterNetwork(nn.Module):
    """
    The attention-based routing network.
    
    Takes sensory input features and outputs routing weights
    for each sub-network in the cascade.
    """

    def __init__(self, input_dim: int = 128, num_modules: int = 4, hidden_dim: int = 64):
        """
        Args:
            input_dim: Concatenated sensory features (64 visual + 64 auditory = 128)
            num_modules: Number of sub-networks to route between
            hidden_dim: Internal representation size
        """
        super().__init__()
        
        self.num_modules = num_modules
        
        # Attention mechanism: input → routing weights
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_modules),
        )
        
        # Temperature parameter for softmax sharpness
        # Higher temp = more uniform routing, lower = more selective
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, sensory_input: torch.Tensor) -> torch.Tensor:
        """
        Compute routing weights for each sub-network.
        
        Returns:
            Softmax-normalized weights of shape (batch, num_modules)
        """
        logits = self.attention(sensory_input)
        # Clamp temperature to prevent div-by-zero or explosion
        temp = torch.clamp(self.temperature, min=0.1, max=10.0)
        weights = torch.softmax(logits / temp, dim=-1)
        return weights


class MetaController:
    """
    The neural router — the thalamus of Genesis.
    
    Learns to dynamically weight sub-network contributions
    based on input characteristics. This makes the thinking
    STRUCTURE itself learnable, not just the thinking content.
    """

    MODULE_NAMES = ["limbic", "binding", "personality", "world_model"]

    def __init__(self, input_dim: int = 128, num_modules: int = 4,
                 hidden_dim: int = 64, lr: float = 0.0003):
        self.input_dim = input_dim
        self.num_modules = num_modules

        self.network = to_device(RouterNetwork(input_dim, num_modules, hidden_dim))
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        # Routing history for introspection
        self._routing_history: List[Dict[str, float]] = []
        self._max_history = 100
        self._total_routes = 0

        # Running average of routing weights (personality fingerprint)
        self._avg_weights = np.ones(num_modules, dtype=np.float32) / num_modules

        total_params = sum(p.numel() for p in self.network.parameters())
        logger.info("Meta-controller initialized (%d params, %d modules, device=%s)",
                     total_params, num_modules, DEVICE)

    def route(self, visual_embedding: np.ndarray,
              text_embedding: np.ndarray) -> Dict[str, float]:
        """
        Compute routing weights for the current input.
        
        Args:
            visual_embedding: 64-dim visual embedding
            text_embedding: 64-dim phoneme embedding
            
        Returns:
            Dict mapping module name → routing weight (0-1, sums to 1)
        """
        # Concatenate sensory inputs
        combined = np.concatenate([
            visual_embedding.flatten(),
            text_embedding.flatten(),
        ]).astype(np.float32)

        # Pad/truncate to expected input dim
        if len(combined) < self.input_dim:
            combined = np.pad(combined, (0, self.input_dim - len(combined)))
        elif len(combined) > self.input_dim:
            combined = combined[:self.input_dim]

        with torch.no_grad():
            input_tensor = torch.from_numpy(combined).unsqueeze(0).to(DEVICE)
            weights = self.network(input_tensor).squeeze(0).cpu().numpy()

        # Build named routing dict
        routing = {}
        for i, name in enumerate(self.MODULE_NAMES):
            routing[name] = float(weights[i])

        # Update running average (exponential moving average)
        alpha = 0.05
        self._avg_weights = (1 - alpha) * self._avg_weights + alpha * weights

        # Record history
        self._routing_history.append(routing)
        if len(self._routing_history) > self._max_history:
            self._routing_history.pop(0)
        self._total_routes += 1

        return routing

    def learn_from_surprise(self, visual_embedding: np.ndarray,
                            text_embedding: np.ndarray,
                            surprise: float):
        """
        Train the router using the world model's surprise signal.
        
        Low surprise = good routing → reinforce.
        High surprise = bad routing → adjust.
        
        We want to minimize surprise by routing to the right modules.
        """
        combined = np.concatenate([
            visual_embedding.flatten(),
            text_embedding.flatten(),
        ]).astype(np.float32)

        if len(combined) < self.input_dim:
            combined = np.pad(combined, (0, self.input_dim - len(combined)))
        elif len(combined) > self.input_dim:
            combined = combined[:self.input_dim]

        input_tensor = torch.from_numpy(combined).unsqueeze(0).to(DEVICE)
        weights = self.network(input_tensor)

        # Loss: encourage the router to produce weights that minimize surprise
        # If surprise is high, the current routing was suboptimal
        # We use the surprise as a scalar loss multiplied by entropy penalty
        # to encourage the router to be more decisive (less uniform)
        entropy = -(weights * torch.log(weights + 1e-8)).sum()
        
        # Surprise-weighted loss: high surprise → learn more
        # Also add small entropy regularization to prevent collapse to single module
        loss = surprise * entropy * 0.1 - 0.01 * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()

    def get_routing_personality(self) -> Dict[str, float]:
        """
        The routing fingerprint — how this mind prefers to think.
        
        This IS personality at the structural level.
        """
        return {
            name: float(self._avg_weights[i])
            for i, name in enumerate(self.MODULE_NAMES)
        }

    def get_dominant_module(self) -> Tuple[str, float]:
        """Which module does this mind rely on most?"""
        idx = int(np.argmax(self._avg_weights))
        return self.MODULE_NAMES[idx], float(self._avg_weights[idx])

    def save_weights(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'state_dict': self.network.state_dict(),
            'avg_weights': self._avg_weights,
            'total_routes': self._total_routes,
        }, path)
        logger.info("Meta-controller saved (%d routes)", self._total_routes)

    def load_weights(self, path: Path):
        if path.exists():
            try:
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                self.network.load_state_dict(checkpoint['state_dict'])
                self._avg_weights = checkpoint.get('avg_weights', self._avg_weights)
                self._total_routes = checkpoint.get('total_routes', 0)
                logger.info("Meta-controller loaded (%d prior routes)", self._total_routes)
            except RuntimeError as e:
                logger.warning("Meta-controller weights incompatible (architecture changed), reinitializing: %s", e)
                path.unlink(missing_ok=True)

    def get_stats(self) -> Dict:
        personality = self.get_routing_personality()
        dominant, dominant_weight = self.get_dominant_module()
        return {
            "total_routes": self._total_routes,
            "routing_personality": personality,
            "dominant_module": dominant,
            "dominant_weight": round(dominant_weight, 3),
            "temperature": float(self.network.temperature.data),
            "params": sum(p.numel() for p in self.network.parameters()),
        }

    def __repr__(self) -> str:
        dominant, weight = self.get_dominant_module()
        return f"MetaController(routes={self._total_routes}, dominant={dominant}@{weight:.2f})"
