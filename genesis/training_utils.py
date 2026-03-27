"""
Genesis Mind — Training Utilities

Shared training infrastructure for all neural networks:
    - Gradient clipping
    - Learning rate warmup + scheduling
    - Proper weight initialization
    - Safe backward pass (NaN guard)

Every network in Genesis uses these utilities to ensure
stable, reproducible training from real-world noisy input.
"""

import math
import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger("genesis.training_utils")


def init_weights(module: nn.Module):
    """
    Initialize weights using best practices for each layer type.

    - Linear: Xavier uniform (good for ReLU pipelines)
    - GRU: Orthogonal (prevents vanishing/exploding gradients in RNNs)
    - LayerNorm: ones/zeros
    """
    for name, param in module.named_parameters():
        if 'weight' in name:
            if 'gru' in name.lower() or 'rnn' in name.lower():
                nn.init.orthogonal_(param)
            elif param.dim() >= 2:
                nn.init.xavier_uniform_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
    logger.debug("Initialized weights for %s", module.__class__.__name__)


def safe_backward(loss: torch.Tensor, optimizer: optim.Optimizer,
                  parameters, max_norm: float = 1.0,
                  scheduler: Optional[object] = None) -> float:
    """
    Clip-safe backward pass with NaN guard.

    Returns the gradient norm (pre-clipping) for monitoring.
    If loss is NaN, skips the backward pass entirely.
    """
    if torch.isnan(loss) or torch.isinf(loss):
        logger.warning("NaN/Inf loss detected — skipping backward pass")
        return 0.0

    optimizer.zero_grad()
    loss.backward()

    # Compute and clip gradient norm
    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)

    # NaN gradient guard
    if torch.isnan(grad_norm):
        logger.warning("NaN gradients detected — zeroing gradients")
        optimizer.zero_grad()
        return 0.0

    optimizer.step()

    if scheduler is not None:
        scheduler.step()

    return float(grad_norm)


class WarmupScheduler:
    """
    Linear warmup then cosine annealing LR scheduler.

    - Steps 0..warmup_steps: LR linearly ramps from lr/10 to lr
    - Steps warmup_steps..: cosine decay back toward lr/10
    - If mode="constant": LR stays constant after warmup

    Usage:
        scheduler = WarmupScheduler(optimizer, warmup_steps=50, mode="cosine")
        # In training loop:
        scheduler.step()
    """

    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int = 50,
                 mode: str = "cosine", T_max: int = 1000):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.mode = mode
        self.T_max = T_max
        self._step = 0
        self._base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    def step(self):
        self._step += 1
        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = self._base_lrs[i]
            if self._step <= self.warmup_steps:
                # Linear warmup from base_lr/10 to base_lr
                warmup_factor = 0.1 + 0.9 * (self._step / self.warmup_steps)
                pg['lr'] = base_lr * warmup_factor
            elif self.mode == "cosine":
                # Cosine decay
                progress = (self._step - self.warmup_steps) / max(1, self.T_max)
                cosine_factor = 0.1 + 0.9 * (1 + math.cos(math.pi * progress)) / 2
                pg['lr'] = base_lr * cosine_factor
            # else: constant — LR stays at base_lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']

    @property
    def current_step(self) -> int:
        return self._step


class GrowthWarmup:
    """
    Post-growth learning rate warmup.

    After neuroplasticity grows a network, new neurons need gentle
    initial training to avoid destabilizing existing knowledge.

    Reduces LR by 10x for `warmup_steps` steps, then restores.
    """

    def __init__(self, warmup_steps: int = 20):
        self.warmup_steps = warmup_steps
        self._remaining = 0

    def trigger(self):
        """Called after a growth event."""
        self._remaining = self.warmup_steps
        logger.info("Growth warmup triggered: %d steps at 0.1x LR", self.warmup_steps)

    @property
    def is_active(self) -> bool:
        return self._remaining > 0

    def get_lr_multiplier(self) -> float:
        """Returns the LR multiplier (0.1 during warmup, 1.0 otherwise)."""
        if self._remaining > 0:
            # Linearly ramp from 0.1 to 1.0 over warmup_steps
            progress = 1.0 - (self._remaining / self.warmup_steps)
            self._remaining -= 1
            return 0.1 + 0.9 * progress
        return 1.0

    def apply(self, optimizer: optim.Optimizer, base_lrs: list):
        """Apply the warmup multiplier to an optimizer."""
        mult = self.get_lr_multiplier()
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg['lr'] = base_lr * mult
