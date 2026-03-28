"""
Genesis Mind — GridWorld: Treasure Hunt

A simple 2D grid environment where Genesis learns to navigate
toward treasure while avoiding walls.

This is NOT a traditional RL gym. The game state is converted
into a 64-dim "visual embedding" and fed directly into the
subconscious cascade. Rewards flow through the neurochemistry
(dopamine for treasure, cortisol for walls), and the world model
learns to predict next states.

Over time, Genesis should:
    1. Stop bumping into walls (limbic avoidance learning)
    2. Navigate toward treasure (dopamine-guided exploration)
    3. Predict what happens next (world model accuracy)
    4. Develop preferences (personality shaping)

    ┌───────────────────┐
    │ . . . . .         │
    │ . # . . .         │   G = Genesis
    │ . # G . .         │   # = Wall
    │ . . . ★ .         │   ★ = Treasure
    │ . . . . .         │   . = Empty
    └───────────────────┘
"""

import logging
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import IntEnum

logger = logging.getLogger("genesis.games.grid_world")


class Tile(IntEnum):
    EMPTY = 0
    WALL = 1
    TREASURE = 2
    AGENT = 3


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTION_NAMES = {
    Action.UP: "↑",
    Action.DOWN: "↓",
    Action.LEFT: "←",
    Action.RIGHT: "→",
}

ACTION_DELTAS = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass
class StepResult:
    """Result of a single game step."""
    action: Action
    reward: float
    done: bool
    hit_wall: bool
    found_treasure: bool
    position: Tuple[int, int]
    steps: int
    total_reward: float


class GridWorld:
    """
    A simple 2D navigation environment.

    The grid contains:
        - Empty spaces (navigable)
        - Walls (impassable, negative reward)
        - Treasures (goal, large positive reward)

    The environment is designed to test whether Genesis's neural
    cascade can learn optimal behavior through reward signals.
    """

    def __init__(self, size: int = 7, n_walls: int = 6, n_treasures: int = 2,
                 step_penalty: float = -0.01, wall_penalty: float = -0.3,
                 treasure_reward: float = 1.0, max_steps: int = 100):
        self.size = size
        self.n_walls = n_walls
        self.n_treasures = n_treasures
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.treasure_reward = treasure_reward
        self.max_steps = max_steps

        # State
        self.grid = np.zeros((size, size), dtype=np.int32)
        self.agent_pos = (0, 0)
        self.steps = 0
        self.total_reward = 0.0
        self.treasures_found = 0
        self.walls_hit = 0
        self.done = False

        # History
        self.episode_history: List[StepResult] = []

        # Stats across episodes
        self.episodes_played = 0
        self.total_treasures_found = 0
        self.total_walls_hit = 0
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset the environment to a new random layout."""
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self.steps = 0
        self.total_reward = 0.0
        self.treasures_found = 0
        self.walls_hit = 0
        self.done = False
        self.episode_history = []

        # Place walls randomly (not on corners)
        placed = 0
        while placed < self.n_walls:
            r, c = random.randint(1, self.size - 2), random.randint(1, self.size - 2)
            if self.grid[r, c] == Tile.EMPTY:
                self.grid[r, c] = Tile.WALL
                placed += 1

        # Place treasures
        placed = 0
        while placed < self.n_treasures:
            r, c = random.randint(0, self.size - 1), random.randint(0, self.size - 1)
            if self.grid[r, c] == Tile.EMPTY and (r, c) != (0, 0):
                self.grid[r, c] = Tile.TREASURE
                placed += 1

        # Agent starts at (0, 0)
        self.agent_pos = (0, 0)

        return self._get_state_embedding()

    def step(self, action: Action) -> StepResult:
        """Take one step in the environment."""
        if self.done:
            return StepResult(
                action=action, reward=0.0, done=True,
                hit_wall=False, found_treasure=False,
                position=self.agent_pos, steps=self.steps,
                total_reward=self.total_reward,
            )

        self.steps += 1
        dr, dc = ACTION_DELTAS[action]
        new_r = self.agent_pos[0] + dr
        new_c = self.agent_pos[1] + dc

        reward = self.step_penalty  # Small penalty per step
        hit_wall = False
        found_treasure = False

        # Check boundaries
        if new_r < 0 or new_r >= self.size or new_c < 0 or new_c >= self.size:
            # Hit boundary wall
            reward += self.wall_penalty
            hit_wall = True
            self.walls_hit += 1
        elif self.grid[new_r, new_c] == Tile.WALL:
            # Hit internal wall
            reward += self.wall_penalty
            hit_wall = True
            self.walls_hit += 1
        else:
            # Valid move
            self.agent_pos = (new_r, new_c)

            if self.grid[new_r, new_c] == Tile.TREASURE:
                reward += self.treasure_reward
                found_treasure = True
                self.treasures_found += 1
                self.total_treasures_found += 1
                self.grid[new_r, new_c] = Tile.EMPTY  # Collect the treasure

        self.total_reward += reward

        # Check if all treasures collected or max steps
        remaining = np.sum(self.grid == Tile.TREASURE)
        if remaining == 0 or self.steps >= self.max_steps:
            self.done = True
            self.episodes_played += 1
            self.total_walls_hit += self.walls_hit
            self.episode_rewards.append(self.total_reward)
            self.episode_lengths.append(self.steps)

        result = StepResult(
            action=action, reward=reward, done=self.done,
            hit_wall=hit_wall, found_treasure=found_treasure,
            position=self.agent_pos, steps=self.steps,
            total_reward=self.total_reward,
        )
        self.episode_history.append(result)
        return result

    def _get_state_embedding(self) -> np.ndarray:
        """
        Convert the current game state into a 64-dim embedding
        that can be fed directly into Genesis's subconscious cascade.

        Encoding:
            [0:49]  = flattened 7x7 grid (normalized tile types)
            [49:51] = agent position (row, col) normalized to [0, 1]
            [51:53] = distance to nearest treasure (dx, dy) normalized
            [53:55] = walls hit ratio, treasures found ratio
            [55:57] = step count normalized, total reward normalized
            [57:64] = padding zeros
        """
        embedding = np.zeros(64, dtype=np.float32)

        # Flattened grid (normalized: 0=empty, 0.33=wall, 0.67=treasure)
        flat_grid = self.grid.flatten().astype(np.float32) / 3.0
        embedding[:min(49, len(flat_grid))] = flat_grid[:49]

        # Agent position (normalized)
        embedding[49] = self.agent_pos[0] / max(1, self.size - 1)
        embedding[50] = self.agent_pos[1] / max(1, self.size - 1)

        # Distance to nearest treasure
        treasure_locs = np.argwhere(self.grid == Tile.TREASURE)
        if len(treasure_locs) > 0:
            distances = np.abs(treasure_locs - np.array(self.agent_pos)).sum(axis=1)
            nearest_idx = np.argmin(distances)
            nearest = treasure_locs[nearest_idx]
            embedding[51] = (nearest[0] - self.agent_pos[0]) / self.size
            embedding[52] = (nearest[1] - self.agent_pos[1]) / self.size

        # Performance metrics
        embedding[53] = self.walls_hit / max(1, self.steps)
        embedding[54] = self.treasures_found / max(1, self.n_treasures)
        embedding[55] = self.steps / self.max_steps
        embedding[56] = np.clip(self.total_reward / 2.0, -1.0, 1.0)

        return embedding

    def get_valid_actions(self) -> List[Action]:
        """Return actions that don't hit walls or boundaries."""
        valid = []
        for action in Action:
            dr, dc = ACTION_DELTAS[action]
            new_r = self.agent_pos[0] + dr
            new_c = self.agent_pos[1] + dc
            if (0 <= new_r < self.size and 0 <= new_c < self.size
                    and self.grid[new_r, new_c] != Tile.WALL):
                valid.append(action)
        return valid

    def render(self) -> str:
        """Render the grid as a string for terminal display."""
        lines = []
        lines.append(f"  ┌{'───' * self.size}┐")
        for r in range(self.size):
            row = "  │"
            for c in range(self.size):
                if (r, c) == self.agent_pos:
                    row += " G "
                elif self.grid[r, c] == Tile.WALL:
                    row += " # "
                elif self.grid[r, c] == Tile.TREASURE:
                    row += " ★ "
                else:
                    row += " · "
            row += "│"
            lines.append(row)
        lines.append(f"  └{'───' * self.size}┘")
        lines.append(f"  Step: {self.steps}/{self.max_steps}  "
                     f"Reward: {self.total_reward:.2f}  "
                     f"Treasures: {self.treasures_found}/{self.n_treasures}  "
                     f"Walls hit: {self.walls_hit}")
        return "\n".join(lines)

    def get_stats(self) -> Dict:
        """Get lifetime statistics."""
        return {
            "episodes_played": self.episodes_played,
            "total_treasures_found": self.total_treasures_found,
            "total_walls_hit": self.total_walls_hit,
            "avg_reward": (
                sum(self.episode_rewards) / max(1, len(self.episode_rewards))
            ),
            "avg_length": (
                sum(self.episode_lengths) / max(1, len(self.episode_lengths))
            ),
            "best_reward": max(self.episode_rewards) if self.episode_rewards else 0,
            "last_5_avg": (
                sum(self.episode_rewards[-5:]) / max(1, len(self.episode_rewards[-5:]))
                if self.episode_rewards else 0
            ),
        }
