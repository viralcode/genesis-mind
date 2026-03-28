"""
Genesis Mind — Game Trainer

Connects the GridWorld game to Genesis's neural cascade.

This is the critical integration: game state flows through
the subconscious EXACTLY like visual/auditory input, and
rewards flow through the neurochemistry EXACTLY like real
emotional experiences.

The training loop:
    1. Game state → 64-dim embedding (like a visual embedding)
    2. Feed to subconscious cascade (limbic → binding → personality → world model)
    3. World model predicts next state → surprise signal
    4. Meta-controller selects action based on routing weights
    5. Execute action in game
    6. Reward → dopamine (treasure) or cortisol (wall)
    7. Store experience in hippocampal replay buffer
    8. Repeat

Over episodes, Genesis should:
    - Develop wall avoidance (limbic learning)
    - Navigate toward treasures (dopamine-guided)
    - Predict game outcomes (world model training)
    - Improve average reward (the real test)
"""

import logging
import time
import numpy as np
from typing import Dict, Optional

from genesis.games.grid_world import GridWorld, Action, ACTION_NAMES

logger = logging.getLogger("genesis.games.trainer")


class GameTrainer:
    """
    Trains Genesis to play GridWorld by feeding game states
    through the full neural cascade.

    This is a genuine test of the architecture: can the
    subconscious cascade learn optimal behavior from reward?
    """

    def __init__(self, mind, grid_size: int = 7, n_walls: int = 6,
                 n_treasures: int = 2, max_steps: int = 80):
        """
        Args:
            mind: The Genesis Mind instance (has subconscious, neurochemistry, etc.)
        """
        self.mind = mind
        self.game = GridWorld(
            size=grid_size,
            n_walls=n_walls,
            n_treasures=n_treasures,
            max_steps=max_steps,
        )
        self._action_history: list = []
        self._reward_history: list = []
        self._episode_surprises: list = []

        # Exploration parameters (decay over episodes)
        self.epsilon = 0.9       # Start with 90% random exploration
        self.epsilon_decay = 0.985  # Decay per episode
        self.epsilon_min = 0.1   # Always keep 10% exploration

        logger.info("Game trainer initialized: %dx%d grid, %d walls, %d treasures",
                     grid_size, grid_size, n_walls, n_treasures)

    def select_action(self, state_embedding: np.ndarray) -> Action:
        """
        Select an action using the neural cascade.

        Strategy:
            - With probability epsilon: random action (exploration)
            - Otherwise: use the world model to pick the action
              with the lowest predicted surprise (exploitation)
        """
        import random

        # Epsilon-greedy exploration
        if random.random() < self.epsilon:
            valid = self.game.get_valid_actions()
            if valid:
                return random.choice(valid)
            return random.choice(list(Action))

        # Exploitation: simulate each action, pick lowest surprise
        best_action = Action.UP
        best_score = float('-inf')

        subconscious = self.mind.subconscious
        state = subconscious.personality.get_consciousness_state()

        for action in Action:
            # Create a hypothetical next state by shifting the embedding
            hypothetical = state_embedding.copy()
            # Encode the action direction into the embedding
            hypothetical[57 + action] = 1.0

            # Ask the world model: how surprising would this be?
            try:
                surprise = subconscious.world_model.predict_and_learn(
                    hypothetical[:64], state
                )
                # Lower surprise = more predictable = better known path
                # But we also want SOME novelty (curiosity-driven)
                score = -surprise  # Prefer predictable outcomes
                if score > best_score:
                    best_score = score
                    best_action = action
            except Exception:
                pass

        # Still enforce valid actions when possible
        valid = self.game.get_valid_actions()
        if valid and best_action not in valid:
            best_action = random.choice(valid)

        return best_action

    def play_episode(self, render: bool = True, verbose: bool = True) -> Dict:
        """
        Play one complete episode of GridWorld.

        Each step feeds the game state through the full subconscious
        cascade and rewards flow through neurochemistry.

        Returns:
            Episode report with reward, steps, treasures, etc.
        """
        state_embedding = self.game.reset()
        episode_surprises = []
        episode_actions = []

        if render and verbose:
            print("\n" + self.game.render())
            print()

        while not self.game.done:
            # 1. Feed current state through the subconscious cascade
            context_vec = self.mind.proprioception.get_context_vector()
            result = self.mind.subconscious.process_experience(
                visual_embedding=state_embedding,
                text_embedding=None,
                context=context_vec,
                train=True,
                emotional_intensity=abs(self.game.total_reward) / 2.0,
                drive_hunger=self.mind.drives.curiosity_hunger.level,
            )

            surprise = result.get('surprise', 0.0)
            episode_surprises.append(surprise)

            # 2. Select action using neural cascade
            action = self.select_action(state_embedding)
            episode_actions.append(action)

            # 3. Execute action in game
            step_result = self.game.step(action)

            # 4. Reward → Neurochemistry (this is the learning signal!)
            if step_result.found_treasure:
                # TREASURE! Big dopamine spike — this is how Genesis learns
                # that finding treasure is good
                self.mind.neurochemistry.on_successful_learning()
                self.mind.neurochemistry.dopamine.spike(0.25)
                self.mind.drives.on_novel_stimulus()
                if verbose:
                    logger.info("    ★ TREASURE FOUND! Dopamine spike!")

            elif step_result.hit_wall:
                # WALL! Cortisol spike — this is how Genesis learns
                # that hitting walls is bad
                self.mind.neurochemistry.cortisol.spike(0.15)
                self.mind.neurochemistry.dopamine.suppress(0.05)
                if verbose:
                    logger.debug("    # Wall hit. Cortisol spike.")

            # 5. Get new state embedding for next step
            state_embedding = self.game._get_state_embedding()

            # 6. Render if requested
            if render and verbose and self.game.steps % 10 == 0:
                print(self.game.render())
                print()

        # Episode complete — record stats
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self._reward_history.append(self.game.total_reward)
        self._episode_surprises.append(
            sum(episode_surprises) / max(1, len(episode_surprises))
        )

        # Final render
        if render and verbose:
            print(self.game.render())
            print()

        report = {
            "episode": self.game.episodes_played,
            "reward": round(self.game.total_reward, 3),
            "steps": self.game.steps,
            "treasures": self.game.treasures_found,
            "walls_hit": self.game.walls_hit,
            "epsilon": round(self.epsilon, 3),
            "avg_surprise": round(
                sum(episode_surprises) / max(1, len(episode_surprises)), 4
            ),
            "actions": "".join(ACTION_NAMES[a] for a in episode_actions),
        }

        if verbose:
            logger.info(
                "Episode %d complete: reward=%.2f, steps=%d, "
                "treasures=%d/%d, walls=%d, ε=%.2f",
                report["episode"], report["reward"], report["steps"],
                report["treasures"], self.game.n_treasures,
                report["walls_hit"], report["epsilon"],
            )

        return report

    def train(self, n_episodes: int = 20, render_every: int = 5,
              verbose: bool = True) -> Dict:
        """
        Train Genesis over multiple episodes.

        This is the main training loop. After training, you should
        see the average reward increase and wall hits decrease.

        Args:
            n_episodes: How many games to play
            render_every: Render the grid every N episodes
            verbose: Print detailed logs
        """
        if verbose:
            print("\n" + "=" * 60)
            print("  GENESIS GAME TRAINING — GridWorld Treasure Hunt")
            print("=" * 60)
            print(f"  Grid: {self.game.size}x{self.game.size}")
            print(f"  Walls: {self.game.n_walls}  Treasures: {self.game.n_treasures}")
            print(f"  Episodes: {n_episodes}  Max steps: {self.game.max_steps}")
            print(f"  Epsilon: {self.epsilon:.2f} → {self.epsilon_min:.2f}")
            print("=" * 60 + "\n")

        start_time = time.time()
        reports = []

        for ep in range(n_episodes):
            render = (ep % render_every == 0) or (ep == n_episodes - 1)
            report = self.play_episode(
                render=render and verbose,
                verbose=verbose,
            )
            reports.append(report)

            if verbose and ep % render_every == 0:
                stats = self.game.get_stats()
                print(f"  [Episode {ep+1}/{n_episodes}]  "
                      f"Avg reward: {stats['last_5_avg']:.2f}  "
                      f"Total treasures: {stats['total_treasures_found']}  "
                      f"Total walls: {stats['total_walls_hit']}\n")

        elapsed = time.time() - start_time

        # Final summary
        summary = {
            "episodes": n_episodes,
            "elapsed_sec": round(elapsed, 1),
            "lifetime_stats": self.game.get_stats(),
            "final_epsilon": round(self.epsilon, 3),
            "reward_trend": {
                "first_5": round(
                    np.mean([r["reward"] for r in reports[:5]]), 3
                ) if len(reports) >= 5 else 0,
                "last_5": round(
                    np.mean([r["reward"] for r in reports[-5:]]), 3
                ) if len(reports) >= 5 else 0,
            },
            "wall_trend": {
                "first_5_avg": round(
                    np.mean([r["walls_hit"] for r in reports[:5]]), 1
                ) if len(reports) >= 5 else 0,
                "last_5_avg": round(
                    np.mean([r["walls_hit"] for r in reports[-5:]]), 1
                ) if len(reports) >= 5 else 0,
            },
        }

        if verbose:
            print("\n" + "=" * 60)
            print("  TRAINING COMPLETE")
            print("=" * 60)
            print(f"  Episodes:        {n_episodes}")
            print(f"  Time:            {elapsed:.1f}s")
            print(f"  Final epsilon:   {self.epsilon:.3f}")
            print(f"  Reward (first 5): {summary['reward_trend']['first_5']:.3f}")
            print(f"  Reward (last 5):  {summary['reward_trend']['last_5']:.3f}")
            print(f"  Walls (first 5):  {summary['wall_trend']['first_5_avg']:.1f}")
            print(f"  Walls (last 5):   {summary['wall_trend']['last_5_avg']:.1f}")
            trend = summary['reward_trend']['last_5'] - summary['reward_trend']['first_5']
            if trend > 0.1:
                print(f"  Trend:           📈 IMPROVING (+{trend:.3f})")
            elif trend < -0.1:
                print(f"  Trend:           📉 Declining ({trend:.3f})")
            else:
                print(f"  Trend:           ➡️  Stable ({trend:+.3f})")
            print("=" * 60 + "\n")

        return summary

    def get_status(self) -> Dict:
        """Get current trainer status for the dashboard."""
        return {
            "game_stats": self.game.get_stats(),
            "epsilon": round(self.epsilon, 3),
            "episodes_played": self.game.episodes_played,
            "reward_history": self._reward_history[-20:],
        }
