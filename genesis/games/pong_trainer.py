"""
Genesis Mind — Pong Trainer (REAL Reinforcement Learning)

NO teacher. NO ball-tracker. NO cheating.

Genesis controls the paddle via a policy network trained with
REINFORCE + value baseline (Actor-Critic / A2C). The network
starts with RANDOM actions and must discover paddle-ball
relationships purely from reward signals:

    +1.0  when Genesis scores
    -1.0  when CPU scores
    +0.01 per step if paddle is near ball's y-position (shaped)
    -0.01 per step if paddle is far from ball's y-position

Architecture:
    PongPolicyNet (Actor):  64-dim state → 3 action probabilities
    PongValueNet  (Critic): 64-dim state → scalar value estimate

Training:
    At the end of each rally (when someone scores), compute
    discounted returns, advantages, and update both networks
    via policy gradient with value baseline.

The subconscious cascade processes game embeddings in the
background, so the full neural cascade (limbic, binding,
personality, world model) learns from Pong in real-time.
"""

import logging
import os
import time
import threading
import webbrowser
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from flask import Flask, send_file, request, jsonify

logger = logging.getLogger("genesis.games.pong")


# ═══════════════════════════════════════════════════════════
# Policy Network (Actor) — Chooses actions
# ═══════════════════════════════════════════════════════════

class PongPolicyNet(nn.Module):
    """
    The Actor: maps game state to action probabilities.
    No teacher. Learns entirely from reward signals.
    """
    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 3),  # [UP, STAY, DOWN]
        )
        # Small init → starts near-uniform distribution
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# Value Network (Critic) — Estimates state value
# ═══════════════════════════════════════════════════════════

class PongValueNet(nn.Module):
    """
    The Critic: estimates how good a state is.
    Reduces variance in policy gradient updates.
    """
    def __init__(self, input_dim=64, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)


ACTION_MAP = {0: 'UP', 1: 'STAY', 2: 'DOWN'}


class PongTrainer:
    """
    REAL reinforcement learning Pong trainer.

    No teacher. No ball-tracker. No shortcuts.
    Genesis discovers paddle control from reward signals alone.
    """

    def __init__(self, mind, port: int = 5051):
        self.mind = mind
        self.port = port
        self.decisions = 0
        self.last_action = 'STAY'

        # Performance tracking
        self.genesis_scores = 0
        self.cpu_scores = 0
        self.total_rallies = 0
        self.max_rally = 0
        self.current_rally = 0
        self.start_time = time.time()
        self.episodes = 0

        # ═══ ACTOR-CRITIC NETWORKS ═══
        self.policy_net = PongPolicyNet(input_dim=64, hidden_dim=64)
        self.value_net = PongValueNet(input_dim=64, hidden_dim=64)

        self.policy_optimizer = optim.Adam(
            self.policy_net.parameters(), lr=3e-3
        )
        self.value_optimizer = optim.Adam(
            self.value_net.parameters(), lr=3e-3
        )

        self.gamma = 0.99      # Discount factor
        self.entropy_coef = 0.15  # Strong exploration — prevents STAY collapse

        # ═══ EPISODE TRAJECTORY ═══
        # Store ONLY raw data — no graph tensors (thread safety)
        self._ep_states = []     # list of np.ndarray
        self._ep_actions = []    # list of int
        self._ep_rewards = []    # list of float
        self._ep_lock = threading.Lock()

        # ═══ Training stats ═══
        self.training_steps = 0
        self.policy_loss = 0.0
        self.value_loss = 0.0
        self.avg_entropy = 1.1   # log(3) ≈ 1.1 at start (uniform)
        self._min_entropy_floor = 0.5  # Minimum entropy enforced early on
        self.avg_return = 0.0
        self._return_history = deque(maxlen=50)

        # ═══ Action probabilities for UI ═══
        self._last_probs = [0.33, 0.34, 0.33]

        # Ball position tracking for shaped rewards
        self._prev_ball_y = 0.5
        self._prev_ball_x = 0.5
        self._prev_embedding = None
        self.last_surprise = 0.0

        # Subconscious integration queue
        self._pending_embeddings = deque(maxlen=20)

        # Build Flask app
        self.app = Flask(__name__, static_folder=None)
        self._register_routes()

        # Background threads
        self._running = True
        self._brain_thread = threading.Thread(
            target=self._brain_integration_loop, name="pong-brain", daemon=True
        )

        policy_params = sum(p.numel() for p in self.policy_net.parameters())
        value_params = sum(p.numel() for p in self.value_net.parameters())
        logger.info(
            "Pong RL trainer initialized — policy: %d params, value: %d params, "
            "NO TEACHER, pure reinforcement learning",
            policy_params, value_params,
        )

    def _register_routes(self):
        """Register HTTP endpoints."""

        @self.app.route('/')
        def serve_game():
            html_path = os.path.join(
                os.path.dirname(__file__), 'pong', 'index.html'
            )
            return send_file(html_path)

        @self.app.route('/api/pong/action', methods=['POST'])
        def get_action():
            data = request.json
            if not data:
                return jsonify({'action': 'STAY'})

            action, surprise = self._decide(data)
            self.decisions += 1
            self.last_action = action
            self.last_surprise = float(surprise)

            nc = self.mind.neurochemistry
            return jsonify({
                'action': action,
                'decisions': self.decisions,
                'surprise': round(float(surprise), 4),
                'episodes': self.episodes,
                'training_steps': self.training_steps,
                'policy_loss': round(float(self.policy_loss), 5),
                'value_loss': round(float(self.value_loss), 5),
                'entropy': round(float(self.avg_entropy), 4),
                'avg_return': round(float(self.avg_return), 4),
                'action_probs': [round(float(p), 3) for p in self._last_probs],
                'neural_params': (
                    sum(p.numel() for p in self.policy_net.parameters()) +
                    sum(p.numel() for p in self.value_net.parameters())
                ),
                'neurochemistry': {
                    'dopamine': round(float(nc.dopamine.level), 3),
                    'cortisol': round(float(nc.cortisol.level), 3),
                    'serotonin': round(float(nc.serotonin.level), 3),
                },
            })

        @self.app.route('/api/pong/score', methods=['POST'])
        def on_score():
            data = request.json
            if not data:
                return jsonify({'ok': True})

            scorer = data.get('scorer', '')
            rally = data.get('rally', 0)

            if scorer == 'genesis':
                self.genesis_scores += 1
                self.mind.neurochemistry.on_successful_learning()
                self.mind.neurochemistry.dopamine.spike(0.3)
                self.mind.drives.on_novel_stimulus()
                # Terminal reward: POSITIVE
                self._end_episode(terminal_reward=1.0)
                logger.info(
                    "★ GENESIS SCORES! (%d-%d) Rally: %d  +1.0 reward",
                    self.genesis_scores, self.cpu_scores, rally,
                )

            elif scorer == 'cpu':
                self.cpu_scores += 1
                self.mind.neurochemistry.cortisol.spike(0.2)
                self.mind.neurochemistry.dopamine.suppress(0.1)
                # Terminal reward: NEGATIVE
                self._end_episode(terminal_reward=-1.0)
                logger.info(
                    "✗ CPU scores. (%d-%d) Rally: %d  -1.0 reward",
                    self.genesis_scores, self.cpu_scores, rally,
                )

            # Track rallies
            if rally > 0:
                self.total_rallies += 1
                self.max_rally = max(self.max_rally, rally)
                if rally >= 5:
                    self.mind.neurochemistry.serotonin.spike(
                        min(0.15, rally * 0.02)
                    )
                    logger.info("🎯 Rally of %d! Serotonin boost.", rally)

            self.current_rally = 0
            return jsonify({'ok': True})

    # ═══════════════════════════════════════════════════════
    # DECIDE — 100% Neural, No Teacher
    # ═══════════════════════════════════════════════════════

    def _decide(self, game_state: dict) -> tuple:
        """
        Choose an action using the policy network.

        NO teacher. NO ball-tracker. The network must figure
        out paddle control entirely from reward signals.
        """
        embedding = self._state_to_embedding(game_state)

        # Compute surprise
        if self._prev_embedding is not None:
            diff = np.linalg.norm(embedding - self._prev_embedding)
            surprise = min(1.0, diff / 2.0)
        else:
            surprise = 0.5
        self._prev_embedding = embedding.copy()

        # ─── Policy network forward pass (NO GRAPH — inference only) ───
        with torch.no_grad():
            state_tensor = torch.from_numpy(embedding).unsqueeze(0)
            logits = self.policy_net(state_tensor)
            probs = torch.softmax(logits, dim=-1)

        # Sample action from distribution
        dist = torch.distributions.Categorical(probs)
        action_idx = dist.sample().item()  # plain int

        self._last_probs = probs.squeeze(0).tolist()

        # ─── Compute step reward (reward shaping) ───
        ball_y = game_state.get('ball_y', 0.5)
        ball_vx = game_state.get('ball_vx', 0.0)
        paddle_center = game_state.get('paddle_center', 0.5)

        distance = abs(ball_y - paddle_center)
        if ball_vx < 0:
            # Ball approaching: strong reward for being close to ball
            step_reward = 0.2 * (1.0 - distance) - 0.05
        else:
            # Ball moving away: mild reward for centering
            step_reward = 0.05 * (1.0 - distance)
        step_reward += 0.005  # tiny survival bonus

        # ─── Store trajectory step (RAW DATA ONLY — no graph tensors) ───
        with self._ep_lock:
            self._ep_states.append(embedding.copy())
            self._ep_actions.append(action_idx)
            self._ep_rewards.append(step_reward)
        self.current_rally += 1

        # ─── Queue for subconscious ───
        if len(self._pending_embeddings) < 20:
            self._pending_embeddings.append(embedding.copy())

        action = ACTION_MAP[action_idx]
        return action, surprise

    # ═══════════════════════════════════════════════════════
    # EPISODE END — REINFORCE + Value Baseline Update
    # ═══════════════════════════════════════════════════════

    def _end_episode(self, terminal_reward: float):
        """
        A point was scored. Train both networks on the episode trajectory.

        Recomputes the full forward pass from stored states/actions.
        No graph tensors cross thread boundaries — fully thread-safe.
        """
        self.episodes += 1

        # ─── Snapshot and clear under lock ───
        with self._ep_lock:
            if len(self._ep_states) == 0:
                return
            ep_states = list(self._ep_states)
            ep_actions = list(self._ep_actions)
            ep_rewards = list(self._ep_rewards)
            self._ep_states.clear()
            self._ep_actions.clear()
            self._ep_rewards.clear()

        T = len(ep_rewards)
        if T == 0:
            return

        # Add terminal reward to the last step
        ep_rewards[-1] += terminal_reward

        # ─── Compute discounted returns ───
        returns = []
        R = 0.0
        for t in reversed(range(T)):
            R = ep_rewards[t] + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float32)

        ep_return = returns[0].item()
        self._return_history.append(ep_return)
        self.avg_return = float(np.mean(self._return_history))

        # ─── Recompute forward pass from raw data (fresh graph) ───
        state_tensors = torch.from_numpy(
            np.array(ep_states, dtype=np.float32)
        )
        action_tensors = torch.tensor(ep_actions, dtype=torch.long)

        # Policy forward pass
        logits = self.policy_net(state_tensors)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(action_tensors)
        entropies = dist.entropy()

        # Value forward pass
        values = self.value_net(state_tensors).squeeze()
        if values.dim() == 0:
            values = values.unsqueeze(0)

        # ─── Advantages ───
        advantages = returns - values.detach()
        if advantages.dim() == 0:
            advantages = advantages.unsqueeze(0)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (
                advantages.std() + 1e-8
            )

        # ─── Policy gradient loss ───
        policy_loss = -(log_probs * advantages).mean()
        # Adaptive entropy floor: force exploration in early training
        min_floor = max(0.1, self._min_entropy_floor * (1.0 - self.training_steps / 500))
        if self.avg_entropy < min_floor:
            # Boost entropy coefficient when policy collapses
            effective_entropy_coef = self.entropy_coef * 3.0
        else:
            effective_entropy_coef = self.entropy_coef
        entropy_bonus = -effective_entropy_coef * entropies.mean()
        total_policy_loss = policy_loss + entropy_bonus

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(), max_norm=0.5
        )
        self.policy_optimizer.step()

        # ─── Value loss (MSE) ───
        # Recompute values (weights changed after policy step)
        values2 = self.value_net(state_tensors).squeeze()
        if values2.dim() == 0:
            values2 = values2.unsqueeze(0)
        value_loss = nn.functional.mse_loss(values2, returns)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.value_net.parameters(), max_norm=0.5
        )
        self.value_optimizer.step()

        # ─── Stats ───
        self.training_steps += 1
        self.policy_loss = total_policy_loss.item()
        self.value_loss = value_loss.item()
        self.avg_entropy = entropies.mean().item()

        if self.episodes % 10 == 0:
            logger.info(
                "🧠 RL Episode %d: return=%.3f policy_loss=%.4f "
                "value_loss=%.4f entropy=%.3f score=%d-%d max_rally=%d",
                self.episodes, ep_return, self.policy_loss,
                self.value_loss, self.avg_entropy,
                self.genesis_scores, self.cpu_scores, self.max_rally,
            )

    # ═══════════════════════════════════════════════════════
    # STATE EMBEDDING — Raw game state to 64-dim vector
    # ═══════════════════════════════════════════════════════

    def _state_to_embedding(self, game_state: dict) -> np.ndarray:
        """Convert Pong game state to a 64-dim embedding."""
        emb = np.zeros(64, dtype=np.float32)

        # Ball state [0:8]
        emb[0] = game_state.get('ball_x', 0.5)
        emb[1] = game_state.get('ball_y', 0.5)
        emb[2] = game_state.get('ball_vx', 0.0)
        emb[3] = game_state.get('ball_vy', 0.0)
        emb[4] = np.sqrt(emb[2]**2 + emb[3]**2)
        emb[5] = emb[0] - self._prev_ball_x
        emb[6] = emb[1] - self._prev_ball_y
        emb[7] = emb[0]
        self._prev_ball_x = float(emb[0])

        # Genesis paddle [8:16]
        emb[8] = game_state.get('paddle_y', 0.5)
        emb[9] = game_state.get('paddle_center', 0.5)
        action_code = {'UP': 1.0, 'DOWN': -1.0, 'STAY': 0.0}
        emb[10] = action_code.get(self.last_action, 0.0)
        emb[11] = game_state.get('genesis_score', 0) / 10.0
        emb[12] = emb[1] - emb[9]  # ball_y - paddle_center (key signal!)

        # CPU paddle [16:24]
        emb[16] = game_state.get('cpu_y', 0.5)
        emb[17] = game_state.get('cpu_score', 0) / 10.0

        # Game context [24:32]
        rally = game_state.get('rally', 0)
        emb[24] = rally / 20.0
        emb[25] = (self.genesis_scores - self.cpu_scores) / 10.0
        emb[26] = min(1.0, (time.time() - self.start_time) / 300)
        emb[27] = self.decisions / 1000.0

        # Trajectory prediction [32:48]
        bx, by = float(emb[0]), float(emb[1])
        bvx, bvy = float(emb[2]), float(emb[3])
        for i in range(8):
            bx += bvx * 0.1
            by += bvy * 0.1
            if by <= 0.02 or by >= 0.98:
                bvy = -bvy
            emb[32 + i * 2] = bx
            emb[33 + i * 2] = by

        # Neurochemistry [48:64]
        nc = self.mind.neurochemistry
        emb[48] = nc.dopamine.level
        emb[49] = nc.cortisol.level
        emb[50] = nc.serotonin.level
        emb[51] = nc.oxytocin.level
        emb[52] = abs(nc.dopamine.level - nc.cortisol.level)

        self._prev_ball_y = float(emb[1])
        return emb

    # ═══════════════════════════════════════════════════════
    # BRAIN INTEGRATION — Real subconscious processing
    # ═══════════════════════════════════════════════════════

    def _brain_integration_loop(self):
        """Feed game states into the subconscious cascade."""
        logger.info("Pong brain integration thread started")
        while self._running:
            try:
                if not self._pending_embeddings:
                    time.sleep(0.5)
                    continue

                embedding = self._pending_embeddings.popleft()
                context_vec = self.mind.proprioception.get_context_vector()
                try:
                    self.mind.subconscious.process_experience(
                        visual_embedding=embedding,
                        text_embedding=None,
                        context=context_vec,
                        emotional_intensity=float(self.mind.neurochemistry.cortisol.level),
                        drive_hunger=float(self.mind.neurochemistry.dopamine.level),
                        train=False,
                    )
                except Exception as integrate_err:
                    logger.warning("Benign integration skip: %s", integrate_err)
                time.sleep(0.2)
            except Exception as e:
                logger.error("Pong brain integration error: %s", e)
                time.sleep(1.0)

    # ═══════════════════════════════════════════════════════
    # START / STATUS
    # ═══════════════════════════════════════════════════════

    def start(self, open_browser: bool = True):
        """Start the Pong server and brain integration."""
        def run_server():
            import logging as _logging
            log = _logging.getLogger('werkzeug')
            log.setLevel(_logging.WARNING)
            self.app.run(host='0.0.0.0', port=self.port, threaded=True)

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        self._brain_thread.start()

        url = f"http://localhost:{self.port}"
        logger.info("Pong RL server running at %s", url)
        print(f"\n  🏓 GENESIS PONG running at: {url}")
        print(f"  ⚡ PURE REINFORCEMENT LEARNING — No teacher, no ball-tracker")
        print(f"  ⚡ Genesis starts RANDOM and must learn from reward alone")
        print(f"  ⚡ Watch the action probabilities converge!\n")

        if open_browser:
            time.sleep(0.5)
            webbrowser.open(url)

    def get_status(self) -> dict:
        """Get current game statistics."""
        elapsed = time.time() - self.start_time
        return {
            'genesis_scores': self.genesis_scores,
            'cpu_scores': self.cpu_scores,
            'decisions': self.decisions,
            'episodes': self.episodes,
            'max_rally': self.max_rally,
            'total_rallies': self.total_rallies,
            'elapsed_sec': round(elapsed, 1),
            'decisions_per_sec': round(self.decisions / max(1, elapsed), 1),
            'training_steps': self.training_steps,
            'policy_loss': round(self.policy_loss, 5),
            'value_loss': round(self.value_loss, 5),
            'avg_entropy': round(self.avg_entropy, 4),
            'avg_return': round(self.avg_return, 4),
        }
