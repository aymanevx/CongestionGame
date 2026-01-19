"""
Single-Agent DQN Training Script
=================================

Trains a Deep Q-Network (DQN) agent to navigate a congestion game
where it must reach a goal while minimizing costs from edge congestion.

The agent learns optimal navigation policies against a background of
other agents using greedy goal-seeking strategies.

Key Components:
- QNet: Neural network for Q-value estimation
- ReplayBuffer: Experience buffer for off-policy learning
- Greedy baseline: Other agents use Manhattan distance heuristic
"""

import os
import json
import random
from collections import deque
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from env_6x6 import GridCongestionEnv, EnvConfig


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def plot_training_rewards(episode_rewards: List[float], save_path: str, window_size: int = 100):
    """
    Plot training rewards with moving average.
    
    Args:
        episode_rewards: List of rewards for each episode
        save_path: Directory to save the plot
        window_size: Size of moving average window
    """
    import numpy as np
    
    # Calculate moving average
    rewards_array = np.array(episode_rewards)
    moving_avg = np.convolve(rewards_array, np.ones(window_size)/window_size, mode='valid')
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot 1: Episode rewards with moving average
    ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    ax1.plot(range(window_size-1, len(episode_rewards)), moving_avg, 
             color='orange', linewidth=2, label=f'Moving Avg (window={window_size})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Episode Reward')
    ax1.set_title('Training Rewards Over Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Moving average only (zoomed)
    ax2.plot(range(window_size-1, len(episode_rewards)), moving_avg, 
             color='green', linewidth=2)
    ax2.fill_between(range(window_size-1, len(episode_rewards)), moving_avg, alpha=0.3, color='green')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Average Reward')
    ax2.set_title(f'Moving Average (window={window_size})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(save_path, 'training_rewards.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"\nSaved plot: {plot_file}")
    
    # Also save data to CSV for external analysis
    csv_file = os.path.join(save_path, 'training_rewards.csv')
    with open(csv_file, 'w') as f:
        f.write('episode,reward,moving_avg\n')
        for i, reward in enumerate(episode_rewards):
            moving_avg_val = moving_avg[i - window_size + 1] if i >= window_size - 1 else 'N/A'
            f.write(f'{i+1},{reward},{moving_avg_val}\n')
    print(f"Saved data: {csv_file}")
    
    plt.close()


def obs_to_vec(obs: Dict[str, Any], size: int, max_steps: int) -> torch.Tensor:
    """
    Convert observation dictionary to normalized feature vector.
    
    Observation encoding:
    [self_row, self_col, goal_row, goal_col,
     other1_row, other1_col, other2_row, other2_col, ...,
     arrived_flag, timestep_progress]
    
    All spatial coordinates are normalized to [0, 1] using grid size.
    
    Args:
        obs: Observation dictionary from environment
        size: Grid size (for normalization)
        max_steps: Maximum episode steps (for timestep normalization)
        
    Returns:
        Normalized feature vector as torch.Tensor
    """
    self_pos = int(obs["self_pos"])
    goal = int(obs["goal"])
    others_pos = list(obs["others_pos"])  # Length = n_agents - 1
    arrived = 1.0 if obs["arrived"] else 0.0
    t = int(obs["t"])

    # Convert positions to (row, col) coordinates
    sr, sc = divmod(self_pos, size)
    gr, gc = divmod(goal, size)

    # Normalization factor (avoid division by zero)
    denom = (size - 1) if size > 1 else 1
    
    # Build feature vector
    feats = [
        sr / denom, sc / denom,      # Self position (normalized)
        gr / denom, gc / denom,      # Goal position (normalized)
    ]

    # Add positions of other agents
    for p in others_pos:
        r, c = divmod(int(p), size)
        feats.extend([r / denom, c / denom])

    # Add timestep progress and arrival flag
    tf = t / max_steps if max_steps > 0 else 0.0
    feats.extend([arrived, tf])

    return torch.tensor(feats, dtype=torch.float32)


# ============================================================================
# NEURAL NETWORK ARCHITECTURE
# ============================================================================

class QNet(nn.Module):
    """
    Q-Network for DQN agent.
    
    A simple feedforward neural network that estimates Q-values for each
    possible action given an observation state.
    
    Architecture: 
    Input -> Hidden(ReLU) -> Hidden(ReLU) -> Output(Q-values)
    """
    
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        """
        Initialize Q-network.
        
        Args:
            obs_dim: Observation feature dimension
            n_actions: Number of possible actions (5 for grid movement)
            hidden: Number of neurons in hidden layers
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input feature vector
            
        Returns:
            Q-value estimates for each action
        """
        return self.net(x)


# ============================================================================
# EXPERIENCE REPLAY BUFFER
# ============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    
    Stores transitions (state, action, reward, next_state, done) and
    allows random sampling of mini-batches for gradient updates.
    """
    
    def __init__(self, capacity: int = 100_000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store (uses FIFO eviction)
        """
        self.buf = deque(maxlen=capacity)

    def push(self, s: torch.Tensor, a: int, r: float, s2: torch.Tensor, done: float):
        """
        Add a transition to the buffer.
        
        Args:
            s: Current state
            a: Action taken
            r: Reward received
            s2: Next state
            done: Episode termination flag
        """
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        """
        Sample a mini-batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(s2),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        """Return current buffer size."""
        return len(self.buf)


# ============================================================================
# ACTION SELECTION UTILITY
# ============================================================================

def masked_argmax(qvals: torch.Tensor, valid_actions: List[int]) -> int:
    """
    Select best action from valid actions only.
    
    Finds the action with highest Q-value among those that are valid
    (respecting environment constraints like grid boundaries).
    
    Args:
        qvals: Q-values for all actions
        valid_actions: List of valid action indices
        
    Returns:
        Index of best valid action
    """
    best_a = valid_actions[0]
    best_q = qvals[best_a].item()
    for a in valid_actions[1:]:
        q = qvals[a].item()
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


# ============================================================================
# GREEDY BASELINE POLICY
# ============================================================================

def manhattan(node_a: int, node_b: int, size: int) -> int:
    """
    Calculate Manhattan distance between two nodes on a grid.
    
    Args:
        node_a, node_b: Node indices
        size: Grid size
        
    Returns:
        Manhattan distance
    """
    ar, ac = divmod(node_a, size)
    br, bc = divmod(node_b, size)
    return abs(ar - br) + abs(ac - bc)


def greedy_to_goal_action(env: GridCongestionEnv, agent_i: int) -> int:
    """
    Greedy policy: select action that minimizes Manhattan distance to goal.
    
    This is used by non-trained agents in the environment. When multiple
    actions achieve equal distance, uses a deterministic tie-breaking rule.
    
    Args:
        env: Environment instance
        agent_i: Agent index
        
    Returns:
        Best greedy action
    """
    # Agents that already arrived stay put
    if env.arrived[agent_i]:
        return 0  # STAY

    u = env.pos[agent_i]
    goal = env.goal
    size = env.size

    valid = env.valid_actions(agent_i)

    # Preference order for tie-breaking: prioritize movement over staying
    # (can be adjusted to change agent behavior)
    pref_order = [1, 2, 3, 4, 0]  # UP, DOWN, LEFT, RIGHT, STAY

    ur, uc = divmod(u, size)

    best_a = valid[0]
    best_d = 10**9
    best_rank = 10**9

    # Evaluate each valid action
    for a in valid:
        nr, nc = ur, uc
        if a == 1:        # UP
            nr = max(0, ur - 1)
        elif a == 2:      # DOWN
            nr = min(size - 1, ur + 1)
        elif a == 3:      # LEFT
            nc = max(0, uc - 1)
        elif a == 4:      # RIGHT
            nc = min(size - 1, uc + 1)
        # a == 0: STAY

        v = nr * size + nc
        d = manhattan(v, goal, size)
        rank = pref_order.index(a) if a in pref_order else 999

        # Choose action with minimum distance (or best rank if tied)
        if (d < best_d) or (d == best_d and rank < best_rank):
            best_d = d
            best_rank = rank
            best_a = a

    return best_a


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_single_agent_dqn(
    agent_id: int = 0,
    episodes: int = 3000,
    seed: int = 0,
    save_dir: str = "checkpoints_single",
    device: str = "cpu",
):
    """
    Train a single DQN agent in the multi-agent congestion environment.
    
    The trained agent learns to reach the goal efficiently while the
    other agents use fixed greedy policies. Training uses standard DQN
    with epsilon-greedy exploration and experience replay.
    
    Args:
        agent_id: Which agent to train (typically 0)
        episodes: Number of training episodes
        seed: Random seed
        save_dir: Directory to save checkpoints
        device: Compute device ("cpu" or "cuda")
    """
    os.makedirs(save_dir, exist_ok=True)
    set_seed(seed)

    # ---- Environment Setup ----
    cfg = EnvConfig(
        size=6, 
        n_agents=10, 
        max_steps=60, 
        base_cost=1.0, 
        goal_reward=20.0, 
        seed=seed
    )
    env = GridCongestionEnv(cfg)

    # ---- Network Setup ----
    # Observation dimension: 2(self) + 2(goal) + 2*(n_agents-1) + 1(arrived) + 1(t)
    obs_dim = 2 + 2 + 2 * (cfg.n_agents - 1) + 1 + 1
    n_actions = 5  # STAY, UP, DOWN, LEFT, RIGHT

    q = QNet(obs_dim, n_actions, hidden=128).to(device)
    q_targ = QNet(obs_dim, n_actions, hidden=128).to(device)
    q_targ.load_state_dict(q.state_dict())
    q_targ.eval()

    # ---- Optimizer and Buffer ----
    opt = optim.Adam(q.parameters(), lr=5e-4)
    rb = ReplayBuffer(capacity=100_000)

    # ---- Hyperparameters ----
    gamma = 0.99                    # Discount factor
    batch_size = 256                # Mini-batch size
    learning_starts = 3000          # Steps before training begins
    target_update_every = 500       # Steps between target network updates
    train_every = 1                 # Train every N steps
    max_grad_norm = 10.0            # Gradient clipping threshold

    # Epsilon decay schedule
    eps_start, eps_end = 1.0, 0.05
    eps_decay_steps = 200_000

    # ---- Training Loop ----
    global_step = 0
    returns_window = deque(maxlen=100)  # Rolling window for evaluation
    best_window100 = -1e9
    all_episode_rewards = []  # Store all episode rewards for plotting

    for ep in range(1, episodes + 1):
        obs_list = env.reset(global_obs=False)
        done_global = False
        ep_return = 0.0

        while not done_global:
            # ---- Epsilon-Greedy Exploration ----
            frac = max(0.0, (eps_decay_steps - global_step) / eps_decay_steps)
            eps = eps_end + (eps_start - eps_end) * frac

            agent_already_arrived = bool(obs_list[agent_id]["arrived"])

            # ---- Trained Agent Action Selection ----
            if agent_already_arrived:
                a_train = 0  # STAY
            else:
                s = obs_to_vec(obs_list[agent_id], cfg.size, cfg.max_steps).to(device)
                valid = env.valid_actions(agent_id)

                # Epsilon-greedy: random action or best Q-value
                if random.random() < eps:
                    a_train = random.choice(valid)
                else:
                    with torch.no_grad():
                        qvals = q(s)
                    a_train = masked_argmax(qvals, valid)

            # ---- Other Agents: Greedy Baseline ----
            actions = []
            for i in range(cfg.n_agents):
                if i == agent_id:
                    actions.append(a_train)
                else:
                    actions.append(greedy_to_goal_action(env, i))

            # ---- Environment Step ----
            next_obs_list, rewards, terminated, truncated, info = env.step(actions, global_obs=False)
            done_global = terminated or truncated

            r = float(rewards[agent_id])
            ep_return += r

            # ---- Store Experience (only if agent not already arrived) ----
            if not agent_already_arrived:
                s2 = obs_to_vec(next_obs_list[agent_id], cfg.size, cfg.max_steps).to(device)

                # Episode termination for this agent
                done_agent = float(next_obs_list[agent_id]["arrived"] or truncated)

                rb.push(
                    obs_to_vec(obs_list[agent_id], cfg.size, cfg.max_steps).cpu(),
                    a_train,
                    r,
                    s2.cpu(),
                    done_agent,
                )

            obs_list = next_obs_list
            global_step += 1

            # ---- DQN Training ----
            if global_step > learning_starts and len(rb) >= batch_size and (global_step % train_every == 0):
                # Sample mini-batch
                S, A, R, S2, D = rb.sample(batch_size)
                S = S.to(device)
                A = A.to(device)
                R = R.to(device)
                S2 = S2.to(device)
                D = D.to(device)

                # Compute Q-values for actions taken
                q_sa = q(S).gather(1, A.view(-1, 1)).squeeze(1)

                # Compute target Q-values
                with torch.no_grad():
                    max_next = q_targ(S2).max(dim=1).values
                    target = R + gamma * (1.0 - D) * max_next

                # Loss and optimization step
                loss = nn.functional.smooth_l1_loss(q_sa, target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), max_grad_norm)
                opt.step()

            # ---- Target Network Update ----
            if global_step % target_update_every == 0:
                q_targ.load_state_dict(q.state_dict())

        # ---- Episode Logging ----
        returns_window.append(ep_return)
        all_episode_rewards.append(ep_return)
        if ep % 50 == 0:
            w = sum(returns_window) / len(returns_window)
            print(
                f"ep={ep:5d}  return={ep_return:8.2f}  window100={w:8.2f}  "
                f"eps={eps:5.3f}  steps={global_step}"
            )

            # Save checkpoint if performance improved
            if w > best_window100:
                best_window100 = w
                torch.save(q.state_dict(), os.path.join(save_dir, f"agent{agent_id}_best.pt"))

    # ---- Save Final Model and Metadata ----
    final_path = os.path.join(save_dir, f"agent{agent_id}_final.pt")
    torch.save(q.state_dict(), final_path)

    meta = {
        "agent_id": agent_id,
        "env_config": {
            "size": cfg.size,
            "n_agents": cfg.n_agents,
            "max_steps": cfg.max_steps,
            "base_cost": cfg.base_cost,
            "goal_reward": cfg.goal_reward,
            "seed": cfg.seed,
        },
        "train": {
            "episodes": episodes,
            "seed": seed,
            "gamma": gamma,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "target_update_every": target_update_every,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay_steps": eps_decay_steps,
            "lr": 5e-4,
        },
        "best_window100_return": best_window100,
        "final_checkpoint": final_path,
    }
    with open(os.path.join(save_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved final model: {final_path}")
    print(f"Saved meta: {os.path.join(save_dir, 'training_meta.json')}")
    print(f"Best window100 return: {best_window100:.2f}")
    
    # ---- Plot and save training rewards ----
    plot_training_rewards(all_episode_rewards, save_dir, window_size=100)


if __name__ == "__main__":
    train_single_agent_dqn(
        agent_id=0,
        episodes=20000,
        seed=0,
        save_dir="checkpoints_single",
        device="cpu",  # Use "cuda" if GPU available
    )
