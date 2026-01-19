# Congestion Game - Multi-Agent DQN Learning

A reinforcement learning project (Deep Q-Learning) to solve a multi-agent congestion game on a 6x6 grid.

## Project Description

This project implements a congestion game environment where multiple agents (4 by default) must navigate from a starting point to a goal on a grid. The movement cost increases when multiple agents use the same edges simultaneously, creating a congestion dynamic.

### Key Features

- **Multi-agent Environment** : 6x6 grid with N agents (configurable)
- **Congestion Model** : Edge costs increase with power alpha based on the number of agents using them
- **Reinforcement Learning** : Use of Deep Q-Networks (DQN) to train agents
- **Two Training Modes** : 
  - Single agent (single DQN)
  - Two simultaneous agents (one with fixed policy, one learning)
- **Visualization** : Animation of agent trajectories on the grid

## Project Structure

```
CongestionGame/
├── env_6x6.py                              # Congestion game environment
├── train_single_dqn.py                     # Single agent training
├── train_second_agent_with_fixed_agent.py  # Two-agent training
├── visualize_dqn_agent.py                  # Visualization and animation
├── checkpoints_single/                     # Checkpoints for single agent
│   ├── agent0_best.pt
│   ├── agent0_final.pt
│   └── training_meta.json
└── checkpoints_agent1/                     # Checkpoints for agent 1
    ├── agent1_best.pt
    ├── agent1_final.pt
    └── training_meta.json
```

## Installation

### Requirements

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib

### Install Dependencies

```bash
pip install torch numpy matplotlib
```

## Usage

### 1. Train a Single Agent

Trains a DQN agent to navigate alone on the grid towards its goal:

```bash
python train_single_dqn.py
```

Configuration options (modify in code):
- `EnvConfig.n_agents` : Number of agents (default: 4)
- `EnvConfig.max_steps` : Maximum steps per episode (default: 60)
- `EnvConfig.congestion_alpha` : Penalization coefficient (default: 2.5)
- `EnvConfig.goal_reward` : Reward for reaching goal (default: 20.0)

### 2. Train Two Agents (One Fixed, One Learning)

Trains one agent while another agent follows a fixed policy:

```bash
python train_second_agent_with_fixed_agent.py
```

This training tests how an agent adapts to the presence of other agents with predefined behaviors.

### 3. Visualize a Trained Agent

Displays an animation of trajectories and rewards:

```bash
python visualize_dqn_agent.py
```

The visualization shows:
- Grid with agent positions
- Path traveled by each agent
- Target goals
- Evolution of rewards

## Neural Network Architecture

The Q-Network used in DQN is a multi-layer network:

```
Input (obs_dim) → Hidden Layer (128) → ReLU → Hidden Layer (128) → ReLU → Output (5 actions)
```

### Observation Space

The encoded observation contains:
- Self position (normalized)
- Goal position (normalized)
- Other agents positions (normalized)
- Boolean indicating arrival
- Step normalized by max_steps

Total dimension: 2 + 2 + 2*(n_agents-1) + 1 + 1

### Action Space

5 discrete actions:
- 0 : STAY (stay in place)
- 1 : UP (move up)
- 2 : DOWN (move down)
- 3 : LEFT (move left)
- 4 : RIGHT (move right)

## Environment Configuration

### EnvConfig

```python
@dataclass
class EnvConfig:
    size: int = 6                    # Grid size (6x6)
    n_agents: int = 4               # Number of agents
    max_steps: int = 60             # Maximum steps per episode
    base_cost: float = 1.0          # Base cost for movement
    goal_reward: float = 20.0       # Reward for reaching goal
    congestion_alpha: float = 2.5   # Penalization coefficient (2.5 = quadratic)
    seed: Optional[int] = None      # Seed for reproducibility
```

## Congestion Model

The cost of an edge depends on the number of agents k using it simultaneously:

```
Cost = base_cost * k^alpha
```

With alpha = 2.5 by default, the cost increases super-linearly, encouraging agents to diverge to avoid congestion.

## DQN Training

Training hyperparameters:

- Replay buffer size: 10000
- Batch size: 32
- Initial epsilon (exploration): 1.0
- Final epsilon: 0.01
- Epsilon decay: exponential
- Learning rate: 1e-3
- Target network update: every 1000 steps

## Results and Checkpoints

Best models and final models are saved in checkpoint folders:

- `checkpoints_single/` : Models for single agent
- `checkpoints_agent1/` : Models for agent 1 in multi-agent environment

Each folder contains:
- `agent0_best.pt` / `agent1_best.pt` : Best model found
- `agent0_final.pt` / `agent1_final.pt` : Model after complete training
- `training_meta.json` : Training metadata

## Applied Optimizations

- Observation normalization
- Experience replay for better stability
- Separate target network to reduce correlation
- Epsilon-greedy exploration schedule
- Automatic best model saving

## Extension Points

Possibilities for project improvement:

- Add more agents and test scalability
- Implement more advanced algorithms (PPO, A3C)
- Add static obstacles on the grid
- Modify goals dynamically
- Implement multi-agent RL with joint training
- Add real-time performance graphs

## License

Not specified. Please consult the appropriate license for your usage.