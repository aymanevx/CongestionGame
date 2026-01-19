"""
Congestion Game Environment on NxN Grid
========================================

A multi-agent reinforcement learning environment where agents navigate a grid
and must reach a common goal, incurring increasing costs when congestion occurs.

Key Features:
- Congestion modeled on edges (paths between grid cells)
- Super-linear cost penalty based on edge congestion
- Support for multiple agents with individual observations
"""

import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


# ============================================================================
# ACTION DEFINITIONS
# ============================================================================
# Discrete action space compatible with DQN training
STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4

ACTION_NAMES = {0: "STAY", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT"}


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================
@dataclass(frozen=True)
class EnvConfig:
    """
    Configuration for the GridCongestionEnv.
    
    Attributes:
        size: Grid dimension (size x size)
        n_agents: Number of agents in the environment
        max_steps: Maximum steps per episode
        base_cost: Base cost for each movement (before congestion penalty)
        goal_reward: Reward bonus when reaching the goal
        seed: Random seed for reproducibility
        congestion_alpha: Congestion penalty exponent (1.0=linear, 2.0=quadratic, 2.5=super-linear)
    """
    size: int = 6
    n_agents: int = 4
    max_steps: int = 60
    base_cost: float = 1.0
    goal_reward: float = 20.0
    seed: Optional[int] = None
    # Congestion cost formula: base_cost * (num_agents_on_edge ** congestion_alpha)
    congestion_alpha: float = 2.5

# ============================================================================
# GRID CONGESTION ENVIRONMENT CLASS
# ============================================================================
class GridCongestionEnv:
    """
    Multi-agent congestion game on an NxN grid.
    
    Agents start at random positions and must collectively reach a goal,
    competing for path resources. When multiple agents use the same edge
    in one step, they incur a super-linear cost penalty.
    
    Edge congestion model:
    - If k agents move along the same edge in step t, each pays:
      cost = base_cost * (k ** congestion_alpha)
    """

    def __init__(self, config: EnvConfig = EnvConfig()):
        """
        Initialize the environment with given configuration.
        
        Args:
            config: EnvConfig object containing environment parameters
        """
        self.cfg = config
        self.rng = random.Random(config.seed)

        self.size = self.cfg.size
        self.n_agents = self.cfg.n_agents
        self.n_nodes = self.size * self.size

        # Episode state
        self.t: int = 0                          # Current timestep
        self.goal: int = -1                      # Goal node index
        self.starts: List[int] = []              # Starting positions for each agent
        self.pos: List[int] = []                 # Current position of each agent
        self.cost: List[float] = []              # Accumulated cost for each agent
        self.arrived: List[bool] = []            # Whether each agent has reached the goal

    # ========================================================================
    # UTILITY METHODS - Coordinate Transformations
    # ========================================================================
    
    def to_rc(self, node: int) -> Tuple[int, int]:
        """
        Convert node index to (row, column) coordinates.
        
        Args:
            node: Index in range [0, size*size)
            
        Returns:
            Tuple of (row, column)
        """
        return divmod(node, self.size)

    def to_node(self, r: int, c: int) -> int:
        """
        Convert (row, column) coordinates to node index.
        
        Args:
            r: Row coordinate
            c: Column coordinate
            
        Returns:
            Node index
        """
        return r * self.size + c

    @staticmethod
    def _edge(u: int, v: int) -> Tuple[int, int]:
        """
        Create a canonical (undirected) edge representation.
        Ensures consistent representation regardless of direction: edge(3,5) == edge(5,3)
        
        Args:
            u, v: Node indices
            
        Returns:
            Tuple (min_node, max_node) representing the edge
        """
        return (u, v) if u < v else (v, u)

    # ========================================================================
    # ENVIRONMENT INITIALIZATION - Goal and Starting Positions
    # ========================================================================
    
    def _sample_goal_and_starts(self) -> Tuple[int, List[int]]:
        """
        Randomly sample a goal node and distinct starting positions for all agents.
        
        Returns:
            Tuple of (goal_node, list_of_starting_positions)
        """
        nodes = list(range(self.n_nodes))
        goal = self.rng.choice(nodes)
        starts = self.rng.sample(
            [x for x in nodes if x != goal], 
            self.n_agents
        )
        return goal, starts

    # ========================================================================
    # OBSERVATION GENERATION
    # ========================================================================
    
    def _obs_agent(self, i: int) -> Dict[str, Any]:
        """
        Generate agent-specific observation (partially observable).
        
        Args:
            i: Agent index
            
        Returns:
            Dictionary containing:
                - self_pos: This agent's current position
                - goal: Goal node index
                - arrived: Whether this agent reached the goal
                - t: Current timestep
                - others_pos: Tuple of other agents' positions
        """
        return {
            "self_pos": self.pos[i],
            "goal": self.goal,
            "arrived": self.arrived[i],
            "t": self.t,
            "others_pos": tuple(self.pos[j] for j in range(self.n_agents) if j != i),
        }

    def _obs_global(self) -> Dict[str, Any]:
        """
        Generate full global observation (for analysis/visualization).
        
        Returns:
            Dictionary containing global state
        """
        return {
            "pos": tuple(self.pos),
            "goal": self.goal,
            "arrived": tuple(self.arrived),
            "t": self.t,
        }

    # ========================================================================
    # MAIN API - Reset and Step
    # ========================================================================
    
    def reset(self, *, seed: Optional[int] = None, global_obs: bool = False):
        """
        Reset the environment to initial state.
        
        Args:
            seed: Optional new random seed
            global_obs: If True, return global observation; else return list of agent observations
            
        Returns:
            Initial observation(s)
        """
        if seed is not None:
            self.rng.seed(seed)

        self.t = 0
        self.goal, self.starts = self._sample_goal_and_starts()
        self.pos = list(self.starts)
        self.cost = [0.0] * self.n_agents
        self.arrived = [False] * self.n_agents

        return self._obs_global() if global_obs else [self._obs_agent(i) for i in range(self.n_agents)]

    def step(self, actions: List[int], *, global_obs: bool = False):
        """
        Execute one step of the environment with given actions.
        
        Process:
        1. Move agents based on actions (respecting grid boundaries)
        2. Detect edge congestion (multiple agents on same edge)
        3. Calculate movement costs with congestion penalties
        4. Check if agents reached goal
        5. Return observations, rewards, termination flags
        
        Args:
            actions: List of action indices for each agent (length must equal n_agents)
            global_obs: If True, return global observation; else return list of agent observations
            
        Returns:
            Tuple of (observations, rewards, terminated, truncated, info)
                - observations: Current state observations
                - rewards: Reward for each agent (negative costs + goal bonus)
                - terminated: True if all agents reached goal
                - truncated: True if max_steps reached
                - info: Additional information (timestep, edge counts, costs)
        """
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions)}")

        self.t += 1
        new_pos = list(self.pos)
        edges = [None] * self.n_agents

        # ---- PHASE 1: Execute movements ----
        for i in range(self.n_agents):
            # Agents that already arrived stay at goal
            if self.arrived[i]:
                continue

            u = self.pos[i]
            r, c = self.to_rc(u)
            nr, nc = r, c
            a = actions[i]

            # Apply movement based on action
            if a == UP:
                nr = max(0, r - 1)
            elif a == DOWN:
                nr = min(self.size - 1, r + 1)
            elif a == LEFT:
                nc = max(0, c - 1)
            elif a == RIGHT:
                nc = min(self.size - 1, c + 1)
            # a == STAY: nr, nc unchanged

            v = self.to_node(nr, nc)
            new_pos[i] = v
            # Track edge only if agent actually moved
            if v != u:
                edges[i] = self._edge(u, v)

        # ---- PHASE 2: Calculate edge congestion ----
        # Count how many agents use each edge this step
        edge_counts = Counter(e for e in edges if e is not None)

        # ---- PHASE 3: Calculate rewards and costs ----
        rewards = [0.0] * self.n_agents

        for i in range(self.n_agents):
            # Skip agents that already arrived
            if self.arrived[i]:
                continue

            step_cost = self.cfg.base_cost

            # Apply congestion penalty if agent moved
            if edges[i] is not None:
                num_agents_on_edge = edge_counts[edges[i]]
                # Super-linear congestion: cost = base_cost * (k ** alpha)
                step_cost *= (num_agents_on_edge ** self.cfg.congestion_alpha)

            # Accumulate cost
            self.cost[i] += step_cost
            reward = -step_cost  # Negative reward (minimizing cost)

            # Goal bonus
            if new_pos[i] == self.goal:
                self.arrived[i] = True
                reward += self.cfg.goal_reward

            rewards[i] = reward

        self.pos = new_pos

        # ---- PHASE 4: Check termination conditions ----
        terminated = all(self.arrived)
        truncated = (self.t >= self.cfg.max_steps)

        info = {
            "t": self.t,
            "edge_counts": edge_counts,
            "cost": tuple(self.cost),
        }

        obs = self._obs_global() if global_obs else [self._obs_agent(i) for i in range(self.n_agents)]
        return obs, rewards, terminated, truncated, info

    # ========================================================================
    # HELPER METHODS - Valid Actions and Sampling
    # ========================================================================
    
    def valid_actions(self, i: int) -> List[int]:
        """
        Get list of valid actions for agent i (respecting boundary constraints).
        Agents that arrived can only STAY.
        
        Args:
            i: Agent index
            
        Returns:
            List of valid action indices
        """
        if self.arrived[i]:
            return [STAY]

        r, c = self.to_rc(self.pos[i])
        acts = [STAY]
        if r > 0:
            acts.append(UP)
        if r < self.size - 1:
            acts.append(DOWN)
        if c > 0:
            acts.append(LEFT)
        if c < self.size - 1:
            acts.append(RIGHT)
        return acts

    def sample_actions(self) -> List[int]:
        """
        Sample random valid actions for all agents.
        
        Returns:
            List of randomly sampled valid actions
        """
        return [self.rng.choice(self.valid_actions(i)) for i in range(self.n_agents)]

    def render_ascii(self) -> str:
        """
        Render environment as ASCII art.
        
        Legend:
        - G: Goal
        - 0-9: Agent indices
        - *: Multiple agents on same cell
        - .: Empty cell
        
        Returns:
            ASCII representation of current state
        """
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        gr, gc = self.to_rc(self.goal)
        grid[gr][gc] = "G"

        for i, p in enumerate(self.pos):
            r, c = self.to_rc(p)
            # If cell is empty, mark with agent index; otherwise mark as collision
            grid[r][c] = str(i) if grid[r][c] == "." else "*"

        return "\n".join(" ".join(row) for row in grid)





# ============================================================================
# TESTING AND DEMONSTRATION
# ============================================================================
if __name__ == "__main__":
    # Create environment with super-linear congestion penalty
    env = GridCongestionEnv(
        EnvConfig(
            congestion_alpha=2.5,  # Very punitive for congestion
            max_steps=30
        )
    )

    # Initialize episode
    obs = env.reset()
    print(env.render_ascii(), "\n")

    # Run episode with random actions
    done = False
    while not done:
        actions = env.sample_actions()
        obs, rewards, terminated, truncated, info = env.step(actions)
        done = terminated or truncated

        # Print step summary
        print(
            f"t={info['t']:2d} "
            f"pos={[o['self_pos'] for o in obs]:20s} "
            f"rewards={[f'{r:5.1f}' for r in rewards]} "
            f"cost={[f'{c:5.1f}' for c in info['cost']]}"
        )
