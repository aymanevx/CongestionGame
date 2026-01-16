import random
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any


# Actions discrètes (DQN-friendly)
STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4


@dataclass(frozen=True)
class EnvConfig:
    size: int = 6
    n_agents: int = 4
    max_steps: int = 60
    base_cost: float = 1.0
    goal_reward: float = 20.0
    seed: Optional[int] = None


class GridCongestionEnv:
    """
    Congestion game sur grille NxN (défaut 6x6), multi-agent.

    - n_agents agents se déplacent simultanément.
    - Goal commun aléatoire à chaque épisode.
    - Starts aléatoires, tous différents et != goal.
    - Pas de collision: plusieurs agents peuvent être sur la même case.
    - Congestion sur les ARÊTES: si k agents prennent la même arête au même step,
      le coût de step pour ces agents est multiplié par k.
    - Actions discrètes: 0 stay, 1 up, 2 down, 3 left, 4 right.
    """

    def __init__(self, config: EnvConfig = EnvConfig()):
        self.cfg = config
        self.rng = random.Random(config.seed)

        self.size = self.cfg.size
        self.n_agents = self.cfg.n_agents
        self.n_nodes = self.size * self.size

        # dynamiques
        self.t: int = 0
        self.goal: int = -1
        self.starts: List[int] = []
        self.pos: List[int] = []
        self.cost: List[float] = []
        self.arrived: List[bool] = []

    # ---------- utils index/coords ----------
    def to_rc(self, node: int) -> Tuple[int, int]:
        return divmod(node, self.size)

    def to_node(self, r: int, c: int) -> int:
        return r * self.size + c

    @staticmethod
    def _edge(u: int, v: int) -> Tuple[int, int]:
        return (u, v) if u < v else (v, u)

    # ---------- sampling ----------
    def _sample_goal_and_starts(self) -> Tuple[int, List[int]]:
        if self.n_nodes - 1 < self.n_agents:
            raise ValueError("Not enough cells for unique starts != goal.")

        nodes = list(range(self.n_nodes))
        goal = self.rng.choice(nodes)

        candidates = [x for x in nodes if x != goal]
        starts = self.rng.sample(candidates, k=self.n_agents)
        return goal, starts

    # ---------- observation ----------
    def _obs_agent(self, i: int) -> Dict[str, Any]:
        return {
            "self_pos": self.pos[i],
            "goal": self.goal,
            "arrived": self.arrived[i],
            "t": self.t,
            # IMPORTANT: positions des autres agents
            "others_pos": tuple(self.pos[j] for j in range(self.n_agents) if j != i),
        }


    def _obs_global(self) -> Dict[str, Any]:
        return {
            "pos": tuple(self.pos),
            "goal": self.goal,
            "arrived": tuple(self.arrived),
            "t": self.t,
        }

    # ---------- API ----------
    def reset(self, *, seed: Optional[int] = None, global_obs: bool = False):
        if seed is not None:
            self.rng.seed(seed)

        self.t = 0
        self.goal, self.starts = self._sample_goal_and_starts()
        self.pos = list(self.starts)
        self.cost = [0.0] * self.n_agents
        self.arrived = [False] * self.n_agents

        if global_obs:
            return self._obs_global()
        return [self._obs_agent(i) for i in range(self.n_agents)]

    def step(self, actions: List[int], *, global_obs: bool = False):
        """
        actions: liste d'actions discrètes (len == n_agents)
        returns: obs, rewards, terminated, truncated, info
        """
        if len(actions) != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} actions, got {len(actions)}.")

        self.t += 1

        new_pos = list(self.pos)
        edges = [None] * self.n_agents  # type: ignore

        # 1) apply moves
        for i in range(self.n_agents):
            if self.arrived[i]:
                continue

            u = self.pos[i]
            r, c = self.to_rc(u)

            a = actions[i]
            nr, nc = r, c

            if a == UP:
                nr = max(0, r - 1)
            elif a == DOWN:
                nr = min(self.size - 1, r + 1)
            elif a == LEFT:
                nc = max(0, c - 1)
            elif a == RIGHT:
                nc = min(self.size - 1, c + 1)
            elif a == STAY:
                pass
            else:
                # action invalide -> stay (robuste)
                nr, nc = r, c

            v = self.to_node(nr, nc)
            new_pos[i] = v

            if v != u:  # uniquement si déplacement
                edges[i] = self._edge(u, v)

        # 2) congestion on edges
        edge_counts = Counter(e for e in edges if e is not None)

        # 3) rewards/costs
        rewards = [0.0] * self.n_agents

        for i in range(self.n_agents):
            if self.arrived[i]:
                rewards[i] = 0.0
                continue

            step_cost = self.cfg.base_cost
            if edges[i] is not None:
                step_cost *= edge_counts[edges[i]]

            self.cost[i] += step_cost
            r = -step_cost

            if new_pos[i] == self.goal:
                self.arrived[i] = True
                r += self.cfg.goal_reward

            rewards[i] = r

        self.pos = new_pos

        terminated = all(self.arrived)
        truncated = (self.t >= self.cfg.max_steps)

        info = {
            "t": self.t,
            "edge_counts": edge_counts,
            "cost": tuple(self.cost),
        }

        if global_obs:
            obs = self._obs_global()
        else:
            obs = [self._obs_agent(i) for i in range(self.n_agents)]

        return obs, rewards, terminated, truncated, info

    # ---------- helpers for agents ----------
    def valid_actions(self, i: int) -> List[int]:
        """Actions valides depuis la position courante de l'agent i."""
        if self.arrived[i]:
            return [STAY]

        u = self.pos[i]
        r, c = self.to_rc(u)
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
        """Policy random, utile pour tester."""
        return [self.rng.choice(self.valid_actions(i)) for i in range(self.n_agents)]

    def render_ascii(self) -> str:
        """Debug simple en ASCII."""
        grid = [["." for _ in range(self.size)] for _ in range(self.size)]
        gr, gc = self.to_rc(self.goal)
        grid[gr][gc] = "G"

        for i, p in enumerate(self.pos):
            r, c = self.to_rc(p)
            ch = str(i)
            # si déjà quelque chose (ex: G), on marque *
            grid[r][c] = ch if grid[r][c] == "." else "*"

        return "\n".join(" ".join(row) for row in grid)


# ----------------- TEST -----------------
if __name__ == "__main__":
    env = GridCongestionEnv(EnvConfig(seed=None, max_steps=30))
    obs = env.reset(global_obs=False)
    print("RESET obs[0]:", obs[0])
    print(env.render_ascii(), "\n")

    done = False
    while not done:
        actions = env.sample_actions()
        obs, rewards, terminated, truncated, info = env.step(actions, global_obs=False)
        done = terminated or truncated

        print(
            f"t={info['t']} pos={[o['self_pos'] for o in obs]} goal={env.goal} "
            f"rewards={tuple(round(r,1) for r in rewards)} "
            f"arrived={tuple(o['arrived'] for o in obs)}"
        )
