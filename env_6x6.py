import random
from collections import Counter

class GridEnv6x6:
    """
    Congestion game sur une grille 6x6.
    - 4 agents se déplacent simultanément.
    - Goal commun aléatoire à chaque épisode (reset).
    - Starts aléatoires, tous différents et != goal.
    - Pas de collision: plusieurs agents peuvent être sur la même case.
    - Congestion sur les ARÊTES: si k agents prennent la même arête au même step,
      le coût du step (pour ceux-là) augmente (k).
    """

    def __init__(self, seed=None, size=6, n_agents=4, max_steps=60,
                 base_cost=1.0, goal_reward=20.0):
        self.rng = random.Random(seed)
        self.size = size
        self.n_agents = n_agents
        self.max_steps = max_steps

        self.base_cost = float(base_cost)
        self.goal_reward = float(goal_reward)

        self.roads = self._build_roads()

        # Nombre total de cases
        self.n_nodes = self.size * self.size
        self.nodes = list(range(self.n_nodes))

        # Variables dynamiques
        self.starts = None
        self.goal = None
        self.t = 0
        self.pos = None
        self.cost = None
        self.arrived = None

    def _build_roads(self):
        roads = {}
        for r in range(self.size):
            for c in range(self.size):
                u = r * self.size + c
                neigh = []
                if r > 0:
                    neigh.append((r - 1) * self.size + c)      # up
                if r < self.size - 1:
                    neigh.append((r + 1) * self.size + c)      # down
                if c > 0:
                    neigh.append(r * self.size + (c - 1))      # left
                if c < self.size - 1:
                    neigh.append(r * self.size + (c + 1))      # right
                roads[u] = neigh
        return roads

    @staticmethod
    def _edge(u, v):
        # arête non orientée (u-v == v-u)
        return (u, v) if u < v else (v, u)

    def _sample_goal_and_starts(self):
        """Goal commun + starts tous différents et != goal."""
        if self.n_nodes - 1 < self.n_agents:
            raise ValueError("Pas assez de cases pour des starts uniques")

        goal = self.rng.choice(self.nodes)

        candidates = [x for x in self.nodes if x != goal]
        starts = self.rng.sample(candidates, k=self.n_agents)

        return goal, starts

    def reset(self):
        self.t = 0
        self.goal, self.starts = self._sample_goal_and_starts()

        self.pos = list(self.starts)
        self.cost = [0.0 for _ in range(self.n_agents)]
        self.arrived = [False for _ in range(self.n_agents)]

        return self._get_obs()

    def _get_obs(self):
        return {
            "pos": tuple(self.pos),
            "goal": self.goal,
            "starts": tuple(self.starts),
            "cost": tuple(self.cost),
            "arrived": tuple(self.arrived),
        }

    def step(self, dests):
        """
        dests : liste des destinations proposées (len = n_agents)
        Retour : obs, rewards, done, info
        """
        self.t += 1

        edges = [None] * self.n_agents
        new_pos = list(self.pos)

        # 1) Mouvements
        for i in range(self.n_agents):
            if self.arrived[i]:
                new_pos[i] = self.pos[i]
                continue

            u = self.pos[i]
            v = dests[i]

            if v in self.roads[u]:
                new_pos[i] = v
                edges[i] = self._edge(u, v)
            else:
                new_pos[i] = u

        # 2) Congestion
        edge_counts = Counter(e for e in edges if e is not None)

        rewards = [0.0 for _ in range(self.n_agents)]

        # 3) Coûts + rewards
        for i in range(self.n_agents):
            if self.arrived[i]:
                continue

            step_cost = self.base_cost
            if edges[i] is not None:
                step_cost *= edge_counts[edges[i]]

            self.cost[i] += step_cost
            r = -step_cost

            if new_pos[i] == self.goal:
                self.arrived[i] = True
                r += self.goal_reward

            rewards[i] = r

        self.pos = new_pos
        done = all(self.arrived) or self.t >= self.max_steps

        info = {
            "t": self.t,
            "edge_counts": edge_counts,
        }

        return self._get_obs(), rewards, done, info

    def auto_step(self):
        """Actions aléatoires"""
        return [self.rng.choice(self.roads[p]) for p in self.pos]


# ----------------- TEST -----------------
if __name__ == "__main__":
    print("=== TEST ENV 6x6 FINAL ===")
    env = GridEnv6x6(seed=None, max_steps=30, goal_reward=20.0)

    obs = env.reset()
    print("RESET:", obs)

    done = False
    while not done:
        dests = env.auto_step()
        obs, rewards, done, info = env.step(dests)
        print(
            f"t={info['t']} pos={obs['pos']} goal={obs['goal']} "
            f"rewards={tuple(round(r,1) for r in rewards)} "
            f"arrived={obs['arrived']}"
        )

    print("=== FIN ===")


