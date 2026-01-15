import random
from collections import Counter

class GridEnv6x6:
    """
    Congestion game sur une grille 6x6.
    - 4 agents se déplacent simultanément.
    - Tous ont le MEME goal (objectif commun).
    - Pas de collision: plusieurs agents peuvent être sur la même case.
    - Congestion sur les ARÊTES: si k agents prennent la même arête au même step,
      le coût du step (pour ceux-là) augmente (k).
    """

    def __init__(self, seed=None, size=6, n_agents=4, max_steps=60,
                 base_cost=1.0, goal_reward=0.0, goal_node=35):
        self.rng = random.Random(seed)
        self.size = size
        self.n_agents = n_agents
        self.max_steps = max_steps

        self.base_cost = float(base_cost)
        self.goal_reward = float(goal_reward)  # optionnel (souvent 0 dans congestion game pur)
        self.goal = int(goal_node)

        self.roads = self._build_roads()

        # starts (exemple) — tu peux changer
        self.starts = [0, 5, 30, 34]

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
                if r > 0: neigh.append((r - 1) * self.size + c)          # up
                if r < self.size - 1: neigh.append((r + 1) * self.size + c)  # down
                if c > 0: neigh.append(r * self.size + (c - 1))          # left
                if c < self.size - 1: neigh.append(r * self.size + (c + 1))  # right
                roads[u] = neigh
        return roads

    @staticmethod
    def _edge(u, v):
        # arête non orientée (u-v == v-u)
        return (u, v) if u < v else (v, u)

    def reset(self):
        self.t = 0
        self.pos = list(self.starts)
        self.cost = [0.0 for _ in range(self.n_agents)]
        self.arrived = [False for _ in range(self.n_agents)]
        return self._get_obs()

    def _get_obs(self):
        return {
            "pos": tuple(self.pos),
            "goal": self.goal,
            "cost": tuple(self.cost),
            "arrived": tuple(self.arrived),
        }

    def step(self, dests):
        """
        dests: liste (len=4) de destinations proposées.
        Retour: obs, rewards, done, info
        rewards = -coût (minimiser coût <=> maximiser reward)
        """
        self.t += 1

        # 1) appliquer mouvements (pas de collision, donc on accepte tous)
        edges = [None] * self.n_agents
        new_pos = list(self.pos)

        for i in range(self.n_agents):
            if self.arrived[i]:
                # agent déjà arrivé: il reste au goal
                new_pos[i] = self.pos[i]
                edges[i] = None
                continue

            u = self.pos[i]
            v = dests[i]

            if v in self.roads[u]:
                new_pos[i] = v
                edges[i] = self._edge(u, v)
            else:
                # mouvement invalide => reste
                new_pos[i] = u
                edges[i] = None

        # 2) congestion sur arêtes
        edge_counts = Counter([e for e in edges if e is not None])

        rewards = [0.0 for _ in range(self.n_agents)]

        # 3) coûts + rewards
        for i in range(self.n_agents):
            if self.arrived[i]:
                rewards[i] = 0.0
                continue

            step_cost = self.base_cost  # coût minimal (temps / effort)
            if edges[i] is not None:
                k = edge_counts[edges[i]]     # nb d'agents sur la même arête
                step_cost = self.base_cost * k

            self.cost[i] += step_cost

            # reward = -coût (logique RL)
            r = -step_cost

            # goal atteint ?
            if new_pos[i] == self.goal:
                self.arrived[i] = True
                r += self.goal_reward  # souvent 0 en congestion game pur

            rewards[i] = r

        # appliquer positions
        self.pos = new_pos

        # done si tous arrivés OU max_steps
        done = all(self.arrived) or (self.t >= self.max_steps)

        obs = self._get_obs()
        info = {"t": self.t, "edge_counts": edge_counts}
        return obs, rewards, done, info

    def auto_step(self):
        """Actions random: chaque agent choisit un voisin au hasard."""
        dests = []
        for i in range(self.n_agents):
            u = self.pos[i]
            dests.append(self.rng.choice(self.roads[u]))
        return dests


if __name__ == "__main__":
    print("=== TEST ENV 6x6 (même goal, pas de collision) ===")
    env = GridEnv6x6(seed=42, max_steps=15, goal_node=35, goal_reward=0.0)
    obs = env.reset()
    print("reset:", obs)

    done = False
    while not done:
        dests = env.auto_step()
        obs, rewards, done, info = env.step(dests)
        print(
            f"t={info['t']} pos={obs['pos']} "
            f"cost={tuple(round(c,1) for c in obs['cost'])} "
            f"rewards={tuple(round(r,1) for r in rewards)} "
            f"arrived={obs['arrived']} done={done}"
        )
    print("=== FIN TEST ===")
