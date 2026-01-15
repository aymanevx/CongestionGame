import random
from collections import Counter

class GridEnv6x6:
    """
    Congestion game sur une grille 6x6.
    - 4 agents se déplacent simultanément.
    - Goal commun aléatoire à chaque épisode (reset).
    - Starts aléatoires à chaque épisode (reset).
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

        # Starts/goal seront tirés au reset
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
                if r > 0: neigh.append((r - 1) * self.size + c)              # up
                if r < self.size - 1: neigh.append((r + 1) * self.size + c)  # down
                if c > 0: neigh.append(r * self.size + (c - 1))              # left
                if c < self.size - 1: neigh.append(r * self.size + (c + 1))  # right
                roads[u] = neigh
        return roads

    @staticmethod
    def _edge(u, v):
        # arête non orientée (u-v == v-u)
        return (u, v) if u < v else (v, u)

    def _sample_goal_and_starts(self):
        """Tire un goal commun et des starts aléatoires (différents du goal)."""
        goal = self.rng.choice(self.nodes)

        # On évite le goal pour les starts
        candidates = [x for x in self.nodes if x != goal]

        # Starts différents (recommandé)
        if self.n_agents <= len(candidates):
            starts = self.rng.sample(candidates, k=self.n_agents)
        else:
            starts = [self.rng.choice(candidates) for _ in range(self.n_agents)]

        return goal, starts

    def reset(self):
        self.t = 0

        # Goal + starts aléatoires à chaque épisode
        self.goal, self.starts = self._sample_goal_and_starts()

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
            "starts": tuple(self.starts),
        }

    def step(self, dests):
        """
        dests: liste (len=4) de destinations proposées.
        Retour: obs, rewards, done, info
        rewards = -coût (minimiser coût <=> maximiser reward)
        """
        self.t += 1

        # Pour savoir qui arrive AUJOURD'HUI
        newly_arrived = [False for _ in range(self.n_agents)]

        # 1) appliquer mouvements (pas de collision)
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

            step_cost = self.base_cost
            if edges[i] is not None:
                k = edge_counts[edges[i]]
                step_cost = self.base_cost * k

            self.cost[i] += step_cost

            # reward = -coût
            r = -step_cost

            # goal atteint ?
            if new_pos[i] == self.goal:
                self.arrived[i] = True
                newly_arrived[i] = True
                r += self.goal_reward

            rewards[i] = r

        # appliquer positions
        self.pos = new_pos

        # done si tous arrivés OU max_steps
        done = all(self.arrived) or (self.t >= self.max_steps)

        obs = self._get_obs()
        info = {
            "t": self.t,
            "edge_counts": edge_counts,
            "newly_arrived": tuple(newly_arrived),
        }
        return obs, rewards, done, info

    def auto_step(self):
        """Actions random: chaque agent choisit un voisin au hasard."""
        dests = []
        for i in range(self.n_agents):
            u = self.pos[i]
            dests.append(self.rng.choice(self.roads[u]))
        return dests


if __name__ == "__main__":
    print("=== TEST ENV 6x6 (goal + starts aléatoires, pas de collision) ===")

    # seed=None => ça change à chaque exécution
    env = GridEnv6x6(seed=None, max_steps=30, goal_reward=20.0)

    obs = env.reset()
    print("reset:", obs)

    done = False
    while not done:
        dests = env.auto_step()
        obs, rewards, done, info = env.step(dests)
        print(
            f"t={info['t']} pos={obs['pos']} goal={obs['goal']} "
            f"cost={tuple(round(c,1) for c in obs['cost'])} "
            f"rewards={tuple(round(r,1) for r in rewards)} "
            f"newly_arrived={info['newly_arrived']} "
            f"arrived={obs['arrived']} done={done}"
        )

    print("=== FIN TEST ===")
