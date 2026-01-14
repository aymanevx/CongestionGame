import random
from collections import Counter

class GridEnv6x6:
    def __init__(self, seed=None, size=6, n_agents=4, max_steps=60,
                 goal_reward=20.0, step_penalty=1.0, collision_penalty=5.0):
        self.rng = random.Random(seed)
        self.size = size
        self.n_agents = n_agents
        self.max_steps = max_steps

        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.collision_penalty = collision_penalty

        self.roads = self._build_roads()

        # Exemples (tu peux changer)
        self.starts = [0, 5, 30, 35]
        self.goals  = [35, 30, 5, 0]

        self.t = 0
        self.pos = None
        self.arrived = None  # True si agent déjà arrivé
        self.cost = None     # coût cumulé (utile pour analyses)

    def _build_roads(self):
        roads = {}
        for r in range(self.size):
            for c in range(self.size):
                u = r * self.size + c
                neigh = []
                if r > 0: neigh.append((r - 1) * self.size + c)
                if r < self.size - 1: neigh.append((r + 1) * self.size + c)
                if c > 0: neigh.append(r * self.size + (c - 1))
                if c < self.size - 1: neigh.append(r * self.size + (c + 1))
                roads[u] = neigh
        return roads

    @staticmethod
    def _edge(u, v):
        return (u, v) if u < v else (v, u)

    def reset(self):
        self.t = 0
        self.pos = list(self.starts)
        self.arrived = [False for _ in range(self.n_agents)]
        self.cost = [0 for _ in range(self.n_agents)]
        return self._get_obs()

    def _get_obs(self):
        # obs simple (compatible vizu): pos/goals/cost/arrived
        return {
            "pos": tuple(self.pos),
            "goals": tuple(self.goals),
            "cost": tuple(self.cost),
            "arrived": tuple(self.arrived),
        }

    def step(self, dests, learner_idx=0):
        """
        dests: liste de destinations (len=4).
        learner_idx: quel agent on considère comme 'celui qu'on entraîne'
                    (utile pour arrêter l'épisode quand il atteint son goal)
        """
        self.t += 1

        rewards = [0.0 for _ in range(self.n_agents)]

        # 1) Propositions de mouvement (les agents arrivés ne bougent plus)
        proposed_pos = list(self.pos)
        edges = [None] * self.n_agents

        for i in range(self.n_agents):
            if self.arrived[i]:
                proposed_pos[i] = self.pos[i]
                edges[i] = None
                continue

            u = self.pos[i]
            v = dests[i]
            if v in self.roads[u]:
                proposed_pos[i] = v
                edges[i] = self._edge(u, v)
            else:
                proposed_pos[i] = u
                edges[i] = None

        # 2) Collisions: plusieurs agents veulent la même case => annulation
        counts_pos = Counter(proposed_pos)
        collisions = [False] * self.n_agents
        for i in range(self.n_agents):
            if self.arrived[i]:
                continue
            if counts_pos[proposed_pos[i]] > 1:
                collisions[i] = True
                proposed_pos[i] = self.pos[i]
                edges[i] = None

        # 3) Congestion sur arêtes (après annulation collisions)
        edge_counts = Counter([e for e in edges if e is not None])

        # 4) Calcul rewards + coûts + appliquer mouvements
        for i in range(self.n_agents):
            if self.arrived[i]:
                rewards[i] = 0.0
                continue

            r = -self.step_penalty  # pénalité de temps

            if collisions[i]:
                r -= self.collision_penalty

            if edges[i] is not None:
                k = edge_counts[edges[i]]  # nb d'agents sur la même arête
                # pénalité congestion (plus k est grand, plus c'est mauvais)
                r -= float(k)
                self.cost[i] += 1 * k
            else:
                # mouvement annulé/invalide => coût minimal
                self.cost[i] += 1

            # appliquer position
            self.pos[i] = proposed_pos[i]

            # goal atteint ?
            if self.pos[i] == self.goals[i]:
                self.arrived[i] = True
                r += self.goal_reward

            rewards[i] = r

        # 5) Done: on stoppe quand le learner arrive OU max_steps
        done = self.arrived[learner_idx] or (self.t >= self.max_steps)

        obs = self._get_obs()
        info = {
            "t": self.t,
            "collisions": tuple(collisions),
            "edge_counts": edge_counts,
            "arrived": tuple(self.arrived),
        }
        return obs, rewards, done, info

    def auto_step(self):
        # random pour tous les agents non arrivés
        dests = []
        for i in range(self.n_agents):
            u = self.pos[i]
            dests.append(self.rng.choice(self.roads[u]))
        return dests

if __name__ == "__main__":
    print("=== TEST ENV 6x6 (agents random) ===")

    env = GridEnv6x6(seed=42, max_steps=20)
    obs = env.reset()
    print("reset:", obs)

    done = False
    while not done:
        dests = env.auto_step()
        obs, rewards, done, info = env.step(dests, learner_idx=0)

        print(
            f"t={info['t']} "
            f"pos={obs['pos']} "
            f"rewards={tuple(round(r,1) for r in rewards)} "
            f"arrived={obs['arrived']} "
            f"collisions={info['collisions']} "
            f"done={done}"
        )

    print("=== FIN TEST ===")
