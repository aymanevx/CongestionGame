import random

class SquareEnv2:
    # 4 endroits: 0-1
    #            | |
    #            3-2
    def __init__(self, seed=None, start=1, goal=2):
        self.rng = random.Random(seed)
        self.roads = {
            0: [1, 3],
            1: [0, 2],
            2: [1, 3],
            3: [0, 2],
        }
        self.start = start
        self.goal = goal
        self.pos = [start, start]   # positions des 2 agents: [A, B]
        self.cost = [0, 0]          # coût cumulé de A et B

    def reset(self):
        self.pos = [self.start, self.start]
        self.cost = [0, 0]
        return {"pos": tuple(self.pos), "goal": self.goal, "cost": tuple(self.cost)}

    def _auto_dest(self, agent_idx):
        """Même règle qu'avant, mais pour un agent donné."""
        p = self.pos[agent_idx]
        neighbors = self.roads[p]
        if self.goal in neighbors:
            return self.goal
        return self.rng.choice(neighbors)

    @staticmethod
    def _edge(u, v):
        """Représentation non orientée d'une arête (u-v == v-u)."""
        return (u, v) if u < v else (v, u)

    def step(self, dests):
        """
        dests = [dest_A, dest_B]
        Pas simultané.
        """
        # coût de base: 1 par agent
        step_cost = 1

        # arêtes demandées (si action invalide => l'agent reste sur place)
        edges = []
        for i in (0, 1):
            u = self.pos[i]
            v = dests[i]
            if v in self.roads[u]:
                edges.append(self._edge(u, v))
            else:
                edges.append(None)  # pas de mouvement valide

        # congestion: même arête utilisée par les deux => coût 2 au lieu de 1
        if edges[0] is not None and edges[0] == edges[1]:
            step_cost = 2

        # appliquer le coût
        self.cost[0] += step_cost
        self.cost[1] += step_cost

        # appliquer les mouvements valides
        for i in (0, 1):
            u = self.pos[i]
            v = dests[i]
            if v in self.roads[u]:
                self.pos[i] = v

        done = (self.pos[0] == self.goal and self.pos[1] == self.goal)
        obs = {"pos": tuple(self.pos), "goal": self.goal, "cost": tuple(self.cost)}
        return obs, done

    def auto_step(self):
        """Choisit automatiquement une action pour chaque agent puis fait un step."""
        dests = [self._auto_dest(0), self._auto_dest(1)]
        return self.step(dests)


# --- Exemple d'usage ---
env = SquareEnv2(seed=None, start=0, goal=2)

obs = env.reset()
print("reset:", obs)

done = False
t = 0
while not done and t < 20:
    obs, done = env.auto_step()
    t += 1
    print(f"t={t} pos={obs['pos']} cost={obs['cost']} done={done}")
