import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from env_6x6 import GridEnv6x6


def node_to_rc(node, size):
    """Convertit un noeud (0..35) en (row, col)."""
    return node // size, node % size


def collect_history(env, learner_idx=0):
    """Joue un épisode random et enregistre l'historique des positions."""
    obs = env.reset()
    history = [obs["pos"]]

    done = False
    while not done:
        # auto_step renvoie maintenant les destinations (actions) des agents
        dests = env.auto_step()

        # step renvoie obs, rewards, done, info
        obs, rewards, done, info = env.step(dests, learner_idx=learner_idx)
        history.append(obs["pos"])

    return history


def animate_agents(history, goals, size):
    """Animation matplotlib : positions des agents + goals."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title("Congestion Game 6x6 - Agents (random)")
    ax.set_xlim(-0.5, size - 0.5)
    ax.set_ylim(-0.5, size - 0.5)
    ax.set_xticks(range(size))
    ax.set_yticks(range(size))
    ax.grid(True)

    # Goals en X
    goals_rc = [node_to_rc(g, size) for g in goals]
    gx = [c for r, c in goals_rc]
    gy = [r for r, c in goals_rc]
    ax.scatter(gx, gy, marker="X", s=200)

    # Agents
    scat = ax.scatter([], [], s=250)
    step_text = ax.text(0.02, 1.02, "", transform=ax.transAxes)

    def init():
        scat.set_offsets(np.empty((0, 2)))
        step_text.set_text("")
        return scat, step_text

    def update(frame):
        positions = history[frame]
        pts = []
        for p in positions:
            r, c = node_to_rc(p, size)
            pts.append([c, r])  # x=col, y=row
        scat.set_offsets(np.array(pts))
        step_text.set_text(f"t = {frame}")
        return scat, step_text

    anim = FuncAnimation(fig, update, frames=len(history), init_func=init, interval=400, blit=True)
    plt.show()


def show_heatmap(history, size):
    """Heatmap : nombre de visites par case (tous agents confondus)."""
    heatmap = np.zeros((size, size), dtype=int)

    for positions in history:
        for p in positions:
            r, c = node_to_rc(p, size)
            heatmap[r, c] += 1

    plt.figure(figsize=(8, 6))
    plt.imshow(heatmap, origin="lower", aspect="auto")
    plt.colorbar(label="Nombre de visites (trafic)")
    plt.xticks(ticks=np.arange(size), labels=np.arange(size))
    plt.yticks(ticks=np.arange(size), labels=np.arange(size))
    plt.xlabel("colonne")
    plt.ylabel("ligne")
    plt.title("Heatmap - trafic (4 agents random)")
    plt.show()


def main():
    env = GridEnv6x6(seed=42, max_steps=60)
    size = env.size

    history = collect_history(env, learner_idx=0)

    # Animation (voitures + goals)
    animate_agents(history, env.goals, size)

    # Heatmap (trafic)
    show_heatmap(history, size)


if __name__ == "__main__":
    main()

