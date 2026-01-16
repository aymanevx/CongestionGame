import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

import torch
import torch.nn as nn

from env_6x6 import GridCongestionEnv, EnvConfig


# ----------------------------
# Même encodage que train_single_dqn.py
# ----------------------------
def obs_to_vec(obs, size: int, max_steps: int) -> torch.Tensor:
    self_pos = int(obs["self_pos"])
    goal = int(obs["goal"])
    others_pos = list(obs["others_pos"])
    arrived = 1.0 if obs["arrived"] else 0.0
    t = int(obs["t"])

    sr, sc = divmod(self_pos, size)
    gr, gc = divmod(goal, size)

    denom = (size - 1) if size > 1 else 1
    feats = [
        sr / denom, sc / denom,
        gr / denom, gc / denom,
    ]

    for p in others_pos:
        r, c = divmod(int(p), size)
        feats.extend([r / denom, c / denom])

    tf = t / max_steps if max_steps > 0 else 0.0
    feats.extend([arrived, tf])

    return torch.tensor(feats, dtype=torch.float32)


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def node_to_rc(node: int, size: int):
    return node // size, node % size


def masked_argmax(qvals: torch.Tensor, valid_actions):
    best_a = valid_actions[0]
    best_q = qvals[best_a].item()
    for a in valid_actions[1:]:
        q = qvals[a].item()
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


# ----------------------------
# Greedy "hyper optimisateur" pour les autres agents
# ----------------------------
def manhattan(node_a: int, node_b: int, size: int) -> int:
    ar, ac = divmod(node_a, size)
    br, bc = divmod(node_b, size)
    return abs(ar - br) + abs(ac - bc)


def greedy_to_goal_action(env: GridCongestionEnv, agent_i: int) -> int:
    """
    Choisit l'action valide qui minimise la distance Manhattan au goal après le déplacement.
    Tie-break déterministe via pref_order.
    """
    # si déjà arrivé -> stay
    if env.arrived[agent_i]:
        return 0  # STAY

    u = env.pos[agent_i]
    goal = env.goal
    size = env.size
    valid = env.valid_actions(agent_i)

    # Tie-break: tu peux changer cet ordre si tu veux
    pref_order = [1, 2, 3, 4, 0]  # UP, DOWN, LEFT, RIGHT, STAY

    ur, uc = divmod(u, size)

    best_a = valid[0]
    best_d = 10**9
    best_rank = 10**9

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
        # a == 0 -> stay

        v = nr * size + nc
        d = manhattan(v, goal, size)
        rank = pref_order.index(a) if a in pref_order else 999

        if (d < best_d) or (d == best_d and rank < best_rank):
            best_d = d
            best_rank = rank
            best_a = a

    return best_a


def load_trained_agent(checkpoint_path: str, cfg: EnvConfig, device: str = "cpu"):
    obs_dim = 2 + 2 + 2 * (cfg.n_agents - 1) + 1 + 1
    n_actions = 5
    model = QNet(obs_dim, n_actions, hidden=128).to(device)
    sd = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def pick_ckpt(dirpath: str, agent_id: int):
    best_path = os.path.join(dirpath, f"agent{agent_id}_best.pt")
    final_path = os.path.join(dirpath, f"agent{agent_id}_final.pt")
    if os.path.exists(best_path):
        return best_path
    if os.path.exists(final_path):
        return final_path
    return None


def visualize_episode_step_by_step(
    env: GridCongestionEnv,
    model0: nn.Module,
    model1: nn.Module,
    episode_num: int = 1,
    device: str = "cpu",
    interval_ms: int = 500,
):
    agent0_id = 0
    agent1_id = 1

    obs_list = env.reset(global_obs=False)
    size = env.size
    goal = env.goal
    starts = list(env.starts)

    history_pos = [tuple(env.pos)]
    history_costs = [tuple(env.cost)]
    history_rewards0 = []
    history_rewards1 = []

    done_global = False
    episode_reward0 = 0.0
    episode_reward1 = 0.0
    arrived0 = False
    arrived1 = False

    while not done_global:
        actions = []

        # --- Action agent0 (DQN greedy) ---
        if obs_list[agent0_id]["arrived"]:
            a0 = 0
        else:
            s0 = obs_to_vec(obs_list[agent0_id], env.size, env.cfg.max_steps).to(device)
            valid0 = env.valid_actions(agent0_id)
            with torch.no_grad():
                qvals0 = model0(s0)
            a0 = masked_argmax(qvals0, valid0)

        # --- Action agent1 (DQN greedy) ---
        if obs_list[agent1_id]["arrived"]:
            a1 = 0
        else:
            s1 = obs_to_vec(obs_list[agent1_id], env.size, env.cfg.max_steps).to(device)
            valid1 = env.valid_actions(agent1_id)
            with torch.no_grad():
                qvals1 = model1(s1)
            a1 = masked_argmax(qvals1, valid1)

        # --- Autres agents GREEDY vers le goal ---
        for i in range(env.n_agents):
            if i == agent0_id:
                actions.append(a0)
            elif i == agent1_id:
                actions.append(a1)
            else:
                actions.append(greedy_to_goal_action(env, i))

        next_obs_list, rewards, terminated, truncated, info = env.step(actions, global_obs=False)
        done_global = terminated or truncated

        r0 = float(rewards[agent0_id])
        r1 = float(rewards[agent1_id])
        episode_reward0 += r0
        episode_reward1 += r1
        history_rewards0.append(r0)
        history_rewards1.append(r1)

        if next_obs_list[agent0_id]["arrived"]:
            arrived0 = True
        if next_obs_list[agent1_id]["arrived"]:
            arrived1 = True

        obs_list = next_obs_list
        history_pos.append(tuple(env.pos))
        history_costs.append(tuple(env.cost))

    # ----------------------------
    # Plot + animation
    # ----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    ax1.set_title(f"Episode {episode_num} - Positions des agents", fontsize=14, fontweight="bold")
    ax1.set_xlim(-0.5, size - 0.5)
    ax1.set_ylim(-0.5, size - 0.5)
    ax1.set_xticks(range(size))
    ax1.set_yticks(range(size))
    ax1.grid(True, alpha=0.3)
    ax1.invert_yaxis()
    ax1.set_xlabel("Colonne")
    ax1.set_ylabel("Ligne")

    # Goal
    gr, gc = node_to_rc(goal, size)
    ax1.scatter([gc], [gr], marker="*", s=800, c="green", zorder=10, edgecolors="darkgreen", linewidth=2)

    # Starts
    for i, start in enumerate(starts):
        sr, sc = node_to_rc(start, size)
        if i == 0:
            color = "darkred"
        elif i == 1:
            color = "darkorange"
        else:
            color = "darkblue"
        ax1.scatter([sc], [sr], marker="x", s=300, c=color, alpha=0.5, linewidth=2)

    # Agents
    scat = ax1.scatter([], [], s=300, alpha=0.85, edgecolors="black", linewidth=2)

    # Info panel
    ax2.axis("off")
    info_text = ax2.text(
        0.05, 0.95, "",
        transform=ax2.transAxes,
        verticalalignment="top",
        fontsize=11,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85),
    )

    # Couleurs : agent0 rouge, agent1 orange, autres bleu
    colors = ["blue"] * env.n_agents
    colors[0] = "red"
    colors[1] = "orange"

    # Labels adaptés
    agent_names = [f"Agent {i} (Greedy)" for i in range(env.n_agents)]
    agent_names[0] = "Agent 0 (DQN)"
    agent_names[1] = "Agent 1 (DQN)"

    def update(frame: int):
        positions = history_pos[frame]
        pts = []
        cols = []

        for i, p in enumerate(positions):
            r, c = node_to_rc(p, size)
            pts.append([c, r])
            cols.append(colors[i])

        scat.set_offsets(np.array(pts))
        scat.set_color(cols)

        costs = history_costs[frame]
        info_str = f"Step: {frame}/{len(history_pos) - 1}\n"
        info_str += f"Goal node: {goal}\n"
        info_str += f"Arrived agent0: {arrived0} | agent1: {arrived1}\n"
        info_str += "\nCoûts cumulés:\n"
        for i in range(env.n_agents):
            info_str += f"  {agent_names[i]}: {costs[i]:.1f}\n"

        info_str += f"\nReward total agent0: {episode_reward0:.1f}\n"
        info_str += f"Reward total agent1: {episode_reward1:.1f}\n"

        info_text.set_text(info_str)
        return scat, info_text

    anim = FuncAnimation(fig, update, frames=len(history_pos), interval=interval_ms, blit=True, repeat=True)

    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="green", markersize=15, label="Goal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Agent 0 (DQN)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="orange", markersize=10, label="Agent 1 (DQN)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Agents greedy"),
        Line2D([0], [0], marker="x", color="darkred", linewidth=2, label="Départ agent 0"),
        Line2D([0], [0], marker="x", color="darkorange", linewidth=2, label="Départ agent 1"),
        Line2D([0], [0], marker="x", color="darkblue", linewidth=2, label="Départs greedy"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=10)

    plt.tight_layout()
    plt.show()

    return (episode_reward0, arrived0, episode_reward1, arrived1, history_costs[-1])


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = EnvConfig(size=6, n_agents=4, max_steps=60, base_cost=1.0, goal_reward=20.0, seed=None)
    env = GridCongestionEnv(cfg)

    # --- Load agent 0 ---
    ckpt0 = pick_ckpt("checkpoints_single", 0)
    if ckpt0 is None:
        print("✗ Aucun checkpoint trouvé pour agent0 dans checkpoints_single/")
        return

    # --- Load agent 1 ---
    ckpt1 = pick_ckpt("checkpoints_agent1", 1)
    if ckpt1 is None:
        print("✗ Aucun checkpoint trouvé pour agent1 dans checkpoints_agent1/")
        return

    print(f"=== Chargement agent0: {ckpt0} ===")
    model0 = load_trained_agent(ckpt0, cfg, device=device)
    print("✓ agent0 chargé.")

    print(f"=== Chargement agent1: {ckpt1} ===")
    model1 = load_trained_agent(ckpt1, cfg, device=device)
    print("✓ agent1 chargé.")

    num_episodes = 5
    for ep in range(num_episodes):
        print(f"\nEpisode {ep+1}/{num_episodes}...")
        r0, a0, r1, a1, final_costs = visualize_episode_step_by_step(
            env, model0, model1, episode_num=ep + 1, device=device, interval_ms=500
        )
        print(f"  Agent0 -> reward: {r0:.1f}, arrivé: {a0}")
        print(f"  Agent1 -> reward: {r1:.1f}, arrivé: {a1}")
        print(f"  Coûts finaux: {final_costs}")

    print("\n=== Terminé ===")


if __name__ == "__main__":
    main()
