import os
import json
import random
from collections import deque
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim

from env_6x6 import GridCongestionEnv, EnvConfig


# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def obs_to_vec(obs: Dict[str, Any], size: int, max_steps: int) -> torch.Tensor:
    """
    Observation enrichie avec positions des autres agents (nécessite others_pos dans env).
    Vector = [self_r, self_c, goal_r, goal_c,
              o1_r, o1_c, o2_r, o2_c, o3_r, o3_c,
              arrived, t_norm]
    dim = 2 + 2 + 2*(n_agents-1) + 1 + 1
    """
    self_pos = int(obs["self_pos"])
    goal = int(obs["goal"])
    others_pos = list(obs["others_pos"])  # longueur = n_agents-1
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


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buf = deque(maxlen=capacity)

    def push(self, s: torch.Tensor, a: int, r: float, s2: torch.Tensor, done: float):
        self.buf.append((s, a, r, s2, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s, a, r, s2, d = zip(*batch)
        return (
            torch.stack(s),
            torch.tensor(a, dtype=torch.int64),
            torch.tensor(r, dtype=torch.float32),
            torch.stack(s2),
            torch.tensor(d, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buf)


def masked_argmax(qvals: torch.Tensor, valid_actions: List[int]) -> int:
    # qvals shape: [n_actions]
    best_a = valid_actions[0]
    best_q = qvals[best_a].item()
    for a in valid_actions[1:]:
        q = qvals[a].item()
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


# ----------------------------
# Hyper-optimisateur (greedy direct vers goal)
# ----------------------------

def manhattan(node_a: int, node_b: int, size: int) -> int:
    ar, ac = divmod(node_a, size)
    br, bc = divmod(node_b, size)
    return abs(ar - br) + abs(ac - bc)


def greedy_to_goal_action(env: GridCongestionEnv, agent_i: int) -> int:
    """
    Choisit l'action valide qui minimise la distance Manhattan au goal après mouvement.
    Tie-break déterministe via l'ordre pref_order.
    """
    if env.arrived[agent_i]:
        return 0  # STAY

    u = env.pos[agent_i]
    goal = env.goal
    size = env.size

    valid = env.valid_actions(agent_i)

    # ordre de préférence pour départager à distance égale
    # (tu peux changer l'ordre si tu veux une autre "personnalité")
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


# ----------------------------
# Training (1 agent only)
# ----------------------------

def train_single_agent_dqn(
    agent_id: int = 0,
    episodes: int = 3000,
    seed: int = 0,
    save_dir: str = "checkpoints_single",
    device: str = "cpu",
):
    os.makedirs(save_dir, exist_ok=True)
    set_seed(seed)

    # Env
    cfg = EnvConfig(size=6, n_agents=4, max_steps=60, base_cost=1.0, goal_reward=20.0, seed=seed)
    env = GridCongestionEnv(cfg)

    # Obs dim: 2(self) + 2(goal) + 2*(n_agents-1) + 1(arrived) + 1(t)
    obs_dim = 2 + 2 + 2 * (cfg.n_agents - 1) + 1 + 1
    n_actions = 5

    q = QNet(obs_dim, n_actions, hidden=128).to(device)
    q_targ = QNet(obs_dim, n_actions, hidden=128).to(device)
    q_targ.load_state_dict(q.state_dict())
    q_targ.eval()

    opt = optim.Adam(q.parameters(), lr=5e-4)
    rb = ReplayBuffer(capacity=100_000)

    gamma = 0.99
    batch_size = 256
    learning_starts = 3000
    target_update_every = 500
    train_every = 1
    max_grad_norm = 10.0

    # Epsilon decay
    eps_start, eps_end = 1.0, 0.05
    eps_decay_steps = 200_000

    global_step = 0
    returns_window = deque(maxlen=100)
    best_window100 = -1e9

    for ep in range(1, episodes + 1):
        obs_list = env.reset(global_obs=False)
        done_global = False
        ep_return = 0.0

        while not done_global:
            # epsilon schedule
            frac = max(0.0, (eps_decay_steps - global_step) / eps_decay_steps)
            eps = eps_end + (eps_start - eps_end) * frac

            agent_already_arrived = bool(obs_list[agent_id]["arrived"])

            # action agent entraîné
            if agent_already_arrived:
                a_train = 0  # STAY
            else:
                s = obs_to_vec(obs_list[agent_id], cfg.size, cfg.max_steps).to(device)
                valid = env.valid_actions(agent_id)

                if random.random() < eps:
                    a_train = random.choice(valid)
                else:
                    with torch.no_grad():
                        qvals = q(s)
                    a_train = masked_argmax(qvals, valid)

            # actions autres agents: hyper-optimisateurs (greedy direct)
            actions = []
            for i in range(cfg.n_agents):
                if i == agent_id:
                    actions.append(a_train)
                else:
                    actions.append(greedy_to_goal_action(env, i))

            next_obs_list, rewards, terminated, truncated, info = env.step(actions, global_obs=False)
            done_global = terminated or truncated

            r = float(rewards[agent_id])
            ep_return += r

            # Stockage / apprentissage seulement si l'agent n'était pas déjà arrivé
            if not agent_already_arrived:
                s2 = obs_to_vec(next_obs_list[agent_id], cfg.size, cfg.max_steps).to(device)

                # done au niveau agent
                done_agent = float(next_obs_list[agent_id]["arrived"] or truncated)

                rb.push(
                    obs_to_vec(obs_list[agent_id], cfg.size, cfg.max_steps).cpu(),
                    a_train,
                    r,
                    s2.cpu(),
                    done_agent,
                )

            obs_list = next_obs_list
            global_step += 1

            # Learn
            if global_step > learning_starts and len(rb) >= batch_size and (global_step % train_every == 0):
                S, A, R, S2, D = rb.sample(batch_size)
                S = S.to(device)
                A = A.to(device)
                R = R.to(device)
                S2 = S2.to(device)
                D = D.to(device)

                q_sa = q(S).gather(1, A.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    max_next = q_targ(S2).max(dim=1).values
                    target = R + gamma * (1.0 - D) * max_next

                loss = nn.functional.smooth_l1_loss(q_sa, target)

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), max_grad_norm)
                opt.step()

            # Target update
            if global_step % target_update_every == 0:
                q_targ.load_state_dict(q.state_dict())

        # Logs
        returns_window.append(ep_return)
        if ep % 50 == 0:
            w = sum(returns_window) / len(returns_window)
            print(
                f"ep={ep:5d}  return={ep_return:8.2f}  window100={w:8.2f}  "
                f"eps={eps:5.3f}  steps={global_step}"
            )

            if w > best_window100:
                best_window100 = w
                torch.save(q.state_dict(), os.path.join(save_dir, f"agent{agent_id}_best.pt"))

    # Save final
    final_path = os.path.join(save_dir, f"agent{agent_id}_final.pt")
    torch.save(q.state_dict(), final_path)

    meta = {
        "agent_id": agent_id,
        "env_config": {
            "size": cfg.size,
            "n_agents": cfg.n_agents,
            "max_steps": cfg.max_steps,
            "base_cost": cfg.base_cost,
            "goal_reward": cfg.goal_reward,
            "seed": cfg.seed,
        },
        "train": {
            "episodes": episodes,
            "seed": seed,
            "gamma": gamma,
            "batch_size": batch_size,
            "learning_starts": learning_starts,
            "target_update_every": target_update_every,
            "eps_start": eps_start,
            "eps_end": eps_end,
            "eps_decay_steps": eps_decay_steps,
            "lr": 5e-4,
        },
        "best_window100_return": best_window100,
        "final_checkpoint": final_path,
    }
    with open(os.path.join(save_dir, "training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nSaved final model: {final_path}")
    print(f"Saved meta: {os.path.join(save_dir, 'training_meta.json')}")
    print(f"Best window100 return: {best_window100:.2f}")


if __name__ == "__main__":
    train_single_agent_dqn(
        agent_id=0,
        episodes=3000,
        seed=0,
        save_dir="checkpoints_single",
        device="cpu",  # "cuda" si dispo
    )
