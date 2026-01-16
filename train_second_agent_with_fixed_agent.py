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
    best_a = valid_actions[0]
    best_q = qvals[best_a].item()
    for a in valid_actions[1:]:
        q = qvals[a].item()
        if q > best_q:
            best_q = q
            best_a = a
    return best_a


def load_fixed_agent(checkpoint_path: str, cfg: EnvConfig, device: str) -> nn.Module:
    obs_dim = 2 + 2 + 2 * (cfg.n_agents - 1) + 1 + 1
    n_actions = 5
    model = QNet(obs_dim, n_actions, hidden=128).to(device)
    sd = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(sd)
    model.eval()
    return model


def load_state_dict_safely(path: str, device: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint introuvable: {path}")
    return torch.load(path, map_location=device)


def freeze_feature_layers(q: QNet, freeze: bool):
    """
    Gel/dégel des 2 premières Linear (feature extractor).
    net = [Linear, ReLU, Linear, ReLU, Linear]
    """
    layers_to_freeze = [0, 2]
    for idx in layers_to_freeze:
        layer = q.net[idx]
        if isinstance(layer, nn.Linear):
            for p in layer.parameters():
                p.requires_grad = (not freeze)


# ----------------------------
# Greedy "hyper optimisateur" pour agents non-DQN (2 & 3)
# ----------------------------
def manhattan(node_a: int, node_b: int, size: int) -> int:
    ar, ac = divmod(node_a, size)
    br, bc = divmod(node_b, size)
    return abs(ar - br) + abs(ac - bc)


def greedy_to_goal_action(env: GridCongestionEnv, agent_i: int) -> int:
    """
    Choisit l'action valide qui minimise la distance Manhattan au goal après déplacement.
    Tie-break déterministe via pref_order.
    """
    if env.arrived[agent_i]:
        return 0  # STAY

    u = env.pos[agent_i]
    goal = env.goal
    size = env.size
    valid = env.valid_actions(agent_i)

    # ordre de préférence pour départager à distance égale
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
# Training: agent 1 apprend, agent 0 fixé
# + warm start agent1 depuis agent0
# + agents 2 & 3 greedy vers le goal
# ----------------------------
def train_agent1_with_fixed_agent0_warmstart(
    agent0_ckpt: str,
    train_agent_id: int = 1,
    episodes: int = 4000,
    seed: int = 0,
    save_dir: str = "checkpoints_agent1",
    device: str = "cpu",
    warmstart_from_agent0: bool = True,
    freeze_steps: int = 20_000,   # 0 = désactivé
):
    os.makedirs(save_dir, exist_ok=True)
    set_seed(seed)

    cfg = EnvConfig(size=6, n_agents=4, max_steps=60, base_cost=1.0, goal_reward=20.0, seed=seed)
    env = GridCongestionEnv(cfg)

    # Agent 0 figé (DQN chargé)
    fixed_id = 0
    fixed_agent = load_fixed_agent(agent0_ckpt, cfg, device=device)

    # Agent 1 (train)
    obs_dim = 2 + 2 + 2 * (cfg.n_agents - 1) + 1 + 1
    n_actions = 5
    q = QNet(obs_dim, n_actions, hidden=128).to(device)
    q_targ = QNet(obs_dim, n_actions, hidden=128).to(device)

    # ---- WARM START depuis agent0 ----
    if warmstart_from_agent0:
        sd0 = load_state_dict_safely(agent0_ckpt, device=device)
        q.load_state_dict(sd0)
        q_targ.load_state_dict(sd0)
    else:
        q_targ.load_state_dict(q.state_dict())

    q_targ.eval()

    # Optim
    opt = optim.Adam(q.parameters(), lr=5e-4)
    rb = ReplayBuffer(capacity=100_000)

    gamma = 0.99
    batch_size = 256
    learning_starts = 2000
    target_update_every = 500
    max_grad_norm = 10.0

    # avec warm-start: eps plus bas
    eps_start, eps_end = 0.30, 0.05
    eps_decay_steps = 150_000

    global_step = 0
    returns_window = deque(maxlen=100)
    best_window100 = -1e9

    for ep in range(1, episodes + 1):
        obs_list = env.reset(global_obs=False)
        done_global = False
        ep_return = 0.0

        while not done_global:
            # Freeze feature layers au début (optionnel)
            if freeze_steps > 0:
                freeze_feature_layers(q, freeze=(global_step < freeze_steps))

            frac = max(0.0, (eps_decay_steps - global_step) / eps_decay_steps)
            eps = eps_end + (eps_start - eps_end) * frac

            actions = []

            # -------- Agent 0 (fixe DQN greedy) --------
            if obs_list[fixed_id]["arrived"]:
                a0 = 0
            else:
                s0 = obs_to_vec(obs_list[fixed_id], cfg.size, cfg.max_steps).to(device)
                valid0 = env.valid_actions(fixed_id)
                with torch.no_grad():
                    qvals0 = fixed_agent(s0)
                a0 = masked_argmax(qvals0, valid0)

            # -------- Agent 1 (train) --------
            agent_already_arrived = bool(obs_list[train_agent_id]["arrived"])
            if agent_already_arrived:
                a1 = 0
                s1 = None
            else:
                s1 = obs_to_vec(obs_list[train_agent_id], cfg.size, cfg.max_steps).to(device)
                valid1 = env.valid_actions(train_agent_id)

                if random.random() < eps:
                    a1 = random.choice(valid1)
                else:
                    with torch.no_grad():
                        qvals1 = q(s1)
                    a1 = masked_argmax(qvals1, valid1)

            # -------- Agents 2 & 3 GREEDY vers le goal --------
            for i in range(cfg.n_agents):
                if i == fixed_id:
                    actions.append(a0)
                elif i == train_agent_id:
                    actions.append(a1)
                else:
                    actions.append(greedy_to_goal_action(env, i))

            next_obs_list, rewards, terminated, truncated, info = env.step(actions, global_obs=False)
            done_global = terminated or truncated

            r1 = float(rewards[train_agent_id])
            ep_return += r1

            # stocker transition agent 1 seulement si pas déjà arrivé
            if (not agent_already_arrived) and (s1 is not None):
                s2 = obs_to_vec(next_obs_list[train_agent_id], cfg.size, cfg.max_steps).to(device)
                done_agent = float(next_obs_list[train_agent_id]["arrived"] or truncated)
                rb.push(s1.cpu(), a1, r1, s2.cpu(), done_agent)

            obs_list = next_obs_list
            global_step += 1

            # learn
            if global_step > learning_starts and len(rb) >= batch_size:
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

            if global_step % target_update_every == 0:
                q_targ.load_state_dict(q.state_dict())

        returns_window.append(ep_return)

        if ep % 50 == 0:
            w = sum(returns_window) / len(returns_window)
            print(
                f"ep={ep:5d}  return(agent{train_agent_id})={ep_return:8.2f}  "
                f"window100={w:8.2f}  eps={eps:5.3f}  steps={global_step}"
            )

            if w > best_window100:
                best_window100 = w
                torch.save(q.state_dict(), os.path.join(save_dir, f"agent{train_agent_id}_best.pt"))

    final_path = os.path.join(save_dir, f"agent{train_agent_id}_final.pt")
    torch.save(q.state_dict(), final_path)

    meta = {
        "fixed_agent_id": fixed_id,
        "fixed_agent_ckpt": agent0_ckpt,
        "trained_agent_id": train_agent_id,
        "warmstart_from_agent0": warmstart_from_agent0,
        "freeze_steps": freeze_steps,
        "other_agents_policy": "greedy_to_goal_action (agents != 0,1)",
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
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent0_best = os.path.join("checkpoints_single", "agent0_best.pt")
    agent0_final = os.path.join("checkpoints_single", "agent0_final.pt")
    agent0_ckpt = agent0_best if os.path.exists(agent0_best) else agent0_final

    if not os.path.exists(agent0_ckpt):
        raise FileNotFoundError(
            f"Aucun checkpoint agent0 trouvé. Cherché:\n- {agent0_best}\n- {agent0_final}"
        )

    train_agent1_with_fixed_agent0_warmstart(
        agent0_ckpt=agent0_ckpt,
        train_agent_id=1,
        episodes=2000,
        seed=0,
        save_dir="checkpoints_agent1",
        device=device,
        warmstart_from_agent0=True,
        freeze_steps=20_000,  # mets 0 si tu veux désactiver le freeze
    )
