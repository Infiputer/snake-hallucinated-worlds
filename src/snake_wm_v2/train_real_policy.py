from __future__ import annotations

import argparse
import csv
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb

from .common import POLICY_SPECS, RewardConfig, count_parameters, set_seed, shaped_reward, wandb_kwargs, write_json
from .env import SnakeEnv
from .models import CNNPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN PPO policy in the true Snake simulator")
    p.add_argument("--out", required=True)
    p.add_argument("--policy", choices=tuple(POLICY_SPECS), default="medium")
    p.add_argument("--init-policy", default=None)
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--rollout-steps", type=int, default=64)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--max-episode-steps", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto")
    p.add_argument("--project", default="snake-wm-pretraining-v1")
    return p.parse_args()


def frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(frames.astype(np.float32) / 255.0).permute(0, 3, 1, 2).to(device)


def save_policy(path: Path, policy: CNNPolicy, opt: torch.optim.Optimizer, update: int, cfg: argparse.Namespace, init_meta: dict | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "update": update,
            "policy_variant": policy.variant,
            "policy_param_count": count_parameters(policy),
            "init_policy": init_meta,
            "args": vars(cfg),
        },
        path,
    )


@torch.no_grad()
def evaluate_policy(policy: CNNPolicy, episodes: int, max_steps: int, seed: int, device: torch.device) -> dict[str, float]:
    reward_cfg = RewardConfig()
    returns, steps, lengths, apples, deaths, wins = [], [], [], [], [], []
    policy.eval()
    for ep in range(episodes):
        env = SnakeEnv(seed=seed + ep)
        result = env.reset()
        total = 0.0
        eaten = 0
        for step in range(max_steps):
            obs = frames_to_tensor(result.frame[None], device)
            logits, _ = policy(obs)
            action = int(torch.argmax(logits, dim=1).item())
            result = env.step(action)
            total += shaped_reward(result.reward, result.status, reward_cfg)
            eaten += int(result.reward > 0)
            if result.done:
                break
        returns.append(total)
        steps.append(step + 1)
        lengths.append(result.length)
        apples.append(eaten)
        deaths.append(int(result.status == "dead"))
        wins.append(int(result.status == "win"))
    policy.train()
    return {
        "eval/return_mean": float(np.mean(returns)),
        "eval/steps_mean": float(np.mean(steps)),
        "eval/length_mean": float(np.mean(lengths)),
        "eval/apples_mean": float(np.mean(apples)),
        "eval/death_rate": float(np.mean(deaths)),
        "eval/win_rate": float(np.mean(wins)),
    }


class RealBatchEnv:
    def __init__(self, num_envs: int, max_episode_steps: int, seed: int):
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.seed = int(seed)
        self.reward_cfg = RewardConfig()
        self.envs = [SnakeEnv(seed=self.seed + i) for i in range(self.num_envs)]
        self.results = [env.reset() for env in self.envs]
        self.steps = np.zeros(self.num_envs, dtype=np.int32)
        self.ep_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.ep_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.completed_returns: deque[float] = deque(maxlen=200)
        self.completed_lengths: deque[int] = deque(maxlen=200)

    @property
    def obs(self) -> np.ndarray:
        return np.stack([r.frame for r in self.results], axis=0)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.float32)
        for i, action in enumerate(actions.tolist()):
            result = self.envs[i].step(int(action))
            reward = shaped_reward(result.reward, result.status, self.reward_cfg)
            self.ep_returns[i] += reward
            self.ep_lengths[i] += 1
            self.steps[i] += 1
            done = result.done or self.steps[i] >= self.max_episode_steps
            rewards[i] = reward
            dones[i] = float(done)
            if done:
                self.completed_returns.append(float(self.ep_returns[i]))
                self.completed_lengths.append(int(self.ep_lengths[i]))
                self.envs[i] = SnakeEnv(seed=self.seed + i + len(self.completed_returns) * self.num_envs)
                result = self.envs[i].reset()
                self.steps[i] = 0
                self.ep_returns[i] = 0.0
                self.ep_lengths[i] = 0
            self.results[i] = result
        return self.obs, rewards, dones

    def recent_return(self) -> float:
        return float(np.mean(self.completed_returns)) if self.completed_returns else 0.0

    def recent_length(self) -> float:
        return float(np.mean(self.completed_lengths)) if self.completed_lengths else 0.0


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = CNNPolicy(cfg.policy).to(device)
    init_meta = None
    if cfg.init_policy:
        init = torch.load(cfg.init_policy, map_location=device)
        policy.load_state_dict(init["policy_state_dict"])
        init_meta = {"path": cfg.init_policy, "policy_variant": init.get("policy_variant"), "update": init.get("update"), "world_model": init.get("world_model")}
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    env = RealBatchEnv(cfg.num_envs, cfg.max_episode_steps, cfg.seed)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    curve_path = out_dir / "eval_curve.csv"
    run = wandb.init(name=out_dir.name, config=vars(cfg), **wandb_kwargs(cfg.project, cfg.wandb_mode))
    write_json(out_dir / "metadata.json", {"policy_variant": cfg.policy, "policy_params": count_parameters(policy), "init_policy": init_meta, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "args": vars(cfg)})
    curve_rows: list[dict[str, float | int]] = []
    obs_np = env.obs
    t0 = time.time()
    total_env_steps = 0
    for update in range(1, cfg.updates + 1):
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        for _ in range(cfg.rollout_steps):
            obs = frames_to_tensor(obs_np, device)
            logits, value = policy(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_obs_np, reward_np, done_np = env.step(action.detach().cpu().numpy())
            obs_buf.append(obs.detach())
            act_buf.append(action.detach())
            logp_buf.append(dist.log_prob(action).detach())
            rew_buf.append(torch.from_numpy(reward_np).to(device))
            done_buf.append(torch.from_numpy(done_np).to(device))
            val_buf.append(value.detach())
            obs_np = next_obs_np
            total_env_steps += cfg.num_envs
        with torch.no_grad():
            _, next_value = policy(frames_to_tensor(obs_np, device))
            rewards = torch.stack(rew_buf)
            dones = torch.stack(done_buf)
            values = torch.stack(val_buf)
            advantages = torch.zeros_like(rewards)
            last_adv = torch.zeros(cfg.num_envs, device=device)
            for t in reversed(range(cfg.rollout_steps)):
                next_nonterminal = 1.0 - dones[t]
                next_val = next_value if t == cfg.rollout_steps - 1 else values[t + 1]
                delta = rewards[t] + cfg.gamma * next_val * next_nonterminal - values[t]
                last_adv = delta + cfg.gamma * cfg.gae_lambda * next_nonterminal * last_adv
                advantages[t] = last_adv
            returns = advantages + values
        b_obs = torch.stack(obs_buf).reshape(-1, 3, 128, 128)
        b_actions = torch.stack(act_buf).reshape(-1)
        b_logp = torch.stack(logp_buf).reshape(-1)
        b_adv = advantages.reshape(-1).detach()
        b_returns = returns.reshape(-1).detach()
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        idx = torch.randperm(b_obs.shape[0], device=device)
        pi_loss_total = value_loss_total = entropy_total = 0.0
        opt_steps = 0
        for _ in range(cfg.epochs):
            for start in range(0, b_obs.shape[0], cfg.minibatch_size):
                mb = idx[start:start + cfg.minibatch_size]
                logits, value = policy(b_obs[mb])
                dist = Categorical(logits=logits)
                new_logp = dist.log_prob(b_actions[mb])
                ratio = (new_logp - b_logp[mb]).exp()
                pi_loss = -torch.min(ratio * b_adv[mb], torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * b_adv[mb]).mean()
                value_loss = F.mse_loss(value, b_returns[mb])
                entropy = dist.entropy().mean()
                loss = pi_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
                opt.step()
                pi_loss_total += float(pi_loss.item())
                value_loss_total += float(value_loss.item())
                entropy_total += float(entropy.item())
                opt_steps += 1
        if update == 1 or update % 5 == 0:
            metrics = {
                "real/train_return_recent": env.recent_return(),
                "real/train_length_recent": env.recent_length(),
                "real/reward_mean": float(rewards.mean().item()),
                "real/done_rate": float(dones.mean().item()),
                "real/pi_loss": pi_loss_total / max(opt_steps, 1),
                "real/value_loss": value_loss_total / max(opt_steps, 1),
                "real/entropy": entropy_total / max(opt_steps, 1),
                "env_steps": total_env_steps,
                "time/min": (time.time() - t0) / 60.0,
            }
            wandb.log(metrics, step=update)
            print({"update": update, **metrics}, flush=True)
        if update == 1 or update % cfg.eval_every == 0 or update == cfg.updates:
            eval_metrics = evaluate_policy(policy, cfg.eval_episodes, cfg.max_episode_steps, cfg.seed + 10000 + update, device)
            row = {"update": update, "env_steps": total_env_steps, **eval_metrics}
            curve_rows.append(row)
            with curve_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                writer.writeheader()
                writer.writerows(curve_rows)
            write_json(out_dir / "latest_eval.json", row)
            wandb.log(eval_metrics | {"eval/env_steps": total_env_steps}, step=update)
            print(row, flush=True)
        if update == 1 or update % cfg.save_every == 0 or update == cfg.updates:
            save_policy(out_dir / f"policy_update_{update:06d}.pt", policy, opt, update, cfg, init_meta)
            save_policy(out_dir / "latest.pt", policy, opt, update, cfg, init_meta)
    run.finish()


if __name__ == "__main__":
    main()
