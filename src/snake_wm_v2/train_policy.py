from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
import wandb

from .common import POLICY_SPECS, count_parameters, set_seed, wandb_kwargs, write_json
from .models import CNNPolicy, LatentSnakeWorldModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN PPO policy inside frozen no-overlay world model")
    p.add_argument("--world-model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policy", choices=tuple(POLICY_SPECS), default="small")
    p.add_argument("--init-policy", default=None)
    p.add_argument("--updates", type=int, default=500)
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
    p.add_argument("--reward-mode", choices=("wm", "length_delta", "mixed"), default="wm")
    p.add_argument("--length-reward-scale", type=float, default=1.0)
    p.add_argument("--wm-reward-scale", type=float, default=1.0)
    p.add_argument("--death-penalty", type=float, default=1.0)
    p.add_argument("--length-delta-clamp", type=float, default=0.0, help="Clamp absolute length delta if >0; 0 keeps the length-head delta unclamped")
    p.add_argument("--ppo-reward-clamp", type=float, default=5.0, help="Clamp absolute PPO reward if >0; 0 disables final reward clipping")
    p.add_argument("--done-threshold", type=float, default=0.8)
    p.add_argument("--max-episode-steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto")
    p.add_argument("--project", default="snake-hallucinated-worlds-v2")
    return p.parse_args()


def load_world_model(path: str | Path, device: torch.device) -> LatentSnakeWorldModel:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["model_config"]
    model = LatentSnakeWorldModel(int(cfg["context"]), str(cfg["variant"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class HallucinatedBatchEnv:
    def __init__(
        self,
        model: LatentSnakeWorldModel,
        dataset_path: str | Path,
        num_envs: int,
        max_episode_steps: int,
        done_threshold: float,
        device: torch.device,
        seed: int,
        reward_mode: str = "wm",
        length_reward_scale: float = 1.0,
        wm_reward_scale: float = 1.0,
        death_penalty: float = 1.0,
        length_delta_clamp: float = 0.0,
        ppo_reward_clamp: float = 5.0,
    ):
        self.model = model
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.done_threshold = float(done_threshold)
        self.device = device
        self.reward_mode = reward_mode
        self.length_reward_scale = float(length_reward_scale)
        self.wm_reward_scale = float(wm_reward_scale)
        self.death_penalty = float(death_penalty)
        self.length_delta_clamp = float(length_delta_clamp)
        self.ppo_reward_clamp = float(ppo_reward_clamp)
        self.rng = np.random.default_rng(seed)
        dataset_path = Path(dataset_path)
        self.frames = np.load(dataset_path / "frames.npy", mmap_mode="r")
        self.context_indices = np.load(dataset_path / "context_indices.npy", mmap_mode="r")[:, -model.context:]
        self.prev_rewards = np.load(dataset_path / "prev_rewards.npy", mmap_mode="r").astype(np.float32)
        self.lengths = np.load(dataset_path / "lengths.npy", mmap_mode="r").astype(np.float32)
        self.dataset_len = len(self.context_indices)
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.context = torch.empty(self.num_envs, model.context, 3, 128, 128, device=device)
        self.prev_reward = torch.empty(self.num_envs, device=device)
        self.length_estimate = torch.empty(self.num_envs, device=device)
        self.reset(torch.arange(self.num_envs, device=device))

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        idx = self.rng.integers(0, self.dataset_len, size=int(env_ids.numel()))
        context = self.frames[self.context_indices[idx]].astype(np.float32) / 255.0
        self.context[env_ids] = torch.from_numpy(np.asarray(context)).permute(0, 1, 4, 2, 3).to(self.device)
        self.prev_reward[env_ids] = torch.from_numpy(self.prev_rewards[idx]).to(self.device)
        self.length_estimate[env_ids] = torch.from_numpy(np.asarray(self.lengths[idx], dtype=np.float32)).to(self.device)
        self.steps[env_ids] = 0

    @property
    def obs(self) -> torch.Tensor:
        return self.context[:, -1]

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.model(self.context, self.prev_reward, action)
        wm_reward = out["reward"]
        if self.ppo_reward_clamp > 0:
            wm_reward = wm_reward.clamp(-self.ppo_reward_clamp, self.ppo_reward_clamp)
        next_length = out["length"]
        length_delta = next_length - self.length_estimate
        if self.length_delta_clamp > 0:
            length_delta = length_delta.clamp(-self.length_delta_clamp, self.length_delta_clamp)
        done_prob = torch.sigmoid(out["done_logit"])
        self.steps += 1
        done = (done_prob > self.done_threshold) | (self.steps >= self.max_episode_steps)
        if self.reward_mode == "wm":
            reward = wm_reward
        elif self.reward_mode == "length_delta":
            reward = self.length_reward_scale * length_delta - self.death_penalty * done.float()
        else:
            reward = self.wm_reward_scale * wm_reward + self.length_reward_scale * length_delta - self.death_penalty * done.float()
        if self.ppo_reward_clamp > 0:
            reward = reward.clamp(-self.ppo_reward_clamp, self.ppo_reward_clamp)
        next_frame = out["frame"].detach()
        self.context = torch.cat([self.context[:, 1:], next_frame.unsqueeze(1)], dim=1) if self.model.context > 1 else next_frame.unsqueeze(1)
        self.prev_reward = reward.detach()
        self.length_estimate = next_length.detach()
        self.reset(torch.nonzero(done, as_tuple=False).flatten())
        return self.obs, reward, done.float()


def save_policy(path: Path, policy: CNNPolicy, opt: torch.optim.Optimizer, update: int, cfg: argparse.Namespace, world_model_cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy_state_dict": policy.state_dict(), "optimizer_state_dict": opt.state_dict(), "update": update, "policy_variant": policy.variant, "policy_param_count": count_parameters(policy), "world_model": world_model_cfg, "args": vars(cfg)}, path)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_ckpt = torch.load(cfg.world_model, map_location="cpu")
    world_model_cfg = wm_ckpt["model_config"]
    world_model = load_world_model(cfg.world_model, device)
    policy = CNNPolicy(cfg.policy).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    if cfg.init_policy:
        init = torch.load(cfg.init_policy, map_location=device)
        policy.load_state_dict(init["policy_state_dict"])
    env = HallucinatedBatchEnv(
        world_model,
        cfg.dataset,
        cfg.num_envs,
        cfg.max_episode_steps,
        cfg.done_threshold,
        device,
        cfg.seed,
        cfg.reward_mode,
        cfg.length_reward_scale,
        cfg.wm_reward_scale,
        cfg.death_penalty,
        cfg.length_delta_clamp,
        cfg.ppo_reward_clamp,
    )
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(name=out_dir.name, config=vars(cfg), **wandb_kwargs(cfg.project, cfg.wandb_mode))
    write_json(out_dir / "metadata.json", {"dataset": cfg.dataset, "world_model_checkpoint": cfg.world_model, "world_model": world_model_cfg, "policy_variant": cfg.policy, "policy_params": count_parameters(policy), "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "args": vars(cfg)})
    obs = env.obs
    t0 = time.time()
    for update in range(1, cfg.updates + 1):
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
        for _ in range(cfg.rollout_steps):
            logits, value = policy(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_obs, reward, done = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(dist.log_prob(action))
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value)
            obs = next_obs
        with torch.no_grad():
            _, next_value = policy(obs)
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
        b_obs = torch.stack(obs_buf).reshape(-1, 3, 128, 128).detach()
        b_actions = torch.stack(act_buf).reshape(-1).detach()
        b_logp = torch.stack(logp_buf).reshape(-1).detach()
        b_adv = advantages.reshape(-1).detach()
        b_returns = returns.reshape(-1).detach()
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n = b_obs.shape[0]
        idx = torch.randperm(n, device=device)
        pi_loss_total = value_loss_total = entropy_total = 0.0
        opt_steps = 0
        for _ in range(cfg.epochs):
            for start in range(0, n, cfg.minibatch_size):
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
            metrics = {"ppo/hallucinated_return_mean": float(rewards.sum(dim=0).mean().item()), "ppo/reward_mean": float(rewards.mean().item()), "ppo/done_rate": float(dones.mean().item()), "ppo/pi_loss": pi_loss_total / max(opt_steps, 1), "ppo/value_loss": value_loss_total / max(opt_steps, 1), "ppo/entropy": entropy_total / max(opt_steps, 1), "time/min": (time.time() - t0) / 60.0}
            wandb.log(metrics, step=update)
            print({"update": update, **metrics}, flush=True)
        if update == 1 or update % cfg.save_every == 0 or update == cfg.updates:
            save_policy(out_dir / f"policy_update_{update:06d}.pt", policy, opt, update, cfg, world_model_cfg)
            save_policy(out_dir / "latest.pt", policy, opt, update, cfg, world_model_cfg)
    run.finish()


if __name__ == "__main__":
    main()
