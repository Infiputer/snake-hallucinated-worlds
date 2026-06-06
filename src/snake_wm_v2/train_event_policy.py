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
from .event_model import EventSnakeWorldModel
from .models import CNNPolicy


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PPO policy inside hard-event Snake world model")
    p.add_argument("--world-model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--policy", choices=tuple(POLICY_SPECS), default="small")
    p.add_argument("--updates", type=int, default=250)
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
    p.add_argument("--reward-decoder", choices=("hard", "prob"), default="hard")
    p.add_argument("--max-episode-steps", type=int, default=256)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto")
    p.add_argument("--project", default="snake-hallucinated-worlds-event")
    return p.parse_args()


def load_event_world_model(path: str | Path, device: torch.device) -> EventSnakeWorldModel:
    ckpt = torch.load(path, map_location="cpu")
    cfg = ckpt["model_config"]
    model = EventSnakeWorldModel(str(cfg["variant"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


class EventHallucinatedBatchEnv:
    def __init__(
        self,
        model: EventSnakeWorldModel,
        dataset_path: str | Path,
        num_envs: int,
        max_episode_steps: int,
        device: torch.device,
        seed: int,
        reward_decoder: str = "hard",
    ):
        self.model = model
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)
        self.device = device
        self.reward_decoder = reward_decoder
        self.rng = np.random.default_rng(seed)
        dataset_path = Path(dataset_path)
        self.frames = np.load(dataset_path / "frames.npy", mmap_mode="r")
        self.context_indices = np.load(dataset_path / "context_indices.npy", mmap_mode="r")[:, -1]
        self.dataset_len = len(self.context_indices)
        _, height, width, channels = self.frames.shape
        if channels != 3:
            raise ValueError(f"expected RGB frames, got shape {self.frames.shape}")
        self.frame_shape = (3, int(height), int(width))
        self.steps = torch.zeros(self.num_envs, dtype=torch.long, device=device)
        self.frame = torch.empty(self.num_envs, *self.frame_shape, device=device)
        self.reset(torch.arange(self.num_envs, device=device))

    def reset(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        idx = self.rng.integers(0, self.dataset_len, size=int(env_ids.numel()))
        frames = self.frames[self.context_indices[idx]].astype(np.float32) / 255.0
        self.frame[env_ids] = torch.from_numpy(np.asarray(frames)).permute(0, 3, 1, 2).to(self.device)
        self.steps[env_ids] = 0

    @property
    def obs(self) -> torch.Tensor:
        return self.frame

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        out = self.model(self.frame, action)
        apple = out["apple_logits"].argmax(dim=1)
        apple_prob = torch.softmax(out["apple_logits"], dim=1)[:, 1]
        death = out["death_logits"].argmax(dim=1).bool()
        self.steps += 1
        timeout = self.steps >= self.max_episode_steps
        done = death | timeout
        apple_reward = apple.float() if self.reward_decoder == "hard" else apple_prob
        reward = torch.where(death, torch.full_like(apple_reward, -1.0), apple_reward)
        self.frame = out["frame"].detach()
        self.reset(torch.nonzero(done, as_tuple=False).flatten())
        info = {"apple": apple.float(), "apple_reward": apple_reward.float(), "death": death.float()}
        return self.obs, reward, done.float(), info


def save_policy(path: Path, policy: CNNPolicy, opt: torch.optim.Optimizer, update: int, cfg: argparse.Namespace, world_model_cfg: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "update": update,
        "policy_variant": policy.variant,
        "policy_param_count": count_parameters(policy),
        "world_model": world_model_cfg,
        "event_policy": True,
        "args": vars(cfg),
    }, path)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_ckpt = torch.load(cfg.world_model, map_location="cpu")
    world_model_cfg = wm_ckpt["model_config"]
    world_model = load_event_world_model(cfg.world_model, device)
    policy = CNNPolicy(cfg.policy).to(device)
    opt = torch.optim.AdamW(policy.parameters(), lr=cfg.lr)
    env = EventHallucinatedBatchEnv(world_model, cfg.dataset, cfg.num_envs, cfg.max_episode_steps, device, cfg.seed, cfg.reward_decoder)
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(name=out_dir.name, config=vars(cfg), **wandb_kwargs(cfg.project, cfg.wandb_mode))
    write_json(out_dir / "metadata.json", {
        "dataset": cfg.dataset,
        "world_model_checkpoint": cfg.world_model,
        "world_model": world_model_cfg,
        "policy_variant": cfg.policy,
        "policy_params": count_parameters(policy),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "reward": "hard argmax death terminates with -1; apple reward is argmax class or softmax probability",
        "args": vars(cfg),
    })
    obs = env.obs
    t0 = time.time()
    for update in range(1, cfg.updates + 1):
        obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf, apple_buf, apple_reward_buf, death_buf = [], [], [], [], [], [], [], [], []
        for _ in range(cfg.rollout_steps):
            logits, value = policy(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            next_obs, reward, done, info = env.step(action)
            obs_buf.append(obs)
            act_buf.append(action)
            logp_buf.append(dist.log_prob(action))
            rew_buf.append(reward)
            done_buf.append(done)
            val_buf.append(value)
            apple_buf.append(info["apple"])
            apple_reward_buf.append(info["apple_reward"])
            death_buf.append(info["death"])
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
        b_obs = torch.stack(obs_buf).reshape(-1, *env.frame_shape).detach()
        b_actions = torch.stack(act_buf).reshape(-1).detach()
        b_logp = torch.stack(logp_buf).reshape(-1).detach()
        b_adv = advantages.reshape(-1).detach()
        b_returns = returns.reshape(-1).detach()
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
        n = b_obs.shape[0]
        pi_loss_total = value_loss_total = entropy_total = 0.0
        opt_steps = 0
        for _ in range(cfg.epochs):
            idx = torch.randperm(n, device=device)
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
            metrics = {
                "ppo/hallucinated_return_mean": float(rewards.sum(dim=0).mean().item()),
                "ppo/reward_mean": float(rewards.mean().item()),
                "ppo/apple_rate": float(torch.stack(apple_buf).mean().item()),
                "ppo/apple_reward_mean": float(torch.stack(apple_reward_buf).mean().item()),
                "ppo/death_rate": float(torch.stack(death_buf).mean().item()),
                "ppo/done_rate": float(dones.mean().item()),
                "ppo/pi_loss": pi_loss_total / max(opt_steps, 1),
                "ppo/value_loss": value_loss_total / max(opt_steps, 1),
                "ppo/entropy": entropy_total / max(opt_steps, 1),
                "time/min": (time.time() - t0) / 60.0,
            }
            wandb.log(metrics, step=update)
            print({"update": update, **metrics}, flush=True)
        if update == 1 or update % cfg.save_every == 0 or update == cfg.updates:
            save_policy(out_dir / f"policy_update_{update:06d}.pt", policy, opt, update, cfg, world_model_cfg)
            save_policy(out_dir / "latest.pt", policy, opt, update, cfg, world_model_cfg)
    run.finish()


if __name__ == "__main__":
    main()
