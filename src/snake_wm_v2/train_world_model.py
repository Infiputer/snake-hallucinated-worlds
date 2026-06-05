from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import wandb

from .common import WORLD_MODEL_SPECS, count_parameters, set_seed, wandb_kwargs, write_json
from .dataset import SnakeWorldModelDataset
from .models import LatentSnakeWorldModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train no-overlay latent Snake world model")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variant", choices=tuple(WORLD_MODEL_SPECS), default="wm_1m")
    p.add_argument("--context", type=int, choices=(1, 2, 5), default=1)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--init-checkpoint", default=None, help="Optional world-model checkpoint to initialize from before fine-tuning")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto")
    p.add_argument("--project", default="snake-hallucinated-worlds-v2")
    return p.parse_args()


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def compute_losses(model: LatentSnakeWorldModel, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out = model(batch["context"], batch["prev_reward"], batch["action"])
    frame_l1 = F.l1_loss(out["frame"], batch["next_frame"])
    reward_mse = F.mse_loss(out["reward"], batch["reward"])
    done_bce = F.binary_cross_entropy_with_logits(out["done_logit"], batch["done"])
    length_huber = F.smooth_l1_loss(out["length"], batch["length"])
    total = frame_l1 + 0.5 * reward_mse + 0.5 * done_bce + 0.1 * length_huber
    return total, {"total": total.detach(), "frame_l1": frame_l1.detach(), "reward_mse": reward_mse.detach(), "done_bce": done_bce.detach(), "length_huber": length_huber.detach()}


@torch.no_grad()
def evaluate(model: LatentSnakeWorldModel, loader: DataLoader, device: torch.device, max_batches: int = 16) -> dict[str, float]:
    model.eval()
    losses = {"total": [], "frame_l1": [], "reward_mse": [], "done_bce": [], "length_huber": []}
    all_p: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch_to_device(batch, device)
        _, parts = compute_losses(model, batch)
        for k in losses:
            losses[k].append(float(parts[k].item()))
        out = model(batch["context"], batch["prev_reward"], batch["action"])
        all_p.append(torch.sigmoid(out["done_logit"]).cpu().numpy())
        all_y.append(batch["done"].cpu().numpy())
    y = np.concatenate(all_y).astype(bool) if all_y else np.zeros(0, dtype=bool)
    p = np.concatenate(all_p) if all_p else np.zeros(0, dtype=np.float32)
    pred = p > 0.5
    tp = int((pred & y).sum())
    fp = int((pred & ~y).sum())
    tn = int((~pred & ~y).sum())
    fn = int((~pred & y).sum())
    metrics = {f"val/{k}": float(np.mean(v)) for k, v in losses.items() if v}
    metrics.update({"val/done_tp": tp, "val/done_fp": fp, "val/done_tn": tn, "val/done_fn": fn, "val/done_precision": tp / max(tp + fp, 1), "val/done_recall": tp / max(tp + fn, 1), "val/done_fpr": fp / max(fp + tn, 1)})
    model.train()
    return metrics


@torch.no_grad()
def save_rollout_gif(model: LatentSnakeWorldModel, dataset: SnakeWorldModelDataset, device: torch.device, path: Path, steps: int = 24) -> None:
    model.eval()
    sample = dataset[0]
    context = sample["context"].unsqueeze(0).to(device)
    prev_reward = sample["prev_reward"].view(1).to(device)
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(steps):
        action = torch.tensor([int(rng.integers(0, 4))], device=device)
        out = model(context, prev_reward, action)
        frame = out["frame"].squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        frames.append((frame * 255).astype(np.uint8))
        next_frame = out["frame"].detach()
        context = torch.cat([context[:, 1:], next_frame.unsqueeze(1)], dim=1) if model.context > 1 else next_frame.unsqueeze(1)
        prev_reward = out["reward"].detach()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path.as_posix(), frames, fps=6)
    model.train()


def save_checkpoint(path: Path, model: LatentSnakeWorldModel, opt: torch.optim.Optimizer, step: int, cfg: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": opt.state_dict(), "step": step, "model_config": model.config(), "args": vars(cfg), "param_count": count_parameters(model)}, path)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = SnakeWorldModelDataset(cfg.dataset, cfg.context, "train")
    val_ds = SnakeWorldModelDataset(cfg.dataset, cfg.context, "val")
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(cfg.num_workers, 2), pin_memory=device.type == "cuda")
    train_iter = iter(train_loader)
    model = LatentSnakeWorldModel(cfg.context, cfg.variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    if cfg.init_checkpoint:
        init = torch.load(cfg.init_checkpoint, map_location="cpu")
        init_cfg = init["model_config"]
        if int(init_cfg["context"]) != int(cfg.context) or str(init_cfg["variant"]) != str(cfg.variant):
            raise ValueError(f"checkpoint config {init_cfg} does not match requested context={cfg.context} variant={cfg.variant}")
        model.load_state_dict(init["model_state_dict"])
    scaler = GradScaler(enabled=device.type == "cuda")
    run = wandb.init(name=out_dir.name, config=vars(cfg), **wandb_kwargs(cfg.project, cfg.wandb_mode))
    write_json(out_dir / "metadata.json", {"dataset": cfg.dataset, "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "param_count": count_parameters(model), "model_config": model.config(), "args": vars(cfg)})
    t0 = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
        batch = batch_to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            loss, parts = compute_losses(model, batch)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        if step == 1 or step % 50 == 0:
            metrics = {f"train/{k}": float(v.item()) for k, v in parts.items()}
            metrics["step"] = step
            metrics["time/min"] = (time.time() - t0) / 60.0
            wandb.log(metrics, step=step)
            print(metrics, flush=True)
        if step == 1 or step % cfg.val_every == 0 or step == cfg.steps:
            metrics = evaluate(model, val_loader, device)
            wandb.log(metrics, step=step)
            write_json(out_dir / "latest_val.json", {"step": step, **metrics})
        if step == 1 or step % cfg.save_every == 0 or step == cfg.steps:
            save_checkpoint(out_dir / f"wm_step_{step:07d}.pt", model, opt, step, cfg)
            save_checkpoint(out_dir / "latest.pt", model, opt, step, cfg)
            save_rollout_gif(model, val_ds, device, out_dir / "rollouts" / f"step_{step:07d}.gif")
    run.finish()


if __name__ == "__main__":
    main()
