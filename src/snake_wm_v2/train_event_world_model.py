from __future__ import annotations

import argparse
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import wandb

from .common import WORLD_MODEL_SPECS, count_parameters, set_seed, wandb_kwargs, write_json
from .dataset import SnakeWorldModelDataset
from .event_model import EventSnakeWorldModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train context-1 Snake event world model")
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--variant", choices=tuple(WORLD_MODEL_SPECS), default="wm_1m")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--val-every", type=int, default=500)
    p.add_argument("--wandb-mode", choices=("auto", "online", "offline", "disabled"), default="auto")
    p.add_argument("--project", default="snake-hallucinated-worlds-event")
    return p.parse_args()


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def targets(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    apple = (batch["reward"] > 0).long()
    death = (batch["reward"] < 0).long()
    return apple, death


def make_class_weights(dataset: SnakeWorldModelDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rewards = np.asarray(dataset.file.rewards[dataset.indices], dtype=np.float32)
    apple_counts = np.bincount((rewards > 0).astype(np.int64), minlength=2).astype(np.float64)
    death_counts = np.bincount((rewards < 0).astype(np.int64), minlength=2).astype(np.float64)

    def weights(counts: np.ndarray) -> np.ndarray:
        total = max(float(counts.sum()), 1.0)
        out = total / np.maximum(2.0 * counts, 1.0)
        return np.clip(out, 0.1, 20.0).astype(np.float32)

    apple_w = weights(apple_counts)
    death_w = weights(death_counts)
    meta = {
        "apple_negative": float(apple_counts[0]),
        "apple_positive": float(apple_counts[1]),
        "death_negative": float(death_counts[0]),
        "death_positive": float(death_counts[1]),
        "apple_weight_0": float(apple_w[0]),
        "apple_weight_1": float(apple_w[1]),
        "death_weight_0": float(death_w[0]),
        "death_weight_1": float(death_w[1]),
    }
    return torch.tensor(apple_w, device=device), torch.tensor(death_w, device=device), meta


def compute_losses(
    model: EventSnakeWorldModel,
    batch: dict[str, torch.Tensor],
    apple_weights: torch.Tensor,
    death_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    out = model(batch["context"], batch["action"])
    apple_target, death_target = targets(batch)
    frame_l1 = F.l1_loss(out["frame"], batch["next_frame"])
    apple_ce = F.cross_entropy(out["apple_logits"], apple_target, weight=apple_weights)
    death_ce = F.cross_entropy(out["death_logits"], death_target, weight=death_weights)
    total = frame_l1 + 0.25 * apple_ce + 0.25 * death_ce
    return total, {
        "total": total.detach(),
        "frame_l1": frame_l1.detach(),
        "apple_ce": apple_ce.detach(),
        "death_ce": death_ce.detach(),
    }


def binary_metrics(pred: np.ndarray, target: np.ndarray, prefix: str) -> dict[str, float]:
    pred = pred.astype(bool)
    target = target.astype(bool)
    tp = int((pred & target).sum())
    fp = int((pred & ~target).sum())
    tn = int((~pred & ~target).sum())
    fn = int((~pred & target).sum())
    return {
        f"val/{prefix}_tp": tp,
        f"val/{prefix}_fp": fp,
        f"val/{prefix}_tn": tn,
        f"val/{prefix}_fn": fn,
        f"val/{prefix}_accuracy": (tp + tn) / max(tp + fp + tn + fn, 1),
        f"val/{prefix}_precision": tp / max(tp + fp, 1),
        f"val/{prefix}_recall": tp / max(tp + fn, 1),
        f"val/{prefix}_fpr": fp / max(fp + tn, 1),
    }


@torch.no_grad()
def evaluate(
    model: EventSnakeWorldModel,
    loader: DataLoader,
    device: torch.device,
    apple_weights: torch.Tensor,
    death_weights: torch.Tensor,
    max_batches: int = 16,
) -> dict[str, float]:
    model.eval()
    losses = {"total": [], "frame_l1": [], "apple_ce": [], "death_ce": []}
    apple_preds: list[np.ndarray] = []
    apple_targets: list[np.ndarray] = []
    death_preds: list[np.ndarray] = []
    death_targets: list[np.ndarray] = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        batch = batch_to_device(batch, device)
        _, parts = compute_losses(model, batch, apple_weights, death_weights)
        out = model(batch["context"], batch["action"])
        apple_target, death_target = targets(batch)
        for k in losses:
            losses[k].append(float(parts[k].item()))
        apple_preds.append(out["apple_logits"].argmax(dim=1).cpu().numpy())
        apple_targets.append(apple_target.cpu().numpy())
        death_preds.append(out["death_logits"].argmax(dim=1).cpu().numpy())
        death_targets.append(death_target.cpu().numpy())
    metrics = {f"val/{k}": float(np.mean(v)) for k, v in losses.items() if v}
    if apple_preds:
        metrics.update(binary_metrics(np.concatenate(apple_preds), np.concatenate(apple_targets), "apple"))
        metrics.update(binary_metrics(np.concatenate(death_preds), np.concatenate(death_targets), "death"))
    model.train()
    return metrics


@torch.no_grad()
def save_rollout_gif(model: EventSnakeWorldModel, dataset: SnakeWorldModelDataset, device: torch.device, path: Path, steps: int = 24) -> None:
    model.eval()
    sample = dataset[0]
    frame = sample["context"][-1:].to(device)
    rng = np.random.default_rng(0)
    frames = []
    for _ in range(steps):
        action = torch.tensor([int(rng.integers(0, 4))], device=device)
        out = model(frame, action)
        img = out["frame"].squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        frames.append((img * 255).astype(np.uint8))
        frame = out["frame"].detach()
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path.as_posix(), frames, fps=6)
    model.train()


def save_checkpoint(path: Path, model: EventSnakeWorldModel, opt: torch.optim.Optimizer, step: int, cfg: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "step": step,
        "model_config": model.config(),
        "args": vars(cfg),
        "param_count": count_parameters(model),
    }, path)


def main() -> None:
    cfg = parse_args()
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ds = SnakeWorldModelDataset(cfg.dataset, 1, "train")
    val_ds = SnakeWorldModelDataset(cfg.dataset, 1, "val")
    apple_weights, death_weights, class_meta = make_class_weights(train_ds, device)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(cfg.num_workers, 2), pin_memory=device.type == "cuda")
    train_iter = iter(train_loader)
    model = EventSnakeWorldModel(cfg.variant).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    scaler = GradScaler(enabled=device.type == "cuda")
    run = wandb.init(name=out_dir.name, config=vars(cfg), **wandb_kwargs(cfg.project, cfg.wandb_mode))
    write_json(out_dir / "metadata.json", {
        "dataset": cfg.dataset,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "param_count": count_parameters(model),
        "model_config": model.config(),
        "class_balance": class_meta,
        "args": vars(cfg),
    })
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
            loss, parts = compute_losses(model, batch, apple_weights, death_weights)
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
            metrics = evaluate(model, val_loader, device, apple_weights, death_weights)
            wandb.log(metrics, step=step)
            write_json(out_dir / "latest_val.json", {"step": step, **metrics})
        if step == 1 or step % cfg.save_every == 0 or step == cfg.steps:
            save_checkpoint(out_dir / f"wm_step_{step:07d}.pt", model, opt, step, cfg)
            save_checkpoint(out_dir / "latest.pt", model, opt, step, cfg)
            save_rollout_gif(model, val_ds, device, out_dir / "rollouts" / f"step_{step:07d}.gif")
    run.finish()


if __name__ == "__main__":
    main()

