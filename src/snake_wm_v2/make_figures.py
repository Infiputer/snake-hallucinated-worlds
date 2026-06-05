from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .dataset import SnakeWorldModelDataset
from .env import DOWN, RIGHT, SnakeEnv
from .train_policy import load_world_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create paper figures")
    p.add_argument("--dataset", required=True)
    p.add_argument("--world-model", default=None)
    p.add_argument("--eval-summary", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100)
    return p.parse_args()


def to_pil(frame: np.ndarray, scale: int = 3) -> Image.Image:
    img = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    return img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST) if scale != 1 else img


def label(img: Image.Image, text: str, bar_h: int = 24) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + bar_h), "white")
    out.paste(img, (0, bar_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    draw.text((6, 4), text, fill=(20, 20, 20), font=font)
    return out


def label_bottom(img: Image.Image, text: str, bar_h: int = 42) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + bar_h), "white")
    out.paste(img, (0, 0))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except Exception:
        font = ImageFont.load_default()
    draw.multiline_text((6, img.height + 5), text, fill=(20, 20, 20), font=font, spacing=2)
    return out


def arrow_cell(height: int, width: int = 48) -> Image.Image:
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = height // 2 - 10
    x0, x1 = 8, width - 10
    draw.line((x0, y, x1, y), fill=(35, 35, 35), width=4)
    draw.polygon([(x1, y), (x1 - 10, y - 7), (x1 - 10, y + 7)], fill=(35, 35, 35))
    return img


def hstack(images: list[Image.Image], gap: int = 8) -> Image.Image:
    out = Image.new("RGB", (sum(i.width for i in images) + gap * (len(images) - 1), max(i.height for i in images)), "white")
    x = 0
    for img in images:
        out.paste(img, (x, 0))
        x += img.width + gap
    return out


def vstack(images: list[Image.Image], gap: int = 8) -> Image.Image:
    out = Image.new("RGB", (max(i.width for i in images), sum(i.height for i in images) + gap * (len(images) - 1)), "white")
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.height + gap
    return out


def make_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.clip(np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.float32) * 3.0, 0, 255).astype(np.uint8)


def make_environment_sequence(out_dir: Path) -> None:
    env = SnakeEnv(seed=7)
    frames: list[tuple[np.ndarray, str]] = [(env.reset().frame, "Initial board\nsnake, rocks, apples")]
    result = None
    for action in [RIGHT, RIGHT, RIGHT, RIGHT]:
        result = env.step(action)
    frames.append((result.frame, "Policy actions\nmove toward apple"))
    for action in [RIGHT, DOWN, DOWN]:
        result = env.step(action)
    frames.append((result.frame, "Apple eaten\nsnake length increases"))

    death_env = SnakeEnv(seed=11)
    death_env.reset()
    death_env.snake = [(7, 7), (7, 8), (8, 8), (8, 7), (9, 7), (9, 8), (9, 9), (8, 9), (7, 9)]
    death_env.direction = RIGHT
    death_env.last_action = RIGHT
    frames.append((death_env.frame, "Before collision\nhead faces its body"))
    result = death_env.step(RIGHT)
    frames.append((result.frame, "Self-intersection\nepisode ends"))

    cells = [label_bottom(to_pil(frame, 2), text) for frame, text in frames]
    pieces: list[Image.Image] = []
    for i, cell in enumerate(cells):
        pieces.append(cell)
        if i + 1 < len(cells):
            pieces.append(arrow_cell(cell.height))
    hstack(pieces, gap=0).save(out_dir / "snake_environment_sequence.png")


def plot_eval_summary(path: Path, out_dir: Path) -> None:
    rows = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return
    hall = np.asarray([float(r["hallucinated_return"]) for r in rows])
    real = np.asarray([float(r["real_return"]) for r in rows])
    colors = {"small": "#2a9d8f", "medium": "#e9c46a", "large": "#e76f51"}
    fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=180)
    for policy in sorted(set(r["policy"] for r in rows)):
        mask = np.asarray([r["policy"] == policy for r in rows])
        ax.scatter(hall[mask], real[mask], label=policy, color=colors.get(policy), s=38)
    mn = min(float(hall.min()), float(real.min())) - 0.5
    mx = max(float(hall.max()), float(real.max())) + 0.5
    ax.plot([mn, mx], [mn, mx], color="0.65", linewidth=1)
    ax.set_xlabel("Hallucinated return")
    ax.set_ylabel("Real-simulator return")
    ax.set_title("Transfer: hallucinated vs real")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "hallucinated_vs_real.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = np.load(Path(args.dataset) / "frames.npy", mmap_mode="r")
    make_environment_sequence(out_dir)
    meta_colors = sorted(set(tuple(map(int, frames[i, 0, 0])) for i in np.linspace(0, len(frames) - 1, min(len(frames), 500), dtype=int)))
    example = label(to_pil(np.asarray(frames[min(20, len(frames) - 1)]), 5), f"Simulator Snake frame; sampled corner colors={meta_colors}", 30)
    example.save(out_dir / "snake_simulator_frame.png")
    if args.world_model:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_world_model(args.world_model, device)
        ds = SnakeWorldModelDataset(args.dataset, model.context, "val")
        sample = ds[0]
        context = sample["context"].unsqueeze(0).to(device)
        prev_reward = sample["prev_reward"].view(1).to(device)
        actions = np.load(Path(args.dataset) / "actions.npy", mmap_mode="r")
        indices = ds.indices[:args.steps]
        real_next_idx = np.load(Path(args.dataset) / "next_frame_indices.npy", mmap_mode="r")[indices]
        real = frames[real_next_idx].astype(np.uint8)
        preds = []
        pred_lengths = []
        pred_done = []
        with torch.no_grad():
            for idx in indices:
                action = torch.tensor([int(actions[idx])], device=device)
                out = model(context, prev_reward, action)
                pred = out["frame"].squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
                preds.append((pred * 255).astype(np.uint8))
                pred_lengths.append(float(out["length"].item()))
                pred_done.append(float(torch.sigmoid(out["done_logit"]).item()))
                nf = out["frame"].detach()
                context = torch.cat([context[:, 1:], nf.unsqueeze(1)], dim=1) if model.context > 1 else nf.unsqueeze(1)
                prev_reward = out["reward"].detach()
        pred_arr = np.stack(preds)
        mae = np.abs(real.astype(np.float32) - pred_arr.astype(np.float32)).mean(axis=(1, 2, 3)) / 255.0
        steps = [s for s in [1, 5, 10, 25, 50, 75, 100] if s <= len(indices)]
        real_cells = [label(to_pil(real[s - 1], 2), f"t={s}", 20) for s in steps]
        pred_cells = [label(to_pil(pred_arr[s - 1], 2), f"t={s}", 20) for s in steps]
        diff_cells = [label(to_pil(make_diff(real[s - 1], pred_arr[s - 1]), 2), f"MAE={mae[s - 1]:.3f}", 20) for s in steps]
        grid = vstack([hstack([label(Image.new("RGB", (130, 256), "white"), "Real", 20)] + real_cells), hstack([label(Image.new("RGB", (130, 256), "white"), "World model", 20)] + pred_cells), hstack([label(Image.new("RGB", (130, 256), "white"), "|diff| x3", 20)] + diff_cells)])
        grid.save(out_dir / "wm_vs_sim_rollout.png")
        fig, ax1 = plt.subplots(figsize=(7.2, 3.2), dpi=180)
        xs = np.arange(1, len(mae) + 1)
        ax1.plot(xs, mae, color="#0f4c81", label="pixel MAE")
        ax1.set_xlabel("Autoregressive step")
        ax1.set_ylabel("Mean absolute pixel error")
        ax1.grid(True, alpha=0.25)
        ax2 = ax1.twinx()
        ax2.plot(xs, pred_done, color="#d1495b", alpha=0.8, label="done probability")
        ax2.set_ylabel("Predicted done probability")
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left")
        fig.tight_layout()
        fig.savefig(out_dir / "wm_drift_curve.png", bbox_inches="tight")
        plt.close(fig)
    if args.eval_summary:
        plot_eval_summary(Path(args.eval_summary), out_dir)
    print(out_dir.as_posix())


if __name__ == "__main__":
    main()
