from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .dataset import SnakeWorldModelDataset
from .event_model import EventSnakeWorldModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Pac-Man real-vs-world-model rollout figure")
    p.add_argument("--world-model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--out", default="papers/figures/pacman_wm_vs_sim_rollout.png")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed-index", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/lato/Lato-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def to_uint8(tensor: torch.Tensor) -> np.ndarray:
    arr = tensor.detach().clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def image_cell(arr: np.ndarray, size: int = 128) -> Image.Image:
    return Image.fromarray(arr).resize((size, size), Image.Resampling.NEAREST)


def main() -> None:
    cfg = parse_args()
    device = torch.device(cfg.device)
    ckpt = torch.load(cfg.world_model, map_location="cpu")
    model = EventSnakeWorldModel(ckpt["model_config"]["variant"]).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = SnakeWorldModelDataset(cfg.dataset, 1, "all")
    start = min(max(int(cfg.seed_index), 0), max(len(ds) - cfg.steps - 1, 0))
    sample = ds[start]
    frame = sample["context"][-1:].to(device)
    real_frames: list[np.ndarray] = []
    pred_frames: list[np.ndarray] = []
    diffs: list[np.ndarray] = []
    labels: list[str] = []

    with torch.no_grad():
        for offset in range(cfg.steps):
            item = ds[start + offset]
            action = item["action"].view(1).to(device)
            out = model(frame, action)
            pred = to_uint8(out["frame"])
            real = (item["next_frame"].permute(1, 2, 0).numpy() * 255.0 + 0.5).astype(np.uint8)
            diff = np.clip(np.abs(real.astype(np.int16) - pred.astype(np.int16)) * 3, 0, 255).astype(np.uint8)
            real_frames.append(real)
            pred_frames.append(pred)
            diffs.append(diff)
            labels.append(f"t={offset + 1:02d}")
            frame = out["frame"].detach()

    cell = 128
    label_w = 108
    cell_w = cell + 18
    top = 70
    row_h = 168
    width = label_w + cfg.steps * cell_w + 32
    height = top + 3 * row_h + 36
    canvas = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 18), "Pac-Man world-model rollout against dataset frames", font=font(24), fill=(242, 234, 210))
    for i, label in enumerate(labels):
        draw.text((label_w + i * cell_w + 44, top - 28), label, font=font(13), fill=(169, 180, 148))
    rows = [("Real", real_frames, (155, 216, 92)), ("WM", pred_frames, (114, 167, 255)), ("Diff x3", diffs, (255, 114, 95))]
    for ridx, (name, frames, color) in enumerate(rows):
        y = top + ridx * row_h
        draw.text((18, y + 52), name, font=font(16), fill=color)
        for i, arr in enumerate(frames):
            x = label_w + i * cell_w
            draw.rounded_rectangle((x - 5, y - 5, x + cell + 5, y + cell + 5), radius=8, fill=(7, 10, 6), outline=(52, 67, 41))
            canvas.paste(image_cell(arr, cell), (x, y))
    out = Path(cfg.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(out)


if __name__ == "__main__":
    main()
