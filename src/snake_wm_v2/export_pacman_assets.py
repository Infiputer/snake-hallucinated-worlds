from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from .pacman_env import PacmanEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Pac-Man environment images for README/paper")
    p.add_argument("--out", default="images")
    p.add_argument("--paper-out", default="papers/figures")
    p.add_argument("--seed", type=int, default=123)
    return p.parse_args()


def font(size: int) -> ImageFont.ImageFont:
    for path in ("/usr/share/fonts/truetype/lato/Lato-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


def labeled_frame(frame, label: str) -> Image.Image:
    img = Image.fromarray(frame).resize((256, 256), Image.Resampling.NEAREST)
    canvas = Image.new("RGB", (300, 324), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((0, 0, 299, 323), radius=18, fill=(7, 10, 6), outline=(52, 67, 41))
    canvas.paste(img, (22, 46))
    draw.text((22, 16), label, fill=(242, 234, 210), font=font(18))
    return canvas


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    paper = Path(args.paper_out)
    out.mkdir(parents=True, exist_ok=True)
    paper.mkdir(parents=True, exist_ok=True)

    env = PacmanEnv(seed=args.seed, random_map=False)
    sequence = [("start", env.reset().frame)]
    for label, action in [("move right", 3), ("move up", 0), ("pellet path", 0), ("ghosts move", 2)]:
        sequence.append((label, env.step(action).frame))

    sheet = Image.new("RGB", (4 * 300 + 3 * 16, 324), (0, 0, 0))
    for i, (label, frame) in enumerate(sequence[:4]):
        sheet.paste(labeled_frame(frame, label), (i * 316, 0))
    sheet.save(out / "pacman_environment_sequence.png")
    sheet.save(paper / "pacman_environment_sequence.png")

    env = PacmanEnv(seed=args.seed, random_map=True)
    frames = [env.reset().frame]
    for action in [3, 3, 0, 0, 2, 2, 1, 3, 0, 0, 3, 3]:
        frames.append(env.step(action).frame)
    imageio.mimsave((out / "pacman_random_map.gif").as_posix(), frames, fps=4)

    Image.fromarray(frames[0]).resize((512, 512), Image.Resampling.NEAREST).save(out / "pacman_random_map.png")
    Image.fromarray(frames[0]).resize((512, 512), Image.Resampling.NEAREST).save(paper / "pacman_random_map.png")
    print(out / "pacman_environment_sequence.png")
    print(out / "pacman_random_map.gif")


if __name__ == "__main__":
    main()
