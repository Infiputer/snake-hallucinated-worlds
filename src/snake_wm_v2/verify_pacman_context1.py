from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np

from .pacman_env import ACTION_VECTORS, BOARD, FRAME_SIZE, SCALE, PacmanEnv, pt


GHOST_COLORS = [(255, 79, 99), (85, 215, 255), (255, 157, 242), (255, 180, 71)]
BACKGROUND = (5, 5, 5)
WALL_BLUE = (30, 85, 255)
WALL_DARK = (0, 12, 80)
PELLET = (255, 230, 166)
PACMAN = (255, 210, 31)
WHITE = (255, 255, 255)


@dataclass(frozen=True)
class ParsedFrame:
    player: tuple[int, int]
    ghosts: tuple[tuple[int, int], ...]
    walls: frozenset[tuple[int, int]]
    pellets: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class PredictedStep:
    frame: np.ndarray
    pellet_event: bool
    death_event: bool
    pellets_remaining: int


def has_color(crop: np.ndarray, color: tuple[int, int, int]) -> bool:
    target = np.asarray(color, dtype=np.uint8)
    return bool(np.any(np.all(crop == target, axis=-1)))


def parse_frame(frame: np.ndarray) -> ParsedFrame:
    if frame.shape != (FRAME_SIZE, FRAME_SIZE, 3):
        raise ValueError(f"expected {(FRAME_SIZE, FRAME_SIZE, 3)} frame, got {frame.shape}")
    player: tuple[int, int] | None = None
    ghosts: list[tuple[int, int] | None] = [None] * len(GHOST_COLORS)
    walls: set[tuple[int, int]] = set()
    pellets: set[tuple[int, int]] = set()

    for y in range(BOARD):
        for x in range(BOARD):
            cell = pt(x, y)
            crop = frame[y * SCALE : (y + 1) * SCALE, x * SCALE : (x + 1) * SCALE]
            if has_color(crop, PACMAN):
                if player is not None:
                    raise ValueError(f"multiple Pac-Man cells: {player}, {cell}")
                player = cell
            for i, color in enumerate(GHOST_COLORS):
                if has_color(crop, color):
                    if ghosts[i] is not None:
                        raise ValueError(f"ghost {i} appears in multiple cells: {ghosts[i]}, {cell}")
                    ghosts[i] = cell
            if has_color(crop, WALL_BLUE) or has_color(crop, WALL_DARK):
                walls.add(cell)
            elif has_color(crop, PELLET):
                pellets.add(cell)

    if player is None:
        raise ValueError("could not parse Pac-Man position from frame")
    if any(g is None for g in ghosts):
        raise ValueError(f"could not parse all ghosts from frame: {ghosts}")
    return ParsedFrame(player, tuple(g for g in ghosts if g is not None), frozenset(walls), frozenset(pellets))


def blocked(cell: tuple[int, int], walls: frozenset[tuple[int, int]]) -> bool:
    return not (0 <= cell[0] < BOARD and 0 <= cell[1] < BOARD) or cell in walls


def render(parsed: ParsedFrame) -> np.ndarray:
    img = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
    img[:, :] = BACKGROUND
    for wall in parsed.walls:
        rect(img, wall, WALL_BLUE, inset=2)
        rect(img, wall, WALL_DARK, inset=5)
    for i, ghost in enumerate(parsed.ghosts):
        rect(img, ghost, GHOST_COLORS[i], inset=3)
        rect(img, ghost, WHITE, inset=6)
    for pellet in parsed.pellets:
        rect(img, pellet, PELLET, inset=6)
    rect(img, parsed.player, PACMAN, inset=2)
    return img


def rect(img: np.ndarray, cell: tuple[int, int], color: tuple[int, int, int], inset: int = 0) -> None:
    x0 = cell[0] * SCALE + inset
    y0 = cell[1] * SCALE + inset
    x1 = (cell[0] + 1) * SCALE - inset
    y1 = (cell[1] + 1) * SCALE - inset
    img[y0:y1, x0:x1] = color


def transition_from_frame(parsed: ParsedFrame, action: int) -> PredictedStep:
    dx, dy = ACTION_VECTORS[int(action)]
    nxt = pt(parsed.player[0] + int(dx), parsed.player[1] + int(dy))
    player = parsed.player if blocked(nxt, parsed.walls) else nxt

    pellets = set(parsed.pellets)
    pellet_event = player in pellets
    if pellet_event:
        pellets.remove(player)

    current_ghosts = set(parsed.ghosts)
    occupied: set[tuple[int, int]] = set()
    new_ghosts: list[tuple[int, int]] = []
    for ghost in parsed.ghosts:
        candidates: list[tuple[int, int, tuple[int, int]]] = []
        blocked_ghosts = (current_ghosts - {ghost}) | occupied
        for ghost_action, (gx, gy) in enumerate(ACTION_VECTORS):
            ghost_nxt = pt(ghost[0] + int(gx), ghost[1] + int(gy))
            if not blocked(ghost_nxt, parsed.walls) and ghost_nxt not in blocked_ghosts:
                dist = abs(ghost_nxt[0] - player[0]) + abs(ghost_nxt[1] - player[1])
                candidates.append((dist, ghost_action, ghost_nxt))
        candidates.sort()
        chosen = candidates[0][2] if candidates else ghost
        new_ghosts.append(chosen)
        occupied.add(chosen)

    death_event = player in new_ghosts
    next_parsed = ParsedFrame(player, tuple(new_ghosts), parsed.walls, frozenset(pellets))
    return PredictedStep(render(next_parsed), pellet_event, death_event, len(pellets))


def clone_env(env: PacmanEnv) -> PacmanEnv:
    clone = PacmanEnv(seed=0, random_map=False)
    clone.player = env.player
    clone.ghosts = list(env.ghosts)
    clone.walls = set(env.walls)
    clone.pellets = list(env.pellets)
    clone.status = env.status
    return clone


def assert_frame_state_matches_env(parsed: ParsedFrame, env: PacmanEnv) -> None:
    if parsed.player != env.player:
        raise AssertionError(f"player parse mismatch: parsed={parsed.player} env={env.player}")
    if tuple(parsed.ghosts) != tuple(env.ghosts):
        raise AssertionError(f"ghost parse mismatch: parsed={parsed.ghosts} env={env.ghosts}")
    if set(parsed.walls) != set(env.walls):
        raise AssertionError("wall parse mismatch")
    if set(parsed.pellets) != set(env.pellets):
        raise AssertionError("pellet parse mismatch")


def verify_env(env: PacmanEnv, rng: np.random.Generator, max_steps: int) -> tuple[int, int]:
    states = 0
    action_checks = 0
    for _ in range(max_steps):
        if env.status != "play":
            break
        parsed = parse_frame(env.frame)
        assert_frame_state_matches_env(parsed, env)
        for action in range(4):
            predicted = transition_from_frame(parsed, action)
            actual_env = clone_env(env)
            before_pellets = len(actual_env.pellets)
            actual = actual_env.step(action)
            actual_pellet_event = len(actual_env.pellets) == before_pellets - 1
            actual_death_event = actual_env.status == "dead"
            if predicted.pellet_event != actual_pellet_event:
                raise AssertionError(f"pellet event mismatch action={action}")
            if predicted.death_event != actual_death_event:
                raise AssertionError(f"death event mismatch action={action}")
            if predicted.pellets_remaining != len(actual_env.pellets):
                raise AssertionError(f"pellet count mismatch action={action}")
            if not np.array_equal(predicted.frame, actual.frame):
                raise AssertionError(f"next-frame mismatch action={action}")
            action_checks += 1
        env.step(int(rng.integers(0, 4)))
        states += 1
    return states, action_checks


def mirrored_inner_count(walls: set[tuple[int, int]]) -> int:
    inner = {w for w in walls if 0 < w[0] < BOARD - 1 and 0 < w[1] < BOARD - 1}
    return sum(1 for x, y in inner if (BOARD - 1 - x, y) in inner)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify Pac-Man is context-1 deterministic from RGB frame + action")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    total_states = 0
    total_action_checks = 0
    for random_map in (False, True):
        for seed in range(args.seeds):
            env = PacmanEnv(seed=seed, random_map=random_map)
            mirrored = mirrored_inner_count(set(env.walls))
            if mirrored:
                raise AssertionError(f"map symmetry check failed random_map={random_map} seed={seed}: {mirrored}")
            states, action_checks = verify_env(env, rng, args.max_steps)
            total_states += states
            total_action_checks += action_checks
    print(
        "context1_determinism_verified "
        f"seeds={args.seeds} modes=2 live_states={total_states} action_checks={total_action_checks}"
    )


if __name__ == "__main__":
    main()
