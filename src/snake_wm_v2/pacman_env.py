from __future__ import annotations

from dataclasses import dataclass

import numpy as np


UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_VECTORS = np.asarray([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int64)
BOARD = 16
SCALE = 16
FRAME_SIZE = BOARD * SCALE


@dataclass
class PacmanStep:
    frame: np.ndarray
    reward: float
    done: bool
    pellets: int
    status: str


def pt(x: int, y: int) -> tuple[int, int]:
    return (int(x), int(y))


def border() -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(BOARD):
        cells.extend([pt(i, 0), pt(i, BOARD - 1), pt(0, i), pt(BOARD - 1, i)])
    return cells


FIXED_INNER_WALLS = [
    pt(2, 2), pt(3, 2), pt(4, 2), pt(6, 2), pt(9, 2), pt(11, 2), pt(12, 2), pt(13, 2),
    pt(2, 4), pt(4, 4), pt(5, 4), pt(7, 4), pt(8, 4), pt(10, 4), pt(11, 4), pt(13, 4),
    pt(4, 5), pt(11, 5),
    pt(2, 6), pt(4, 6), pt(6, 6), pt(9, 6), pt(11, 6), pt(13, 6),
    pt(6, 7), pt(9, 7),
    pt(3, 8), pt(4, 8), pt(6, 8), pt(9, 8), pt(11, 8), pt(12, 8),
    pt(2, 9), pt(4, 9), pt(6, 9), pt(9, 9), pt(11, 9), pt(13, 9),
    pt(4, 10), pt(11, 10),
    pt(2, 11), pt(4, 11), pt(5, 11), pt(7, 11), pt(8, 11), pt(10, 11), pt(11, 11), pt(13, 11),
    pt(2, 13), pt(3, 13), pt(4, 13), pt(6, 13), pt(9, 13), pt(11, 13), pt(12, 13), pt(13, 13),
]


class PacmanEnv:
    """Small deterministic Pac-Man-like visual environment.

    The reward interface intentionally matches the Snake event setup:
    pellet eaten is a discrete +1 event and death is a discrete terminal event
    with reward -1. There are no floating-point shaping rewards.
    """

    def __init__(self, seed: int = 0, random_map: bool = False):
        self.rng = np.random.default_rng(seed)
        self.random_map = bool(random_map)
        self.reset()

    def reset(self) -> PacmanStep:
        self.player = pt(8, 12)
        self.ghosts = [pt(7, 6), pt(8, 6), pt(7, 9), pt(8, 9)]
        self.walls = set(self._random_walls() if self.random_map else border() + FIXED_INNER_WALLS)
        self.pellets = [p for p in self._reachable(self.player) if p != self.player and p not in self.ghosts]
        self.status = "play"
        return self._result(0.0, False)

    def _random_walls(self) -> list[tuple[int, int]]:
        start = self.player
        safe = {start, *self.ghosts}
        for _ in range(200):
            walls = set(border())
            for y in range(2, 14):
                for x in range(1, 8):
                    a, b = pt(x, y), pt(BOARD - 1 - x, y)
                    if a in safe or b in safe:
                        continue
                    if self.rng.random() < 0.24:
                        walls.add(a)
                        walls.add(b)
            if len(self._reachable(start, walls)) >= 125:
                return sorted(walls)
        return border() + FIXED_INNER_WALLS

    def _reachable(self, start: tuple[int, int], walls: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
        walls = self.walls if walls is None else walls
        q = [start]
        seen = {start}
        for cell in q:
            for dx, dy in ACTION_VECTORS:
                nxt = pt(cell[0] + int(dx), cell[1] + int(dy))
                if 0 <= nxt[0] < BOARD and 0 <= nxt[1] < BOARD and nxt not in walls and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return q

    def _blocked(self, cell: tuple[int, int]) -> bool:
        return not (0 <= cell[0] < BOARD and 0 <= cell[1] < BOARD) or cell in self.walls

    def _move_ghosts(self) -> None:
        new_ghosts = []
        current_ghosts = set(self.ghosts)
        occupied: set[tuple[int, int]] = set()
        for ghost in self.ghosts:
            candidates = []
            blocked_ghosts = (current_ghosts - {ghost}) | occupied
            for action, (dx, dy) in enumerate(ACTION_VECTORS):
                nxt = pt(ghost[0] + int(dx), ghost[1] + int(dy))
                if not self._blocked(nxt) and nxt not in blocked_ghosts:
                    dist = abs(nxt[0] - self.player[0]) + abs(nxt[1] - self.player[1])
                    candidates.append((dist, action, nxt))
            candidates.sort()
            chosen = candidates[0][2] if candidates else ghost
            new_ghosts.append(chosen)
            occupied.add(chosen)
        self.ghosts = new_ghosts

    def step(self, action: int) -> PacmanStep:
        if self.status != "play":
            return self._result(0.0, True)
        dx, dy = ACTION_VECTORS[int(action)]
        nxt = pt(self.player[0] + int(dx), self.player[1] + int(dy))
        if not self._blocked(nxt):
            self.player = nxt
        reward = 0.0
        if self.player in self.pellets:
            self.pellets.remove(self.player)
            reward = 1.0
        self._move_ghosts()
        if self.player in self.ghosts:
            self.status = "dead"
            return self._result(-1.0, True)
        if not self.pellets:
            self.status = "win"
            return self._result(reward, True)
        return self._result(reward, False)

    def _result(self, reward: float, done: bool) -> PacmanStep:
        return PacmanStep(self.frame, float(reward), bool(done), int(len(self.pellets)), self.status)

    @property
    def frame(self) -> np.ndarray:
        img = np.zeros((FRAME_SIZE, FRAME_SIZE, 3), dtype=np.uint8)
        img[:, :] = (5, 5, 5)
        for wall in self.walls:
            self._rect(img, wall, (30, 85, 255), inset=2)
            self._rect(img, wall, (0, 12, 80), inset=5)
        for pellet in self.pellets:
            self._rect(img, pellet, (255, 230, 166), inset=6)
        colors = [(255, 79, 99), (85, 215, 255), (255, 157, 242), (255, 180, 71)]
        for i, ghost in enumerate(self.ghosts):
            self._rect(img, ghost, colors[i % len(colors)], inset=3)
            self._rect(img, ghost, (255, 255, 255), inset=6)
        self._rect(img, self.player, (255, 210, 31), inset=2)
        return img

    @staticmethod
    def _rect(img: np.ndarray, cell: tuple[int, int], color: tuple[int, int, int], inset: int = 0) -> None:
        x0 = cell[0] * SCALE + inset
        y0 = cell[1] * SCALE + inset
        x1 = (cell[0] + 1) * SCALE - inset
        y1 = (cell[1] + 1) * SCALE - inset
        img[y0:y1, x0:x1] = color
