from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np


BOARD = 16
WIN_LENGTH = 16
TILE_SIZE = 8
CANVAS_SIZE = BOARD * TILE_SIZE
UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
ACTION_VECTORS = np.array([[0, -1], [0, 1], [-1, 0], [1, 0]], dtype=np.int32)
OPPOSITE = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
ROCKS = [(8, 5), (8, 6), (9, 5), (9, 6), (4, 12), (5, 12), (11, 8), (12, 8)]
ROCKS_SET = set(ROCKS)
INITIAL_APPLES = [(12, 3), (3, 4), (10, 11), (6, 6), (13, 13), (2, 2), (14, 2), (2, 13), (7, 3), (13, 6), (6, 13), (1, 7), (14, 10), (10, 2)]


@dataclass
class StepResult:
    frame: np.ndarray
    length: float
    reward: float
    status: str
    done: bool
    last_action: int
    apples_remaining: int


class SnakeEnv:
    """Deterministic Snake environment with no terminal visual overlay."""

    def __init__(self, seed: int | None = None):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> StepResult:
        self.snake: List[Tuple[int, int]] = [(5, 9), (4, 9), (3, 9)]
        self.direction = RIGHT
        self.last_action = RIGHT
        self.apples: List[Tuple[int, int]] = list(INITIAL_APPLES)
        self.status = "play"
        self._rock_set = ROCKS_SET
        return self._result(0.0, False)

    def _result(self, reward: float, done: bool) -> StepResult:
        return StepResult(self.frame, float(len(self.snake)), float(reward), self.status, bool(done), int(self.last_action), len(self.apples))

    def _apple_index(self, x: int, y: int) -> int:
        for i, (ax, ay) in enumerate(self.apples):
            if ax == x and ay == y:
                return i
        return -1

    def _inside(self, x: int, y: int) -> bool:
        return 0 <= x < BOARD and 0 <= y < BOARD

    def _render_board(self) -> np.ndarray:
        frame = np.zeros((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8)
        light = np.array([170, 215, 81], dtype=np.uint8)
        dark = np.array([162, 209, 73], dtype=np.uint8)
        for y in range(BOARD):
            for x in range(BOARD):
                frame[y * TILE_SIZE:(y + 1) * TILE_SIZE, x * TILE_SIZE:(x + 1) * TILE_SIZE] = light if (x + y) % 2 == 0 else dark
        return frame

    def _render(self) -> np.ndarray:
        frame = self._render_board()
        for x, y in self._rock_set:
            x0, y0 = x * TILE_SIZE + 1, y * TILE_SIZE + 1
            x1, y1 = x0 + TILE_SIZE - 2, y0 + TILE_SIZE - 2
            cv2.rectangle(frame, (x0, y0), (x1, y1), (131, 183, 68), thickness=-1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (67, 99, 28), thickness=1)
        for ax, ay in self.apples:
            cx = (ax + 0.5) * TILE_SIZE
            cy = (ay + 0.54) * TILE_SIZE
            center = (int(round(cx)), int(round(cy)))
            radius = max(1, int(round(TILE_SIZE * 0.26)))
            cv2.circle(frame, center, radius, (231, 71, 60), thickness=-1)
            cv2.line(frame, (center[0] + 1, center[1] - int(TILE_SIZE * 0.32)), (center[0] - 1, center[1] - int(TILE_SIZE * 0.32)), (122, 74, 29), 1)
            cv2.ellipse(frame, (int(round(cx + TILE_SIZE * 0.15)), int(round(cy - TILE_SIZE * 0.28))), (int(round(TILE_SIZE * 0.14)), int(round(TILE_SIZE * 0.08))), -31, 0, 360, (79, 174, 63), thickness=-1)
        for i, (x, y) in enumerate(reversed(self.snake)):
            x0, y0 = x * TILE_SIZE + 1, y * TILE_SIZE + 1
            x1, y1 = x0 + TILE_SIZE - 2, y0 + TILE_SIZE - 2
            color = (59, 141, 242) if i > 0 else (47, 128, 237)
            cv2.rectangle(frame, (x0, y0), (x1, y1), color, thickness=-1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (16, 58, 120), thickness=1)
        if self.snake and self.status == "play":
            head_x, head_y = self.snake[0]
            dir_xy = ACTION_VECTORS[self.direction]
            px, py = -dir_xy[1], dir_xy[0]
            cx = int(round((head_x + 0.5) * TILE_SIZE + dir_xy[0] * TILE_SIZE * 0.16))
            cy = int(round((head_y + 0.5) * TILE_SIZE + dir_xy[1] * TILE_SIZE * 0.16))
            for side in (-1, 1):
                ex = int(round(cx + px * side * TILE_SIZE * 0.11))
                ey = int(round(cy + py * side * TILE_SIZE * 0.11))
                cv2.circle(frame, (ex, ey), max(1, int(round(TILE_SIZE * 0.075))), (255, 255, 255), thickness=-1)
                cv2.circle(frame, (int(round(ex + dir_xy[0] * TILE_SIZE * 0.025)), int(round(ey + dir_xy[1] * TILE_SIZE * 0.025))), max(1, int(round(TILE_SIZE * 0.033))), (17, 24, 39), thickness=-1)
        return frame

    @property
    def frame(self) -> np.ndarray:
        return self._render()

    def step(self, action: int) -> StepResult:
        if self.status != "play":
            return self._result(0.0, True)
        if action not in (UP, DOWN, LEFT, RIGHT):
            action = self.direction
        if action == OPPOSITE.get(self.direction, self.direction):
            action = self.direction
        dx, dy = ACTION_VECTORS[action]
        head_x, head_y = self.snake[0]
        next_x, next_y = head_x + int(dx), head_y + int(dy)
        apple_idx = self._apple_index(next_x, next_y)
        eating = apple_idx >= 0
        body = self.snake if eating else self.snake[:-1]
        hit = not self._inside(next_x, next_y) or (next_x, next_y) in self._rock_set or any(seg_x == next_x and seg_y == next_y for seg_x, seg_y in body)
        self.last_action = int(action)
        if hit:
            self.status = "dead"
            return self._result(0.0, True)
        self.direction = int(action)
        self.snake = [(next_x, next_y)] + self.snake
        reward = 0.0
        if eating:
            self.apples.pop(apple_idx)
            reward = 1.0
        else:
            self.snake.pop()
        if len(self.snake) >= WIN_LENGTH:
            self.status = "win"
            return self._result(reward, True)
        return self._result(reward, False)
