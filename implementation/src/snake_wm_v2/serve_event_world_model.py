from __future__ import annotations

import argparse
import base64
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image

from .env import BOARD, DOWN, INITIAL_APPLES, LEFT, RIGHT, ROCKS_SET, UP, SnakeEnv
from .event_model import EventSnakeWorldModel


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Snake World Model Simulator</title>
  <style>
    :root {
      --ink: #f2ead2;
      --muted: #a9b494;
      --paper: #11170f;
      --panel: #182114;
      --line: #344329;
      --accent: #9bd85c;
      --danger: #ff725f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background: #000;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    }
    main {
      width: min(1040px, calc(100vw - 32px));
      margin: 34px auto;
      display: grid;
      grid-template-columns: minmax(280px, 560px) 1fr;
      gap: 28px;
      align-items: start;
    }
    h1 {
      font-size: clamp(34px, 5vw, 68px);
      line-height: .9;
      letter-spacing: -0.05em;
      margin: 0 0 14px;
    }
    p { color: var(--muted); font-size: 18px; line-height: 1.45; margin: 0 0 18px; }
    .screen {
      background: #070a06;
      border: 1px solid #2b3a22;
      border-radius: 22px;
      padding: 18px;
      box-shadow: 0 28px 80px rgba(0, 0, 0, .55), inset 0 0 0 1px rgba(255,255,255,.03);
    }
    img {
      display: block;
      width: 100%;
      aspect-ratio: 1;
      image-rendering: pixelated;
      border-radius: 12px;
      background: #6fb35c;
    }
    .card {
      background: rgba(24, 33, 20, .86);
      border: 1px solid var(--line);
      border-radius: 22px;
      padding: 24px;
      box-shadow: 0 18px 60px rgba(0, 0, 0, .36), inset 0 0 0 1px rgba(255,255,255,.025);
      backdrop-filter: blur(8px);
    }
    .stats {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin: 18px 0;
    }
    .stat {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255,255,255,.045);
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .08em;
      margin-bottom: 4px;
    }
    .value {
      font: 700 25px/1.1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(3, 74px);
      gap: 10px;
      justify-content: center;
      margin-top: 20px;
    }
    button {
      border: 1px solid #2a3b25;
      background: #24351e;
      color: #f2ead2;
      border-radius: 16px;
      font: 700 18px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      padding: 16px 14px;
      cursor: pointer;
      box-shadow: 0 8px 0 #070b06;
      transform: translateY(0);
    }
    button:active { transform: translateY(5px); box-shadow: 0 3px 0 #070b06; }
    .gear {
      position: fixed;
      right: 18px;
      top: 18px;
      z-index: 30;
      width: 54px;
      height: 54px;
      padding: 0;
      border-radius: 50%;
      font-size: 24px;
      box-shadow: 0 7px 0 #121d10;
    }
    .modal {
      position: fixed;
      inset: 0;
      z-index: 20;
      display: none;
      place-items: center;
      padding: 18px;
      background: rgba(5, 8, 5, .72);
    }
    .modal.open { display: grid; }
    .editor {
      width: min(620px, 100%);
      max-height: calc(100vh - 36px);
      overflow: auto;
      background: #121a0f;
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 30px 90px rgba(0, 0, 0, .28);
    }
    .editorHead {
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: start;
      margin-bottom: 14px;
    }
    .editor h2 {
      margin: 0 0 8px;
      font-size: 32px;
      letter-spacing: -.04em;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(16, minmax(13px, 1fr));
      border: 1px solid #769f45;
      border-radius: 14px;
      overflow: hidden;
      aspect-ratio: 1;
      background: #26351f;
      width: min(100%, 500px);
      margin: 12px auto 0;
    }
    .cell {
      border: 0;
      border-radius: 0;
      padding: 0;
      box-shadow: none;
      min-width: 0;
      background: #3f642d;
      position: relative;
    }
    .cell.light { background: #466d31; }
    .cell.dark { background: #263b20; }
    .cell.rock { background: #1f2d17; cursor: not-allowed; }
    .cell.snake { background: #2f80ed; cursor: not-allowed; }
    .cell.apple::after {
      content: "";
      position: absolute;
      inset: 23%;
      border-radius: 50%;
      background: #e7473c;
      box-shadow: inset 0 -2px 0 rgba(0,0,0,.14);
    }
    .editorActions {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(104px, 1fr));
      gap: 10px;
      margin-top: 12px;
    }
    .editorActions button, .editorHead button {
      background: #355d25;
      border-color: #5b873d;
      font-size: 14px;
      box-shadow: 0 6px 0 #0c1309;
      padding: 12px 10px;
      min-height: 48px;
    }
    .wide {
      display: flex;
      gap: 10px;
      margin-top: 16px;
    }
    .wide button {
      width: 100%;
      background: var(--accent);
      border-color: #5b873d;
      box-shadow: 0 8px 0 #0c1309;
    }
    .status {
      margin-top: 16px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(155,216,92,.10);
      color: var(--accent);
      font-weight: 700;
    }
    .status.dead {
      background: rgba(255,114,95,.12);
      color: var(--danger);
    }
    code {
      background: rgba(242,234,210,.10);
      padding: 2px 6px;
      border-radius: 7px;
    }
    @media (max-width: 820px) {
      main { grid-template-columns: 1fr; margin: 18px auto; }
      .screen { order: 2; }
      .card { order: 1; }
    }
  </style>
</head>
<body>
  <button id="settings" class="gear" title="Initial frame settings">⚙</button>
  <div id="modal" class="modal">
    <section class="editor">
      <div class="editorHead">
        <div>
          <h2>Initial WM frame</h2>
          <p>Click cells to edit apples or rocks. The snake stays fixed. Press start to seed the world model from this frame.</p>
        </div>
        <button id="closeSettings">close</button>
      </div>
      <div id="appleCount" class="label">editing apples | apples: 0 | rocks: 0</div>
      <div id="grid" class="grid"></div>
      <div class="editorActions">
        <button id="editApples">edit apples</button>
        <button id="editRocks">edit rocks</button>
        <button id="defaultLayout">default</button>
        <button id="clearMode">clear mode</button>
        <button id="randomApples">random apples</button>
        <button id="randomRocks">random rocks</button>
        <button id="startCustom">start WM</button>
      </div>
    </section>
  </div>
  <main>
    <section class="screen">
      <img id="frame" alt="World model predicted Snake frame" />
    </section>
    <section class="card">
      <h1>Snake, hallucinated.</h1>
      <p>This is the learned event world model, not the real simulator. Press arrow keys or WASD; each move asks the WM to predict the next RGB frame plus apple/death events.</p>
      <div class="stats">
        <div class="stat"><div class="label">step</div><div class="value" id="step">0</div></div>
        <div class="stat"><div class="label">apples</div><div class="value" id="apples">0</div></div>
        <div class="stat"><div class="label">P(apple)</div><div class="value" id="pApple">0.00</div></div>
        <div class="stat"><div class="label">P(death)</div><div class="value" id="pDeath">0.00</div></div>
      </div>
      <div class="controls">
        <div></div><button data-action="0">↑</button><div></div>
        <button data-action="2">←</button><button id="reset">reset</button><button data-action="3">→</button>
        <div></div><button data-action="1">↓</button><div></div>
      </div>
      <div class="wide">
        <button id="continue">continue after death</button>
      </div>
      <div id="status" class="status">ready</div>
      <p style="margin-top:18px">Action ids: <code>0 up</code>, <code>1 down</code>, <code>2 left</code>, <code>3 right</code>.</p>
    </section>
  </main>
  <script>
    const frame = document.getElementById("frame");
    const ids = {
      step: document.getElementById("step"),
      apples: document.getElementById("apples"),
      pApple: document.getElementById("pApple"),
      pDeath: document.getElementById("pDeath"),
      status: document.getElementById("status"),
    };
    let busy = false;
    let allowDeadStep = false;
    let editor = {board: 16, mode: "apple", apples: [], default_apples: [], rocks: [], default_rocks: [], snake: []};
    const grid = document.getElementById("grid");
    const modal = document.getElementById("modal");
    const appleCount = document.getElementById("appleCount");

    function draw(s) {
      frame.src = s.image;
      ids.step.textContent = s.step;
      ids.apples.textContent = s.predicted_apples;
      ids.pApple.textContent = Number(s.apple_prob).toFixed(2);
      ids.pDeath.textContent = Number(s.death_prob).toFixed(2);
      ids.status.textContent = s.done ? "death head fired; reset or continue" : "running";
      ids.status.className = "status" + (s.done ? " dead" : "");
    }
    async function post(path, body = {}) {
      const res = await fetch(path, {
        method: "POST",
        headers: {"content-type": "application/json"},
        body: JSON.stringify(body)
      });
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }
    function keyOf(x, y) { return `${x},${y}`; }
    function cellsSet(items) { return new Set(items.map(p => keyOf(p[0], p[1]))); }
    function renderEditor() {
      const appleSet = cellsSet(editor.apples);
      const rockSet = cellsSet(editor.rocks);
      const snakeSet = cellsSet(editor.snake);
      grid.innerHTML = "";
      grid.style.gridTemplateColumns = `repeat(${editor.board}, minmax(13px, 1fr))`;
      for (let y = 0; y < editor.board; y++) {
        for (let x = 0; x < editor.board; x++) {
          const btn = document.createElement("button");
          btn.className = "cell";
          btn.classList.add(((x + y) % 2 === 0) ? "light" : "dark");
          btn.dataset.x = x;
          btn.dataset.y = y;
          if (rockSet.has(keyOf(x, y))) btn.classList.add("rock");
          if (snakeSet.has(keyOf(x, y))) btn.classList.add("snake");
          if (appleSet.has(keyOf(x, y))) btn.classList.add("apple");
          btn.addEventListener("click", () => toggleCell(x, y));
          grid.appendChild(btn);
        }
      }
      appleCount.textContent = `editing ${editor.mode}s | apples: ${editor.apples.length} | rocks: ${editor.rocks.length}`;
    }
    function toggleCell(x, y) {
      const snakeSet = cellsSet(editor.snake);
      if (snakeSet.has(keyOf(x, y))) return;
      if (editor.mode === "rock") {
        const idx = editor.rocks.findIndex(p => p[0] === x && p[1] === y);
        if (idx >= 0) editor.rocks.splice(idx, 1);
        else {
          editor.apples = editor.apples.filter(p => !(p[0] === x && p[1] === y));
          editor.rocks.push([x, y]);
        }
      } else {
        const rockSet = cellsSet(editor.rocks);
        if (rockSet.has(keyOf(x, y))) return;
        const idx = editor.apples.findIndex(p => p[0] === x && p[1] === y);
        if (idx >= 0) editor.apples.splice(idx, 1);
        else editor.apples.push([x, y]);
      }
      renderEditor();
    }
    async function loadConfig() {
      const res = await fetch("/api/config");
      editor = await res.json();
      editor.mode = "apple";
      editor.apples = editor.default_apples.map(p => [p[0], p[1]]);
      editor.rocks = editor.default_rocks.map(p => [p[0], p[1]]);
      renderEditor();
    }
    async function reset() {
      busy = true;
      try { draw(await post("/api/reset")); }
      finally { busy = false; }
    }
    async function customReset() {
      busy = true;
      try {
        draw(await post("/api/custom_reset", {apples: editor.apples, rocks: editor.rocks}));
        modal.classList.remove("open");
      } finally { busy = false; }
    }
    async function step(action) {
      if (busy) return;
      busy = true;
      try { draw(await post("/api/step", {action, allow_dead_step: allowDeadStep})); }
      finally { busy = false; }
    }
    document.querySelectorAll("[data-action]").forEach(btn => {
      btn.addEventListener("click", () => step(Number(btn.dataset.action)));
    });
    document.getElementById("reset").addEventListener("click", reset);
    document.getElementById("settings").addEventListener("click", () => modal.classList.add("open"));
    document.getElementById("closeSettings").addEventListener("click", () => modal.classList.remove("open"));
    modal.addEventListener("click", (e) => { if (e.target === modal) modal.classList.remove("open"); });
    document.getElementById("editApples").addEventListener("click", () => {
      editor.mode = "apple";
      renderEditor();
    });
    document.getElementById("editRocks").addEventListener("click", () => {
      editor.mode = "rock";
      renderEditor();
    });
    document.getElementById("defaultLayout").addEventListener("click", () => {
      editor.apples = editor.default_apples.map(p => [p[0], p[1]]);
      editor.rocks = editor.default_rocks.map(p => [p[0], p[1]]);
      renderEditor();
    });
    document.getElementById("clearMode").addEventListener("click", () => {
      if (editor.mode === "rock") editor.rocks = [];
      else editor.apples = [];
      renderEditor();
    });
    document.getElementById("randomApples").addEventListener("click", () => {
      const blocked = cellsSet(editor.rocks);
      for (const p of editor.snake) blocked.add(keyOf(p[0], p[1]));
      const cells = [];
      for (let y = 0; y < editor.board; y++) {
        for (let x = 0; x < editor.board; x++) {
          if (!blocked.has(keyOf(x, y))) cells.push([x, y]);
        }
      }
      for (let i = cells.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [cells[i], cells[j]] = [cells[j], cells[i]];
      }
      editor.apples = cells.slice(0, editor.default_apples.length);
      renderEditor();
    });
    document.getElementById("randomRocks").addEventListener("click", () => {
      const blocked = cellsSet(editor.snake);
      for (const p of editor.apples) blocked.add(keyOf(p[0], p[1]));
      const cells = [];
      for (let y = 0; y < editor.board; y++) {
        for (let x = 0; x < editor.board; x++) {
          if (!blocked.has(keyOf(x, y))) cells.push([x, y]);
        }
      }
      for (let i = cells.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [cells[i], cells[j]] = [cells[j], cells[i]];
      }
      editor.rocks = cells.slice(0, editor.default_rocks.length);
      renderEditor();
    });
    document.getElementById("startCustom").addEventListener("click", customReset);
    document.getElementById("continue").addEventListener("click", () => {
      allowDeadStep = !allowDeadStep;
      document.getElementById("continue").textContent = allowDeadStep ? "stop at death" : "continue after death";
    });
    window.addEventListener("keydown", (e) => {
      const map = {
        ArrowUp: 0, w: 0, W: 0,
        ArrowDown: 1, s: 1, S: 1,
        ArrowLeft: 2, a: 2, A: 2,
        ArrowRight: 3, d: 3, D: 3,
      };
      if (e.key === "r" || e.key === "R") return reset();
      if (map[e.key] !== undefined) {
        e.preventDefault();
        step(map[e.key]);
      }
    });
    loadConfig().then(reset).catch(err => {
      ids.status.textContent = err.message;
      ids.status.className = "status dead";
    });
  </script>
</body>
</html>
"""


def tensor_to_data_url(frame: torch.Tensor) -> str:
    arr = frame.detach().float().clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()
    arr = (arr * 255.0).round().astype(np.uint8)
    image = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def validate_cells(cells: object, name: str, blocked: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if not isinstance(cells, list):
        raise ValueError(f"{name} must be a list of [x, y] cells")
    seen: set[tuple[int, int]] = set()
    clean: list[tuple[int, int]] = []
    for item in cells:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"each {name} cell must be [x, y]")
        x, y = int(item[0]), int(item[1])
        cell = (x, y)
        if not (0 <= x < BOARD and 0 <= y < BOARD):
            raise ValueError(f"{name} outside board: {cell}")
        if cell in blocked:
            raise ValueError(f"{name} overlaps blocked cell: {cell}")
        if cell in seen:
            continue
        seen.add(cell)
        clean.append(cell)
    return clean


def validate_apples(apples: object, snake: list[tuple[int, int]], rocks: set[tuple[int, int]]) -> list[tuple[int, int]]:
    if not isinstance(apples, list):
        raise ValueError("apples must be a list of [x, y] cells")
    blocked = set(rocks) | set(snake)
    seen: set[tuple[int, int]] = set()
    clean: list[tuple[int, int]] = []
    for item in apples:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each apple must be [x, y]")
        x, y = int(item[0]), int(item[1])
        cell = (x, y)
        if not (0 <= x < BOARD and 0 <= y < BOARD):
            raise ValueError(f"apple outside board: {cell}")
        if cell in blocked:
            raise ValueError(f"apple overlaps snake or rock: {cell}")
        if cell in seen:
            continue
        seen.add(cell)
        clean.append(cell)
    return clean


def validate_rocks(rocks: object, snake: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return set(validate_cells(rocks, "rock", set(snake)))


class WorldModelSession:
    def __init__(self, checkpoint: Path, device: str):
        ckpt = torch.load(checkpoint, map_location="cpu")
        cfg = ckpt.get("model_config") or {}
        variant = cfg.get("variant", "wm_1m")
        self.device = torch.device(device)
        self.model = EventSnakeWorldModel(variant=variant).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.lock = threading.Lock()
        self.frame: torch.Tensor | None = None
        self.step_count = 0
        self.predicted_apples = 0
        self.done = False
        self.apple_prob = 0.0
        self.death_prob = 0.0
        self.last_action = RIGHT
        self.reset()

    def reset(self) -> dict:
        env = SnakeEnv(seed=123)
        result = env.reset()
        with self.lock:
            self.seed_frame_unlocked(result.frame)
            return self.state_unlocked()

    def custom_reset(self, apples: object, rocks: object | None = None) -> dict:
        env = SnakeEnv(seed=123)
        env.reset()
        custom_rocks = validate_rocks(rocks if rocks is not None else [list(p) for p in ROCKS_SET], env.snake)
        env._rock_set = custom_rocks
        env.apples = validate_apples(apples, env.snake, custom_rocks)
        with self.lock:
            self.seed_frame_unlocked(env.frame)
            return self.state_unlocked()

    def seed_frame_unlocked(self, frame_np: np.ndarray) -> None:
        frame = torch.from_numpy(frame_np.astype(np.float32) / 255.0).permute(2, 0, 1)[None]
        self.frame = frame.to(self.device)
        self.step_count = 0
        self.predicted_apples = 0
        self.done = False
        self.apple_prob = 0.0
        self.death_prob = 0.0
        self.last_action = RIGHT

    def step(self, action: int, allow_dead_step: bool = False) -> dict:
        if action not in (UP, DOWN, LEFT, RIGHT):
            raise ValueError(f"invalid action {action}; expected 0, 1, 2, or 3")
        with self.lock:
            if self.frame is None:
                return self.reset()
            if self.done and not allow_dead_step:
                return self.state_unlocked()
            action_t = torch.tensor([action], dtype=torch.long, device=self.device)
            with torch.inference_mode():
                out = self.model(self.frame, action_t)
                apple_probs = torch.softmax(out["apple_logits"], dim=-1)
                death_probs = torch.softmax(out["death_logits"], dim=-1)
                apple_class = int(apple_probs.argmax(dim=-1).item())
                death_class = int(death_probs.argmax(dim=-1).item())
                self.apple_prob = float(apple_probs[0, 1].item())
                self.death_prob = float(death_probs[0, 1].item())
                self.frame = out["frame"].detach().clamp(0, 1)
            self.step_count += 1
            self.predicted_apples += apple_class
            self.done = bool(death_class)
            self.last_action = action
            return self.state_unlocked()

    def state_unlocked(self) -> dict:
        assert self.frame is not None
        return {
            "image": tensor_to_data_url(self.frame),
            "step": self.step_count,
            "predicted_apples": self.predicted_apples,
            "done": self.done,
            "apple_prob": self.apple_prob,
            "death_prob": self.death_prob,
            "last_action": self.last_action,
        }


def make_handler(session: WorldModelSession):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def send_json(self, data: dict, status: int = 200) -> None:
            raw = json.dumps(data).encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path in ("/", "/index.html"):
                raw = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if path == "/api/state":
                with session.lock:
                    self.send_json(session.state_unlocked())
                return
            if path == "/api/config":
                env = SnakeEnv(seed=123)
                env.reset()
                self.send_json({
                    "board": BOARD,
                    "default_apples": [list(p) for p in INITIAL_APPLES],
                    "default_rocks": [list(p) for p in sorted(ROCKS_SET)],
                    "rocks": [list(p) for p in sorted(ROCKS_SET)],
                    "snake": [list(p) for p in env.snake],
                })
                return
            self.send_error(404)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            size = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(size) if size else b"{}"
            try:
                data = json.loads(body.decode("utf-8") or "{}")
                if path == "/api/reset":
                    self.send_json(session.reset())
                    return
                if path == "/api/custom_reset":
                    self.send_json(session.custom_reset(data.get("apples", []), data.get("rocks")))
                    return
                if path == "/api/step":
                    self.send_json(session.step(int(data.get("action")), bool(data.get("allow_dead_step", False))))
                    return
                self.send_error(404)
            except Exception as exc:
                self.send_json({"error": str(exc)}, status=400)

    return Handler


def parse_location(location: str | None, host: str, port: int) -> tuple[str, int]:
    if not location:
        return host, port
    raw = location.removeprefix("http://").removeprefix("https://").rstrip("/")
    if ":" not in raw:
        return raw, port
    loc_host, loc_port = raw.rsplit(":", 1)
    return loc_host or host, int(loc_port)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/local_event_random_eval_small_only_20260605_220114/hard/world_models/wm_1m_ctx1_events/latest.pt"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8055)
    parser.add_argument("--location", default=None, help="host:port shorthand, for example localhost:2453")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    host, port = parse_location(args.location, args.host, args.port)

    session = WorldModelSession(args.checkpoint, args.device)
    server = ThreadingHTTPServer((host, port), make_handler(session))
    print(f"serving Snake WM simulator at http://{host}:{port}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {args.device}")
    server.serve_forever()


if __name__ == "__main__":
    main()
