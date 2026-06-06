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


GAMES_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Low-res world model game candidates</title>
  <style>
    :root {
      --ink: #f2ead2;
      --muted: #a9b494;
      --line: #344329;
      --green: #9bd85c;
      --blue: #72a7ff;
      --red: #ff725f;
      --gold: #f0c95a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      background: #000;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
    }
    main {
      width: min(1120px, calc(100vw - 32px));
      margin: 34px auto 60px;
    }
    .top {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 18px;
      align-items: end;
      margin-bottom: 26px;
    }
    h1 {
      font-size: clamp(38px, 7vw, 82px);
      line-height: .88;
      letter-spacing: -0.06em;
      margin: 0 0 14px;
    }
    p {
      color: var(--muted);
      font-size: 18px;
      line-height: 1.45;
      max-width: 760px;
      margin: 0;
    }
    a {
      color: inherit;
      text-decoration: none;
    }
    .back {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 12px 16px;
      color: var(--green);
      background: rgba(155, 216, 92, .08);
      font: 700 14px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: nowrap;
    }
    .games {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
      gap: 18px;
    }
    .game {
      min-height: 310px;
      border: 1px solid var(--line);
      border-radius: 24px;
      padding: 18px;
      background: #070a06;
      box-shadow: 0 22px 70px rgba(0, 0, 0, .45), inset 0 0 0 1px rgba(255,255,255,.025);
      display: grid;
      grid-template-rows: auto 1fr auto;
      gap: 15px;
      transition: transform .14s ease, border-color .14s ease;
    }
    .game:hover {
      transform: translateY(-4px);
      border-color: var(--green);
    }
    .thumb {
      aspect-ratio: 1;
      border-radius: 16px;
      overflow: hidden;
      border: 1px solid #25351d;
      background: #111;
      image-rendering: pixelated;
      display: grid;
      grid-template-columns: repeat(16, 1fr);
      grid-template-rows: repeat(16, 1fr);
    }
    .cell:nth-child(odd) { background: #1b2616; }
    .cell:nth-child(even) { background: #22301b; }
    .wall { background: #344329 !important; }
    .coin { background: var(--gold) !important; box-shadow: inset 0 0 0 2px #9e7a1e; }
    .enemy { background: var(--red) !important; }
    .player { background: var(--blue) !important; }
    .pellet { background: #f2ead2 !important; transform: scale(.32); border-radius: 999px; }
    .water { background: #1f5f8f !important; }
    .lava { background: #b93b2d !important; }
    .key { background: #c48cff !important; }
    .goal { background: var(--green) !important; }
    h2 {
      margin: 0 0 6px;
      font-size: 28px;
      letter-spacing: -.035em;
    }
    .meta {
      color: var(--muted);
      font: 700 12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      text-transform: uppercase;
      letter-spacing: .08em;
    }
    .why {
      color: var(--muted);
      font-size: 16px;
      line-height: 1.35;
      margin-top: 9px;
    }
    .tag {
      display: inline-flex;
      width: fit-content;
      border: 1px solid #405832;
      border-radius: 999px;
      padding: 8px 10px;
      color: var(--green);
      background: rgba(155, 216, 92, .08);
      font: 700 12px/1 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    @media (max-width: 760px) {
      .top { grid-template-columns: 1fr; align-items: start; }
    }
  </style>
</head>
<body>
  <main>
    <section class="top">
      <div>
        <h1>Next games after Snake.</h1>
        <p>Same visual constraint: tiny RGB frames, crisp pixel art, action-conditioned next-frame prediction, plus discrete event heads for reward and death. Pick a game where the reward interface is still simple enough to test cleanly.</p>
      </div>
      <a class="back" href="/">back to Snake WM</a>
    </section>
    <section class="games">
      <a class="game" href="/games/pacman">
        <div class="thumb" data-scene="pacman"></div>
        <div>
          <h2>Mini Pac-Man</h2>
          <div class="meta">collect pellets | avoid ghosts | medium risk</div>
          <div class="why">Best next step if we want a recognizable game with discrete rewards and adversarial motion.</div>
        </div>
        <span class="tag">recommended after Snake++</span>
      </a>
      <a class="game" href="/games/maze-key">
        <div class="thumb" data-scene="maze"></div>
        <div>
          <h2>Maze Key Door</h2>
          <div class="meta">key event | door event | low risk</div>
          <div class="why">Clean test of partial objectives: learn navigation, collect a key, then reach a goal.</div>
        </div>
        <span class="tag">clean paper extension</span>
      </a>
      <a class="game" href="/games/collect-coins">
        <div class="thumb" data-scene="coins"></div>
        <div>
          <h2>Coin Collector</h2>
          <div class="meta">multi-collect | no body dynamics | low risk</div>
          <div class="why">Simplest control after Snake. Useful for checking whether event rewards solve reward exploitation generally.</div>
        </div>
        <span class="tag">fastest baseline</span>
      </a>
      <a class="game" href="/games/sokoban">
        <div class="thumb" data-scene="sokoban"></div>
        <div>
          <h2>Mini Sokoban</h2>
          <div class="meta">push blocks | sparse goal | high risk</div>
          <div class="why">More interesting dynamics, but harder credit assignment and more ways for a world model to drift.</div>
        </div>
        <span class="tag">hard dynamics</span>
      </a>
      <a class="game" href="/games/frogger">
        <div class="thumb" data-scene="frogger"></div>
        <div>
          <h2>Mini Frogger</h2>
          <div class="meta">moving hazards | crossing reward | medium risk</div>
          <div class="why">Good for testing moving objects and death prediction without making rewards ambiguous.</div>
        </div>
        <span class="tag">moving hazards</span>
      </a>
      <a class="game" href="/games/snake-plus">
        <div class="thumb" data-scene="snake"></div>
        <div>
          <h2>Snake++</h2>
          <div class="meta">random rocks | random apples | lowest risk</div>
          <div class="why">Most controlled extension: same game, broader layouts, better evidence before changing domains.</div>
        </div>
        <span class="tag">do this first</span>
      </a>
    </section>
  </main>
  <script>
    const scenes = {
      pacman: {wall: [0,1,2,3,4,5,6,7,8,15,16,31,32,47,48,63,64,79,80,95,96,111,112,127,128,143,144,159,160,175,176,191,192,207,208,223,224,239,240,241,242,243,244,245,246,247,248,255], pellet: [18,21,24,27,35,38,41,44,67,70,73,76,99,102,105,108,131,134,137,140,163,166,169,172,195,198,201,204], player: [182], enemy: [85,90,170]},
      maze: {wall: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,32,34,35,36,38,39,41,42,44,47,48,52,55,58,63,64,65,66,68,70,72,74,76,78,79,80,84,86,90,94,95,96,98,100,102,104,106,108,111,112,116,120,124,127,128,130,131,132,134,136,138,140,143,144,159,160,175,176,191,192,207,208,223,224,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255], player: [17], key: [109], goal: [238]},
      coins: {coin: [35,39,44,72,76,105,119,134,151,186,202,217], player: [204], enemy: [85]},
      sokoban: {wall: [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,31,32,47,48,63,64,79,80,95,96,111,112,127,128,143,144,159,160,175,176,191,192,207,208,223,224,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,84,85,86,116,148], player: [197], coin: [101,118], goal: [90,154]},
      frogger: {water: [32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79], lava: [112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159], player: [232], goal: [7], enemy: [116,123,146,153]},
      snake: {wall: [88,89,104,105,196,197,139,140], coin: [37,42,82,110,154,201], player: [147,146,145]}
    };
    document.querySelectorAll('.thumb').forEach((el) => {
      const scene = scenes[el.dataset.scene] || {};
      for (let i = 0; i < 256; i++) {
        const c = document.createElement('div');
        c.className = 'cell';
        for (const [name, ids] of Object.entries(scene)) {
          if (ids.includes(i)) c.classList.add(name);
        }
        el.appendChild(c);
      }
    });
  </script>
</body>
</html>
"""


GAME_CONFIGS = {
    "pacman": {"title": "Mini Pac-Man", "mode": "pacman", "hint": "Eat pellets, avoid ghosts. Arrow keys/WASD move one tile."},
    "maze-key": {"title": "Maze Key Door", "mode": "maze", "hint": "Pick up the key, then reach the green door."},
    "collect-coins": {"title": "Coin Collector", "mode": "coins", "hint": "Collect every coin while avoiding the red chaser."},
    "sokoban": {"title": "Mini Sokoban", "mode": "sokoban", "hint": "Push both crates onto the green goal tiles."},
    "frogger": {"title": "Mini Frogger", "mode": "frogger", "hint": "Cross moving hazard lanes and reach the top goal."},
    "snake-plus": {"title": "Snake++", "mode": "snake", "hint": "Classic Snake with rocks. Eat apples, avoid walls, rocks, and yourself."},
}


GAME_PLAY_HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>__TITLE__</title>
<style>
:root{--ink:#f2ead2;--muted:#a9b494;--line:#344329;--green:#9bd85c;--blue:#72a7ff;--red:#ff725f;--gold:#f0c95a}*{box-sizing:border-box}body{margin:0;min-height:100vh;color:var(--ink);background:#000;font-family:Georgia,"Iowan Old Style","Palatino Linotype",serif}main{width:min(980px,calc(100vw - 32px));margin:32px auto;display:grid;grid-template-columns:minmax(280px,560px) 1fr;gap:26px;align-items:start}h1{margin:0 0 12px;font-size:clamp(38px,6vw,78px);line-height:.88;letter-spacing:-.06em}p{color:var(--muted);font-size:18px;line-height:1.45;margin:0 0 18px}a{color:var(--green);text-decoration:none}.screen,.card{border:1px solid var(--line);border-radius:24px;background:#070a06;box-shadow:0 24px 80px rgba(0,0,0,.48),inset 0 0 0 1px rgba(255,255,255,.025)}.screen{padding:18px}canvas{width:100%;aspect-ratio:1;display:block;image-rendering:pixelated;border-radius:14px;background:#172512}.card{padding:24px}.stats{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:18px 0}.stat{border:1px solid var(--line);border-radius:14px;padding:12px;background:rgba(255,255,255,.04)}.label{color:var(--muted);font:700 12px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;text-transform:uppercase;letter-spacing:.08em;margin-bottom:6px}.value{font:700 26px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}.controls{display:grid;grid-template-columns:repeat(3,74px);gap:10px;justify-content:center;margin-top:18px}button{border:1px solid #2a3b25;background:#24351e;color:#f2ead2;border-radius:16px;font:700 18px/1 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;padding:16px 14px;cursor:pointer;box-shadow:0 8px 0 #070b06}button:active{transform:translateY(5px);box-shadow:0 3px 0 #070b06}.wide{width:100%;margin-top:12px;background:var(--green);color:#10200b}.status{margin-top:16px;padding:12px 14px;border-radius:14px;background:rgba(155,216,92,.1);color:var(--green);font-weight:700}.dead{color:var(--red);background:rgba(255,114,95,.12)}@media(max-width:820px){main{grid-template-columns:1fr;margin:18px auto}}
</style></head><body><main><section class="screen"><canvas id="c" width="128" height="128"></canvas></section><section class="card"><a href="/games">all games</a><h1>__TITLE__</h1><p>__HINT__ Same low-resolution visual style as Snake: 16x16 grid rendered into 128x128 RGB pixels.</p><div class="stats"><div class="stat"><div class="label">score</div><div class="value" id="score">0</div></div><div class="stat"><div class="label">steps</div><div class="value" id="steps">0</div></div><div class="stat"><div class="label">mode</div><div class="value" id="mode">__MODE__</div></div><div class="stat"><div class="label">state</div><div class="value" id="state">run</div></div></div><div class="controls"><div></div><button data-a="0">↑</button><div></div><button data-a="2">←</button><button id="reset">reset</button><button data-a="3">→</button><div></div><button data-a="1">↓</button><div></div></div><button class="wide" id="tick">wait / tick hazards</button><div class="status" id="msg">ready</div></section></main>
<script>
const MODE="__MODE__",N=16,C=8,cv=document.getElementById('c'),x=cv.getContext('2d'),$=id=>document.getElementById(id),D=[[0,-1],[0,1],[-1,0],[1,0]];let S;function pt(x,y){return{x,y}}function same(a,b){return a.x===b.x&&a.y===b.y}function has(a,p){return a.some(q=>same(q,p))}function rm(a,p){let i=a.findIndex(q=>same(q,p));if(i>=0)a.splice(i,1);return i>=0}function inside(p){return p.x>=0&&p.x<N&&p.y>=0&&p.y<N}function add(p,d){return pt(p.x+D[d][0],p.y+D[d][1])}function border(){let w=[];for(let i=0;i<N;i++)w.push(pt(i,0),pt(i,N-1),pt(0,i),pt(N-1,i));return w}
function reset(){S={score:0,steps:0,dead:false,win:false,msg:'ready',walls:[],coins:[],pellets:[],goals:[],boxes:[],rocks:[],hazards:[],ghosts:[],key:false,dir:3,snake:[]};if(MODE==='pacman'){S.player=pt(8,12);S.walls=border().concat([pt(4,4),pt(5,4),pt(10,4),pt(11,4),pt(4,8),pt(5,8),pt(10,8),pt(11,8),pt(7,6),pt(8,6)]);for(let y=2;y<14;y+=2)for(let q=2;q<14;q+=2)S.pellets.push(pt(q,y));S.ghosts=[pt(7,7),pt(11,10)]}if(MODE==='maze'){S.player=pt(1,1);S.keyPos=pt(11,6);S.door=pt(14,14);S.walls=border().concat([pt(3,1),pt(3,2),pt(3,3),pt(3,4),pt(5,4),pt(6,4),pt(7,4),pt(8,4),pt(10,2),pt(10,3),pt(10,4),pt(10,5),pt(10,6),pt(2,8),pt(3,8),pt(4,8),pt(5,8),pt(7,8),pt(8,8),pt(9,8),pt(11,10),pt(12,10),pt(13,10),pt(6,11),pt(6,12),pt(6,13)])}if(MODE==='coins'){S.player=pt(2,13);S.walls=border().concat([pt(5,5),pt(5,6),pt(5,7),pt(10,8),pt(11,8),pt(12,8)]);S.coins=[pt(3,3),pt(8,2),pt(13,4),pt(4,10),pt(9,12),pt(13,13)];S.ghosts=[pt(12,2)]}if(MODE==='sokoban'){S.player=pt(3,12);S.walls=border().concat([pt(3,3),pt(4,3),pt(5,3),pt(10,3),pt(10,4),pt(10,5),pt(6,10),pt(7,10),pt(8,10)]);S.boxes=[pt(6,7),pt(9,7)];S.goals=[pt(6,4),pt(11,11)]}if(MODE==='frogger'){S.player=pt(8,15);S.goals=[pt(8,0)];for(let q=1;q<15;q+=4)S.hazards.push({x:q,y:5,d:1});for(let q=2;q<15;q+=5)S.hazards.push({x:q,y:9,d:-1});for(let q=0;q<15;q+=5)S.hazards.push({x:q,y:12,d:1})}if(MODE==='snake'){S.snake=[pt(5,9),pt(4,9),pt(3,9)];S.player=S.snake[0];S.dir=3;S.rocks=[pt(8,5),pt(8,6),pt(9,5),pt(9,6),pt(4,12),pt(5,12),pt(11,8),pt(12,8)];S.coins=[pt(12,3)]}draw()}
function blocked(p){return!inside(p)||has(S.walls,p)||has(S.rocks,p)}function moveGhosts(){for(const g of S.ghosts){let o=[0,1,2,3].map(d=>add(g,d)).filter(p=>!blocked(p));o.sort((a,b)=>Math.abs(a.x-S.player.x)+Math.abs(a.y-S.player.y)-Math.abs(b.x-S.player.x)-Math.abs(b.y-S.player.y));if(o[0]){g.x=o[0].x;g.y=o[0].y}}}function tickHazards(){for(const h of S.hazards){h.x+=h.d;if(h.x<=0||h.x>=15){h.d*=-1;h.x+=h.d*2}}}
function step(d){if(S.dead||S.win)return;S.steps++;if(MODE==='snake'){if((d===0&&S.dir!==1)||(d===1&&S.dir!==0)||(d===2&&S.dir!==3)||(d===3&&S.dir!==2))S.dir=d;let h=add(S.snake[0],S.dir);if(blocked(h)||has(S.snake,h)){S.dead=true;S.msg='death';return draw()}S.snake.unshift(h);S.player=h;if(rm(S.coins,h)){S.score++;placeApple()}else S.snake.pop();return draw()}let np=add(S.player,d);if(MODE==='sokoban'&&has(S.boxes,np)){let b=S.boxes.find(q=>same(q,np)),bp=add(b,d);if(blocked(bp)||has(S.boxes,bp))np=S.player;else{b.x=bp.x;b.y=bp.y}}if(!blocked(np)&&!(MODE==='sokoban'&&has(S.boxes,np)))S.player=np;if(MODE==='pacman'){if(rm(S.pellets,S.player))S.score++;moveGhosts();if(has(S.ghosts,S.player)){S.dead=true;S.msg='ghost caught you'}if(S.pellets.length===0){S.win=true;S.msg='cleared pellets'}}if(MODE==='maze'){if(!S.key&&same(S.player,S.keyPos)){S.key=true;S.score=1}if(same(S.player,S.door)){if(S.key){S.win=true;S.msg='door reached'}else S.msg='need key first'}}if(MODE==='coins'){if(rm(S.coins,S.player))S.score++;moveGhosts();if(has(S.ghosts,S.player)){S.dead=true;S.msg='caught'}if(S.coins.length===0){S.win=true;S.msg='all coins'}}if(MODE==='sokoban'){S.score=S.boxes.filter(b=>has(S.goals,b)).length;if(S.score===S.goals.length){S.win=true;S.msg='all crates placed'}}if(MODE==='frogger'){tickHazards();if(S.hazards.some(h=>same(h,S.player))){S.dead=true;S.msg='hit hazard'}if(has(S.goals,S.player)){S.score=1;S.win=true;S.msg='crossed'}}draw()}
function waitTick(){if(S.dead||S.win)return;S.steps++;if(MODE==='frogger')tickHazards();if(MODE==='pacman'||MODE==='coins')moveGhosts();if(S.hazards.some(h=>same(h,S.player))||has(S.ghosts,S.player)){S.dead=true;S.msg='caught'}draw()}function placeApple(){for(let y=1;y<15;y++)for(let q=1;q<15;q++){let p=pt(q,y);if(!has(S.snake,p)&&!has(S.rocks,p)){S.coins=[p];return}}}function cell(p,c){x.fillStyle=c;x.fillRect(p.x*C,p.y*C,C,C)}
function draw(){x.imageSmoothingEnabled=false;for(let y=0;y<N;y++)for(let q=0;q<N;q++){x.fillStyle=((q+y)%2)?'#86c94d':'#93d957';x.fillRect(q*C,y*C,C,C)}[...S.walls,...S.rocks].forEach(p=>cell(p,'#405832'));S.goals.forEach(p=>cell(p,'#36c36b'));S.coins.forEach(p=>cell(p,'#e7473c'));S.pellets.forEach(p=>{x.fillStyle='#f2ead2';x.fillRect(p.x*C+3,p.y*C+3,2,2)});if(S.keyPos&&!S.key)cell(S.keyPos,'#c48cff');if(S.door)cell(S.door,S.key?'#36c36b':'#6a4831');S.boxes.forEach(p=>cell(p,'#b7793e'));S.hazards.forEach(p=>cell(p,'#ff725f'));S.ghosts.forEach(p=>cell(p,'#ff725f'));if(S.snake.length)S.snake.forEach((p,i)=>cell(p,i?'#2f80ed':'#72a7ff'));else cell(S.player,'#2f80ed');$('score').textContent=S.score;$('steps').textContent=S.steps;$('state').textContent=S.win?'win':(S.dead?'dead':'run');$('msg').textContent=S.msg||'running';$('msg').className='status'+(S.dead?' dead':'')}
window.addEventListener('keydown',e=>{let m={ArrowUp:0,w:0,W:0,ArrowDown:1,s:1,S:1,ArrowLeft:2,a:2,A:2,ArrowRight:3,d:3,D:3};if(e.key==='r'||e.key==='R')return reset();if(m[e.key]!==undefined){e.preventDefault();step(m[e.key])}});document.querySelectorAll('[data-a]').forEach(b=>b.onclick=()=>step(Number(b.dataset.a)));$('reset').onclick=reset;$('tick').onclick=waitTick;reset();
</script></body></html>"""


def game_page_html(slug: str) -> str:
    cfg = GAME_CONFIGS[slug]
    return (GAME_PLAY_HTML
        .replace("__TITLE__", cfg["title"])
        .replace("__MODE__", cfg["mode"])
        .replace("__HINT__", cfg["hint"]))


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
            if path == "/games":
                raw = GAMES_HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("content-type", "text/html; charset=utf-8")
                self.send_header("content-length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if path.startswith("/games/"):
                slug = path.removeprefix("/games/").strip("/")
                if slug in GAME_CONFIGS:
                    raw = game_page_html(slug).encode("utf-8")
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
