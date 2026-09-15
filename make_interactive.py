#!/usr/bin/env python
"""Turn a phase-grid CSV into a self-contained interactive HTML viewer.

The viewer is plain HTML + CSS + canvas JavaScript with **no external
dependencies** (no plotly, no CDN), so it works offline by double-clicking it.

What it provides:

* hover anywhere and read the exact N, P, loss and standard deviation;
* click to pin a cell, so the readout stays put while you look at it;
* zoom with the mouse wheel (anchored on the cursor) and pan by dragging;
* switch the colour scale between linear / log / asinh to see what a log colour
  scale actually does;
* switch between the four recorded quantities;
* an optional grid of numbers drawn on top of the cells.

Usage
-----
    python make_interactive.py
    python make_interactive.py --csv phase/<name>/phase_grid.csv --open
"""

from __future__ import annotations

import argparse
import csv
import json
import webbrowser
from pathlib import Path

# (csv key, label, unit-ish description) -- first entry is the default
QUANTITIES = [
    ("final_test_loss", "final test loss（最后一个 epoch）"),
    ("best_test_loss", "best test loss（所有测试点的最小值）"),
    ("final_test_analytic", "解析 test loss  |w-w̄|² + σ²"),
    ("final_train_loss", "final train loss（P<N 时降到机器精度）"),
]

PALETTE = [
    # (position 0..1, rgb) -- a viridis-like ramp, perceptually ordered
    (0.00, (68, 1, 84)),
    (0.15, (72, 40, 120)),
    (0.30, (62, 74, 137)),
    (0.45, (49, 104, 142)),
    (0.60, (38, 130, 142)),
    (0.75, (31, 158, 137)),
    (0.88, (74, 193, 109)),
    (1.00, (253, 231, 37)),
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build the interactive N-P viewer.")
    p.add_argument("--csv", type=str, default=None,
                   help="phase grid CSV; default: the newest one under phase/")
    p.add_argument("--out", type=str, default=None,
                   help="output HTML; default: beside the CSV as interactive.html")
    p.add_argument("--open", action="store_true", help="open it in the browser")
    return p.parse_args(argv)


def newest_grid() -> Path:
    candidates = sorted(Path("phase").glob("*/phase_grid.csv"),
                        key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit("no phase/*/phase_grid.csv found -- run phase_diagram.py first")
    return candidates[-1]


def load_grid(csv_path: Path) -> dict:
    with csv_path.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    n_values = sorted({int(r["N"]) for r in rows})
    p_values = sorted({int(r["P"]) for r in rows})
    idx_n = {n: i for i, n in enumerate(n_values)}
    idx_p = {p: i for i, p in enumerate(p_values)}

    data: dict[str, dict] = {}
    for key, label in QUANTITIES:
        values = [[None] * len(n_values) for _ in p_values]
        stds = [[None] * len(n_values) for _ in p_values]
        for r in rows:
            i, j = idx_p[int(r["P"])], idx_n[int(r["N"])]
            values[i][j] = float(r[key])
            stds[i][j] = float(r.get(f"{key}_std") or "nan")
        data[key] = {"label": label, "values": values, "stds": stds}

    lr_used = [[None] * len(n_values) for _ in p_values]
    for r in rows:
        lr_used[idx_p[int(r["P"])]][idx_n[int(r["N"])]] = float(r["lr_used"])

    meta = {}
    if rows:
        first = rows[0]
        meta = {
            "seeds": first.get("seeds", ""),
            "lr_requested": first.get("lr_requested", ""),
            "source": csv_path.name,
        }
    return {"n_values": n_values, "p_values": p_values, "quantities": data,
            "lr_used": lr_used, "meta": meta}


HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>N-P 相图交互查看器</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 "Microsoft YaHei", "Noto Sans CJK SC", system-ui, sans-serif;
         background: #fafafa; color: #1a1a1a; }
  header { padding: 14px 20px 6px; }
  h1 { font-size: 17px; margin: 0 0 4px; }
  .hint { color: #666; font-size: 12.5px; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 18px; align-items: center;
             padding: 10px 20px; background: #fff; border-bottom: 1px solid #e3e3e3;
             position: sticky; top: 0; z-index: 5; }
  .toolbar label { font-size: 13px; color: #444; }
  select, button { font: inherit; padding: 3px 8px; border: 1px solid #bbb;
                   border-radius: 5px; background: #fff; }
  button { cursor: pointer; }
  button:hover { background: #f0f0f0; }
  #readout { padding: 10px 20px; font-family: ui-monospace, Consolas, monospace;
             font-size: 13px; background: #fff; border-bottom: 1px solid #e3e3e3;
             min-height: 62px; }
  #readout b { color: #0b5; }
  .pinned { color: #b00; font-weight: 600; }
  .wrap { display: flex; justify-content: center; padding: 14px 20px 30px; }
  canvas { touch-action: none; cursor: crosshair; border-radius: 6px; }
  footer { padding: 8px 20px 30px; color: #666; font-size: 12.5px; max-width: 900px; }
  code { background: #eee; padding: 1px 5px; border-radius: 3px; }
</style>
</head>
<body>
<header>
  <h1>N-P 相图交互查看器</h1>
  <div class="hint">
    滚轮缩放（以光标为中心）· 拖动平移 · 点击格子固定读数 · 悬停即读数值
  </div>
</header>

<div class="toolbar">
  <label>显示的量
    <select id="quantity"></select>
  </label>
  <label>颜色刻度
    <select id="scale">
      <option value="linear">线性 linear</option>
      <option value="log">对数 log</option>
      <option value="asinh">asinh（折中）</option>
    </select>
  </label>
  <label><input type="checkbox" id="showNumbers"> 在格子上显示数值</label>
  <button id="reset">重置视图</button>
</div>

<div id="readout">把鼠标移到图上任意位置。</div>

<div class="wrap"><canvas id="cv" width="880" height="640"></canvas></div>

<footer>
  <p><b>关于对数坐标：</b>坐标轴的对数刻度与颜色的对数刻度是两回事。这两相图的
  <b>坐标轴（N 和 P）永远是线性的</b>；上面的「颜色刻度」只决定<em>数值如何映射到颜色</em>。
  切到 <code>log</code> 会放大低 loss 区域的差异（否则它们都挤成一个颜色），
  切到 <code>linear</code> 则是原始比例。</p>
  <p id="protect"></p>
</footer>

<script id="grid-data" type="application/json">__DATA__</script>
<script>
const GRID = JSON.parse(document.getElementById('grid-data').textContent);
const PALETTE = __PALETTE__;

const cv = document.getElementById('cv');
const ctx = cv.getContext('2d');
const readout = document.getElementById('readout');
const quantitySel = document.getElementById('quantity');
const scaleSel = document.getElementById('scale');
const numbersChk = document.getElementById('showNumbers');
const resetBtn = document.getElementById('reset');

// ---- plot geometry -----------------------------------------------------
const M = { left: 62, right: 96, top: 26, bottom: 52 };
const W = () => cv.width - M.left - M.right;
const H = () => cv.height - M.top - M.bottom;
const nCols = GRID.n_values.length, nRows = GRID.p_values.length;

let view = { x0: 0, y0: 0, x1: nCols, y1: nRows };  // in pixel-cell units
let pinned = null;

function resetView() { view = { x0: 0, y0: 0, x1: nCols, y1: nRows }; pinned = null; draw(); }

// ---- colour ------------------------------------------------------------
function ramp(t) {
  t = Math.min(1, Math.max(0, t));
  for (let i = 1; i < PALETTE.length; i++) {
    if (t <= PALETTE[i][0]) {
      const [p0, c0] = PALETTE[i - 1], [p1, c1] = PALETTE[i];
      const f = (t - p0) / (p1 - p0 || 1);
      return [0, 1, 2].map(k => Math.round(c0[k] + f * (c1[k] - c0[k])));
    }
  }
  return PALETTE[PALETTE.length - 1][1];
}

function makeNorm(mode, lo, hi) {
  if (mode === 'log') {
    const a = Math.log10(Math.max(lo, hi * 1e-6)), b = Math.log10(hi);
    return v => (Math.log10(Math.max(v, hi * 1e-6)) - a) / (b - a || 1);
  }
  if (mode === 'asinh') {
    const w = Math.max(hi * 0.05, 1e-12);
    const f = v => Math.asinh(v / w);
    const a = f(lo), b = f(hi);
    return v => (f(v) - a) / (b - a || 1);
  }
  return v => (v - lo) / (hi - lo || 1);
}

function flat(key) {
  const out = [];
  const vals = GRID.quantities[key].values;
  for (let i = 0; i < nRows; i++) for (let j = 0; j < nCols; j++) {
    const v = vals[i][j];
    if (v !== null && isFinite(v)) out.push(v);
  }
  return out;
}

// ---- drawing -----------------------------------------------------------
function draw() {
  const key = quantitySel.value;
  const q = GRID.quantities[key];
  const vals = flat(key);
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const norm = makeNorm(scaleSel.value, lo, hi);

  const w = W(), h = H();
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, cv.width, cv.height);

  const cw = w / (view.x1 - view.x0), ch = h / (view.y1 - view.y0);
  // only draw visible cells
  for (let i = Math.floor(view.y0); i < Math.ceil(view.y1); i++) {
    if (i < 0 || i >= nRows) continue;
    for (let j = Math.floor(view.x0); j < Math.ceil(view.x1); j++) {
      if (j < 0 || j >= nCols) continue;
      const v = q.values[i][j];
      const px = M.left + (j - view.x0) * cw, py = M.top + (i - view.y0) * ch;
      if (v === null || !isFinite(v)) { ctx.fillStyle = '#ddd'; }
      else { const c = ramp(norm(v)); ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`; }
      ctx.fillRect(px, py, Math.ceil(cw) + 1, Math.ceil(ch) + 1);
      if (numbersChk.checked && cw > 34 && ch > 16) {
        ctx.fillStyle = norm(v) > 0.55 ? '#111' : '#fff';
        ctx.font = '10px ui-monospace, monospace';
        ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText(fmtShort(v), px + cw / 2, py + ch / 2);
      }
    }
  }

  // P = N diagonal, drawn in cell coordinates
  ctx.save();
  ctx.beginPath();
  ctx.rect(M.left, M.top, w, h);
  ctx.clip();
  ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.setLineDash([6, 5]);
  ctx.beginPath();
  const diag = (n, p) => [M.left + (n - view.x0) * cw, M.top + (p - view.y0) * ch];
  const nMin = GRID.n_values[0], nMax = GRID.n_values[nCols - 1];
  const [ax, ay] = diag(nMin * 0.8, GRID.p_values[0] * 0.8);
  const [bx, by] = diag(nMax * 1.2, nMax * 1.2);
  ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
  ctx.restore();

  // pinned cell marker
  if (pinned) {
    const [i, j] = pinned;
    ctx.strokeStyle = '#e00'; ctx.lineWidth = 2.5; ctx.setLineDash([]);
    ctx.strokeRect(M.left + (j - view.x0) * cw, M.top + (i - view.y0) * ch,
                   cw, ch);
  }

  // axes
  ctx.strokeStyle = '#888'; ctx.lineWidth = 1; ctx.setLineDash([]);
  ctx.strokeRect(M.left, M.top, w, h);
  ctx.fillStyle = '#333'; ctx.font = '12px ui-monospace, monospace';
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (let j = 0; j < nCols; j++) {
    const px = M.left + (j + 0.5 - view.x0) * cw;
    if (px < M.left - 1 || px > M.left + w + 1) continue;
    ctx.fillText(GRID.n_values[j], px, M.top + h + 7);
    ctx.strokeStyle = '#ccc'; ctx.beginPath();
    ctx.moveTo(px, M.top + h); ctx.lineTo(px, M.top + h + 4); ctx.stroke();
  }
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (let i = 0; i < nRows; i++) {
    const py = M.top + (i + 0.5 - view.y0) * ch;
    if (py < M.top - 1 || py > M.top + h + 1) continue;
    ctx.fillText(GRID.p_values[i], M.left - 8, py);
    ctx.strokeStyle = '#ccc'; ctx.beginPath();
    ctx.moveTo(M.left - 4, py); ctx.lineTo(M.left, py); ctx.stroke();
  }
  ctx.fillStyle = '#000'; ctx.font = '13px "Microsoft YaHei", sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  ctx.fillText('N  (输入维度 / student 参数量)', M.left + w / 2, cv.height - 12);
  ctx.save();
  ctx.translate(16, M.top + h / 2); ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'top';
  ctx.fillText('P  (训练样本数)', 0, 0);
  ctx.restore();

  // colourbar
  const bx0 = M.left + w + 34, bw = 16, by0 = M.top, bh = h;
  for (let k = 0; k < bh; k++) {
    const t = 1 - k / bh;
    const c = ramp(t);
    ctx.fillStyle = `rgb(${c[0]},${c[1]},${c[2]})`;
    ctx.fillRect(bx0, by0 + k, bw, 1);
  }
  ctx.strokeStyle = '#888'; ctx.strokeRect(bx0, by0, bw, bh);
  ctx.fillStyle = '#333'; ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
  const mode = scaleSel.value;
  for (let k = 0; k <= 4; k++) {
    const t = k / 4;
    ctx.fillText(fmtShort(invNorm(norm, t, lo, hi, mode)), bx0 + bw + 6,
                 by0 + bh * (1 - t));
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  ctx.fillText(mode === 'linear' ? '线性刻度' : mode + ' 颜色刻度',
               bx0 + bw / 2 - 14, by0 + bh + 10);
}

function invNorm(norm, t, lo, hi, mode) {
  // invert the (monotone) norm so the colourbar can be labelled
  let a = lo, b = hi;
  for (let k = 0; k < 40; k++) {
    const m = (a + b) / 2;
    if (norm(m) < t) a = m; else b = m;
  }
  return (a + b) / 2;
}

function fmtShort(v) {
  if (v === null || !isFinite(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 1e-3 || a >= 1e4) return v.toExponential(1);
  return v.toFixed(a < 0.1 ? 4 : 3);
}

// ---- interaction -------------------------------------------------------
function cellFromEvent(ev) {
  const r = cv.getBoundingClientRect();
  const x = (ev.clientX - r.left) * (cv.width / r.width);
  const y = (ev.clientY - r.top) * (cv.height / r.height);
  const cw = W() / (view.x1 - view.x0), ch = H() / (view.y1 - view.y0);
  const j = Math.floor(view.x0 + (x - M.left) / cw);
  const i = Math.floor(view.y0 + (y - M.top) / ch);
  if (i < 0 || i >= nRows || j < 0 || j >= nCols) return null;
  return [i, j];
}

function describe(i, j, tag) {
  const key = quantitySel.value;
  const q = GRID.quantities[key];
  const v = q.values[i][j], s = q.stds[i][j];
  const N = GRID.n_values[j], P = GRID.p_values[i];
  const lr = GRID.lr_used[i][j];
  const lrNote = (lr !== null && lr < 0.0999)
      ? `　<span class="pinned">lr 已降为 ${lr.toFixed(3)}（防发散）</span>` : '';
  return `${tag ? '<span class="pinned">[已固定] </span>' : ''}`
    + `N = <b>${N}</b>　P = <b>${P}</b>　P/N = <b>${(P / N).toFixed(3)}</b>　|　`
    + `${q.label} = <b>${v === null ? '—' : v.toExponential(4)}</b>`
    + (isFinite(s) ? `　(多种子标准差 ${s.toExponential(2)})` : '')
    + `<br>坐标：第 ${j + 1}/${nCols} 列（N），第 ${i + 1}/${nRows} 行（P）`
    + lrNote;
}

cv.addEventListener('mousemove', ev => {
  const cell = cellFromEvent(ev);
  if (cell) {
    const [i, j] = cell;
    if (!pinned) readout.innerHTML = describe(i, j, false);
  } else if (!pinned) {
    readout.textContent = '把鼠标移到图上任意位置。';
  }
});

cv.addEventListener('click', ev => {
  const cell = cellFromEvent(ev);
  if (!cell) return;
  pinned = (pinned && pinned[0] === cell[0] && pinned[1] === cell[1]) ? null : cell;
  readout.innerHTML = pinned ? describe(pinned[0], pinned[1], true) : '已取消固定。';
  draw();
});

cv.addEventListener('wheel', ev => {
  ev.preventDefault();
  const r = cv.getBoundingClientRect();
  const x = (ev.clientX - r.left) * (cv.width / r.width);
  const y = (ev.clientY - r.top) * (cv.height / r.height);
  const cw = W() / (view.x1 - view.x0), ch = H() / (view.y1 - view.y0);
  const ux = view.x0 + (x - M.left) / cw;     // anchor in cell units
  const uy = view.y0 + (y - M.top) / ch;
  const k = ev.deltaY < 0 ? 0.85 : 1 / 0.85;  // zoom in / out
  let w = (view.x1 - view.x0) * k, h = (view.y1 - view.y0) * k;
  w = Math.min(Math.max(w, 1), nCols * 1.5);
  h = Math.min(Math.max(h, 1), nRows * 1.5);
  view.x0 = ux - (ux - view.x0) * (w / (view.x1 - view.x0));
  view.x1 = view.x0 + w;
  view.y0 = uy - (uy - view.y0) * (h / (view.y1 - view.y0));
  view.y1 = view.y0 + h;
  draw();
}, { passive: false });

let dragging = null;
cv.addEventListener('pointerdown', ev => {
  const r = cv.getBoundingClientRect();
  dragging = { x: ev.clientX, y: ev.clientY, view: { ...view } };
  cv.setPointerCapture(ev.pointerId);
});
cv.addEventListener('pointermove', ev => {
  if (!dragging) return;
  const r = cv.getBoundingClientRect();
  const cw = W() / (dragging.view.x1 - dragging.view.x0);
  const ch = H() / (dragging.view.y1 - dragging.view.y0);
  const dx = (ev.clientX - dragging.x) * (cv.width / r.width) / cw;
  const dy = (ev.clientY - dragging.y) * (cv.height / r.height) / ch;
  view.x0 = dragging.view.x0 - dx; view.x1 = dragging.view.x1 - dx;
  view.y0 = dragging.view.y0 - dy; view.y1 = dragging.view.y1 - dy;
  draw();
});
cv.addEventListener('pointerup', () => { dragging = null; });

quantitySel.addEventListener('change', draw);
scaleSel.addEventListener('change', draw);
numbersChk.addEventListener('change', draw);
resetBtn.addEventListener('click', resetView);

// ---- populate ----------------------------------------------------------
for (const [key, label] of __QLIST__) {
  const o = document.createElement('option');
  o.value = key; o.textContent = label;
  quantitySel.appendChild(o);
}
document.getElementById('protect').innerHTML =
  `数据来源：<code>${GRID.meta.source || ''}</code>，每个格点 ${GRID.meta.seeds || '?'} 个种子平均；`
  + `请求的 lr=${GRID.meta.lr_requested || '?'}。格子边框即网格分辨率，不是连续场。`;
draw();
</script>
</body>
</html>
"""


def build_html(grid: dict) -> str:
    payload = json.dumps(grid, ensure_ascii=False, separators=(",", ":"))
    qlist = json.dumps([[k, lab] for k, lab in QUANTITIES], ensure_ascii=False)
    return (HTML
            .replace("__DATA__", payload)
            .replace("__PALETTE__", json.dumps(PALETTE))
            .replace("__QLIST__", qlist))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv) if args.csv else newest_grid()
    out_path = Path(args.out) if args.out else csv_path.with_name("interactive.html")

    grid = load_grid(csv_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(build_html(grid), encoding="utf-8")

    cells = len(grid["n_values"]) * len(grid["p_values"])
    print(f"[interactive] {csv_path}")
    print(f"              {cells} cells, {len(grid['n_values'])} N x "
          f"{len(grid['p_values'])} P, {len(QUANTITIES)} quantities")
    print(f"[output]      {out_path}  ({out_path.stat().st_size / 1024:.0f} KB)")
    if args.open:
        webbrowser.open(out_path.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
