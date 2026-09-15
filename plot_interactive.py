#!/usr/bin/env python
"""Zero-dependency interactive figures (curves), the counterpart of
``make_interactive.py`` which handles the phase heat maps.

The viewer is plain HTML + canvas JavaScript with no plotly, no CDN and no
network access, so the generated file works offline.

Interaction model:

* **hover**  -- a crosshair follows the cursor and a floating box lists every
  visible series at that x, with the nearest sample highlighted;
* **drag**   -- pan; **wheel** -- zoom on the cursor; **shift+drag** -- rubber-band
  zoom into a region;
* **legend** -- click a series to show/hide it;
* **magnifier** -- an optional second panel that follows the cursor and shows a
  zoomed window around it, which is what makes a 2000-point curve readable;
* **y 轴**   -- switch between linear and log, which is a genuine axis change
  (unlike the colour scale of the heat maps).

Usage as a library::

    from plot_interactive import Series, write_interactive_curves
    write_interactive_curves(path, title, xlabel, ylabel, series, base="loss.csv")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Series:
    """One curve: x/y samples plus presentation choices."""

    name: str
    x: np.ndarray
    y: np.ndarray
    color: str = "#1f77b4"
    width: float = 1.6
    markers: bool = False
    visible: bool = True

    def to_json(self) -> dict:
        return {
            "name": self.name,
            "x": [float(v) for v in self.x],
            "y": [float(v) for v in self.y],
            "color": self.color,
            "width": self.width,
            "markers": bool(self.markers),
            "visible": bool(self.visible),
        }


def hlines_to_series(name: str, y: float, x0: float, x1: float,
                     color: str = "#888", width: float = 1.2) -> Series:
    """A horizontal reference line expressed as a 2-point series."""
    return Series(name=name, x=np.array([x0, x1]), y=np.array([y, y]),
                  color=color, width=width)


TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.5 "Microsoft YaHei", "Noto Sans CJK SC", system-ui, sans-serif;
         background: #fafafa; color: #1a1a1a; }
  header { padding: 12px 20px 4px; }
  h1 { font-size: 17px; margin: 0 0 2px; }
  .sub { color: #666; font-size: 12.5px; }
  .toolbar { display: flex; flex-wrap: wrap; gap: 16px; align-items: center;
             padding: 8px 20px; background: #fff; border-bottom: 1px solid #e3e3e3;
             position: sticky; top: 0; z-index: 5; }
  .toolbar label { font-size: 13px; color: #444; }
  select, button, input { font: inherit; }
  select, button { padding: 3px 8px; border: 1px solid #bbb; border-radius: 5px; background: #fff; }
  button { cursor: pointer; } button:hover { background: #f0f0f0; }
  #legend { display: flex; flex-wrap: wrap; gap: 14px; padding: 8px 20px 2px; }
  #legend .item { display: flex; align-items: center; gap: 6px; cursor: pointer;
                  font-size: 13px; user-select: none; }
  #legend .item.off { opacity: .35; text-decoration: line-through; }
  #legend .swatch { width: 15px; height: 3px; border-radius: 2px; }
  .readout { padding: 6px 20px; font-family: ui-monospace, Consolas, monospace;
             font-size: 12.5px; background: #fff; border-bottom: 1px solid #e3e3e3;
             min-height: 30px; white-space: pre; overflow-x: auto; }
  .wrap { padding: 10px 20px 26px; }
  canvas { touch-action: none; cursor: crosshair; display: block; border-radius: 6px; }
  .hint { color: #777; font-size: 12px; padding: 0 20px 22px; max-width: 980px; }
  code { background: #eee; padding: 1px 5px; border-radius: 3px; }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="sub">__SUBTITLE__</div>
</header>

<div class="toolbar">
  <label>y 轴
    <select id="yscale">
      <option value="linear">线性</option>
      <option value="log">对数 log</option>
    </select>
  </label>
  <label><input type="checkbox" id="magnify" __MAGDEFAULT__> 放大镜跟随光标</label>
  <button id="reset">重置视图</button>
  <button id="fitx">只看某一区间…</button>
</div>

<div id="legend"></div>
<div class="readout" id="readout">把鼠标移到图上。</div>
<div class="wrap"><canvas id="cv" width="980" height="560"></canvas></div>
<div class="hint">
  悬停读值 · 滚轮缩放 · 拖动平移 · <b>Shift+拖动框选放大</b> · 点图例开关曲线 · 切换 y 轴对数。
  悬停时右上角会显示放大镜，看 2000 个点的曲线细节用它。
  __SOURCE__
</div>

<script id="fig-data" type="application/json">__DATA__</script>
<script>
const FIG = JSON.parse(document.getElementById('fig-data').textContent);
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const readout = document.getElementById('readout');
const yscaleSel = document.getElementById('yscale');
const magnifyChk = document.getElementById('magnify');
const legendBox = document.getElementById('legend');

const M = { left: 74, right: 22, top: 18, bottom: 54 };
const PL = () => M.left, PT = () => M.top;
const PW = () => cv.width - M.left - M.right, PH = () => cv.height - M.top - M.bottom;
// magnifier occupies the top-right corner of the plot area
const MAG = { w: 300, h: 190, pad: 8 };

let full = null;        // full data range
let view = null;        // {x0,x1,y0,y1,log}
let hover = null;       // position in canvas pixels
let band = null;        // rubber-band selection in pixel space
let drag = null;        // active pan / band gesture
let series = [];

function num(v) { return v === null || !isFinite(v); }

function computeFull() {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const s of series) {
    if (!s.visible) continue;
    for (let i = 0; i < s.x.length; i++) {
      const xv = s.x[i], yv = s.y[i];
      if (num(xv) || num(yv)) continue;
      if (xv < x0) x0 = xv; if (xv > x1) x1 = xv;
      if (yv < y0) y0 = yv; if (yv > y1) y1 = yv;
    }
  }
  if (!isFinite(x0)) { x0 = 0; x1 = 1; y0 = 0; y1 = 1; }
  if (x1 === x0) { x1 = x0 + 1; }
  const pad = (y1 - y0) * 0.06 || Math.abs(y1) * 0.1 || 1;
  return { x0, x1, y0: y0 - pad, y1: y1 + pad };
}

function logMode() { return yscaleSel.value === 'log'; }

function yRange() {
  // In log mode the axis must stay positive: use the smallest positive value so
  // nothing silently disappears. Returns the y limits plus the current mode.
  const base = computeFull();
  if (!logMode()) return { y0: base.y0, y1: base.y1, log: false };
  let lo = Infinity, hi = -Infinity;
  for (const s of series) {
    if (!s.visible) continue;
    for (const v of s.y) { if (!num(v) && v > 0) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
  }
  if (!isFinite(lo)) { lo = 1e-12; hi = 1; }
  if (hi <= lo) hi = lo * 10;
  return { y0: lo / 1.4, y1: hi * 1.4, log: true };
}

function resetView() {
  full = computeFull();
  const y = yRange();   // yRange carries y limits only; x comes from the data
  view = { x0: full.x0, x1: full.x1, y0: y.y0, y1: y.y1, log: y.log };
  band = null;
  draw();
}

function applyMode() {
  // switch linear <-> log: x window is preserved, y is refitted
  full = computeFull();
  const y = yRange();
  view = { x0: view ? view.x0 : full.x0, x1: view ? view.x1 : full.x1,
           y0: y.y0, y1: y.y1, log: y.log };
  draw();
}

// ---- transforms --------------------------------------------------------
function X(v) { return PL() + (v - view.x0) / (view.x1 - view.x0) * PW(); }
function Y(v) {
  if (!view.log) return PT() + (1 - (v - view.y0) / (view.y1 - view.y0)) * PH();
  const a = Math.log10(Math.max(view.y0, 1e-300)), b = Math.log10(view.y1);
  return PT() + (1 - (Math.log10(Math.max(v, 1e-300)) - a) / (b - a)) * PH();
}
function invX(px) { return view.x0 + (px - PL()) / PW() * (view.x1 - view.x0); }
function invY(py) {
  if (!view.log) return view.y0 + (1 - (py - PT()) / PH()) * (view.y1 - view.y0);
  const a = Math.log10(Math.max(view.y0, 1e-300)), b = Math.log10(view.y1);
  const t = 1 - (py - PT()) / PH();
  return Math.pow(10, a + t * (b - a));
}

function niceTicks(lo, hi, n) {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const norm = raw / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(v);
  return out;
}

const fmt = v => {
  if (num(v)) return '—';
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a < 1e-3 || a >= 1e5) return v.toExponential(2);
  return a < 1 ? v.toFixed(4) : v.toFixed(3);
};

// ---- drawing -----------------------------------------------------------
function draw() {
  ctx.clearRect(0, 0, cv.width, cv.height);
  ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, cv.width, cv.height);
  const w = PW(), h = PH();

  // grid + y ticks
  ctx.font = '11px ui-monospace, monospace';
  const yTicks = logMode()
    ? logTicks(view.y0, view.y1)
    : niceTicks(view.y0, view.y1, 6);
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (const v of yTicks) {
    const py = Y(v);
    if (py < PT() - 1 || py > PT() + h + 1) continue;
    ctx.strokeStyle = '#e8e8e8'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PL(), py); ctx.lineTo(PL() + w, py); ctx.stroke();
    ctx.fillStyle = '#444';
    ctx.fillText(logMode() ? v.toExponential(0) : fmt(v), PL() - 7, py);
  }
  // x ticks
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  for (const v of niceTicks(view.x0, view.x1, 8)) {
    const px = X(v);
    if (px < PL() - 1 || px > PL() + w + 1) continue;
    ctx.strokeStyle = '#f0f0f0'; ctx.beginPath();
    ctx.moveTo(px, PT()); ctx.lineTo(px, PT() + h); ctx.stroke();
    ctx.fillStyle = '#444'; ctx.fillText(fmt(v), px, PT() + h + 7);
  }

  // clip plot area
  ctx.save();
  ctx.beginPath(); ctx.rect(PL(), PT(), w, h); ctx.clip();

  // fix the browser's unary-minus handling before drawing
  for (const s of series) {
    if (!s.visible || s.x.length === 0) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width;
    ctx.beginPath();
    let pen = false;
    const stride = Math.max(1, Math.floor(s.x.length / (w * 2)));
    for (let i = 0; i < s.x.length; i += stride) {
      const px = X(s.x[i]), py = Y(s.y[i]);
      if (!isFinite(px) || !isFinite(py)) { pen = false; continue; }
      if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
    if (s.markers && s.x.length <= 400) {
      for (let i = 0; i < s.x.length; i++) {
        const px = X(s.x[i]), py = Y(s.y[i]);
        if (!isFinite(px) || !isFinite(py)) continue;
        ctx.fillStyle = s.color;
        ctx.beginPath(); ctx.arc(px, py, 2.6, 0, 6.2832); ctx.fill();
      }
    }
  }

  // hover crosshair + value dots
  if (hover) {
    const hx = hover.px, hy = hover.py;
    ctx.strokeStyle = '#999'; ctx.setLineDash([4, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(hx, PT()); ctx.lineTo(hx, PT() + h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PL(), hy); ctx.lineTo(PL() + w, hy); ctx.stroke();
    ctx.setLineDash([]);
    for (const s of series) {
      if (!s.visible || s.x.length === 0) continue;
      const xv = invX(hx);
      const i = nearestIndex(s, xv);
      if (i < 0) continue;
      const py = Y(s.y[i]);
      if (!isFinite(py)) continue;
      ctx.fillStyle = s.color;
      ctx.beginPath(); ctx.arc(X(s.x[i]), py, 4, 0, 6.2832); ctx.fill();
      ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }

  // rubber band
  if (band) {
    ctx.fillStyle = 'rgba(40,110,200,0.16)';
    ctx.strokeStyle = 'rgba(40,110,200,0.8)'; ctx.lineWidth = 1;
    const x = Math.min(band.x0, band.x1), y = Math.min(band.y0, band.y1);
    const bw = Math.abs(band.x1 - band.x0), bh = Math.abs(band.y1 - band.y0);
    ctx.fillRect(x, y, bw, bh); ctx.strokeRect(x, y, bw, bh);
  }

  // magnifier (drawn inside the clipped plot area)
  if (magnifyChk.checked && hover) drawMagnifier(hover);
  ctx.restore();

  // axes frame
  ctx.strokeStyle = '#999'; ctx.lineWidth = 1;
  ctx.strokeRect(PL(), PT(), w, h);
  ctx.fillStyle = '#222'; ctx.font = '13px "Microsoft YaHei", sans-serif';
  ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic';
  ctx.fillText(FIG.xlabel, PL() + w / 2, cv.height - 14);
  ctx.save();
  ctx.translate(15, PT() + h / 2); ctx.rotate(-Math.PI / 2);
  ctx.textBaseline = 'top';
  ctx.fillText(FIG.ylabel + (logMode() ? '（对数轴）' : ''), 0, 0);
  ctx.restore();
}

function logTicks(lo, hi) {
  const out = [];
  const a = Math.floor(Math.log10(Math.max(lo, 1e-300)));
  const b = Math.ceil(Math.log10(Math.max(hi, 1e-300)));
  for (let e = a; e <= b; e++) out.push(Math.pow(10, e));
  return out.filter(v => v >= lo && v <= hi * 1.0001);
}

function nearestIndex(s, xv) {
  // binary search on a monotonically increasing x
  const x = s.x;
  if (xv <= x[0]) return 0;
  if (xv >= x[x.length - 1]) return x.length - 1;
  let lo = 0, hi = x.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (x[mid] <= xv) lo = mid; else hi = mid;
  }
  return (xv - x[lo] <= x[hi] - xv) ? lo : hi;
}

function drawMagnifier(pos) {
  const x0 = PL() + PW() - MAG.w - MAG.pad, y0 = PT() + MAG.pad;
  const half = (view.x1 - view.x0) * 0.04;   // window width in data units
  const cx = invX(pos.px);
  let lo = cx - half, hi = cx + half;
  // y window: spread of the visible series inside that x window
  let ylo = Infinity, yhi = -Infinity;
  for (const s of series) {
    if (!s.visible) continue;
    for (let i = 0; i < s.x.length; i++) {
      if (s.x[i] < lo || s.x[i] > hi) continue;
      if (num(s.y[i])) continue;
      ylo = Math.min(ylo, s.y[i]); yhi = Math.max(yhi, s.y[i]);
    }
  }
  if (!isFinite(ylo)) { ylo = view.y0; yhi = view.y1; }
  const pad = (yhi - ylo) * 0.15 || Math.abs(yhi) * 0.1 || 1;
  ylo -= pad; yhi += pad;

  ctx.save();
  ctx.beginPath(); ctx.rect(x0, y0, MAG.w, MAG.h); ctx.clip();
  ctx.fillStyle = 'rgba(255,255,255,0.94)'; ctx.fillRect(x0, y0, MAG.w, MAG.h);
  const mx = v => x0 + (v - lo) / (hi - lo) * MAG.w;
  const my = v => y0 + (1 - (v - ylo) / (yhi - ylo)) * MAG.h;
  for (const s of series) {
    if (!s.visible) continue;
    ctx.strokeStyle = s.color; ctx.lineWidth = s.width + 0.4;
    ctx.beginPath();
    let pen = false;
    for (let i = 0; i < s.x.length; i++) {
      if (s.x[i] < lo || s.x[i] > hi) { pen = false; continue; }
      const px = mx(s.x[i]), py = my(s.y[i]);
      if (!isFinite(px) || !isFinite(py)) { pen = false; continue; }
      if (!pen) { ctx.moveTo(px, py); pen = true; } else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }
  ctx.strokeStyle = '#ccc'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(mx(cx), y0); ctx.lineTo(mx(cx), y0 + MAG.h); ctx.stroke();
  ctx.restore();
  ctx.strokeStyle = '#bbb'; ctx.strokeRect(x0, y0, MAG.w, MAG.h);
  ctx.fillStyle = '#666'; ctx.font = '11px ui-monospace, monospace';
  ctx.textAlign = 'left'; ctx.textBaseline = 'bottom';
  ctx.fillText('放大镜 x∈[' + fmt(lo) + ', ' + fmt(hi) + ']', x0 + 6, y0 + MAG.h - 5);
}

// ---- readout -----------------------------------------------------------
function updateReadout() {
  if (!hover) { readout.textContent = '把鼠标移到图上。'; return; }
  const xv = invX(hover.px);
  let best = null;
  for (const s of series) {
    if (!s.visible || s.x.length === 0) continue;
    const i = nearestIndex(s, xv);
    if (i < 0) continue;
    if (!best || Math.abs(s.x[i] - xv) < Math.abs(best.x - xv)) best = { s, i };
  }
  if (!best) { readout.textContent = `x = ${fmt(xv)}`; return; }
  const parts = [`x = ${fmt(best.s.x[best.i])}`];
  for (const s of series) {
    if (!s.visible || s.x.length === 0) continue;
    const i = nearestIndex(s, xv);
    if (i < 0) continue;
    parts.push(`${s.name} = ${fmt(s.y[i])}`);
  }
  readout.textContent = parts.join('   |   ');
}

// ---- interaction -------------------------------------------------------
function pos(ev) {
  const r = cv.getBoundingClientRect();
  return { px: (ev.clientX - r.left) * (cv.width / r.width),
           py: (ev.clientY - r.top) * (cv.height / r.height) };
}
const inPlot = p => p.px >= PL() && p.px <= PL() + PW() && p.py >= PT() && p.py <= PT() + PH();

cv.addEventListener('mousemove', ev => {
  const p = pos(ev);
  hover = inPlot(p) ? p : null;
  if (drag && !ev.shiftKey) {
    const cw = PW() / (drag.view.x1 - drag.view.x0), chh = PH() / (drag.view.y1 - drag.view.y0);
    const dx = (p.px - drag.px) / cw, dy = (p.py - drag.py) / chh;
    view.x0 = drag.view.x0 - dx; view.x1 = drag.view.x1 - dx;
    view.y0 = drag.view.y0 - dy; view.y1 = drag.view.y1 - dy;
  } else if (drag && ev.shiftKey) {
    band = { x0: drag.px, y0: drag.py, x1: p.px, y1: p.py };
  }
  updateReadout(); draw();
});

cv.addEventListener('pointerdown', ev => {
  const p = pos(ev);
  if (!inPlot(p)) return;
  drag = { px: p.px, py: p.py, view: { ...view }, shifted: ev.shiftKey };
  band = ev.shiftKey ? { x0: p.px, y0: p.py, x1: p.px, y1: p.py } : null;
  cv.setPointerCapture(ev.pointerId);
});

cv.addEventListener('pointerup', ev => {
  if (drag && drag.shifted && band) {
    const x0 = invX(Math.min(band.x0, band.x1)), x1 = invX(Math.max(band.x0, band.x1));
    const y0 = invY(Math.max(band.y0, band.y1)), y1 = invY(Math.min(band.y0, band.y1));
    if (Math.abs(x1 - x0) > 0 && Math.abs(y1 - y0) > 0
        && (view.log ? y0 > 0 && y1 > 0 : true)) {
      view = { x0, x1, y0, y1 };
    }
  }
  drag = null; band = null; draw();
});

cv.addEventListener('wheel', ev => {
  ev.preventDefault();
  const p = pos(ev);
  if (!inPlot(p)) return;
  const k = ev.deltaY < 0 ? 0.82 : 1 / 0.82;
  const ax = invX(p.px), ay = invY(p.py);
  view.x0 = ax - (ax - view.x0) * k; view.x1 = ax + (view.x1 - ax) * k;
  if (!view.log) {
    view.y0 = ay - (ay - view.y0) * k; view.y1 = ay + (view.y1 - ay) * k;
  }
  hover = p; updateReadout(); draw();
}, { passive: false });

yscaleSel.addEventListener('change', applyMode);
magnifyChk.addEventListener('change', draw);
document.getElementById('reset').addEventListener('click', resetView);
document.getElementById('fitx').addEventListener('click', () => {
  const a = prompt('x 轴区间，例如 0 200', `${fmt(view.x0)} ${fmt(view.x1)}`);
  if (!a) return;
  const [s0, s1] = a.split(/[\\s,]+/).map(Number);
  if (isFinite(s0) && isFinite(s1) && s1 > s0) { view.x0 = s0; view.x1 = s1; draw(); }
});

// ---- build legend ------------------------------------------------------
function buildLegend() {
  legendBox.innerHTML = '';
  for (const s of series) {
    const d = document.createElement('div');
    d.className = 'item' + (s.visible ? '' : ' off');
    d.innerHTML = `<span class="swatch" style="background:${s.color}"></span>${s.name}`;
    d.onclick = () => { s.visible = !s.visible; toggleSeries(); };
    legendBox.appendChild(d);
  }
}

function toggleSeries() {
  // the axis limits depend on which series are visible, so refit y (keep x)
  const y = yRange();
  view.y0 = y.y0; view.y1 = y.y1; view.log = y.log;
  buildLegend(); draw();
}

series = FIG.series.map(s => ({ ...s }));
buildLegend();
resetView();
</script>
</body>
</html>
"""


def build_html(title: str, subtitle: str, xlabel: str, ylabel: str,
               series: list[Series], source: str = "",
               magnify_default: bool = False) -> str:
    payload = json.dumps({"xlabel": xlabel, "ylabel": ylabel,
                          "series": [s.to_json() for s in series]},
                         ensure_ascii=False, separators=(",", ":"))
    return (TEMPLATE
            .replace("__TITLE__", title)
            .replace("__SUBTITLE__", subtitle)
            .replace("__DATA__", payload)
            .replace("__MAGDEFAULT__", "checked" if magnify_default else "")
            .replace("__SOURCE__", source))


def write_interactive_curves(path: str | Path, title: str, xlabel: str, ylabel: str,
                             series: list[Series], subtitle: str = "",
                             source: str = "", magnify_default: bool = False) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(title, subtitle, xlabel, ylabel, series,
                              source=source, magnify_default=magnify_default),
                   encoding="utf-8")
    return out


def series_from_columns(x, y, name: str, **kw) -> Series:
    """Convenience constructor that coerces to float arrays."""
    return Series(name=name, x=np.asarray(x, dtype=float),
                  y=np.asarray(y, dtype=float), **kw)


# ----------------------------------------------------------------------
def read_loss_csv(path: str | Path) -> dict[str, Series]:
    """Read one of this project's loss.csv files back into interactive series."""
    import csv as _csv

    cols: dict[str, list] = {"epoch": [], "train_loss": [], "test_loss": [],
                             "test_loss_analytic": [], "weight_distance": []}
    with Path(path).open(encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            if not row.get("epoch"):
                continue
            for key in cols:
                value = row.get(key) or ""
                cols[key].append(float(value) if value else None)

    colors = {"train_loss": "#1f77b4", "test_loss": "#d62728",
              "test_loss_analytic": "#2ca02c", "weight_distance": "#9467bd"}

    def pick(key: str) -> tuple[list[float], list[float]]:
        xs = [e for e, y in zip(cols["epoch"], cols[key]) if y is not None]
        ys = [y for y in cols[key] if y is not None]
        return xs, ys

    out: dict[str, Series] = {}
    for key in ("train_loss", "test_loss", "test_loss_analytic", "weight_distance"):
        xs, ys = pick(key)
        if xs:
            out[key] = Series(name=key, x=np.array(xs), y=np.array(ys),
                              color=colors[key],
                              markers=key == "test_loss",
                              visible=key != "weight_distance")
    return out


def _cli(argv: list[str] | None = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Build a zero-dependency interactive curve figure from a loss.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--csv", required=True, help="path to a run's loss.csv")
    p.add_argument("--out", default=None, help="output HTML (default: beside the CSV)")
    p.add_argument("--title", default=None)
    p.add_argument("--publish", type=str, nargs="?", const="docs/loss_curves.html",
                   default=None,
                   help="also copy into the repo so the figure is version-controlled")
    args = p.parse_args(argv)

    csv_path = Path(args.csv)
    s = read_loss_csv(csv_path)
    if not s:
        raise SystemExit(f"no usable columns in {csv_path}")

    # the noise floor is the analytic floor minus the weight-error term, so draw
    # it only when the analytic column makes it recoverable
    title = args.title or f"{csv_path.parent.name} — 交互曲线"
    out = Path(args.out) if args.out else csv_path.with_name("loss_curves_interactive.html")
    write_interactive_curves(
        out, title=title, xlabel="epoch", ylabel="MSE",
        series=[s["train_loss"], s["test_loss"], s["test_loss_analytic"]],
        subtitle="悬停读值 · 滚轮缩放 · 拖动平移 · Shift+拖动框选 · 点图例开关",
        source=f"数据来源：<code>{csv_path.name}</code>",
        magnify_default=True,
    )
    print(f"[output] {out}  ({out.stat().st_size / 1024:.0f} KB)")
    if args.publish:
        pub = Path(args.publish)
        pub.parent.mkdir(parents=True, exist_ok=True)
        pub.write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"[publish] {pub}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
