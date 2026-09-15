// Headless checks for the interactive curve viewer.
// Stubs enough DOM/canvas to run the generated <script>, then verifies the axis
// transforms, the click->data mapping, series toggling and the readout.
//
// Run:  node docs/_check_curves.mjs <figure.html>

import { readFileSync } from 'node:fs';

const file = process.argv[2];
const html = readFileSync(file, 'utf8');

const dataMatch = html.match(/<script id="fig-data" type="application\/json">([\s\S]*?)<\/script>/);
const mainMatch = html.match(/<script>\s*([\s\S]*?)\s*<\/script>\s*<\/body>/);
if (!dataMatch || !mainMatch) { console.error('could not find script blocks'); process.exit(1); }
const FIG = JSON.parse(dataMatch[1]);

// ---- DOM / canvas stubs ------------------------------------------------
const handlers = {};
const ctx = new Proxy({}, {
  get: (_, p) => {
    if (p === 'canvas') return { width: 980, height: 560 };
    return typeof p === 'string' &&
      /^(clearRect|fillRect|strokeRect|beginPath|moveTo|lineTo|stroke|fill|fillText|save|restore|translate|rotate|rect|clip|setLineDash|arc)$/.test(p)
      ? () => {} : (p === 'measureText' ? () => ({ width: 10 }) : () => {});
  },
  set: () => true,
});
function el(id) {
  return {
    id, value: '', checked: false, textContent: '', innerHTML: '', className: '',
    style: {}, children: [], onclick: null,
    appendChild(c) { this.children.push(c); }, addEventListener(t, fn) { (handlers[id] ??= {})[t] = fn; },
    getContext: () => ctx, setPointerCapture() {}, getBoundingClientRect: () => ({ left: 0, top: 0, width: 980, height: 560 }),
    width: 980, height: 560,
  };
}
const els = {};
globalThis.document = {
  getElementById: (id) => {
    if (id === 'fig-data') return { textContent: dataMatch[1] };
    if (!els[id]) {
      els[id] = el(id);
      if (id === 'yscale') els[id].value = 'linear';
    }
    return els[id];
  },
  createElement: () => el('div'),
};
globalThis.prompt = () => null;
globalThis.window = {};

let api;
try {
  api = new Function(`${mainMatch[1]}
; return { FIG, X: (...a) => X(...a), Y: (...a) => Y(...a), invX, invY, nearestIndex,
             series: () => series, view: () => view, fmt, buildLegend, applyMode, resetView,
             computeFull, yRange, toggleSeries };`)();
} catch (e) { console.error('viewer script threw:', e.message); process.exit(1); }

let failures = 0;
const ok = (n, c, d = '') => { console.log(`[${c ? 'PASS' : 'FAIL'}] ${n}${d ? ' -- ' + d : ''}`); if (!c) failures++; };

// 1. payload
const S = api.FIG.series;
ok('payload parsed', S.length >= 2 && S.every(s => s.x.length === s.y.length && s.x.length > 0),
   `${S.length} series, lengths ${S.map(s => s.x.length).join('/')}`);
ok('all samples finite',
   S.every(s => s.x.every(Number.isFinite) && s.y.every(Number.isFinite)));

// 2. axis transforms are exact inverses, and the plot corners map to the limits
const v = api.view();
ok('view has x and y limits', Number.isFinite(v.x0) && v.x0 < v.x1 && v.y0 < v.y1,
   `x[${api.fmt(v.x0)},${api.fmt(v.x1)}] y[${api.fmt(v.y0)},${api.fmt(v.y1)}]`);

let worstX = 0, worstY = 0;
for (let k = 0; k <= 20; k++) {
  const xv = v.x0 + (v.x1 - v.x0) * k / 20;
  worstX = Math.max(worstX, Math.abs(api.invX(api.X(xv)) - xv) / Math.max(1e-12, v.x1 - v.x0));
  const yv = v.y0 + (v.y1 - v.y0) * k / 20;
  worstY = Math.max(worstY, Math.abs(api.invY(api.Y(yv)) - yv) / Math.max(1e-12, v.y1 - v.y0));
}
ok('x transform round-trips', worstX < 1e-9, `worst rel err ${worstX.toExponential(2)}`);
ok('y transform round-trips', worstY < 1e-9, `worst rel err ${worstY.toExponential(2)}`);

// 3. nearestIndex picks the truly nearest sample (brute-force cross-check)
let mismatches = 0;
for (const s of S) {
  if (s.x.length < 3) continue;
  for (let k = 0; k < 50; k++) {
    const target = s.x[0] + (s.x[s.x.length - 1] - s.x[0]) * (k / 49);
    let best = 0, bd = Infinity;
    for (let i = 0; i < s.x.length; i++) {
      const d = Math.abs(s.x[i] - target);
      if (d < bd) { bd = d; best = i; }
    }
    // binary search must land on one of the two neighbours of the target
    const got = api.nearestIndex(s, target);
    if (Math.abs(s.x[got] - target) > bd + 1e-9) mismatches++;
  }
}
ok('nearest-sample lookup matches brute force', mismatches === 0, `${mismatches} misses`);

// 4. log mode refits the y axis to positive values only
api.applyMode();
els.yscale.value = 'log';
api.applyMode();
const lv = api.view();
ok('log mode keeps the y axis positive', lv.y0 > 0 && lv.y1 > lv.y0,
   `y[${lv.y0.toExponential(2)}, ${lv.y1.toExponential(2)}]`);
let logRound = 0;
for (let k = 1; k <= 10; k++) {
  const yv = Math.pow(10, Math.log10(lv.y0) + (Math.log10(lv.y1) - Math.log10(lv.y0)) * k / 11);
  logRound = Math.max(logRound, Math.abs(api.invY(api.Y(yv)) - yv) / yv);
}
ok('log y transform round-trips', logRound < 1e-9, `worst rel err ${logRound.toExponential(2)}`);

els.yscale.value = 'linear';
api.resetView();

// 5. toggling a series removes it from the y range without breaking the view
const before = { ...api.view() };
api.series()[0].visible = false;
api.toggleSeries();
const after = api.view();
ok('toggling a series keeps the view valid',
   Number.isFinite(after.y0) && after.y0 < after.y1 && after.x0 === before.x0,
   `y[${api.fmt(after.y0)},${api.fmt(after.y1)}]`);
api.series()[0].visible = true;
api.toggleSeries();

// 6. readout never produces NaN
const bad = [0, 1e-35, 2.5e-4, 1, 12345.678, 1e9].map(api.fmt).filter(s => /NaN|undefined/.test(s));
ok('number formatting is safe', bad.length === 0, bad.join(','));

// 7. sanity: every series carries a readable name, and a loss figure contains
//    both a train and a test curve (a sweep overlay names runs instead, so the
//    train/test expectation only applies when those names are present)
ok('every series has a non-empty name', S.every(s => typeof s.name === 'string' && s.name.trim().length > 0),
   S.map(s => s.name).join(' | ').slice(0, 80));
const names = S.map(s => s.name.toLowerCase());
const isRunOverlay = names.some(n => n.includes('p/n='));
ok('loss figure carries train and test curves (or is a run overlay)',
   isRunOverlay || (names.some(n => n.includes('train')) && names.some(n => n.includes('test'))),
   isRunOverlay ? 'run overlay' : names.join(' | '));

console.log(failures === 0 ? '\nALL CURVE VIEWER CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
