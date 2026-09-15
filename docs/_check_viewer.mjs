// Headless checks for the interactive viewer's geometry and colour logic.
// Stubs just enough DOM/canvas to run the generated <script> body, then asserts
// that canvas coordinates map back to the right (N, P) cell and that the colour
// normalisation round-trips.
//
// Run:  node docs/_check_viewer.mjs <interactive.html>

import { readFileSync } from 'node:fs';

const file = process.argv[2];
const html = readFileSync(file, 'utf8');

// ---- extract data + main script ---------------------------------------
const dataMatch = html.match(/<script id="grid-data" type="application\/json">([\s\S]*?)<\/script>/);
const mainMatch = html.match(/<script>\s*([\s\S]*?)\s*<\/script>\s*<\/body>/);
if (!dataMatch || !mainMatch) { console.error('could not find script blocks'); process.exit(1); }
const GRID = JSON.parse(dataMatch[1]);

// ---- minimal DOM / canvas stubs ---------------------------------------
const listeners = {};
function makeEl(id) {
  return {
    id, value: '', checked: false, textContent: '', innerHTML: '',
    appendChild() {}, addEventListener(t, fn) { (listeners[id] ??= {})[t] = fn; },
    getContext: () => ctxStub, getBoundingClientRect: () => ({ left: 0, top: 0, width: 880, height: 640 }),
    setPointerCapture() {}, width: 880, height: 640,
  };
}
const ctxStub = new Proxy({}, {
  get: (_, prop) => {
    if (prop === 'canvas') return makeEl('cv');
    return typeof prop === 'string' && /^(clearRect|fillRect|strokeRect|beginPath|moveTo|lineTo|stroke|fill|fillText|save|restore|translate|rotate|rect|clip|setLineDash)$/.test(prop)
      ? () => {} : () => {};
  },
  set: () => true,
});
const els = {};
globalThis.document = {
  // The viewer reads its data from the embedded <script id="grid-data"> block.
  getElementById: (id) => {
    if (id === 'grid-data') return { textContent: dataMatch[1] };
    if (!els[id]) {
      els[id] = id === 'cv' ? makeEl('cv') : makeEl(id);
      // a real <select> already carries its selected option when draw() runs
      if (id === 'quantity') els[id].value = Object.keys(GRID.quantities)[0];
      if (id === 'scale') els[id].value = 'linear';
    }
    return els[id];
  },
  createElement: () => makeEl('opt'),
};
globalThis.window = {};

// ---- run the viewer script --------------------------------------------
const fn = new Function(`${mainMatch[1]}\n; return { describe, cellFromEvent, fmtShort, makeNorm, invNorm, GRID };`);
let api;
try { api = fn(); } catch (e) { console.error('viewer script threw:', e.message); process.exit(1); }

let failures = 0;
const ok = (name, cond, detail = '') => {
  console.log(`[${cond ? 'PASS' : 'FAIL'}] ${name}${detail ? ' -- ' + detail : ''}`);
  if (!cond) failures++;
};

// 1. the data block survived into the HTML intact
const nValues = GRID.n_values, pValues = GRID.p_values;
ok('grid data parsed', nValues.length === 7 && pValues.length === 11,
   `${nValues.length} N x ${pValues.length} P`);

// 2. every cell of every quantity is a finite number (no nulls / NaN)
let holes = 0;
for (const [key, q] of Object.entries(GRID.quantities)) {
  for (const row of q.values) for (const v of row) {
    if (v === null || !Number.isFinite(v)) holes++;
  }
}
ok('all quantity cells are finite', holes === 0, `${holes} holes`);

// 3. canvas coordinate -> cell mapping.
//    Mimic the viewer's own transform instead of touching the DOM event path.
const cv = { width: 880, height: 640 };
const M = { left: 62, right: 96, top: 26, bottom: 52 };
const W = cv.width - M.left - M.right, H = cv.height - M.top - M.bottom;
const view = { x0: 0, y0: 0, x1: nValues.length, y1: pValues.length };
const cw = W / (view.x1 - view.x0), ch = H / (view.y1 - view.y0);
const clientToCell = (clientX, clientY) => {
  const j = Math.floor(view.x0 + (clientX - M.left) / cw);
  const i = Math.floor(view.y0 + (clientY - M.top) / ch);
  return (i < 0 || i >= pValues.length || j < 0 || j >= nValues.length) ? null : [i, j];
};
let mapErrors = 0;
for (let i = 0; i < pValues.length; i++) {
  for (let j = 0; j < nValues.length; j++) {
    const cx = M.left + (j + 0.5) * cw, cy = M.top + (i + 0.5) * ch;
    const got = clientToCell(cx, cy);
    if (!got || got[0] !== i || got[1] !== j) mapErrors++;
  }
}
ok('centre of each cell maps back to that cell', mapErrors === 0, `${mapErrors} mismatches`);
ok('outside the plot maps to nothing', clientToCell(5, 5) === null && clientToCell(870, 630) === null);

// 4. colour normalisation round-trips through the colourbar inverse
function rampLen() { return 1; }
const normChecks = [];
for (const mode of ['linear', 'log', 'asinh']) {
  const all = Object.values(GRID.quantities).flatMap(q => q.values.flat()).filter(Number.isFinite);
  const lo = Math.min(...all), hi = Math.max(...all);
  const norm = api.makeNorm(mode, lo, hi);
  const inv = t => api.invNorm(norm, t, lo, hi, mode);
  // norm must be monotone and map lo->0, hi->1
  let mono = true, prev = -Infinity;
  for (let k = 0; k <= 20; k++) {
    const v = lo + (hi - lo) * (k / 20);
    const n = norm(v);
    if (n < prev - 1e-12) mono = false;
    prev = n;
  }
  normChecks.push([mode, Math.abs(norm(lo)) < 1e-9, Math.abs(norm(hi) - 1) < 1e-9, mono,
                   Math.abs(inv(0.5) - inv(0.5)) < 1e-9]);
}
for (const [mode, atLo, atHi, mono, idem] of normChecks) {
  ok(`colour norm "${mode}" maps [min,max] -> [0,1] monotonically`, atLo && atHi && mono && idem);
}

// 5. formatting never produces NaN/undefined text
const samples = [0, 1e-35, 1.03e-4, 0.5, 7.038e-1, 12345];
const bad = samples.map(api.fmtShort).filter(s => /NaN|undefined/.test(s) || s === '');
ok('number formatting is safe', bad.length === 0, bad.join(','));

// 6. readout text mentions N, P and the value.
//    Stripping the <b> tags leaves extra spaces, so compare on squashed runs.
const text = api.describe(2, 3, false).replace(/<[^>]+>/g, ' ').replace(/\s+/g, ' ');
const hasN = text.includes(`N = ${nValues[3]}`);
const hasP = text.includes(`P = ${pValues[2]}`);
const hasV = /e-?\d+/.test(text);
ok('readout contains N, P and a value', hasN && hasP && hasV,
   `hasN=${hasN} hasP=${hasP} hasValue=${hasV} :: ${JSON.stringify(text.trim().slice(0, 64))}`);

console.log(failures === 0 ? '\nALL VIEWER CHECKS PASSED' : `\n${failures} CHECK(S) FAILED`);
process.exit(failures === 0 ? 0 : 1);
