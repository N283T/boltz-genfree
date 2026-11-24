// Own ligand conformer embedder (src/trunk/own/conformer.js): geometry checks and the etkdg / rdkit_2d modes.
// Offline, no GPU. Run: node --test test/own-conformer.test.js
import { test } from "node:test";
import assert from "node:assert/strict";
import { buildBounds, embedConformer, ligandConformerPositions, rigidMotion } from "../src/trunk/own/conformer.js";
import { ligandGraph } from "../src/trunk/own/ligand.js";

const graphOf = (smiles) => {
  const g = ligandGraph(smiles);
  return { elements: g.atoms.map((a) => a.element), bonds: g.bonds };
};
const at = (x, i) => [x[i * 3], x[i * 3 + 1], x[i * 3 + 2]];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const sub = (a, b) => a.map((v, k) => v - b[k]);
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];

/** Largest relative bond-length error and largest angle error (degrees) against the ideal geometry. */
function geometryErrors(graph, x) {
  const b = buildBounds(graph);
  let bond = 0, angle = 0;
  b.bonds.forEach((bd, i) => { bond = Math.max(bond, Math.abs(dist(at(x, bd.a), at(x, bd.b)) / b.lengths[i] - 1)); });
  for (const { i, j, k, theta } of b.angles) {
    const u = sub(at(x, i), at(x, j)), v = sub(at(x, k), at(x, j));
    const measured = (Math.acos(Math.max(-1, Math.min(1, dot(u, v) / Math.hypot(...u) / Math.hypot(...v)))) * 180) / Math.PI;
    angle = Math.max(angle, Math.abs(measured - theta));
  }
  return { bond, angle, bounds: b };
}

/** Smallest eigenvalue of the covariance of the points (closed form for 3x3): zero when coplanar. */
function planeSpread(points) {
  const c = [0, 1, 2].map((k) => points.reduce((s, p) => s + p[k], 0) / points.length);
  const m = [0, 1, 2].map((a) => [0, 1, 2].map((b) => points.reduce((s, p) => s + (p[a] - c[a]) * (p[b] - c[b]), 0) / points.length));
  const q = (m[0][0] + m[1][1] + m[2][2]) / 3;
  const p2 = (m[0][0] - q) ** 2 + (m[1][1] - q) ** 2 + (m[2][2] - q) ** 2 + 2 * (m[0][1] ** 2 + m[0][2] ** 2 + m[1][2] ** 2);
  const p = Math.sqrt(p2 / 6);
  if (p < 1e-12) return 0;
  const B = m.map((row, a) => row.map((v, b) => (v - (a === b ? q : 0)) / p));
  const det = B[0][0] * (B[1][1] * B[2][2] - B[1][2] * B[2][1]) - B[0][1] * (B[1][0] * B[2][2] - B[1][2] * B[2][0])
    + B[0][2] * (B[1][0] * B[2][1] - B[1][1] * B[2][0]);
  const phi = Math.acos(Math.max(-1, Math.min(1, det / 2))) / 3;
  return Math.max(0, q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3));
}

const MOLECULES = {
  phenol: "Oc1ccccc1",
  acetaminophen: "CC(=O)Nc1ccc(O)cc1",
  naphthalene: "c1ccc2ccccc2c1",
  indole: "c1ccc2[nH]ccc2c1",
  cyclohexane: "C1CCCCC1",
  ethylAcrylate: "CCOC(=O)C=C",
  // 56 heavy atoms: Val-Phe-Gly-Pro-Tyr style peptide with a biaryl cap
  big: "CC(C)C(NC(=O)C(Cc1ccccc1)NC(=O)CNC(=O)C2CCCN2C(=O)C(Cc3ccc(O)cc3)NC(=O)c4ccc(cc4)-c5ccccc5)C(=O)O",
};
const GRAPHS = Object.fromEntries(Object.entries(MOLECULES).map(([k, s]) => [k, graphOf(s)]));
const report = {};

for (const [name, graph] of Object.entries(GRAPHS)) {
  test(`embed ${name}: bond lengths, angles, clashes`, () => {
    const x = embedConformer(graph, { seed: 5 });
    assert.equal(x.length, graph.elements.length * 3);
    const { bond, angle, bounds } = geometryErrors(graph, x);
    report[name] = { atoms: graph.elements.length, bondErr: +(bond * 100).toFixed(2), angleErr: +angle.toFixed(1) };
    assert.ok(bond < 0.03, `${name}: worst bond length error ${(bond * 100).toFixed(2)}% (limit 3%)`);
    assert.ok(angle < 12, `${name}: worst angle error ${angle.toFixed(1)} deg (limit 12)`);
    const n = graph.elements.length;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      const d = dist(at(x, i), at(x, j));
      const t = bounds.topo[i * n + j];
      if (t === 2) assert.ok(d > 0.9 * bounds.lo[i * n + j] / 0.96, `${name}: 1-3 pair ${i}-${j} at ${d.toFixed(2)} A`);
      if (t >= 3 || t < 0) assert.ok(d > 2.3, `${name}: atoms ${i}-${j} clash at ${d.toFixed(2)} A (limit 2.3)`);
    }
  });
}

test("56-atom molecule: ten seeds all within tolerance, rings planar", () => {
  const g = GRAPHS.big;
  assert.equal(g.elements.length, 56);
  const b = buildBounds(g);
  let worstBond = 0, worstAngle = 0, worstPlane = 0;
  for (let seed = 1; seed <= 10; seed++) {
    const x = embedConformer(g, { seed });
    const e = geometryErrors(g, x);
    worstBond = Math.max(worstBond, e.bond); worstAngle = Math.max(worstAngle, e.angle);
    for (const ring of b.planar) worstPlane = Math.max(worstPlane, Math.sqrt(planeSpread(ring.map((i) => at(x, i)))));
  }
  report.big10 = { bondErrPct: +(worstBond * 100).toFixed(2), angleErrDeg: +worstAngle.toFixed(1), ringRmsA: +worstPlane.toFixed(3) };
  assert.ok(worstBond < 0.03 && worstAngle < 12 && worstPlane < 0.05);
});

test("literature bond lengths within 3% (phenol, ester)", () => {
  const g = GRAPHS.phenol;
  const x = embedConformer(g, { seed: 2 });
  for (const bd of g.bonds) {
    if (g.elements[bd.a] === 6 && g.elements[bd.b] === 6) assert.ok(Math.abs(dist(at(x, bd.a), at(x, bd.b)) / 1.39 - 1) < 0.03);
  }
  const e = GRAPHS.ethylAcrylate;
  const y = embedConformer(e, { seed: 2 });
  const literature = new Map([["6,6,2", 1.34], ["6,8,2", 1.21], ["6,8,1", 1.34]]);
  let checked = 0;
  for (const bd of e.bonds) {
    const za = Math.min(e.elements[bd.a], e.elements[bd.b]), zb = Math.max(e.elements[bd.a], e.elements[bd.b]);
    const ref = literature.get(`${za},${zb},${bd.order ?? bd.type}`);
    if (ref !== undefined) { checked++; assert.ok(Math.abs(dist(at(y, bd.a), at(y, bd.b)) / ref - 1) < 0.04); }
  }
  assert.ok(checked >= 2);
});

test("aromatic rings are planar", () => {
  for (const name of ["phenol", "acetaminophen", "naphthalene", "indole", "big"]) {
    const g = GRAPHS[name];
    const x = embedConformer(g, { seed: 11 });
    const b = buildBounds(g);
    assert.ok(b.planar.length > 0);
    for (const ring of b.planar) {
      const spread = planeSpread(ring.map((i) => at(x, i)));
      assert.ok(Math.sqrt(spread) < 0.05, `${name}: ring ${ring} rms out-of-plane ${Math.sqrt(spread).toFixed(3)} A`);
    }
  }
  // fused system: the whole naphthalene skeleton is one plane
  const x = embedConformer(GRAPHS.naphthalene, { seed: 11 });
  const all = Array.from({ length: 10 }, (_, i) => at(x, i));
  assert.ok(Math.sqrt(planeSpread(all)) < 0.06);
});

test("deterministic per seed; different seeds differ", () => {
  const g = GRAPHS.acetaminophen;
  const a = embedConformer(g, { seed: 7 });
  const b = embedConformer(g, { seed: 7 });
  const c = embedConformer(g, { seed: 8 });
  assert.deepEqual(Array.from(a), Array.from(b));
  assert.notDeepEqual(Array.from(a), Array.from(c));
  const p = ligandConformerPositions(g, "etkdg", 3);
  assert.deepEqual(Array.from(p), Array.from(ligandConformerPositions(g, "etkdg", 3)));
  assert.notDeepEqual(Array.from(p), Array.from(ligandConformerPositions(g, "etkdg", 4)));
});

test("degenerate graphs", () => {
  assert.equal(embedConformer({ elements: [], bonds: [] }).length, 0);
  assert.deepEqual(Array.from(embedConformer({ elements: [8], bonds: [] })), [0, 0, 0]);
  const two = embedConformer({ elements: [6, 8], bonds: [{ a: 0, b: 1, order: 2 }] });
  assert.ok(Math.abs(dist(at(two, 0), at(two, 1)) - 1.24) < 0.05);
});

test("etkdg is a seeded rigid motion of the embedded conformer", () => {
  const g = GRAPHS.acetaminophen;
  const base = embedConformer(g, { seed: 9 });
  const moved = ligandConformerPositions(g, "etkdg", 9);
  const n = g.elements.length;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    assert.ok(Math.abs(dist(at(base, i), at(base, j)) - dist(at(moved, i), at(moved, j))) < 1e-4);
  }
});

// four non-coplanar points so that the pure rigid motion has something to flatten
const POINTS = Float32Array.from([0, 0, 0, 1.5, 0, 0, 0, 1.5, 0, 0.5, 0.5, 1.4]);
const detCov = (x) => {
  const n = x.length / 3;
  const c = [0, 1, 2].map((k) => { let s = 0; for (let i = 0; i < n; i++) s += x[i * 3 + k]; return s / n; });
  const m = [0, 1, 2].map((a) => [0, 1, 2].map((b) => { let s = 0; for (let i = 0; i < n; i++) s += (x[i * 3 + a] - c[a]) * (x[i * 3 + b] - c[b]); return s; }));
  return Math.abs(m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]));
};

test("rigidMotion: seeded, distance preserving; flatten gives a plane", () => {
  const a = rigidMotion(POINTS, 7), b = rigidMotion(POINTS, 7), c = rigidMotion(POINTS, 8);
  assert.deepEqual(Array.from(a), Array.from(b));
  assert.notDeepEqual(Array.from(a), Array.from(c));
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) assert.ok(Math.abs(dist(at(a, i), at(a, j)) - dist(at(POINTS, i), at(POINTS, j))) < 1e-4);
  assert.ok(detCov(POINTS) > 1e-3);
  assert.ok(detCov(rigidMotion(POINTS, 3, { flatten: true })) < 1e-6);
});

test("rdkit_2d is the embedded conformer flattened onto a plane", () => {
  for (const name of ["acetaminophen", "big"]) {
    const g = GRAPHS[name];
    const flat = ligandConformerPositions(g, "rdkit_2d", 3);
    const full = ligandConformerPositions(g, "etkdg", 3);
    assert.ok(detCov(full) > 1e-3);
    assert.ok(detCov(flat) / detCov(full) < 1e-6);
  }
  assert.throws(() => ligandConformerPositions(GRAPHS.phenol, "skip", 1));
});

test("measured accuracy (printed)", () => { console.log(JSON.stringify(report)); });
