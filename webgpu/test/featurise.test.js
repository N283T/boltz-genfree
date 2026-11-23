// Pure-JS pieces of the featuriser: A3M column crop and the ligand reference conformer modes.
// No LocalFold or GPU needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { applyLigandConformer, cropA3m } from "../src/trunk/featurise.js";

test("cropA3m keeps the selected query columns and the headers", () => {
  assert.equal(cropA3m(">q\nMKTV\n>a\nM-TV", [0, 2]), ">q\nMT\n>a\nMT");
  assert.equal(cropA3m(">a\nM-T", [1, 2]), ">a\n-T");
});

test("cropA3m moves insertions with the column that follows them", () => {
  assert.equal(cropA3m(">a\nMaKT", [1]), ">a\naK");
  assert.equal(cropA3m(">a\nMaKT", [0, 2]), ">a\nMT");
});

// four non-coplanar atoms so that the 3D conformer is not flat
const POINTS = [[0, 0, 0], [1.5, 0, 0], [0, 1.5, 0], [0.5, 0.5, 1.4]];

function makeBatch() {
  const dense = 2;
  const refPos = new Float32Array(POINTS.length * dense * 3);
  POINTS.forEach((p, t) => refPos.set(p, t * dense * 3));
  return { dense, refPos };
}

const coords = (batch) => POINTS.map((_, t) => Array.from(batch.refPos.slice(t * batch.dense * 3, t * batch.dense * 3 + 3)));
const dist = (a, b) => Math.hypot(...a.map((v, k) => v - b[k]));

/** |det| of the covariance of the centred points: zero exactly when the points are coplanar. */
function covarianceDet(points) {
  const centre = [0, 1, 2].map((k) => points.reduce((s, p) => s + p[k], 0) / points.length);
  const c = points.map((p) => p.map((v, k) => v - centre[k]));
  const m = [0, 1, 2].map((a) => [0, 1, 2].map((b) => c.reduce((s, p) => s + p[a] * p[b], 0)));
  return Math.abs(
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
    - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
    + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]));
}

test("ligand conformer skip zeroes ref_pos", () => {
  const batch = makeBatch();
  applyLigandConformer(batch, 0, POINTS.length, "skip", 1);
  assert.ok(batch.refPos.every((v) => v === 0));
});

test("ligand conformer etkdg is a seeded rigid motion", () => {
  const a = makeBatch();
  const b = makeBatch();
  const c = makeBatch();
  applyLigandConformer(a, 0, POINTS.length, "etkdg", 7);
  applyLigandConformer(b, 0, POINTS.length, "etkdg", 7);
  applyLigandConformer(c, 0, POINTS.length, "etkdg", 8);
  assert.deepEqual(Array.from(a.refPos), Array.from(b.refPos));
  assert.notDeepEqual(Array.from(a.refPos), Array.from(c.refPos));
  const out = coords(a);
  for (let i = 0; i < POINTS.length; i += 1) {
    for (let j = i + 1; j < POINTS.length; j += 1) {
      assert.ok(Math.abs(dist(out[i], out[j]) - dist(POINTS[i], POINTS[j])) < 1e-4);
    }
  }
});

test("ligand conformer rdkit_2d flattens onto a plane and keeps the seed behaviour", () => {
  const flat = makeBatch();
  const full = makeBatch();
  applyLigandConformer(flat, 0, POINTS.length, "rdkit_2d", 3);
  applyLigandConformer(full, 0, POINTS.length, "etkdg", 3);
  assert.ok(covarianceDet(POINTS) > 1e-3);
  assert.ok(covarianceDet(coords(full)) > 1e-3);
  assert.ok(covarianceDet(coords(flat)) < 1e-6);
});

test("ligand conformer only touches the ligand's atom range", () => {
  const batch = makeBatch();
  applyLigandConformer(batch, 2, 2, "skip", 1);
  const before = coords(makeBatch());
  const after = coords(batch);
  assert.deepEqual(after.slice(0, 2), before.slice(0, 2));
  assert.deepEqual(after.slice(2), [[0, 0, 0], [0, 0, 0]]);
});
