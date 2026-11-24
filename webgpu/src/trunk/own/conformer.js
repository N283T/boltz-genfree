// Ligand reference conformers without RDKit or LocalFold: a small ETKDG-like distance-geometry embedder for the
// heavy-atom graph, plus the two Boltz/GenFree conformer modes built on it.
//
// Pipeline (embedConformer): distance bounds from chem/bounds.js (1-2 / 1-3 / 1-4 windows, fixed cis 1-4 in planar
// rings, 2.6 A floor beyond), triangle smoothing, random distances inside the bounds, metric-matrix embedding in 4D
// (Jacobi eigendecomposition), L-BFGS against the bounds with a ramped penalty on the 4th coordinate, projection to
// 3D, and a final L-BFGS that adds improper (planarity) terms at sp2 centres. Several seeded attempts; the first one
// within tolerance wins, else the best. The approach follows the one of LocalFold's shared/chem/conformer.js
// (https://github.com/sokrypton/localfold) but is a much smaller independent implementation.
//
// Not modelled: stereochemistry (chiral centres, E/Z), which gets whichever configuration the embedding finds;
// torsional preferences (only the cis/trans window of each 1-4 pair, planar rings fixed); hydrogens.

import { buildBounds } from "./chem/bounds.js";

/** Deterministic PRNG (mulberry32). */
export function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(random) {
  return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

/** Random rotation matrix from a uniform unit quaternion. */
export function randomRotation(random) {
  const q = [gaussian(random), gaussian(random), gaussian(random), gaussian(random)];
  const norm = Math.hypot(...q);
  const [w, x, y, z] = q.map((v) => v / norm);
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

/** Symmetric eigendecomposition by cyclic Jacobi: eigenvalues (descending) and eigenvectors as rows. */
function jacobiEigen(matrix, n) {
  const a = Float64Array.from(matrix);
  const v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-22) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const theta = (a[q * n + q] - a[p * n + p]) / (2 * apq);
      const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {
        const akp = a[k * n + p], akq = a[k * n + q];
        a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {
        const apk = a[p * n + k], aqk = a[q * n + k];
        a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = v[p * n + k], vkq = v[q * n + k];
        v[p * n + k] = c * vkp - s * vkq; v[q * n + k] = s * vkp + c * vkq;
      }
    }
  }
  const order = Array.from({ length: n }, (_, i) => i).sort((x, y) => a[y * n + y] - a[x * n + x]);
  return { values: order.map((i) => a[i * n + i]), vectors: order.map((i) => v.subarray(i * n, (i + 1) * n)) };
}

/** L-BFGS with Armijo backtracking. fg(x, g) fills g and returns the energy. */
function lbfgs(x, fg, maxIter, m = 8) {
  const size = x.length;
  const g = new Float64Array(size);
  let f = fg(x, g);
  const S = [], Y = [], RHO = [];
  const trial = new Float64Array(size), gNew = new Float64Array(size);
  for (let iter = 0; iter < maxIter; iter++) {
    let gnorm = 0;
    for (let i = 0; i < size; i++) gnorm += g[i] * g[i];
    if (gnorm < 1e-14) break;
    const dir = Float64Array.from(g);
    const alpha = new Array(S.length);
    for (let k = S.length - 1; k >= 0; k--) {
      let dot = 0;
      for (let i = 0; i < size; i++) dot += S[k][i] * dir[i];
      alpha[k] = RHO[k] * dot;
      for (let i = 0; i < size; i++) dir[i] -= alpha[k] * Y[k][i];
    }
    if (S.length) {
      const last = S.length - 1;
      let yy = 0, sy = 0;
      for (let i = 0; i < size; i++) { yy += Y[last][i] ** 2; sy += S[last][i] * Y[last][i]; }
      const gamma = sy / yy;
      for (let i = 0; i < size; i++) dir[i] *= gamma;
    } else {
      const scale = 0.1 / Math.sqrt(gnorm);
      for (let i = 0; i < size; i++) dir[i] *= scale;
    }
    for (let k = 0; k < S.length; k++) {
      let dot = 0;
      for (let i = 0; i < size; i++) dot += Y[k][i] * dir[i];
      const beta = RHO[k] * dot;
      for (let i = 0; i < size; i++) dir[i] += S[k][i] * (alpha[k] - beta);
    }
    let slope = 0;
    for (let i = 0; i < size; i++) { dir[i] = -dir[i]; slope += dir[i] * g[i]; }
    if (slope >= 0) { S.length = Y.length = RHO.length = 0; for (let i = 0; i < size; i++) dir[i] = -g[i] * 0.01; slope = -gnorm * 0.01; }
    let step = 1, fNew = f, ok = false;
    for (let tries = 0; tries < 30; tries++) {
      for (let i = 0; i < size; i++) trial[i] = x[i] + step * dir[i];
      fNew = fg(trial, gNew);
      if (Number.isFinite(fNew) && fNew <= f + 1e-4 * step * slope) { ok = true; break; }
      step *= 0.5;
    }
    if (!ok) { if (S.length) { S.length = Y.length = RHO.length = 0; continue; } break; }
    const s = new Float64Array(size), y = new Float64Array(size);
    let sy = 0;
    for (let i = 0; i < size; i++) { s[i] = trial[i] - x[i]; y[i] = gNew[i] - g[i]; sy += s[i] * y[i]; }
    if (sy > 1e-12) { S.push(s); Y.push(y); RHO.push(1 / sy); if (S.length > m) { S.shift(); Y.shift(); RHO.shift(); } }
    const drop = f - fNew;
    x.set(trial); g.set(gNew); f = fNew;
    if (drop < 1e-13 * (1 + Math.abs(f))) break;
  }
  return f;
}

/** Energy of flat-bottom pair restraints (+ planarity impropers in 3D, + a 4th-coordinate penalty). */
function makeEnergy(bounds, dim, wFourth) {
  const { n, lo, hi, weight, impropers } = bounds;
  return (x, g) => {
    g.fill(0);
    let e = 0;
    for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
      let d2 = 0;
      for (let k = 0; k < dim; k++) { const t = x[i * dim + k] - x[j * dim + k]; d2 += t * t; }
      const d = Math.sqrt(d2) + 1e-12;
      const l = lo[i * n + j], h = hi[i * n + j];
      let dev = 0;
      if (d > h) dev = d - h; else if (d < l) dev = d - l; else continue;
      const w = weight[i * n + j];
      e += w * dev * dev;
      const c = (2 * w * dev) / d;
      for (let k = 0; k < dim; k++) { const t = c * (x[i * dim + k] - x[j * dim + k]); g[i * dim + k] += t; g[j * dim + k] -= t; }
    }
    if (dim === 3) {
      for (const [a, b, c, d] of impropers) {
        const u = [0, 1, 2].map((k) => x[b * 3 + k] - x[a * 3 + k]);
        const v = [0, 1, 2].map((k) => x[c * 3 + k] - x[a * 3 + k]);
        const w = [0, 1, 2].map((k) => x[d * 3 + k] - x[a * 3 + k]);
        const cross = (p, q) => [p[1] * q[2] - p[2] * q[1], p[2] * q[0] - p[0] * q[2], p[0] * q[1] - p[1] * q[0]];
        const gu = cross(v, w), gv = cross(w, u), gw = cross(u, v);
        const vol = u[0] * gu[0] + u[1] * gu[1] + u[2] * gu[2];
        const wt = 20;
        e += wt * vol * vol;
        for (let k = 0; k < 3; k++) {
          g[b * 3 + k] += 2 * wt * vol * gu[k]; g[c * 3 + k] += 2 * wt * vol * gv[k]; g[d * 3 + k] += 2 * wt * vol * gw[k];
          g[a * 3 + k] -= 2 * wt * vol * (gu[k] + gv[k] + gw[k]);
        }
      }
    } else if (wFourth > 0) {
      for (let i = 0; i < n; i++) { const t = x[i * dim + dim - 1]; e += wFourth * t * t; g[i * dim + dim - 1] += 2 * wFourth * t; }
    }
    return e;
  };
}

/** Largest restraint violation (angstrom) over pairs with weight >= 3, and over bonds alone. */
function violations(bounds, x) {
  const { n, lo, hi, weight, topo } = bounds;
  let worst = 0, bond = 0;
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const d = Math.hypot(x[i * 3] - x[j * 3], x[i * 3 + 1] - x[j * 3 + 1], x[i * 3 + 2] - x[j * 3 + 2]);
    const dev = Math.max(d - hi[i * n + j], lo[i * n + j] - d, 0);
    if (topo[i * n + j] === 1) bond = Math.max(bond, dev);
    if (weight[i * n + j] >= 3 || dev > 0.3) worst = Math.max(worst, dev);
  }
  return { worst, bond };
}

function attempt(bounds, random) {
  const { n, lo, hi } = bounds;
  // random distances inside the bounds, metric matrix, 4D coordinates
  const D2 = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) {
    const l = lo[i * n + j], h = Math.min(hi[i * n + j], l + 6 + 0.6 * n);
    const d = l + random() * Math.max(h - l, 0);
    D2[i * n + j] = D2[j * n + i] = d * d;
  }
  const rowMean = new Float64Array(n);
  let all = 0;
  for (let i = 0; i < n; i++) { for (let j = 0; j < n; j++) rowMean[i] += D2[i * n + j]; all += rowMean[i]; rowMean[i] /= n; }
  all /= n * n;
  const G = new Float64Array(n * n);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) G[i * n + j] = 0.5 * (rowMean[i] + rowMean[j] - D2[i * n + j] - all);
  const { values, vectors } = jacobiEigen(G, n);
  const x4 = new Float64Array(n * 4);
  for (let k = 0; k < 4; k++) {
    const scale = Math.sqrt(Math.max(values[k] ?? 0, 0));
    for (let i = 0; i < n; i++) x4[i * 4 + k] = (vectors[k]?.[i] ?? 0) * scale + 0.05 * (random() - 0.5);
  }
  lbfgs(x4, makeEnergy(bounds, 4, 0), 300);
  lbfgs(x4, makeEnergy(bounds, 4, 0.5), 300);
  const x = new Float64Array(n * 3);
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) x[i * 3 + k] = x4[i * 4 + k];
  const energy = makeEnergy(bounds, 3, 0);
  lbfgs(x, energy, 400);
  const f = lbfgs(x, energy, 400);
  return { x, f, ...violations(bounds, x) };
}

/**
 * Embed the heavy-atom graph in 3D.
 * @param {{elements: number[], bonds: {a: number, b: number, order?: number, aromatic?: boolean, type?: number}[]}} graph
 * @param {{seed?: number, attempts?: number}} [options]
 * @returns {Float32Array} n*3 coordinates (angstrom), deterministic per seed
 */
export function embedConformer(graph, { seed = 42, attempts = 30 } = {}) {
  const n = graph.elements.length;
  if (n === 0) return new Float32Array(0);
  if (n === 1) return new Float32Array(3);
  const bounds = buildBounds(graph);
  let best = null;
  for (let t = 0; t < attempts; t++) {
    const random = rng((Math.imul(seed >>> 0, 0x9e3779b1) + Math.imul(t + 1, 0x85ebca6b)) >>> 0);
    const result = attempt(bounds, random);
    if (!best || result.worst + result.bond < best.worst + best.bond) best = result;
    if (result.bond < 0.03 && result.worst < 0.08) { best = result; break; }
  }
  return Float32Array.from(best.x);
}

/**
 * Centre, optionally flatten onto the principal plane (z = 0), then apply a seeded random rotation and a
 * Gaussian translation: Boltz's center_random_augmentation. Pure and deterministic per seed.
 * @param {ArrayLike<number>} coords n*3
 * @returns {Float32Array}
 */
export function rigidMotion(coords, seed, { flatten = false } = {}) {
  const n = coords.length / 3;
  const centre = [0, 0, 0];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) centre[k] += coords[i * 3 + k] / n;
  let work = Array.from({ length: n }, (_, i) => [0, 1, 2].map((k) => coords[i * 3 + k] - centre[k]));
  if (flatten) {
    const cov = new Float64Array(9);
    for (const p of work) for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) cov[a * 3 + b] += p[a] * p[b];
    const { vectors } = jacobiEigen(cov, 3);
    work = work.map((p) => [0, 1].map((a) => p[0] * vectors[a][0] + p[1] * vectors[a][1] + p[2] * vectors[a][2]).concat([0]));
  }
  const random = rng(seed);
  const rotation = randomRotation(random);
  const shift = [gaussian(random), gaussian(random), gaussian(random)];
  const out = new Float32Array(n * 3);
  work.forEach((p, i) => {
    for (let a = 0; a < 3; a++) out[i * 3 + a] = rotation[a][0] * p[0] + rotation[a][1] * p[1] + rotation[a][2] * p[2] + shift[a];
  });
  return out;
}

/**
 * Ligand reference conformer positions for the heavy atoms (`skip` is the caller's: ref_pos stays zero).
 *  etkdg    embedded conformer, centred, seeded random rotation and translation.
 *  rdkit_2d the same conformer flattened onto its principal plane, then rotated and translated the same way.
 * @returns {Float32Array} n*3
 */
export function ligandConformerPositions(graph, mode, seed) {
  if (mode !== "etkdg" && mode !== "rdkit_2d") throw new Error(`ligand conformer mode ${mode}: expected etkdg or rdkit_2d`);
  const conformer = embedConformer(graph, { seed });
  return rigidMotion(conformer, seed, { flatten: mode === "rdkit_2d" });
}

export { buildBounds };
