// Ideal geometry and distance bounds for the ligand heavy-atom graph (hydrogens implicit).
//
// Self-contained: no RDKit, no LocalFold. Lengths come from covalent radii that depend on the hybridisation
// (sp3 / sp2 / sp), with a few explicit double-bond overrides; angles come from the hybridisation of the central
// atom and, inside small rings, from the ring size. The bounds are the 1-2 / 1-3 / 1-4 distance windows of
// ETKDG-style distance geometry (the approach of LocalFold's shared/chem/conformer.js, which was the reference for
// what is modelled here, not a copy of it).
//
// Graph input: {elements: number[] (atomic numbers), bonds: {a, b, order?, aromatic?, type?}[]}. A bond is
// aromatic when `aromatic` is set or `type` is 4 (Boltz BOND_TYPE.AROMATIC) or 12 (canonical.js RDKit AROMATIC);
// otherwise its order is `order` or `type` (1, 2, 3). Stereo (chirality, E/Z) is not modelled.

// Covalent radii [sp3, sp2, sp] in angstrom.
const RADII = {
  1: [0.32, 0.32, 0.32], 5: [0.82, 0.78, 0.74], 6: [0.77, 0.74, 0.70], 7: [0.70, 0.67, 0.64], 8: [0.66, 0.62, 0.60],
  9: [0.57, 0.57, 0.57], 14: [1.17, 1.13, 1.09], 15: [1.07, 1.03, 1.00], 16: [1.04, 1.00, 0.96], 17: [0.99, 0.99, 0.99],
  34: [1.18, 1.14, 1.10], 35: [1.14, 1.14, 1.14], 53: [1.33, 1.33, 1.33],
};
const DOUBLE_OVERRIDE = new Map([["8,16", 1.45], ["8,15", 1.48], ["6,16", 1.62], ["7,8", 1.23], ["7,16", 1.55]]);
const HYB = { SP3: 0, SP2: 1, SP: 2 };

const key = (x, y) => (x < y ? `${x},${y}` : `${y},${x}`);

export function normaliseBonds(graph) {
  return graph.bonds.map((b) => {
    const aromatic = Boolean(b.aromatic) || b.type === 4 || b.type === 12;
    const order = aromatic ? 1.5 : (b.order ?? b.type ?? 1);
    return { a: b.a, b: b.b, order, aromatic };
  });
}

function adjacency(n, bonds) {
  const adj = Array.from({ length: n }, () => []);
  bonds.forEach((b, index) => { adj[b.a].push({ to: b.b, bond: index }); adj[b.b].push({ to: b.a, bond: index }); });
  return adj;
}

/** Hybridisation per atom (0 sp3, 1 sp2, 2 sp). N/O next to an unsaturated C/N count as sp2 (conjugated, planar). */
export function hybridisation(n, bonds, adj) {
  const unsaturated = new Uint8Array(n);
  const hyb = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let doubles = 0, triples = 0, aromatic = false;
    for (const { bond } of adj[i]) {
      const b = bonds[bond];
      if (b.aromatic) aromatic = true;
      else if (b.order === 2) doubles++;
      else if (b.order === 3) triples++;
    }
    unsaturated[i] = aromatic || doubles || triples ? 1 : 0;
    hyb[i] = aromatic ? HYB.SP2 : (triples || doubles >= 2) ? HYB.SP : doubles ? HYB.SP2 : HYB.SP3;
  }
  return { hyb, unsaturated };
}

/** Finish the hybridisation with the conjugation rule; sulfur/phosphorus stay as the bond pattern says. */
export function atomHybrid(graph, bonds, adj) {
  const n = graph.elements.length;
  const { hyb, unsaturated } = hybridisation(n, bonds, adj);
  const out = Uint8Array.from(hyb);
  for (let i = 0; i < n; i++) {
    const z = graph.elements[i];
    if (out[i] !== HYB.SP3) continue;
    const nb = adj[i];
    if (z === 7 && nb.length <= 3) {
      if (nb.some(({ to }) => unsaturated[to] && (graph.elements[to] === 6 || graph.elements[to] === 7))) out[i] = HYB.SP2;
    } else if (z === 8 && nb.length === 2) {
      if (nb.some(({ to }) => unsaturated[to] && (graph.elements[to] === 6 || graph.elements[to] === 7))) out[i] = HYB.SP2;
    }
  }
  return out;
}

/** Ideal bond length in angstrom. */
export function idealBondLength(za, zb, bond, hybA, hybB) {
  const radii = (z) => RADII[z] ?? RADII[6];
  if (bond.aromatic) {
    const sum = radii(za)[1] + radii(zb)[1];
    return sum - (za === 6 && zb === 6 ? 0.08 : 0.06);
  }
  if (bond.order === 2) {
    const override = DOUBLE_OVERRIDE.get(key(za, zb));
    if (override) return override;
    return radii(za)[1] + radii(zb)[1] - 0.14;
  }
  if (bond.order === 3) return radii(za)[2] + radii(zb)[2] - 0.2;
  return radii(za)[hybA] + radii(zb)[hybB];
}

/** Shortest cycle through `bond` (atom path a..b), or null; `limit` caps the ring size. */
function ringThrough(adj, bonds, bondIndex, limit) {
  const { a, b } = bonds[bondIndex];
  const parent = new Map([[a, -1]]);
  const queue = [a];
  for (let head = 0; head < queue.length; head++) {
    const u = queue[head];
    for (const { to, bond } of adj[u]) {
      if (bond === bondIndex || parent.has(to)) continue;
      parent.set(to, u);
      if (to === b) {
        const path = [];
        for (let v = b; v !== -1; v = parent.get(v)) path.push(v);
        return path.length <= limit ? path.reverse() : null;
      }
      queue.push(to);
    }
  }
  return null;
}

/** Size of the smallest ring containing the path p-j-q (j central), or 0 when there is none within `limit`. */
function ringSizeThrough(adj, j, p, q, limit = 8) {
  const dist = new Map([[p, 0]]);
  const queue = [p];
  for (let head = 0; head < queue.length; head++) {
    const u = queue[head];
    for (const { to } of adj[u]) {
      if (to === j || dist.has(to)) continue;
      const d = dist.get(u) + 1;
      if (d + 2 > limit) continue;
      if (to === q) return d + 2;
      dist.set(to, d);
      queue.push(to);
    }
  }
  return 0;
}

const SP3_RING_ANGLE = { 3: 60, 4: 90, 5: 104, 6: 111, 7: 114, 8: 116 };
const deg = Math.PI / 180;

/** Distance a-c from lengths a-b, b-c and the angle at b (degrees). */
const law = (x, y, theta) => Math.sqrt(x * x + y * y - 2 * x * y * Math.cos(theta * deg));

/** Distance 1-4 for lengths a, b, c, angles t1, t2 (radians) and torsion phi (cis = 0). */
function d14(a, b, c, t1, t2, phi) {
  const lx = b - c * Math.cos(t2) - a * Math.cos(t1);
  const ly = c * Math.sin(t2) * Math.cos(phi) - a * Math.sin(t1);
  const lz = c * Math.sin(t2) * Math.sin(phi);
  return Math.sqrt(lx * lx + ly * ly + lz * lz);
}

/**
 * Distance bounds for the graph.
 * @returns {{n, lo: Float64Array, hi: Float64Array, weight: Float64Array, topo: Int16Array, bonds, hyb,
 *            lengths: number[], angles: {i, j, k, theta}[], planar: number[][], impropers: number[][]}}
 *   n*n symmetric matrices (row-major); `weight` is the restraint stiffness per pair class.
 */
export function buildBounds(graph) {
  const n = graph.elements.length;
  const bonds = normaliseBonds(graph);
  const adj = adjacency(n, bonds);
  const hyb = atomHybrid(graph, bonds, adj);
  const lengths = bonds.map((b) => idealBondLength(graph.elements[b.a], graph.elements[b.b], b, hyb[b.a], hyb[b.b]));

  // topological distances
  const topo = new Int16Array(n * n).fill(-1);
  for (let s = 0; s < n; s++) {
    topo[s * n + s] = 0;
    const queue = [s];
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      for (const { to } of adj[u]) if (topo[s * n + to] < 0) { topo[s * n + to] = topo[s * n + u] + 1; queue.push(to); }
    }
  }
  const lo = new Float64Array(n * n);
  const hi = new Float64Array(n * n).fill(99);
  const weight = new Float64Array(n * n).fill(1);
  const set = (i, j, l, h, w) => { lo[i * n + j] = lo[j * n + i] = l; hi[i * n + j] = hi[j * n + i] = h; weight[i * n + j] = weight[j * n + i] = w; };

  const lengthOf = new Map();
  bonds.forEach((b, index) => { lengthOf.set(key(b.a, b.b), lengths[index]); set(b.a, b.b, lengths[index] - 0.01, lengths[index] + 0.01, 20); });

  // rings: shortest cycle through every ring bond, de-duplicated
  const rings = [];
  const seenRing = new Set();
  bonds.forEach((_, index) => {
    const ring = ringThrough(adj, bonds, index, 8);
    if (!ring) return;
    const id = ring.slice().sort((p, q) => p - q).join(",");
    if (!seenRing.has(id)) { seenRing.add(id); rings.push(ring); }
  });
  const planar = rings.filter((r) => (r.length === 5 || r.length === 6 || r.length === 7)
    && r.every((v) => hyb[v] === HYB.SP2 || hyb[v] === HYB.SP));

  // 1-3: angle per pair of neighbours
  const angles = [];
  const impropers = [];
  for (let j = 0; j < n; j++) {
    const nb = adj[j].map((e) => e.to);
    const planarCentre = hyb[j] === HYB.SP2;
    const pairs = [];
    for (let x = 0; x < nb.length; x++) for (let y = x + 1; y < nb.length; y++) {
      const size = hyb[j] === HYB.SP ? 0 : ringSizeThrough(adj, j, nb[x], nb[y]);
      let theta = hyb[j] === HYB.SP ? 180 : planarCentre ? 120 : 109.5;
      if (size) theta = planarCentre ? ((size - 2) * 180) / size : SP3_RING_ANGLE[size];
      pairs.push({ i: nb[x], k: nb[y], theta, ring: size > 0 });
    }
    if (planarCentre && nb.length === 3) {
      // exocyclic angles share what the ring angles leave over from 360 degrees
      const open = pairs.filter((p) => !p.ring);
      if (open.length && open.length < 3) {
        const used = pairs.filter((p) => p.ring).reduce((s, p) => s + p.theta, 0);
        for (const p of open) p.theta = (360 - used) / open.length;
      }
    }
    if (planarCentre && nb.length === 3) impropers.push([j, nb[0], nb[1], nb[2]]);
    for (const p of pairs) {
      const l1 = lengthOf.get(key(j, p.i)), l2 = lengthOf.get(key(j, p.k));
      const d = law(l1, l2, p.theta);
      const slack = p.ring ? 0.06 : 0.04;
      set(p.i, p.k, d - slack, d + slack, 8);
      angles.push({ i: p.i, j, k: p.k, theta: p.theta });
    }
  }

  // 1-4: cis / trans window, fixed cis inside planar rings
  const window14 = new Map();
  for (let j = 0; j < n; j++) for (const { to: k } of adj[j]) {
    if (k < j) continue;
    for (const { to: i } of adj[j]) for (const { to: l } of adj[k]) {
      if (i === k || l === j || i === l || topo[i * n + l] !== 3) continue;
      const a = lengthOf.get(key(i, j)), b = lengthOf.get(key(j, k)), c = lengthOf.get(key(k, l));
      const t1 = angles.find((q) => q.j === j && ((q.i === i && q.k === k) || (q.i === k && q.k === i))).theta * deg;
      const t2 = angles.find((q) => q.j === k && ((q.i === j && q.k === l) || (q.i === l && q.k === j))).theta * deg;
      const cis = d14(a, b, c, t1, t2, 0), trans = d14(a, b, c, t1, t2, Math.PI);
      const id = key(i, l);
      const prev = window14.get(id);
      const next = prev ? [Math.max(prev[0], cis), Math.min(prev[1], trans)] : [cis, trans];
      if (next[0] > next[1]) next[0] = next[1] = (next[0] + next[1]) / 2;
      window14.set(id, next);
    }
  }
  for (const [id, [cis, trans]] of window14) {
    const [i, l] = id.split(",").map(Number);
    set(i, l, cis - 0.02, trans + 0.02, 3);
  }
  for (const ring of planar) {
    const r = ring.length;
    // every four consecutive ring atoms coplanar (volume restraint, see conformer.js)
    for (let p = 0; p < r; p++) impropers.push([ring[p], ring[(p + 1) % r], ring[(p + 2) % r], ring[(p + 3) % r]]);
    if (r <= 6) {
      // planar five/six rings: tight 1-3 windows keep the ring from puckering
      for (let p = 0; p < r; p++) {
        const i = ring[p], j = ring[(p + 1) % r], k = ring[(p + 2) % r];
        const q = angles.find((e) => e.j === j && ((e.i === i && e.k === k) || (e.i === k && e.k === i)));
        const d = law(lengthOf.get(key(i, j)), lengthOf.get(key(j, k)), q.theta);
        set(i, k, d - 0.02, d + 0.02, 8);
      }
    }
    for (let p = 0; p < r; p++) {
      const i = ring[p], j = ring[(p + 1) % r], k = ring[(p + 2) % r], l = ring[(p + 3) % r];
      if (topo[i * n + l] !== 3) continue;
      const a = lengthOf.get(key(i, j)), b = lengthOf.get(key(j, k)), c = lengthOf.get(key(k, l));
      const t1 = angles.find((q) => q.j === j && ((q.i === i && q.k === k) || (q.i === k && q.k === i))).theta * deg;
      const t2 = angles.find((q) => q.j === k && ((q.i === j && q.k === l) || (q.i === l && q.k === j))).theta * deg;
      const cis = d14(a, b, c, t1, t2, 0);
      set(i, l, cis - 0.015, cis + 0.015, 8);
    }
  }

  // everything farther away: a 2.6 A floor (peri contacts of fused rings are ~2.5 A), then triangle smoothing
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (i !== j && topo[i * n + j] !== 1 && topo[i * n + j] !== 2 && topo[i * n + j] !== 3) {
      lo[i * n + j] = topo[i * n + j] < 0 ? 3.0 : 2.6;
    }
  }
  smooth(n, lo, hi);
  return { n, lo, hi, weight, topo, bonds, hyb, lengths, angles, planar, impropers };
}

/** Floyd-Warshall triangle-inequality smoothing of the bounds. */
export function smooth(n, lo, hi) {
  for (let k = 0; k < n; k++) for (let i = 0; i < n; i++) {
    if (i === k) continue;
    for (let j = i + 1; j < n; j++) {
      if (j === k) continue;
      const up = hi[i * n + k] + hi[k * n + j];
      if (hi[i * n + j] > up) hi[i * n + j] = hi[j * n + i] = up;
      const low = Math.max(lo[i * n + k] - hi[k * n + j], lo[j * n + k] - hi[k * n + i]);
      if (lo[i * n + j] < low) lo[i * n + j] = lo[j * n + i] = low;
    }
  }
  for (let i = 0; i < n * n; i++) if (lo[i] > hi[i]) lo[i] = hi[i];
}
