// WGSL kernels for the affinity module. Plain f32, one generic kernel per op.
// These are deliberately simple (correctness first); the LocalFold pair-track
// kernels are the optimised alternative, see webgpu/DESIGN.md.

/** Dispatch geometry for 1D work folded over x and y (see `spread` in context.js). */
export const GROUP_WIDTH = 32768;

const HEAD = `
struct P { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
const GW: u32 = ${GROUP_WIDTH}u;
`;

/** LayerNorm over rows of width c, or over the concatenation [A | B] (width 2c). eps 1e-5. */
export const LAYERNORM = `${HEAD}
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> Wt: array<f32>;
@group(0) @binding(3) var<storage, read> Bi: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@group(0) @binding(5) var<uniform> p: P; // a = rows, b = c, c = 1 when concatenating A and B
var<workgroup> sh: array<f32, 64>;
fn val(row: u32, e: u32) -> f32 {
  if (e < p.b) { return A[row * p.b + e]; }
  return B[row * p.b + (e - p.b)];
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let row = wid.y * GW + wid.x;
  if (row >= p.a) { return; }
  let width = select(p.b, 2u * p.b, p.c == 1u);
  var s = 0.0;
  for (var e = li; e < width; e += 64u) { s += val(row, e); }
  sh[li] = s;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { sh[li] += sh[li + st]; }
    workgroupBarrier();
  }
  let mean = sh[0] / f32(width);
  workgroupBarrier();
  var v = 0.0;
  for (var e = li; e < width; e += 64u) { let d = val(row, e) - mean; v += d * d; }
  sh[li] = v;
  workgroupBarrier();
  for (var st = 32u; st > 0u; st >>= 1u) {
    if (li < st) { sh[li] += sh[li + st]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sh[0] / f32(width) + 1e-5);
  for (var e = li; e < width; e += 64u) {
    O[row * width + e] = (val(row, e) - mean) * inv * Wt[e] + Bi[e];
  }
}`;

/**
 * out[r, o] = epilogue(sum_i A[r, i] * W[o, i] + bias[o]).
 * epilogue: 0 store, 1 relu, 2 sigmoid, 3 out += r, 4 out = r * aux, 5 out += r * aux, 6 silu
 */
export const MATMUL = `${HEAD}
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read> Bias: array<f32>;
@group(0) @binding(3) var<storage, read> Aux: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@group(0) @binding(5) var<uniform> p: P; // a = rows, b = inDim, c = outDim, d = epilogue, e = hasBias
var<workgroup> tA: array<f32, 256>;
var<workgroup> tW: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let col = wid.x * 16u + lid.x;
  let row = wid.y * 16u + lid.y;
  let wrow = wid.x * 16u + lid.y; // output channel loaded by this thread into the W tile
  var acc = 0.0;
  for (var k0 = 0u; k0 < p.b; k0 += 16u) {
    let k = k0 + lid.x;
    var a = 0.0;
    var w = 0.0;
    if (row < p.a && k < p.b) { a = A[row * p.b + k]; }
    if (wrow < p.c && k < p.b) { w = W[wrow * p.b + k]; }
    tA[lid.y * 16u + lid.x] = a;
    tW[lid.y * 16u + lid.x] = w;
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) { acc += tA[lid.y * 16u + kk] * tW[lid.x * 16u + kk]; }
    workgroupBarrier();
  }
  if (row >= p.a || col >= p.c) { return; }
  var r = acc;
  if (p.e == 1u) { r += Bias[col]; }
  let idx = row * p.c + col;
  switch (p.d) {
    case 1u: { O[idx] = max(r, 0.0); }
    case 2u: { O[idx] = 1.0 / (1.0 + exp(-r)); }
    case 3u: { O[idx] = O[idx] + r; }
    case 4u: { O[idx] = r * Aux[idx]; }
    case 5u: { O[idx] = O[idx] + r * Aux[idx]; }
    case 6u: { O[idx] = r / (1.0 + exp(-r)); }
    default: { O[idx] = r; }
  }
}`;

/** x[r, :] *= mask[r] over rows of width c. */
export const SCALE_ROWS = `${HEAD}
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> M: array<f32>;
@group(0) @binding(2) var<uniform> p: P; // a = rows * c, b = c
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = (wid.y * GW + wid.x) * 64u + li;
  if (t >= p.a) { return; }
  X[t] = X[t] * M[t / p.b];
}`;

/** x += y, elementwise. */
export const ADD = `${HEAD}
@group(0) @binding(0) var<storage, read_write> X: array<f32>;
@group(0) @binding(1) var<storage, read> Y: array<f32>;
@group(0) @binding(2) var<uniform> p: P; // a = count
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = (wid.y * GW + wid.x) * 64u + li;
  if (t >= p.a) { return; }
  X[t] = X[t] + Y[t];
}`;

/** z[(i, j), d] += s1[i, d] + s2[j, d] */
export const PAIR_ADD_ROWCOL = `${HEAD}
@group(0) @binding(0) var<storage, read_write> Z: array<f32>;
@group(0) @binding(1) var<storage, read> S1: array<f32>;
@group(0) @binding(2) var<storage, read> S2: array<f32>;
@group(0) @binding(3) var<uniform> p: P; // a = n, b = c
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = (wid.y * GW + wid.x) * 64u + li;
  let total = p.a * p.a * p.b;
  if (t >= total) { return; }
  let d = t % p.b;
  let pair = t / p.b;
  let j = pair % p.a;
  let i = pair / p.a;
  Z[t] = Z[t] + S1[i * p.b + d] + S2[j * p.b + d];
}`;

/** out[p, :] = table[bins[p], :] */
export const EMBED_GATHER = `${HEAD}
@group(0) @binding(0) var<storage, read> Bins: array<u32>;
@group(0) @binding(1) var<storage, read> Table: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<uniform> p: P; // a = rows, b = c
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = (wid.y * GW + wid.x) * 64u + li;
  if (t >= p.a * p.b) { return; }
  O[t] = Table[Bins[t / p.b] * p.b + (t % p.b)];
}`;

/** Triangle multiplication contraction over the packed [a | b] projections (width 2c). */
export const TRI_CONTRACT = `${HEAD}
@group(0) @binding(0) var<storage, read> Pr: array<f32>;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<uniform> p: P; // a = n, b = c, c = 1 for outgoing
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = p.a;
  let c = p.b;
  let t = (wid.y * GW + wid.x) * 64u + li;
  if (t >= n * n * c) { return; }
  let d = t % c;
  let pair = t / c;
  let j = pair % n;
  let i = pair / n;
  var acc = 0.0;
  for (var k = 0u; k < n; k++) {
    var ia: u32;
    var ib: u32;
    if (p.c == 1u) {
      ia = (i * n + k) * 2u * c + d;
      ib = (j * n + k) * 2u * c + c + d;
    } else {
      ia = (k * n + i) * 2u * c + d;
      ib = (k * n + j) * 2u * c + c + d;
    }
    acc += Pr[ia] * Pr[ib];
  }
  O[t] = acc;
}`;

/**
 * Triangle attention for one direction, one thread per (row i, query j, head h),
 * online softmax over the keys. Logit order matches the PyTorch reference:
 * (q.k + mask_bias) + triangle_bias, with q pre-scaled by 1/sqrt(width).
 * `starting` reads pair (i, j) at i*n+j; the ending node reads the transpose.
 */
export const triAttention = (heads, width) => `${HEAD}
const H: u32 = ${heads}u;
const W: u32 = ${width}u;
const HW: u32 = ${heads * width}u;
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> G: array<f32>;
@group(0) @binding(4) var<storage, read> TB: array<f32>;
@group(0) @binding(5) var<storage, read> M: array<f32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
@group(0) @binding(7) var<uniform> p: P; // a = n, b = 1 for the starting node
fn at(i: u32, j: u32) -> u32 {
  if (p.b == 1u) { return i * p.a + j; }
  return j * p.a + i;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = p.a;
  let t = (wid.y * GW + wid.x) * 64u + li;
  if (t >= n * n * H) { return; }
  let h = t % H;
  let pair = t / H;
  let j = pair % n;
  let i = pair / n;
  let pq = at(i, j);
  let scale = 1.0 / sqrt(f32(W));
  var acc: array<f32, ${width}>;
  for (var d = 0u; d < W; d++) { acc[d] = 0.0; }
  var m = -3.0e38;
  var denom = 0.0;
  for (var kk = 0u; kk < n; kk++) {
    let pk = at(i, kk);
    var dot = 0.0;
    for (var d = 0u; d < W; d++) { dot += (Q[pq * HW + h * W + d] * scale) * K[pk * HW + h * W + d]; }
    var l = dot + 1.0e9 * (M[pk] - 1.0);
    l = l + TB[at(j, kk) * H + h];
    let mn = max(m, l);
    let corr = exp(m - mn);
    let e = exp(l - mn);
    denom = denom * corr + e;
    for (var d = 0u; d < W; d++) { acc[d] = acc[d] * corr + e * V[pk * HW + h * W + d]; }
    m = mn;
  }
  for (var d = 0u; d < W; d++) {
    let g = 1.0 / (1.0 + exp(-G[pq * HW + h * W + d]));
    O[pq * HW + h * W + d] = acc[d] / denom * g;
  }
}`;

/** Partial masked sums over chunks of 64 pairs: partial[chunk, d] = sum_p z[p, d] * m[p]. */
export const POOL_PARTIAL = `${HEAD}
@group(0) @binding(0) var<storage, read> Z: array<f32>;
@group(0) @binding(1) var<storage, read> M: array<f32>;
@group(0) @binding(2) var<storage, read_write> Part: array<f32>;
@group(0) @binding(3) var<uniform> p: P; // a = rows, b = c
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let chunk = wid.x;
  let d = wid.y * 64u + li;
  if (d >= p.b) { return; }
  var acc = 0.0;
  for (var q = chunk * 64u; q < min(chunk * 64u + 64u, p.a); q++) { acc += Z[q * p.b + d] * M[q]; }
  Part[chunk * p.b + d] = acc;
}`;

/** out[d] = sum_chunk partial[chunk, d] / denominator (bitcast in p.d). */
export const POOL_REDUCE = `${HEAD}
@group(0) @binding(0) var<storage, read> Part: array<f32>;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<uniform> p: P; // a = chunks, b = c, d = bitcast<u32>(denominator)
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d = gid.x;
  if (d >= p.b) { return; }
  var acc = 0.0;
  for (var q = 0u; q < p.a; q++) { acc += Part[q * p.b + d]; }
  O[d] = acc / bitcast<f32>(p.d);
}`;
