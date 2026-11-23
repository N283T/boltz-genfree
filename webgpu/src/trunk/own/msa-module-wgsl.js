// WGSL kernels specific to the Boltz-2 MSAModule (plain f32). Pair-track kernels
// (LayerNorm, matmul with epilogues, triangle multiplication / attention) are reused
// from src/gpu/wgsl.js. Written from the PyTorch reference (src/boltz/model/modules/
// trunkv2.py, layers/pair_averaging.py, layers/outer_product_mean.py); LocalFold
// (sokrypton/localfold) was consulted only for the Boltz-2 dialect flags
// (msaUpdateBeforeOuterProduct, opmBiasAfterNorm, swapTransposedBias).
import { GROUP_WIDTH } from "../../gpu/wgsl.js";

const HEAD = `
struct P { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
const GW: u32 = ${GROUP_WIDTH}u;
`;

/**
 * m[s, i, :] = msa_proj([onehot(msa[s,i]), has_deletion, deletion_value, is_paired]) + s_proj(s)[i, :]
 * The one-hot product is a column lookup. p: a = S*N, b = N, c = number of tokens (33), d = input width (36).
 */
export const MSA_EMBED = `${HEAD}
@group(0) @binding(0) var<storage, read> Msa: array<f32>;
@group(0) @binding(1) var<storage, read> Paired: array<f32>;
@group(0) @binding(2) var<storage, read> HasDel: array<f32>;
@group(0) @binding(3) var<storage, read> DelVal: array<f32>;
@group(0) @binding(4) var<storage, read> Wt: array<f32>;
@group(0) @binding(5) var<storage, read> Sp: array<f32>;
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
@group(0) @binding(7) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = (wid.y * GW + wid.x) * 64u + li;
  if (t >= p.a * 64u) { return; }
  let o = t % 64u;
  let row = t / 64u;
  let i = row % p.b;
  let tok = min(u32(Msa[row]), p.c - 1u);
  let base = o * p.d;
  O[t] = Wt[base + tok] + Wt[base + p.c] * HasDel[row] + Wt[base + p.c + 1u] * DelVal[row]
    + Wt[base + p.c + 2u] * Paired[row] + Sp[i * 64u + o];
}`;

/**
 * Pair-weighted-averaging weights: W[h, i, :] = softmax_j(bias[i, j, h] + (1 - mask[i, j]) * -1e6).
 * p: a = N, b = heads.
 */
export const PWA_WEIGHTS = `${HEAD}
@group(0) @binding(0) var<storage, read> Bias: array<f32>;
@group(0) @binding(1) var<storage, read> Mask: array<f32>;
@group(0) @binding(2) var<storage, read_write> W: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
fn logit(i: u32, j: u32, h: u32) -> f32 {
  return Bias[(i * p.a + j) * p.b + h] + (1.0 - Mask[i * p.a + j]) * -1.0e6;
}
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = (wid.y * GW + wid.x) * 64u + li;
  if (t >= p.a * p.b) { return; }
  let h = t % p.b;
  let i = t / p.b;
  var m = -3.0e38;
  for (var j = 0u; j < p.a; j++) { m = max(m, logit(i, j, h)); }
  var s = 0.0;
  for (var j = 0u; j < p.a; j++) { s += exp(logit(i, j, h) - m); }
  for (var j = 0u; j < p.a; j++) { W[(h * p.a + i) * p.a + j] = exp(logit(i, j, h) - m) / s; }
}`;

/**
 * o[s, i, h, :] = g[s, i, h, :] * sum_j W[h, i, j] v[s, j, h, :], channel = h * width + d.
 * p: a = sequences in the chunk, b = N, c = width per head, d = channels (heads * width).
 */
export const PWA_APPLY = `${HEAD}
@group(0) @binding(0) var<storage, read> V: array<f32>;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = p.b;
  let t = (wid.y * GW + wid.x) * 64u + li;
  if (t >= p.a * n * p.d) { return; }
  let ch = t % p.d;
  let row = t / p.d;
  let i = row % n;
  let s = row / n;
  let h = ch / p.c;
  var acc = 0.0;
  for (var j = 0u; j < n; j++) { acc += W[(h * n + i) * n + j] * V[(s * n + j) * p.d + ch]; }
  O[t] = acc * G[t];
}`;

/** Num[i, j] = max(1, sum_s mask[s, i] * mask[s, j]). p: a = S, b = N. */
export const OPM_COUNT = `${HEAD}
@group(0) @binding(0) var<storage, read> M: array<f32>;
@group(0) @binding(1) var<storage, read_write> Num: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = p.b;
  let t = (wid.y * GW + wid.x) * 64u + li;
  if (t >= n * n) { return; }
  let i = t / n;
  let j = t % n;
  var acc = 0.0;
  for (var s = 0u; s < p.a; s++) { acc += M[s * n + i] * M[s * n + j]; }
  Num[t] = max(acc, 1.0);
}`;

/**
 * Outer-product mean for pair rows i0 .. i0+ci-1 (masks already applied to A and B):
 * O[(i - i0, j), c * H + d] = (sum_s A[s, i, c] * B[s, j, d]) / Num[i, j].
 * Tiled as the matrix product A^T B with (i, c) as rows and (j, d) as columns.
 * p: a = S, b = N, c = i0, d = ci, e = hidden width H.
 */
export const OPM_OUTER = `${HEAD}
@group(0) @binding(0) var<storage, read> A: array<f32>;
@group(0) @binding(1) var<storage, read> B: array<f32>;
@group(0) @binding(2) var<storage, read> Num: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
var<workgroup> tA: array<f32, 256>;
var<workgroup> tB: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let hw = p.e;
  let width = p.b * hw;
  let rows = p.d * hw;
  let colT = wid.x * 16u + lid.x;
  let rowT = wid.y * 16u + lid.y;
  let aRow = wid.y * 16u + lid.x; // A tile is loaded with k along y and the output row along x
  var acc = 0.0;
  for (var k0 = 0u; k0 < p.a; k0 += 16u) {
    let k = k0 + lid.y;
    var a = 0.0;
    var b = 0.0;
    if (k < p.a && aRow < rows) { a = A[k * width + p.c * hw + aRow]; }
    if (k < p.a && colT < width) { b = B[k * width + colT]; }
    tA[lid.y * 16u + lid.x] = a;
    tB[lid.y * 16u + lid.x] = b;
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) { acc += tA[kk * 16u + lid.y] * tB[kk * 16u + lid.x]; }
    workgroupBarrier();
  }
  if (rowT >= rows || colT >= width) { return; }
  let il = rowT / hw;
  let c = rowT % hw;
  let j = colT / hw;
  let d = colT % hw;
  O[(il * p.b + j) * hw * hw + c * hw + d] = acc / Num[(p.c + il) * p.b + j];
}`;
