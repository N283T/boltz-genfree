// WGSL kernels specific to the Boltz-2 trunk PairformerLayer single track.
// The pair track reuses the generic f32 kernels in src/gpu/wgsl.js.
import { GROUP_WIDTH } from "../../gpu/wgsl.js";

/**
 * AttentionPairBias core (src/boltz/model/layers/attentionv2.py): one thread per (query i, head h),
 * online softmax over all keys j. logit = q.k / sqrt(d) + pairBias[i, j, h] + (1 - mask[j]) * -1e6;
 * output is sigmoid(g) * (softmax . v), laid out [n, heads * d] ready for proj_o.
 */
export const singleAttention = (heads, width) => `
struct P { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
const GW: u32 = ${GROUP_WIDTH}u;
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
@group(0) @binding(7) var<uniform> p: P; // a = n
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = p.a;
  let t = (wid.y * GW + wid.x) * 64u + li;
  if (t >= n * H) { return; }
  let h = t % H;
  let i = t / H;
  let inv = 1.0 / sqrt(f32(W));
  var acc: array<f32, ${width}>;
  for (var d = 0u; d < W; d++) { acc[d] = 0.0; }
  var m = -3.0e38;
  var denom = 0.0;
  for (var j = 0u; j < n; j++) {
    var dot = 0.0;
    for (var d = 0u; d < W; d++) { dot += Q[i * HW + h * W + d] * K[j * HW + h * W + d]; }
    var l = dot * inv + TB[(i * n + j) * H + h];
    l = l + (1.0 - M[j]) * -1.0e6;
    let mn = max(m, l);
    let corr = exp(m - mn);
    let e = exp(l - mn);
    denom = denom * corr + e;
    for (var d = 0u; d < W; d++) { acc[d] = acc[d] * corr + e * V[j * HW + h * W + d]; }
    m = mn;
  }
  for (var d = 0u; d < W; d++) {
    let g = 1.0 / (1.0 + exp(-G[i * HW + h * W + d]));
    O[i * HW + h * W + d] = acc[d] / denom * g;
  }
}`;
