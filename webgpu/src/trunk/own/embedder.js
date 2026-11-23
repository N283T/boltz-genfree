// Boltz-2 InputEmbedder + initial / recycling embeddings in WGSL (f32), written from Boltz's PyTorch reference
// (src/boltz/model/modules/{encodersv2,transformersv2,trunkv2}.py and Boltz2.forward); nothing is imported from
// LocalFold. Structure-free route only: no templates, no contact / pocket conditioning (contact encoding is the learned
// "unspecified" vector), no cyclic positional encoding, recycling steps are driven by the caller.
//
//   const emb = new Embedder(ctx, ownWeights);
//   const sIn = emb.sInputs(features);                  // [N,384]   (affinity: {affinity: true})
//   const { sInit, zInit } = emb.initial(features, sIn); // [N,384], [N,N,128]
//   const { s, z } = emb.recycle(sInit, zInit, prevS, prevZ); // prev = null -> zeros (pass 0)
//
// `features` is any accessor with get(name)/shape(name) over the oracle's `feat/*` names: a TensorBundle of an oracle
// dump or `featureBundle(featuriseOwn(...))`. Atom windows are the oracle's: 32 queries x 128 keys, key slot k of query
// block b is atom 32b - 48 + k, slots outside the (padded) atom list are zero with mask 0 - identical to `windows`
// from featurise.js, but derived here from the padded atom count so the embedder also runs on raw oracle features.
//
// Returned buffers are persistent (the caller destroys them); everything else is scratch freed by ctx.release().
// Every returned buffer carries `.tokens` (N). Pass `taps` (a Map) to collect named intermediates as {buffer, floats}.

import { GROUP_WIDTH, LAYERNORM, MATMUL, ADD, PAIR_ADD_ROWCOL } from "../../gpu/wgsl.js";

export const ATOM_QUERIES = 32;
export const ATOM_KEYS = 128;
/** Every non-layer tensor group the embedder, initial embeddings and recycling read. */
const EMBEDDER_PREFIXES = ["input_embedder/", "s_init/", "z_init_1/", "z_init_2/", "rel_pos/", "token_bonds/", "token_bonds_type/",
  "contact_conditioning/", "s_norm/", "z_norm/", "s_recycle/", "z_recycle/"];
const ATOM_S = 128;
const ATOM_Z = 16;
const HEADS = 4;
const DEPTH = 3;
const TOKEN_S = 384;
const TOKEN_Z = 128;
const REL_FEATURES = 139;
const AE = "input_embedder/atom_attention_encoder/atom_encoder/diffusion_transformer/layers";

const HEAD = `
struct P { a: u32, b: u32, c: u32, d: u32, e: u32, f: u32, g: u32, h: u32 }
const GW: u32 = ${GROUP_WIDTH}u;
`;
const GID = "(wid.y * GW + wid.x) * 64u + li";

const RELU = `${HEAD}
@group(0) @binding(0) var<storage, read> X: array<f32>;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<uniform> p: P; // a = count
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = ${GID};
  if (t >= p.a) { return; }
  O[t] = max(X[t], 0.0);
}`;

/** p[blk, w, h, ch] before p_mlp: ref_pos / distance / mask embeddings gated by the window validity v, plus c_to_p q and k. */
const ATOM_PAIR_INIT = `${HEAD}
@group(0) @binding(0) var<storage, read> At: array<f32>; // [P,5]: pos xyz, uid, mask
@group(0) @binding(1) var<storage, read> CQ: array<f32>; // [P,16] = c_to_p_trans_q(relu(c))
@group(0) @binding(2) var<storage, read> CK: array<f32>; // [P,16]
@group(0) @binding(3) var<storage, read> W: array<f32>;  // [16,5]: pos xyz, dist, mask
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@group(0) @binding(5) var<uniform> p: P; // a = atoms (padded), b = rows * 16
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = ${GID};
  if (t >= p.b) { return; }
  let ch = t % 16u;
  let r = t / 16u;
  let h = r % 128u;
  let w = (r / 128u) % 32u;
  let blk = r / 4096u;
  let qi = blk * 32u + w;
  let kj = i32(blk * 32u) - 48 + i32(h);
  let valid = kj >= 0 && kj < i32(p.a);
  var v = 0.0;
  var d = vec3<f32>(0.0);
  var ck = 0.0;
  let qpos = vec3<f32>(At[qi * 5u], At[qi * 5u + 1u], At[qi * 5u + 2u]);
  if (valid) {
    let k = u32(kj);
    let kpos = vec3<f32>(At[k * 5u], At[k * 5u + 1u], At[k * 5u + 2u]);
    d = kpos - qpos;
    if (At[qi * 5u + 4u] > 0.5 && At[k * 5u + 4u] > 0.5 && At[qi * 5u + 3u] == At[k * 5u + 3u]) { v = 1.0; }
    ck = CK[k * 16u + ch];
  } else {
    d = -qpos;
  }
  let dn = 1.0 / (1.0 + dot(d, d));
  var val = (W[ch * 5u] * d.x + W[ch * 5u + 1u] * d.y + W[ch * 5u + 2u] * d.z) * v;
  val += W[ch * 5u + 3u] * dn * v;
  val += W[ch * 5u + 4u] * v * v;
  O[t] = val + CQ[qi * 16u + ch] + ck;
}`;

/** b = sigmoid(scale) * normalised a + bias (AdaLN, after the three projections). */
const ADALN_COMBINE = `${HEAD}
@group(0) @binding(0) var<storage, read> An: array<f32>;
@group(0) @binding(1) var<storage, read> Sig: array<f32>;
@group(0) @binding(2) var<storage, read> Sb: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
@group(0) @binding(4) var<uniform> p: P; // a = count
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = ${GID};
  if (t >= p.a) { return; }
  O[t] = Sig[t] * An[t] + Sb[t];
}`;

/**
 * Windowed attention of the atom transformer, one thread per (atom, head), online softmax over the 128 key slots.
 * logits = q.k / sqrt(32) + bias[layer*4 + head] + (1 - keyMask) * -1e6; out-of-range key slots have k = v = 0 and mask 0.
 * Output is multiplied by sigmoid(G).
 */
const ATOM_ATTENTION = `${HEAD}
@group(0) @binding(0) var<storage, read> Q: array<f32>;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read> G: array<f32>;
@group(0) @binding(4) var<storage, read> Bias: array<f32>; // [rows, 12]
@group(0) @binding(5) var<storage, read> M: array<f32>;    // atom mask [P]
@group(0) @binding(6) var<storage, read_write> O: array<f32>;
@group(0) @binding(7) var<uniform> p: P; // a = atoms (padded), b = layer
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = ${GID};
  if (t >= p.a * ${HEADS}u) { return; }
  let h = t % ${HEADS}u;
  let qi = t / ${HEADS}u;
  let blk = qi / 32u;
  let w = qi % 32u;
  let scale = 1.0 / sqrt(32.0);
  var q: array<f32, 32>;
  for (var d = 0u; d < 32u; d++) { q[d] = Q[qi * 128u + h * 32u + d]; }
  var acc: array<f32, 32>;
  for (var d = 0u; d < 32u; d++) { acc[d] = 0.0; }
  var m = -3.0e38;
  var denom = 0.0;
  for (var kk = 0u; kk < 128u; kk++) {
    let kj = i32(blk * 32u) - 48 + i32(kk);
    let valid = kj >= 0 && kj < i32(p.a);
    var dotp = 0.0;
    var mk = 0.0;
    if (valid) {
      let base = u32(kj) * 128u + h * 32u;
      for (var d = 0u; d < 32u; d++) { dotp += q[d] * K[base + d]; }
      mk = M[u32(kj)];
    }
    let row = (blk * 32u + w) * 128u + kk;
    var l = dotp * scale + Bias[row * 12u + p.b * ${HEADS}u + h];
    l = l + (1.0 - mk) * -1.0e6;
    let mn = max(m, l);
    let corr = exp(m - mn);
    let e = exp(l - mn);
    denom = denom * corr + e;
    if (valid) {
      let base = u32(kj) * 128u + h * 32u;
      for (var d = 0u; d < 32u; d++) { acc[d] = acc[d] * corr + e * V[base + d]; }
    } else {
      for (var d = 0u; d < 32u; d++) { acc[d] = acc[d] * corr; }
    }
    m = mn;
  }
  for (var d = 0u; d < 32u; d++) {
    let g = 1.0 / (1.0 + exp(-G[qi * 128u + h * 32u + d]));
    O[qi * 128u + h * 32u + d] = acc[d] / denom * g;
  }
}`;

/** h = silu(gates) * x * ab with Sw = [x | gates] (width 2c) and Ab (width c). */
const SWIGLU_MUL = `${HEAD}
@group(0) @binding(0) var<storage, read> Sw: array<f32>;
@group(0) @binding(1) var<storage, read> Ab: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<uniform> p: P; // a = rows * c, b = c
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = ${GID};
  if (t >= p.a) { return; }
  let row = t / p.b;
  let j = t % p.b;
  let x = Sw[row * 2u * p.b + j];
  let g = Sw[row * 2u * p.b + p.b + j];
  O[t] = (g / (1.0 + exp(-g))) * x * Ab[t];
}`;

/** a[t, :] = sum over atoms of token t of QA[atom, :] / (count_t + 1e-6)  (atom_to_token_mean^T @ q_to_a). */
const ATOM_POOL = `${HEAD}
@group(0) @binding(0) var<storage, read> QA: array<f32>;
@group(0) @binding(1) var<storage, read> Tok: array<i32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<uniform> p: P; // a = tokens, b = atoms (padded), c = width
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = ${GID};
  if (t >= p.a * p.c) { return; }
  let tok = t / p.c;
  let d = t % p.c;
  var acc = 0.0;
  var count = 0.0;
  for (var i = 0u; i < p.b; i++) {
    if (u32(Tok[i]) == tok && Tok[i] >= 0) { acc += QA[i * p.c + d]; count += 1.0; }
  }
  O[t] = acc / (count + 1.0e-6);
}`;

/** z[(i,j), :] += rel_pos linear over the one-hot concat [pos 66 | token 66 | same entity 1 | chain 6]. */
const REL_POS = `${HEAD}
@group(0) @binding(0) var<storage, read> Tv: array<f32>;  // [N,8]: asym, residue, entity, token, sym
@group(0) @binding(1) var<storage, read> Wr: array<f32>;  // [128,139]
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<uniform> p: P; // a = n, b = fix_sym_check
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let n = p.a;
  let t = ${GID};
  if (t >= n * n * ${TOKEN_Z}u) { return; }
  let d = t % ${TOKEN_Z}u;
  let pair = t / ${TOKEN_Z}u;
  let j = pair % n;
  let i = pair / n;
  let sameChain = Tv[i * 8u] == Tv[j * 8u];
  let sameRes = Tv[i * 8u + 1u] == Tv[j * 8u + 1u];
  let sameEnt = Tv[i * 8u + 2u] == Tv[j * 8u + 2u];
  var dres = u32(clamp(Tv[i * 8u + 1u] - Tv[j * 8u + 1u] + 32.0, 0.0, 64.0));
  if (!sameChain) { dres = 65u; }
  var dtok = u32(clamp(Tv[i * 8u + 3u] - Tv[j * 8u + 3u] + 32.0, 0.0, 64.0));
  if (!(sameChain && sameRes)) { dtok = 65u; }
  var dch = u32(clamp(Tv[i * 8u + 4u] - Tv[j * 8u + 4u] + 2.0, 0.0, 4.0));
  let useUnrelated = select(sameChain, !sameEnt, p.b == 1u);
  if (useUnrelated) { dch = 5u; }
  let base = d * ${REL_FEATURES}u;
  var acc = Wr[base + dres] + Wr[base + 66u + dtok] + Wr[base + 133u + dch];
  if (sameEnt) { acc += Wr[base + 132u]; }
  O[t] += acc;
}`;

/** z[(i,j), :] += bonds[i,j] * w[:] (token_bonds, Linear(1, 128, no bias)). */
const BOND_LINEAR = `${HEAD}
@group(0) @binding(0) var<storage, read> Bd: array<f32>;
@group(0) @binding(1) var<storage, read> Wb: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<uniform> p: P; // a = n * n
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = ${GID};
  if (t >= p.a * ${TOKEN_Z}u) { return; }
  O[t] += Bd[t / ${TOKEN_Z}u] * Wb[t % ${TOKEN_Z}u];
}`;

/** z[(i,j), :] += table[type[i,j], :] (token_bonds_type embedding). */
const BOND_TYPE = `${HEAD}
@group(0) @binding(0) var<storage, read> Ty: array<f32>;
@group(0) @binding(1) var<storage, read> Table: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@group(0) @binding(3) var<uniform> p: P; // a = n * n
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = ${GID};
  if (t >= p.a * ${TOKEN_Z}u) { return; }
  O[t] += Table[u32(Ty[t / ${TOKEN_Z}u]) * ${TOKEN_Z}u + t % ${TOKEN_Z}u];
}`;

/** z[(i,j), :] += v[:] (the unspecified-contact encoding is the same for every pair). */
const BROADCAST_ADD = `${HEAD}
@group(0) @binding(0) var<storage, read> Vec: array<f32>;
@group(0) @binding(1) var<storage, read_write> O: array<f32>;
@group(0) @binding(2) var<uniform> p: P; // a = rows, b = c
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32) {
  let t = ${GID};
  if (t >= p.a * p.b) { return; }
  O[t] += Vec[t % p.b];
}`;

export class Embedder {
  /** @param {import("../../gpu/context.js").GpuContext} ctx @param {import("../own-weights.js").OwnWeights} weights */
  constructor(ctx, weights, { fixSymCheck = true } = {}) {
    this.ctx = ctx;
    this.weights = weights;
    this.fixSymCheck = fixSymCheck;
    this.cache = new Map();
    this.ones = ctx.upload("emb.ones", new Float32Array(TOKEN_S).fill(1), true);
    this.zeros = ctx.upload("emb.zeros", new Float32Array(TOKEN_S), true);
    this.dummy = ctx.buffer("emb.dummy", 4, true);
  }

  /** Lazy URL packs: Range-fetch the embedder's tensors (about 6 MB; no-op for Node packs). Call before sInputs/initial/recycle. */
  async prepare() { for (const prefix of EMBEDDER_PREFIXES) await this.weights.ensure(prefix); }

  /** Drop the lazily fetched JS copies (uploaded GPU buffers stay cached). */
  releaseWeights() { for (const prefix of EMBEDDER_PREFIXES) this.weights.release(prefix); }

  /** Weight buffer by tensor name, uploaded once. */
  w(name) {
    let buffer = this.cache.get(name);
    if (buffer === undefined) {
      buffer = this.ctx.upload(name, this.weights.get(name), true);
      this.cache.set(name, buffer);
    }
    return buffer;
  }

  destroy() {
    for (const buffer of this.cache.values()) buffer.destroy();
    this.cache.clear();
    this.ones.destroy(); this.zeros.destroy(); this.dummy.destroy();
  }

  // ---- op helpers -------------------------------------------------------------
  layerNorm(a, weight, bias, out, rows, c) {
    this.ctx.dispatch("ln", LAYERNORM, [a, this.dummy, weight, bias, out], [rows, c, 0],
      Math.min(rows, GROUP_WIDTH), Math.ceil(rows / GROUP_WIDTH));
  }

  /** out = epilogue(a . W^T + bias); epilogues as in wgsl.js MATMUL. */
  mm(a, weightName, out, rows, inDim, outDim, epilogue = 0, aux = null, biasName = null) {
    this.ctx.dispatch("matmul", MATMUL,
      [a, this.w(weightName), biasName ? this.w(biasName) : this.dummy, aux ?? this.dummy, out],
      [rows, inDim, outDim, epilogue, biasName ? 1 : 0], Math.ceil(outDim / 16), Math.ceil(rows / 16));
  }

  elementwise(key, code, bindings, params, count) { this.ctx.dispatch1d(key, code, bindings, params, count); }

  scratch(label, floats) { return this.ctx.buffer(label, floats); }

  /** A scratch copy of `src` (zero buffer + ADD). */
  copy(src, floats, label = "emb.copy") {
    const out = this.scratch(label, floats);
    this.elementwise("add", ADD, [out, src], [floats], floats);
    return out;
  }

  tap(taps, name, buffer, floats) { if (taps) taps.set(name, { buffer, floats }); }

  // ---- feature access -----------------------------------------------------------
  #atomsOf(features) {
    const [, atoms] = features.shape("feat/ref_pos");
    if (atoms % ATOM_QUERIES !== 0) throw new Error(`padded atom count ${atoms} is not a multiple of ${ATOM_QUERIES}`);
    return atoms;
  }

  /** Atom-side inputs shared by the embedder: padded atom count, mask, packed [pos, uid, mask], token index per atom. */
  #atomInputs(features) {
    const P = this.#atomsOf(features);
    const T = features.shape("feat/token_pad_mask")[1];
    const mask = features.get("feat/atom_pad_mask");
    const pos = features.get("feat/ref_pos");
    const uid = features.get("feat/ref_space_uid");
    const packed = new Float32Array(P * 5);
    const tokenOfAtom = new Int32Array(P).fill(-1);
    const a2t = features.get("feat/atom_to_token");
    for (let i = 0; i < P; i += 1) {
      packed.set(pos.subarray(i * 3, i * 3 + 3), i * 5);
      packed[i * 5 + 3] = uid[i];
      packed[i * 5 + 4] = mask[i];
      let ones = 0;
      for (let t = 0; t < T; t += 1) {
        const x = a2t[i * T + t];
        if (x === 1) { tokenOfAtom[i] = t; ones += 1; } else if (x !== 0) throw new Error("atom_to_token is not a 0/1 matrix");
      }
      if (ones > 1) throw new Error("atom_to_token row with several tokens");
    }
    // Atom features: ref_pos (3), ref_charge (1), ref_element (128), ref_atom_name_chars (4*64).
    const element = features.get("feat/ref_element");
    const chars = features.get("feat/ref_atom_name_chars");
    const charge = features.get("feat/ref_charge");
    const width = 3 + 1 + 128 + 256;
    const atomFeats = new Float32Array(P * width);
    for (let i = 0; i < P; i += 1) {
      const row = i * width;
      atomFeats.set(pos.subarray(i * 3, i * 3 + 3), row);
      atomFeats[row + 3] = charge[i];
      atomFeats.set(element.subarray(i * 128, i * 128 + 128), row + 4);
      atomFeats.set(chars.subarray(i * 256, i * 256 + 256), row + 132);
    }
    return { P, T, mask, packed, tokenOfAtom, atomFeats, width };
  }

  // ---- atom encoder ---------------------------------------------------------------
  /**
   * AtomEncoder + AtomAttentionEncoder: returns the token-level `a` [N,384] (atom -> token pooling of the relu'd
   * atom transformer output). Taps: atom_enc_q_in, atom_enc_c, atom_enc_p, atom_enc_bias, atom_enc_q, atom_enc_a.
   */
  atomEncoder(features, taps = null) {
    const { P, T, mask, packed, tokenOfAtom, atomFeats, width } = this.#atomInputs(features);
    const ctx = this.ctx;
    const rows = P * ATOM_KEYS; // = (P / 32) * 32 * 128
    const E = "input_embedder/atom_encoder";
    const atomBuf = ctx.upload("emb.atom", packed);
    const maskBuf = ctx.upload("emb.atommask", mask.subarray(0, P));
    const featBuf = ctx.upload("emb.atomfeats", atomFeats);

    const c = this.scratch("emb.c", P * ATOM_S);
    this.mm(featBuf, `${E}/embed_atom_features/weight`, c, P, width, ATOM_S, 0, null, `${E}/embed_atom_features/bias`);
    this.tap(taps, "atom_enc_q_in", c, P * ATOM_S);
    this.tap(taps, "atom_enc_c", c, P * ATOM_S);

    const rc = this.scratch("emb.rc", P * ATOM_S);
    this.elementwise("relu", RELU, [c, rc], [P * ATOM_S], P * ATOM_S);
    const cq = this.scratch("emb.cq", P * ATOM_Z);
    const ck = this.scratch("emb.ck", P * ATOM_Z);
    this.mm(rc, `${E}/c_to_p_trans_q/1/weight`, cq, P, ATOM_S, ATOM_Z);
    this.mm(rc, `${E}/c_to_p_trans_k/1/weight`, ck, P, ATOM_S, ATOM_Z);

    const wp = new Float32Array(ATOM_Z * 5);
    const wpos = this.weights.get(`${E}/embed_atompair_ref_pos/weight`);
    const wdist = this.weights.get(`${E}/embed_atompair_ref_dist/weight`);
    const wmask = this.weights.get(`${E}/embed_atompair_mask/weight`);
    for (let ch = 0; ch < ATOM_Z; ch += 1) {
      wp.set(wpos.subarray(ch * 3, ch * 3 + 3), ch * 5);
      wp[ch * 5 + 3] = wdist[ch];
      wp[ch * 5 + 4] = wmask[ch];
    }
    const wpBuf = ctx.upload("emb.pairw", wp);
    const p = this.scratch("emb.p", rows * ATOM_Z);
    this.elementwise("atom_pair_init", ATOM_PAIR_INIT, [atomBuf, cq, ck, wpBuf, p], [P, rows * ATOM_Z], rows * ATOM_Z);

    // p += p_mlp(p): relu, L, relu, L, relu, L
    const tA = this.scratch("emb.pa", rows * ATOM_Z);
    const tB = this.scratch("emb.pb", rows * ATOM_Z);
    const n16 = rows * ATOM_Z;
    this.elementwise("relu", RELU, [p, tA], [n16], n16);
    this.mm(tA, `${E}/p_mlp/1/weight`, tB, rows, ATOM_Z, ATOM_Z, 1);
    this.mm(tB, `${E}/p_mlp/3/weight`, tA, rows, ATOM_Z, ATOM_Z, 1);
    this.mm(tA, `${E}/p_mlp/5/weight`, p, rows, ATOM_Z, ATOM_Z, 3);
    this.tap(taps, "atom_enc_p", p, n16);

    // per-layer pair bias: LayerNorm(16) then Linear(16 -> depth * heads)
    const pn = tA;
    this.layerNorm(p, this.w("input_embedder/atom_enc_proj_z/0/weight"), this.w("input_embedder/atom_enc_proj_z/0/bias"), pn, rows, ATOM_Z);
    const bias = this.scratch("emb.bias", rows * DEPTH * HEADS);
    this.mm(pn, "input_embedder/atom_enc_proj_z/1/weight", bias, rows, ATOM_Z, DEPTH * HEADS);
    this.tap(taps, "atom_enc_bias", bias, rows * DEPTH * HEADS);

    // 3-layer windowed atom transformer
    const a = this.copy(c, P * ATOM_S, "emb.a");
    const an = this.scratch("emb.an", P * ATOM_S);
    const sn = this.scratch("emb.sn", P * ATOM_S);
    const sig = this.scratch("emb.sig", P * ATOM_S);
    const sb = this.scratch("emb.sb", P * ATOM_S);
    const b = this.scratch("emb.b", P * ATOM_S);
    const qb = this.scratch("emb.q", P * ATOM_S);
    const kb = this.scratch("emb.k", P * ATOM_S);
    const vb = this.scratch("emb.v", P * ATOM_S);
    const gb = this.scratch("emb.g", P * ATOM_S);
    const ob = this.scratch("emb.o", P * ATOM_S);
    const gate = this.scratch("emb.gate", P * ATOM_S);
    const sw = this.scratch("emb.sw", P * 4 * ATOM_S);
    const ab = this.scratch("emb.ab", P * 2 * ATOM_S);
    const hb = this.scratch("emb.h", P * 2 * ATOM_S);
    const count = P * ATOM_S;
    const adaln = (prefix, source) => {
      this.layerNorm(source, this.ones, this.zeros, an, P, ATOM_S);
      this.layerNorm(c, this.w(`${prefix}/s_norm/weight`), this.zeros, sn, P, ATOM_S);
      this.mm(sn, `${prefix}/s_scale/weight`, sig, P, ATOM_S, ATOM_S, 2, null, `${prefix}/s_scale/bias`);
      this.mm(sn, `${prefix}/s_bias/weight`, sb, P, ATOM_S, ATOM_S);
      this.elementwise("adaln_combine", ADALN_COMBINE, [an, sig, sb, b], [count], count);
    };
    for (let layer = 0; layer < DEPTH; layer += 1) {
      const L = `${AE}/${layer}`;
      adaln(`${L}/adaln`, a);
      this.mm(b, `${L}/pair_bias_attn/proj_q/weight`, qb, P, ATOM_S, ATOM_S, 0, null, `${L}/pair_bias_attn/proj_q/bias`);
      this.mm(b, `${L}/pair_bias_attn/proj_k/weight`, kb, P, ATOM_S, ATOM_S);
      this.mm(b, `${L}/pair_bias_attn/proj_v/weight`, vb, P, ATOM_S, ATOM_S);
      this.mm(b, `${L}/pair_bias_attn/proj_g/weight`, gb, P, ATOM_S, ATOM_S);
      this.elementwise("atom_attention", ATOM_ATTENTION, [qb, kb, vb, gb, bias, maskBuf, ob], [P, layer], P * HEADS);
      this.mm(c, `${L}/output_projection/0/weight`, gate, P, ATOM_S, ATOM_S, 2, null, `${L}/output_projection/0/bias`);
      this.mm(ob, `${L}/pair_bias_attn/proj_o/weight`, a, P, ATOM_S, ATOM_S, 5, gate);
      // conditioned transition
      adaln(`${L}/transition/adaln`, a);
      this.mm(b, `${L}/transition/swish_gate/0/weight`, sw, P, ATOM_S, 4 * ATOM_S);
      this.mm(b, `${L}/transition/a_to_b/weight`, ab, P, ATOM_S, 2 * ATOM_S);
      this.elementwise("swiglu_mul", SWIGLU_MUL, [sw, ab, hb], [P * 2 * ATOM_S, 2 * ATOM_S], P * 2 * ATOM_S);
      this.mm(c, `${L}/transition/output_projection/0/weight`, gate, P, ATOM_S, ATOM_S, 2, null, `${L}/transition/output_projection/0/bias`);
      this.mm(hb, `${L}/transition/b_to_a/weight`, a, P, 2 * ATOM_S, ATOM_S, 5, gate);
      ctx.flush();
    }
    this.tap(taps, "atom_enc_q", a, count);

    // atom -> token: relu(Linear(128 -> 384)) then mean over the atoms of each token
    const qa = this.scratch("emb.qa", P * TOKEN_S);
    this.mm(a, "input_embedder/atom_attention_encoder/atom_to_token_trans/0/weight", qa, P, ATOM_S, TOKEN_S, 1);
    const tokBuf = ctx.upload("emb.tok", tokenOfAtom);
    const pooled = this.scratch("emb.pooled", T * TOKEN_S);
    this.elementwise("atom_pool", ATOM_POOL, [qa, tokBuf, pooled], [T, P, TOKEN_S], T * TOKEN_S);
    this.tap(taps, "atom_enc_a", pooled, T * TOKEN_S);
    return pooled;
  }

  // ---- (a) s_inputs ------------------------------------------------------------------
  /** InputEmbedder.forward: s_inputs [N,384]. `affinity` selects profile_affinity / deletion_mean_affinity. */
  sInputs(features, { affinity = false, taps = null } = {}) {
    const ctx = this.ctx;
    const T = features.shape("feat/token_pad_mask")[1];
    const a = this.atomEncoder(features, taps);
    const resType = features.get("feat/res_type");
    const profile = features.get(affinity ? "feat/profile_affinity" : "feat/profile");
    const deletion = features.get(affinity ? "feat/deletion_mean_affinity" : "feat/deletion_mean");
    const prof = new Float32Array(T * 34);
    for (let t = 0; t < T; t += 1) {
      prof.set(profile.subarray(t * 33, t * 33 + 33), t * 34);
      prof[t * 34 + 33] = deletion[t];
    }
    const s = ctx.buffer("emb.s_inputs", T * TOKEN_S, true);
    s.tokens = T;
    this.mm(ctx.upload("emb.restype", resType), "input_embedder/res_type_encoding/weight", s, T, 33, TOKEN_S, 3);
    this.mm(ctx.upload("emb.profile", prof), "input_embedder/msa_profile_encoding/weight", s, T, 34, TOKEN_S, 3);

    // method / modified / cyclic / mol_type conditioning (small per-token lookups, done on the CPU)
    const method = features.get("feat/method_feature");
    const modified = features.get("feat/modified");
    const cyclic = features.get("feat/cyclic_period");
    const molType = features.get("feat/mol_type");
    const wm = this.weights.get("input_embedder/method_conditioning_init/weight");
    const wd = this.weights.get("input_embedder/modified_conditioning_init/weight");
    const wc = this.weights.get("input_embedder/cyclic_conditioning_init/weight");
    const wt = this.weights.get("input_embedder/mol_type_conditioning_init/weight");
    const cond = new Float32Array(T * TOKEN_S);
    for (let t = 0; t < T; t += 1) {
      const cy = Math.min(cyclic[t], 1);
      for (let d = 0; d < TOKEN_S; d += 1) {
        cond[t * TOKEN_S + d] = wm[method[t] * TOKEN_S + d] + wd[modified[t] * TOKEN_S + d] + wc[d] * cy + wt[molType[t] * TOKEN_S + d];
      }
    }
    this.elementwise("add", ADD, [s, ctx.upload("emb.cond", cond)], [T * TOKEN_S], T * TOKEN_S);
    // target_feat_sum (oracle tap) = everything except the atom-encoder term
    if (taps) this.tap(taps, "target_feat_sum", this.copy(s, T * TOKEN_S, "emb.tfs"), T * TOKEN_S);
    this.elementwise("add", ADD, [s, a], [T * TOKEN_S], T * TOKEN_S);
    ctx.flush();
    return s;
  }

  // ---- (b) initial embeddings --------------------------------------------------------------
  /**
   * s_init = Linear(s_inputs); z_init = z_init_1(s)[i] + z_init_2(s)[j] + rel_pos + token_bonds + token_bonds_type +
   * contact (unspecified encoding). Taps: z_init_terms/{z_init_1_2,rel_pos,token_bonds,type_bonds,contact}, z_init, s_init.
   */
  initial(features, sInputs, { taps = null } = {}) {
    const ctx = this.ctx;
    const n = sInputs.tokens ?? features.shape("feat/token_pad_mask")[1];
    const pairs = n * n;
    const floats = pairs * TOKEN_Z;
    const cc = features.get("feat/contact_conditioning");
    for (let i = 0; i < pairs; i += 1) {
      if (cc[i * 5] !== 1) throw new Error("contact conditioning other than 'unspecified' is not supported");
    }
    const sInit = ctx.buffer("emb.s_init", n * TOKEN_S, true);
    sInit.tokens = n;
    this.mm(sInputs, "s_init/weight", sInit, n, TOKEN_S, TOKEN_S);
    this.tap(taps, "s_init", sInit, n * TOKEN_S);

    const z1 = this.scratch("emb.z1", n * TOKEN_Z);
    const z2 = this.scratch("emb.z2", n * TOKEN_Z);
    this.mm(sInputs, "z_init_1/weight", z1, n, TOKEN_S, TOKEN_Z);
    this.mm(sInputs, "z_init_2/weight", z2, n, TOKEN_S, TOKEN_Z);

    const tv = new Float32Array(n * 8);
    const asym = features.get("feat/asym_id"); const res = features.get("feat/residue_index");
    const ent = features.get("feat/entity_id"); const tok = features.get("feat/token_index"); const sym = features.get("feat/sym_id");
    const cyc = features.get("feat/cyclic_period");
    for (let t = 0; t < n; t += 1) {
      if (cyc[t] > 0) throw new Error("cyclic positional encoding is not supported");
      tv.set([asym[t], res[t], ent[t], tok[t], sym[t]], t * 8);
    }
    const tvBuf = ctx.upload("emb.tokvec", tv);
    const bonds = ctx.upload("emb.bonds", features.get("feat/token_bonds"));
    const types = ctx.upload("emb.types", features.get("feat/type_bonds"));

    const terms = {
      z_init_1_2: (out) => this.elementwise("pair_add", PAIR_ADD_ROWCOL, [out, z1, z2], [n, TOKEN_Z], floats),
      rel_pos: (out) => this.elementwise("rel_pos", REL_POS, [tvBuf, this.w("rel_pos/linear_layer/weight"), out],
        [n, this.fixSymCheck ? 1 : 0], floats),
      token_bonds: (out) => this.elementwise("bond_linear", BOND_LINEAR, [bonds, this.w("token_bonds/weight"), out], [pairs], floats),
      type_bonds: (out) => this.elementwise("bond_type", BOND_TYPE, [types, this.w("token_bonds_type/weight"), out], [pairs], floats),
      contact: (out) => this.elementwise("broadcast_add", BROADCAST_ADD, [this.w("contact_conditioning/encoding_unspecified"), out],
        [pairs, TOKEN_Z], floats),
    };
    const zInit = ctx.buffer("emb.z_init", floats, true);
    zInit.tokens = n;
    for (const [name, apply] of Object.entries(terms)) {
      apply(zInit);
      if (taps) {
        const own = this.scratch(`emb.term.${name}`, floats);
        apply(own);
        this.tap(taps, `z_init_terms/${name}`, own, floats);
      }
    }
    this.tap(taps, "z_init", zInit, floats);
    ctx.flush();
    return { sInit, zInit };
  }

  // ---- (c) recycling -----------------------------------------------------------------------
  /**
   * s = s_init + s_recycle(s_norm(prevS)); z = z_init + z_recycle(z_norm(prevZ)); prev = null means zeros (pass 0, so
   * LN(0) = bias is included). These (s, z) enter the MSA module and the pairformer.
   */
  recycle(sInit, zInit, prevS = null, prevZ = null, { taps = null } = {}) {
    const ctx = this.ctx;
    const n = sInit.tokens ?? zInit.tokens;
    if (n === undefined) throw new Error("recycle: sInit / zInit must come from initial() (they carry .tokens)");
    const sFloats = n * TOKEN_S;
    const zFloats = n * n * TOKEN_Z;
    const s = ctx.buffer("emb.s", sFloats, true);
    const z = ctx.buffer("emb.z", zFloats, true);
    s.tokens = n; z.tokens = n;
    const zeroS = prevS ?? this.scratch("emb.zero_s", sFloats);
    const zeroZ = prevZ ?? this.scratch("emb.zero_z", zFloats);
    const sn = this.scratch("emb.s_norm", sFloats);
    const zn = this.scratch("emb.z_norm", zFloats);
    // The recycling terms alone are what the oracle taps as s_recycle_out / z_recycle_out.
    const sTerm = this.scratch("emb.s_term", sFloats);
    const zTerm = this.scratch("emb.z_term", zFloats);
    this.layerNorm(zeroS, this.w("s_norm/weight"), this.w("s_norm/bias"), sn, n, TOKEN_S);
    this.mm(sn, "s_recycle/weight", sTerm, n, TOKEN_S, TOKEN_S);
    this.layerNorm(zeroZ, this.w("z_norm/weight"), this.w("z_norm/bias"), zn, n * n, TOKEN_Z);
    this.mm(zn, "z_recycle/weight", zTerm, n * n, TOKEN_Z, TOKEN_Z);
    this.tap(taps, "s_recycle_out", sTerm, sFloats);
    this.tap(taps, "z_recycle_out", zTerm, zFloats);
    this.elementwise("add", ADD, [s, sInit], [sFloats], sFloats);
    this.elementwise("add", ADD, [s, sTerm], [sFloats], sFloats);
    this.elementwise("add", ADD, [z, zInit], [zFloats], zFloats);
    this.elementwise("add", ADD, [z, zTerm], [zFloats], zFloats);
    ctx.flush();
    return { s, z };
  }
}
