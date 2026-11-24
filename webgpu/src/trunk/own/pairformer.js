// Boltz-2 trunk PairformerModule (64 x PairformerLayer) on WebGPU, plain f32.
//
// Structure (src/boltz/model/layers/pairformer.py, eval, no dropout, no chunking):
//   z += tri_mul_out(z); z += tri_mul_in(z); z += tri_att_start(z); z += tri_att_end(z);
//   z += transition_z(z)
//   s += AttentionPairBias(LN(s), z); s += transition_s(s)
// The z track does not depend on s. Pair-track kernels are the generic f32 kernels of
// src/gpu/wgsl.js (same math as the affinity PairformerNoSeq block); only the single-track
// attention kernel is new (pairformer-wgsl.js). Weights are streamed: one layer's tensors
// are uploaded, used, and destroyed before the next layer, so the whole pack is never
// resident on the GPU. The kernel design follows the AF3/Boltz pairformer structure also
// implemented in LocalFold (sokrypton/localfold), used only as a reference; no code is imported.
import { GpuContext } from "../../gpu/context.js";
import * as K from "../../gpu/wgsl.js";
import { singleAttention } from "./pairformer-wgsl.js";

const ROW_CHUNK = 16384; // rows per transition chunk

export class Pairformer {
  /** @param {GPUDevice | GpuContext} deviceOrCtx @param {import("../own-weights.js").OwnWeights} weights */
  constructor(deviceOrCtx, weights, { prefix = "pairformer_module" } = {}) {
    this.ctx = deviceOrCtx instanceof GpuContext ? deviceOrCtx : new GpuContext(deviceOrCtx);
    this.device = this.ctx.device;
    this.weights = weights;
    this.prefix = prefix;
    const arch = weights.arch ?? {};
    this.sDim = arch.tokenS ?? 384;
    this.zDim = arch.tokenZ ?? 128;
    this.heads = arch.pairformerHeads ?? 16; // single-track attention heads
    this.triHeads = 4; // pairwise_num_heads, pairwise_head_width (Boltz-2 defaults)
    this.triWidth = 32;
    this.numBlocks = arch.pairformerBlocks ?? 64;
    this.dummy = this.ctx.buffer("pf.dummy", 4, true);
  }

  // ---- weight streaming ------------------------------------------------------
  async #loadLayer(i) {
    await this.weights.ensure(`${this.prefix}/layers/${i}/`); // lazy URL packs: Range-fetch this layer only
    const slice = this.weights.slice(`${this.prefix}/layers/${i}`);
    const buffers = new Map();
    for (const [name, t] of Object.entries(slice)) {
      const buf = this.ctx.upload(`pf.l${i}.${name}`, t.data, true);
      buffers.set(name, buf);
    }
    return buffers;
  }

  // ---- op helpers ------------------------------------------------------------
  #ln(W, prefix, a, out, rows, c) {
    this.ctx.dispatch("ln", K.LAYERNORM, [a, this.dummy, W.get(`${prefix}/weight`), W.get(`${prefix}/bias`), out],
      [rows, c, 0], Math.min(rows, K.GROUP_WIDTH), Math.ceil(rows / K.GROUP_WIDTH));
  }

  /** out = epilogue(a . W^T [+ bias]); epilogues as in K.MATMUL. */
  #mm(W, a, weight, out, rows, inDim, outDim, epilogue = 0, aux = null, bias = null) {
    this.ctx.dispatch("matmul", K.MATMUL,
      [a, W.get(weight), bias ? W.get(bias) : this.dummy, aux ?? this.dummy, out],
      [rows, inDim, outDim, epilogue, bias ? 1 : 0], Math.ceil(outDim / 16), Math.ceil(rows / 16));
  }

  #slice(buffer, row, rows, width) { return { buffer, offset: row * width * 4, size: rows * width * 4 }; }

  /** x += transition(x), SwiGLU, in row chunks. */
  #transition(W, prefix, x, norm, u, h, rows, c, hidden) {
    this.#ln(W, `${prefix}/norm`, x, norm, rows, c);
    for (let r0 = 0; r0 < rows; r0 += ROW_CHUNK) {
      const n = Math.min(ROW_CHUNK, rows - r0);
      const ns = this.#slice(norm, r0, n, c);
      const xs = this.#slice(x, r0, n, c);
      this.#mm(W, ns, `${prefix}/fc1/weight`, u, n, c, hidden, 6);
      this.#mm(W, ns, `${prefix}/fc2/weight`, h, n, c, hidden, 4, u);
      this.#mm(W, h, `${prefix}/fc3/weight`, xs, n, hidden, c, 3);
      this.ctx.flush();
    }
  }

  #triMul(W, prefix, S, outgoing) {
    const { n, rows, c, pairMask } = S;
    this.#ln(W, `${prefix}/norm_in`, S.z, S.x, rows, c);
    this.#mm(W, S.x, `${prefix}/g_in/weight`, S.g2, rows, c, 2 * c, 2);
    this.#mm(W, S.x, `${prefix}/p_in/weight`, S.p2, rows, c, 2 * c, 4, S.g2);
    this.ctx.dispatch1d("scale_rows", K.SCALE_ROWS, [S.p2, pairMask], [rows * 2 * c, 2 * c], rows * 2 * c);
    this.ctx.dispatch1d("contract", K.TRI_CONTRACT, [S.p2, S.hidden], [n, c, outgoing ? 1 : 0], rows * c);
    this.#ln(W, `${prefix}/norm_out`, S.hidden, S.y, rows, c);
    this.#mm(W, S.x, `${prefix}/g_out/weight`, S.gate, rows, c, c, 2);
    this.#mm(W, S.y, `${prefix}/p_out/weight`, S.z, rows, c, c, 5, S.gate);
    this.ctx.flush();
  }

  #triAtt(W, prefix, S, starting) {
    const { n, rows, c, pairMask } = S;
    const heads = this.triHeads;
    const hw = heads * this.triWidth;
    this.#ln(W, `${prefix}/layer_norm`, S.z, S.x, rows, c);
    this.#mm(W, S.x, `${prefix}/linear/weight`, S.tbias, rows, c, heads, 0);
    this.#mm(W, S.x, `${prefix}/mha/linear_q/weight`, S.q, rows, c, hw, 0);
    this.#mm(W, S.x, `${prefix}/mha/linear_k/weight`, S.k, rows, c, hw, 0);
    this.#mm(W, S.x, `${prefix}/mha/linear_v/weight`, S.v, rows, c, hw, 0);
    this.#mm(W, S.x, `${prefix}/mha/linear_g/weight`, S.g, rows, c, hw, 0);
    this.ctx.flush();
    this.ctx.dispatch1d(`triatt:${heads}x${this.triWidth}`, K.triAttention(heads, this.triWidth),
      [S.q, S.k, S.v, S.g, S.tbias, pairMask, S.o], [n, starting ? 1 : 0], n * n * heads);
    this.ctx.flush();
    this.#mm(W, S.o, `${prefix}/mha/linear_o/weight`, S.z, rows, hw, c, 3);
    this.ctx.flush();
  }

  #singleTrack(W, S, s) {
    const { n, rows, c } = S;
    const d = this.sDim;
    const A = "attention";
    this.#ln(W, "pre_norm_s", s, S.sn, n, d);
    this.#ln(W, `${A}/proj_z/0`, S.z, S.x, rows, c);
    this.#mm(W, S.x, `${A}/proj_z/1/weight`, S.pbias, rows, c, this.heads, 0);
    this.#mm(W, S.sn, `${A}/proj_q/weight`, S.sq, n, d, d, 0, null, `${A}/proj_q/bias`);
    this.#mm(W, S.sn, `${A}/proj_k/weight`, S.sk, n, d, d, 0);
    this.#mm(W, S.sn, `${A}/proj_v/weight`, S.sv, n, d, d, 0);
    this.#mm(W, S.sn, `${A}/proj_g/weight`, S.sg, n, d, d, 0);
    this.ctx.flush();
    const width = d / this.heads;
    this.ctx.dispatch1d(`sattn:${this.heads}x${width}`, singleAttention(this.heads, width),
      [S.sq, S.sk, S.sv, S.sg, S.pbias, S.tokMask, S.so], [n], n * this.heads);
    this.#mm(W, S.so, `${A}/proj_o/weight`, s, n, d, d, 3);
    this.ctx.flush();
    this.#transition(W, "transition_s", s, S.sn, S.su, S.sh, n, d, 4 * d);
  }

  /**
   * Run the trunk pairformer.
   * @param {GPUBuffer|Float32Array} s [n, 384]  (updated in place when a GPUBuffer)
   * @param {GPUBuffer|Float32Array} z [n, n, 128] (updated in place when a GPUBuffer)
   * @param {Float32Array} mask token_pad_mask [n]; the pair mask is mask[i] * mask[j]
   * @param {{layers?: number, startLayer?: number, singleTrack?: boolean, tapLayers?: number[],
   *          taps?: Map<string, Float32Array>, onLayer?: (i: number, bufs: {s, z}) => Promise<void>|void,
   *          timings?: number[]}} options
   *   `tapLayers` + `taps` store `layer{i}_s` / `layer{i}_z` (after layer i); `onLayer` runs after each
   *   layer (buffers are only valid during the call); `timings` receives wall ms per layer.
   * @returns {Promise<{s: GPUBuffer, z: GPUBuffer, release: () => void}>} the buffers are owned by the
   *   caller when they were passed in; created from arrays they are freed by `release()`.
   */
  async run(s, z, mask, options = {}) {
    const { layers = this.numBlocks, startLayer = 0, singleTrack = true, tapLayers = [], taps, onLayer, timings } = options;
    const ctx = this.ctx;
    const n = mask.length;
    const rows = n * n;
    const c = this.zDim;
    const d = this.sDim;
    const owned = [];
    const asBuffer = (x, label) => {
      if (x instanceof Float32Array) { const b = ctx.upload(label, x, true); owned.push(b); return b; }
      return x;
    };
    const sBuf = asBuffer(s, "pf.s");
    const zBuf = asBuffer(z, "pf.z");
    const scratch = [];
    const buf = (label, floats) => { const b = ctx.buffer(`pf.${label}`, floats, true); scratch.push(b); return b; };
    const pairMaskData = new Float32Array(rows);
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) pairMaskData[i * n + j] = mask[i] * mask[j];
    const hw = this.triHeads * this.triWidth;
    const S = { n, rows, c, z: zBuf };
    S.pairMask = asBuffer(pairMaskData, "pf.pairMask"); scratch.push(S.pairMask); owned.pop();
    S.tokMask = asBuffer(mask, "pf.tokMask"); scratch.push(S.tokMask); owned.pop();
    S.x = buf("x", rows * c);
    S.y = buf("y", rows * c);
    S.hidden = buf("hidden", rows * c);
    S.gate = buf("gate", rows * c);
    S.p2 = buf("p2", rows * 2 * c);
    S.g2 = buf("g2", rows * 2 * c);
    S.tbias = buf("tbias", rows * this.triHeads);
    for (const name of ["q", "k", "v", "g", "o"]) S[name] = buf(name, rows * hw);
    const wide = Math.min(rows, ROW_CHUNK);
    S.u = buf("u", wide * 4 * c);
    S.h = buf("h", wide * 4 * c);
    if (singleTrack) {
      S.sn = buf("sn", n * d);
      S.pbias = buf("pbias", rows * this.heads);
      for (const name of ["sq", "sk", "sv", "sg", "so"]) S[name] = buf(name, n * d);
      S.su = buf("su", n * 4 * d);
      S.sh = buf("sh", n * 4 * d);
    }

    try {
      for (let i = startLayer; i < Math.min(this.numBlocks, startLayer + layers); i++) {
        const t0 = performance.now();
        const W = await this.#loadLayer(i);
        this.#triMul(W, "tri_mul_out", S, true);
        this.#triMul(W, "tri_mul_in", S, false);
        this.#triAtt(W, "tri_att_start", S, true);
        this.#triAtt(W, "tri_att_end", S, false);
        this.#transition(W, "transition_z", zBuf, S.x, S.u, S.h, rows, c, 4 * c);
        if (singleTrack) this.#singleTrack(W, S, sBuf);
        ctx.flush();
        await this.device.queue.onSubmittedWorkDone();
        timings?.push(performance.now() - t0);
        if (taps && tapLayers.includes(i)) {
          if (singleTrack) taps.set(`layer${i}_s`, await ctx.read(sBuf, n * d));
          taps.set(`layer${i}_z`, await ctx.read(zBuf, rows * c));
        }
        if (onLayer) await onLayer(i, { s: sBuf, z: zBuf });
        for (const b of W.values()) b.destroy();
        this.weights.release(`${this.prefix}/layers/${i}/`);
        await ctx.release();
      }
      await this.device.queue.onSubmittedWorkDone();
    } finally {
      for (const b of scratch) b.destroy();
    }
    return { s: sBuf, z: zBuf, release: () => { for (const b of owned) b.destroy(); } };
  }

  destroy() { this.dummy.destroy(); }
}
