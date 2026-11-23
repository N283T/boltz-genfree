// Boltz-2 MSAModule on WebGPU (f32), written from the PyTorch reference
// (src/boltz/model/modules/trunkv2.py: MSAModule / MSALayer). No LocalFold import;
// LocalFold (sokrypton/localfold) was only used to confirm the Boltz-2 dialect:
// msaUpdateBeforeOuterProduct (m is updated before the outer-product mean),
// opmBiasAfterNorm (divide by the pair count, then proj_o with its bias) and
// swapTransposedBias (handled by the shared triangle-attention kernel).
//
// Per block (this repository's MSALayer order):
//   m += PWA(m, z)            pair-weighted averaging, 8 heads x 32, bias from z
//   m += transition(m)        SwiGLU, hidden 4 * 64
//   z += OPM(m)               outer-product mean over the MSA rows
//   z  = PairformerNoSeqLayer(z)   tri-mul out/in, tri-att start/end, transition (z-track of the affinity block)
// The module returns z (input plus all updates), and Boltz2.forward then does
// `z = z + msa_module(z, ...)`, so the caller adds its own z back on top.
import { AffinityGpu } from "../../gpu/affinity-gpu.js";
import * as K from "../../gpu/wgsl.js";
import * as W from "./msa-module-wgsl.js";

const C_Z = 128;
const C_M = 64;
const PWA_HEADS = 8;
const PWA_WIDTH = 32;
const OPM_HIDDEN = 32;
const TRI_HEADS = 4;
const TRI_WIDTH = 32;
const NUM_TOKENS = 33;
const ROW_CHUNK = 16384; // rows per matmul/transition chunk

const BLOCK_ROOT = "b"; // pseudo root: the weights of the block currently resident on the GPU

/** Pair-track ops are inherited from AffinityGpu; its weight cache is fed one MSA block at a time. */
class PairOps extends AffinityGpu {
  constructor(ctx, bundle) {
    super(ctx.device, bundle);
    this.dummy.destroy();
    this.ctx = ctx;
    this.dummy = ctx.buffer("msa.dummy", 4, true);
  }

  /** Drop the resident block weights (call after the GPU is idle). */
  dropWeights() {
    for (const buffer of this.weights.values()) buffer.destroy();
    this.weights.clear();
  }

  /** Residual SwiGLU transition on rows [baseRow, baseRow + rows) of `buf`, in chunks. */
  transitionRows(prefix, buf, baseRow, rows, c, hidden, x, u, h) {
    for (let r0 = 0; r0 < rows; r0 += ROW_CHUNK) {
      const n = Math.min(ROW_CHUNK, rows - r0);
      const target = this.slice(buf, baseRow + r0, n, c);
      this.layerNorm(BLOCK_ROOT, `${prefix}/norm`, target, x, n, c);
      this.matmul(BLOCK_ROOT, x, `${prefix}/fc1/weight`, u, n, c, hidden, 6);
      this.matmul(BLOCK_ROOT, x, `${prefix}/fc2/weight`, h, n, c, hidden, 4, u);
      this.matmul(BLOCK_ROOT, h, `${prefix}/fc3/weight`, target, n, hidden, c, 3);
      this.ctx.flush();
    }
  }
}

const bytes = (floats) => Math.max(16, floats * 4);

export class MsaModule {
  /** @param {import("../../gpu/context.js").GpuContext} ctx @param {import("../own-weights.js").OwnWeights} weights */
  constructor(ctx, weights) {
    this.ctx = ctx;
    this.weights = weights;
    this.blocks = weights.arch.msaBlocks ?? 4;
    this.current = null;
    const self = this;
    this.ops = new PairOps(ctx, { get: (key) => self.#tensor(key) });
  }

  #tensor(key) {
    const name = key.slice(BLOCK_ROOT.length + 1);
    const tensor = this.current?.[name];
    if (tensor === undefined) throw new Error(`MSA block has no tensor ${name}`);
    return tensor.data;
  }

  /** Rows of MSA processed per sequence chunk (a multiple that keeps 128-byte-per-row slices 256-aligned). */
  static sequenceChunk(n, rows) {
    let chunk = Math.max(1, Math.floor(ROW_CHUNK / n));
    if ((chunk * n) % 2 === 1) chunk = Math.max(1, chunk - 1);
    return Math.min(chunk, rows);
  }

  /** Largest buffers and total scratch bytes for `rows` MSA rows over `n` tokens. */
  static estimate(n, rows) {
    const pair = n * n;
    const chunkRows = MsaModule.sequenceChunk(n, rows) * n;
    const m = rows * n * C_M;
    const ab = rows * n * OPM_HIDDEN;
    const pairBuffers = [
      pair * C_Z * 6, // z, x, y, hidden, gate, pair q/k/v/g/o share the same size class (counted loosely)
      pair * 2 * C_Z * 2, // p2, g2
      pair * TRI_HEADS * TRI_WIDTH * 5,
      pair * PWA_HEADS * 2, // pwa bias + weights
    ].reduce((a, b) => a + b, 0);
    const scratch = chunkRows * (C_M + 3 * PWA_HEADS * PWA_WIDTH) + Math.min(pair, ROW_CHUNK) * 4 * C_Z * 2
      + Math.min(pair, ROW_CHUNK) * 1024;
    const largestFloats = Math.max(m, ab, pair * 2 * C_Z, rows * n * 4);
    return { largestBufferBytes: largestFloats * 4, totalBytes: (m + 2 * ab + pairBuffers + scratch + rows * n * 4 * 4) * 4 };
  }

  /**
   * Run the MSA module.
   * @param {Float32Array|GPUBuffer} z     [N*N*128] module input (a GPUBuffer is updated in place)
   * @param {Float32Array|GPUBuffer} sInputs [N*384]
   * @param {{msa, msaPaired, hasDeletion, deletionValue, msaMask}} msaFeatures  each [S*N] (S >= 1), row-major [s, i]
   * @param {{tokenPadMask: ArrayLike<number>}} masks  [N] token pad mask (1 = real)
   * @param {{taps?: Map<string, Float32Array>}} options  taps: msa_embed_m, msa_block{i}_{m_after_pwa, m_after_trans,
   *   z_after_opm, z_after_tri, z_out}, z_msa_module_out
   * @returns {Promise<Float32Array|GPUBuffer>} z_out (the module's full output, not a delta)
   */
  async run(z, sInputs, msaFeatures, masks, { taps } = {}) {
    const ctx = this.ctx;
    const ops = this.ops;
    const padMask = masks.tokenPadMask;
    const n = padMask.length;
    const pairRows = n * n;
    const S = msaFeatures.msa.length / n;
    if (!Number.isInteger(S) || S < 1) throw new Error(`MSA features must hold S >= 1 rows of ${n} tokens`);
    const est = MsaModule.estimate(n, S);
    const limits = ctx.device.limits;
    const maxBuf = Math.min(limits.maxBufferSize ?? Infinity, limits.maxStorageBufferBindingSize ?? Infinity);
    if (est.largestBufferBytes > maxBuf) {
      throw new Error(`MSA module: ${S} rows x ${n} tokens needs a ${est.largestBufferBytes} byte buffer, device limit ${maxBuf}`);
    }

    const owned = [];
    const buf = (label, floats) => { const b = ctx.buffer(label, floats, true); owned.push(b); return b; };
    const up = (label, data) => { const b = ctx.upload(label, data, true); owned.push(b); return b; };
    const tap = async (name, buffer, floats) => { if (taps) taps.set(name, await ctx.read(buffer, floats)); };
    const idle = async () => { await ctx.read(ops.dummy, 1); }; // flush, wait, reap uniform blocks

    try {
      const zBuf = z instanceof Float32Array ? up("msa.z", z) : z;
      const sBuf = sInputs instanceof Float32Array ? up("msa.s_inputs", sInputs) : sInputs;
      const pairMaskData = new Float32Array(pairRows);
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) pairMaskData[i * n + j] = padMask[i] * padMask[j];
      const pairMask = up("msa.pair_mask", pairMaskData);

      // ---- embedding -----------------------------------------------------------
      const sProj = up("msa.s_proj", this.weights.get("msa_module/s_proj/weight"));
      const mProj = up("msa.msa_proj", this.weights.get("msa_module/msa_proj/weight"));
      const sp = buf("msa.sp", n * C_M);
      ops.ctx.dispatch("matmul", K.MATMUL, [sBuf, sProj, ops.dummy, ops.dummy, sp], [n, 384, C_M, 0, 0], Math.ceil(C_M / 16), Math.ceil(n / 16));
      const f32 = (a) => (a instanceof Float32Array ? a : Float32Array.from(a));
      const feat = {
        msa: up("msa.feat.msa", f32(msaFeatures.msa)),
        paired: up("msa.feat.paired", f32(msaFeatures.msaPaired)),
        hasDel: up("msa.feat.has_deletion", f32(msaFeatures.hasDeletion)),
        delVal: up("msa.feat.deletion_value", f32(msaFeatures.deletionValue)),
        mask: up("msa.feat.mask", f32(msaFeatures.msaMask)),
      };
      const m = buf("msa.m", S * n * C_M);
      ctx.dispatch1d("msa_embed", W.MSA_EMBED, [feat.msa, feat.paired, feat.hasDel, feat.delVal, mProj, sp, m],
        [S * n, n, NUM_TOKENS, NUM_TOKENS + 3], S * n * C_M);
      ctx.flush();
      await tap("msa_embed_m", m, S * n * C_M);

      // ---- scratch ---------------------------------------------------------------
      const sc = MsaModule.sequenceChunk(n, S);
      const chunkRows = sc * n;
      const hw = PWA_HEADS * PWA_WIDTH;
      const xm = buf("msa.xm", chunkRows * C_M);
      const bv = buf("msa.v", chunkRows * hw);
      const bg = buf("msa.g", chunkRows * hw);
      const bo = buf("msa.o", chunkRows * hw);
      const wide = Math.min(pairRows, ROW_CHUNK);
      const u = buf("msa.u", wide * 4 * C_Z);
      const h = buf("msa.h", wide * 4 * C_Z);
      const pwaBias = buf("msa.pwa_bias", pairRows * PWA_HEADS);
      const pwaW = buf("msa.pwa_w", pairRows * PWA_HEADS);
      const opmA = buf("msa.opm_a", S * n * OPM_HIDDEN);
      const opmB = buf("msa.opm_b", S * n * OPM_HIDDEN);
      const opmNum = buf("msa.opm_num", pairRows);
      const outerRows = Math.max(1, Math.floor(ROW_CHUNK / n));
      const outer = buf("msa.opm_outer", outerRows * n * OPM_HIDDEN * OPM_HIDDEN);
      const s = { n, rows: pairRows, c: C_Z, mask: pairMask, z: zBuf };
      s.x = buf("msa.x", pairRows * C_Z);
      s.y = buf("msa.y", pairRows * C_Z);
      s.hidden = buf("msa.hidden", pairRows * C_Z);
      s.gate = buf("msa.gate", pairRows * C_Z);
      s.p2 = buf("msa.p2", pairRows * 2 * C_Z);
      s.g2 = buf("msa.g2", pairRows * 2 * C_Z);
      s.bias = buf("msa.tri_bias", pairRows * TRI_HEADS);
      for (const name of ["q", "k", "v", "g", "o"]) s[name] = buf(`msa.tri_${name}`, pairRows * TRI_HEADS * TRI_WIDTH);

      ctx.dispatch1d("opm_count", W.OPM_COUNT, [feat.mask, opmNum], [S, n], pairRows);
      ctx.flush();

      for (let block = 0; block < this.blocks; block++) {
        this.current = this.weights.slice(`msa_module/layers/${block}`);
        const pre = `msa_block${block}`;
        const pwa = "pair_weighted_averaging";

        // m += PWA(m, z)
        ops.layerNorm(BLOCK_ROOT, `${pwa}/norm_z`, zBuf, s.x, pairRows, C_Z);
        ops.matmul(BLOCK_ROOT, s.x, `${pwa}/proj_z/weight`, pwaBias, pairRows, C_Z, PWA_HEADS, 0);
        ctx.dispatch1d("pwa_weights", W.PWA_WEIGHTS, [pwaBias, pairMask, pwaW], [n, PWA_HEADS], n * PWA_HEADS);
        ctx.flush();
        for (let s0 = 0; s0 < S; s0 += sc) {
          const cs = Math.min(sc, S - s0);
          const rows = cs * n;
          const mc = ops.slice(m, s0 * n, rows, C_M);
          ops.layerNorm(BLOCK_ROOT, `${pwa}/norm_m`, mc, xm, rows, C_M);
          ops.matmul(BLOCK_ROOT, xm, `${pwa}/proj_m/weight`, bv, rows, C_M, hw, 0);
          ops.matmul(BLOCK_ROOT, xm, `${pwa}/proj_g/weight`, bg, rows, C_M, hw, 2);
          ctx.dispatch1d("pwa_apply", W.PWA_APPLY, [bv, bg, pwaW, bo], [cs, n, PWA_WIDTH, hw], rows * hw);
          ops.matmul(BLOCK_ROOT, bo, `${pwa}/proj_o/weight`, mc, rows, hw, C_M, 3);
          ctx.flush();
        }
        await tap(`${pre}_m_after_pwa`, m, S * n * C_M);

        // m += transition(m)
        for (let s0 = 0; s0 < S; s0 += sc) {
          const cs = Math.min(sc, S - s0);
          ops.transitionRows("msa_transition", m, s0 * n, cs * n, C_M, 4 * C_M, xm, bv, bg);
        }
        await tap(`${pre}_m_after_trans`, m, S * n * C_M);

        // z += OPM(m): a, b = mask * proj(LN(m)); mean over rows of the outer product, then proj_o (+ bias)
        const opm = "outer_product_mean";
        for (let s0 = 0; s0 < S; s0 += sc) {
          const cs = Math.min(sc, S - s0);
          const rows = cs * n;
          ops.layerNorm(BLOCK_ROOT, `${opm}/norm`, ops.slice(m, s0 * n, rows, C_M), xm, rows, C_M);
          ops.matmul(BLOCK_ROOT, xm, `${opm}/proj_a/weight`, ops.slice(opmA, s0 * n, rows, OPM_HIDDEN), rows, C_M, OPM_HIDDEN, 0);
          ops.matmul(BLOCK_ROOT, xm, `${opm}/proj_b/weight`, ops.slice(opmB, s0 * n, rows, OPM_HIDDEN), rows, C_M, OPM_HIDDEN, 0);
          ctx.flush();
        }
        for (const t of [opmA, opmB]) ctx.dispatch1d("scale_rows", K.SCALE_ROWS, [t, feat.mask], [S * n * OPM_HIDDEN, OPM_HIDDEN], S * n * OPM_HIDDEN);
        ctx.flush();
        for (let i0 = 0; i0 < n; i0 += outerRows) {
          const ci = Math.min(outerRows, n - i0);
          const hwid = OPM_HIDDEN;
          ctx.dispatch("opm_outer", W.OPM_OUTER, [opmA, opmB, opmNum, outer], [S, n, i0, ci, hwid],
            Math.ceil((n * hwid) / 16), Math.ceil((ci * hwid) / 16));
          ops.matmul(BLOCK_ROOT, outer, `${opm}/proj_o/weight`, ops.slice(zBuf, i0 * n, ci * n, C_Z),
            ci * n, hwid * hwid, C_Z, 3, null, `${opm}/proj_o/bias`);
          ctx.flush();
        }
        await tap(`${pre}_z_after_opm`, zBuf, pairRows * C_Z);

        // z = PairformerNoSeqLayer(z)
        const layer = "pairformer_layer";
        ops.triangleMultiplication(BLOCK_ROOT, `${layer}/tri_mul_out`, s, true);
        ops.triangleMultiplication(BLOCK_ROOT, `${layer}/tri_mul_in`, s, false);
        ops.triangleAttention(BLOCK_ROOT, `${layer}/tri_att_start`, s, true, TRI_HEADS, TRI_WIDTH);
        ops.triangleAttention(BLOCK_ROOT, `${layer}/tri_att_end`, s, false, TRI_HEADS, TRI_WIDTH);
        ops.transitionRows(`${layer}/transition_z`, zBuf, 0, pairRows, C_Z, 4 * C_Z, s.x, u, h);
        // The reference's "z_after_tri" hook sits on the whole pair layer, so it equals z_out.
        await tap(`${pre}_z_after_tri`, zBuf, pairRows * C_Z);
        await tap(`${pre}_z_out`, zBuf, pairRows * C_Z);

        // Stream weights: free this block's tensors once the GPU is done with them.
        await idle();
        ops.dropWeights();
        this.current = null;
      }
      await tap("z_msa_module_out", zBuf, pairRows * C_Z);
      if (z instanceof Float32Array) return await ctx.read(zBuf, pairRows * C_Z);
      return zBuf;
    } finally {
      await idle();
      ops.dropWeights();
      this.current = null;
      for (const b of owned) {
        if (b !== z && b !== sInputs) b.destroy();
      }
    }
  }
}
