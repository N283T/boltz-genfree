// Boltz-2 trunk driver on the own WebGPU modules (no LocalFold): the sequence of Boltz2.forward
// (src/boltz/model/models/boltz2.py) up to the trunk outputs and the affinity branch inputs.
//
//   s_inputs = input_embedder(feats)                      Embedder.sInputs
//   s_init = s_init(s_inputs); z_init = z_init_1[i] + z_init_2[j] + rel_pos + bonds + contact      Embedder.initial
//   s = z = 0
//   for pass in 0..recycling_steps:
//     s = s_init + s_recycle(s_norm(s));  z = z_init + z_recycle(z_norm(z))                         Embedder.recycle
//     z = z + msa_module(z, s_inputs, feats)       (the module returns z + its updates, so z_in is counted twice)
//     s, z = pairformer_module(s, z, mask)
//   affinity branch: s_inputs_affinity = input_embedder(feats with profile_affinity / deletion_mean_affinity),
//                    z_affinity = z * (protein-ligand cross-pair mask)   (applied by the caller, see chain.js)
//
// Recycling steps default to 0 (GenFree). All state stays on the GPU between stages (s, z, s_inputs, z_init, s_init
// are GPUBuffers); only the final s, z and the two s_inputs are read back. Weights stream per module: the embedder keeps
// its small tables resident, the MSA module and the pairformer upload one block / layer at a time.
//
// MSA rows: Boltz subsamples to `num_subsampled_msa` (1024) rows with torch.randperm (GenFree seeds it per record and
// step). Here a deeper alignment is subsampled with a seeded JS shuffle per pass, which is not torch's stream; any
// alignment of at most 1024 rows is used whole and in order.
import { GpuContext } from "../../gpu/context.js";
import { ADD } from "../../gpu/wgsl.js";
import { Embedder } from "./embedder.js";
import { MsaModule } from "./msa-module.js";
import { Pairformer } from "./pairformer.js";

export const MSA_SUBSAMPLE = 1024;
const TOKEN_Z = 128;
const TOKEN_S = 384;

function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The five MSA features of one pass, `limit` rows at most (seeded subsample, rows kept in drawn order). */
export function msaFeaturesFor(features, tokens, { seed = 0, pass = 0, limit = MSA_SUBSAMPLE } = {}) {
  const names = { msa: "feat/msa", msaPaired: "feat/msa_paired", hasDeletion: "feat/has_deletion", deletionValue: "feat/deletion_value", msaMask: "feat/msa_mask" };
  const all = Object.fromEntries(Object.entries(names).map(([key, name]) => [key, features.get(name)]));
  const rows = all.msa.length / tokens;
  if (rows <= limit) return { features: all, rows };
  const order = Int32Array.from({ length: rows }, (_, i) => i);
  const random = rng((seed * 2654435761 + pass * 40503 + 1) >>> 0);
  for (let i = rows - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    const t = order[i]; order[i] = order[j]; order[j] = t;
  }
  const picked = {};
  for (const [key, array] of Object.entries(all)) {
    const out = new Float32Array(limit * tokens);
    for (let r = 0; r < limit; r += 1) out.set(array.subarray(order[r] * tokens, (order[r] + 1) * tokens), r * tokens);
    picked[key] = out;
  }
  return { features: picked, rows: limit };
}

export class OwnTrunk {
  /** @param {GPUDevice} device @param {import("../own-weights.js").OwnWeights} weights */
  constructor(device, weights) {
    this.device = device;
    this.weights = weights;
    this.ctx = new GpuContext(device); // embedder + driver kernels
    this.embedder = new Embedder(this.ctx, weights);
    this.msa = new MsaModule(new GpuContext(device), weights);
    this.pairformer = new Pairformer(device, weights);
  }

  #copy(source, floats) {
    const target = this.ctx.buffer("run.copy", floats, true);
    this.ctx.flush();
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(source, 0, target, 0, floats * 4);
    this.device.queue.submit([encoder.finish()]);
    return target;
  }

  /**
   * @param {{get: Function, shape: Function}} features accessor over the `feat/*` tensors (featureBundle(featuriseOwn(..)) or an oracle dump)
   * @param {{recycles?: number, seed?: number, onStage?: Function, taps?: Map, affinity?: boolean}} [options]
   * @returns {Promise<{pair: Float32Array, single: Float32Array, sInputsTrunk: Float32Array, sInputsAffinity: Float32Array,
   *                    tokens: number, msaRows: number, timings: object}>}
   */
  async run(features, { recycles = 0, seed = 0, onStage, taps = null, affinity = true } = {}) {
    const { ctx, embedder } = this;
    const timings = {};
    const mark = async (name, work) => {
      const start = performance.now();
      const value = await work();
      timings[name] = (timings[name] ?? 0) + performance.now() - start;
      onStage?.(name, timings[name]);
      return value;
    };
    const n = features.shape("feat/token_pad_mask")[1];
    const mask = Float32Array.from(features.get("feat/token_pad_mask"));
    const zFloats = n * n * TOKEN_Z;
    const sFloats = n * TOKEN_S;
    const keep = []; // buffers destroyed at the end
    let msaRows = 0;
    let prevS = null;
    let prevZ = null;
    try {
      await embedder.prepare(); // lazy URL packs: the ~6 MB of embedder tensors by Range; layers stream per block below
      const { sIn, sInit, zInit } = await mark("embed", async () => {
        const sIn = embedder.sInputs(features, { taps });
        const { sInit, zInit } = embedder.initial(features, sIn, { taps });
        await ctx.release();
        return { sIn, sInit, zInit };
      });
      keep.push(sIn, sInit, zInit);

      for (let pass = 0; pass <= recycles; pass += 1) {
        const { s, z } = await mark("recycle", async () => {
          const out = embedder.recycle(sInit, zInit, prevS, prevZ, { taps });
          await ctx.release();
          return out;
        });
        if (prevS !== null) { prevS.destroy(); prevZ.destroy(); }
        // z = z + msa_module(z, s_inputs, feats): the module updates its input in place, so run it on a copy
        await mark("msa", async () => {
          const rows = msaFeaturesFor(features, n, { seed, pass });
          msaRows = rows.rows;
          const zCopy = this.#copy(z, zFloats);
          await this.msa.run(zCopy, sIn, rows.features, { tokenPadMask: mask }, { taps: pass === 0 ? taps : undefined });
          ctx.dispatch1d("add", ADD, [z, zCopy], [zFloats], zFloats);
          await ctx.release();
          zCopy.destroy();
        });
        const out = await mark("pairformer", () => this.pairformer.run(s, z, mask, { taps: pass === recycles ? taps : undefined }));
        prevS = out.s;
        prevZ = out.z;
      }
      const pair = await mark("readback", () => this.pairformer.ctx.read(prevZ, zFloats));
      const single = await this.pairformer.ctx.read(prevS, sFloats);
      const sInputsTrunk = await ctx.read(sIn, sFloats);
      let sInputsAffinity = null;
      if (affinity) {
        const sAff = await mark("embed-affinity", async () => {
          const buffer = embedder.sInputs(features, { affinity: true, taps: null });
          const data = await ctx.read(buffer, sFloats);
          await ctx.release();
          buffer.destroy();
          return data;
        });
        sInputsAffinity = sAff;
      }
      return { pair, single, sInputsTrunk, sInputsAffinity, tokens: n, msaRows, timings };
    } finally {
      embedder.releaseWeights();
      prevS?.destroy();
      prevZ?.destroy();
      for (const buffer of keep) buffer.destroy();
      await ctx.release();
    }
  }

  async destroy() {
    this.embedder.destroy();
    this.pairformer.destroy();
    await this.ctx.release();
    await this.msa.ctx.release();
  }
}
