// WebGPU implementation of the Boltz-2 AffinityModule (structure-free path).
import { affinityMasks, buildStructureFreeBins } from "../distogram.js";
import { boundariesOf, moduleConfig } from "../module-config.js";
import { GpuContext } from "./context.js";
import * as K from "./wgsl.js";

const ROW_CHUNK = 16384; // rows per transition chunk; keeps the 4c hidden buffers modest

export class AffinityGpu {
  /** @param {GPUDevice} device @param {import("../bundle.js").TensorBundle} bundle */
  constructor(device, bundle) {
    this.device = device;
    this.bundle = bundle;
    this.ctx = new GpuContext(device);
    this.weights = new Map();
    this.dummy = this.ctx.buffer("dummy", 4, true);
  }

  /** Weight buffer, uploaded once. */
  w(root, path) {
    const key = `${root}/${path}`;
    let buffer = this.weights.get(key);
    if (buffer === undefined) {
      buffer = this.ctx.upload(key, this.bundle.get(key), true);
      this.weights.set(key, buffer);
    }
    return buffer;
  }

  // ---- op helpers -------------------------------------------------------------
  layerNorm(root, prefix, a, out, rows, c, b = null) {
    this.ctx.dispatch("ln", K.LAYERNORM,
      [a, b ?? this.dummy, this.w(root, `${prefix}/weight`), this.w(root, `${prefix}/bias`), out],
      [rows, c, b === null ? 0 : 1], Math.min(rows, K.GROUP_WIDTH), Math.ceil(rows / K.GROUP_WIDTH));
  }

  /** out = epilogue(a . W^T + bias). `weight` is a tensor path, `bias` an optional path. */
  matmul(root, a, weightPath, out, rows, inDim, outDim, epilogue = 0, aux = null, biasPath = null) {
    this.ctx.dispatch("matmul", K.MATMUL,
      [a, this.w(root, weightPath), biasPath ? this.w(root, biasPath) : this.dummy, aux ?? this.dummy, out],
      [rows, inDim, outDim, epilogue, biasPath ? 1 : 0], Math.ceil(outDim / 16), Math.ceil(rows / 16));
  }

  slice(buffer, row, rows, width) {
    return { buffer, offset: row * width * 4, size: rows * width * 4 };
  }

  /** z += transition(z), SwiGLU with hidden width `hidden`, in row chunks. */
  transition(root, prefix, z, x, u, h, rows, c, hidden) {
    this.layerNorm(root, `${prefix}/norm`, z, x, rows, c);
    for (let r0 = 0; r0 < rows; r0 += ROW_CHUNK) {
      const n = Math.min(ROW_CHUNK, rows - r0);
      const xs = this.slice(x, r0, n, c);
      const zs = this.slice(z, r0, n, c);
      this.matmul(root, xs, `${prefix}/fc1/weight`, u, n, c, hidden, 6);
      this.matmul(root, xs, `${prefix}/fc2/weight`, h, n, c, hidden, 4, u);
      this.matmul(root, h, `${prefix}/fc3/weight`, zs, n, hidden, c, 3);
      this.ctx.flush();
    }
  }

  triangleMultiplication(root, prefix, s, outgoing) {
    const { n, rows, c, mask } = s;
    this.layerNorm(root, `${prefix}/norm_in`, s.z, s.x, rows, c);
    this.matmul(root, s.x, `${prefix}/g_in/weight`, s.g2, rows, c, 2 * c, 2);
    this.matmul(root, s.x, `${prefix}/p_in/weight`, s.p2, rows, c, 2 * c, 4, s.g2);
    this.ctx.dispatch1d("scale_rows", K.SCALE_ROWS, [s.p2, mask], [rows * 2 * c, 2 * c], rows * 2 * c);
    this.ctx.dispatch1d("contract", K.TRI_CONTRACT, [s.p2, s.hidden], [n, c, outgoing ? 1 : 0], rows * c);
    this.layerNorm(root, `${prefix}/norm_out`, s.hidden, s.y, rows, c);
    this.matmul(root, s.x, `${prefix}/g_out/weight`, s.gate, rows, c, c, 2);
    this.matmul(root, s.y, `${prefix}/p_out/weight`, s.z, rows, c, c, 5, s.gate);
    this.ctx.flush();
  }

  triangleAttention(root, prefix, s, starting, heads, width) {
    const { n, rows, c, mask } = s;
    const hw = heads * width;
    this.layerNorm(root, `${prefix}/layer_norm`, s.z, s.x, rows, c);
    this.matmul(root, s.x, `${prefix}/linear/weight`, s.bias, rows, c, heads, 0);
    this.matmul(root, s.x, `${prefix}/mha/linear_q/weight`, s.q, rows, c, hw, 0);
    this.matmul(root, s.x, `${prefix}/mha/linear_k/weight`, s.k, rows, c, hw, 0);
    this.matmul(root, s.x, `${prefix}/mha/linear_v/weight`, s.v, rows, c, hw, 0);
    this.matmul(root, s.x, `${prefix}/mha/linear_g/weight`, s.g, rows, c, hw, 0);
    this.ctx.flush();
    this.ctx.dispatch1d(`triatt:${heads}x${width}`, K.triAttention(heads, width),
      [s.q, s.k, s.v, s.g, s.bias, mask, s.o], [n, starting ? 1 : 0], n * n * heads);
    this.ctx.flush();
    this.matmul(root, s.o, `${prefix}/mha/linear_o/weight`, s.z, rows, hw, c, 3);
    this.ctx.flush();
  }

  pool(z, poolMask, rows, c, out) {
    const chunks = Math.ceil(rows / 64);
    const part = this.ctx.buffer("pool.part", chunks * c);
    this.ctx.dispatch("pool_partial", K.POOL_PARTIAL, [z, poolMask, part], [rows, c], chunks, Math.ceil(c / 64));
    const total = poolMask.sum;
    this.ctx.dispatch("pool_reduce", K.POOL_REDUCE, [part, out], [chunks, c, 0, { f32: Math.fround(total + 1e-7) }], Math.ceil(c / 64));
  }

  /**
   * Run one ensemble member.
   * @param {object} input {sInputs, zAffinity, padMask, molType, affinityTokenMask}
   * @param {{mode?: string, seed?: number, taps?: Map<string, Float32Array>, bins?: Uint32Array}} options
   */
  async runModule(root, input, options = {}) {
    const { mode = "zero_distogram_features", seed = 0, taps, bins: givenBins } = options;
    const config = moduleConfig(this.bundle, root);
    const { tokenZ: c, tokenS: sDim, headsTokenS, triHeads: heads, triHeadWidth: width } = config;
    const n = input.padMask.length;
    const rows = n * n;
    const ctx = this.ctx;
    const masks = affinityMasks(input.padMask, input.molType, input.affinityTokenMask);
    const built = buildStructureFreeBins({
      boundaries: boundariesOf(config), receptor: masks.receptor, ligand: masks.ligand, mode, seed,
    });
    const bins = givenBins ?? built.bins;

    const poolMask = ctx.upload("poolMask", masks.poolMask);
    poolMask.sum = masks.poolMask.reduce((a, b) => a + b, 0);
    const pairMask = ctx.upload("pairMask", masks.crossPairMask);
    const z = ctx.upload("z", input.zAffinity);
    const s = { n, rows, c, mask: pairMask, z };
    const wideRows = Math.min(rows, ROW_CHUNK);
    s.x = ctx.buffer("x", rows * c);
    s.y = ctx.buffer("y", rows * c);
    s.hidden = ctx.buffer("hidden", rows * c);
    s.gate = ctx.buffer("gate", rows * c);
    s.p2 = ctx.buffer("p2", rows * 2 * c);
    s.g2 = ctx.buffer("g2", rows * 2 * c);
    s.bias = ctx.buffer("bias", rows * heads);
    for (const name of ["q", "k", "v", "g", "o"]) s[name] = ctx.buffer(name, rows * heads * width);
    const u = ctx.buffer("u", wideRows * 4 * c);
    const h = ctx.buffer("h", wideRows * 4 * c);
    const pooled = ctx.buffer("pooled", c);
    const tap = async (name, buffer, floats) => {
      if (taps) taps.set(name, await ctx.read(buffer, floats));
    };
    const tapPool = async (name) => {
      if (!taps) return;
      this.pool(z, poolMask, rows, c, pooled);
      await tap(name, pooled, c);
    };

    // z = z_linear(z_norm(z)) + s_to_z_prod_in1(s)[:, None] + s_to_z_prod_in2(s)[None]
    this.layerNorm(root, "z_norm", z, s.x, rows, c);
    this.matmul(root, s.x, "z_linear/weight", z, rows, c, c, 0);
    await tap("z_linear", z, rows * c);
    const sBuf = ctx.upload("s_inputs", input.sInputs);
    const s1 = ctx.buffer("s1", n * c);
    const s2 = ctx.buffer("s2", n * c);
    this.matmul(root, sBuf, "s_to_z_prod_in1/weight", s1, n, sDim, c);
    this.matmul(root, sBuf, "s_to_z_prod_in2/weight", s2, n, sDim, c);
    ctx.dispatch1d("pair_add", K.PAIR_ADD_ROWCOL, [z, s1, s2], [n, c], rows * c);
    await tapPool("pooled_processed");

    // distogram features and the pairwise conditioner
    if (built.conditioning) {
      const embed = this.bundle.get(`${root}/dist_bin_pairwise_embed/weight`);
      let distogram = ctx.buffer("distogram", rows * c); // zero-initialised: zero_distogram_features
      if (built.features !== "zero") {
        let table = this.w(root, "dist_bin_pairwise_embed/weight");
        if (built.features === "mean") {
          const mean = new Float32Array(c);
          const numBins = embed.length / c;
          for (let b = 0; b < numBins; b++) for (let d = 0; d < c; d++) mean[d] += embed[b * c + d] / numBins;
          table = ctx.upload("mean_embedding", mean);
        }
        const binBuf = ctx.upload("bins", built.features === "mean" ? new Uint32Array(rows) : bins);
        ctx.dispatch1d("gather", K.EMBED_GATHER, [binBuf, table, distogram], [rows, c], rows * c);
      }
      const prefix = "pairwise_conditioner";
      this.layerNorm(root, `${prefix}/dim_pairwise_init_proj/0`, z, s.p2, rows, c, distogram);
      const update = ctx.buffer("conditioner", rows * c);
      this.matmul(root, s.p2, `${prefix}/dim_pairwise_init_proj/1/weight`, update, rows, 2 * c, c);
      ctx.flush();
      for (let t = 0; t < config.conditionerTransitions; t++) {
        this.transition(root, `${prefix}/transitions/${t}`, update, s.x, u, h, rows, c, 2 * c);
      }
      await tap("conditioner_update", update, rows * c);
      ctx.dispatch1d("add", K.ADD, [z, update], [rows * c], rows * c);
    } else if (taps) {
      taps.set("conditioner_update", new Float32Array(rows * c));
    }
    ctx.flush();
    await tapPool("pooled_conditioned");

    for (let layer = 0; layer < config.pairformerBlocks; layer++) {
      const p = `pairformer_stack/layers/${layer}`;
      this.triangleMultiplication(root, `${p}/tri_mul_out`, s, true);
      this.triangleMultiplication(root, `${p}/tri_mul_in`, s, false);
      this.triangleAttention(root, `${p}/tri_att_start`, s, true, heads, width);
      this.triangleAttention(root, `${p}/tri_att_end`, s, false, heads, width);
      this.transition(root, `${p}/transition_z`, z, s.x, u, h, rows, c, 4 * c);
      if (taps && layer < (taps.layers ?? 0)) await tap(`layer${layer}`, z, rows * c);
    }
    await tap("pairformer_out", z, rows * c);

    // affinity heads
    this.pool(z, poolMask, rows, c, pooled);
    await tap("pooled_pairformer", pooled, c);
    const a = ctx.buffer("head.a", headsTokenS);
    const b = ctx.buffer("head.b", headsTokenS);
    const g = ctx.buffer("head.g", headsTokenS);
    const hh = "affinity_heads";
    this.matmul(root, pooled, `${hh}/affinity_out_mlp/0/weight`, a, 1, c, c, 1, null, `${hh}/affinity_out_mlp/0/bias`);
    this.matmul(root, a, `${hh}/affinity_out_mlp/2/weight`, g, 1, c, headsTokenS, 1, null, `${hh}/affinity_out_mlp/2/bias`);
    await tap("pooled_hidden", g, headsTokenS);
    const head = (name, out) => {
      this.matmul(root, g, `${hh}/${name}/0/weight`, a, 1, headsTokenS, headsTokenS, 1, null, `${hh}/${name}/0/bias`);
      this.matmul(root, a, `${hh}/${name}/2/weight`, b, 1, headsTokenS, headsTokenS, 1, null, `${hh}/${name}/2/bias`);
      this.matmul(root, b, `${hh}/${name}/4/weight`, out, 1, headsTokenS, 1, 0, null, `${hh}/${name}/4/bias`);
    };
    const value = ctx.buffer("value", 1);
    const score = ctx.buffer("score", 1);
    const logit = ctx.buffer("logit", 1);
    head("to_affinity_pred_value", value);
    head("to_affinity_pred_score", score);
    this.matmul(root, score, `${hh}/to_affinity_logits_binary/weight`, logit, 1, 1, 1, 0, null, `${hh}/to_affinity_logits_binary/bias`);
    const out = { value: (await ctx.read(value, 1))[0], logit: (await ctx.read(logit, 1))[0] };
    out.probability = 1 / (1 + Math.exp(-out.logit));
    taps?.set("affinity_pred_value", Float32Array.of(out.value));
    taps?.set("affinity_logits_binary", Float32Array.of(out.logit));
    await ctx.release();
    return out;
  }

  /** Ensemble of the members present in the pack (Boltz2._run_affinity_mode). */
  async score(input, options = {}) {
    const roots = ["affinity1", "affinity2"].filter((root) => this.bundle.meta.modules?.[root]);
    const members = [];
    for (const root of roots) {
      members.push(await this.runModule(root, input, options.taps ? { ...options, taps: options.taps(root) } : options));
    }
    const value = members.reduce((acc, m) => acc + m.value, 0) / members.length;
    const probability = members.reduce((acc, m) => acc + m.probability, 0) / members.length;
    return { value, probability, members };
  }

  async destroy() {
    await this.ctx.release();
    for (const buffer of this.weights.values()) buffer.destroy();
    this.dummy.destroy();
    this.weights.clear();
  }
}
