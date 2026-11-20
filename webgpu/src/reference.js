// CPU reference of the Boltz-2 AffinityModule, structure-free path.
//
// Written independently of the WebGPU kernels so the two can be compared, and
// against the PyTorch oracle (webgpu/tools/dump_oracle.py). Storage is float32,
// accumulation is float64. Layouts are PyTorch's: pair tensors are [N, N, C]
// row-major, linear weights are [out, in].
import { affinityMasks, buildStructureFreeBins } from "./distogram.js";
import { boundariesOf, moduleConfig, weightsFor } from "./module-config.js";

const EPS = 1e-5;
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const silu = (x) => x * sigmoid(x);

function layerNorm(x, rows, c, weight, bias) {
  const out = new Float32Array(x.length);
  for (let r = 0; r < rows; r++) {
    let mean = 0;
    for (let i = 0; i < c; i++) mean += x[r * c + i];
    mean /= c;
    let variance = 0;
    for (let i = 0; i < c; i++) variance += (x[r * c + i] - mean) ** 2;
    variance /= c;
    const inv = 1 / Math.sqrt(variance + EPS);
    for (let i = 0; i < c; i++) {
      out[r * c + i] = (x[r * c + i] - mean) * inv * (weight ? weight[i] : 1) + (bias ? bias[i] : 0);
    }
  }
  return out;
}

/** y[r, o] = sum_i x[r, i] * w[o, i] (+ b[o]) */
function linear(x, rows, inDim, w, outDim, b) {
  const out = new Float32Array(rows * outDim);
  for (let r = 0; r < rows; r++) {
    for (let o = 0; o < outDim; o++) {
      let acc = b ? b[o] : 0;
      for (let i = 0; i < inDim; i++) acc += x[r * inDim + i] * w[o * inDim + i];
      out[r * outDim + o] = acc;
    }
  }
  return out;
}

function transition(z, rows, c, hidden, w, prefix) {
  const x = layerNorm(z, rows, c, w(`${prefix}/norm/weight`), w(`${prefix}/norm/bias`));
  const a = linear(x, rows, c, w(`${prefix}/fc1/weight`), hidden);
  const b = linear(x, rows, c, w(`${prefix}/fc2/weight`), hidden);
  for (let i = 0; i < a.length; i++) a[i] = silu(a[i]) * b[i];
  return linear(a, rows, hidden, w(`${prefix}/fc3/weight`), c);
}

function triangleMultiplication(z, n, c, mask, w, prefix, outgoing) {
  const rows = n * n;
  const x = layerNorm(z, rows, c, w(`${prefix}/norm_in/weight`), w(`${prefix}/norm_in/bias`));
  const p = linear(x, rows, c, w(`${prefix}/p_in/weight`), 2 * c);
  const g = linear(x, rows, c, w(`${prefix}/g_in/weight`), 2 * c);
  for (let i = 0; i < p.length; i++) p[i] = p[i] * sigmoid(g[i]) * mask[Math.floor(i / (2 * c))];
  const hidden = new Float32Array(rows * c);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      for (let d = 0; d < c; d++) {
        let acc = 0;
        for (let k = 0; k < n; k++) {
          const a = outgoing ? p[(i * n + k) * 2 * c + d] : p[(k * n + i) * 2 * c + d];
          const b = outgoing ? p[(j * n + k) * 2 * c + c + d] : p[(k * n + j) * 2 * c + c + d];
          acc += a * b;
        }
        hidden[(i * n + j) * c + d] = acc;
      }
    }
  }
  const normed = layerNorm(hidden, rows, c, w(`${prefix}/norm_out/weight`), w(`${prefix}/norm_out/bias`));
  const out = linear(normed, rows, c, w(`${prefix}/p_out/weight`), c);
  const gate = linear(x, rows, c, w(`${prefix}/g_out/weight`), c);
  for (let i = 0; i < out.length; i++) out[i] *= sigmoid(gate[i]);
  return out;
}

function triangleAttention(z, n, c, heads, width, mask, w, prefix, starting) {
  const rows = n * n;
  const at = (i, j) => (starting ? i * n + j : j * n + i); // pair index of the (possibly transposed) view
  const x = layerNorm(z, rows, c, w(`${prefix}/layer_norm/weight`), w(`${prefix}/layer_norm/bias`));
  const bias = linear(x, rows, c, w(`${prefix}/linear/weight`), heads);
  const q = linear(x, rows, c, w(`${prefix}/mha/linear_q/weight`), heads * width);
  const k = linear(x, rows, c, w(`${prefix}/mha/linear_k/weight`), heads * width);
  const v = linear(x, rows, c, w(`${prefix}/mha/linear_v/weight`), heads * width);
  const g = linear(x, rows, c, w(`${prefix}/mha/linear_g/weight`), heads * width);
  const o = new Float32Array(rows * heads * width);
  const scale = 1 / Math.sqrt(width);
  const logits = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    for (let h = 0; h < heads; h++) {
      for (let j = 0; j < n; j++) {
        const pq = at(i, j);
        let max = -Infinity;
        for (let kk = 0; kk < n; kk++) {
          const pk = at(i, kk);
          let dot = 0;
          for (let d = 0; d < width; d++) dot += q[pq * heads * width + h * width + d] * scale * k[pk * heads * width + h * width + d];
          const maskBias = 1e9 * (mask[at(i, kk)] - 1);
          logits[kk] = Math.fround(Math.fround(dot + maskBias) + bias[at(j, kk) * heads + h]);
          if (logits[kk] > max) max = logits[kk];
        }
        let sum = 0;
        for (let kk = 0; kk < n; kk++) { logits[kk] = Math.exp(logits[kk] - max); sum += logits[kk]; }
        for (let d = 0; d < width; d++) {
          let acc = 0;
          for (let kk = 0; kk < n; kk++) acc += (logits[kk] / sum) * v[at(i, kk) * heads * width + h * width + d];
          o[pq * heads * width + h * width + d] = acc * sigmoid(g[pq * heads * width + h * width + d]);
        }
      }
    }
  }
  return linear(o, rows, heads * width, w(`${prefix}/mha/linear_o/weight`), c);
}

function addInPlace(a, b) { for (let i = 0; i < a.length; i++) a[i] += b[i]; }

/** One PairformerNoSeq layer, updating `z` in place; mirrors PairformerNoSeqLayer.forward. */
export function pairformerLayer(z, n, config, mask, w, prefix) {
  const c = config.tokenZ;
  const { triHeads: heads, triHeadWidth: width } = config;
  addInPlace(z, triangleMultiplication(z, n, c, mask, w, `${prefix}/tri_mul_out`, true));
  addInPlace(z, triangleMultiplication(z, n, c, mask, w, `${prefix}/tri_mul_in`, false));
  addInPlace(z, triangleAttention(z, n, c, heads, width, mask, w, `${prefix}/tri_att_start`, true));
  addInPlace(z, triangleAttention(z, n, c, heads, width, mask, w, `${prefix}/tri_att_end`, false));
  addInPlace(z, transition(z, n * n, c, 4 * c, w, `${prefix}/transition_z`));
}

export function pool(z, n, c, poolMask) {
  const out = new Float64Array(c);
  let total = 0;
  for (let p = 0; p < n * n; p++) {
    total += poolMask[p];
    if (poolMask[p] === 0) continue;
    for (let d = 0; d < c; d++) out[d] += z[p * c + d] * poolMask[p];
  }
  return Float32Array.from(out, (value) => value / (total + 1e-7));
}

/**
 * Run one AffinityModule. `trace` (optional Map) receives intermediates under the
 * oracle's names (without the `affinityN/mode/` prefix).
 */
export function runAffinityModule(bundle, root, input, options = {}) {
  const { mode = "zero_distogram_features", seed = 0, trace, bins: givenBins } = options;
  const config = moduleConfig(bundle, root);
  const w = weightsFor(bundle, root);
  const { tokenZ: c, tokenS: sDim, headsTokenS } = config;
  const { sInputs, zAffinity, padMask, molType, affinityTokenMask } = input;
  const n = padMask.length;
  const rows = n * n;
  const masks = affinityMasks(padMask, molType, affinityTokenMask);
  const boundaries = boundariesOf(config);
  const built = buildStructureFreeBins({
    boundaries, receptor: masks.receptor, ligand: masks.ligand, mode, seed,
  });
  const bins = givenBins ?? built.bins;
  const embedWeight = w("dist_bin_pairwise_embed/weight"); // [numBins, c]

  // z = z_linear(z_norm(z)) + s_to_z_prod_in1(s)[:, None] + s_to_z_prod_in2(s)[None]
  let z = layerNorm(zAffinity, rows, c, w("z_norm/weight"), w("z_norm/bias"));
  z = linear(z, rows, c, w("z_linear/weight"), c);
  trace?.set("z_linear", Float32Array.from(z));
  const s1 = linear(sInputs, n, sDim, w("s_to_z_prod_in1/weight"), c);
  const s2 = linear(sInputs, n, sDim, w("s_to_z_prod_in2/weight"), c);
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let d = 0; d < c; d++) z[(i * n + j) * c + d] += s1[i * c + d] + s2[j * c + d];
  trace?.set("pooled_processed", pool(z, n, c, masks.poolMask));

  // distogram features
  const distogram = new Float32Array(rows * c);
  if (built.features === "embed") {
    for (let p = 0; p < rows; p++) distogram.set(embedWeight.subarray(bins[p] * c, bins[p] * c + c), p * c);
  } else if (built.features === "mean") {
    const bins_ = config.numDistBins;
    const mean = new Float32Array(c);
    for (let b = 0; b < bins_; b++) for (let d = 0; d < c; d++) mean[d] += embedWeight[b * c + d] / bins_;
    for (let p = 0; p < rows; p++) distogram.set(mean, p * c);
  }

  // pairwise conditioner
  if (built.conditioning) {
    const cat = new Float32Array(rows * 2 * c);
    for (let p = 0; p < rows; p++) {
      cat.set(z.subarray(p * c, p * c + c), p * 2 * c);
      cat.set(distogram.subarray(p * c, p * c + c), p * 2 * c + c);
    }
    const normed = layerNorm(cat, rows, 2 * c, w("pairwise_conditioner/dim_pairwise_init_proj/0/weight"), w("pairwise_conditioner/dim_pairwise_init_proj/0/bias"));
    let update = linear(normed, rows, 2 * c, w("pairwise_conditioner/dim_pairwise_init_proj/1/weight"), c);
    for (let t = 0; t < config.conditionerTransitions; t++) {
      const delta = transition(update, rows, c, 2 * c, w, `pairwise_conditioner/transitions/${t}`);
      addInPlace(update, delta);
    }
    trace?.set("conditioner_update", Float32Array.from(update));
    addInPlace(z, update);
  } else {
    trace?.set("conditioner_update", new Float32Array(rows * c));
  }
  trace?.set("pooled_conditioned", pool(z, n, c, masks.poolMask));

  for (let layer = 0; layer < config.pairformerBlocks; layer++) {
    pairformerLayer(z, n, config, masks.crossPairMask, w, `pairformer_stack/layers/${layer}`);
    trace?.set(`layer${layer}`, Float32Array.from(z));
  }
  trace?.set("pairformer_out", Float32Array.from(z));

  // affinity heads
  const pooled = pool(z, n, c, masks.poolMask);
  trace?.set("pooled_pairformer", pooled);
  let g = linear(pooled, 1, c, w("affinity_heads/affinity_out_mlp/0/weight"), c, w("affinity_heads/affinity_out_mlp/0/bias"));
  g = g.map((x) => Math.max(0, x));
  g = linear(g, 1, c, w("affinity_heads/affinity_out_mlp/2/weight"), headsTokenS, w("affinity_heads/affinity_out_mlp/2/bias"));
  g = g.map((x) => Math.max(0, x));
  trace?.set("pooled_hidden", g);
  const head = (name) => {
    let h = linear(g, 1, headsTokenS, w(`affinity_heads/${name}/0/weight`), headsTokenS, w(`affinity_heads/${name}/0/bias`)).map((x) => Math.max(0, x));
    h = linear(h, 1, headsTokenS, w(`affinity_heads/${name}/2/weight`), headsTokenS, w(`affinity_heads/${name}/2/bias`)).map((x) => Math.max(0, x));
    return linear(h, 1, headsTokenS, w(`affinity_heads/${name}/4/weight`), 1, w(`affinity_heads/${name}/4/bias`))[0];
  };
  const value = head("to_affinity_pred_value");
  const score = head("to_affinity_pred_score");
  const logit = w("affinity_heads/to_affinity_logits_binary/weight")[0] * score + w("affinity_heads/to_affinity_logits_binary/bias")[0];
  trace?.set("affinity_pred_value", Float32Array.of(value));
  trace?.set("affinity_logits_binary", Float32Array.of(logit));
  trace?.set("affinity_distogram_bins", Float32Array.from(bins));
  return { value, logit, probability: sigmoid(logit) };
}

/** Ensemble of the two members, as Boltz2._run_affinity_mode. */
export function scoreEnsemble(bundle, input, options = {}) {
  const roots = ["affinity1", "affinity2"].filter((root) => bundle.meta.modules?.[root]);
  const members = roots.map((root) => runAffinityModule(bundle, root, input, options));
  const value = members.reduce((a, m) => a + m.value, 0) / members.length;
  const probability = members.reduce((a, m) => a + m.probability, 0) / members.length;
  return { value, probability, members };
}
