// Own WGSL InputEmbedder / initial / recycling embeddings against the PyTorch oracle dumps, stage by stage.
// Inputs are the oracle's `feat/*` tensors (not our featuriser), so errors localise to the embedder.
// Runs only with BOLTZ_GENFREE_GPU_TESTS=1; skips visibly without a WebGPU adapter, the trunk pack or the dumps.
import { after, test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { compare } from "../src/compare.js";
import { createNodeDevice } from "../src/device.js";
import { GpuContext } from "../src/gpu/context.js";
import { OwnWeights } from "../src/trunk/own-weights.js";
import { Embedder } from "../src/trunk/own/embedder.js";

const here = (p) => new URL(`../${p}`, import.meta.url).pathname;
const PACK = here("packs/boltz2-trunk-own/boltz2-trunk-own.json");
const CASES = ["phenol-nomsa", "acetaminophen-nomsa", "phenol-msa"];
const TOL = 1e-4; // relative RMS, f32 against the PyTorch f32 oracle

const enabled = process.env.BOLTZ_GENFREE_GPU_TESTS === "1";
let gpu;
if (enabled) { try { gpu = await createNodeDevice(); } catch (error) { gpu = { error }; } }
after(() => gpu?.device?.destroy());

let weights;

for (const name of CASES) {
  const dump = here(`oracle-dumps/${name}-chain.json`);
  const skip = !enabled ? "set BOLTZ_GENFREE_GPU_TESTS=1 to run GPU tests"
    : gpu.error ? `no WebGPU: ${gpu.error.message}`
      : !existsSync(PACK) ? "trunk pack missing (packs/boltz2-trunk-own)"
        : !existsSync(dump) && `oracle dump missing: ${name}`;
  test(`own embedder vs oracle (${name}) - stage relRMS <= ${TOL}`, { skip }, async () => {
    weights ??= await OwnWeights.open(PACK);
    const oracle = await TensorBundle.open(dump);
    const ctx = new GpuContext(gpu.device);
    const embedder = new Embedder(ctx, weights);
    const taps = new Map();
    const n = oracle.shape("feat/token_pad_mask")[1];
    const results = [];
    const check = async (stage, buffer, floats, key = `trunk/${stage}`) => {
      if (!oracle.has(key)) return;
      const actual = await ctx.read(buffer, floats);
      const stat = compare(actual, oracle.get(key));
      results.push({ stage, ...stat });
    };
    try {
      const sIn = embedder.sInputs(oracle, { taps });
      const { sInit, zInit } = embedder.initial(oracle, sIn, { taps });
      const { s, z } = embedder.recycle(sInit, zInit, null, null, { taps });
      const sInAff = embedder.sInputs(oracle, { affinity: true });
      for (const [stage, { buffer, floats }] of taps) await check(stage, buffer, floats);
      await check("s_inputs_trunk", sIn, n * 384);
      await check("s_inputs_affinity", sInAff, n * 384);
      await check("pairformer_in_s", s, n * 384);

      // Recycling with a non-zero previous pass must equal the same formula on the CPU.
      const prevS = ctx.upload("test.prevS", oracle.get("trunk/s_init"));
      const prevZ = ctx.upload("test.prevZ", oracle.get("trunk/z_init"));
      const again = embedder.recycle(sInit, zInit, prevS, prevZ);
      const sNext = await ctx.read(again.s, n * 384);
      const zNext = await ctx.read(again.z, 64 * 128);
      const ln = (v, w, b) => {
        const mean = v.reduce((a, c) => a + c, 0) / v.length;
        const variance = v.reduce((a, c) => a + (c - mean) ** 2, 0) / v.length;
        return v.map((c, i) => (c - mean) / Math.sqrt(variance + 1e-5) * w[i] + b[i]);
      };
      const project = (w, v, out) => Array.from({ length: out }, (_, o) => v.reduce((a, c, i) => a + c * w[o * v.length + i], 0));
      const sRef = new Float32Array(n * 384);
      for (let t = 0; t < n; t += 1) {
        const normed = ln(Array.from(oracle.get("trunk/s_init").subarray(t * 384, t * 384 + 384)), weights.get("s_norm/weight"), weights.get("s_norm/bias"));
        const proj = project(weights.get("s_recycle/weight"), normed, 384);
        for (let d = 0; d < 384; d += 1) sRef[t * 384 + d] = oracle.get("trunk/s_init")[t * 384 + d] + proj[d];
      }
      const zRef = new Float32Array(64 * 128);
      for (let q = 0; q < 64; q += 1) {
        const row = Array.from(oracle.get("trunk/z_init").subarray(q * 128, q * 128 + 128));
        const proj = project(weights.get("z_recycle/weight"), ln(row, weights.get("z_norm/weight"), weights.get("z_norm/bias")), 128);
        for (let d = 0; d < 128; d += 1) zRef[q * 128 + d] = row[d] + proj[d];
      }
      const sStat = compare(sNext, sRef);
      const zStat = compare(zNext, zRef);
      results.push({ stage: "recycle(prev=s_init/z_init) vs CPU s", ...sStat }, { stage: "recycle(prev=s_init/z_init) vs CPU z[:64]", ...zStat });

      console.log(`  ${name}: ${n} tokens`);
      for (const r of results) console.log(`    ${r.stage.padEnd(28)} relRMS ${r.relRms.toExponential(2)}  maxAbs ${r.maxAbs.toExponential(2)}`);
      assert.ok(results.length >= 18, `expected at least 18 checked stages, got ${results.length}`);
      const bad = results.filter((r) => !(r.relRms <= TOL));
      assert.equal(bad.length, 0, `stages over ${TOL}: ${bad.map((r) => `${r.stage} ${r.relRms.toExponential(2)}`).join(", ")}`);
    } finally {
      await ctx.release();
      embedder.destroy();
    }
  });
}
