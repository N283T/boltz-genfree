// Own Boltz-2 MSAModule (WGSL f32) against the PyTorch trunk oracle dumps.
// Needs a WebGPU adapter plus oracle-dumps/*-chain.* and packs/boltz2-trunk-own; skips visibly otherwise.
// Run: BOLTZ_GENFREE_GPU_TESTS=1 node --test test/own-msa-module.test.js
// BOLTZ_GENFREE_MSA_STRESS=1 additionally runs a synthetic 8192-row case (timing only, no oracle).
import { after, test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { compare } from "../src/compare.js";
import { createNodeDevice } from "../src/device.js";
import { GpuContext } from "../src/gpu/context.js";
import { OwnWeights } from "../src/trunk/own-weights.js";
import { MsaModule } from "../src/trunk/own/msa-module.js";

const here = (p) => new URL(`../${p}`, import.meta.url).pathname;
const PACK = here("packs/boltz2-trunk-own/boltz2-trunk-own.json");
const TOL = 1e-4; // relative RMS, f32 against the PyTorch f32 oracle
const CASES = [
  { name: "phenol-msa", dump: "oracle-dumps/phenol-msa-chain.json", rows: 75 },
  { name: "phenol-nomsa", dump: "oracle-dumps/phenol-nomsa-chain.json", rows: 1 },
  { name: "acetaminophen-nomsa", dump: "oracle-dumps/acetaminophen-nomsa-chain.json", rows: 1 },
];
const STAGES = (blocks) => ["msa_embed_m", ...Array.from({ length: blocks }, (_, b) =>
  ["m_after_pwa", "z_after_opm", "m_after_trans", "z_after_tri", "z_out"].map((s) => `msa_block${b}_${s}`)).flat(),
"z_msa_module_out"];

const enabled = process.env.BOLTZ_GENFREE_GPU_TESTS === "1";
let gpu;
if (enabled) { try { gpu = await createNodeDevice(); } catch (error) { gpu = { error }; } }
after(() => gpu?.device?.destroy());

// The oracle ran with subsample_msa (torch.randperm), so its MSA rows are a permutation of the feature rows. Row order
// cannot change z (the outer-product mean sums over rows); m is compared after undoing the permutation, which is
// recovered by exact matching of the embedding rows (the embedding itself is checked after the same alignment).
const rowPermutation = (mine, ref, rows, width) => {
  const perm = new Int32Array(rows).fill(-1);
  const taken = new Set();
  for (let r = 0; r < rows; r++) {
    let best = -1;
    let bestErr = Infinity;
    for (let s = 0; s < rows; s++) {
      if (taken.has(s)) continue;
      let err = 0;
      for (let k = 0; k < width; k += 5) { const d = ref[r * width + k] - mine[s * width + k]; err += d * d; }
      if (err < bestErr) { bestErr = err; best = s; }
      if (err === 0) break;
    }
    perm[r] = best;
    taken.add(best);
  }
  return perm;
};
const permuteRows = (a, perm, width) => {
  const out = new Float32Array(a.length);
  perm.forEach((s, r) => out.set(a.subarray(s * width, (s + 1) * width), r * width));
  return out;
};

const load = async (c) => {
  const dump = await TensorBundle.open(here(c.dump));
  const features = {
    msa: dump.get("feat/msa"), msaPaired: dump.get("feat/msa_paired"), hasDeletion: dump.get("feat/has_deletion"),
    deletionValue: dump.get("feat/deletion_value"), msaMask: dump.get("feat/msa_mask"),
  };
  return { dump, features };
};

for (const c of CASES) {
  const skip = !enabled ? "set BOLTZ_GENFREE_GPU_TESTS=1" : gpu.error ? `no WebGPU: ${gpu.error.message}`
    : !(existsSync(here(c.dump)) && existsSync(PACK)) && "oracle dump or trunk pack missing";
  test(`own MSAModule vs oracle (${c.name}) - every stage relRMS <= ${TOL}`, { skip }, async () => {
    const { dump, features } = await load(c);
    const weights = await OwnWeights.open(PACK);
    const ctx = new GpuContext(gpu.device);
    const module = new MsaModule(ctx, weights);
    const taps = new Map();
    const t0 = performance.now();
    const out = await module.run(dump.get("trunk/msa_in_z"), dump.get("trunk/s_inputs_trunk"), features,
      { tokenPadMask: dump.get("feat/token_pad_mask") }, { taps });
    const seconds = ((performance.now() - t0) / 1000).toFixed(2);
    const n = dump.get("feat/token_pad_mask").length;
    console.log(`  ${c.name}: ${n} tokens x ${features.msa.length / n} rows, ${seconds} s`);
    assert.equal(features.msa.length / n, c.rows);
    const failures = [];
    const rowWidth = n * 64;
    const perm = rowPermutation(taps.get("msa_embed_m"), dump.get("trunk/msa_embed_m"), c.rows, rowWidth);
    if (perm.some((s, r) => s !== r)) console.log(`  ${c.name}: oracle MSA rows are permuted (subsample_msa); aligning m stages`);
    for (const stage of STAGES(weights.arch.msaBlocks ?? 4)) {
      const mine = taps.get(stage);
      const stat = compare(stage.includes("_m_") || stage === "msa_embed_m" ? permuteRows(mine, perm, rowWidth) : mine,
        dump.get(`trunk/${stage}`));
      console.log(`  ${c.name} ${stage}: relRMS ${stat.relRms.toExponential(2)} maxAbs ${stat.maxAbs.toExponential(2)}`);
      if (!(stat.relRms <= TOL)) failures.push(`${stage}: ${stat.relRms}`);
    }
    assert.deepEqual(failures, []);
    assert.ok(compare(out, dump.get("trunk/z_msa_module_out")).relRms <= TOL);
    // Boltz2.forward adds the module output to its input: z = z + msa_module(z, ...).
    if (dump.has("trunk/pairformer_in_z")) {
      const z = dump.get("trunk/msa_in_z");
      const sum = Float32Array.from(out, (v, i) => v + z[i]);
      const stat = compare(sum, dump.get("trunk/pairformer_in_z"));
      console.log(`  ${c.name} pairformer_in_z = z_in + module_out: relRMS ${stat.relRms.toExponential(2)}`);
      assert.ok(stat.relRms <= TOL);
    }
    await ctx.release();
  });
}

test("own MSAModule synthetic depth (timing, finite output)", {
  skip: !enabled ? "set BOLTZ_GENFREE_GPU_TESTS=1" : gpu.error ? `no WebGPU: ${gpu.error.message}`
    : process.env.BOLTZ_GENFREE_MSA_STRESS !== "1" ? "set BOLTZ_GENFREE_MSA_STRESS=1" : !existsSync(PACK) && "trunk pack missing",
}, async () => {
  const c = CASES[0];
  const { dump, features } = await load(c);
  const n = dump.get("feat/token_pad_mask").length;
  const base = features.msa.length / n;
  const rows = Number(process.env.BOLTZ_GENFREE_MSA_ROWS ?? 8192);
  const tile = (a) => { const o = new Float32Array(rows * n); for (let r = 0; r < rows; r++) o.set(a.subarray((r % base) * n, (r % base + 1) * n), r * n); return o; };
  const big = Object.fromEntries(Object.entries(features).map(([k, v]) => [k, tile(v)]));
  const ctx = new GpuContext(gpu.device);
  const module = new MsaModule(ctx, await OwnWeights.open(PACK));
  const t0 = performance.now();
  const out = await module.run(dump.get("trunk/msa_in_z"), dump.get("trunk/s_inputs_trunk"), big, { tokenPadMask: dump.get("feat/token_pad_mask") });
  console.log(`  synthetic ${rows} rows x ${n} tokens: ${((performance.now() - t0) / 1000).toFixed(1)} s`);
  assert.ok(out.every(Number.isFinite));
  await ctx.release();
});
