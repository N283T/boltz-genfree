// Own WebGPU trunk PairformerModule against the PyTorch oracle dumps. GPU tests are gated by
// BOLTZ_GENFREE_GPU_TESTS=1 and skip visibly without a GPU, the trunk pack or the dumps.
import { after, test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { compare } from "../src/compare.js";
import { createNodeDevice } from "../src/device.js";
import { OwnWeights } from "../src/trunk/own-weights.js";
import { Pairformer } from "../src/trunk/own/pairformer.js";

const here = (p) => new URL(`../${p}`, import.meta.url).pathname;
const PACK = here("packs/boltz2-trunk-own/boltz2-trunk-own.json");
const LAYERS = [0, 1, 2, 7, 31, 63];
// relRMS tolerance (task targets: 1e-4 at layer 0, 2e-3 at layer 63; measured errors are 100x lower).
const tolFor = (layer) => (layer === 0 ? 1e-5 : 1e-4); // measured: ~4e-8 at layer 0, ~1e-6 at layer 63
const CASES = ["phenol-nomsa", "acetaminophen-nomsa", "phenol-msa", "big200-nomsa"];

const enabled = process.env.BOLTZ_GENFREE_GPU_TESTS === "1";
let gpu;
if (enabled) { try { gpu = await createNodeDevice(); } catch (error) { gpu = { error }; } }
after(() => gpu?.device?.destroy());

let weights;
for (const name of CASES) {
  const dumpPath = here(`oracle-dumps/${name}-chain.json`);
  const skip = !enabled ? "set BOLTZ_GENFREE_GPU_TESTS=1"
    : gpu.error ? `no WebGPU: ${gpu.error.message}`
      : !existsSync(PACK) ? "trunk pack missing (tools/convert_boltz_trunk.py)"
        : !existsSync(dumpPath) && "oracle dump missing";
  test(`own pairformer vs oracle (${name})`, { skip, timeout: 600000 }, async (t) => {
    const dump = await TensorBundle.open(dumpPath);
    if (!dump.has("trunk/pairformer_in_z")) { t.skip("dump predates pairformer taps"); return; }
    weights ??= await OwnWeights.open(PACK);
    const s = dump.get("trunk/pairformer_in_s");
    const z = dump.get("trunk/pairformer_in_z");
    const mask = Float32Array.from(dump.get("feat/token_pad_mask"));
    const n = mask.length;
    const pf = new Pairformer(gpu.device, weights);
    const taps = new Map();
    const timings = [];
    const out = await pf.run(s, z, mask, { tapLayers: LAYERS, taps, timings });
    const finalZ = await pf.ctx.read(out.z, n * n * 128);
    const finalS = await pf.ctx.read(out.s, n * 384);
    out.release();
    pf.destroy();
    await pf.ctx.release();
    const sorted = [...timings].sort((a, b) => a - b);
    console.log(`  ${name} n=${n}: ${sorted[sorted.length >> 1].toFixed(0)} ms/layer (median), total ${(timings.reduce((a, b) => a + b, 0) / 1000).toFixed(1)} s`);
    const check = (label, actual, key, tol) => {
      const st = compare(actual, dump.get(key));
      console.log(`  ${name} ${label}: relRMS ${st.relRms.toExponential(2)} maxAbs ${st.maxAbs.toExponential(2)}`);
      assert.ok(st.relRms <= tol, `${key}: relRMS ${st.relRms} > ${tol}`);
    };
    for (const l of LAYERS) {
      if (dump.has(`trunk/pairformer_layer${l}_z`)) check(`layer${l}_z`, taps.get(`layer${l}_z`), `trunk/pairformer_layer${l}_z`, tolFor(l));
      if (dump.has(`trunk/pairformer_layer${l}_s`)) check(`layer${l}_s`, taps.get(`layer${l}_s`), `trunk/pairformer_layer${l}_s`, tolFor(l));
    }
    check("final z", finalZ, "trunk/z", 1e-4);
    check("final s", finalS, "trunk/s", 1e-4);
  });
}
