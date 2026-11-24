// The from-scratch chain (own featuriser -> own embedder -> recycling -> own MSA module -> own pairformer -> affinity
// module) against GenFree's PyTorch run.
// Needs BOLTZ_GENFREE_GPU_TESTS=1, a WebGPU adapter, packs/boltz2-trunk-own, packs/genfree-affinity and the oracle dumps
// (tools/dump_trunk_oracle.py); skips visibly otherwise.
//   exact      : the oracle's protein reference positions / atom-name one-hots are imported (engine parity)
//   standalone : everything from the own featuriser (Boltz's random reference-conformer augmentation is not reproduced)
import { after, test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { compare } from "../src/compare.js";
import { createNodeDevice } from "../src/device.js";
import { createOwnChain } from "../src/trunk/own/chain.js";

const here = (p) => new URL(`../${p}`, import.meta.url).pathname;
const TRUNK = here("packs/boltz2-trunk-own/boltz2-trunk-own.json");
const AFFINITY = here("packs/genfree-affinity.json");
// Tolerances set well above the measured error (docs/webgpu.md).
const CASES = [
  { name: "phenol-nomsa", exact: { z: 2e-3, value: 3e-3 }, js: { value: 0.06 } },
  { name: "acetaminophen-nomsa", exact: { z: 2e-3, value: 3e-3 }, js: { value: 0.06 } },
  { name: "salicylic-nomsa", exact: { z: 2e-3, value: 3e-3 }, js: { value: 0.06 } },
  { name: "phenol-msa", exact: { z: 3e-2, value: 2e-2 }, js: { value: 0.15 } },
];

const enabled = process.env.BOLTZ_GENFREE_GPU_TESTS === "1";
let gpu;
if (enabled) { try { gpu = await createNodeDevice(); } catch (error) { gpu = { error }; } }
const packsMissing = !existsSync(TRUNK) || !existsSync(AFFINITY);
const reason = (dump) => (!enabled ? "set BOLTZ_GENFREE_GPU_TESTS=1 to run GPU tests"
  : gpu.error ? `no WebGPU: ${gpu.error.message}`
    : packsMissing ? "packs missing (packs/boltz2-trunk-own, packs/genfree-affinity)"
      : dump && !existsSync(dump) ? `oracle dump missing: ${dump}` : undefined);

let chain;
after(async () => { await chain?.destroy(); gpu?.device?.destroy(); });
const getChain = async () => (chain ??= await createOwnChain({ trunkPack: TRUNK, affinityPack: AFFINITY, device: gpu.device }));
const inputOf = (oracle) => ({
  sequence: oracle.meta.sequence, smiles: oracle.meta.smiles,
  a3m: oracle.meta.msa ? readFileSync(new URL(`../../${oracle.meta.msa}`, import.meta.url), "utf8") : null,
});
const refValue = (oracle) => (oracle.has("final/zero_distogram_features/affinity_pred_value")
  ? oracle.get("final/zero_distogram_features/affinity_pred_value")[0] : oracle.meta.affinity_pred_value);
const seconds = (ms) => (ms / 1000).toFixed(1);

for (const c of CASES) {
  const dump = here(`oracle-dumps/${c.name}-chain.json`);
  test(`own chain vs PyTorch (${c.name})`, { skip: reason(dump), timeout: 900000 }, async () => {
    const oracle = await TensorBundle.open(dump);
    const engine = await getChain();
    const input = inputOf(oracle);
    const ref = refValue(oracle);
    const exact = await engine.score(input, { oracle });
    const z = compare(exact.trunk.pair, oracle.get("trunk/z")).relRms;
    const s = compare(exact.trunk.single, oracle.get("trunk/s")).relRms;
    const standalone = await engine.score(input, {});
    console.log(`  ${c.name} (${exact.tokens} tokens, ${exact.msaRows} MSA rows): exact z relRMS ${z.toExponential(2)} s ${s.toExponential(2)}, `
      + `value ${exact.value.toFixed(5)} vs ${ref.toFixed(5)} (diff ${(exact.value - ref).toExponential(2)}); `
      + `standalone value ${standalone.value.toFixed(5)} (diff ${(standalone.value - ref).toFixed(4)}); `
      + `wall ${seconds(exact.timings.totalMs)} s exact / ${seconds(standalone.timings.totalMs)} s standalone`);
    assert.ok(z <= c.exact.z, `trunk z relRMS ${z}`);
    assert.ok(Math.abs(exact.value - ref) <= c.exact.value, `exact-feature value ${exact.value} vs ${ref}`);
    assert.ok(Math.abs(standalone.value - ref) <= c.js.value, `standalone value ${standalone.value} vs ${ref}`);
  });
}

test("own chain, standalone, 211 tokens (big200-nomsa)", { skip: reason(here("oracle-dumps/big200-nomsa-chain.json")), timeout: 900000 }, async () => {
  const oracle = await TensorBundle.open(here("oracle-dumps/big200-nomsa-chain.json"));
  const engine = await getChain();
  const ref = refValue(oracle);
  const result = await engine.score(inputOf(oracle), {});
  const z = compare(result.trunk.pair, oracle.get("trunk/z")).relRms;
  console.log(`  big200-nomsa (${result.tokens} tokens): standalone value ${result.value.toFixed(5)} vs ${ref.toFixed(5)} `
    + `(diff ${(result.value - ref).toFixed(4)}), trunk z relRMS vs oracle ${z.toExponential(2)}, wall ${seconds(result.timings.totalMs)} s `
    + `[${Object.entries(result.timings).map(([k, v]) => `${k} ${seconds(v)}`).join(", ")}]`);
  assert.ok(Math.abs(result.value - ref) <= 0.06, `standalone value ${result.value} vs ${ref}`);
});

test("own chain, recycling_steps=1 (phenol-nomsa-recycle1)", { skip: reason(here("oracle-dumps/phenol-nomsa-recycle1-chain.json")), timeout: 900000 }, async () => {
  const oracle = await TensorBundle.open(here("oracle-dumps/phenol-nomsa-recycle1-chain.json"));
  const engine = await getChain();
  const result = await engine.score(inputOf(oracle), { oracle, recycles: 1 });
  const z = compare(result.trunk.pair, oracle.get("trunk/z")).relRms;
  const s = compare(result.trunk.single, oracle.get("trunk/s")).relRms;
  const ref = refValue(oracle);
  console.log(`  recycle1: trunk z relRMS ${z.toExponential(2)}, s ${s.toExponential(2)}, value ${result.value.toFixed(5)} vs ${ref.toFixed(5)}, wall ${seconds(result.timings.totalMs)} s`);
  assert.ok(z <= 2e-3, `trunk z relRMS ${z}`);
  assert.ok(s <= 2e-3, `trunk s relRMS ${s}`);
  assert.ok(Math.abs(result.value - ref) <= 3e-3, `value ${result.value} vs ${ref}`);
});
