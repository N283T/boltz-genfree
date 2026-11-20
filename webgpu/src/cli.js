#!/usr/bin/env node
// Score one protein-ligand pair with Boltz-GenFree on WebGPU (or the CPU reference).
//
//   node src/cli.js --pack packs/genfree-affinity.json --trunk oracle-dumps/ckpt-dump.json \
//                   [--mode zero_distogram_features] [--seed 0] [--cpu] [--json]
//
// --pack   weight pack from webgpu/tools/convert_weights.py
// --trunk  tensor bundle with the trunk outputs (input/s_inputs, input/z_trunk, masks)
import { parseArgs } from "node:util";
import { TensorBundle } from "./bundle.js";
import { STRUCTURE_FREE_MODES } from "./distogram.js";
import { loadTrunkInput } from "./trunk-input.js";
import { scoreEnsemble } from "./reference.js";

const { values } = parseArgs({
  options: {
    pack: { type: "string" },
    trunk: { type: "string" },
    mode: { type: "string", default: "zero_distogram_features" },
    seed: { type: "string", default: "0" },
    cpu: { type: "boolean", default: false },
    json: { type: "boolean", default: false },
    "mw-correction": { type: "string" },
  },
});
if (!values.pack || !values.trunk) {
  console.error("usage: node src/cli.js --pack <pack.json> --trunk <trunk.json> [--mode M] [--seed N] [--cpu] [--json] [--mw-correction <Da>]");
  process.exit(2);
}
if (!STRUCTURE_FREE_MODES.includes(values.mode)) {
  console.error(`unknown mode ${values.mode}; expected one of ${STRUCTURE_FREE_MODES.join(", ")}`);
  process.exit(2);
}

const bundle = await TensorBundle.open(values.pack);
const input = await loadTrunkInput(values.trunk);
const options = { mode: values.mode, seed: Number(values.seed) };
const started = performance.now();
let result;
let backend;
if (values.cpu) {
  backend = "cpu-reference";
  result = scoreEnsemble(bundle, input, options);
} else {
  const { createNodeDevice } = await import("./device.js");
  const { AffinityGpu } = await import("./gpu/affinity-gpu.js");
  const { device, adapter } = await createNodeDevice();
  backend = `webgpu (${adapter.info?.vendor ?? "?"} ${adapter.info?.architecture ?? ""})`.trim();
  const engine = new AffinityGpu(device, bundle);
  result = await engine.score(input, options);
  await engine.destroy();
}
const elapsedMs = performance.now() - started;

let value = result.value;
// Optional molecular-weight correction (Boltz2._run_affinity_mode, off by default in GenFree).
if (values["mw-correction"] !== undefined) {
  value = 1.03525938 * value - 0.59992683 * Number(values["mw-correction"]) ** 0.3 + 2.83288489;
}
const report = {
  backend, mode: values.mode, tokens: input.padMask.length,
  affinity_pred_value: value,
  affinity_probability_binary: result.probability,
  members: result.members.map((m) => ({ affinity_pred_value: m.value, affinity_probability_binary: m.probability })),
  elapsed_ms: Math.round(elapsedMs),
};
console.log(values.json ? JSON.stringify(report) : [
  `backend      ${report.backend}`,
  `mode         ${report.mode}  (${report.tokens} tokens)`,
  `pred value   ${value.toFixed(6)}`,
  `P(binder)    ${result.probability.toFixed(6)}`,
  `elapsed      ${report.elapsed_ms} ms`,
].join("\n"));
