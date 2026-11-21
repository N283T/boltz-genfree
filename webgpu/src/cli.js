#!/usr/bin/env node
// Boltz-GenFree on WebGPU.
//
// Full chain (needs LocalFold for the trunk, see docs/webgpu.md):
//   LOCALFOLD_DIR=/path/to/localfold node src/cli.js --sequence MKT... [--a3m target.a3m] --smiles 'Oc1ccccc1'
//
// Affinity module only, from a stored trunk output:
//   node src/cli.js --trunk oracle-dumps/ckpt-dump.json [--cpu]
import { parseArgs } from "node:util";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";
import { TensorBundle } from "./bundle.js";
import { STRUCTURE_FREE_MODES } from "./distogram.js";
import { loadTrunkInput } from "./trunk-input.js";
import { scoreEnsemble } from "./reference.js";

const here = (p) => resolve(new URL(`../${p}`, import.meta.url).pathname);
const { values } = parseArgs({
  options: {
    sequence: { type: "string" },
    "sequence-file": { type: "string" },
    a3m: { type: "string" },
    smiles: { type: "string" },
    localfold: { type: "string", default: process.env.LOCALFOLD_DIR ?? "" },
    "trunk-weights": { type: "string" },
    pack: { type: "string", default: here("packs/genfree-affinity.json") },
    trunk: { type: "string" },
    mode: { type: "string", default: "zero_distogram_features" },
    seed: { type: "string", default: "0" },
    recycles: { type: "string", default: "0" },
    precision: { type: "string", default: "f32" },
    "ligand-conformer": { type: "string", default: "skip" },
    "ligand-atoms": { type: "string" },
    "allow-extended": { type: "boolean", default: false },
    cpu: { type: "boolean", default: false },
    json: { type: "boolean", default: false },
    "mw-correction": { type: "string" },
  },
});

const usage = `usage:
  node src/cli.js --sequence <protein> --smiles <SMILES> [--a3m file] [--localfold DIR] [--trunk-weights DIR|URL]
                  [--recycles 0..5] [--precision f32|auto] [--ligand-atoms ligand.json] [--allow-extended] [--json]
  node src/cli.js --trunk <trunk dump manifest> [--cpu]       (affinity module on a stored trunk output)`;
if (!STRUCTURE_FREE_MODES.includes(values.mode)) {
  console.error(`unknown mode ${values.mode}; expected one of ${STRUCTURE_FREE_MODES.join(", ")}`);
  process.exit(2);
}
const fullChain = values.sequence !== undefined || values["sequence-file"] !== undefined;
if (!fullChain && !values.trunk) {
  console.error(usage);
  process.exit(2);
}

const mwCorrect = (value, mw) => (mw === undefined ? value : 1.03525938 * value - 0.59992683 * Number(mw) ** 0.3 + 2.83288489);
const affinityPack = await TensorBundle.open(values.pack);
let report;
const started = performance.now();

if (fullChain) {
  if (!values.smiles) { console.error(usage); process.exit(2); }
  if (values.cpu) { console.error("--cpu applies to --trunk dumps only; the trunk needs WebGPU"); process.exit(2); }
  const sequence = (values.sequence ?? readFileSync(values["sequence-file"], "utf8")).replace(/\s+/g, "");
  const a3m = values.a3m ? readFileSync(values.a3m, "utf8") : null;
  const trunkWeights = values["trunk-weights"] ?? (existsSync(here("packs/boltz2-trunk-f32/manifest.json")) ? here("packs/boltz2-trunk-f32") : undefined);
  if (!trunkWeights) {
    console.error("no trunk weights: run tools/convert_trunk_weights.py (packs/boltz2-trunk-f32) or pass --trunk-weights DIR|URL");
    process.exit(2);
  }
  const { createNodeDevice } = await import("./device.js");
  const { createChain } = await import("./trunk/chain.js");
  const { device, adapter } = await createNodeDevice({ localfoldDir: values.localfold });
  const chain = await createChain({ localfold: values.localfold, trunkWeights, affinityPack, device });
  const result = await chain.score({ sequence, a3m, smiles: values.smiles }, {
    recycles: Number(values.recycles), precision: values.precision, mode: values.mode, seed: Number(values.seed),
    ligandConformer: values["ligand-conformer"], allowExtended: values["allow-extended"],
    ligandAtoms: values["ligand-atoms"] ? JSON.parse(readFileSync(values["ligand-atoms"], "utf8")) : undefined,
  });
  report = {
    backend: `webgpu (${adapter.info?.vendor ?? "?"} ${adapter.info?.architecture ?? ""})`.trim(),
    mode: values.mode, tokens: result.tokens, receptor_tokens: result.receptorTokens, ligand_tokens: result.ligandTokens,
    msa_rows: result.msaRows, recycles: Number(values.recycles), precision: values.precision,
    affinity_pred_value: mwCorrect(result.value, values["mw-correction"]),
    affinity_probability_binary: result.probability,
    members: result.members.map((m) => ({ affinity_pred_value: m.value, affinity_probability_binary: m.probability })),
    timings_ms: Object.fromEntries(Object.entries(result.timings).map(([k, v]) => [k, Math.round(v)])),
    weight_load_ms: Math.round(chain.weightLoadMs),
    notes: result.notes,
  };
  await chain.destroy();
  device.destroy();
} else {
  const input = await loadTrunkInput(values.trunk);
  const options = { mode: values.mode, seed: Number(values.seed) };
  let result;
  let backend;
  if (values.cpu) {
    backend = "cpu-reference";
    result = scoreEnsemble(affinityPack, input, options);
  } else {
    const { createNodeDevice } = await import("./device.js");
    const { AffinityGpu } = await import("./gpu/affinity-gpu.js");
    const { device, adapter } = await createNodeDevice();
    backend = `webgpu (${adapter.info?.vendor ?? "?"} ${adapter.info?.architecture ?? ""})`.trim();
    const engine = new AffinityGpu(device, affinityPack);
    result = await engine.score(input, options);
    await engine.destroy();
    device.destroy();
  }
  report = {
    backend, mode: values.mode, tokens: input.padMask.length,
    affinity_pred_value: mwCorrect(result.value, values["mw-correction"]),
    affinity_probability_binary: result.probability,
    members: result.members.map((m) => ({ affinity_pred_value: m.value, affinity_probability_binary: m.probability })),
    elapsed_ms: Math.round(performance.now() - started),
  };
}

if (values.json) {
  console.log(JSON.stringify(report));
} else {
  const lines = [
    `backend      ${report.backend}`,
    `mode         ${report.mode}  (${report.tokens} tokens${report.receptor_tokens ? `: ${report.receptor_tokens} receptor + ${report.ligand_tokens} ligand` : ""})`,
    `pred value   ${report.affinity_pred_value.toFixed(6)}`,
    `P(binder)    ${report.affinity_probability_binary.toFixed(6)}`,
  ];
  if (report.timings_ms) {
    const t = report.timings_ms;
    lines.push(`timings      featurise ${t.featuriseMs} ms, trunk ${Object.entries(t).filter(([k]) => k.startsWith("trunk-pass")).reduce((a, [, v]) => a + v, 0)} ms, affinity ${t.affinityMs} ms, total ${t.totalMs} ms (weights ${report.weight_load_ms} ms)`);
  } else {
    lines.push(`elapsed      ${report.elapsed_ms} ms`);
  }
  for (const note of report.notes ?? []) lines.push(`note         ${note}`);
  console.log(lines.join("\n"));
}
process.exit(0);
