#!/usr/bin/env node
// Compare the WebGPU chain against a PyTorch oracle dump (tools/dump_trunk_oracle.py).
//
//   LOCALFOLD_DIR=... node tools/check_chain.js --dump oracle-dumps/phenol-nomsa-chain.json \
//       [--trunk-weights packs/boltz2-trunk-f32] [--features exact|names|js] [--precision f32|auto] [--recycles 0]
//
// --features exact : atom positions and ligand atom names imported from the oracle (engine parity)
//            names : ligand atom names imported, positions from LocalFold's featuriser
//            ligand-atoms: positions from LocalFold's featuriser, ligand atoms from --ligand-atoms (tools/ligand_atoms.py)
//            js    : everything from LocalFold's featuriser (the standalone route)
import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { createNodeDevice } from "../src/device.js";
import { createChain } from "../src/trunk/chain.js";

const { values } = parseArgs({
  options: {
    dump: { type: "string" },
    "trunk-weights": { type: "string", default: "packs/boltz2-trunk-f32" },
    "affinity-pack": { type: "string", default: "packs/genfree-affinity.json" },
    features: { type: "string", default: "exact" },
    precision: { type: "string", default: "f32" },
    recycles: { type: "string", default: "0" },
    "ligand-atoms": { type: "string" },
    localfold: { type: "string", default: process.env.LOCALFOLD_DIR ?? "" },
  },
});
const relRms = (a, b) => {
  let e = 0;
  let s = 0;
  for (let i = 0; i < b.length; i += 1) { e += (a[i] - b[i]) ** 2; s += b[i] ** 2; }
  return Math.sqrt(e / Math.max(s, 1e-30));
};
const oracle = await TensorBundle.open(values.dump);
const meta = oracle.meta;
const a3m = meta.msa ? readFileSync(meta.msa, "utf8") : null;
const { device } = await createNodeDevice({ localfoldDir: values.localfold });
const affinityPack = await TensorBundle.open(values["affinity-pack"]);
const chain = await createChain({ localfold: values.localfold, trunkWeights: values["trunk-weights"], affinityPack, device });
const mode = values.features;
const result = await chain.score({ sequence: meta.sequence, a3m, smiles: meta.smiles }, {
  precision: values.precision, recycles: Number(values.recycles),
  oracle: mode === "js" || mode === "ligand-atoms" ? undefined : oracle, importPositions: mode === "exact", importNames: true,
  ligandAtoms: values["ligand-atoms"] ? JSON.parse(readFileSync(values["ligand-atoms"], "utf8")) : undefined,
});
const n = result.tokens;
const refZ = oracle.get("trunk/z");
const refS = oracle.get("trunk/s");
const report = {
  features: mode, precision: values.precision, tokens: n, msaRows: result.msaRows,
  s_inputs_trunk_relRMS: relRms(result.trunk.sInputsTrunk, oracle.get("trunk/s_inputs_trunk")),
  s_inputs_affinity_relRMS: relRms(result.trunk.sInputsAffinity, oracle.get("input/s_inputs")),
  z_relRMS: relRms(result.trunk.pair, refZ),
  s_relRMS: relRms(result.trunk.single, refS),
  affinity_pred_value: { webgpu: result.value, pytorch: oracle.get("final/zero_distogram_features/affinity_pred_value")[0] },
  affinity_probability_binary: { webgpu: result.probability, pytorch: oracle.get("final/zero_distogram_features/affinity_probability_binary")[0] },
  members: result.members.map((m, i) => ({
    value: m.value, pytorch: oracle.get(`final/zero_distogram_features/affinity_pred_value${i + 1}`)[0],
  })),
  timings_ms: Object.fromEntries(Object.entries(result.timings).map(([k, v]) => [k, Math.round(v)])),
  weight_load_ms: Math.round(chain.weightLoadMs),
};
console.log(JSON.stringify(report, null, 2));
await chain.destroy();
device.destroy();
process.exit(0);
