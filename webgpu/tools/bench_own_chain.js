#!/usr/bin/env node
// Time and memory of the own chain (no LocalFold).
//   node tools/bench_own_chain.js --case big200-nomsa [--exact] [--recycles 0]
//   node tools/bench_own_chain.js --long 384          # big200's protein extended to 384 residues + acetaminophen (395 tokens)
// Reports wall time per stage, peak process RSS (sampled) and the peak of live GPU buffer bytes created through the device.
import { parseArgs } from "node:util";
import { readFileSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { compare } from "../src/compare.js";
import { createNodeDevice } from "../src/device.js";
import { createOwnChain } from "../src/trunk/own/chain.js";

const { values } = parseArgs({
  options: {
    case: { type: "string", default: "big200-nomsa" }, long: { type: "string" }, exact: { type: "boolean", default: false },
    recycles: { type: "string", default: "0" },
  },
});
const here = (p) => new URL(`../${p}`, import.meta.url).pathname;
const oracle = await TensorBundle.open(here(`oracle-dumps/${values.case}-chain.json`));
const input = {
  sequence: oracle.meta.sequence, smiles: oracle.meta.smiles,
  a3m: oracle.meta.msa ? readFileSync(new URL(`../../${oracle.meta.msa}`, import.meta.url), "utf8") : null,
};
const options = { recycles: Number(values.recycles) };
if (values.long) {
  const length = Number(values.long);
  input.sequence = input.sequence.repeat(Math.ceil(length / input.sequence.length)).slice(0, length);
  options.allowExtended = true;
} else if (values.exact) options.oracle = oracle;

const { device } = await createNodeDevice();
let live = 0;
let peakGpu = 0;
const create = device.createBuffer.bind(device);
device.createBuffer = (descriptor) => {
  const buffer = create(descriptor);
  live += descriptor.size;
  peakGpu = Math.max(peakGpu, live);
  const destroy = buffer.destroy.bind(buffer);
  let done = false;
  buffer.destroy = () => { if (!done) { done = true; live -= descriptor.size; } destroy(); };
  return buffer;
};
let peakRss = 0;
const sample = setInterval(() => { peakRss = Math.max(peakRss, process.memoryUsage().rss); }, 20);
const chain = await createOwnChain({
  trunkPack: here("packs/boltz2-trunk-own/boltz2-trunk-own.json"), affinityPack: here("packs/genfree-affinity.json"), device,
});
const t0 = performance.now();
const result = await chain.score(input, { ...options, onStage: undefined });
const wall = performance.now() - t0;
clearInterval(sample);
peakRss = Math.max(peakRss, process.memoryUsage().rss);
const gb = (b) => (b / 2 ** 30).toFixed(2);
console.log(JSON.stringify({
  case: values.long ? `long-${values.long}` : values.case, tokens: result.tokens, msaRows: result.msaRows,
  value: result.value, probability: result.probability,
  ...(values.long ? { finite: result.trunk.pair.every(Number.isFinite) } : {
    refValue: oracle.has("final/zero_distogram_features/affinity_pred_value") ? oracle.get("final/zero_distogram_features/affinity_pred_value")[0] : oracle.meta.affinity_pred_value,
    trunkZRelRms: compare(result.trunk.pair, oracle.get("trunk/z")).relRms,
  }),
  wallSeconds: +(wall / 1000).toFixed(1),
  stageSeconds: Object.fromEntries(Object.entries(result.timings).map(([k, v]) => [k, +(v / 1000).toFixed(2)])),
  peakRssGiB: +gb(peakRss), peakGpuBufferGiB: +gb(peakGpu),
}, null, 1));
await chain.destroy();
device.destroy();
