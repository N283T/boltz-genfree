// WebGPU implementation against the PyTorch oracle. Skips (visibly) without a WebGPU adapter.
import { after, test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { compare } from "../src/compare.js";
import { loadOracle } from "../src/oracle-input.js";
import { createNodeDevice } from "../src/device.js";
import { AffinityGpu } from "../src/gpu/affinity-gpu.js";

const here = (p) => new URL(`../${p}`, import.meta.url).pathname;
// Explicit tolerances (relative RMS against the PyTorch float32 oracle).
const CASES = [
  { name: "small", dump: "oracle-dumps/small-dump.json", weights: "oracle-dumps/small-weights.json", stageTol: 1e-4, finalTol: 5e-5 },
  { name: "ckpt131", dump: "oracle-dumps/ckpt131-dump.json", weights: "packs/genfree-affinity.json", stageTol: 1e-3, finalTol: 5e-4 },
  { name: "ckpt", dump: "oracle-dumps/ckpt-dump.json", weights: "packs/genfree-affinity.json", stageTol: 1e-3, finalTol: 5e-4 },
];
const STAGES = ["z_linear", "pooled_processed", "conditioner_update", "pooled_conditioned", "layer0", "layer1", "pairformer_out", "pooled_pairformer", "pooled_hidden"];

let gpu;
try { gpu = await createNodeDevice(); } catch (error) { gpu = { error }; }

// Dawn keeps the process alive after the last test unless the device is destroyed.
after(() => gpu.device?.destroy());

for (const c of CASES) {
  const present = existsSync(here(c.dump)) && existsSync(here(c.weights));
  const skip = gpu.error ? `no WebGPU: ${gpu.error.message}` : !present && "oracle dump or weights missing";
  test(`webgpu vs oracle (${c.name}) - stage relRMS <= ${c.stageTol}`, { skip }, async () => {
    const { dump, input } = await loadOracle(here(c.dump));
    const bundle = await TensorBundle.open(here(c.weights));
    const engine = new AffinityGpu(gpu.device, bundle);
    try {
      for (const mode of dump.meta.modes) {
        if (mode === "normal_random") continue; // JS Box-Muller stream differs from torch.randn; see distogram.js
        const traces = {};
        const ens = await engine.score(input, {
          mode, seed: dump.meta.seed,
          taps: (root) => { const t = new Map(); t.layers = 2; traces[root] = t; return t; },
        });
        for (const root of ["affinity1", "affinity2"]) {
          for (const stage of STAGES) {
            const key = `${root}/${mode}/${stage}`;
            if (!dump.has(key) || !traces[root].has(stage)) continue;
            const stat = compare(traces[root].get(stage), dump.get(key));
            console.log(`  ${c.name} ${root} ${mode} ${stage}: relRMS ${stat.relRms.toExponential(2)} maxAbs ${stat.maxAbs.toExponential(2)}`);
            assert.ok(stat.relRms <= c.stageTol, `${key}: relRMS ${stat.relRms}`);
          }
        }
        const expectedValue = dump.get(`final/${mode}/affinity_pred_value`)[0];
        const expectedProb = dump.get(`final/${mode}/affinity_probability_binary`)[0];
        console.log(`  ${c.name} final ${mode}: value ${ens.value.toFixed(6)} vs ${expectedValue.toFixed(6)}, prob ${ens.probability.toFixed(6)} vs ${expectedProb.toFixed(6)}`);
        assert.ok(Math.abs(ens.value - expectedValue) <= Math.max(1e-4, c.finalTol * Math.abs(expectedValue)), `value ${ens.value} vs ${expectedValue}`);
        assert.ok(Math.abs(ens.probability - expectedProb) <= 1e-4, `probability ${ens.probability} vs ${expectedProb}`);
      }
    } finally {
      engine.destroy();
    }
  });
}
