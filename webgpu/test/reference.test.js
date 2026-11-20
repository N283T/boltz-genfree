// CPU reference against the PyTorch oracle. Needs oracle dumps:
//   uv run python webgpu/tools/dump_oracle.py --random-small 7
//   uv run python webgpu/tools/dump_oracle.py --checkpoint ~/.boltz/boltz2_aff.ckpt   (+ convert_weights.py)
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { compare } from "../src/compare.js";
import { loadOracle } from "../src/oracle-input.js";
import { runAffinityModule, scoreEnsemble } from "../src/reference.js";
import { STRUCTURE_FREE_MODES } from "../src/distogram.js";

const CASES = [
  { name: "small", dump: "oracle-dumps/small-dump.json", weights: "oracle-dumps/small-weights.json", tol: 2e-5 },
  { name: "ckpt", dump: "oracle-dumps/ckpt-dump.json", weights: "packs/genfree-affinity.json", tol: 5e-4 },
];
const STAGES = ["z_linear", "pooled_processed", "conditioner_update", "pooled_conditioned", "layer0", "pairformer_out", "pooled_pairformer", "pooled_hidden", "affinity_pred_value", "affinity_logits_binary"];

for (const c of CASES) {
  const present = existsSync(new URL(`../${c.dump}`, import.meta.url)) && existsSync(new URL(`../${c.weights}`, import.meta.url));
  test(`cpu reference vs oracle (${c.name}) - relRMS <= ${c.tol}`, { skip: !present && "oracle dump or weights missing" }, async () => {
    const { dump, input } = await loadOracle(new URL(`../${c.dump}`, import.meta.url).pathname);
    const bundle = await TensorBundle.open(new URL(`../${c.weights}`, import.meta.url).pathname);
    const seed = dump.meta.seed;
    for (const mode of STRUCTURE_FREE_MODES) {
      for (const root of ["affinity1", "affinity2"]) {
        const trace = new Map();
        const oracleBins = dump.get(`${root}/${mode}/affinity_distogram_bins`);
        runAffinityModule(bundle, root, input, { mode, seed, trace });
        // bins generated on the JS side must equal the torch generator's (uniform) or be valid (normal)
        if (mode !== "normal_random") assert.deepEqual(Array.from(trace.get("affinity_distogram_bins")), Array.from(oracleBins), `${root}/${mode} bins`);
        for (const stage of STAGES) {
          // the PyTorch module never calls the conditioner in zero_conditioning mode
          if (!dump.has(`${root}/${mode}/${stage}`)) continue;
          const stat = compare(trace.get(stage), dump.get(`${root}/${mode}/${stage}`));
          // normal_random bins differ from torch's randn stream; skip numeric stages for it
          if (mode === "normal_random") continue;
          assert.ok(stat.relRms <= c.tol, `${root}/${mode}/${stage}: relRMS ${stat.relRms} maxAbs ${stat.maxAbs}`);
        }
      }
      if (mode !== "normal_random") {
        const ens = scoreEnsemble(bundle, input, { mode, seed });
        const expected = dump.get(`final/${mode}/affinity_pred_value`)[0];
        assert.ok(Math.abs(ens.value - expected) <= Math.max(1e-4, c.tol * Math.abs(expected) * 10), `final value ${ens.value} vs ${expected}`);
        const eprob = dump.get(`final/${mode}/affinity_probability_binary`)[0];
        assert.ok(Math.abs(ens.probability - eprob) <= 1e-4, `final prob ${ens.probability} vs ${eprob}`);
      }
    }
  });
}
