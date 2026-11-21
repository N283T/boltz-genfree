// Full chain (featurisation -> LocalFold trunk -> affinity) against GenFree's PyTorch run.
// Needs LOCALFOLD_DIR, packs/boltz2-trunk-f32, packs/genfree-affinity and oracle dumps from
// tools/dump_trunk_oracle.py; skips visibly otherwise.
import { after, test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { createNodeDevice } from "../src/device.js";
import { createChain } from "../src/trunk/chain.js";
import { checkSize, normalizeA3m } from "../src/trunk/featurise.js";
import { ligandAtomsFromSmiles } from "../src/trunk/canonical.js";

const here = (p) => new URL(`../${p}`, import.meta.url).pathname;

test("size limits mirror crop_full_screen", () => {
  assert.doesNotThrow(() => checkSize({ receptorTokens: 200, ligandTokens: 56, atoms: 1600 }));
  assert.throws(() => checkSize({ receptorTokens: 201, ligandTokens: 10, atoms: 1000 }), /receptor <= 200/);
  assert.throws(() => checkSize({ receptorTokens: 150, ligandTokens: 129, atoms: 1000 }, { allowExtended: true }), /ligand has 129/);
  assert.doesNotThrow(() => checkSize({ receptorTokens: 400, ligandTokens: 20, atoms: 4000 }, { allowExtended: true }));
});

test("A3M rows are deduplicated like boltz.data.parse.a3m", () => {
  const out = normalizeA3m(">q\nMKT\n>a\nM-T\n>b\nMKT\n#c\n>d\nmKT\n");
  assert.equal(out.split("\n").filter((l) => l.startsWith(">")).length, 2);
});

test("ligand atom names follow RDKit canonical ranks (phenol)", () => {
  const r = ligandAtomsFromSmiles("Oc1ccccc1");
  assert.deepEqual(r.atomNames, ["O1", "C6", "C4", "C2", "C1", "C3", "C5"]);
  assert.ok(r.exact);
});

const localfold = process.env.LOCALFOLD_DIR ?? "";
const CASES = [
  { name: "phenol-nomsa", exact: { z: 2e-3, value: 3e-3 }, js: { value: 0.06 } },
  { name: "acetaminophen-nomsa", exact: { z: 2e-3, value: 3e-3 }, js: { value: 0.06 } },
  { name: "salicylic-nomsa", exact: { z: 2e-3, value: 3e-3 }, js: { value: 0.06 } },
  { name: "phenol-msa", exact: { z: 3e-2, value: 2e-2 }, js: { value: 0.15 } },
];
const missing = !localfold || !existsSync(here("packs/boltz2-trunk-f32/manifest.json")) || !existsSync(here("packs/genfree-affinity.json"));
let gpu;
try { gpu = missing ? { error: new Error("LOCALFOLD_DIR / packs missing") } : await createNodeDevice({ localfoldDir: localfold }); } catch (error) { gpu = { error }; }
let chain;
after(async () => { await chain?.destroy(); gpu.device?.destroy(); });
const relRms = (a, b) => { let e = 0; let s = 0; for (let i = 0; i < b.length; i += 1) { e += (a[i] - b[i]) ** 2; s += b[i] ** 2; } return Math.sqrt(e / s); };

for (const c of CASES) {
  const dump = here(`oracle-dumps/${c.name}-chain.json`);
  const skip = gpu.error ? gpu.error.message : !existsSync(dump) && "oracle dump missing";
  test(`chain vs PyTorch (${c.name})`, { skip }, async () => {
    const oracle = await TensorBundle.open(dump);
    chain ??= await createChain({
      localfold, trunkWeights: here("packs/boltz2-trunk-f32"),
      affinityPack: await TensorBundle.open(here("packs/genfree-affinity.json")), device: gpu.device,
    });
    const input = { sequence: oracle.meta.sequence, smiles: oracle.meta.smiles,
      a3m: oracle.meta.msa ? readFileSync(new URL(`../../${oracle.meta.msa}`, import.meta.url), "utf8") : null };
    const ref = oracle.get("final/zero_distogram_features/affinity_pred_value")[0];
    const exact = await chain.score(input, { oracle });
    assert.ok(relRms(exact.trunk.pair, oracle.get("trunk/z")) <= c.exact.z, "trunk z");
    assert.ok(Math.abs(exact.value - ref) <= c.exact.value, `exact-feature value ${exact.value} vs ${ref}`);
    const standalone = await chain.score(input, {});
    assert.ok(Math.abs(standalone.value - ref) <= c.js.value, `standalone value ${standalone.value} vs ${ref}`);
  });
}
