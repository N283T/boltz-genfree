#!/usr/bin/env node
// Check tools/convert_trunk_weights.py against LocalFold's published int5 boltz2 bundle: every tensor of
// the f32 pack should equal the dequantised int5 tensor up to int5 round-off (relRMS ~0.04; norms and
// biases exactly). Template tensors are excluded (zeros here by design).
//   LOCALFOLD_DIR=... node tools/compare_trunk_pack.mjs packs/boltz2-trunk-f32 <int5 bundle dir>
import { loadLocalFold } from "../src/trunk/localfold.js";
import { openTrunkStore } from "../src/trunk/weights.js";
import { TensorBundle } from "../src/bundle.js";

const [packDir, int5Dir] = process.argv.slice(2);
const lf = await loadLocalFold(process.env.LOCALFOLD_DIR);
const pack = await TensorBundle.open(`${packDir}/manifest.json`);
const { store, close } = await openTrunkStore(lf, int5Dir);
const rows = [];
for (const name of Object.keys(pack.records)) {
  if (name.includes("template_embedding")) continue;
  const a = pack.get(name);
  const b = await store.tensor(name);
  if (a.length !== b.length) { rows.push([Infinity, name, "length"]); continue; }
  let e = 0;
  let s = 0;
  for (let i = 0; i < a.length; i += 1) { e += (a[i] - b[i]) ** 2; s += b[i] ** 2; }
  rows.push([Math.sqrt(e / Math.max(s, 1e-30)), name]);
}
rows.sort((x, y) => y[0] - x[0]);
console.log(`${rows.length} tensors compared; worst ${rows[0][0].toFixed(4)} (${rows[0][1]}); median ${rows[rows.length >> 1][0].toFixed(4)}`);
for (const r of rows.slice(0, 3)) console.log(r[0].toFixed(4), r[1]);
await close();
process.exit(0);
