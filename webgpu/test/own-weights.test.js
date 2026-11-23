// Unit tests for the PyTorch-named trunk pack (tools/convert_boltz_trunk.py). No GPU, no LocalFold.
// Needs ~/.boltz/boltz2_aff.ckpt (or BOLTZ_CKPT); generates packs/boltz2-trunk-own if absent.
// Skips visibly when the checkpoint and pack are both missing.
import { before, test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { homedir } from "node:os";
import { OwnWeights } from "../src/trunk/own-weights.js";

const here = (p) => new URL(`../${p}`, import.meta.url).pathname;
const manifest = here("packs/boltz2-trunk-own/boltz2-trunk-own.json");
const ckpt = process.env.BOLTZ_CKPT ?? `${homedir()}/.boltz/boltz2_aff.ckpt`;
const SAMPLE = [
  "s_init.weight",
  "z_init_1.weight",
  "rel_pos.linear_layer.weight",
  "contact_conditioning.encoding_unspecified",
  "input_embedder.res_type_encoding.weight",
  "input_embedder.atom_attention_encoder.atom_encoder.diffusion_transformer.layers.2.transition.swish_gate.0.weight",
  "msa_module.layers.3.pairformer_layer.tri_mul_in.p_in.weight",
  "msa_module.layers.0.outer_product_mean.proj_o.bias",
  "pairformer_module.layers.0.attention.proj_q.bias",
  "pairformer_module.layers.63.transition_z.fc3.weight",
];

let skip;
let w;
let reference; // name -> number[] from the checkpoint

before(async () => {
  if (!existsSync(manifest)) {
    if (!existsSync(ckpt)) { skip = "checkpoint and generated pack are both absent"; return; }
    const r = spawnSync("uv", ["run", "--with", "torch", "--with", "numpy", "python",
      here("tools/convert_boltz_trunk.py"), "--checkpoint", ckpt, "--out-dir", here("packs/boltz2-trunk-own")],
    { encoding: "utf8" });
    if (r.status !== 0) { skip = `pack generation failed: ${r.stderr.slice(-300)}`; return; }
  }
  w = await OwnWeights.open(manifest);
  if (existsSync(ckpt)) {
    const code = `import json,sys,torch\ns=torch.load(sys.argv[1],map_location='cpu',weights_only=False)['state_dict']\n` +
      `print(json.dumps({k:s[k].flatten().tolist() for k in sys.argv[2:]}))`;
    const r = spawnSync("uv", ["run", "--with", "torch", "python", "-c", code, ckpt, ...SAMPLE],
      { encoding: "utf8", maxBuffer: 1 << 28 });
    if (r.status === 0) reference = JSON.parse(r.stdout);
  }
});

const guard = (name, fn) => test(name, async (t) => {
  if (skip) return t.skip(skip);
  await fn(t);
});

guard("meta carries the model name and architecture", () => {
  assert.equal(w.model, "boltz2");
  const a = w.arch;
  assert.deepEqual([a.tokenS, a.tokenZ, a.msaS, a.msaBlocks, a.pairformerBlocks, a.pairformerHeads], [384, 128, 64, 4, 64, 16]);
});

guard("every expected prefix is present", () => {
  for (const p of ["input_embedder/atom_encoder", "input_embedder/atom_attention_encoder", "s_init", "z_init_1", "z_init_2", "rel_pos",
    "token_bonds", "token_bonds_type", "contact_conditioning", "s_norm", "z_norm", "s_recycle", "z_recycle"]) {
    assert.ok(w.names(`${p}/`).length > 0, p);
  }
  for (let i = 0; i < w.arch.msaBlocks; i++) assert.ok(w.has(`msa_module/layers/${i}/pairformer_layer/tri_mul_out/p_in/weight`), `msa ${i}`);
  for (let i = 0; i < w.arch.pairformerBlocks; i++) {
    for (const t of ["attention/proj_q/bias", "attention/proj_o/weight", "tri_att_end/mha/linear_o/weight", "transition_z/fc3/weight"]) {
      assert.ok(w.has(`pairformer_module/layers/${i}/${t}`), `pairformer ${i} ${t}`);
    }
  }
});

guard("shapes are consistent with meta dims", () => {
  const { tokenS: s, tokenZ: z, msaS, pairformerHeads: h } = w.arch;
  assert.deepEqual(w.shape("s_init/weight"), [s, s]);
  assert.deepEqual(w.shape("z_init_1/weight"), [z, s]);
  assert.deepEqual(w.shape("z_recycle/weight"), [z, z]);
  assert.deepEqual(w.shape("s_norm/weight"), [s]);
  assert.deepEqual(w.shape("msa_module/s_proj/weight"), [msaS, s]);
  assert.deepEqual(w.shape("pairformer_module/layers/5/attention/proj_z/1/weight"), [h, z]);
  assert.deepEqual(w.shape("pairformer_module/layers/5/transition_s/fc1/weight"), [4 * s, s]);
  assert.deepEqual(w.shape("pairformer_module/layers/5/transition_z/fc1/weight"), [4 * z, z]);
  const slice = w.slice("pairformer_module/layers/0/tri_mul_out");
  assert.deepEqual(slice["p_in/weight"].shape, [2 * z, z]);
  assert.ok(slice["p_in/weight"].data instanceof Float32Array);
});

guard("no template, distogram, confidence, structure or affinity tensors leaked in", () => {
  const bad = /^(template_module|distogram_module|confidence_module|structure_module|diffusion_conditioning|affinity)/;
  assert.deepEqual(w.names().filter((n) => bad.test(n)), []);
});

guard("values round-trip against the checkpoint", (t) => {
  if (!reference) return t.skip("could not read the checkpoint with torch");
  for (const name of SAMPLE) {
    const got = w.get(name.replaceAll(".", "/"));
    const want = reference[name];
    assert.equal(got.length, want.length, name);
    for (let i = 0; i < want.length; i += Math.max(1, want.length >> 12)) assert.equal(got[i], Math.fround(want[i]), `${name}[${i}]`);
  }
});

test("manifest is readable without loading tensors", (t) => {
  if (!existsSync(manifest)) return t.skip("pack absent");
  assert.equal(JSON.parse(readFileSync(manifest, "utf8")).meta.model, "boltz2");
});
