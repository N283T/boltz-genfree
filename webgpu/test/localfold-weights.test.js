// Unit tests for the LocalFold int5 trunk loader (src/trunk/localfold-weights.js). No GPU, no network.
// Part 1 runs on a synthetic bundle (int5 reference encoded by tools/map_localfold_weights.py --write-fixture).
// Part 2 needs the downloaded LocalFold shards (BOLTZ_LOCALFOLD_DIR, default /private/tmp/claude-501/lf-boltz2) and the
// f32 pack packs/boltz2-trunk-own; it skips visibly when either is absent.
import { after, before, test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { LocalFoldWeights, decodeInt5, halfToFloat } from "../src/trunk/localfold-weights.js";
import { OwnWeights } from "../src/trunk/own-weights.js";

const here = (p) => new URL(`../${p}`, import.meta.url).pathname;
const fixture = JSON.parse(readFileSync(here("test/fixtures/localfold-int5.json"), "utf8"));
const shard = Uint8Array.from(Buffer.from(fixture.shard, "base64"));
const expected = new Float32Array(Uint8Array.from(Buffer.from(fixture.expected, "base64")).buffer);
const original = new Float32Array(Uint8Array.from(Buffer.from(fixture.original, "base64")).buffer);

function decodeRange(first, count) {
  const g0 = Math.floor(first / 32);
  const g1 = Math.ceil((first + count) / 32);
  return decodeInt5(
    shard.subarray(fixture.codeAt + g0 * 20, fixture.codeAt + g1 * 20),
    shard.subarray(fixture.scaleAt + g0 * 2, fixture.scaleAt + g1 * 2),
    shard.subarray(fixture.zeroAt + g0 * 2, fixture.zeroAt + g1 * 2),
    first - g0 * 32, count,
  );
}

test("halfToFloat: normal, subnormal, signed, special values", () => {
  assert.equal(halfToFloat(0x3c00), 1);
  assert.equal(halfToFloat(0xc000), -2);
  assert.equal(halfToFloat(0x3555), 0.333251953125);
  assert.equal(halfToFloat(0x0001), 2 ** -24);
  assert.equal(halfToFloat(0x7bff), 65504);
  assert.equal(halfToFloat(0x7c00), Infinity);
  assert.ok(Number.isNaN(halfToFloat(0x7e00)));
  assert.equal(halfToFloat(0), 0);
});

test("int5 decode of the whole synthetic tensor matches the Python reference", () => {
  const out = decodeRange(0, fixture.count);
  assert.equal(out.length, fixture.count);
  for (let i = 0; i < out.length; i += 1) assert.ok(Math.abs(out[i] - expected[i]) <= 1e-5 * Math.max(1, Math.abs(expected[i])), `element ${i}`);
  // the reference encoder is lossy by at most half a quantisation step per group
  let worst = 0;
  for (let i = 0; i < out.length; i += 1) worst = Math.max(worst, Math.abs(out[i] - original[i]));
  assert.ok(worst < 0.5, `encode error ${worst}`);
});

test("int5 range decode reads only the groups it needs (unaligned start and end)", () => {
  for (const [first, count] of [[0, 1], [31, 2], [57, 57], [114, 57], [33, 100], [fixture.count - 3, 3]]) {
    const out = decodeRange(first, count);
    assert.deepEqual(Array.from(out), Array.from(expected.subarray(first, first + count)), `range ${first}+${count}`);
  }
});

test("int5 decode refuses truncated input", () => {
  assert.throws(() => decodeInt5(new Uint8Array(10), new Uint8Array(2), new Uint8Array(2), 0, 32), /too short/);
});

// ---------------------------------------------------------------- recipes on a synthetic bundle

const f32 = Float32Array.from({ length: 2 * 2 * 3 }, (_, i) => i + 1); // stacked [2, 2, 3]
const small = Float32Array.from([10, 20, 30]);
let dir;
let manifest;
let bytes;
const SHARD = "toy.int5.bin";

function toyBundle() {
  const parts = [shard];
  let at = shard.length;
  const place = (array) => { const start = at; parts.push(new Uint8Array(array.buffer.slice(0))); at += array.byteLength; return start; };
  const stackedAt = place(f32);
  const smallAt = place(small);
  bytes = Buffer.concat(parts);
  manifest = {
    formatVersion: 1, source: "synthetic", bundle: { encoding: "int5-le" },
    tensors: {
      "t/int": { file: SHARD, shape: [3, 57], byteOffset: fixture.codeAt, scaleOffset: fixture.scaleAt, zeroOffset: fixture.zeroAt, dtype: "int5", block: 32 },
      "t/stack": { file: SHARD, shape: [2, 2, 3], byteOffset: stackedAt, dtype: "float32" },
      "t/small": { file: SHARD, shape: [3], byteOffset: smallAt, dtype: "float32" },
    },
  };
}

const leaf = (src, ops, layer = false) => ({ src, ...(layer ? { layer: true } : {}), ...(ops ? { ops } : {}) });
const toyMap = {
  version: 1, meta: { model: "toy", architecture: { blocks: 2 } }, aliases: { T: "t/" },
  rules: [
    { pack: "p/{L}/stack", layers: 2, shape: [2, 3], recipe: leaf("${T}stack", undefined, true) },
    { pack: "p/{L}/transposed", layers: 2, shape: [3, 2], recipe: leaf("${T}stack", [["transpose", [1, 0]]], true) },
    { pack: "p/{L}/flip", layers: 2, shape: [2, 3], recipe: leaf("${T}stack", [["flip", 1]], true) },
    { pack: "p/{L}/slice", layers: 2, shape: [2, 2], recipe: leaf("${T}stack", [["slice", 1, 1, 3]], true) },
    { pack: "p/{L}/take", layers: 2, shape: [2, 2], recipe: leaf("${T}stack", [["take", 1, [2, 0]]], true) },
    { pack: "p/scaled", layers: null, shape: [3], recipe: leaf("${T}small", [["scale", -2]]) },
    { pack: "p/column", layers: null, shape: [3, 1], recipe: leaf("${T}small", [["reshape", [-1, 1]]]) },
    { pack: "p/glued", layers: null, shape: [3, 3], recipe: { concat: 1, parts: [{ zeros: [3, 1] }, leaf("${T}small", [["reshape", [3, 1]]]), { add: [leaf("${T}small", [["reshape", [3, 1]]]), leaf("${T}small", [["reshape", [3, 1]]])] }] } },
    { pack: "p/{L}/int", layers: 3, shape: [57], recipe: leaf("${T}int", undefined, true) },
  ],
};

before(() => {
  dir = mkdtempSync(join(tmpdir(), "localfold-toy-"));
  toyBundle();
  writeFileSync(join(dir, SHARD), bytes);
  writeFileSync(join(dir, "manifest.json"), JSON.stringify(manifest));
});
after(() => { rmSync(dir, { recursive: true, force: true }); });

const approx = (a, b) => assert.deepEqual(Array.from(a), Array.from(b));

test("recipes: layer, transpose, flip, slice, take, scale, reshape(-1), concat, add, zeros", async () => {
  const w = await LocalFoldWeights.open(join(dir, "manifest.json"), { map: toyMap });
  try {
    assert.equal(w.model, "toy");
    assert.deepEqual(w.arch, { blocks: 2 });
    assert.equal(w.names("p/1/").length, 6);
    assert.equal(w.names("p/").length, 2 * 5 + 2 + 3 + 1);
    assert.equal(w.has("p/0/stack"), true);
    assert.equal(w.has("p/2/stack"), false);
    assert.deepEqual(w.shape("p/1/transposed"), [3, 2]);
    approx(w.get("p/0/stack"), [1, 2, 3, 4, 5, 6]);
    approx(w.get("p/1/stack"), [7, 8, 9, 10, 11, 12]);
    approx(w.get("p/1/transposed"), [7, 10, 8, 11, 9, 12]);
    approx(w.get("p/0/flip"), [3, 2, 1, 6, 5, 4]);
    approx(w.get("p/1/slice"), [8, 9, 11, 12]);
    approx(w.get("p/0/take"), [3, 1, 6, 4]);
    approx(w.get("p/scaled"), [-20, -40, -60]);
    approx(w.get("p/column"), [10, 20, 30]);
    approx(w.get("p/glued"), [0, 10, 20, 0, 20, 40, 0, 30, 60]);
    // the int5 layer slices (57 elements: not group aligned) equal the reference decode
    for (let layer = 0; layer < 3; layer += 1) {
      approx(w.get(`p/${layer}/int`), expected.subarray(layer * 57, (layer + 1) * 57));
    }
    // every get returns a fresh array
    assert.notEqual(w.get("p/scaled"), w.get("p/scaled"));
    assert.throws(() => w.get("nope"), /no tensor/);
    const slice = w.slice("p/1");
    assert.deepEqual(Object.keys(slice).sort(), ["flip", "int", "slice", "stack", "take", "transposed"]);
    assert.deepEqual(slice.stack.shape, [2, 3]);
    assert.throws(() => w.slice("q"), /no tensors/);
    await w.ensure("p/");
    w.release("p/");
    w.close();
    approx(w.get("p/scaled"), [-20, -40, -60]); // reads reopen the files after close
  } finally { w.close(); }
});

test("open fails early on a recipe that points at a missing bundle tensor, and on foreign manifests", async () => {
  const bad = { ...toyMap, rules: [{ pack: "x", layers: null, shape: [1], recipe: leaf("${T}missing") }] };
  await assert.rejects(LocalFoldWeights.open(join(dir, "manifest.json"), { map: bad }), /no tensor t\/missing/);
  writeFileSync(join(dir, "foreign.json"), JSON.stringify({ ...manifest, bundle: {} }));
  await assert.rejects(LocalFoldWeights.open(join(dir, "foreign.json"), { map: toyMap }), /int5 bundle/);
});

test("a recipe with the wrong declared shape is reported with the tensor name", async () => {
  const bad = { ...toyMap, rules: [{ pack: "x", layers: null, shape: [4], recipe: leaf("${T}small") }] };
  const w = await LocalFoldWeights.open(join(dir, "manifest.json"), { map: bad });
  assert.throws(() => w.get("x"), /x: recipe produced 3, expected 4/);
  w.close();
});

test("HTTP: ensure fetches byte ranges, get needs ensure, release drops the layer, small tensors stay", async () => {
  const requests = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    if (url.endsWith("manifest.json")) return new Response(JSON.stringify(manifest));
    const range = /bytes=(\d+)-(\d+)/.exec(init.headers?.Range ?? "");
    requests.push(range ? [Number(range[1]), Number(range[2]) + 1] : null);
    if (!range) return new Response(bytes, { status: 200 });
    return new Response(bytes.subarray(Number(range[1]), Number(range[2]) + 1), { status: 206 });
  };
  try {
    const w = await LocalFoldWeights.fromUrl("http://host/dir/manifest.json", { map: toyMap, smallElements: 100 });
    assert.equal(requests.length, 0, "nothing but the manifest is fetched up front");
    assert.throws(() => w.get("p/0/stack"), /ensure/);
    await w.ensure("p/1/");
    assert.ok(requests.length > 0 && requests.every((r) => r !== null), "Range requests only");
    approx(w.get("p/1/stack"), [7, 8, 9, 10, 11, 12]);
    approx(w.get("p/1/int"), expected.subarray(57, 114));
    assert.throws(() => w.get("p/0/int"), /ensure/, "layer 0 of the large tensor was not requested");
    const before = requests.length;
    await w.ensure("p/1/"); // already held
    assert.equal(requests.length, before);
    w.release("p/1/");
    assert.throws(() => w.get("p/1/int"), /ensure/, "large layer slice dropped");
    approx(w.get("p/1/stack"), [7, 8, 9, 10, 11, 12]); // small tensors were fetched whole and are kept
    approx(w.get("p/0/stack"), [1, 2, 3, 4, 5, 6]);
    await w.ensure("p/");
    approx(w.get("p/2/int"), expected.subarray(114, 171));
  } finally { globalThis.fetch = realFetch; }
});

test("HTTP: a server that ignores Range (200 with the whole file) still works", async () => {
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (input) => (String(input).endsWith("manifest.json") ? new Response(JSON.stringify(manifest)) : new Response(bytes, { status: 200 }));
  try {
    const w = await LocalFoldWeights.fromUrl("http://host/manifest.json", { map: toyMap });
    await w.ensure("p/0/");
    approx(w.get("p/0/int"), expected.subarray(0, 57));
    approx(w.get("p/0/take"), [3, 1, 6, 4]);
  } finally { globalThis.fetch = realFetch; }
});

// ---------------------------------------------------------------- the real bundle against the f32 pack

const lfDir = process.env.BOLTZ_LOCALFOLD_DIR ?? "/private/tmp/claude-501/lf-boltz2";
const lfManifest = join(lfDir, "manifest.json");
const shardDir = join(lfDir, "shards");
const packManifest = here("packs/boltz2-trunk-own/boltz2-trunk-own.json");
const realSkip = !existsSync(lfManifest) || !existsSync(join(shardDir, "weights-23.int5.bin")) || !existsSync(packManifest)
  ? "LocalFold shards or the f32 trunk pack are not present" : false;

function relL2(a, b) {
  let d = 0;
  let r = 0;
  for (let i = 0; i < a.length; i += 1) { const e = a[i] - b[i]; d += e * e; r += b[i] * b[i]; }
  return Math.sqrt(d / (r + 1e-30));
}

test("real bundle: every rule pattern (first and last layer) matches the f32 pack", { skip: realSkip }, async () => {
  const lf = await LocalFoldWeights.open(lfManifest, { shardDir });
  const own = await OwnWeights.open(packManifest);
  try {
    const map = JSON.parse(readFileSync(here("src/trunk/localfold-map.json"), "utf8"));
    const sample = new Set();
    for (const rule of map.rules) {
      sample.add(rule.pack.replace("{L}", "0"));
      if (rule.layers) sample.add(rule.pack.replace("{L}", String(rule.layers - 1)));
    }
    assert.ok(sample.size > 150, `only ${sample.size} patterns`);
    const bad = [];
    for (const name of sample) {
      assert.deepEqual(lf.shape(name), own.shape(name), name);
      const a = lf.get(name);
      const b = own.get(name);
      assert.equal(a.length, b.length, name);
      const error = relL2(a, b);
      // int5 noise is about 0.04; the one documented zero-filled LayerNorm offset is the only larger error
      if (error > 0.15 && name !== "input_embedder/atom_enc_proj_z/0/bias") bad.push(`${name}: ${error.toFixed(3)}`);
    }
    assert.deepEqual(bad, []);
  } finally { lf.close(); own.close(); }
});

test("real bundle: exactly the f32 pack's tensor names, shapes and float32 layouts", { skip: realSkip }, async () => {
  const lf = await LocalFoldWeights.open(lfManifest, { shardDir });
  const own = await OwnWeights.open(packManifest);
  try {
    assert.deepEqual([...lf.names("")].sort(), [...own.names("")].sort());
    for (const name of lf.names("")) assert.deepEqual(lf.shape(name), own.shape(name), name);
    assert.deepEqual(lf.arch, own.arch);
    assert.equal(lf.model, own.model);
    assert.ok(lf.get("s_init/weight") instanceof Float32Array);
  } finally { lf.close(); own.close(); }
});

test("real bundle: float32 tensors are bit-exact and a stacked layer slice decodes alone", { skip: realSkip }, async () => {
  const lf = await LocalFoldWeights.open(lfManifest, { shardDir });
  const own = await OwnWeights.open(packManifest);
  try {
    for (const name of ["pairformer_module/layers/17/transition_s/norm/weight", "pairformer_module/layers/63/transition_s/norm/bias", "contact_conditioning/encoder/weight", "s_norm/weight"]) {
      assert.deepEqual(Array.from(lf.get(name)), Array.from(own.get(name)), name);
    }
    const slice = lf.slice("pairformer_module/layers/5");
    assert.equal(Object.keys(slice).length, own.names("pairformer_module/layers/5/").length);
    assert.deepEqual(slice["attention/proj_q/weight"].shape, [384, 384]);
    // the unavailable tensors are zero, exactly as documented in the map
    assert.ok(lf.get("input_embedder/atom_enc_proj_z/0/bias").every((v) => v === 0));
    const res = lf.get("input_embedder/res_type_encoding/weight"); // (384, 33), column 0 is the pad class
    assert.ok(Array.from({ length: 384 }, (_, i) => res[i * 33]).every((v) => v === 0));
  } finally { lf.close(); own.close(); }
});

// ---------------------------------------------------------------- whole-shard mode (cross-origin hosts)

function mockCaches(store = new Map()) {
  const calls = { match: 0, put: 0, delete: 0 };
  const caches = {
    store, calls,
    open: async () => ({
      match: async (url) => { calls.match += 1; const e = store.get(url); return e ? new Response(e.bytes, { headers: e.headers }) : undefined; },
      put: async (url, response) => { calls.put += 1; store.set(url, { bytes: new Uint8Array(await response.arrayBuffer()), headers: Object.fromEntries(response.headers) }); },
      delete: async (url) => { calls.delete += 1; return store.delete(url); },
    }),
  };
  return caches;
}

async function withShardHost(run, { caches } = {}) {
  const realFetch = globalThis.fetch;
  const realCaches = Object.getOwnPropertyDescriptor(globalThis, "caches");
  const log = [];
  globalThis.fetch = async (input, init = {}) => {
    const url = String(input);
    log.push({ url, range: init.headers?.Range });
    if (url.endsWith("manifest.json")) return new Response(JSON.stringify(manifest));
    return new Response(bytes, { status: 200, headers: { "content-length": String(bytes.length) } });
  };
  Object.defineProperty(globalThis, "caches", { value: caches, configurable: true, writable: true });
  try { await run(log); } finally {
    globalThis.fetch = realFetch;
    if (realCaches) Object.defineProperty(globalThis, "caches", realCaches); else delete globalThis.caches;
  }
}

test("whole-shard: one GET per shard, no Range, prefetch reports progress, ensure/release are no-ops", async () => {
  const events = [];
  await withShardHost(async (log) => {
    const w = await LocalFoldWeights.fromUrl("http://host/dir/manifest.json", { map: toyMap, wholeShards: true, onProgress: (e) => events.push(e) });
    const shardGets = log.filter((r) => r.url.endsWith(SHARD));
    assert.equal(shardGets.length, 1);
    assert.ok(log.every((r) => r.range === undefined));
    assert.ok(events.length >= 2);
    const last = events.at(-1);
    assert.deepEqual(Object.keys(last).sort(), ["file", "fromCache", "loadedBytes", "shardsDone", "shardsTotal", "totalBytes"]);
    assert.equal(last.loadedBytes, bytes.length);
    assert.equal(last.shardsDone, 1);
    assert.equal(last.shardsTotal, 1);
    assert.equal(last.fromCache, false);
    await w.ensure("p/1/");
    approx(w.get("p/1/int"), expected.subarray(57, 114));
    w.release("p/1/");
    approx(w.get("p/1/int"), expected.subarray(57, 114)); // shards are shared: release keeps them
    await w.ensure("p/");
    assert.equal(log.filter((r) => r.url.endsWith(SHARD)).length, 1);
    w.close();
    assert.throws(() => w.get("p/1/int"), /ensure/);
  }, { caches: undefined });
});

test("whole-shard: Cache API is filled under the stable URL and a second open skips the network", async () => {
  const caches = mockCaches();
  await withShardHost(async (log) => {
    await LocalFoldWeights.fromUrl("http://host/dir/manifest.json", { map: toyMap, wholeShards: true });
    assert.equal(caches.calls.put, 1);
    assert.ok(caches.store.has(`http://host/dir/${SHARD}`));
    const events = [];
    const w = await LocalFoldWeights.fromUrl("http://host/dir/manifest.json", { map: toyMap, wholeShards: true, onProgress: (e) => events.push(e) });
    assert.equal(log.filter((r) => r.url.endsWith(SHARD)).length, 1, "second open read the cache");
    assert.equal(events.at(-1).fromCache, true);
    assert.equal(events.at(-1).loadedBytes, bytes.length);
    approx(w.get("p/2/int"), expected.subarray(114, 171));
  }, { caches });
});

test("whole-shard: a truncated cache entry is dropped and refetched", async () => {
  const caches = mockCaches();
  await withShardHost(async (log) => {
    caches.store.set(`http://host/dir/${SHARD}`, { bytes: bytes.subarray(0, 100), headers: { "content-length": "100", "x-localfold-size": "100" } });
    const w = await LocalFoldWeights.fromUrl("http://host/dir/manifest.json", { map: toyMap, wholeShards: true });
    assert.equal(log.filter((r) => r.url.endsWith(SHARD)).length, 1);
    assert.equal(caches.calls.delete, 1);
    assert.equal(caches.store.get(`http://host/dir/${SHARD}`).bytes.length, bytes.length);
    approx(w.get("p/0/int"), expected.subarray(0, 57));
    // an entry whose stored size disagrees with its body is also refetched
    caches.store.set(`http://host/dir/${SHARD}`, { bytes, headers: { "content-length": String(bytes.length + 5), "x-localfold-size": String(bytes.length + 5) } });
    await LocalFoldWeights.fromUrl("http://host/dir/manifest.json", { map: toyMap, wholeShards: true });
    assert.equal(log.filter((r) => r.url.endsWith(SHARD)).length, 2);
  }, { caches });
});

test("whole-shard: a failing Cache API falls back to plain fetch; prefetch:false defers the download to ensure", async () => {
  const broken = { open: async () => { throw new Error("SecurityError"); } };
  await withShardHost(async (log) => {
    const w = await LocalFoldWeights.fromUrl("http://host/manifest.json", { map: toyMap, wholeShards: true, prefetch: false });
    assert.equal(log.filter((r) => r.url.endsWith(SHARD)).length, 0);
    await w.ensure("p/0/");
    assert.equal(log.filter((r) => r.url.endsWith(SHARD)).length, 1);
    approx(w.get("p/0/int"), expected.subarray(0, 57));
  }, { caches: broken });
});
