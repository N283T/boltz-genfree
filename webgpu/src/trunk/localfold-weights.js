// Boltz-2 trunk weights read straight from LocalFold's public int5 bundle (sokrypton/localfold, boltz2-int5), exposed
// with the OwnWeights interface: PyTorch parameter names with "." replaced by "/", PyTorch layouts ((out, in) linear
// weights), float32 arrays. Nothing is hosted by us and no f32 pack is needed.
//
// The bundle uses AF3/Haiku names and layouts (layer-stacked tensors, (in, out) weights, fused projections). The
// recipes that turn one into the other live in localfold-map.json, written and verified against the f32 pack by
// tools/map_localfold_weights.py (recipe language documented there). This file only evaluates them.
//
// int5 format: groups of `block` (32) consecutive elements of the row-major flattened tensor; every element is a
// 5-bit code packed LSB-first (20 bytes per group); value = code * scale[group] + zero[group], scale and zero being
// float16 arrays at scaleOffset / zeroOffset. float32 tensors are plain little-endian float32 at byteOffset.
//
// Memory stays bounded: a `get` decodes only the bytes of the requested layer slice of a stacked tensor.
//   Node:     LocalFoldWeights.open(manifestPath): shards are read by byte range with fs.readSync; `ensure`/`release`
//             are no-ops and every `get` returns a fresh array.
//   Browser:  LocalFoldWeights.fromUrl(manifestUrl): nothing but the manifest is fetched up front. `await ensure(prefix)`
//             Range-fetches the (raw, 5/32 of f32 size) bytes of every tensor under `prefix`, `get` decodes them
//             synchronously, `release(prefix)` drops them again. Small tensors are fetched whole and kept.

const ASSET_MAP = "./localfold-map.json";
const GAP = 1 << 15; // merge fetches closer than this
const SMALL_ELEMENTS = 1 << 16; // tensors up to this size are fetched whole (one request, kept for all layers; `smallElements` option)
const CONCURRENCY = 8;

/** IEEE float16 bits to number. */
export function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  if (exponent === 0) return sign * fraction * 2 ** -24;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * (1 + fraction / 1024) * 2 ** (exponent - 15);
}

/**
 * Decode `count` elements starting at element `first` of an int5 tensor.
 * @param {Uint8Array} codes bytes of groups [firstGroup, ...) (group g starts at byte (g - firstGroup) * block * 5 / 8)
 * @param {Uint8Array} scales float16 little-endian scales of groups [firstGroup, ...)
 * @param {Uint8Array} zeros   float16 little-endian zero points of the same groups
 */
export function decodeInt5(codes, scales, zeros, first, count, block = 32) {
  const groupBytes = (block * 5) / 8;
  if (!Number.isInteger(groupBytes)) throw new Error(`int5 at block ${block} does not pack into whole bytes`);
  const firstGroup = Math.floor(first / block);
  const lastGroup = Math.ceil((first + count) / block);
  if (codes.length < (lastGroup - firstGroup) * groupBytes) throw new Error("int5: codes too short");
  if (scales.length < (lastGroup - firstGroup) * 2 || zeros.length < (lastGroup - firstGroup) * 2) throw new Error("int5: scales or zeros too short");
  const out = new Float32Array(count);
  const end = first + count;
  for (let group = firstGroup; group < lastGroup; group += 1) {
    const g = group - firstGroup;
    const scale = halfToFloat(scales[2 * g] | (scales[2 * g + 1] << 8));
    const zero = halfToFloat(zeros[2 * g] | (zeros[2 * g + 1] << 8));
    const start = Math.max(group * block, first);
    const stop = Math.min((group + 1) * block, end);
    let bit = g * groupBytes * 8 + (start - group * block) * 5;
    for (let index = start; index < stop; index += 1) {
      const byte = bit >> 3;
      // codes[byte + 1] is undefined past the last byte of a tensor, which reads as 0 here
      const value = ((codes[byte] | (codes[byte + 1] << 8)) >> (bit & 7)) & 31;
      out[index - first] = value * scale + zero;
      bit += 5;
    }
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ tiny ndarray ops

const size = (shape) => shape.reduce((a, b) => a * b, 1);

function reshape(a, shape) {
  const known = size(shape.filter((d) => d !== -1));
  const out = shape.map((d) => (d === -1 ? a.data.length / known : d));
  if (size(out) !== a.data.length) throw new Error(`cannot reshape ${a.shape} to ${shape}`);
  return { data: a.data, shape: out };
}

function transpose(a, perm) {
  const rank = a.shape.length;
  const shape = perm.map((p) => a.shape[p]);
  const inStrides = new Array(rank);
  let stride = 1;
  for (let d = rank - 1; d >= 0; d -= 1) { inStrides[d] = stride; stride *= a.shape[d]; }
  const strides = perm.map((p) => inStrides[p]);
  const out = new Float32Array(a.data.length);
  const index = new Array(rank).fill(0);
  const last = rank - 1;
  const lastStride = strides[last];
  const lastSize = shape[last];
  let source = 0;
  for (let o = 0; o < out.length; o += lastSize) {
    for (let i = 0, s = source; i < lastSize; i += 1, s += lastStride) out[o + i] = a.data[s];
    // advance the odometer over all but the last axis
    for (let d = last - 1; d >= 0; d -= 1) {
      index[d] += 1;
      source += strides[d];
      if (index[d] < shape[d]) break;
      source -= strides[d] * shape[d];
      index[d] = 0;
    }
  }
  return { data: out, shape };
}

/** out[..., k, ...] = a[..., indices[k], ...] along `axis`. */
function gather(a, axis, indices) {
  const outer = size(a.shape.slice(0, axis));
  const inner = size(a.shape.slice(axis + 1));
  const n = a.shape[axis];
  const shape = a.shape.slice();
  shape[axis] = indices.length;
  const out = new Float32Array(outer * indices.length * inner);
  for (let o = 0; o < outer; o += 1) {
    for (let k = 0; k < indices.length; k += 1) {
      const from = (o * n + indices[k]) * inner;
      out.set(a.data.subarray(from, from + inner), (o * indices.length + k) * inner);
    }
  }
  return { data: out, shape };
}

function concat(parts, axis) {
  const outer = size(parts[0].shape.slice(0, axis));
  const inner = size(parts[0].shape.slice(axis + 1));
  const total = parts.reduce((sum, p) => sum + p.shape[axis], 0);
  const shape = parts[0].shape.slice();
  shape[axis] = total;
  const out = new Float32Array(outer * total * inner);
  let offset = 0;
  for (const p of parts) {
    const chunk = p.shape[axis] * inner;
    for (let o = 0; o < outer; o += 1) out.set(p.data.subarray(o * chunk, (o + 1) * chunk), (o * total + offset) * inner);
    offset += p.shape[axis];
  }
  return { data: out, shape };
}

function applyOps(a, ops = []) {
  let x = a;
  for (const [kind, ...args] of ops) {
    if (kind === "reshape") x = reshape(x, args[0]);
    else if (kind === "transpose") x = transpose(x, args[0]);
    else if (kind === "slice") x = gather(x, args[0], Array.from({ length: args[2] - args[1] }, (_, i) => args[1] + i));
    else if (kind === "flip") x = gather(x, args[0], Array.from({ length: x.shape[args[0]] }, (_, i) => x.shape[args[0]] - 1 - i));
    else if (kind === "take") x = gather(x, args[0], args[1]);
    else if (kind === "scale") { const out = new Float32Array(x.data.length); for (let i = 0; i < out.length; i += 1) out[i] = x.data[i] * args[0]; x = { data: out, shape: x.shape }; }
    else throw new Error(`unknown recipe op ${kind}`);
  }
  return x;
}

// ------------------------------------------------------------------------------------------------ byte sources

/** Sync reads of byte ranges from local shard files (Node). */
class NodeSource {
  constructor(fs, resolve) { this.fs = fs; this.resolve = resolve; this.files = new Map(); }

  handle(file) {
    let entry = this.files.get(file);
    if (entry === undefined) {
      const fd = this.fs.openSync(this.resolve(file), "r");
      entry = { fd, size: this.fs.fstatSync(fd).size };
      this.files.set(file, entry);
    }
    return entry;
  }

  read(file, start, end) {
    const { fd, size: fileSize } = this.handle(file);
    if (start < 0 || end > fileSize) throw new Error(`${file}: range ${start}-${end} outside the file (${fileSize} bytes)`);
    const out = new Uint8Array(end - start);
    let done = 0;
    while (done < out.length) {
      const n = this.fs.readSync(fd, out, done, out.length - done, start + done);
      if (n === 0) throw new Error(`${file}: unexpected end of file`);
      done += n;
    }
    return out;
  }

  close() { for (const { fd } of this.files.values()) this.fs.closeSync(fd); this.files.clear(); }
}

/** Range-fetched byte blocks kept by (file, start, end), reference counted per ensured tensor (browser). */
class BlockCache {
  constructor(base) { this.base = base; this.blocks = new Map(); } // file -> block[]

  find(file, start, end) {
    return (this.blocks.get(file) ?? []).find((b) => b.start <= start && end <= b.end);
  }

  read(file, start, end) {
    const block = this.find(file, start, end);
    if (block === undefined) return undefined;
    return block.bytes.subarray(start - block.start, end - block.start);
  }

  async fetchRange(file, start, end) {
    const response = await fetch(new URL(file, this.base), { headers: { Range: `bytes=${start}-${end - 1}` } });
    if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
    const buffer = new Uint8Array(await response.arrayBuffer());
    // a server that ignores Range answers 200 with the whole file
    if (response.status !== 206) return buffer.subarray(start, end);
    if (buffer.length !== end - start) throw new Error(`${file}: short HTTP response`);
    return buffer;
  }

  /** Make sure every range is held by some block; returns nothing. Ranges are merged when close. */
  async load(ranges) {
    const missing = ranges.filter((r) => this.find(r.file, r.start, r.end) === undefined)
      .sort((a, b) => (a.file < b.file ? -1 : a.file > b.file ? 1 : a.start - b.start) || Number(Boolean(a.permanent)) - Number(Boolean(b.permanent)));
    const merged = [];
    for (const r of missing) {
      const prev = merged[merged.length - 1];
      if (prev && prev.file === r.file && Boolean(prev.permanent) === Boolean(r.permanent) && r.start - prev.end <= GAP) {
        prev.end = Math.max(prev.end, r.end);
      } else merged.push({ ...r });
    }
    let next = 0;
    const worker = async () => {
      while (next < merged.length) {
        const job = merged[next++];
        const bytes = await this.fetchRange(job.file, job.start, job.end);
        const list = this.blocks.get(job.file) ?? [];
        list.push({ start: job.start, end: job.end, bytes, refs: 0, permanent: Boolean(job.permanent) });
        this.blocks.set(job.file, list);
      }
    };
    await Promise.all(Array.from({ length: Math.min(CONCURRENCY, merged.length) }, worker));
  }

  retain(file, start, end) {
    const block = this.find(file, start, end);
    if (block === undefined) throw new Error(`${file}: range ${start}-${end} was not fetched`);
    block.refs += 1;
    return block;
  }

  drop(block, file) {
    block.refs -= 1;
    if (block.refs > 0 || block.permanent) return;
    const list = this.blocks.get(file);
    const at = list.indexOf(block);
    if (at >= 0) list.splice(at, 1);
  }
}


/**
 * Whole-shard source for cross-origin hosts (Hugging Face): every needed shard file is fetched in full with one GET
 * (ranged per-tensor requests cost ~100-300 ms each there) and kept in memory; persisted with the Cache API under the
 * stable shard URL. Same interface as BlockCache, but `release` never drops shard bytes (they are shared).
 */
class ShardStore {
  constructor(base, records, { onProgress, cacheName = "localfold-int5", totalBytes } = {}) {
    this.base = base;
    this.onProgress = onProgress;
    this.cacheName = cacheName;
    this.shards = new Map(); // file -> Uint8Array
    this.pending = new Map(); // file -> Promise
    this.expected = new Map(); // file -> minimum byte size implied by the manifest
    for (const r of Object.values(records)) {
      const count = size(r.shape);
      const groups = Math.ceil(count / (r.block ?? 32));
      const end = r.dtype === "float32" ? r.byteOffset + 4 * count
        : Math.max(r.byteOffset + groups * (((r.block ?? 32) * 5) / 8), r.scaleOffset + 2 * groups, r.zeroOffset + 2 * groups);
      this.expected.set(r.file, Math.max(this.expected.get(r.file) ?? 0, end));
    }
    this.files = [...this.expected.keys()];
    this.totalBytes = totalBytes ?? [...this.expected.values()].reduce((a, b) => a + b, 0);
    this.loadedBytes = 0;
    this.shardsDone = 0;
  }

  url(file) { return new URL(file, this.base).href; }

  progress(file, fromCache) {
    this.onProgress?.({ loadedBytes: this.loadedBytes, totalBytes: this.totalBytes, file, shardsDone: this.shardsDone, shardsTotal: this.files.length, fromCache });
  }

  find(file, start, end) { const b = this.shards.get(file); return b && end <= b.length ? b : undefined; }

  read(file, start, end) { return this.shards.get(file)?.subarray(start, end); }

  retain(file, start, end) {
    if (this.find(file, start, end) === undefined) throw new Error(`${file}: range ${start}-${end} was not fetched`);
    return {};
  }

  drop() {}

  close() { this.shards.clear(); this.pending.clear(); this.loadedBytes = 0; this.shardsDone = 0; }

  async load(ranges) { await this.loadFiles([...new Set(ranges.map((r) => r.file))]); }

  /** Download (or read from the Cache API) these shards, at most 4 at a time. */
  async loadFiles(files) {
    const todo = files.filter((f) => !this.shards.has(f));
    let next = 0;
    const worker = async () => {
      while (next < todo.length) {
        const file = todo[next++];
        if (!this.pending.has(file)) this.pending.set(file, this.#shard(file).finally(() => this.pending.delete(file)));
        await this.pending.get(file);
      }
    };
    await Promise.all(Array.from({ length: Math.min(4, todo.length) }, worker));
  }

  async #open() {
    try { return globalThis.caches ? await globalThis.caches.open(this.cacheName) : null; } catch { return null; }
  }

  async #shard(file) {
    const url = this.url(file);
    const min = this.expected.get(file);
    const cache = await this.#open();
    if (cache) {
      try {
        const hit = await cache.match(url);
        if (hit) {
          const bytes = new Uint8Array(await hit.arrayBuffer());
          const stored = Number(hit.headers.get("x-localfold-size") ?? hit.headers.get("content-length"));
          if (bytes.length >= min && bytes.length === stored) return this.#done(file, bytes, true);
          try { await cache.delete(url); } catch { /* ignore */ }
        }
      } catch { /* fall through to the network */ }
    }
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
    const length = Number(response.headers.get("content-length"));
    let bytes;
    if (response.body?.getReader && length > 0) {
      bytes = new Uint8Array(length);
      const reader = response.body.getReader();
      let at = 0;
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        if (at + value.length > length) throw new Error(`${file}: longer than its content-length`);
        bytes.set(value, at);
        at += value.length;
        this.loadedBytes += value.length;
        this.progress(file, false);
      }
      if (at !== length) throw new Error(`${file}: truncated download (${at} of ${length} bytes)`);
    } else {
      bytes = new Uint8Array(await response.arrayBuffer());
      this.loadedBytes += bytes.length;
    }
    if (bytes.length < min) throw new Error(`${file}: ${bytes.length} bytes, the manifest needs at least ${min}`);
    if (cache) {
      try {
        await cache.put(url, new Response(bytes, { headers: { "content-length": String(bytes.length), "x-localfold-size": String(bytes.length) } }));
      } catch { /* quota or private window: keep going without persistence */ }
    }
    return this.#done(file, bytes, false, true);
  }

  #done(file, bytes, fromCache, counted = false) {
    this.shards.set(file, bytes);
    if (!counted) this.loadedBytes += bytes.length;
    this.shardsDone += 1;
    this.progress(file, fromCache);
    return bytes;
  }
}

// ------------------------------------------------------------------------------------------------ the loader

function expandAlias(src, aliases) {
  return src.replace(/\$\{(\w+)\}/g, (_, key) => {
    if (!(key in aliases)) throw new Error(`recipe alias ${key} is not defined`);
    return aliases[key];
  });
}

function* leaves(node) {
  if ("src" in node) yield node;
  for (const part of node.parts ?? node.add ?? []) yield* leaves(part);
}

export class LocalFoldWeights {
  /**
   * @param {object} manifest the LocalFold manifest ({ tensors: { name: { file, shape, byteOffset, dtype, ... } } })
   * @param {object} map parsed localfold-map.json
   * @param {{ smallElements?: number }} [options]
   * @param {{ read?: (file, start, end) => Uint8Array, cache?: BlockCache, close?: () => void }} source
   */
  constructor(manifest, map, source, { smallElements = SMALL_ELEMENTS } = {}) {
    this.smallElements = smallElements;
    if (map.version !== 1) throw new Error("unsupported localfold-map version");
    if (manifest.bundle?.encoding !== "int5-le" && manifest.quantisation?.bits !== 5) {
      throw new Error("not a LocalFold int5 bundle manifest");
    }
    this.records = manifest.tensors;
    this.map = map;
    this.source = source;
    this.meta = { ...map.meta, source: manifest.source, bundle: manifest.bundle };
    this.arch = map.meta.architecture ?? {};
    this.model = map.meta.model;
    this.entries = new Map(); // pack name -> { name, shape, recipe, layer }
    for (const rule of map.rules) {
      const count = rule.layers ?? 1;
      for (let layer = 0; layer < count; layer += 1) {
        const name = rule.pack.replace("{L}", String(layer));
        this.entries.set(name, { name, shape: rule.shape, recipe: rule.recipe, layer: rule.layers === null ? null : layer });
      }
      for (const node of leaves(rule.recipe)) {
        const src = expandAlias(node.src, map.aliases);
        if (!(src in this.records)) throw new Error(`bundle has no tensor ${src} (needed for ${rule.pack})`);
      }
    }
    this.ensured = new Map(); // pack name -> [{ block, file }]
  }

  /** Node: read shards by byte range from the directory of `manifestPath` (or `options.shardDir`). */
  static async open(manifestPath, options = {}) {
    const fs = await import("node:fs");
    const { dirname, resolve } = await import("node:path");
    const path = resolve(manifestPath);
    const manifest = JSON.parse(await fs.promises.readFile(path, "utf8"));
    const map = options.map ?? JSON.parse(await fs.promises.readFile(new URL(ASSET_MAP, import.meta.url), "utf8"));
    const base = options.shardDir ? resolve(options.shardDir) : dirname(path);
    const source = new NodeSource(fs, (file) => resolve(base, file));
    return new LocalFoldWeights(manifest, map, { read: (file, start, end) => source.read(file, start, end), close: () => source.close() }, options);
  }

  /**
   * Browser (or Node >= 18) over HTTP; shard URLs are relative to `manifestUrl` (or `options.baseUrl`).
   * Always lazy: call `await weights.ensure(prefix)` before reading tensors. `lazy: false` fetches the whole bundle.
   */
  static async fromUrl(manifestUrl, options = {}) {
    const { lazy = true, map, baseUrl, onProgress } = options;
    const url = new URL(manifestUrl, globalThis.location?.href);
    const manifest = await (await fetch(url)).json();
    const resolved = map ?? await (await fetch(new URL(ASSET_MAP, import.meta.url))).json();
    const base = baseUrl ? new URL(baseUrl, globalThis.location?.href) : url;
    // cross-origin hosts (Hugging Face) charge ~100-300 ms per request: fetch whole shards there; same-origin servers use Range
    const cross = globalThis.location !== undefined && base.origin !== globalThis.location.origin;
    const wholeShards = options.wholeShards ?? cross;
    const cache = wholeShards
      ? new ShardStore(base, manifest.tensors, { onProgress, cacheName: options.cacheName, totalBytes: manifest.bundle?.bytes })
      : new BlockCache(base);
    const weights = new LocalFoldWeights(manifest, resolved, { cache }, options);
    weights.wholeShards = wholeShards;
    if (wholeShards && (options.prefetch ?? true)) await cache.loadFiles(cache.files);
    if (!lazy) await weights.ensure("");
    return weights;
  }

  // ---- OwnWeights interface

  /** Lazy URL bundles: Range-fetch every tensor under `prefix` (no-op for Node). */
  async ensure(prefix) {
    const cache = this.source.cache;
    if (cache === undefined) return;
    const todo = this.names(prefix).filter((name) => !this.ensured.has(name));
    const plans = new Map(todo.map((name) => [name, this.#plan(this.entries.get(name))]));
    await cache.load([...plans.values()].flatMap((plan) => plan.fetch));
    for (const [name, plan] of plans) {
      this.ensured.set(name, plan.need.map((r) => ({ file: r.file, block: cache.retain(r.file, r.start, r.end) })));
    }
  }

  release(prefix) {
    const cache = this.source.cache;
    if (cache === undefined) return;
    for (const name of this.names(prefix)) {
      const held = this.ensured.get(name);
      if (held === undefined) continue;
      for (const { file, block } of held) cache.drop(block, file);
      this.ensured.delete(name);
    }
  }

  close() { this.source.close?.(); this.source.cache?.close?.(); this.ensured.clear(); }

  has(name) { return this.entries.has(name); }

  get(name) { return this.#evaluate(this.#entry(name)).data; }

  shape(name) { return this.#entry(name).shape; }

  names(prefix = "") { return [...this.entries.keys()].filter((n) => n.startsWith(prefix)); }

  /** All tensors under `prefix` as { relativeName: { data: Float32Array, shape } }. */
  slice(prefix) {
    const base = prefix.endsWith("/") ? prefix : `${prefix}/`;
    const out = {};
    for (const name of this.names(base)) out[name.slice(base.length)] = { data: this.get(name), shape: this.shape(name) };
    if (Object.keys(out).length === 0) throw new Error(`pack has no tensors under ${base}`);
    return out;
  }

  // ---- internals

  #entry(name) {
    const entry = this.entries.get(name);
    if (entry === undefined) throw new Error(`bundle has no tensor ${name}`);
    return entry;
  }

  /** Byte layout of one leaf: the element range it needs and the byte ranges to read. */
  #leafLayout(node, layer) {
    const name = expandAlias(node.src, this.map.aliases);
    const record = this.records[name];
    const total = size(record.shape);
    const perLayer = node.layer ? total / record.shape[0] : total;
    const first = node.layer ? layer * perLayer : 0;
    if (node.layer && !(layer >= 0 && layer < record.shape[0])) throw new Error(`${name}: layer ${layer} out of range`);
    const shape = node.layer ? record.shape.slice(1) : record.shape;
    if (record.dtype === "float32") {
      const need = [{ file: record.file, start: record.byteOffset + 4 * first, end: record.byteOffset + 4 * (first + perLayer) }];
      const whole = [{ file: record.file, start: record.byteOffset, end: record.byteOffset + 4 * total }];
      return { record, shape, first, count: perLayer, total, need, whole };
    }
    if (record.dtype !== "int5") throw new Error(`${name}: unsupported dtype ${record.dtype}`);
    const { block = 32 } = record;
    const groupBytes = (block * 5) / 8;
    const groups = Math.ceil(total / block);
    const g0 = Math.floor(first / block);
    const g1 = Math.ceil((first + perLayer) / block);
    const range = (at, unit, a, b) => ({ file: record.file, start: at + unit * a, end: at + unit * b });
    const need = [range(record.byteOffset, groupBytes, g0, g1), range(record.scaleOffset, 2, g0, g1), range(record.zeroOffset, 2, g0, g1)];
    const whole = [range(record.byteOffset, groupBytes, 0, groups), range(record.scaleOffset, 2, 0, groups), range(record.zeroOffset, 2, 0, groups)];
    return { record, shape, first, count: perLayer, total, need, whole, g0, block };
  }

  #walk(node, layer, visit) {
    if ("src" in node) visit(this.#leafLayout(node, layer));
    for (const part of node.parts ?? node.add ?? []) this.#walk(part, layer, visit);
  }

  /** Ranges one pack tensor reads (`need`) and what to fetch for them (small tensors are fetched whole and kept). */
  #plan(entry) {
    const need = [];
    const fetch = [];
    this.#walk(entry.recipe, entry.layer, (layout) => {
      need.push(...layout.need);
      if (layout.total <= this.smallElements) fetch.push(...layout.whole.map((r) => ({ ...r, permanent: true })));
      else fetch.push(...layout.need);
    });
    return { need, fetch };
  }

  #bytes(range, name) {
    const direct = this.source.read;
    if (direct) return direct(range.file, range.start, range.end);
    const bytes = this.source.cache.read(range.file, range.start, range.end);
    if (bytes === undefined) throw new Error(`${name}: not loaded, await weights.ensure(prefix) first`);
    return bytes;
  }

  #leaf(node, layer, name) {
    const layout = this.#leafLayout(node, layer);
    const { record, shape, count } = layout;
    const [codes, scale, zero] = layout.need.map((range) => this.#bytes(range, name));
    if (record.dtype === "float32") {
      const data = new Float32Array(count);
      new Uint8Array(data.buffer).set(codes);
      return { data, shape };
    }
    // decodeInt5 wants `first` relative to the first fetched group
    return { data: decodeInt5(codes, scale, zero, layout.first - layout.g0 * layout.block, count, layout.block), shape };
  }

  #node(node, layer, name) {
    let x;
    if ("src" in node) x = this.#leaf(node, layer, name);
    else if ("zeros" in node) x = { data: new Float32Array(size(node.zeros)), shape: node.zeros };
    else if ("concat" in node) x = concat(node.parts.map((p) => this.#node(p, layer, name)), node.concat);
    else if ("add" in node) {
      const [a, b] = node.add.map((p) => this.#node(p, layer, name));
      if (a.data.length !== b.data.length) throw new Error(`${name}: add of different sizes`);
      for (let i = 0; i < a.data.length; i += 1) a.data[i] += b.data[i];
      x = a;
    } else throw new Error(`${name}: bad recipe node`);
    return applyOps(x, node.ops);
  }

  #evaluate(entry) {
    const { name } = entry;
    const result = this.#node(entry.recipe, entry.layer, name);
    if (result.shape.join("x") !== entry.shape.join("x")) {
      throw new Error(`${name}: recipe produced ${result.shape}, expected ${entry.shape}`);
    }
    return result;
  }
}
