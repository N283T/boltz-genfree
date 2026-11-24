// Reader for the float32 tensor bundles written by webgpu/tools/tensor_bundle.py.
// The layout follows the tensor-manifest layout of LocalFold (sokrypton/localfold, shared/bundles/tensor-store.js):
// { formatVersion, meta, tensors: { name: { file, byteOffset, dtype, shape } } }.
//
// Reads are lazy and by byte range: opening a bundle reads only the manifest, `get(name)` reads that tensor's bytes
// straight into a fresh, aligned Float32Array (no whole-file buffer, no second copy).
//   cache: true  (default for `open`)  the first read is kept, later `get` calls return the same array
//   cache: false (OwnWeights)          every `get` reads again; the caller owns the array and it can be freed at once,
//                                      so streaming a multi-hundred-MB pack costs only one tensor at a time
// Browser: `fromUrl(url)` fetches whole files as before; `fromUrl(url, { lazy: true })` fetches nothing up front and
// uses HTTP Range requests: `await bundle.ensure(names)` loads those tensors, `get` then returns them synchronously,
// and `release(names)` drops them again.

class Entry {
  constructor(record, read, cache) {
    this.shape = record.shape;
    this.count = record.shape.reduce((a, b) => a * b, 1);
    this.record = record;
    this.read = read;
    this.cache = cache;
    this.value = null;
  }

  get data() {
    if (this.value !== null) return this.value;
    const data = this.read(this);
    if (this.cache) this.value = data;
    return data;
  }
}

export class TensorBundle {
  constructor(manifest, tensors, closer = null) {
    this.meta = manifest.meta ?? {};
    this.records = manifest.tensors;
    this.tensors = tensors;
    this.closer = closer;
  }

  /** Open a bundle from disk (Node). `cache: false` re-reads a tensor on every `get` (weights streamed per layer). */
  static async open(manifestPath, { cache = true } = {}) {
    // Node-only imports are lazy so this module also loads in a browser.
    const fs = await import("node:fs");
    const { dirname, resolve } = await import("node:path");
    const path = resolve(manifestPath);
    const manifest = JSON.parse(await fs.promises.readFile(path, "utf8"));
    const base = dirname(path);
    const files = new Map(); // file -> { fd, size }
    const handle = (file) => {
      let entry = files.get(file);
      if (entry === undefined) {
        const fd = fs.openSync(resolve(base, file), "r");
        entry = { fd, size: fs.fstatSync(fd).size };
        files.set(file, entry);
      }
      return entry;
    };
    const read = (entry) => {
      const { record, count } = entry;
      const file = handle(record.file);
      if (record.byteOffset + count * 4 > file.size) throw new Error(`tensor points outside ${record.file}`);
      const out = new Float32Array(count);
      const view = new Uint8Array(out.buffer);
      let done = 0;
      while (done < view.length) {
        const n = fs.readSync(file.fd, view, done, view.length - done, record.byteOffset + done);
        if (n === 0) throw new Error(`${record.file}: unexpected end of file`);
        done += n;
      }
      return out;
    };
    const closer = () => { for (const { fd } of files.values()) fs.closeSync(fd); files.clear(); };
    const bundle = TensorBundle.#build(manifest, path, read, cache, closer);
    // Fail at open (not at first use) when a record points outside its file.
    for (const [name, entry] of bundle.tensors) {
      const { record } = entry;
      if (record.byteOffset + entry.count * 4 > handle(record.file).size) throw new Error(`${name} points outside ${record.file}`);
    }
    return bundle;
  }

  /**
   * Open a bundle over HTTP (browser or Node >= 18). `manifestUrl` may be relative to the page.
   * With `lazy: true` nothing but the manifest is fetched; call `await ensure(names)` before `get`.
   */
  static async fromUrl(manifestUrl, { lazy = false } = {}) {
    const base = new URL(manifestUrl, globalThis.location?.href);
    const manifest = await (await fetch(base)).json();
    if (lazy) {
      const loaded = new Map(); // name -> Float32Array
      const bundle = TensorBundle.#build(manifest, String(base), (entry) => {
        const data = loaded.get(entry.name);
        if (data === undefined) throw new Error(`${entry.name}: not loaded, await bundle.ensure([name]) first`);
        return data;
      }, false, null);
      for (const [name, entry] of bundle.tensors) entry.name = name;
      bundle.loaded = loaded;
      bundle.url = base;
      return bundle;
    }
    const files = new Map();
    const read = (entry) => {
      const { record, count } = entry;
      const bytes = files.get(record.file);
      const end = record.byteOffset + count * 4;
      if (end > bytes.byteLength) throw new Error(`tensor points outside ${record.file}`);
      const copy = new Float32Array(count);
      new Uint8Array(copy.buffer).set(bytes.subarray(record.byteOffset, end));
      return copy;
    };
    for (const record of Object.values(manifest.tensors)) {
      if (files.has(record.file)) continue;
      const response = await fetch(new URL(record.file, base));
      if (!response.ok) throw new Error(`${record.file}: HTTP ${response.status}`);
      files.set(record.file, new Uint8Array(await response.arrayBuffer()));
    }
    return TensorBundle.#build(manifest, String(base), read, true, null);
  }

  static #build(manifest, label, read, cache, closer) {
    if (manifest.formatVersion !== 1 || manifest.tensors === undefined) {
      throw new Error(`${label} is not a version 1 tensor bundle`);
    }
    const tensors = new Map();
    for (const [name, record] of Object.entries(manifest.tensors)) {
      if (record.dtype !== "float32") throw new Error(`${name}: unsupported dtype ${record.dtype}`);
      tensors.set(name, new Entry(record, read, cache));
    }
    return new TensorBundle(manifest, tensors, closer);
  }

  /** Lazy URL bundles: fetch the byte ranges of `names` (adjacent tensors of one file are merged into one request). */
  async ensure(names) {
    if (this.loaded === undefined) return;
    const wanted = names.filter((name) => !this.loaded.has(name)).map((name) => {
      const entry = this.tensors.get(name);
      if (entry === undefined) throw new Error(`bundle has no tensor ${name}`);
      return { name, entry };
    }).sort((a, b) => (a.entry.record.file < b.entry.record.file ? -1 : a.entry.record.file > b.entry.record.file ? 1 : a.entry.record.byteOffset - b.entry.record.byteOffset));
    const gap = 1 << 16;
    let i = 0;
    while (i < wanted.length) {
      const file = wanted[i].entry.record.file;
      const start = wanted[i].entry.record.byteOffset;
      let end = start + wanted[i].entry.count * 4;
      let j = i + 1;
      while (j < wanted.length && wanted[j].entry.record.file === file && wanted[j].entry.record.byteOffset - end <= gap) {
        end = Math.max(end, wanted[j].entry.record.byteOffset + wanted[j].entry.count * 4);
        j += 1;
      }
      const response = await fetch(new URL(file, this.url), { headers: { Range: `bytes=${start}-${end - 1}` } });
      if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
      const buffer = await response.arrayBuffer();
      // a server that ignores Range answers 200 with the whole file
      const offset = response.status === 206 ? 0 : start;
      for (const { name, entry } of wanted.slice(i, j)) {
        const from = entry.record.byteOffset - start + offset;
        if (from + entry.count * 4 > buffer.byteLength) throw new Error(`${name}: short HTTP response`);
        const data = new Float32Array(entry.count);
        new Uint8Array(data.buffer).set(new Uint8Array(buffer, from, entry.count * 4));
        this.loaded.set(name, data);
      }
      i = j;
    }
  }

  /** Drop lazily loaded URL tensors. */
  release(names) { for (const name of names) this.loaded?.delete(name); }

  /** Close file handles (Node). Reads after `close` reopen them. */
  close() { this.closer?.(); }

  has(name) { return this.tensors.has(name); }

  get(name) {
    const tensor = this.tensors.get(name);
    if (tensor === undefined) throw new Error(`bundle has no tensor ${name}`);
    return tensor.data;
  }

  shape(name) {
    const tensor = this.tensors.get(name);
    if (tensor === undefined) throw new Error(`bundle has no tensor ${name}`);
    return tensor.shape;
  }
}
