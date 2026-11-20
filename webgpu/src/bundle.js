// Reader for the float32 tensor bundles written by webgpu/tools/tensor_bundle.py.
// The layout is LocalFold's manifest layout (shared/bundles/tensor-store.js):
// { formatVersion, meta, tensors: { name: { file, byteOffset, dtype, shape } } }.

export class TensorBundle {
  constructor(manifest, tensors) {
    this.meta = manifest.meta ?? {};
    this.records = manifest.tensors;
    this.tensors = tensors;
  }

  /** Open a bundle from disk (Node). */
  static async open(manifestPath) {
    // Node-only imports are lazy so this module also loads in a browser.
    const { readFile } = await import("node:fs/promises");
    const { dirname, resolve } = await import("node:path");
    const path = resolve(manifestPath);
    const manifest = JSON.parse(await readFile(path, "utf8"));
    return TensorBundle.#load(manifest, path, async (file) => new Uint8Array(await readFile(resolve(dirname(path), file))));
  }

  /** Open a bundle over HTTP (browser or Node >= 18). `manifestUrl` may be relative to the page. */
  static async fromUrl(manifestUrl) {
    const base = new URL(manifestUrl, globalThis.location?.href);
    const manifest = await (await fetch(base)).json();
    return TensorBundle.#load(manifest, String(base), async (file) => {
      const response = await fetch(new URL(file, base));
      if (!response.ok) throw new Error(`${file}: HTTP ${response.status}`);
      return new Uint8Array(await response.arrayBuffer());
    });
  }

  static async #load(manifest, label, readBytes) {
    if (manifest.formatVersion !== 1 || manifest.tensors === undefined) {
      throw new Error(`${label} is not a version 1 tensor bundle`);
    }
    const files = new Map();
    const tensors = new Map();
    for (const [name, record] of Object.entries(manifest.tensors)) {
      if (record.dtype !== "float32") throw new Error(`${name}: unsupported dtype ${record.dtype}`);
      if (!files.has(record.file)) files.set(record.file, await readBytes(record.file));
      const bytes = files.get(record.file);
      const count = record.shape.reduce((a, b) => a * b, 1);
      const end = record.byteOffset + count * 4;
      if (end > bytes.byteLength) throw new Error(`${name} points outside ${record.file}`);
      // Copy: the file buffer's byteOffset is not 4-aligned in general.
      const copy = new Float32Array(count);
      new Uint8Array(copy.buffer).set(bytes.subarray(record.byteOffset, end));
      tensors.set(name, { data: copy, shape: record.shape });
    }
    return new TensorBundle(manifest, tensors);
  }

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
