// Reader for the float32 tensor bundles written by webgpu/tools/tensor_bundle.py.
// The layout is LocalFold's manifest layout (shared/bundles/tensor-store.js):
// { formatVersion, meta, tensors: { name: { file, byteOffset, dtype, shape } } }.
import { readFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";

export class TensorBundle {
  constructor(manifest, tensors) {
    this.meta = manifest.meta ?? {};
    this.records = manifest.tensors;
    this.tensors = tensors;
  }

  static async open(manifestPath) {
    const path = resolve(manifestPath);
    const manifest = JSON.parse(await readFile(path, "utf8"));
    if (manifest.formatVersion !== 1 || manifest.tensors === undefined) {
      throw new Error(`${path} is not a version 1 tensor bundle`);
    }
    const files = new Map();
    const tensors = new Map();
    for (const [name, record] of Object.entries(manifest.tensors)) {
      if (record.dtype !== "float32") throw new Error(`${name}: unsupported dtype ${record.dtype}`);
      if (!files.has(record.file)) files.set(record.file, await readFile(resolve(dirname(path), record.file)));
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
