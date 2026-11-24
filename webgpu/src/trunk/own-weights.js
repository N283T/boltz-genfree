// Loader for the Boltz-2 trunk pack written by tools/convert_boltz_trunk.py.
// Tensor names are the PyTorch parameter names with "." replaced by "/"; layouts are
// untouched ((out, in) linear weights). Node: OwnWeights.open(path); browser: fromUrl(url).
//
// Nothing is held in memory: every `get` reads that tensor's byte range (Node: fs.readSync on a kept file handle) and
// returns a fresh array, so the modules, which already stream weights layer by layer, never hold the whole pack.
// Browser: fromUrl(url, { lazy: true }) fetches nothing up front; `await weights.ensure(prefix)` loads one layer's
// tensors by HTTP Range request before the module reads them, `release(prefix)` drops them again.

import { TensorBundle } from "../bundle.js";

export class OwnWeights {
  constructor(bundle) {
    this.bundle = bundle;
    this.meta = bundle.meta;
    this.arch = bundle.meta.architecture ?? {};
    this.model = bundle.meta.model;
  }

  static async open(manifestPath) { return new OwnWeights(await TensorBundle.open(manifestPath, { cache: false })); }

  static async fromUrl(manifestUrl, options = {}) { return new OwnWeights(await TensorBundle.fromUrl(manifestUrl, options)); }

  /** Lazy URL packs: load every tensor under `prefix` (no-op for Node packs and eagerly fetched URL packs). */
  async ensure(prefix) { await this.bundle.ensure(this.names(prefix)); }

  release(prefix) { this.bundle.release(this.names(prefix)); }

  close() { this.bundle.close(); }

  has(name) { return this.bundle.has(name); }

  get(name) { return this.bundle.get(name); }

  shape(name) { return this.bundle.shape(name); }

  names(prefix = "") { return [...this.bundle.tensors.keys()].filter((n) => n.startsWith(prefix)); }

  /** All tensors under `prefix` as { relativeName: { data: Float32Array, shape } }. */
  slice(prefix) {
    const base = prefix.endsWith("/") ? prefix : `${prefix}/`;
    const out = {};
    for (const name of this.names(base)) {
      out[name.slice(base.length)] = { data: this.bundle.get(name), shape: this.bundle.shape(name) };
    }
    if (Object.keys(out).length === 0) throw new Error(`pack has no tensors under ${base}`);
    return out;
  }
}
