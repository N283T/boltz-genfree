// Loader for the Boltz-2 trunk pack written by tools/convert_boltz_trunk.py.
// Tensor names are the PyTorch parameter names with "." replaced by "/"; layouts are
// untouched ((out, in) linear weights). Node: OwnWeights.open(path); browser: fromUrl(url).

import { TensorBundle } from "../bundle.js";

export class OwnWeights {
  constructor(bundle) {
    this.bundle = bundle;
    this.meta = bundle.meta;
    this.arch = bundle.meta.architecture ?? {};
    this.model = bundle.meta.model;
  }

  static async open(manifestPath) { return new OwnWeights(await TensorBundle.open(manifestPath)); }

  static async fromUrl(manifestUrl) { return new OwnWeights(await TensorBundle.fromUrl(manifestUrl)); }

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
