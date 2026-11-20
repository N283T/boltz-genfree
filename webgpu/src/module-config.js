// Reads the per-module configuration and weights of a genfree affinity pack
// (webgpu/tools/convert_weights.py) and resolves the tensors one module needs.

export function moduleConfig(bundle, root) {
  const config = bundle.meta.modules?.[root];
  if (config === undefined) throw new Error(`pack has no module ${root}`);
  return config;
}

/** Weight accessor: `w("pairformer_stack/layers/0/transition_z/fc1/weight")`. */
export function weightsFor(bundle, root) {
  return (path) => bundle.get(`${root}/${path}`);
}

export function boundariesOf(config) {
  return Float32Array.from(config.boundaries);
}
