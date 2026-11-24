// The trunk -> affinity boundary (see DESIGN.md).
//
// Boltz2.forward hands the affinity module three things: s_inputs (the input
// embedding), the trunk pair representation z, and token masks. z is multiplied
// by the protein-ligand pair mask before the module sees it (z_affinity).
import { TensorBundle } from "./bundle.js";
import { affinityMasks } from "./distogram.js";

/** Multiply z [N, N, C] by the cross-pair mask [N, N], as Boltz2.forward does. */
export function maskTrunkPair(z, crossPairMask, channels) {
  const out = new Float32Array(z.length);
  for (let p = 0; p < crossPairMask.length; p++) {
    if (crossPairMask[p] === 0) continue;
    out.set(z.subarray(p * channels, (p + 1) * channels), p * channels);
  }
  return out;
}

/**
 * Module inputs from a trunk dump bundle with tensors
 *   input/s_inputs [N, S], input/z_trunk [N, N, C], input/token_pad_mask [N],
 *   input/mol_type [N], input/affinity_token_mask [N]
 * (the names `webgpu/tools/dump_oracle.py` uses for its inputs; a leading batch
 * dimension of 1 is accepted).
 */
export function inputFromBundle(bundle) {
  const toInt = (name) => Int32Array.from(bundle.get(name));
  const padMask = toInt("input/token_pad_mask");
  const molType = toInt("input/mol_type");
  const affinityTokenMask = toInt("input/affinity_token_mask");
  const shape = bundle.shape("input/z_trunk"); // [1, N, N, C] (batch of one) or [N, N, C]
  const channels = shape[shape.length - 1];
  const n = shape[shape.length - 2];
  if (shape.length === 4 && shape[0] !== 1) throw new Error("only a batch of one is supported");
  const { crossPairMask } = affinityMasks(padMask, molType, affinityTokenMask);
  if (padMask.length !== n) throw new Error(`mask length ${padMask.length} != tokens ${n}`);
  return {
    sInputs: bundle.get("input/s_inputs"),
    zAffinity: maskTrunkPair(bundle.get("input/z_trunk"), crossPairMask, channels),
    padMask, molType, affinityTokenMask,
  };
}

export async function loadTrunkInput(manifestPath) {
  return inputFromBundle(await TensorBundle.open(manifestPath));
}
