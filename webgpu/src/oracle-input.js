import { TensorBundle } from "./bundle.js";

/** Load an oracle dump and turn its `input/*` tensors into module inputs. */
export async function loadOracle(dumpManifest) {
  const dump = await TensorBundle.open(dumpManifest);
  const toInt = (name) => Int32Array.from(dump.get(name));
  const input = {
    sInputs: dump.get("input/s_inputs"),
    zAffinity: dump.get("input/z_affinity"),
    padMask: toInt("input/token_pad_mask"),
    molType: toInt("input/mol_type"),
    affinityTokenMask: toInt("input/affinity_token_mask"),
  };
  return { dump, input };
}
