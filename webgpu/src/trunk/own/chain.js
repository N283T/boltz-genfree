// The whole chain without LocalFold: sequence (+ A3M) + SMILES -> own featuriser -> own embedder -> recycling ->
// own MSA module -> own pairformer -> affinity module -> affinity value and probability.
// Same contract as ../chain.js: `createOwnChain(...)` returns { score(input, options), destroy() }.
import { AffinityGpu } from "../../gpu/affinity-gpu.js";
import { TensorBundle } from "../../bundle.js";
import { affinityMasks } from "../../distogram.js";
import { maskTrunkPair } from "../../trunk-input.js";
import { OwnWeights } from "../own-weights.js";
import { featuriseOwn, featureBundle } from "./featurise.js";
import { OwnTrunk } from "./run.js";

/** `base` features with the named `feat/*` tensors taken from `oracle` (same shapes required). */
function overlay(base, oracle, names) {
  const replaced = new Set(names.map((name) => `feat/${name}`));
  for (const name of replaced) {
    const mine = base.shape(name).join("x");
    const theirs = oracle.shape(name).join("x");
    if (mine !== theirs) throw new Error(`oracle ${name} has shape ${theirs}, the own featuriser made ${mine}`);
  }
  return {
    has: (name) => base.has(name),
    shape: (name) => base.shape(name),
    get: (name) => (replaced.has(name) ? oracle.get(name) : base.get(name)),
  };
}

/**
 * @param {{trunkPack: OwnWeights|string, affinityPack: TensorBundle|string, device: GPUDevice, onStatus?: Function}} args
 *   `trunkPack`: an OwnWeights or the path of packs/boltz2-trunk-own/boltz2-trunk-own.json; `affinityPack`: a TensorBundle
 *   or the path of packs/genfree-affinity.json.
 */
export async function createOwnChain({ trunkPack, affinityPack, device, onStatus }) {
  onStatus?.("opening packs");
  const weights = typeof trunkPack === "string" ? await OwnWeights.open(trunkPack) : trunkPack;
  const bundle = typeof affinityPack === "string" ? await TensorBundle.open(affinityPack) : affinityPack;
  const trunkEngine = new OwnTrunk(device, weights);
  const affinity = new AffinityGpu(device, bundle);
  return {
    device, weights,
    /**
     * @param {{sequence: string, a3m?: string|null, smiles: string}} input
     * @param {{recycles?: number, mode?: string, seed?: number, allowExtended?: boolean, maxMsa?: number,
     *          ligandAtoms?: object, oracle?: object, importPositions?: boolean, importNames?: boolean,
     *          onStage?: Function, taps?: Map}} [options]
     *   `oracle` (a dump bundle) replaces the protein reference positions and the atom-name one-hots with GenFree's PyTorch
     *   ones (the 'exact' path: separates engine error from the featuriser's seeded-augmentation difference).
     */
    async score(input, options = {}) {
      const t0 = performance.now();
      const { oracle, recycles = 0, mode = "zero_distogram_features", seed = 0, ...rest } = options;
      const made = featuriseOwn({ ...input, ...rest, seed });
      let features = featureBundle(made);
      if (oracle) {
        const names = [];
        if (options.importPositions ?? true) names.push("ref_pos");
        if (options.importNames ?? true) names.push("ref_atom_name_chars");
        features = overlay(features, oracle, names);
      }
      const featuriseMs = performance.now() - t0;
      const trunk = await trunkEngine.run(features, { recycles, seed, onStage: options.onStage, taps: options.taps });
      const t1 = performance.now();
      const padMask = Int32Array.from(features.get("feat/token_pad_mask"), (v) => (v > 0 ? 1 : 0));
      const molType = Int32Array.from(features.get("feat/mol_type"));
      const affinityTokenMask = Int32Array.from(features.get("feat/affinity_token_mask"));
      const { crossPairMask } = affinityMasks(padMask, molType, affinityTokenMask);
      const zAffinity = maskTrunkPair(trunk.pair, crossPairMask, 128);
      const result = await affinity.score({ sInputs: trunk.sInputsAffinity, zAffinity, padMask, molType, affinityTokenMask }, { mode, seed });
      const affinityMs = performance.now() - t1;
      return {
        value: result.value, probability: result.probability, members: result.members,
        tokens: made.tokens, receptorTokens: made.receptorTokens, ligandTokens: made.ligandTokens, msaRows: made.msaRows,
        notes: oracle ? [] : made.notes,
        trunk: { pair: trunk.pair, single: trunk.single, sInputsTrunk: trunk.sInputsTrunk, sInputsAffinity: trunk.sInputsAffinity },
        timings: { featuriseMs, ...trunk.timings, affinityMs, totalMs: performance.now() - t0 },
      };
    },
    async destroy() {
      await affinity.destroy();
      await trunkEngine.destroy();
    },
  };
}
