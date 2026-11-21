// The whole chain: sequence (+ A3M) + SMILES -> featurisation -> Boltz-2 trunk (LocalFold, WebGPU)
// -> affinity module (this repo, WebGPU) -> affinity value and probability.
import { AffinityGpu } from "../gpu/affinity-gpu.js";
import { affinityMasks } from "../distogram.js";
import { maskTrunkPair } from "../trunk-input.js";
import { loadLocalFold } from "./localfold.js";
import { adaptWeights, loadTrunkWeights, openTrunkStore } from "./weights.js";
import { affinityBatch, buildBatch, importOracleAtomFeatures, tokenMasks } from "./featurise.js";
import { runTrunk } from "./run.js";

/** Load LocalFold, the trunk weights and the affinity pack once; score many inputs with `engine.score`. */
export async function createChain({ localfold, trunkWeights, affinityPack, device, onStatus }) {
  const lf = await loadLocalFold(localfold);
  onStatus?.("opening trunk weights");
  const { store, close } = await openTrunkStore(lf, trunkWeights);
  if (typeof store.prefetch === "function") store.prefetch();
  onStatus?.("decoding trunk weights");
  const started = performance.now();
  const weights = await loadTrunkWeights(lf, store);
  const weightLoadMs = performance.now() - started;
  const affinity = new AffinityGpu(device, affinityPack);
  return {
    lf, device, weights, weightLoadMs,
    /**
     * @param {{sequence: string, a3m?: string|null, smiles: string}} input
     * @param {{recycles?: number, precision?: "f32"|"auto", mode?: string, seed?: number, ligandConformer?: string,
     *          allowExtended?: boolean, oracle?: object, importPositions?: boolean, importNames?: boolean}} [options]
     */
    async score(input, options = {}) {
      const t0 = performance.now();
      const { batch, rows, receptorTokens, ligandTokens, notes } = await buildBatch(lf, { ...input, ...options });
      if (options.oracle) {
        importOracleAtomFeatures(batch, options.oracle, {
          refPos: options.importPositions ?? true, atomNames: options.importNames ?? true,
        });
      }
      const featuriseMs = performance.now() - t0;
      const trunk = await runTrunk(lf, device, weights, batch, {
        recycles: options.recycles ?? 0, precision: options.precision ?? "f32", seed: options.seed ?? 0,
        onStage: options.onStage,
      });
      const t1 = performance.now();
      const affinityInput = affinityBatch(batch);
      const sInputs = await lf.buildTargetFeat(affinityInput, weights.targetFeat, device);
      const masks = tokenMasks(batch);
      const { crossPairMask } = affinityMasks(masks.padMask, masks.molType, masks.affinityTokenMask);
      const zAffinity = maskTrunkPair(trunk.pair, crossPairMask, 128);
      const result = await affinity.score(
        { sInputs, zAffinity, ...masks },
        { mode: options.mode ?? "zero_distogram_features", seed: options.seed ?? 0 },
      );
      const affinityMs = performance.now() - t1;
      return {
        value: result.value, probability: result.probability, members: result.members,
        tokens: batch.tokens, receptorTokens, ligandTokens, msaRows: rows.depth, notes: options.oracle ? [] : notes,
        trunk: { pair: trunk.pair, single: trunk.single, sInputsTrunk: trunk.targetFeat, sInputsAffinity: sInputs },
        timings: { featuriseMs, ...trunk.timings, affinityMs, totalMs: performance.now() - t0 },
      };
    },
    async destroy() {
      await affinity.destroy();
      await close();
    },
  };
}

export { adaptWeights };
