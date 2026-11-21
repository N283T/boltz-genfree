// Run LocalFold's WebGPU Boltz-2 trunk on a batch and read back the tensors the affinity module needs.
// The driver mirrors webgpu/af3/fold.js (foldHolding) for the trunk part only: input embedding
// (`target_feat`, which is Boltz's s_inputs), template/MSA/pairformer passes, optional recycling.

const F32 = { stagedPrecision: "f32", weightPrecision: "f32", accumulatePrecision: "f32", pairWeightPrecision: "f32" };

/**
 * @param {object} lf loadLocalFold result
 * @param {GPUDevice} device
 * @param {{trunk: object, targetFeat: object}} weights adapted weights (weights.js)
 * @param {object} batch from buildBatch
 * @param {{recycles?: number, precision?: "f32"|"auto", seed?: number, onStage?: Function, msaCap?: number}} [options]
 * @returns {Promise<{pair: Float32Array, single: Float32Array, sInputs: Float32Array, timings: object}>}
 */
export async function runTrunk(lf, device, weights, batch, options = {}) {
  const { recycles = 0, precision = "f32", seed = 0, onStage, msaCap = 1024 } = options;
  const timings = {};
  const mark = async (name, work) => {
    const start = performance.now();
    const value = await work();
    timings[name] = (timings[name] ?? 0) + performance.now() - start;
    onStage?.(name, timings[name]);
    return value;
  };
  const tokens = batch.tokens;

  const targetFeat = await mark("input-embedding", () => lf.buildTargetFeat(batch, weights.targetFeat, device));
  const pairMask = new Float32Array(tokens * tokens);
  for (let i = 0; i < tokens; i += 1) {
    for (let j = 0; j < tokens; j += 1) pairMask[i * tokens + j] = batch.seqMask[i] * batch.seqMask[j];
  }
  const knobs = precision === "f32" ? F32 : { pairWeightPrecision: lf.defaultPairWeightPrecision(128) };
  const gpu = new lf.Af3TrunkGpu(device, { ...knobs, residentWeights: true });

  let previousPair;
  let previousSingle;
  let trunk;
  for (let pass = 0; pass <= recycles; pass += 1) {
    const chosen = lf.msaDrawRows(batch.numMsa === undefined ? "prefix" : "pass", seed, pass, batch.sequences, msaCap);
    const rows = chosen.length;
    const gather = (array, Type, fill) => {
      const out = new Type(rows * tokens).fill(fill);
      for (let r = 0; r < rows; r += 1) out.set(array.subarray(chosen[r] * tokens, (chosen[r] + 1) * tokens), r * tokens);
      return out;
    };
    trunk = await mark(`trunk-pass-${pass}`, () => gpu.run({
      tokens, sequences: rows, templates: 1,
      targetFeat, features: batch.features,
      msaRows: gather(batch.msa, Int32Array, 21),
      deletionMatrix: gather(batch.deletionMatrix, Float32Array, 0),
      msaMask: gather(batch.msaMask, Float32Array, 0),
      msaQueryDrawn: chosen[0] === 0,
      isDna: batch.isDna,
      bondMatrix: batch.bondMatrix, bondOrderMatrix: batch.bondOrderMatrix,
      asymId: batch.asymId,
      pairMask, seqMask: batch.seqMask, previousPair, previousSingle,
      contactClasses: lf.af3ContactClasses(batch, tokens),
    }, weights.trunk, weights.trunk.dialect, {
      keepOutputs: false, readback: { pair: true, single: true, logits: false },
    }));
    previousPair = trunk.pair;
    previousSingle = trunk.single;
  }
  return { pair: trunk.pair, single: trunk.single, targetFeat, timings };
}
