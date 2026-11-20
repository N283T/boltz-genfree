// Structure-free affinity distogram bins, generated without coordinates.
// Mirrors build_structure_free_affinity_distogram_bins and
// intervene_affinity_distogram_features in
// src/boltz/model/modules/affinity_distogram.py.
//
// The five structure-free modes (STRUCTURE_FREE_AFFINITY_MODES):
//   zero_distogram_features  bins all 0; embedded features replaced by zeros   (GenFree default)
//   uniform_mean_embedding   bins all 0; embedded features replaced by the mean embedding row
//   zero_conditioning        bins all 0; the pairwise conditioner is skipped
//   uniform_random           symmetric uniform random bin per valid pair
//   normal_random            symmetric N(mean, std) distance per valid pair, then binned

export const STRUCTURE_FREE_MODES = Object.freeze([
  "uniform_random",
  "normal_random",
  "zero_distogram_features",
  "uniform_mean_embedding",
  "zero_conditioning",
]);

/** MT19937, the generator behind torch.Generator on CPU. */
export class MT19937 {
  constructor(seed) {
    this.mt = new Uint32Array(624);
    this.index = 624;
    this.mt[0] = seed >>> 0;
    for (let i = 1; i < 624; i++) {
      const prev = this.mt[i - 1] ^ (this.mt[i - 1] >>> 30);
      this.mt[i] = (Math.imul(1812433253, prev) + i) >>> 0;
    }
  }

  next32() {
    const mt = this.mt;
    if (this.index >= 624) {
      for (let k = 0; k < 624; k++) {
        const y = (mt[k] & 0x80000000) | (mt[(k + 1) % 624] & 0x7fffffff);
        mt[k] = mt[(k + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      this.index = 0;
    }
    let y = mt[this.index++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  }
}

/** Valid-pair mask: both tokens are receptor or ligand (padding excluded). */
export function validPairMask(receptor, ligand) {
  const n = receptor.length;
  const mask = new Uint8Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      mask[i * n + j] = (receptor[i] || ligand[i]) && (receptor[j] || ligand[j]) ? 1 : 0;
    }
  }
  return mask;
}

function binOf(distance, boundaries) {
  let count = 0;
  for (const b of boundaries) if (distance > b) count++;
  return count;
}

/**
 * @param {{boundaries: number[], receptor: ArrayLike<number>, ligand: ArrayLike<number>,
 *          mode: string, seed?: number, normalMean?: number, normalStd?: number}} options
 * @returns {{bins: Uint32Array, validPairMask: Uint8Array, features: "embed"|"zero"|"mean",
 *            conditioning: boolean}}
 */
export function buildStructureFreeBins(options) {
  const { boundaries, receptor, ligand, mode, seed = 0, normalMean = 12, normalStd = 4 } = options;
  if (!STRUCTURE_FREE_MODES.includes(mode)) {
    throw new Error(`mode ${JSON.stringify(mode)} is not coordinate-free; expected one of: ${STRUCTURE_FREE_MODES.join(", ")}`);
  }
  if (receptor.length !== ligand.length) throw new Error("receptor and ligand masks must have the same length");
  if (mode === "normal_random" && !(normalStd > 0)) throw new Error("normal_std must be positive");
  const n = receptor.length;
  const valid = validPairMask(receptor, ligand);
  const bins = new Uint32Array(n * n);
  const base = {
    validPairMask: valid,
    features: mode === "zero_distogram_features" ? "zero" : mode === "uniform_mean_embedding" ? "mean" : "embed",
    conditioning: mode !== "zero_conditioning",
  };
  if (mode !== "uniform_random" && mode !== "normal_random") return { bins, ...base };

  const numBins = boundaries.length + 1;
  const rng = new MT19937(seed);
  // Symmetric by construction: sample a full matrix like the reference, keep the
  // strict upper triangle and mirror it. The diagonal stays 0.
  const sampled = new Uint32Array(n * n);
  if (mode === "uniform_random") {
    // torch.randint(0, numBins) on CPU: one 32-bit draw per element, `draw % range`.
    for (let k = 0; k < n * n; k++) sampled[k] = rng.next32() % numBins;
  } else {
    // Box-Muller from the same MT stream. NOT bit-identical to torch.randn
    // (torch uses a vectorised batched transform); statistically equivalent.
    for (let k = 0; k < n * n; k += 2) {
      const u1 = (rng.next32() + 1) / 4294967297;
      const u2 = (rng.next32() + 1) / 4294967297;
      const r = Math.sqrt(-2 * Math.log(u1));
      const z0 = r * Math.cos(2 * Math.PI * u2);
      const z1 = r * Math.sin(2 * Math.PI * u2);
      sampled[k] = binOf(Math.max(0, z0 * normalStd + normalMean), boundaries);
      if (k + 1 < n * n) sampled[k + 1] = binOf(Math.max(0, z1 * normalStd + normalMean), boundaries);
    }
  }
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const bin = valid[i * n + j] ? sampled[i * n + j] : 0;
      bins[i * n + j] = bin;
      bins[j * n + i] = bin;
    }
  }
  return { bins, ...base };
}

/** Pair weights used by pool_affinity_pairs and the pairformer mask. */
export function affinityMasks(padMask, molType, affinityTokenMask) {
  const n = padMask.length;
  const receptor = new Uint8Array(n);
  const ligand = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    receptor[i] = molType[i] === 0 && padMask[i] ? 1 : 0;
    ligand[i] = affinityTokenMask[i] && padMask[i] ? 1 : 0;
  }
  const cross = new Float32Array(n * n); // includes the ligand-ligand diagonal (pairformer mask)
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      cross[i * n + j] =
        (ligand[i] && receptor[j]) || (receptor[i] && ligand[j]) || (ligand[i] && ligand[j]) ? 1 : 0;
    }
  }
  const pool = new Float32Array(cross); // pooling excludes the diagonal
  for (let i = 0; i < n; i++) pool[i * n + i] = 0;
  return { receptor, ligand, crossPairMask: cross, poolMask: pool };
}
