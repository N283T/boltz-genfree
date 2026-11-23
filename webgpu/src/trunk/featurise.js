// Protein sequence (+ optional A3M) and a ligand SMILES -> the batch LocalFold's Boltz-2 trunk
// consumes, adapted to the feature route GenFree uses (src/boltz/stfree_runtime.py):
//
//   - one protein chain "A" and one ligand "B" given as SMILES, ligand tokens = heavy atoms
//   - `override_method="other"` (handled in weights.js), no templates, no constraints
//   - `ligand_reference_conformer: skip`: ligand `ref_pos` is zeroed (the default). The other
//     GenFree modes (`etkdg`, `rdkit_2d`) need RDKit; `conformer` here uses LocalFold's own
//     JavaScript conformer builder as an approximation of `etkdg` (not bit-comparable).
//   - Boltz's `type_bonds` is the RDKit bond type plus one (0 = no bond); LocalFold's bond-order
//     plane holds the bond order itself for the same embedding table, so every bonded entry gets +1
//   - the affinity input embedding uses `profile_affinity` / `deletion_mean_affinity`: the one-hot of
//     the query residue types and zeros, not the full-MSA profile the trunk used
//   - the size limits of `crop_full_screen` (docs/usage.md)
//
// Everything below the batch (atoms, tokens, MSA rows) is LocalFold's featuriser.

export const LIMITS = { receptorTokens: 200, ligandTokens: 128, totalTokens: 256, totalAtoms: 2048 };
const RESTYPES = 31;

/** Throw GenFree's size errors; `allowExtended` lifts the receptor/total/atom limits (not the ligand's). */
export function checkSize({ receptorTokens, ligandTokens, atoms }, { allowExtended = false } = {}) {
  if (ligandTokens > LIMITS.ligandTokens) {
    throw new Error(`ligand has ${ligandTokens} tokens, the limit is ${LIMITS.ligandTokens}`);
  }
  if (allowExtended) return;
  const total = receptorTokens + ligandTokens;
  if (receptorTokens > LIMITS.receptorTokens || total > LIMITS.totalTokens || atoms > LIMITS.totalAtoms) {
    throw new Error(`input has ${receptorTokens} receptor + ${ligandTokens} ligand tokens (${atoms} atoms); without `
      + `--allow-extended GenFree requires receptor <= ${LIMITS.receptorTokens}, total <= ${LIMITS.totalTokens} `
      + `and atoms <= ${LIMITS.totalAtoms}, and nothing is cropped silently`);
  }
}

/**
 * Boltz's A3M parser (src/boltz/data/parse/a3m.py): comment lines are skipped, a row whose residues
 * (gaps removed, upper-cased, insertions kept) repeat an earlier row is dropped, and at most
 * `maxSeqs` rows are kept (GenFree's `max_msa_seqs`, default 8192).
 */
export function normalizeA3m(text, maxSeqs = 8192) {
  const out = [];
  const seen = new Set();
  let header = ">query";
  let kept = 0;
  for (const raw of text.split("\n")) {
    const line = raw.trim().replaceAll("\0", "");
    if (line === "" || line.startsWith("#")) continue;
    if (line.startsWith(">")) { header = line; continue; }
    const key = line.replaceAll("-", "").toUpperCase();
    if (seen.has(key)) continue;
    seen.add(key);
    out.push(header, line);
    kept += 1;
    if (kept >= maxSeqs) break;
  }
  return out.join("\n") + "\n";
}

/** Deterministic PRNG (mulberry32) for the seeded ligand rotation. */
function rng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gaussian(random) {
  return Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
}

/** Random rotation matrix from a uniform unit quaternion. */
function randomRotation(random) {
  let q = [gaussian(random), gaussian(random), gaussian(random), gaussian(random)];
  const norm = Math.hypot(...q);
  q = q.map((v) => v / norm);
  const [w, x, y, z] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

/**
 * Ligand reference conformer modes (GenFree `ligand_reference_conformer`).
 *  skip     ref_pos = 0 (GenFree's default; exact).
 *  etkdg    a conformer from LocalFold's JavaScript embedder (seeded by `etkdgSeed`), centred and given a seeded
 *           random rotation and translation, as Boltz's `center_random_augmentation` does to RDKit's ETKDG one.
 *  rdkit_2d the same conformer flattened onto its principal plane (z = 0, a stand-in for RDKit's 2D depiction), then
 *           rotated and translated the same way.
 * Neither reproduces RDKit's coordinates or torch's random stream; see docs/webgpu.md for the measured effect.
 */
export function applyLigandConformer(batch, first, count, mode, seed) {
  const dense = batch.dense;
  const base = first * dense * 3;
  if (mode === "skip") { batch.refPos.fill(0, base, (first + count) * dense * 3); return; }
  const pos = [];
  for (let t = 0; t < count; t += 1) pos.push(Array.from(batch.refPos.slice((first + t) * dense * 3, (first + t) * dense * 3 + 3)));
  const centre = [0, 1, 2].map((k) => pos.reduce((a, p) => a + p[k], 0) / count);
  let work = pos.map((p) => p.map((v, k) => v - centre[k]));
  if (mode === "rdkit_2d") {
    // principal axes by power iteration on the covariance, drop the smallest axis
    const cov = [0, 1, 2].map((a) => [0, 1, 2].map((b) => work.reduce((s, p) => s + p[a] * p[b], 0)));
    const axes = [];
    for (let k = 0; k < 3; k += 1) {
      let v = [1, 0.3 + k, 0.1];
      for (let it = 0; it < 100; it += 1) {
        v = [0, 1, 2].map((a) => cov[a].reduce((s, c, b) => s + c * v[b], 0));
        for (const u of axes) { const d = v.reduce((s, x, i) => s + x * u[i], 0); v = v.map((x, i) => x - d * u[i]); }
        const n = Math.hypot(...v) || 1;
        v = v.map((x) => x / n);
      }
      axes.push(v);
    }
    work = work.map((p) => [0, 1].map((a) => p.reduce((s, x, i) => s + x * axes[a][i], 0)).concat([0]));
  }
  const random = rng(seed);
  const rotation = randomRotation(random);
  const shift = [gaussian(random), gaussian(random), gaussian(random)];
  work.forEach((p, t) => {
    const out = [0, 1, 2].map((a) => rotation[a].reduce((s, r, k) => s + r * p[k], 0) + shift[a]);
    batch.refPos.set(out, (first + t) * dense * 3);
  });
}

/** Column-crop an A3M to the query positions in `keep` (insertions before a dropped column go with it). */
export function cropA3m(text, keep) {
  const wanted = new Set(keep);
  const out = [];
  for (const line of text.split("\n")) {
    if (line === "" || line.startsWith(">")) { out.push(line); continue; }
    let column = 0;
    let pending = "";
    let result = "";
    for (const ch of line) {
      if (ch !== "-" && ch >= "a" && ch <= "z") { pending += ch; continue; }
      if (wanted.has(column)) result += pending + ch;
      pending = "";
      column += 1;
    }
    out.push(result);
  }
  return out.join("\n");
}

/**
 * @param {object} lf loadLocalFold result
 * @param {{sequence: string, a3m?: string|null, smiles: string, ligandKind?: "smiles"|"ccd",
 *          ligandConformer?: "skip"|"etkdg"|"rdkit_2d"|"conformer", etkdgSeed?: number, conformerSeed?: number,
 *          receptorIndices?: number[], ligandAtoms?: {smiles: string, atomNames: string[]},
 *          allowExtended?: boolean, maxMsa?: number, maxMsaSeqs?: number}} input
 *   `receptorIndices` (sorted residue indices, a reference crop) keeps only those residues; their original indices
 *   stay in the relative-position features, as in Boltz's crop.
 */
export async function buildBatch(lf, input) {
  const { sequence, a3m = null, smiles, allowExtended = false, ligandKind = "smiles" } = input;
  const ligandConformer = input.ligandConformer === "conformer" ? "etkdg" : (input.ligandConformer ?? "skip");
  if (!["skip", "etkdg", "rdkit_2d"].includes(ligandConformer)) throw new Error(`unknown ligand conformer mode ${ligandConformer}`);
  if (!/^[A-Za-z]+$/.test(sequence)) throw new Error("sequence must contain letters only (one protein chain)");
  const dialect = lf.dialectFor("boltz2");
  const notes = [];
  const fullSequence = sequence.toUpperCase();
  const indices = input.receptorIndices ?? [...fullSequence].map((_, i) => i);
  const kept = indices.map((i) => fullSequence[i]).join("");

  let ligand;
  let ligandAtoms = input.ligandAtoms;
  if (ligandKind === "ccd") {
    const response = await fetch(lf.ccdUrl(smiles));
    if (!response.ok) throw new Error(`CCD component ${smiles}: HTTP ${response.status}`);
    ligand = lf.parseCcdComponent(await response.text());
    ligandAtoms = undefined;
  } else {
    // GenFree standardises the SMILES with RDKit (the atoms then follow the canonical SMILES) and names the
    // atoms from RDKit's canonical ranks; `ligandAtoms` (tools/ligand_atoms.py) carries both.
    if (ligandAtoms === undefined) {
      const { ligandAtomsFromSmiles } = await import("./canonical.js");
      const derived = ligandAtomsFromSmiles(smiles);
      ligandAtoms = { smiles: derived.smiles, atomNames: derived.atomNames };
      if (!derived.exact) {
        notes.push(`ligand atom order/names are approximate (${derived.warnings.join("; ") || "outside the RDKit-free subset"}); `
          + "stereo ligands need tools/ligand_atoms.py (--ligand-atoms) to match GenFree exactly");
      }
    }
    ligand = await lf.smilesComponent(ligandAtoms.smiles, input.etkdgSeed === undefined ? {} : { seed: input.etkdgSeed });
  }

  // Unknown residues: Boltz's UNK has the five atoms N CA C O CB, LocalFold's only four; featurise X as alanine
  // (the same five atoms) and re-label the token, its query MSA column and its profile afterwards.
  const unknown = [];
  [...kept].forEach((letter, i) => { if (!/[ACDEFGHIKLMNPQRSTVWY]/.test(letter)) unknown.push(i); });
  const featSequence = [...kept].map((letter, i) => (unknown.includes(i) ? "A" : letter)).join("");

  let alignment = null;
  if (a3m) {
    let text = normalizeA3m(a3m, input.maxMsaSeqs);
    if (input.receptorIndices !== undefined) text = cropA3m(text, indices);
    const lines = text.split("\n");
    lines[1] = featSequence;
    alignment = { paired: null, unpaired: lines.join("\n") };
  }
  const { batch, rows } = lf.af3BatchFromA3m(featSequence, alignment, {
    ...(input.maxMsa === undefined ? {} : { maxSequences: input.maxMsa }),
    ...lf.featuriserDialect(dialect),
    ligands: [ligand],
  });
  const tokens = batch.tokens;
  const ligandSpan = batch.ligandSpans[0];
  const receptorTokens = ligandSpan.from;
  const ligandTokens = ligandSpan.count;
  if (receptorTokens !== kept.length || receptorTokens + ligandTokens !== tokens) {
    throw new Error(`unexpected tokenisation: ${receptorTokens} receptor + ${ligandTokens} ligand != ${tokens} tokens`);
  }
  if (input.receptorIndices === undefined) checkSize({ receptorTokens, ligandTokens, atoms: batch.atomCount }, { allowExtended });
  else if (ligandTokens > LIMITS.ligandTokens) checkSize({ receptorTokens: 0, ligandTokens, atoms: 0 });

  // the LocalFold batch carries the query twice (paired and unpaired block); Boltz's alignment holds it once
  if (batch.sequences >= 2) {
    const same = Array.from({ length: receptorTokens }, (_, t) => batch.msa[tokens + t] === batch.msa[t]).every(Boolean);
    if (same) {
      const drop = (array, Type) => { const out = new Type(array.length - tokens); out.set(array.subarray(0, tokens)); out.set(array.subarray(2 * tokens), tokens); return out; };
      batch.msa = drop(batch.msa, batch.msa.constructor);
      batch.msaMask = drop(batch.msaMask, batch.msaMask.constructor);
      batch.deletionMatrix = drop(batch.deletionMatrix, batch.deletionMatrix.constructor);
      const before = batch.sequences;
      batch.sequences = before - 1;
      rows.depth = Math.max(1, rows.depth);
      // the profile was already taken over the de-duplicated rows; re-labelled X tokens move one query weight
      batch.dedupedFrom = before;
    }
  }
  const unk = 20;
  for (const t of unknown) {
    const old = batch.aatype[t];
    batch.aatype[t] = unk;
    batch.msa[t] = unk;
    const weight = 1 / batch.sequences;
    batch.profile[t * RESTYPES + old] -= weight;
    batch.profile[t * RESTYPES + unk] += weight;
  }

  // original residue positions of a cropped receptor
  if (input.receptorIndices !== undefined) {
    const { residueIndex } = batch.features;
    indices.forEach((original, t) => { residueIndex[t] = original + 1; });
  }

  if (ligandAtoms !== undefined) {
    if (ligandAtoms.atomNames.length !== ligandTokens) {
      throw new Error(`ligand atom names: ${ligandAtoms.atomNames.length} given, ${ligandTokens} ligand atoms`);
    }
    ligandAtoms.atomNames.forEach((name, i) => {
      const base = ((receptorTokens + i) * batch.dense) * 4;
      for (let k = 0; k < 4; k += 1) batch.refAtomNameChars[base + k] = k < name.length ? name.charCodeAt(k) - 32 : 0;
    });
  }

  // Boltz's type_bonds = bond type + 1
  for (let i = 0; i < batch.bondOrderMatrix.length; i += 1) if (batch.bondMatrix[i] !== 0) batch.bondOrderMatrix[i] += 1;

  applyLigandConformer(batch, receptorTokens, ligandTokens, ligandConformer, (input.conformerSeed ?? input.etkdgSeed ?? 42) >>> 0);
  return { batch, rows, receptorTokens, ligandTokens, notes };
}

/** Per-token masks the affinity module reads (token_pad_mask, mol_type, affinity_token_mask). */
export function tokenMasks(batch) {
  const padMask = Int32Array.from(batch.seqMask, (v) => (v > 0 ? 1 : 0));
  const molType = Int32Array.from(batch.isLigand, (v) => (v ? 3 : 0));
  const affinityTokenMask = Int32Array.from(batch.isLigand, (v) => (v ? 1 : 0));
  return { padMask, molType, affinityTokenMask };
}

/** The batch with the affinity input embedding's profile: one-hot of the query residue types, zero deletion mean. */
export function affinityBatch(batch) {
  const profile = new Float32Array(batch.tokens * RESTYPES);
  for (let token = 0; token < batch.tokens; token += 1) {
    const code = batch.aatype[token];
    if (code >= 0 && code < RESTYPES) profile[token * RESTYPES + code] = 1;
  }
  return { ...batch, profile, deletionMean: new Float32Array(batch.tokens) };
}

/**
 * Parity aid: replace atom features with those GenFree's PyTorch featuriser produced (a dump written by
 * tools/dump_trunk_oracle.py). Used to separate engine differences from featuriser differences: Boltz-2
 * rotates and shifts every residue's reference conformer with a seeded torch RNG and takes ligand atom
 * names from RDKit's canonical ranks, which this port does not reproduce.
 */
export function importOracleAtomFeatures(batch, bundle, { refPos = true, atomNames = true } = {}) {
  const tokens = batch.tokens;
  const dense = batch.dense;
  const pad = bundle.get("feat/atom_pad_mask");
  const toToken = bundle.get("feat/atom_to_token");
  const pos = bundle.get("feat/ref_pos");
  const chars = bundle.get("feat/ref_atom_name_chars"); // [atoms, 4, 64] one-hot
  const slots = new Int32Array(tokens);
  for (let atom = 0; atom < pad.length; atom += 1) {
    if (pad[atom] === 0) continue;
    let token = -1;
    for (let t = 0; t < tokens; t += 1) {
      if (toToken[atom * tokens + t] > 0.5) { token = t; break; }
    }
    const slot = slots[token]++;
    if (refPos) for (let k = 0; k < 3; k += 1) batch.refPos[(token * dense + slot) * 3 + k] = pos[atom * 3 + k];
    if (atomNames) {
      for (let k = 0; k < 4; k += 1) {
        let best = 0;
        for (let c = 1; c < 64; c += 1) if (chars[(atom * 4 + k) * 64 + c] > chars[(atom * 4 + k) * 64 + best]) best = c;
        batch.refAtomNameChars[(token * dense + slot) * 4 + k] = best;
      }
    }
  }
}
