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

/**
 * @param {object} lf loadLocalFold result
 * @param {{sequence: string, a3m?: string|null, smiles: string, ligandConformer?: "skip"|"conformer",
 *          ligandAtoms?: {smiles: string, atomNames: string[]}, allowExtended?: boolean, maxMsa?: number}} input
 */
export async function buildBatch(lf, input) {
  const { sequence, a3m = null, smiles, ligandConformer = "skip", allowExtended = false } = input;
  if (!/^[A-Za-z]+$/.test(sequence)) throw new Error("sequence must contain letters only (one protein chain)");
  const dialect = lf.dialectFor("boltz2");
  const notes = [];
  // GenFree standardises the SMILES with RDKit (the atoms then follow the canonical SMILES) and names the
  // atoms from RDKit's canonical ranks; `ligandAtoms` (tools/ligand_atoms.py) carries both.
  let ligandAtoms = input.ligandAtoms;
  if (ligandAtoms === undefined) {
    const { ligandAtomsFromSmiles } = await import("./canonical.js");
    const derived = ligandAtomsFromSmiles(smiles);
    ligandAtoms = { smiles: derived.smiles, atomNames: derived.atomNames };
    if (!derived.exact) {
      notes.push(`ligand atom order/names are approximate (${derived.warnings.join("; ") || "outside the RDKit-free subset"}); `
        + "stereo ligands need tools/ligand_atoms.py (--ligand-atoms) to match GenFree exactly");
    }
  }
  const ligand = await lf.smilesComponent(ligandAtoms?.smiles ?? smiles, {});
  const alignment = a3m ? { paired: null, unpaired: normalizeA3m(a3m, input.maxMsaSeqs) } : null;
  const { batch, rows } = lf.af3BatchFromA3m(sequence.toUpperCase(), alignment, {
    ...(input.maxMsa === undefined ? {} : { maxSequences: input.maxMsa }),
    ...lf.featuriserDialect(dialect),
    ligands: [ligand],
  });
  const tokens = batch.tokens;
  const ligandSpan = batch.ligandSpans[0];
  const receptorTokens = ligandSpan.from;
  const ligandTokens = ligandSpan.count;
  if (receptorTokens !== sequence.length || receptorTokens + ligandTokens !== tokens) {
    throw new Error(`unexpected tokenisation: ${receptorTokens} receptor + ${ligandTokens} ligand != ${tokens} tokens`);
  }
  checkSize({ receptorTokens, ligandTokens, atoms: batch.atomCount }, { allowExtended });

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

  const dense = batch.dense;
  if (ligandConformer === "skip") {
    for (let token = receptorTokens; token < tokens; token += 1) {
      batch.refPos.fill(0, token * dense * 3, (token + 1) * dense * 3);
    }
  } else if (ligandConformer !== "conformer") {
    throw new Error(`ligand conformer mode ${ligandConformer} is not available here (skip or conformer)`);
  }
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
