// A pure-JS Boltz-2 featuriser for GenFree's route: one protein chain plus exactly one SMILES ligand, an optional A3M.
// It reproduces the tensors Boltz's `Boltz2Featurizer` hands the trunk and the affinity module (names, shapes and
// values of the `feat/*` tensors in webgpu/oracle-dumps/*-chain.json), without importing LocalFold. The code is
// written from Boltz's featurizerv2.py; the conformer tables in conformers.js come from Boltz's residue molecules
// (conformer 0 agrees with LocalFold's boltz2-conformers.js, Beer-Ware licence, attribution in that file).
//
// Route settings (src/boltz/stfree_runtime.py): `override_method="other"`, no templates, no constraints, protein
// reference conformers from Boltz's residue molecules, ligand `ref_pos` zero (`ligand_reference_conformer: skip`).
//
// Layout: token t < receptorTokens is residue t (20 amino acids; anything else is UNK with the atoms N CA C O CB);
// token receptorTokens + i is ligand heavy atom i. Atoms are the compact list in token order, padded with zeros to a
// multiple of 32. Every tensor is Float32Array with a leading batch dimension of 1, as in the oracle dumps.
//
// Known differences from Boltz (see test/own-featurise.test.js):
//   * protein `ref_pos`: Boltz picks one of the ten stored conformers of each residue at random, centres it and
//     applies a random rotation and translation, all from numpy/torch generators; here the pick, rotation and
//     translation come from a seeded JS generator (`refPosAugment: "none"` keeps the centred conformer; `conformers`
//     pins the picks). Given the same pick, pairwise distances inside a residue are identical to Boltz's.
//   * ligand `ref_pos`: 0 under `skip` (default); `etkdg` / `rdkit_2d` use ./conformer.js (approximate, seeded).

import { PROTEIN_CONFORMERS } from "./conformers.js";
import {
  ELEMENT_OF_LETTER, METHOD_OTHER, MOL_NONPOLYMER, MOL_PROTEIN, NUM_BACKBONE_CLASSES, NUM_ELEMENTS, NUM_TOKENS,
  PROTEIN_BACKBONE, RESIDUE_CHARGES, THREE_LETTER, TOKEN_ID, TOKEN_UNK, atomNameCodes, chiralityOf, letterToTokenId,
} from "./boltz-const.js";
import { ligandConformerPositions } from "./conformer.js";
import { ligandGraph, ligandGraphFromCcd } from "./ligand.js";
import { buildMsa } from "./msa.js";
import { atomWindows } from "./windows.js";

/** GenFree's `crop_full_screen` limits (docs/usage.md), as in ../featurise.js. */
export const LIMITS = { receptorTokens: 200, ligandTokens: 128, totalTokens: 256, totalAtoms: 2048 };
const WINDOW = 32;

/** One-letter code -> residue name used for the conformer table; anything unknown is UNK. */
function residueName(letter) {
  const token = letterToTokenId(letter);
  if (token === undefined || token === TOKEN_UNK || token === 1) return "UNK";
  return THREE_LETTER[token - 2];
}

// mulberry32 + Box-Muller: a small seeded generator (the same recipe as ../featurise.js, kept separate here)
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
const gaussian = (random) => Math.sqrt(-2 * Math.log(1 - random())) * Math.cos(2 * Math.PI * random());
function randomRotation(random) {
  const q = [gaussian(random), gaussian(random), gaussian(random), gaussian(random)];
  const norm = Math.hypot(...q);
  const [w, x, y, z] = q.map((v) => v / norm);
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ];
}

/**
 * @param {{sequence: string, smiles: string, a3m?: string|null, ligandAtoms?: {smiles: string, atomNames: string[]},
 *          receptorIndices?: number[], ligandKind?: "smiles"|"ccd", ligandComponent?: object,
 *          allowExtended?: boolean, maxMsa?: number, maxMsaSeqs?: number, seed?: number,
 *          refPosAugment?: "seeded"|"none", conformers?: number[], ligandConformer?: "skip"|"etkdg"|"rdkit_2d",
 *          conformerSeed?: number}} input
 *   `maxMsa` caps the MSA rows of the result (default 8192, Boltz's `max_msa_seqs`; no subsampling to 1024 here).
 * @returns {{features: Record<string, {data: Float32Array, shape: number[]}>, tokens: number, atoms: number,
 *            atomsPadded: number, conformerIndex: Int32Array, receptorTokens: number, ligandTokens: number, msaRows: number, notes: string[],
 *            compact: object, windows: object}}
 */
export function featuriseOwn(input) {
  const { sequence, smiles, a3m = null, allowExtended = false } = input;
  const ligandConformer = input.ligandConformer ?? "skip";
  if (!["skip", "etkdg", "rdkit_2d"].includes(ligandConformer)) {
    throw new Error(`unknown ligand conformer mode ${ligandConformer} (skip, etkdg or rdkit_2d)`);
  }
  if (!/^[A-Za-z]+$/.test(sequence)) throw new Error("sequence must contain letters only (one protein chain)");
  const fullLetters = [...sequence.toUpperCase()];
  // Receptor crop (Boltz's crop of a large receptor): keep only these 0-based residues of the full sequence.
  let keep = null;
  if (input.receptorIndices !== undefined) {
    keep = Array.from(input.receptorIndices);
    if (keep.length === 0) throw new Error("receptorIndices is empty");
    keep.forEach((index, k) => {
      if (!Number.isInteger(index) || index < 0 || index >= fullLetters.length) {
        throw new Error(`receptorIndices[${k}] = ${index} is outside the ${fullLetters.length}-residue sequence`);
      }
      if (k > 0 && index <= keep[k - 1]) throw new Error("receptorIndices must be strictly increasing (Boltz's crop keeps token order)");
    });
  }
  const letters = keep ? keep.map((index) => fullLetters[index]) : fullLetters;
  const n = letters.length;
  const notes = [];

  let ligand;
  if (input.ligandKind === "ccd") {
    if (input.ligandComponent === undefined) {
      throw new Error("ligandKind \"ccd\": pass the parsed component as ligandComponent (chain.score fetches it from `smiles` = the CCD code)");
    }
    ligand = ligandGraphFromCcd(input.ligandComponent);
    notes.push(`ligand: CCD component ${input.ligandComponent.code} (atom order and names from the dictionary; chirality and ref_pos are zero)`);
  } else {
    ligand = ligandGraph(smiles, input.ligandAtoms);
  }
  notes.push(...ligand.notes);
  const m = ligand.atoms.length;
  const T = n + m;

  // ---- tokens and atoms ----
  const names = letters.map(residueName);
  const proteinAtoms = names.map((name) => PROTEIN_CONFORMERS[name].atoms);
  const proteinAtomCount = proteinAtoms.reduce((s, a) => s + a.length, 0);
  const A = proteinAtomCount + m;
  const P = Math.max(1, Math.ceil(A / WINDOW)) * WINDOW;

  if (m > LIMITS.ligandTokens) throw new Error(`ligand has ${m} tokens, the limit is ${LIMITS.ligandTokens}`);
  if (!allowExtended && (n > LIMITS.receptorTokens || T > LIMITS.totalTokens || A > LIMITS.totalAtoms)) {
    throw new Error(`input has ${n} receptor + ${m} ligand tokens (${A} atoms); without --allow-extended GenFree requires `
      + `receptor <= ${LIMITS.receptorTokens}, total <= ${LIMITS.totalTokens} and atoms <= ${LIMITS.totalAtoms}, `
      + "and nothing is cropped silently");
  }

  const f32 = (count) => new Float32Array(count);
  const out = {};
  const put = (name, data, shape) => { out[name] = { data, shape }; };

  const resTypeId = new Int32Array(T);
  const tokenIndex = f32(T); const residueIndex = f32(T); const asymId = f32(T); const entityId = f32(T);
  const symId = f32(T); const molType = f32(T); const resType = f32(T * NUM_TOKENS);
  const padMask = f32(T).fill(1); const affinityMask = f32(T); const methodFeature = f32(T).fill(METHOD_OTHER);
  for (let t = 0; t < T; t += 1) {
    tokenIndex[t] = t;
    if (t < n) {
      resTypeId[t] = TOKEN_ID[names[t]];
      residueIndex[t] = keep ? keep[t] : t; // Boltz keeps each cropped residue's original res_idx
    } else {
      resTypeId[t] = TOKEN_UNK; // a ligand token is the protein UNK token
      asymId[t] = 1; entityId[t] = 1; molType[t] = MOL_NONPOLYMER; affinityMask[t] = 1;
    }
    if (t < n) molType[t] = MOL_PROTEIN;
    resType[t * NUM_TOKENS + resTypeId[t]] = 1;
  }

  // bonds: only the ligand's, symmetric; type_bonds = bond type + 1
  const tokenBonds = f32(T * T);
  const typeBonds = f32(T * T);
  for (const { a, b, type } of ligand.bonds) {
    for (const [i, j] of [[n + a, n + b], [n + b, n + a]]) {
      tokenBonds[i * T + j] = 1;
      typeBonds[i * T + j] = type + 1;
    }
  }
  const contact = f32(T * T * 5);
  for (let i = 0; i < T * T; i += 1) contact[i * 5] = 1; // UNSPECIFIED everywhere

  // ---- atoms ----
  const atomToToken = f32(P * T);
  const atomTokenIndex = new Int32Array(A);
  const refPos = f32(P * 3);
  const refElement = f32(P * NUM_ELEMENTS);
  const refElementId = new Int32Array(A);
  const refCharge = f32(P);
  const refChirality = f32(P);
  const nameChars = f32(P * 4 * 64);
  const nameCodes = new Int32Array(P * 4);
  const refSpaceUid = f32(P);
  const atomPadMask = f32(P);
  const backbone = f32(P * NUM_BACKBONE_CLASSES);

  const random = rng((input.seed ?? 0) >>> 0);
  const conformerIndex = new Int32Array(n);
  const augment = (input.refPosAugment ?? "seeded") === "seeded";
  let atom = 0;
  const setAtom = (token, uid, element, charge, chirality, name, backboneClass) => {
    atomToToken[atom * T + token] = 1;
    atomTokenIndex[atom] = token;
    refElement[atom * NUM_ELEMENTS + element] = 1;
    refElementId[atom] = element;
    refCharge[atom] = charge;
    refChirality[atom] = chirality;
    atomNameCodes(name).forEach((code, k) => { nameChars[(atom * 4 + k) * 64 + code] = 1; nameCodes[atom * 4 + k] = code; });
    refSpaceUid[atom] = uid;
    atomPadMask[atom] = 1;
    backbone[atom * NUM_BACKBONE_CLASSES + backboneClass] = 1;
  };
  for (let t = 0; t < n; t += 1) {
    const residue = names[t];
    const { conformers } = PROTEIN_CONFORMERS[residue];
    const first = atom;
    // Boltz: pick one of the stored conformers at random, centre it (mean over its atoms), rotate and translate it
    // at random. `conformerIndex` (input.conformers) pins the pick, e.g. to reproduce a recorded batch.
    const pick = input.conformers?.[t] ?? Math.min(conformers.length - 1, Math.floor(random() * conformers.length));
    conformerIndex[t] = pick;
    const flat = conformers[pick];
    const points = proteinAtoms[t].map((_, i) => [flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2]]);
    const centre = [0, 1, 2].map((k) => points.reduce((s, p) => s + p[k], 0) / points.length);
    const rotation = augment ? randomRotation(random) : [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const shift = augment ? [gaussian(random), gaussian(random), gaussian(random)] : [0, 0, 0];
    proteinAtoms[t].forEach((name, i) => {
      const element = ELEMENT_OF_LETTER[name[0]];
      setAtom(t, t, element, RESIDUE_CHARGES[`${residue}:${name}`] ?? 0, chiralityOf(residue, name), name,
        PROTEIN_BACKBONE[name] ?? 0);
      const c = points[i].map((v, k) => v - centre[k]);
      for (let a = 0; a < 3; a += 1) {
        refPos[atom * 3 + a] = rotation[a][0] * c[0] + rotation[a][1] * c[1] + rotation[a][2] * c[2] + shift[a];
      }
      atom += 1;
    });
    if (atom === first) throw new Error(`residue ${t} has no atoms`);
  }
  const ligandFirstAtom = atom;
  ligand.atoms.forEach((a, i) => {
    // one residue (reference space) for the whole ligand group; its ref_pos stays zero under `skip`
    setAtom(n + i, n, a.element, a.charge, 0, a.name, 0);
    atom += 1;
  });
  if (ligandConformer !== "skip") {
    // etkdg / rdkit_2d: our own embedder (no RDKit coordinates, no torch random stream: see docs/webgpu.md)
    const positions = ligandConformerPositions(
      { elements: ligand.atoms.map((a) => a.element), bonds: ligand.bonds },
      ligandConformer, (input.conformerSeed ?? input.seed ?? 0) >>> 0);
    for (let i = 0; i < m; i += 1) for (let a = 0; a < 3; a += 1) refPos[(ligandFirstAtom + i) * 3 + a] = positions[i * 3 + a];
    notes.push(`ligand ref_pos: ${ligandConformer} conformer from the built-in embedder (approximate; not RDKit's coordinates)`);
  }

  // ---- MSA ----
  const protein = Int32Array.from(resTypeId.subarray(0, n));
  const msa = buildMsa(protein, m, a3m, {
    maxRows: input.maxMsa ?? 8192, maxSeqs: input.maxMsaSeqs ?? 8192,
    keep, fullTypes: keep ? Int32Array.from(fullLetters, (letter) => TOKEN_ID[residueName(letter)]) : null,
  });
  notes.push(...msa.notes);
  const R = msa.rows;
  const msaMask = f32(R * T).fill(1);
  const hasDeletion = f32(R * T);
  const deletionValue = f32(R * T);
  const msaF = f32(R * T);
  const pairedF = f32(R * T);
  for (let i = 0; i < R * T; i += 1) {
    msaF[i] = msa.msa[i];
    pairedF[i] = msa.paired[i];
    if (msa.deletion[i] > 0) hasDeletion[i] = 1;
    deletionValue[i] = Math.fround(Math.fround(Math.PI / 2) * Math.fround(Math.atan(Math.fround(msa.deletion[i] / 3))));
  }

  put("token_index", tokenIndex, [1, T]); put("residue_index", residueIndex, [1, T]);
  put("asym_id", asymId, [1, T]); put("entity_id", entityId, [1, T]); put("sym_id", symId, [1, T]);
  put("mol_type", molType, [1, T]); put("res_type", resType, [1, T, NUM_TOKENS]);
  put("token_pad_mask", padMask, [1, T]); put("affinity_token_mask", affinityMask, [1, T]);
  put("token_bonds", tokenBonds, [1, T, T, 1]); put("type_bonds", typeBonds, [1, T, T]);
  put("profile", msa.profile, [1, T, NUM_TOKENS]); put("deletion_mean", msa.deletionMean, [1, T]);
  put("profile_affinity", msa.profileAffinity, [1, T, NUM_TOKENS]);
  put("deletion_mean_affinity", msa.deletionMeanAffinity, [1, T]);
  put("contact_conditioning", contact, [1, T, T, 5]); put("contact_threshold", f32(T * T), [1, T, T]);
  put("method_feature", methodFeature, [1, T]); put("modified", f32(T), [1, T]); put("cyclic_period", f32(T), [1, T]);
  put("atom_to_token", atomToToken, [1, P, T]); put("ref_pos", refPos, [1, P, 3]);
  put("ref_element", refElement, [1, P, NUM_ELEMENTS]); put("ref_charge", refCharge, [1, P]);
  put("ref_atom_name_chars", nameChars, [1, P, 4, 64]); put("ref_space_uid", refSpaceUid, [1, P]);
  put("ref_chirality", refChirality, [1, P]); put("atom_pad_mask", atomPadMask, [1, P]);
  put("atom_backbone_feat", backbone, [1, P, NUM_BACKBONE_CLASSES]);
  // Boltz's trunk does not read ref_mask (only the potentials do); it is the real-atom mask here
  put("ref_mask", Float32Array.from(atomPadMask), [1, P]);
  put("msa", msaF, [1, R, T]); put("msa_paired", pairedF, [1, R, T]); put("msa_mask", msaMask, [1, R, T]);
  put("has_deletion", hasDeletion, [1, R, T]); put("deletion_value", deletionValue, [1, R, T]);

  return {
    features: out,
    tokens: T, atoms: A, atomsPadded: P, receptorTokens: n, ligandTokens: m, msaRows: R, notes,
    conformerIndex,
    compact: {
      resType: resTypeId, atomToToken: atomTokenIndex, refElement: refElementId, atomNameCodes: nameCodes,
      msa: msa.msa, msaDeletion: msa.deletion,
    },
    windows: atomWindows(atomTokenIndex),
  };
}

/** `TensorBundle`-like accessor over a featuriseOwn result: `get(name)` / `shape(name)` with the oracle's `feat/` names. */
export function featureBundle(result) {
  const lookup = (name) => {
    const entry = result.features[name.replace(/^feat\//, "")];
    if (entry === undefined) throw new Error(`no feature ${name}`);
    return entry;
  };
  return { has: (name) => result.features[name.replace(/^feat\//, "")] !== undefined, get: (n) => lookup(n).data, shape: (n) => lookup(n).shape };
}

