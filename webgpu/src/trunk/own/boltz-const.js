// Boltz-2 constants the featuriser needs (src/boltz/data/const.py), for one protein chain plus a ligand.
// Nothing here comes from LocalFold; the conformer coordinates live in conformers.js.

/** const.tokens: "<pad>", "-", the 20 amino acids (alphabetical by three-letter code), UNK, RNA, DNA tokens. */
export const NUM_TOKENS = 33;
export const TOKEN_GAP = 1;
export const TOKEN_UNK = 22;
export const THREE_LETTER = [
  "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
  "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
];
export const TOKEN_ID = Object.freeze({
  "<pad>": 0, "-": 1,
  ...Object.fromEntries(THREE_LETTER.map((name, i) => [name, 2 + i])),
  UNK: 22,
});
export const TOKEN_MET = TOKEN_ID.MET;

/** const.prot_letter_to_token, as token ids. J B Z O U are UNK, as in Boltz. */
export const LETTER_TO_TOKEN = Object.freeze({
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", E: "GLU", Q: "GLN", G: "GLY", H: "HIS", I: "ILE",
  L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL",
  X: "UNK", J: "UNK", B: "UNK", Z: "UNK", O: "UNK", U: "UNK", "-": "-",
});
export const letterToTokenId = (letter) => {
  const name = LETTER_TO_TOKEN[letter];
  return name === undefined ? undefined : TOKEN_ID[name];
};

/** Molecule types (const.chain_type_ids): PROTEIN 0, NONPOLYMER 3. */
export const MOL_PROTEIN = 0;
export const MOL_NONPOLYMER = 3;
/** const.method_types_ids["other"]; GenFree runs with `override_method="other"`. */
export const METHOD_OTHER = 4;
/** const.bond_type_ids (RDKit-like): OTHER 0, SINGLE 1, DOUBLE 2, TRIPLE 3, AROMATIC 4. type_bonds is this + 1. */
export const BOND_TYPE = Object.freeze({ SINGLE: 1, DOUBLE: 2, TRIPLE: 3, AROMATIC: 4 });

export const NUM_ELEMENTS = 128;
export const NUM_BACKBONE_CLASSES = 17; // 1 + 4 protein + 12 nucleic
export const PROTEIN_BACKBONE = Object.freeze({ N: 1, CA: 2, C: 3, O: 4 });
/** const.chirality_type_ids: CHI_UNSPECIFIED 0, CHI_TETRAHEDRAL_CW 1, CHI_TETRAHEDRAL_CCW 2. */
export const CHI_CCW = 2;

/**
 * Formal charges that Boltz's residue molecules carry (heavy-atom molecules, RDKit GetFormalCharge).
 * Read off the GenFree oracle dumps (webgpu/oracle-dumps): every other atom of every residue is 0.
 */
export const RESIDUE_CHARGES = Object.freeze({ "LYS:NZ": 1, "ARG:NH2": 1, "HIS:ND1": 1 });

/**
 * Chiral tags (CHI_TETRAHEDRAL_CCW) of Boltz's residue molecules: CA of every residue except GLY, and CB of
 * THR and ILE (the two residues with a second stereocentre). Read off the oracle dumps. The trunk does not read
 * `ref_chirality`; it is produced for parity.
 */
export const RESIDUE_CHIRALITY = Object.freeze({ "THR:CB": CHI_CCW, "ILE:CB": CHI_CCW });
export const chiralityOf = (residue, atom) => {
  if (atom === "CA" && residue !== "GLY") return CHI_CCW;
  return RESIDUE_CHIRALITY[`${residue}:${atom}`] ?? 0;
};

/** Atomic number from an atom name's first letter (protein heavy atoms are C, N, O, S only). */
export const ELEMENT_OF_LETTER = Object.freeze({ C: 6, N: 7, O: 8, S: 16 });

/** Boltz's convert_atom_name: each character as ord - 32, padded with 0 to four. */
export function atomNameCodes(name) {
  if (name.length > 4) throw new Error(`atom name ${name} is longer than four characters`);
  return [0, 1, 2, 3].map((k) => (k < name.length ? name.charCodeAt(k) - 32 : 0));
}
