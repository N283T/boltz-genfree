// Reference-guided receptor crop, the JS counterpart of src/boltz/reference_crop.py (GenFree's `target.crop`).
// Pure functions, no Node or DOM APIs: they read a PDB or mmCIF text, rank receptor residues by their distance to one reference
// ligand, form 10-residue blocks around them and keep the nearest blocks that fit Boltz's affinity budget.
//
// Where it follows gemmi (what the Python side uses) and where it is simpler:
//  - mmCIF: chains are auth_asym_id, residues auth_seq_id; the declared polymer sequence (_entity_poly_seq) and label_seq_id
//    map observed residues onto it, exactly as reference_crop.py does. PDB: no entity records are read, so the observed
//    residues are matched, as gemmi does for PDB files.
//  - Amino-acid recognition uses the 20 standard residues plus a table of common modified ones (MSE, SEP, ...); gemmi's table
//    is larger, so a reference with an exotic modified residue can differ.

/** Heavy-atom count per Boltz protein token (const.ref_atoms); UNK also covers every non-standard letter. */
const ATOMS = {
  ALA: 5, ARG: 11, ASN: 8, ASP: 8, CYS: 6, GLN: 9, GLU: 9, GLY: 4, HIS: 10, ILE: 8,
  LEU: 8, LYS: 9, MET: 8, PHE: 11, PRO: 7, SER: 6, THR: 7, TRP: 14, TYR: 12, VAL: 7, UNK: 5,
};
const LETTER_TO_TOKEN = {
  A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", E: "GLU", Q: "GLN", G: "GLY", H: "HIS", I: "ILE",
  L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL",
};
export const proteinAtomCounts = (sequence) => Array.from(sequence, (letter) => ATOMS[LETTER_TO_TOKEN[letter] ?? "UNK"]);

/** Three-letter residue name -> one-letter code: the standard residues and common modified ones. */
const ONE_LETTER = {
  ALA: "A", ARG: "R", ASN: "N", ASP: "D", CYS: "C", GLN: "Q", GLU: "E", GLY: "G", HIS: "H", ILE: "I",
  LEU: "L", LYS: "K", MET: "M", PHE: "F", PRO: "P", SER: "S", THR: "T", TRP: "W", TYR: "Y", VAL: "V",
  MSE: "M", SEP: "S", TPO: "T", PTR: "Y", CSO: "C", CME: "C", HYP: "P", KCX: "K", LLP: "K", MLY: "K", M3L: "K",
  SEC: "U", PYL: "O", ASX: "B", GLX: "Z", UNK: "X", HID: "H", HIE: "H", HIP: "H", CYX: "C", CSD: "C", OCS: "C",
  SCY: "C", CAS: "C", FME: "M", MHO: "M", NLE: "L", ORN: "A", DAL: "A", DAR: "R", DSN: "S", DLE: "L", DPR: "P",
};
const NUCLEIC = new Set(["A", "C", "G", "U", "I", "T", "N", "DA", "DC", "DG", "DT", "DI", "DU", "DN", "RA", "RC", "RG", "RU"]);
const WATER = new Set(["HOH", "WAT", "DOD", "H2O", "TIP", "SOL"]);
const isAminoAcid = (name) => ONE_LETTER[name] !== undefined;
const isWater = (name) => WATER.has(name);
const isNucleic = (name) => NUCLEIC.has(name);
const oneLetter = (name) => ONE_LETTER[name] ?? "X";

const isHeavy = (atom) => { const e = atom.element.toUpperCase(); return e !== "H" && e !== "D"; };

// ---------------------------------------------------------------------------------------------------------------- parsing

/** @typedef {{name: string, element: string, x: number, y: number, z: number, altloc: string}} Atom */
/** @typedef {{name: string, num: number, icode: string, labelSeq: number|null, entityId: string, entityType: string, atoms: Atom[]}} Residue */
/** @typedef {{chains: Map<string, Residue[]>, entities: Map<string, {type: string, fullSequence: string[]}>}} Structure */

const elementFromName = (name) => {
  const letters = name.replace(/[^A-Za-z]/g, "");
  return letters.length === 0 ? "" : letters[0].toUpperCase();
};

/** Append an atom to its chain, starting a residue when (number, insertion code, name, asym) changes. */
function addAtom(chains, last, chainName, residueKey, residueInit, atom) {
  let residues = chains.get(chainName);
  if (!residues) { residues = []; chains.set(chainName, residues); }
  if (last.get(chainName) !== residueKey) {
    residues.push({ ...residueInit, atoms: [] });
    last.set(chainName, residueKey);
  }
  residues[residues.length - 1].atoms.push(atom);
}

export function parsePdb(text) {
  const chains = new Map(), last = new Map();
  for (const line of text.split(/\r?\n/)) {
    if (line.startsWith("ENDMDL")) break;
    const record = line.slice(0, 6);
    if (record !== "ATOM  " && record !== "HETATM") continue;
    const name = line.slice(12, 16).trim();
    const resName = line.slice(17, 20).trim().toUpperCase();
    const chain = line.slice(21, 22).trim();
    const num = parseInt(line.slice(22, 26), 10);
    const icode = line.slice(26, 27).trim();
    const element = line.slice(76, 78).trim() || elementFromName(name);
    const atom = { name, element, x: parseFloat(line.slice(30, 38)), y: parseFloat(line.slice(38, 46)), z: parseFloat(line.slice(46, 54)), altloc: line.slice(16, 17).trim() };
    addAtom(chains, last, chain, `${num}|${icode}|${resName}`, { name: resName, num, icode, labelSeq: null, entityId: "", entityType: isWater(resName) ? "water" : "unknown" }, atom);
  }
  return { chains, entities: new Map() };
}

/** Tokens of one mmCIF line: whitespace separated, 'single' or "double" quoted values kept whole. */
function cifTokens(line) {
  const out = [];
  let i = 0;
  while (i < line.length) {
    const c = line[i];
    if (c === " " || c === "\t") { i += 1; continue; }
    if (c === "#") break;
    if (c === "'" || c === '"') {
      let j = i + 1;
      while (j < line.length && !(line[j] === c && (j + 1 === line.length || /\s/.test(line[j + 1])))) j += 1;
      out.push(line.slice(i + 1, j));
      i = j + 1;
    } else {
      let j = i;
      while (j < line.length && !/\s/.test(line[j])) j += 1;
      out.push(line.slice(i, j));
      i = j;
    }
  }
  return out;
}

/** The rows of the wanted mmCIF categories: Map(category -> [{item: value}]). */
export function parseCifCategories(text, wanted) {
  const lines = text.split(/\r?\n/);
  const result = new Map();
  const rowsOf = (category) => { if (!result.has(category)) result.set(category, []); return result.get(category); };
  const isControl = (line) => line.startsWith("_") || line.startsWith("loop_") || line.startsWith("data_");
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (line.startsWith("loop_")) {
      i += 1;
      const tags = [];
      while (i < lines.length && lines[i].trimStart().startsWith("_")) { tags.push(lines[i].trim().split(/\s+/)[0]); i += 1; }
      const category = tags[0]?.split(".")[0];
      if (!wanted.has(category)) { while (i < lines.length && !isControl(lines[i])) i += 1; continue; }
      const items = tags.map((t) => t.slice(category.length + 1));
      let values = [];
      while (i < lines.length && !isControl(lines[i])) {
        if (lines[i].startsWith(";")) {   // multi-line text field: one value
          let text1 = lines[i].slice(1);
          i += 1;
          while (i < lines.length && !lines[i].startsWith(";")) { text1 += `\n${lines[i]}`; i += 1; }
          values.push(text1);
          if (i < lines.length) values.push(...cifTokens(lines[i].slice(1)));   // values may follow the closing ';'
          i += 1;
          continue;
        }
        values.push(...cifTokens(lines[i]));
        i += 1;
      }
      const rows = rowsOf(category);
      for (let k = 0; k + items.length <= values.length; k += items.length) {
        const row = {};
        items.forEach((item, n) => { row[item] = values[k + n]; });
        rows.push(row);
      }
    } else if (line.startsWith("_")) {
      const tokens = cifTokens(line);
      const category = tokens[0].split(".")[0];
      if (wanted.has(category)) {
        let value = tokens[1];
        if (value === undefined && i + 1 < lines.length) { i += 1; value = cifTokens(lines[i])[0]; }
        const rows = rowsOf(category);
        if (rows.length === 0) rows.push({});
        rows[0][tokens[0].slice(category.length + 1)] = value;
      }
      i += 1;
    } else i += 1;
  }
  return result;
}

const cifValue = (v) => (v === undefined || v === "." || v === "?" ? "" : v);

export function parseMmcif(text) {
  const cats = parseCifCategories(text, new Set(["_atom_site", "_entity", "_entity_poly_seq"]));
  const entities = new Map();
  for (const row of cats.get("_entity") ?? []) entities.set(row.id, { type: cifValue(row.type).toLowerCase(), fullSequence: [] });
  const poly = new Map();
  for (const row of cats.get("_entity_poly_seq") ?? []) {
    const entry = poly.get(row.entity_id) ?? new Map();
    if (!entry.has(Number(row.num))) entry.set(Number(row.num), row.mon_id);   // first choice at a heterogeneous position
    poly.set(row.entity_id, entry);
  }
  for (const [id, nums] of poly) {
    const e = entities.get(id) ?? { type: "polymer", fullSequence: [] };
    e.fullSequence = [...nums.entries()].sort((a, b) => a[0] - b[0]).map(([, mon]) => mon.toUpperCase());
    entities.set(id, e);
  }
  const chains = new Map(), last = new Map();
  let model = null;
  for (const row of cats.get("_atom_site") ?? []) {
    const m = row.pdbx_PDB_model_num;
    if (model === null) model = m;
    if (m !== model) continue;
    const name = cifValue(row.label_atom_id) || cifValue(row.auth_atom_id);
    const resName = (cifValue(row.auth_comp_id) || cifValue(row.label_comp_id)).toUpperCase();
    const chain = cifValue(row.auth_asym_id) || cifValue(row.label_asym_id);
    const num = parseInt(cifValue(row.auth_seq_id) || cifValue(row.label_seq_id), 10);
    const icode = cifValue(row.pdbx_PDB_ins_code);
    const labelSeqText = cifValue(row.label_seq_id);
    const entityId = cifValue(row.label_entity_id);
    const entity = entities.get(entityId);
    const atom = {
      name, element: cifValue(row.type_symbol) || elementFromName(name),
      x: parseFloat(row.Cartn_x), y: parseFloat(row.Cartn_y), z: parseFloat(row.Cartn_z), altloc: cifValue(row.label_alt_id),
    };
    addAtom(chains, last, chain, `${num}|${icode}|${resName}|${cifValue(row.label_asym_id)}`, {
      name: resName, num, icode, labelSeq: labelSeqText === "" ? null : parseInt(labelSeqText, 10), entityId,
      entityType: entity?.type ?? (isWater(resName) ? "water" : "unknown"),
    }, atom);
  }
  return { chains, entities };
}

/** PDB or mmCIF text -> Structure (first model). */
export function parseStructure(text) {
  return /^\s*data_/m.test(text) || /^_atom_site\./m.test(text) ? parseMmcif(text) : parsePdb(text);
}

// ----------------------------------------------------------------------------------------------------------- reference crop

const residueSequence = (residues) => residues.map((r) => oneLetter(r.name)).join("");

/** A chain's protein residues with the sequence they declare and the position of each in it (reference_crop.py `_protein_candidate`). */
function proteinCandidate(structure, name, allResidues) {
  const residues = allResidues.filter((r) => isAminoAcid(r.name));
  if (residues.length === 0) return null;
  const observed = residueSequence(residues);
  const plain = { name, residues, sequence: observed, observedSequence: observed, sequenceIndices: residues.map((_, i) => i) };
  const entityIds = new Set(residues.map((r) => r.entityId).filter((id) => id));
  if (entityIds.size !== 1) return plain;
  const entity = structure.entities.get([...entityIds][0]);
  if (!entity || entity.fullSequence.length === 0) return plain;
  const sequence = entity.fullSequence.map(oneLetter).join("");
  if (residues.some((r) => r.labelSeq === null)) {
    throw new Error(`Reference protein chain '${name}' declares a full polymer sequence but lacks label_seq_id mapping for observed residues.`);
  }
  const sequenceIndices = residues.map((r) => r.labelSeq - 1);
  if (new Set(sequenceIndices).size !== sequenceIndices.length || sequenceIndices.some((k) => k < 0 || k >= sequence.length)) {
    throw new Error(`Reference protein chain '${name}' has invalid label_seq_id mapping to its declared polymer sequence.`);
  }
  if (sequenceIndices.map((k) => sequence[k]).join("") !== observed) {
    throw new Error(`Reference protein chain '${name}' observed residues do not match its declared polymer sequence at label_seq_id positions.`);
  }
  return { name, residues, sequence, observedSequence: observed, sequenceIndices };
}

function chooseProteinChain(structure, sequence, requested) {
  let candidates = [];
  for (const [name, residues] of structure.chains) {
    const c = proteinCandidate(structure, name, residues);
    if (c) candidates.push(c);
  }
  if (requested !== undefined && requested !== null) {
    candidates = candidates.filter((c) => c.name === requested);
    if (candidates.length === 0) throw new Error(`Reference protein chain '${requested}' was not found.`);
  }
  let exact = candidates.filter((c) => c.sequence === sequence);
  if (exact.length === 0) {
    exact = candidates.filter((c) => c.observedSequence === sequence).map((c) => ({ ...c, sequenceIndices: c.residues.map((_, i) => i) }));
  }
  if (exact.length !== 1) {
    const available = candidates.map((c) => `${c.name}(${c.sequence.length} declared, ${c.residues.length} observed)`).join(", ") || "none";
    const qualifier = requested ? `chain '${requested}'` : "exactly one chain";
    throw new Error(`Reference must contain ${qualifier} whose declared polymer or observed residue sequence exactly matches the receptor sequence; candidates: ${available}.`);
  }
  return exact[0];
}

const heavyCoords = (residue) => residue.atoms.filter(isHeavy).map((a) => [a.x, a.y, a.z]);

/** Non-polymer residues of the first model that can serve as the reference ligand. */
export function ligandCandidates(structure) {
  const out = [];
  for (const [chain, residues] of structure.chains) {
    for (const residue of residues) {
      if (!isWater(residue.name) && !isAminoAcid(residue.name) && !isNucleic(residue.name) && heavyCoords(residue).length > 0) {
        out.push({ chain, residue });
      }
    }
  }
  return out;
}

function chooseLigandResidue(structure, selector) {
  if ((selector.residue === undefined || selector.residue === null) !== (selector.ccd === undefined || selector.ccd === null)) {
    throw new Error("Reference ligand residue and CCD must be specified together.");
  }
  const inChain = ligandCandidates(structure).filter((c) => c.chain === selector.chain).map((c) => c.residue);
  const candidates = selector.residue === undefined || selector.residue === null
    ? inChain
    : inChain.filter((r) => r.num === selector.residue && r.name === String(selector.ccd).toUpperCase());
  if (candidates.length === 0) {
    throw new Error(selector.residue === undefined || selector.residue === null
      ? `Reference ligand chain matched no non-polymer residue: chain='${selector.chain}'.`
      : `Reference ligand selector matched no residue: chain='${selector.chain}', residue=${selector.residue}, ccd='${selector.ccd}'.`);
  }
  if (candidates.length !== 1) {
    throw new Error(selector.residue === undefined || selector.residue === null
      ? "Reference ligand chain matched multiple residues; specify the ligand with chain, residue, and CCD."
      : "Reference ligand selector matched multiple residues; preprocess the reference so chain, residue, and CCD identify exactly one residue.");
  }
  const picked = candidates[0];
  if (picked.icode) throw new Error("Reference ligand uses an insertion code, which is unsupported. Preprocess the reference to a unique residue without an insertion code.");
  if (picked.entityType === "branched") throw new Error("Multi-residue or branched reference ligands are unsupported; provide a preprocessed single-residue ligand.");
  if (picked.atoms.some((a) => a.altloc !== "")) throw new Error("Reference ligand contains alternate locations, which are unsupported. Preprocess the reference to one ligand conformer.");
  return picked;
}

function residueCenter(residue) {
  const ca = residue.atoms.find((a) => a.name === "CA");
  if (ca) return [ca.x, ca.y, ca.z];
  const coords = heavyCoords(residue);
  if (coords.length === 0) return null;
  const mean = [0, 0, 0];
  for (const c of coords) { mean[0] += c[0]; mean[1] += c[1]; mean[2] += c[2]; }
  return mean.map((v) => v / coords.length);
}

const NEIGHBORHOOD = 10;
function blockAround(seed, length) {
  if (length <= NEIGHBORHOOD) return Array.from({ length }, (_, i) => i);
  const start = Math.min(Math.max(0, seed - Math.floor(NEIGHBORHOOD / 2)), length - NEIGHBORHOOD);
  return Array.from({ length: NEIGHBORHOOD }, (_, i) => start + i);
}

/**
 * Rank receptor residues by distance to a reference ligand and form their blocks (reference_crop.py `load_reference_crop`).
 * @param {Structure} structure from `parseStructure`
 * @param {string} sequence the receptor sequence (must match one reference chain exactly)
 * @param {{proteinChain?: string, ligand: {chain: string, residue?: number, ccd?: string}}} options
 * @returns {{proteinChain: string, ligandChain: string, ligandResidue: number, ligandCcd: string, seedOrder: number[], blocks: number[][], distances: number[]}}
 */
export function loadReferenceCrop(structure, sequence, { proteinChain, ligand }) {
  if (structure.chains.size === 0) throw new Error("Crop reference contains no coordinate model.");
  const protein = chooseProteinChain(structure, sequence, proteinChain);
  const lig = chooseLigandResidue(structure, ligand);
  const ligandCoords = heavyCoords(lig);
  const ranked = [];
  protein.residues.forEach((residue, k) => {
    const center = residueCenter(residue);
    if (center === null) return;
    let best = Infinity;
    for (const c of ligandCoords) best = Math.min(best, Math.hypot(c[0] - center[0], c[1] - center[1], c[2] - center[2]));
    ranked.push({ distance: best, index: protein.sequenceIndices[k] });
  });
  if (ranked.length === 0) throw new Error("Reference protein contains no resolved heavy atoms.");
  ranked.sort((a, b) => a.distance - b.distance || a.index - b.index);
  const seedOrder = ranked.map((r) => r.index);
  return {
    proteinChain: protein.name, ligandChain: ligand.chain, ligandResidue: lig.num, ligandCcd: lig.name,
    seedOrder, blocks: seedOrder.map((seed) => blockAround(seed, sequence.length)), distances: ranked.map((r) => r.distance),
  };
}

/**
 * The nearest blocks that fit the affinity budget, never truncating the ligand (reference_crop.py `select_receptor_tokens`).
 * @returns {number[]} sorted 0-based receptor residue indices
 */
export function selectReceptorTokens(crop, sequence, { ligandTokens, maxTokens = 256, maxAtoms = 2048, maxProteinTokens = 200 }) {
  const counts = proteinAtomCounts(sequence);
  const selected = new Set();
  let selectedAtoms = 0;
  for (const block of crop.blocks) {
    const added = block.filter((i) => !selected.has(i));
    const addedAtoms = added.reduce((sum, i) => sum + counts[i], 0);
    if (selected.size + added.length + ligandTokens > maxTokens
      || selectedAtoms + addedAtoms + ligandTokens > maxAtoms
      || selected.size + added.length > maxProteinTokens) break;
    for (const i of added) selected.add(i);
    selectedAtoms += addedAtoms;
  }
  if (selected.size === 0) throw new Error(`No reference-derived receptor block fits after reserving the complete ${ligandTokens}-token ligand.`);
  return [...selected].sort((a, b) => a - b);
}

/** What a UI needs to offer choices: protein chains and ligand residues found in a reference. */
export function describeReference(structure) {
  const chains = [];
  for (const [name, residues] of structure.chains) {
    const c = proteinCandidate(structure, name, residues);
    if (c) chains.push({ name, declared: c.sequence.length, observed: c.residues.length, sequence: c.sequence, observedSequence: c.observedSequence });
  }
  const ligands = ligandCandidates(structure).map(({ chain, residue }) => ({
    chain, residue: residue.num, ccd: residue.name, atoms: heavyCoords(residue).length,
  }));
  return { chains, ligands };
}
