// Chemical Component Dictionary (CCD) ligands: the URL of a component's mmCIF file and a parser for it.
//
// Adapted from LocalFold's shared/af3/featurise/ccd-component.js (`ccdUrl`, the loop reader and the heavy-atom /
// ideal-or-model-coordinate rules of `parseCcdComponent`; Beer-Ware licence, attribution kept here). Differences:
// only what Boltz-2's `parse_ccd_residue` (src/boltz/data/parse/schema.py) reads is kept (no AF3 polymer or glycan
// fields), and bond orders are returned as the bond-type names of the SMILES path ("SINGLE" ...), see below.
//
// What Boltz does with a CCD ligand, and what is mirrored here:
//   * atoms = the heavy atoms of the CCD molecule in the dictionary's order, named by `_chem_comp_atom.atom_id`,
//     element = atomic number, charge = the formal charge (`_chem_comp_atom.charge`);
//   * bonds between heavy atoms; type = RDKit bond type of the CCD molecule (aromatic perceived by RDKit: here the
//     dictionary's `pdbx_aromatic_flag` marks it, value_order otherwise);
//   * `ref_pos` = the stored ideal conformer. GenFree's route uses `ligand_reference_conformer: skip` (ref_pos 0), so
//     the coordinates are parsed (ideal, else model coordinates centred on the heavy atoms) but not used by the featuriser.

export const ELEMENT_SYMBOLS = [
  "H", "HE", "LI", "BE", "B", "C", "N", "O", "F", "NE", "NA", "MG", "AL", "SI", "P", "S", "CL", "AR", "K", "CA",
  "SC", "TI", "V", "CR", "MN", "FE", "CO", "NI", "CU", "ZN", "GA", "GE", "AS", "SE", "BR", "KR", "RB", "SR", "Y", "ZR",
  "NB", "MO", "TC", "RU", "RH", "PD", "AG", "CD", "IN", "SN", "SB", "TE", "I", "XE", "CS", "BA", "LA", "CE", "PR", "ND",
  "PM", "SM", "EU", "GD", "TB", "DY", "HO", "ER", "TM", "YB", "LU", "HF", "TA", "W", "RE", "OS", "IR", "PT", "AU", "HG",
  "TL", "PB", "BI", "PO", "AT", "RN",
];

/** Where the PDB serves one component. */
export function ccdUrl(code) {
  const upper = String(code).trim().toUpperCase();
  if (!/^[A-Z0-9]{1,5}$/.test(upper)) throw new Error(`${code} is not a CCD code: expected 1-5 letters or digits`);
  return `https://files.rcsb.org/ligands/download/${upper}.cif`;
}

const unquote = (value) => value.replace(/^['"]|['"]$/g, "");

/** Column names and rows of one mmCIF category (looped, or the single-row `_category.item value` form). */
export function loopOf(text, prefix) {
  const columns = [];
  const rows = [];
  const single = new Map();
  let inHeader = false;
  let inBody = false;
  for (const line of text.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (trimmed.startsWith(`${prefix}.`)) {
      if (inBody) break;
      const [item, ...value] = trimmed.slice(prefix.length + 1).split(/\s+/);
      columns.push(item);
      if (value.length > 0) single.set(item, value.join(" "));
      inHeader = true;
      continue;
    }
    if (!inHeader) continue;
    // a comment is anything made only of hashes and spaces (files.rcsb.org ends loops with "# #")
    if (trimmed === "" || trimmed === "loop_" || /^[#\s]+$/.test(trimmed)) {
      if (inBody) break;
      continue;
    }
    if (trimmed.startsWith("_")) break;
    const fields = trimmed.match(/'[^']*'|"[^"]*"|\S+/g)?.map(unquote) ?? [];
    if (fields.length !== columns.length) {
      if (inBody) break;
      continue;
    }
    inBody = true;
    rows.push(fields);
  }
  if (rows.length === 0 && single.size > 0) rows.push(columns.map((item) => unquote(single.get(item) ?? "")));
  return { columns, rows };
}

const BOND_NAME = { SING: "SINGLE", DOUB: "DOUBLE", TRIP: "TRIPLE", QUAD: "QUADRUPLE" };

/**
 * Parse one component's mmCIF.
 * @returns {{code: string, atoms: {name: string, element: number, charge: number, x: number, y: number, z: number}[],
 *            bonds: {a: number, b: number, bondType: "SINGLE"|"DOUBLE"|"TRIPLE"|"AROMATIC"}[]}}
 *   heavy atoms only, in file order; `a` < `b` index the atom list
 */
export function parseCcdComponent(text) {
  const code = text.match(/_chem_comp\.id\s+(\S+)/)?.[1]?.replace(/['"]/g, "");
  if (code === undefined) throw new Error("this mmCIF has no _chem_comp.id");
  const atomLoop = loopOf(text, "_chem_comp_atom");
  const at = (row, name) => { const i = atomLoop.columns.indexOf(name); return i < 0 ? undefined : row[i]; };
  const isHydrogen = (row) => ["H", "D"].includes((at(row, "type_symbol") ?? "").toUpperCase());
  const heavy = atomLoop.rows.filter((row) => !isHydrogen(row));
  if (heavy.length === 0) throw new Error(`${code} has no heavy atoms`);
  const monatomic = heavy.length === 1;
  const ideal = heavy.every((row) => ["x", "y", "z"].every((k) => Number.isFinite(Number.parseFloat(at(row, `pdbx_model_Cartn_${k}_ideal`)))));
  const centre = { x: 0, y: 0, z: 0 };
  if (!ideal && !monatomic) {
    for (const k of ["x", "y", "z"]) {
      centre[k] = heavy.reduce((s, row) => s + Number.parseFloat(at(row, `model_Cartn_${k}`)), 0) / heavy.length;
    }
  }
  const coordinate = (row, k) => {
    const value = Number.parseFloat(at(row, ideal ? `pdbx_model_Cartn_${k}_ideal` : `model_Cartn_${k}`));
    if (Number.isFinite(value)) return ideal ? value : value - centre[k];
    if (monatomic) return 0; // a lone atom has no conformer; the origin is as good as any point
    throw new Error(`${code} has no usable ${k} coordinate`);
  };

  const byName = new Map();
  const atoms = heavy.map((row, i) => {
    const name = at(row, "atom_id") ?? "";
    byName.set(name, i);
    const symbol = (at(row, "type_symbol") ?? "").toUpperCase();
    const element = ELEMENT_SYMBOLS.indexOf(symbol) + 1;
    if (element === 0) throw new Error(`${code} atom ${name}: unknown element ${JSON.stringify(symbol)}`);
    return { name, element, charge: Number.parseInt(at(row, "charge") ?? "0", 10) || 0,
      x: coordinate(row, "x"), y: coordinate(row, "y"), z: coordinate(row, "z") };
  });

  const bondLoop = loopOf(text, "_chem_comp_bond");
  const bondAt = (row, name) => { const i = bondLoop.columns.indexOf(name); return i < 0 ? undefined : row[i]; };
  const bonds = [];
  for (const row of bondLoop.rows) {
    const i = byName.get(bondAt(row, "atom_id_1"));
    const j = byName.get(bondAt(row, "atom_id_2"));
    if (i === undefined || j === undefined) continue; // a bond to a hydrogen
    const aromatic = (bondAt(row, "pdbx_aromatic_flag") ?? "N").toUpperCase() === "Y";
    const order = (bondAt(row, "value_order") ?? "SING").toUpperCase();
    bonds.push({ a: Math.min(i, j), b: Math.max(i, j), bondType: aromatic ? "AROMATIC" : (BOND_NAME[order] ?? "SINGLE") });
  }
  return { code, atoms, bonds };
}

/**
 * Fetch and parse a component. `fetchImpl` defaults to the global fetch (Node >= 18 and browsers).
 * @param {string} code 1-5 letter/digit CCD code
 */
export async function fetchCcdComponent(code, fetchImpl = globalThis.fetch) {
  const response = await fetchImpl(ccdUrl(code));
  if (!response.ok) throw new Error(`CCD component ${code}: HTTP ${response.status}`);
  return parseCcdComponent(await response.text());
}
