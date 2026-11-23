// The ligand's atom graph from a SMILES string, with no RDKit and no LocalFold: atoms, formal charges, bonds and
// names come from ../canonical.js (a port of the RDKit pieces GenFree's ligand featuriser depends on).
//
// Token order = heavy-atom order of RDKit's canonical SMILES; names = <ELEMENT><rank> from the AddHs molecule.
// `ligandAtoms` ({smiles, atomNames}, as written by tools/ligand_atoms.py) overrides both, which is how stereo
// ligands get GenFree's exact order and names.

import { _internals, ligandAtomsFromSmiles } from "../canonical.js";
import { BOND_TYPE } from "./boltz-const.js";

// canonical.js bond.type values: 1 2 3 are orders, 12 is RDKit's AROMATIC enum
const BOND_TYPE_OF = new Map([[1, BOND_TYPE.SINGLE], [2, BOND_TYPE.DOUBLE], [3, BOND_TYPE.TRIPLE], [12, BOND_TYPE.AROMATIC]]);

/**
 * @param {string} smiles
 * @param {{smiles: string, atomNames: string[]}} [ligandAtoms]
 * @returns {{atoms: {element: number, charge: number, name: string}[], bonds: {a: number, b: number, type: number}[],
 *            smiles: string, notes: string[]}}
 */
export function ligandGraph(smiles, ligandAtoms) {
  const notes = [];
  let given = ligandAtoms;
  if (given === undefined) {
    const derived = ligandAtomsFromSmiles(smiles);
    given = { smiles: derived.smiles, atomNames: derived.atomNames };
    if (!derived.exact) {
      notes.push(`ligand atom order/names are approximate (${derived.warnings.join("; ") || "outside the RDKit-free subset"}); `
        + "stereo ligands need tools/ligand_atoms.py (--ligand-atoms) to match GenFree exactly");
    }
  }
  // the canonical SMILES lists its atoms in written order, so re-reading it keeps the token order
  const mol = _internals.buildMolecule(_internals.parseSmiles(given.smiles), []);
  if (given.atomNames.length !== mol.n) {
    throw new Error(`ligand atom names: ${given.atomNames.length} given, ${mol.n} ligand atoms`);
  }
  const atoms = Array.from({ length: mol.n }, (_, i) => ({ element: mol.z[i], charge: mol.charge[i], name: given.atomNames[i] }));
  const bonds = mol.bonds.map((bond) => {
    const type = BOND_TYPE_OF.get(bond.type);
    if (type === undefined) throw new Error(`unsupported bond type ${bond.type}`);
    return { a: bond.a, b: bond.b, type };
  });
  return { atoms, bonds, smiles: given.smiles, notes };
}

/**
 * The ligand graph of a parsed CCD component (ccd.js `parseCcdComponent`), in the shape `ligandGraph` returns: tokens in
 * the dictionary's heavy-atom order, names from `atom_id`. A quadruple bond is Boltz's OTHER type (0), which
 * `type_bonds` encodes as 1 like every other type + 1.
 */
export function ligandGraphFromCcd(component) {
  const atoms = component.atoms.map(({ element, charge, name }) => ({ element, charge, name }));
  const bonds = component.bonds.map(({ a, b, bondType }) => ({ a, b, type: BOND_TYPE[bondType] ?? 0 }));
  return { atoms, bonds, smiles: null, notes: [] };
}
