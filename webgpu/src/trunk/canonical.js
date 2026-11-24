// Pure-JS port of the parts of RDKit that GenFree's ligand featuriser depends on:
//
//   standardised SMILES -> RDKit canonical SMILES (heavy-atom order = ligand token order)
//   AddHs molecule      -> CanonicalRankAtoms        (heavy-atom names <ELEMENT><n>)
//
// See webgpu/tools/ligand_atoms.py for the PyTorch-side reference this mirrors. Everything
// here is self-contained (no LocalFold, no dependencies) and runs in Node and browsers.
//
// What is reproduced (validated against RDKit 2026.03 on ~45k molecules, see
// webgpu/tools/check_canonical.mjs):
//   * a SMILES parser (organic subset, bracket atoms, charges, isotopes, ring closures
//     including %nn, explicit H atoms, bond symbols), RDKit valence / implicit-H rules, and
//     Kekulisation of aromatic input,
//   * RDKit's default aromaticity model (symmetrised SSSR, Hueckel counting with RDKit's
//     electron-donor rules, pairs of fused rings),
//   * the parts of chembl_structure_pipeline.standardize_mol that act on neutral organics
//     (nitro / sulfoxide to charge-separated form, imidic acid to amide, a simplified
//     Uncharger, largest fragment) and its exclusion rule (metals, > 7 boron atoms throw),
//   * RDKit's canonical atom ranking (new_canon.cpp: partition refinement with the
//     AtomCompareFunctor, followed by tie breaking), on the H-less graph for SMILES writing
//     and on the AddHs graph for the atom names,
//   * RDKit's canonical SMILES writer (two-pass DFS: ring closures first, non-ring
//     neighbours before ring neighbours, higher bond order first, ring-closure digit reuse).
//
// What is NOT reproduced (`exact: false` plus a warning; the result is then a best effort that
// ignores the feature):
//   * stereo (@, @@, /, \, @TB..., ...): RDKit's chirality-aware ranking changes the atom
//     order and names for most stereo molecules and is not understood well enough to port,
//   * multiple fragments (the largest is kept), remaining formal charges that are not
//     charge-separated +/- pairs, atom maps, elements outside H B C N O F Si P S Cl Se Br I,
//   * [H] atoms the standardiser would keep, and anything else it would change.
//
// Unsupported input (dative bonds, dummy atoms, impossible valences, un-kekulisable aromatics,
// excluded metals) throws an Error, like the RDKit pipeline does.

// ---------------------------------------------------------------------------------------
// Periodic table pieces
// ---------------------------------------------------------------------------------------

const SYMBOLS = ("* H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
  + "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm "
  + "Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split(" ");
const Z_OF = new Map(SYMBOLS.map((s, z) => [s, z]));

// RDKit's valence lists (atomic_data.cpp); absent = unconstrained.
const VALENCES = new Map([
  [1, [1]], [3, [1]], [5, [3]], [6, [4]], [7, [3]], [8, [2]], [9, [1]], [11, [1]], [14, [4]],
  [15, [3, 5, 7]], [16, [2, 4, 6]], [17, [1]], [19, [1]], [33, [3, 5]], [34, [2, 4, 6]],
  [35, [1]], [52, [2, 4, 6]], [53, [1, 3, 5]],
]);
const OUTER_ELECTRONS = new Map([
  [1, 1], [3, 1], [5, 3], [6, 4], [7, 5], [8, 6], [9, 7], [11, 1], [14, 4], [15, 5], [16, 6],
  [17, 7], [19, 1], [33, 5], [34, 6], [35, 7], [52, 6], [53, 7],
]);
// Pauling electronegativity, only to decide whether an exocyclic double bond "steals" the
// pi electron of a ring atom.
const ELECTRONEGATIVITY = new Map([
  [1, 2.2], [5, 2.04], [6, 2.55], [7, 3.04], [8, 3.44], [9, 3.98], [14, 1.9], [15, 2.19],
  [16, 2.58], [17, 3.16], [33, 2.18], [34, 2.55], [35, 2.96], [52, 2.1], [53, 2.66],
]);
const ORGANIC_SUBSET = new Set([5, 6, 7, 8, 9, 15, 16, 17, 35, 53]);
const SUPPORTED_ELEMENTS = new Set([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53]);

const EXCLUDED_METALS = new Set(("Sc Ti V Cr Mn Fe Co Ni Cu Ga Y Zr Nb Mo Tc Ru Rh Pd Cd In Sn La Hf Ta W Re Os "
  + "Ir Pt Au Hg Tl Pb Bi Po Ac Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Th Pa U Np Pu Am Cm Bk Cf Es Fm "
  + "Md No Lr Ge Sb").split(" "));

const SINGLE = 1;
const DOUBLE = 2;
const TRIPLE = 3;
const AROMATIC = 12; // RDKit's Bond::AROMATIC enum value, used as a ranking key

/** Valence list for an atom, using the isoelectronic neutral atom for charged ones. */
function valenceList(z, charge) {
  const eff = charge ? z - charge : z;
  return VALENCES.get(eff) ?? null;
}

// ---------------------------------------------------------------------------------------
// SMILES parser
// ---------------------------------------------------------------------------------------

const AROMATIC_ORGANIC = new Set(["b", "c", "n", "o", "p", "s"]);
const AROMATIC_BRACKET = new Set(["b", "c", "n", "o", "p", "s", "se", "as", "te", "si"]);
const BOND_SYMBOLS = new Set(["-", "=", "#", ":", "$", "/", "\\"]);

class SmilesError extends Error {}

function parseSmiles(text) {
  const atoms = [];
  const bonds = [];
  const flags = { stereo: false, isotope: false, atomMap: false, dummy: false };
  let prev = -1;
  let pending = null;
  const branchStack = [];
  const open = new Map();

  const addBond = (a, b, symbol) => {
    bonds.push({ a, b, symbol });
  };
  const addAtom = (atom) => {
    const index = atoms.length;
    atoms.push(atom);
    if (prev >= 0) {
      addBond(prev, index, pending);
    }
    pending = null;
    prev = index;
  };

  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    if (ch === "(") {
      if (prev < 0) throw new SmilesError("branch without an atom");
      branchStack.push(prev);
      i++;
    } else if (ch === ")") {
      if (!branchStack.length) throw new SmilesError("unbalanced ')'");
      prev = branchStack.pop();
      pending = null;
      i++;
    } else if (ch === ".") {
      prev = -1;
      pending = null;
      i++;
    } else if ((ch === "-" && (text[i + 1] === ">")) || (ch === "<" && text[i + 1] === "-")) {
      throw new SmilesError("dative bonds are not supported");
    } else if (BOND_SYMBOLS.has(ch)) {
      if (ch === "/" || ch === "\\") flags.stereo = true;
      if (ch === "$") throw new SmilesError("quadruple bonds are not supported");
      pending = ch;
      i++;
    } else if ((ch >= "0" && ch <= "9") || ch === "%") {
      let label;
      if (ch === "%") {
        if (text[i + 1] === "(") {
          const end = text.indexOf(")", i);
          label = Number(text.slice(i + 2, end));
          i = end + 1;
        } else {
          label = Number(text.slice(i + 1, i + 3));
          i += 3;
        }
      } else {
        label = Number(ch);
        i++;
      }
      if (prev < 0) throw new SmilesError("ring closure without an atom");
      const entry = open.get(label);
      if (entry) {
        if (entry.atom === prev) throw new SmilesError("ring closure to self");
        const symbol = pending ?? entry.symbol;
        if (pending && entry.symbol && pending !== entry.symbol
          && !(("/\\".includes(pending)) && ("/\\".includes(entry.symbol)))) {
          throw new SmilesError("conflicting ring closure bond symbols");
        }
        addBond(entry.atom, prev, symbol);
        open.delete(label);
      } else {
        open.set(label, { atom: prev, symbol: pending });
      }
      pending = null;
    } else if (ch === "[") {
      const end = text.indexOf("]", i);
      if (end < 0) throw new SmilesError("unterminated bracket atom");
      addAtom(parseBracket(text.slice(i + 1, end), flags));
      i = end + 1;
    } else if (ch === "*") {
      throw new SmilesError("dummy atoms are not supported");
    } else if (ch === ">" || ch === "<") {
      throw new SmilesError("dative bonds are not supported");
    } else {
      let symbol = ch;
      if ((ch === "C" && text[i + 1] === "l") || (ch === "B" && text[i + 1] === "r")) {
        symbol = ch + text[i + 1];
      }
      i += symbol.length;
      const aromatic = AROMATIC_ORGANIC.has(symbol);
      const z = Z_OF.get(aromatic ? symbol.toUpperCase() : symbol);
      if (z === undefined || (!aromatic && !ORGANIC_SUBSET.has(z)) || symbol === "H") {
        throw new SmilesError(`unexpected character '${ch}' in SMILES`);
      }
      addAtom({ z, aromatic, charge: 0, isotope: 0, hExplicit: 0, bracket: false, map: 0 });
    }
  }
  if (branchStack.length) throw new SmilesError("unbalanced '('");
  if (open.size) throw new SmilesError("unclosed ring bond");
  return { atoms, bonds, flags };
}

function parseBracket(body, flags) {
  let i = 0;
  let isotope = 0;
  while (i < body.length && body[i] >= "0" && body[i] <= "9") {
    isotope = isotope * 10 + Number(body[i]);
    i++;
  }
  if (isotope) flags.isotope = true;
  // Element symbol: two letters when that is a real element, else one.
  let symbol = body[i];
  if (!symbol || !/[A-Za-z]/.test(symbol)) throw new SmilesError(`bad bracket atom [${body}]`);
  let aromatic = false;
  if (symbol >= "a" && symbol <= "z") {
    aromatic = true;
    const two = body.slice(i, i + 2);
    if (AROMATIC_BRACKET.has(two) && two.length === 2) symbol = two;
    if (!AROMATIC_BRACKET.has(symbol)) throw new SmilesError(`bad aromatic atom [${body}]`);
  } else {
    const two = body.slice(i, i + 2);
    if (two.length === 2 && /[a-z]/.test(two[1]) && Z_OF.has(two)) symbol = two;
  }
  i += symbol.length;
  const z = Z_OF.get(aromatic ? symbol[0].toUpperCase() + symbol.slice(1) : symbol);
  if (z === undefined) throw new SmilesError(`unknown element in [${body}]`);
  while (body[i] === "@") {
    flags.stereo = true;
    i++;
  }
  // Non-tetrahedral stereo classes (@SP1, @TB5, @OH3, ...): recorded as stereo, then skipped.
  const cls = /^(TH|AL|SP|TB|OH)\d+/.exec(body.slice(i));
  if (cls) i += cls[0].length;
  let hExplicit = 0;
  if (body[i] === "H") {
    i++;
    let digits = "";
    while (i < body.length && body[i] >= "0" && body[i] <= "9") digits += body[i++];
    hExplicit = digits ? Number(digits) : 1;
  }
  let charge = 0;
  if (body[i] === "+" || body[i] === "-") {
    const sign = body[i] === "+" ? 1 : -1;
    let count = 0;
    while (body[i] === (sign > 0 ? "+" : "-")) {
      count++;
      i++;
    }
    let digits = "";
    while (i < body.length && body[i] >= "0" && body[i] <= "9") digits += body[i++];
    charge = sign * (digits ? Number(digits) : count);
  }
  let map = 0;
  if (body[i] === ":") {
    map = Number(body.slice(i + 1)) || 0;
    flags.atomMap = true;
    i = body.length;
  }
  if (i !== body.length) throw new SmilesError(`cannot parse bracket atom [${body}]`);
  return { z, aromatic, charge, isotope, hExplicit, bracket: true, map };
}

// ---------------------------------------------------------------------------------------
// Molecule construction: explicit H removal, Kekulisation, implicit Hs, valence check
// ---------------------------------------------------------------------------------------

/** Build an adjacency structure: adj[i] = [{to, bond}] in bond-creation order. */
function buildAdjacency(n, bonds) {
  const adj = Array.from({ length: n }, () => []);
  bonds.forEach((bond, index) => {
    adj[bond.a].push({ to: bond.b, bond: index });
    adj[bond.b].push({ to: bond.a, bond: index });
  });
  return adj;
}

/** Bridges / ring-bond flags and connected components. */
function ringBonds(n, bonds, adj) {
  const isRing = new Uint8Array(bonds.length);
  const disc = new Int32Array(n).fill(-1);
  const low = new Int32Array(n);
  let time = 0;
  const visit = (u, viaBond) => {
    disc[u] = low[u] = time++;
    for (const { to, bond } of adj[u]) {
      if (bond === viaBond) continue;
      if (disc[to] < 0) {
        visit(to, bond);
        low[u] = Math.min(low[u], low[to]);
        if (low[to] <= disc[u]) isRing[bond] = 1;
      } else {
        low[u] = Math.min(low[u], disc[to]);
        isRing[bond] = 1;
      }
    }
  };
  // Mark tree edges that lie on a cycle: an edge u-to is a ring edge iff low[to] <= disc[u].
  for (let u = 0; u < n; u++) if (disc[u] < 0) visit(u, -1);
  return isRing;
}

function components(n, adj) {
  const comp = new Int32Array(n).fill(-1);
  let count = 0;
  for (let s = 0; s < n; s++) {
    if (comp[s] >= 0) continue;
    const stack = [s];
    comp[s] = count;
    while (stack.length) {
      const u = stack.pop();
      for (const { to } of adj[u]) {
        if (comp[to] < 0) {
          comp[to] = count;
          stack.push(to);
        }
      }
    }
    count++;
  }
  return { comp, count };
}

/** Default bond order of a parsed bond (before Kekulisation); 'ar' is aromatic. */
function initialOrder(bond, atoms) {
  switch (bond.symbol) {
    case "=": return 2;
    case "#": return 3;
    case "-": case "/": case "\\": return 1;
    case ":": return "ar";
    default:
      return atoms[bond.a].aromatic && atoms[bond.b].aromatic ? "ar" : 1;
  }
}

/**
 * Turn the parse tree into a sanitised molecule: explicit hydrogens folded into counts,
 * aromatic input Kekulised, implicit hydrogens assigned, valences checked, RDKit's
 * aromaticity re-perceived.
 */
function buildMolecule(parsed, warnings, approximate = () => {}) {
  let { atoms, bonds } = parsed;

  // Fold plain [H] atoms into their heavy neighbour's hydrogen count.
  const removed = new Set();
  const hFromAtoms = new Array(atoms.length).fill(0);
  {
    const adj0 = buildAdjacency(atoms.length, bonds);
    for (let i = 0; i < atoms.length; i++) {
      const a = atoms[i];
      if (a.z === 1 && a.charge === 0 && a.isotope === 0 && adj0[i].length === 1
        && atoms[adj0[i][0].to].z !== 1 && a.hExplicit === 0) {
        removed.add(i);
        hFromAtoms[adj0[i][0].to]++;
      }
    }
  }
  if (removed.size) {
    const remap = new Map();
    const keep = [];
    atoms.forEach((atom, i) => {
      if (!removed.has(i)) {
        remap.set(i, keep.length);
        keep.push({ ...atom, hExplicit: atom.hExplicit + hFromAtoms[i], fromH: hFromAtoms[i] });
      }
    });
    bonds = bonds.filter((b) => !removed.has(b.a) && !removed.has(b.b))
      .map((b) => ({ ...b, a: remap.get(b.a), b: remap.get(b.b) }));
    atoms = keep;
  } else {
    atoms = atoms.map((a) => ({ ...a }));
  }
  const n = atoms.length;
  if (n === 0) throw new SmilesError("empty molecule");
  const adj = buildAdjacency(n, bonds);
  const orders = bonds.map((bond) => initialOrder(bond, atoms));
  const ring = ringBonds(n, bonds, adj);
  // chembl_structure_pipeline.remove_hs_from_mol keeps [H] atoms on atoms with three ring bonds
  // or in a hypervalent state; those would stay explicit atoms there, which is not modelled.
  for (let i = 0; i < n; i++) {
    if (!atoms[i].fromH) continue;
    const ringCount = adj[i].filter(({ bond }) => ring[bond]).length;
    if (ringCount >= 3) {
      approximate("explicit [H] on an atom with three ring bonds: kept by the standardiser");
      break;
    }
  }

  // Aromatic atoms must lie on a ring.
  for (let i = 0; i < n; i++) {
    if (atoms[i].aromatic && !adj[i].some(({ bond }) => ring[bond])) {
      throw new SmilesError("non-ring atom marked aromatic");
    }
  }
  for (let b = 0; b < bonds.length; b++) {
    if (orders[b] === "ar" && !ring[b]) throw new SmilesError("aromatic bond outside a ring");
  }

  kekulize(atoms, bonds, adj, orders);
  standardiseRules(atoms, bonds, adj, orders, ring, warnings);

  // Hydrogen counts and valence check.
  const nH = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const atom = atoms[i];
    let sum = atom.hExplicit;
    for (const { bond } of adj[i]) sum += orders[bond];
    const list = valenceList(atom.z, atom.charge);
    nH[i] = hydrogenCount(i, atoms, adj, orders);
    if (atom.bracket && list && sum > list[list.length - 1]) {
      throw new SmilesError(`explicit valence ${sum} of ${SYMBOLS[atom.z]} too large`);
    }
  }

  const mol = {
    n,
    z: Int32Array.from(atoms, (a) => a.z),
    charge: Int32Array.from(atoms, (a) => a.charge),
    isotope: Int32Array.from(atoms, (a) => a.isotope),
    nH,
    aromatic: new Uint8Array(n),
    bonds: bonds.map((b, k) => ({ a: b.a, b: b.b, order: orders[k], type: orders[k] })),
    adj,
    isRing: ring,
  };
  perceiveAromaticity(mol);
  return mol;
}

/** Total hydrogens of an atom under RDKit's rules, given the current bond orders. */
function hydrogenCount(i, atoms, adj, orders) {
  const atom = atoms[i];
  if (atom.bracket) return atom.hExplicit;
  let sum = atom.hExplicit;
  for (const { bond } of adj[i]) sum += orders[bond];
  const list = valenceList(atom.z, atom.charge);
  if (!list) return atom.hExplicit;
  const v = list.find((x) => x >= sum);
  if (v === undefined) throw new SmilesError(`explicit valence ${sum} of ${SYMBOLS[atom.z]} too large`);
  return atom.hExplicit + (v - sum);
}

/** Pin an atom's hydrogen count (and charge) by turning it into an explicit bracket atom. */
function pinHydrogens(atom, hydrogens) {
  atom.bracket = true;
  atom.hExplicit = hydrogens;
}

/**
 * The parts of the ChEMBL standardiser (chembl_structure_pipeline) that matter for neutral
 * organic molecules, plus RDKit's own cleanUp: nitro and sulfoxide to charge-separated form,
 * imidic acid to amide, and the Uncharger. Everything it changes is reported in `warnings`
 * so callers can see the input was not already standard.
 */
function standardiseRules(atoms, bonds, adj, orders, ring, warnings) {
  // Nitro: [N;X3](=O)=O -> [N+](=O)[O-]   (also RDKit's pentavalent-N cleanUp)
  for (let i = 0; i < atoms.length; i++) {
    const atom = atoms[i];
    if (atom.z !== 7 || atom.charge !== 0 || atom.aromatic) continue;
    let sum = atom.hExplicit;
    for (const { bond } of adj[i]) sum += orders[bond];
    if (sum !== 5) continue;
    for (const { to, bond } of adj[i]) {
      const other = atoms[to];
      if (orders[bond] === 2 && other.z === 8 && other.charge === 0 && adj[to].length === 1
        && other.hExplicit === 0) {
        orders[bond] = 1;
        atom.charge = 1;
        other.charge = -1;
        pinHydrogens(other, 0);
        warnings.push("pentavalent N(=O) rewritten as charge-separated N+/O-");
        break;
      }
    }
  }
  // Sulfoxide: [!O][S+0;D3](=O)[!O] -> [!O][S+]([O-])[!O]
  for (let i = 0; i < atoms.length; i++) {
    const atom = atoms[i];
    if (atom.z !== 16 || atom.charge !== 0 || atom.aromatic || adj[i].length !== 3) continue;
    let oxygen = -1;
    let ok = true;
    for (const { to, bond } of adj[i]) {
      const other = atoms[to];
      if (orders[bond] === 2 && other.z === 8 && other.charge === 0 && adj[to].length === 1
        && other.hExplicit === 0 && oxygen < 0) {
        oxygen = bond;
      } else if (other.z === 8 || orders[bond] !== 1) {
        ok = false;
      }
    }
    if (ok && oxygen >= 0 && atom.hExplicit === 0) {
      const o = bonds[oxygen].a === i ? bonds[oxygen].b : bonds[oxygen].a;
      orders[oxygen] = 1;
      atom.charge = 1;
      atoms[o].charge = -1;
      pinHydrogens(atoms[o], 0);
      warnings.push("sulfoxide rewritten as S+/O-");
    }
  }
  // Bad amide tautomer: C(OH)=N (non-ring) -> C(=O)N
  for (let c = 0; c < atoms.length; c++) {
    if (atoms[c].z !== 6 || atoms[c].aromatic) continue;
    for (const { to: nIdx, bond: dbl } of adj[c]) {
      if (orders[dbl] !== 2 || ring[dbl] || atoms[nIdx].z !== 7 || atoms[nIdx].aromatic) continue;
      const hN = hydrogenCount(nIdx, atoms, adj, orders);
      if (hN > 1) continue;
      for (const { to: oIdx, bond: sgl } of adj[c]) {
        if (atoms[oIdx].z !== 8 || orders[sgl] !== 1 || adj[oIdx].length !== 1) continue;
        if (hydrogenCount(oIdx, atoms, adj, orders) !== 1 || atoms[oIdx].charge !== 0) continue;
        orders[dbl] = 1;
        orders[sgl] = 2;
        pinHydrogens(atoms[nIdx], hN + 1);
        pinHydrogens(atoms[oIdx], 0);
        warnings.push("imidic acid rewritten as amide");
        break;
      }
    }
  }
  uncharge(atoms, adj, orders, warnings);
}

/** Simplified rdMolStandardize.Uncharger: protonate anions, deprotonate cations that carry H. */
function uncharge(atoms, adj, orders, warnings) {
  const n = atoms.length;
  const hs = atoms.map((_, i) => hydrogenCount(i, atoms, adj, orders));
  const adjacentTo = (i, sign) => adj[i].some(({ to }) => Math.sign(atoms[to].charge) === sign);
  let quaternary = 0;
  for (let i = 0; i < n; i++) {
    if (atoms[i].charge > 0 && hs[i] === 0 && !adjacentTo(i, -1)) quaternary += atoms[i].charge;
  }
  const negatives = [];
  for (let i = 0; i < n; i++) {
    if (atoms[i].charge < 0 && !adjacentTo(i, 1)) negatives.push(i);
  }
  const protonated = [];
  for (let i = 0; i < n; i++) {
    if (atoms[i].charge > 0 && hs[i] > 0 && !adjacentTo(i, -1)) protonated.push(i);
  }
  let surplus = negatives.reduce((sum, i) => sum - atoms[i].charge, 0) - quaternary;
  let changed = false;
  for (const i of protonated) {
    atoms[i].charge -= 1;
    pinHydrogens(atoms[i], hs[i] - 1);
    changed = true;
  }
  const bondSum = (i) => adj[i].reduce((sum, { bond }) => sum + orders[bond], 0);
  for (const i of negatives) {
    const neutral = VALENCES.get(atoms[i].z);
    // Only protonate where the neutral atom can take the extra hydrogen ([B-] stays as it is).
    if (neutral && bondSum(i) + hs[i] + 1 > neutral[neutral.length - 1]) continue;
    while (surplus > 0 && atoms[i].charge < 0) {
      atoms[i].charge += 1;
      pinHydrogens(atoms[i], hs[i] + 1);
      hs[i] += 1;
      surplus--;
      changed = true;
    }
  }
  // A net negative charge left over (nitrate, perchlorate) protonates anions next to cations too.
  let net = atoms.reduce((sum, a) => sum + a.charge, 0);
  for (let i = 0; i < n && net < 0; i++) {
    if (atoms[i].charge < 0) {
      const neutral = VALENCES.get(atoms[i].z);
      if (neutral && bondSum(i) + hs[i] + 1 > neutral[neutral.length - 1]) continue;
      atoms[i].charge += 1;
      pinHydrogens(atoms[i], hs[i] + 1);
      net++;
      changed = true;
    }
  }
  if (changed) warnings.push("charged groups neutralised (simplified Uncharger)");
}

/** Assign Kekule bond orders to the aromatic ('ar') bonds by perfect matching. */
function kekulize(atoms, bonds, adj, orders) {
  const n = atoms.length;
  const need = new Uint8Array(n);
  let any = false;
  for (let i = 0; i < n; i++) {
    if (!atoms[i].aromatic) continue;
    any = true;
    let v = atoms[i].hExplicit;
    for (const { bond } of adj[i]) v += orders[bond] === "ar" ? 1 : orders[bond];
    const list = valenceList(atoms[i].z, atoms[i].charge);
    const dv = list ? list[0] : 4;
    if (dv - v >= 1) need[i] = 1;
  }
  if (!any) return;
  const arAdj = Array.from({ length: n }, () => []);
  bonds.forEach((bond, k) => {
    if (orders[k] === "ar") {
      arAdj[bond.a].push({ to: bond.b, bond: k });
      arAdj[bond.b].push({ to: bond.a, bond: k });
    }
  });
  const matchedWith = new Int32Array(n).fill(-1); // bond index of the double bond
  const taken = new Uint8Array(n);
  // Atoms not needing a double bond are excluded from matching.
  let steps = 0;
  const solve = () => {
    if (++steps > 500000) throw new SmilesError("kekulization search too large");
    // Pick an unmatched needy atom with the fewest free partners.
    let best = -1;
    let bestCount = 99;
    for (let i = 0; i < n; i++) {
      if (!need[i] || taken[i]) continue;
      let count = 0;
      for (const { to } of arAdj[i]) if (need[to] && !taken[to]) count++;
      if (count < bestCount) {
        best = i;
        bestCount = count;
        if (count <= 1) break;
      }
    }
    if (best < 0) return true;
    if (bestCount === 0) return false;
    for (const { to, bond } of arAdj[best]) {
      if (!need[to] || taken[to]) continue;
      taken[best] = taken[to] = 1;
      matchedWith[best] = matchedWith[to] = bond;
      if (solve()) return true;
      taken[best] = taken[to] = 0;
      matchedWith[best] = matchedWith[to] = -1;
    }
    return false;
  };
  if (!solve()) throw new SmilesError("Can't kekulize mol");
  bonds.forEach((bond, k) => {
    if (orders[k] === "ar") {
      orders[k] = matchedWith[bond.a] === k ? 2 : 1;
    }
  });
}

// ---------------------------------------------------------------------------------------
// Rings and aromaticity
// ---------------------------------------------------------------------------------------

function xorInto(target, source) {
  for (let k = 0; k < target.length; k++) target[k] ^= source[k];
}

class Gf2Basis {
  constructor(words) {
    this.words = words;
    this.rows = [];
  }

  reduce(vector) {
    const v = Uint32Array.from(vector);
    for (const row of this.rows) {
      if ((v[row.pivot >> 5] >>> (row.pivot & 31)) & 1) xorInto(v, row.vector);
    }
    return v;
  }

  /** Reduce and report whether the vector was independent (and then absorb it). */
  add(vector) {
    const v = this.reduce(vector);
    for (let w = 0; w < v.length; w++) {
      if (v[w]) {
        const pivot = w * 32 + (31 - Math.clz32(v[w] & -v[w]));
        this.rows.push({ pivot, vector: v });
        return true;
      }
    }
    return false;
  }

  isIndependent(vector) {
    return this.reduce(vector).some((x) => x !== 0);
  }
}

/** Relevant cycles (union of all SSSRs), i.e. RDKit's symmetrised SSSR, as atom/bond lists. */
function symmetrisedSssr(mol) {
  const { n, bonds, adj, isRing } = mol;
  const ringAdj = adj.map((list) => list.filter(({ bond }) => isRing[bond]));
  const ringAtoms = [];
  for (let i = 0; i < n; i++) if (ringAdj[i].length) ringAtoms.push(i);
  if (!ringAtoms.length) return [];
  const words = Math.ceil(bonds.length / 32);

  const candidates = new Map();
  const dist = new Int32Array(n);
  const parent = new Int32Array(n);
  const parentBond = new Int32Array(n);
  const branch = new Int32Array(n);
  for (const root of ringAtoms) {
    dist.fill(-1);
    dist[root] = 0;
    parent[root] = -1;
    parentBond[root] = -1;
    branch[root] = -1;
    const queue = [root];
    for (let head = 0; head < queue.length; head++) {
      const u = queue[head];
      for (const { to, bond } of ringAdj[u]) {
        if (dist[to] < 0) {
          dist[to] = dist[u] + 1;
          parent[to] = u;
          parentBond[to] = bond;
          branch[to] = u === root ? to : branch[u];
          queue.push(to);
        }
      }
    }
    for (const x of queue) {
      for (const { to: y, bond } of ringAdj[x]) {
        if (y < x && dist[y] >= 0) {
          // consider each edge once (y < x); skip tree edges and same-branch edges
          if (parentBond[x] === bond || parentBond[y] === bond) continue;
          if (branch[x] === branch[y] || branch[x] < 0 || branch[y] < 0) continue;
          const cycleBonds = [bond];
          const cycleAtoms = [x];
          for (let u = x; u !== root; u = parent[u]) {
            cycleBonds.push(parentBond[u]);
            cycleAtoms.push(parent[u]);
          }
          for (let u = y; u !== root; u = parent[u]) {
            cycleBonds.push(parentBond[u]);
            cycleAtoms.push(u);
          }
          const key = cycleBonds.slice().sort((p, q) => p - q).join(",");
          if (!candidates.has(key)) {
            candidates.set(key, { bonds: cycleBonds, atoms: [...new Set(cycleAtoms)] });
          }
        }
      }
    }
  }
  const list = [...candidates.values()].sort((p, q) => p.bonds.length - q.bonds.length);
  const { comp } = components(n, ringAdj);
  const ringComps = new Set(ringAtoms.map((a) => comp[a])).size;
  let ringBondCount = 0;
  for (let b = 0; b < bonds.length; b++) if (isRing[b]) ringBondCount++;
  const target = ringBondCount - ringAtoms.length + ringComps;

  const toVector = (cycle) => {
    const v = new Uint32Array(words);
    for (const b of cycle.bonds) v[b >> 5] |= 1 << (b & 31);
    return v;
  };
  const low = new Gf2Basis(words);
  const rings = [];
  let at = 0;
  while (at < list.length && low.rows.length < target) {
    let end = at;
    while (end < list.length && list[end].bonds.length === list[at].bonds.length) end++;
    const group = list.slice(at, end);
    const vectors = group.map(toVector);
    group.forEach((cycle, k) => {
      if (low.isIndependent(vectors[k])) rings.push(cycle);
    });
    vectors.forEach((v) => low.add(v));
    at = end;
  }
  return rings;
}

/** Number of pi electrons an atom donates to a ring, or -1 when it cannot be aromatic. */
function donatedElectrons(mol, i) {
  const { z, charge, nH, adj, bonds, isRing } = mol;
  if (z[i] > 18 && z[i] !== 34 && z[i] !== 52) return -1;
  const eff = charge[i] ? z[i] - charge[i] : z[i];
  const list = VALENCES.get(eff);
  if (!list || list[0] <= 1) return -1;
  const dv = list[0];
  const degree = adj[i].length + nH[i];
  if (degree > 3) return -1;
  // Hypervalent atoms ([SH]=, S=O, ...) never take part in an aromatic ring.
  let valence = nH[i];
  for (const { bond } of adj[i]) valence += bonds[bond].order;
  if (valence > dv) return -1;
  const nlp = Math.max((OUTER_ELECTRONS.get(eff) ?? 0) - dv, 0);
  let res = (dv - degree) + nlp;
  let nMultiple = 0;
  let exocyclic = -1;
  let hasRingMultiple = false;
  for (const { to, bond } of adj[i]) {
    const order = bonds[bond].order;
    if (order === TRIPLE) return -1;
    if (order >= 2) {
      nMultiple++;
      if (isRing[bond]) hasRingMultiple = true;
      else exocyclic = to;
    }
  }
  if (nMultiple > 1) return -1; // cumulene-like
  if (res < 0) return -1;
  if (res > 1 && nMultiple) res = 1;
  if (res >= 2) return 2;
  // A vacant p orbital (no pi electron at all) is accepted for carbon only (tropone, cyclopropenyl).
  if (res === 0) return hasRingMultiple ? 1 : (z[i] === 6 ? 0 : -1);
  // one electron
  if (exocyclic >= 0) {
    const mine = ELECTRONEGATIVITY.get(z[i]) ?? 2.5;
    const theirs = ELECTRONEGATIVITY.get(z[exocyclic]) ?? 2.5;
    return theirs > mine ? 0 : 1;
  }
  return 1;
}

// RDKit only combines pairs of fused rings: larger unions (porphyrins, peri-fused systems)
// were never aromatised by it in testing on ~17k molecules.
const MAX_FUSED_RINGS = 2;
const MAX_FUSED_SUBSETS = 20000;

/** RDKit's default aromaticity model: Hueckel rule on single rings and fused systems. */
function perceiveAromaticity(mol) {
  const { n, bonds } = mol;
  const rings = symmetrisedSssr(mol);
  if (!rings.length) return;
  const electrons = new Int8Array(n);
  for (let i = 0; i < n; i++) {
    electrons[i] = mol.adj[i].some((e) => mol.isRing[e.bond]) ? donatedElectrons(mol, i) : -1;
  }
  const eligible = rings.filter((ring) => ring.atoms.every((a) => electrons[a] >= 0));
  const bondAromatic = new Uint8Array(bonds.length);
  const atomAromatic = new Uint8Array(n);
  const markRing = (ring) => {
    for (const a of ring.atoms) atomAromatic[a] = 1;
    for (const b of ring.bonds) bondAromatic[b] = 1;
  };
  for (const ring of eligible) {
    let sum = 0;
    for (const a of ring.atoms) sum += electrons[a];
    ring.aromatic = sum % 4 === 2;
    if (ring.aromatic) markRing(ring);
  }

  // Fused systems: connected sets of eligible rings sharing a bond.
  const m = eligible.length;
  if (m > 1) {
    const neighbours = Array.from({ length: m }, () => []);
    const bondSets = eligible.map((r) => new Set(r.bonds));
    for (let p = 0; p < m; p++) {
      for (let q = p + 1; q < m; q++) {
        if (eligible[q].bonds.some((b) => bondSets[p].has(b))) {
          neighbours[p].push(q);
          neighbours[q].push(p);
        }
      }
    }
    const seen = new Set();
    let evaluated = 0;
    const grow = (subset) => {
      if (evaluated++ > MAX_FUSED_SUBSETS) return;
      if (subset.length >= 2) {
        const atoms = new Set();
        for (const r of subset) for (const a of eligible[r].atoms) atoms.add(a);
        let sum = 0;
        for (const a of atoms) sum += electrons[a];
        if (sum % 4 === 2) {
          const counts = new Map();
          for (const r of subset) for (const b of eligible[r].bonds) counts.set(b, (counts.get(b) ?? 0) + 1);
          for (const a of atoms) atomAromatic[a] = 1;
          for (const [b, c] of counts) if (c === 1) bondAromatic[b] = 1;
        }
      }
      if (subset.length >= MAX_FUSED_RINGS) return;
      const members = new Set(subset);
      for (const r of subset) {
        for (const q of neighbours[r]) {
          if (members.has(q)) continue;
          const next = [...subset, q].sort((x, y) => x - y);
          const key = next.join(",");
          if (seen.has(key)) continue;
          seen.add(key);
          grow(next);
        }
      }
    };
    for (let r = 0; r < m; r++) {
      const key = String(r);
      if (seen.has(key)) continue;
      seen.add(key);
      grow([r]);
    }
  }

  for (let i = 0; i < n; i++) mol.aromatic[i] = atomAromatic[i];
  bonds.forEach((bond, b) => {
    bond.type = bondAromatic[b] ? AROMATIC : bond.order;
  });
}

// ---------------------------------------------------------------------------------------
// Canonical atom ranking (RDKit new_canon.cpp)
// ---------------------------------------------------------------------------------------

/** RDKit compares formal charges as unsigned integers, so negative charges sort last. */
function unsignedCharge(charge) {
  return charge < 0 ? charge + 4294967296 : charge;
}

/** Lexicographic comparison; a shorter list sorts first when one is a prefix of the other. */
function compareLists(a, b) {
  const common = Math.min(a.length, b.length);
  for (let k = 0; k < common; k++) if (a[k] !== b[k]) return a[k] - b[k];
  return a.length - b.length;
}

/**
 * @param {{n:number, z:Int32Array, deg:Int32Array, nH:Int32Array, charge:Int32Array,
 *   isotope:Int32Array, nbr:Array<Array<[number, number]>>}} g graph; nbr[i] = [[j, bondType]]
 * @returns {Int32Array} canonical ranks with ties broken
 */
function canonicalRanks(g) {
  const { n, z, deg, nH, charge, isotope, nbr } = g;
  const cls = new Int32Array(n);
  const order = Int32Array.from({ length: n }, (_, i) => i);
  const size = new Int32Array(n);
  if (n === 0) return cls;
  size[0] = n;
  const inActive = new Uint8Array(n);
  const touched = new Uint8Array(n);
  let active = [];
  let useNbrs = false;

  // Atom invariants, in RDKit's order: degree, atomic number, isotope, total H count and
  // charge. Empirically (verified against RDKit on thousands of molecules) the sorted list
  // of incident bond types is part of this comparison too: it is what makes an atom with an
  // exocyclic double bond rank above ring atoms with the same neighbour classes.
  const typeLists = nbr.map((list) => list.map(([, t]) => t).sort((a, b) => b - a));
  const baseCompare = (i, j) => (deg[i] - deg[j]) || (z[i] - z[j]) || (isotope[i] - isotope[j])
    || (nH[i] - nH[j]) || (unsignedCharge(charge[i]) - unsignedCharge(charge[j]))
    || compareLists(typeLists[i], typeLists[j]);

  // Sorted (descending) neighbour keys: bond type major, neighbour class minor.
  const neighbourKey = (i) => {
    const list = nbr[i];
    const keys = new Array(list.length);
    for (let k = 0; k < list.length; k++) keys[k] = list[k][1] * 1048576 + cls[list[k][0]] + 1;
    keys.sort((a, b) => b - a);
    return keys;
  };

  const activate = () => {
    // ActivatePartitions: push every multi-member partition, ascending by position.
    active = [];
    inActive.fill(0);
    let i = 0;
    while (i < n) {
      const s = size[i] || 1;
      if (s > 1) {
        active.push(i);
        inActive[i] = 1;
      }
      i += s;
    }
  };

  const refinePartitions = () => {
    while (active.length) {
      const s = active.pop();
      inActive[s] = 0;
      const len = size[s];
      const ids = Array.from(order.subarray(s, s + len));
      let compare;
      if (useNbrs) {
        const keys = new Map();
        for (const id of ids) keys.set(id, neighbourKey(id));
        compare = (i, j) => {
          const base = baseCompare(i, j);
          if (base) return base;
          return compareLists(keys.get(i), keys.get(j));
        };
      } else {
        compare = baseCompare;
      }
      ids.sort(compare); // stable
      for (let k = 0; k < len; k++) {
        order[s + k] = ids[k];
        size[s + k] = 0;
      }
      let groupStart = 0;
      const groupStarts = [];
      for (let k = 1; k <= len; k++) {
        if (k === len || compare(ids[k - 1], ids[k]) !== 0) {
          size[s + groupStart] = k - groupStart;
          groupStarts.push(groupStart);
          groupStart = k;
        }
      }
      // Atoms outside the first tie group get new class ids.
      for (let g = 1; g < groupStarts.length; g++) {
        const end = g + 1 < groupStarts.length ? groupStarts[g + 1] : len;
        for (let k = groupStarts[g]; k < end; k++) cls[ids[k]] = s + groupStarts[g];
      }
      const first = groupStarts.length > 1 ? groupStarts[1] : len;
      for (let k = first; k < len; k++) {
        for (const [nb] of nbr[ids[k]]) touched[cls[nb]] = 1;
      }
      for (let ii = 0; ii < n; ii++) {
        if (touched[ii]) {
          if (size[ii] > 1 && !inActive[ii]) {
            active.push(ii);
            inActive[ii] = 1;
          }
          touched[ii] = 0;
        }
      }
    }
  };

  activate();
  refinePartitions();
  useNbrs = true;
  activate();
  refinePartitions();

  // BreakTies
  const headSize = (atom) => (order[cls[atom]] === atom ? size[cls[atom]] : 0);
  for (let i = 0; i < n; i++) {
    const partition = order[i];
    const oldPart = cls[partition];
    while (headSize(partition) > 1) {
      const len = headSize(partition);
      const offset = cls[partition] + len - 1;
      const index = order[offset];
      cls[index] = offset;
      size[cls[partition]] = len - 1;
      size[offset] = 1;
      if (deg[index] < 1) continue;
      for (const [nb] of nbr[index]) touched[cls[nb]] = 1;
      for (let ii = 0; ii < n; ii++) {
        if (touched[ii]) {
          if (size[ii] > 1 && !inActive[ii]) {
            active.push(ii);
            inActive[ii] = 1;
          }
          touched[ii] = 0;
        }
      }
      refinePartitions();
    }
    if (cls[partition] !== oldPart) i--;
  }
  return cls;
}

/** Ranking graph for a molecule without explicit hydrogens (used for SMILES writing). */
function hydrogenFreeGraph(mol) {
  const { n } = mol;
  const nbr = Array.from({ length: n }, () => []);
  for (const bond of mol.bonds) {
    nbr[bond.a].push([bond.b, bond.type]);
    nbr[bond.b].push([bond.a, bond.type]);
  }
  return {
    n,
    z: mol.z,
    deg: Int32Array.from(nbr, (list) => list.length),
    nH: mol.nH,
    charge: mol.charge,
    isotope: mol.isotope,
    nbr,
  };
}

/** Ranking graph of `AddHs(mol)`; atoms must already be in their final index order. */
function explicitHydrogenGraph(mol) {
  const heavy = mol.n;
  let total = heavy;
  for (let i = 0; i < heavy; i++) total += mol.nH[i];
  const z = new Int32Array(total);
  const deg = new Int32Array(total);
  const nH = new Int32Array(total);
  const charge = new Int32Array(total);
  const isotope = new Int32Array(total);
  const nbr = Array.from({ length: total }, () => []);
  for (let i = 0; i < heavy; i++) {
    z[i] = mol.z[i];
    charge[i] = mol.charge[i];
    isotope[i] = mol.isotope[i];
  }
  // heavy-atom bonds keep their creation order; RDKit's bond list order is not consulted
  // by the ranking except through stable sorting of equal keys, which we do not need.
  for (const bond of mol.bonds) {
    nbr[bond.a].push([bond.b, bond.type]);
    nbr[bond.b].push([bond.a, bond.type]);
  }
  let next = heavy;
  for (let i = 0; i < heavy; i++) {
    for (let k = 0; k < mol.nH[i]; k++) {
      z[next] = 1;
      deg[next] = 1;
      nbr[next].push([i, SINGLE]);
      nbr[i].push([next, SINGLE]);
      next++;
    }
  }
  for (let i = 0; i < heavy; i++) deg[i] = nbr[i].length;
  return { n: total, z, deg, nH, charge, isotope, nbr };
}

// ---------------------------------------------------------------------------------------
// Canonical SMILES writer (RDKit canon.cpp / SmilesWrite.cpp)
// ---------------------------------------------------------------------------------------

const MAX_BONDTYPE = 32;
const MAX_NATOMS = 1000000;

function writeCanonicalSmiles(mol, ranks) {
  const { n, bonds, adj, isRing } = mol;
  const colour = new Uint8Array(n); // 0 white, 1 grey, 2 black
  const out = { order: [], pieces: [] };

  const sortKey = (u, edge, atCycles) => {
    let rank = ranks[edge.to];
    if (atCycles && colour[edge.to] === 1) {
      rank -= (MAX_BONDTYPE + 1) * MAX_NATOMS * MAX_NATOMS;
      rank += (MAX_BONDTYPE - bonds[edge.bond].type) * MAX_NATOMS;
    } else if (isRing[edge.bond]) {
      rank += (MAX_BONDTYPE - bonds[edge.bond].type) * MAX_NATOMS * MAX_NATOMS;
    }
    return rank;
  };

  // Pass 1: find the ring-closure bonds.
  const closures = Array.from({ length: n }, () => []);
  const isClosure = new Uint8Array(bonds.length);
  const findCycles = (u, inBond) => {
    colour[u] = 1;
    const possibles = [];
    for (const edge of adj[u]) {
      if (edge.bond === inBond) continue;
      possibles.push({ key: sortKey(u, edge, true), edge });
    }
    possibles.sort((p, q) => p.key - q.key);
    for (const { edge } of possibles) {
      if (colour[edge.to] === 0) {
        findCycles(edge.to, edge.bond);
      } else if (colour[edge.to] === 1) {
        if (!isClosure[edge.bond]) {
          isClosure[edge.bond] = 1;
          closures[u].push(edge.bond);
          closures[edge.to].push(edge.bond);
        }
      }
    }
    colour[u] = 2;
  };

  // Root: lowest-rank atom (single connected component assumed).
  let root = 0;
  for (let i = 1; i < n; i++) if (ranks[i] < ranks[root]) root = i;
  findCycles(root, -1);

  // Pass 2: build the output.
  colour.fill(0);
  const digitOf = new Map(); // bond -> ring digit while open
  const available = []; // available[k] = true when digit k+1 is free
  const text = [];
  const emitDigit = (d) => (d < 10 ? String(d) : d < 100 ? `%${d}` : `%(${d})`);
  const build = (u, inBond) => {
    colour[u] = 1;
    out.order.push(u);
    text.push(atomToken(mol, u));
    const seenFromHere = new Set();
    const released = [];
    for (const bond of closures[u]) {
      const { a, b } = bonds[bond];
      const other = a === u ? b : a;
      seenFromHere.add(other);
      if (digitOf.has(bond)) {
        const d = digitOf.get(bond);
        text.push(bondToken(mol, bond));
        text.push(emitDigit(d));
        released.push(d);
        digitOf.delete(bond);
      } else {
        let k = available.indexOf(true);
        if (k < 0) {
          k = available.length;
          available.push(false);
        }
        available[k] = false;
        digitOf.set(bond, k + 1);
        text.push(emitDigit(k + 1));
      }
    }
    // RDKit frees a digit only after all closures of this atom are written, so an atom that
    // closes ring 1 and opens a new ring writes "12", not "11".
    for (const d of released) available[d - 1] = true;
    const possibles = [];
    for (const edge of adj[u]) {
      if (edge.bond === inBond) continue;
      if (colour[edge.to] !== 0 || seenFromHere.has(edge.to) || isClosure[edge.bond]) continue;
      possibles.push({ key: sortKey(u, edge, false), edge });
    }
    possibles.sort((p, q) => p.key - q.key);
    possibles.forEach(({ edge }, k) => {
      if (colour[edge.to] !== 0) return;
      const branching = k + 1 < possibles.length;
      if (branching) text.push("(");
      text.push(bondToken(mol, edge.bond));
      build(edge.to, edge.bond);
      if (branching) text.push(")");
    });
    colour[u] = 2;
  };
  build(root, -1);
  return { smiles: text.join(""), order: out.order };
}

function bondToken(mol, bondIndex) {
  const bond = mol.bonds[bondIndex];
  const bothAromatic = mol.aromatic[bond.a] && mol.aromatic[bond.b];
  switch (bond.type) {
    case SINGLE: return bothAromatic ? "-" : "";
    case DOUBLE: return "=";
    case TRIPLE: return "#";
    case AROMATIC: return bothAromatic ? "" : ":";
    default: return "";
  }
}

function atomToken(mol, i) {
  const z = mol.z[i];
  const symbol = SYMBOLS[z];
  const aromatic = mol.aromatic[i] === 1;
  let needsBracket = mol.charge[i] !== 0 || mol.isotope[i] !== 0 || !ORGANIC_SUBSET.has(z);
  if (!needsBracket) {
    if (aromatic && (z === 7 || z === 15) && mol.nH[i] > 0) {
      needsBracket = true;
    } else {
      let sum = 0;
      for (const { bond } of mol.adj[i]) sum += mol.bonds[bond].order;
      const list = VALENCES.get(z);
      const v = list.find((x) => x >= sum);
      const implicit = v === undefined ? 0 : v - sum;
      // RDKit also brackets an atom carrying hydrogens in a non-default valence state, e.g. [PH](=O).
      if (implicit !== mol.nH[i] || (mol.nH[i] > 0 && sum + mol.nH[i] !== list[0])) needsBracket = true;
    }
  }
  const sym = aromatic ? symbol.toLowerCase() : symbol;
  if (!needsBracket) return sym;
  let s = "[";
  if (mol.isotope[i]) s += mol.isotope[i];
  s += sym;
  if (mol.nH[i]) s += mol.nH[i] > 1 ? `H${mol.nH[i]}` : "H";
  const c = mol.charge[i];
  if (c) s += (c > 0 ? "+" : "-") + (Math.abs(c) > 1 ? Math.abs(c) : "");
  return `${s}]`;
}

// ---------------------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------------------

/** True when every formal charge sits in an adjacent +/- pair (nitro, N-oxide, sulfoxide). */
function onlyChargeSeparatedPairs(mol) {
  for (let i = 0; i < mol.n; i++) {
    const c = mol.charge[i];
    if (!c) continue;
    if (Math.abs(c) !== 1) return false;
    if (!mol.adj[i].some(({ to }) => mol.charge[to] === -c)) return false;
  }
  return true;
}

/** Keep only the largest component (by heavy-atom count) of a multi-fragment molecule. */
function largestFragment(parsed) {
  const { atoms, bonds } = parsed;
  const adj = buildAdjacency(atoms.length, bonds);
  const { comp, count } = components(atoms.length, adj);
  if (count <= 1) return { parsed, fragments: 1 };
  const sizes = new Array(count).fill(0);
  const mass = new Array(count).fill(0);
  atoms.forEach((a, i) => {
    if (a.z > 1) sizes[comp[i]]++;
    mass[comp[i]] += a.z;
  });
  let best = 0;
  for (let c = 1; c < count; c++) {
    if (sizes[c] > sizes[best] || (sizes[c] === sizes[best] && mass[c] > mass[best])) best = c;
  }
  const remap = new Map();
  const keep = [];
  atoms.forEach((a, i) => {
    if (comp[i] === best) {
      remap.set(i, keep.length);
      keep.push(a);
    }
  });
  const keptBonds = bonds.filter((b) => comp[b.a] === best)
    .map((b) => ({ ...b, a: remap.get(b.a), b: remap.get(b.b) }));
  return { parsed: { atoms: keep, bonds: keptBonds, flags: parsed.flags }, fragments: count };
}

/** Reorder a molecule so atom k of the result is atom `order[k]` of the input. */
function reorder(mol, order) {
  const position = new Int32Array(mol.n);
  order.forEach((old, k) => { position[old] = k; });
  const n = mol.n;
  const pick = (array) => array.constructor.from({ length: n }, (_, k) => array[order[k]]);
  const bonds = mol.bonds.map((b) => ({
    a: position[b.a], b: position[b.b], order: b.order, type: b.type,
  }));
  return {
    n,
    z: pick(mol.z),
    charge: pick(mol.charge),
    isotope: pick(mol.isotope),
    nH: pick(mol.nH),
    aromatic: pick(mol.aromatic),
    bonds,
    adj: buildAdjacency(n, bonds),
    isRing: mol.isRing,
  };
}

/**
 * Mirror of `webgpu/tools/ligand_atoms.py`: RDKit canonical SMILES of the (standardised)
 * ligand, and the GenFree heavy-atom names in that SMILES' atom order.
 *
 * @param {string} smiles
 * @returns {{smiles: string, atomNames: string[], exact: boolean, warnings: string[]}}
 */
export function ligandAtomsFromSmiles(smiles) {
  const warnings = [];
  let exact = true;
  const approximate = (message) => {
    exact = false;
    warnings.push(message);
  };
  let parsed = parseSmiles(String(smiles).trim());
  const flags = parsed.flags;
  if (flags.stereo) approximate("stereo marks present: CIP-dependent ranking is not modelled");
  if (flags.isotope) warnings.push("isotopes present (ranked and written as RDKit does)");
  if (flags.atomMap) approximate("atom map numbers present");

  // chembl_structure_pipeline.exclude_flag: metals and boranes are rejected.
  let boron = 0;
  for (const atom of parsed.atoms) {
    if (EXCLUDED_METALS.has(SYMBOLS[atom.z])) throw new SmilesError("Molecule is excluded (metal)");
    if (atom.z === 5) boron++;
  }
  if (boron > 7) throw new SmilesError("Molecule is excluded (more than 7 boron atoms)");

  const frag = largestFragment(parsed);
  parsed = frag.parsed;
  if (frag.fragments > 1) {
    approximate(`${frag.fragments} fragments: kept the largest, standardiser behaviour approximated`);
  }

  const mol0 = buildMolecule(parsed, warnings, approximate);
  if (!onlyChargeSeparatedPairs(mol0)) {
    approximate("charged atoms present: the ChEMBL standardiser may neutralise or reionise them");
  }
  for (let i = 0; i < mol0.n; i++) {
    if (!SUPPORTED_ELEMENTS.has(mol0.z[i])) {
      approximate(`element ${SYMBOLS[mol0.z[i]]} is outside the validated subset`);
      break;
    }
  }

  // Canonical SMILES from the H-less molecule.
  const ranks0 = canonicalRanks(hydrogenFreeGraph(mol0));
  const written = writeCanonicalSmiles(mol0, ranks0);

  // Names from AddHs(MolFromSmiles(canonical)): atoms in written order, Hs appended.
  const mol = reorder(mol0, written.order);
  const ranks = canonicalRanks(explicitHydrogenGraph(mol));
  const byRank = Array.from(ranks.keys()).sort((p, q) => ranks[p] - ranks[q]);
  const counts = new Map();
  const names = new Array(mol.n);
  // Hydrogens interleave with heavy atoms in rank order, but names are per element, so only
  // the relative order of the heavy atoms of one element matters.
  for (const i of byRank) {
    const symbol = SYMBOLS[i < mol.n ? mol.z[i] : 1].toUpperCase();
    const count = (counts.get(symbol) ?? 0) + 1;
    counts.set(symbol, count);
    if (i < mol.n) names[i] = symbol + count;
  }
  return { smiles: written.smiles, atomNames: names, exact, warnings };
}

export const _internals = { parseSmiles, buildMolecule, canonicalRanks, hydrogenFreeGraph, explicitHydrogenGraph,
  writeCanonicalSmiles, symmetrisedSssr };
