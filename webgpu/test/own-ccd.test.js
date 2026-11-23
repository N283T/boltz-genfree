// CCD ligands (src/trunk/own/ccd.js, ligand.js, featurise.js, chain.js): offline, on hand-written mmCIF fixtures.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { ccdUrl, fetchCcdComponent, loopOf, parseCcdComponent } from "../src/trunk/own/ccd.js";
import { ligandGraph, ligandGraphFromCcd } from "../src/trunk/own/ligand.js";
import { featuriseOwn } from "../src/trunk/own/featurise.js";
import { BOND_TYPE } from "../src/trunk/own/boltz-const.js";

const fixture = (name) => readFileSync(new URL(`./fixtures/${name}`, import.meta.url), "utf8");

test("ccdUrl", () => {
  assert.equal(ccdUrl(" gol "), "https://files.rcsb.org/ligands/download/GOL.cif");
  assert.throws(() => ccdUrl("a/b"), /not a CCD code/);
  assert.throws(() => ccdUrl("ABCDEF"), /not a CCD code/);
});

test("parseCcdComponent: heavy atoms in file order, names, elements, ideal coordinates, bonds without hydrogens", () => {
  const c = parseCcdComponent(fixture("ccd-GOL.cif"));
  assert.equal(c.code, "GOL");
  assert.deepEqual(c.atoms.map((a) => a.name), ["C1", "O1", "C2", "O2", "C3", "O3"]);
  assert.deepEqual(c.atoms.map((a) => a.element), [6, 8, 6, 8, 6, 8]);
  assert.deepEqual([c.atoms[0].x, c.atoms[0].y, c.atoms[0].z], [-0.5, 0.1, 0.2]); // ideal, not model
  assert.equal(c.bonds.length, 5); // the two bonds to H are dropped
  assert.deepEqual(c.bonds[1], { a: 0, b: 2, bondType: "SINGLE" });
});

test("parseCcdComponent: quoted names, charges, aromatic flag over value order, model coordinates centred", () => {
  const c = parseCcdComponent(fixture("ccd-BNZ-like.cif"));
  assert.deepEqual(c.atoms.map((a) => a.name), ["O1'", "C1", "C2", "N3"]);
  assert.deepEqual(c.atoms.map((a) => a.charge), [0, 0, 0, 1]);
  assert.deepEqual(c.bonds.map((b) => b.bondType), ["SINGLE", "AROMATIC", "TRIPLE"]);
  assert.equal(c.atoms.reduce((s, a) => s + a.x, 0), 0); // ideal missing: model coordinates centred on the atoms
});

test("parseCcdComponent: a monatomic ion written in the single-row form or with '?' coordinates", () => {
  const ion = parseCcdComponent(fixture("ccd-MG.cif"));
  assert.deepEqual(ion.atoms, [{ name: "MG", element: 12, charge: 2, x: 0, y: 0, z: 0 }]);
  assert.deepEqual(ion.bonds, []);
  const single = loopOf("_chem_comp_atom.atom_id MG\n_chem_comp_atom.type_symbol MG\n#\n", "_chem_comp_atom");
  assert.deepEqual(single.rows, [["MG", "MG"]]);
  assert.throws(() => parseCcdComponent("data_X\n_chem_comp.id X\n"), /no heavy atoms/);
  assert.throws(() => parseCcdComponent("data_X\n"), /no _chem_comp.id/);
});

test("ligandGraphFromCcd: same structure and type_bonds encoding as the SMILES path", () => {
  const ccd = ligandGraphFromCcd(parseCcdComponent(fixture("ccd-GOL.cif")));
  const smiles = ligandGraph("OCC(O)CO");
  assert.equal(ccd.atoms.length, smiles.atoms.length);
  assert.deepEqual(Object.keys(ccd), Object.keys(smiles));
  assert.deepEqual(Object.keys(ccd.atoms[0]), Object.keys(smiles.atoms[0]));
  assert.deepEqual(Object.keys(ccd.bonds[0]), Object.keys(smiles.bonds[0]));
  assert.ok(ccd.bonds.every((b) => b.type === BOND_TYPE.SINGLE));
  const arom = ligandGraphFromCcd(parseCcdComponent(fixture("ccd-BNZ-like.cif")));
  assert.deepEqual(arom.bonds.map((b) => b.type), [BOND_TYPE.SINGLE, BOND_TYPE.AROMATIC, BOND_TYPE.TRIPLE]);
});

test("featuriseOwn with a CCD ligand: tokens, names, bonds, type_bonds", () => {
  const component = parseCcdComponent(fixture("ccd-GOL.cif"));
  const r = featuriseOwn({ sequence: "MKTAYI", smiles: "GOL", ligandKind: "ccd", ligandComponent: component });
  assert.equal(r.receptorTokens, 6);
  assert.equal(r.ligandTokens, 6);
  const f = r.features;
  assert.equal(f.affinity_token_mask.data.slice(6).every((v) => v === 1), true);
  const T = 12;
  assert.equal(f.token_bonds.data[6 * T + 7], 1); // C1-O1
  assert.equal(f.type_bonds.data[6 * T + 7], BOND_TYPE.SINGLE + 1);
  assert.equal(f.token_bonds.data[6 * T + 11], 0);
  // atom name "C1" -> codes 'C'-32, '1'-32
  const atom = r.compact.atomToToken.findIndex((t) => t === 6);
  assert.deepEqual(Array.from(r.compact.atomNameCodes.slice(atom * 4, atom * 4 + 4)), ["C".charCodeAt(0) - 32, "1".charCodeAt(0) - 32, 0, 0]);
  assert.throws(() => featuriseOwn({ sequence: "MK", smiles: "GOL", ligandKind: "ccd" }), /ligandComponent/);
});

test("fetchCcdComponent with a mocked fetch (no network)", async () => {
  const seen = [];
  const mock = async (url) => { seen.push(url); return { ok: true, status: 200, text: async () => fixture("ccd-GOL.cif") }; };
  const c = await fetchCcdComponent("gol", mock);
  assert.deepEqual(seen, ["https://files.rcsb.org/ligands/download/GOL.cif"]);
  assert.equal(c.atoms.length, 6);
  await assert.rejects(fetchCcdComponent("ZZZ", async () => ({ ok: false, status: 404 })), /HTTP 404/);
});
