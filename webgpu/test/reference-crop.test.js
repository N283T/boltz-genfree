import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { describeReference, loadReferenceCrop, parseCifCategories, parseStructure, proteinAtomCounts, selectReceptorTokens } from "../src/reference-crop.js";

const fixture = (name) => new URL(`./fixtures/crop/${name}`, import.meta.url);
const expected = JSON.parse(readFileSync(fixture("expected.json"), "utf8"));

// expected.json comes from GenFree's own src/boltz/reference_crop.py (gemmi) run on the same files.
for (const name of Object.keys(expected)) {
  test(`reference crop matches GenFree's Python on ${name}`, () => {
    const e = expected[name];
    const structure = parseStructure(readFileSync(fixture(name), "utf8"));
    const crop = loadReferenceCrop(structure, e.sequence, { proteinChain: e.proteinChain, ligand: e.ligand });
    assert.deepEqual([crop.ligandChain, crop.ligandResidue, crop.ligandCcd], e.ligandOut);
    assert.deepEqual(crop.seedOrder, e.seedOrder);
    for (const [tokens, want] of Object.entries(e.selected)) {
      const got = (() => { try { return selectReceptorTokens(crop, e.sequence, { ligandTokens: Number(tokens) }); } catch (x) { return x.message; } })();
      assert.deepEqual(got, want, `ligand_tokens=${tokens}`);
    }
  });
}

test("the declared mmCIF sequence is used: observed residues map onto it by label_seq_id", () => {
  const e = expected["gsk3b.cif"];
  const structure = parseStructure(readFileSync(fixture("gsk3b.cif"), "utf8"));
  const info = describeReference(structure);
  assert.equal(info.chains[0].declared, e.sequence.length);
  assert.ok(info.chains[0].observed < info.chains[0].declared);
  assert.ok(info.ligands.some((l) => l.ccd === "ANP" && l.residue === 430));
  const observedOnly = e.sequence.slice(0, 50);   // not the declared sequence and not the observed one
  assert.throws(() => loadReferenceCrop(structure, observedOnly, { proteinChain: "A", ligand: e.ligand }), /exactly matches/);
});

test("errors follow GenFree: missing chain, missing or ambiguous ligand, budget too small", () => {
  const e = expected["gsk3b.cif"];
  const structure = parseStructure(readFileSync(fixture("gsk3b.cif"), "utf8"));
  assert.throws(() => loadReferenceCrop(structure, e.sequence, { proteinChain: "Z", ligand: e.ligand }), /chain 'Z' was not found/);
  assert.throws(() => loadReferenceCrop(structure, e.sequence, { proteinChain: "A", ligand: { chain: "A", residue: 999, ccd: "ANP" } }), /matched no residue/);
  assert.throws(() => loadReferenceCrop(structure, e.sequence, { proteinChain: "A", ligand: { chain: "A", residue: 430 } }), /together/);
  const crop = loadReferenceCrop(structure, e.sequence, { proteinChain: "A", ligand: e.ligand });
  assert.throws(() => selectReceptorTokens(crop, e.sequence, { ligandTokens: 250 }), /No reference-derived receptor block fits/);
});

test("mmCIF category reader handles loops, quoted values and text fields", () => {
  const cif = `data_x
_entity.id 1
_entity.type polymer
loop_
_foo.a
_foo.b
x 'q w'
y "e r"
;multi
line
; z
`;
  const cats = parseCifCategories(cif, new Set(["_entity", "_foo"]));
  assert.deepEqual(cats.get("_entity"), [{ id: "1", type: "polymer" }]);
  assert.deepEqual(cats.get("_foo").slice(0, 2), [{ a: "x", b: "q w" }, { a: "y", b: "e r" }]);
  assert.equal(cats.get("_foo")[2].a, "multi\nline");
});

test("protein atom counts follow Boltz's ref_atoms (UNK = 5)", () => {
  assert.deepEqual(proteinAtomCounts("AGWX"), [5, 4, 14, 5]);
});
