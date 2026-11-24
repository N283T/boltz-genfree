// Receptor cropping (`receptorIndices`) of the own featuriser and the A3M crop.
import { test } from "node:test";
import assert from "node:assert/strict";
import { featuriseOwn } from "../src/trunk/own/featurise.js";
import { buildMsa, cropA3m, parseA3m } from "../src/trunk/own/msa.js";
import { TOKEN_ID } from "../src/trunk/own/boltz-const.js";

test("cropA3m keeps the selected query columns and the headers", () => {
  assert.equal(cropA3m(">q\nMKTV\n>a\nM-TV", [0, 2]), ">q\nMT\n>a\nMT");
  assert.equal(cropA3m(">a\nM-T", [1, 2]), ">a\n-T");
});

test("cropA3m moves insertions with the column that follows them", () => {
  assert.equal(cropA3m(">a\nMaKT", [1]), ">a\naK");
  assert.equal(cropA3m(">a\nMaKT", [0, 2]), ">a\nMT");
});

const SEQ = "MKTAYIAKQR";
const A3M = `>q\n${SEQ}\n>h1\nMKTAYIAKQR\n>h2\nMKaTAYIAKQ-\n>h3\nMKTAYIAKQR`;

test("featuriseOwn crops the receptor: tokens, original residue_index, atoms follow the kept residues", () => {
  const keep = [1, 2, 5, 8];
  const full = featuriseOwn({ sequence: SEQ, smiles: "O" });
  const crop = featuriseOwn({ sequence: SEQ, smiles: "O", receptorIndices: keep });
  assert.equal(crop.receptorTokens, 4);
  assert.equal(crop.tokens, 4 + crop.ligandTokens);
  assert.deepEqual(Array.from(crop.features.residue_index.data.slice(0, 4)), keep);
  assert.deepEqual(Array.from(crop.features.token_index.data), [0, 1, 2, 3, 4]);
  assert.deepEqual(Array.from(crop.compact.resType.slice(0, 4)), keep.map((i) => Array.from(full.compact.resType)[i]));
  const atomsOf = (r, t) => Array.from(r.compact.atomToToken).filter((x) => x === t).length;
  keep.forEach((original, t) => assert.equal(atomsOf(crop, t), atomsOf(full, original)));
  assert.ok(Array.from(crop.features.ref_pos.data).every(Number.isFinite));
});

test("the A3M is de-duplicated on the full chain, then cropped; deletions follow their column", () => {
  const keep = [1, 2, 3];
  const r = featuriseOwn({ sequence: SEQ, smiles: "O", a3m: A3M, receptorIndices: keep });
  // h1 and h3 repeat the query, so the full alignment has query + h2; cropping must not re-merge anything
  assert.equal(r.msaRows, 2);
  const T = r.tokens;
  const row1 = Array.from(r.compact.msa.slice(T, T + 3));
  assert.deepEqual(row1, ["K", "T", "A"].map((l) => TOKEN_ID[{ K: "LYS", T: "THR", A: "ALA" }[l]]));
  // h2 has one insertion before column 2 (T): has_deletion is zero for non-query rows in Boltz's featuriser
  assert.equal(r.features.has_deletion.data[T + 1], 0);
});

test("buildMsa crop equals the full alignment cut to the columns", () => {
  const types = Int32Array.from(SEQ, (l) => TOKEN_ID[{ M: "MET", K: "LYS", T: "THR", A: "ALA", Y: "TYR", I: "ILE", Q: "GLN", R: "ARG" }[l]]);
  const keep = [0, 3, 4, 9];
  const full = buildMsa(types, 0, A3M);
  const crop = buildMsa(Int32Array.from(keep, (i) => types[i]), 0, A3M, { keep, fullTypes: types });
  assert.equal(crop.rows, full.rows);
  for (let r = 0; r < full.rows; r += 1) keep.forEach((c, k) => assert.equal(crop.msa[r * 4 + k], full.msa[r * 10 + c]));
  assert.equal(parseA3m(A3M).length, 2);
});

test("receptorIndices validation and size limit on the cropped size", () => {
  assert.throws(() => featuriseOwn({ sequence: SEQ, smiles: "O", receptorIndices: [2, 1] }), /strictly increasing/);
  assert.throws(() => featuriseOwn({ sequence: SEQ, smiles: "O", receptorIndices: [0, 10] }), /outside/);
  assert.throws(() => featuriseOwn({ sequence: SEQ, smiles: "O", receptorIndices: [] }), /empty/);
  const long = "A".repeat(300);
  assert.throws(() => featuriseOwn({ sequence: long, smiles: "O" }), /receptor <= 200/);
  const ok = featuriseOwn({ sequence: long, smiles: "O", receptorIndices: Array.from({ length: 150 }, (_, i) => i * 2) });
  assert.equal(ok.receptorTokens, 150);
  assert.throws(() => featuriseOwn({ sequence: long, smiles: "O", receptorIndices: Array.from({ length: 250 }, (_, i) => i) }), /receptor <= 200/);
});
