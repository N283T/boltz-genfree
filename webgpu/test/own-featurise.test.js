// The pure-JS Boltz-2 featuriser (src/trunk/own) against the PyTorch oracle dumps, feature by feature, plus unit tests of
// its MSA rules, atom windows and ref_pos handling.
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { featuriseOwn, featureBundle } from "../src/trunk/own/featurise.js";
import { PROTEIN_CONFORMERS } from "../src/trunk/own/conformers.js";
import { buildMsa, crc32, parseA3m } from "../src/trunk/own/msa.js";
import { atomWindows, KEYS, QUERIES } from "../src/trunk/own/windows.js";

const root = new URL("../../", import.meta.url).pathname;
const dumps = ["phenol-nomsa", "acetaminophen-nomsa", "salicylic-nomsa", "big200-nomsa", "phenol-msa"];
const dumpPath = (name) => new URL(`../oracle-dumps/${name}-chain.json`, import.meta.url).pathname;

/** Every feature the trunk and the affinity module consume; each must equal the oracle exactly (ref_pos: see below). */
const EXACT = [
  "token_index", "residue_index", "asym_id", "entity_id", "sym_id", "mol_type", "res_type", "token_pad_mask",
  "affinity_token_mask", "token_bonds", "type_bonds", "profile", "deletion_mean", "profile_affinity",
  "deletion_mean_affinity", "contact_conditioning", "contact_threshold", "method_feature", "modified", "cyclic_period",
  "atom_to_token", "ref_element", "ref_charge", "ref_atom_name_chars", "ref_space_uid", "ref_chirality",
  "atom_pad_mask", "atom_backbone_feat", "msa", "msa_paired", "msa_mask", "has_deletion", "deletion_value",
];

const THREE = { A: "ALA", R: "ARG", N: "ASN", D: "ASP", C: "CYS", Q: "GLN", E: "GLU", G: "GLY", H: "HIS", I: "ILE", L: "LEU", K: "LYS", M: "MET", F: "PHE", P: "PRO", S: "SER", T: "THR", W: "TRP", Y: "TYR", V: "VAL" };

function inputOf(oracle, extra = {}) {
  const { sequence, smiles, msa } = oracle.meta;
  const a3m = msa ? readFileSync(`${root}${msa}`, "utf8") : null;
  return { sequence, smiles, a3m, allowExtended: true, ...extra };
}

const distance = (p, i, j) => Math.hypot(p[i * 3] - p[j * 3], p[i * 3 + 1] - p[j * 3 + 1], p[i * 3 + 2] - p[j * 3 + 2]);

/** Atom indices of every reference space (residue) of a feature set. */
function groups(uid, mask) {
  const map = new Map();
  for (let a = 0; a < mask.length; a += 1) {
    if (mask[a] === 0) continue;
    if (!map.has(uid[a])) map.set(uid[a], []);
    map.get(uid[a]).push(a);
  }
  return map;
}

/** Largest difference between the intra-residue distances of two ref_pos tensors over the protein residues. */
function maxIntraResidueDelta(a, b, uid, mask, residues) {
  let worst = 0;
  for (const [u, atoms] of groups(uid, mask)) {
    if (u >= residues) continue;
    for (const i of atoms) for (const j of atoms) worst = Math.max(worst, Math.abs(distance(a, i, j) - distance(b, i, j)));
  }
  return worst;
}

for (const name of dumps) {
  test(`own featuriser matches the oracle exactly: ${name}`, { skip: !existsSync(dumpPath(name)) && "no oracle dump" }, async () => {
    const oracle = await TensorBundle.open(dumpPath(name));
    const result = featuriseOwn(inputOf(oracle));
    const own = featureBundle(result);
    assert.equal(result.tokens, oracle.meta.tokens);
    assert.equal(result.msaRows, oracle.meta.msaRows);
    for (const feature of EXACT) {
      const key = `feat/${feature}`;
      assert.deepEqual(own.shape(feature), oracle.shape(key), `${feature}: shape`);
      const a = own.get(feature);
      const b = oracle.get(key);
      let bad = -1;
      for (let i = 0; i < b.length; i += 1) if (a[i] !== b[i]) { bad = i; break; }
      assert.equal(bad, -1, `${feature}: first difference at ${bad}: own ${a[bad]} oracle ${b[bad]}`);
    }
    // ref_mask is not in the oracle (the trunk does not read it): the real-atom mask
    assert.deepEqual(Array.from(own.get("ref_mask")), Array.from(oracle.get("feat/atom_pad_mask")));
  });

  test(`own protein ref_pos is one of Boltz's stored conformers up to a rigid motion; ligand ref_pos is zero: ${name}`,
    { skip: !existsSync(dumpPath(name)) && "no oracle dump" }, async () => {
      const oracle = await TensorBundle.open(dumpPath(name));
      const n = oracle.meta.sequence.length;
      const uid = oracle.get("feat/ref_space_uid");
      const mask = oracle.get("feat/atom_pad_mask");
      const refPos = oracle.get("feat/ref_pos");
      const byUid = groups(uid, mask);

      // which stored conformer did Boltz draw for each residue? (the one whose intra-residue distances match)
      const picks = [];
      for (let t = 0; t < n; t += 1) {
        const atoms = byUid.get(t);
        const table = PROTEIN_CONFORMERS[THREE[oracle.meta.sequence[t]] ?? "UNK"];
        assert.equal(atoms.length, table.atoms.length, `residue ${t}: atom count`);
        let best = -1;
        let bestError = Infinity;
        table.conformers.forEach((flat, index) => {
          let error = 0;
          for (let i = 0; i < atoms.length; i += 1) for (let j = 0; j < atoms.length; j += 1) {
            const d = Math.hypot(flat[i * 3] - flat[j * 3], flat[i * 3 + 1] - flat[j * 3 + 1], flat[i * 3 + 2] - flat[j * 3 + 2]);
            error = Math.max(error, Math.abs(d - distance(refPos, atoms[i], atoms[j])));
          }
          if (error < bestError) { bestError = error; best = index; }
        });
        assert.ok(bestError < 1e-4, `residue ${t} (${oracle.meta.sequence[t]}): no stored conformer matches, best ${bestError}`);
        picks.push(best);
      }

      // pin those picks: every intra-residue distance then equals the oracle's (float32 round-off of 6-decimal tables)
      const result = featuriseOwn(inputOf(oracle, { conformers: picks, refPosAugment: "none" }));
      const own = result.features.ref_pos.data;
      assert.ok(maxIntraResidueDelta(own, refPos, uid, mask, n) < 1e-4);
      // each residue is centred
      for (let t = 0; t < n; t += 1) {
        const atoms = byUid.get(t);
        for (let k = 0; k < 3; k += 1) assert.ok(Math.abs(atoms.reduce((s, a) => s + own[a * 3 + k], 0) / atoms.length) < 1e-5);
      }
      // ligand and padding rows are zero, here and in the oracle (ligand_reference_conformer: skip)
      for (let a = 0; a < mask.length; a += 1) {
        if (a < result.atoms && uid[a] < n) continue;
        for (let k = 0; k < 3; k += 1) {
          assert.equal(own[a * 3 + k], 0);
          assert.equal(refPos[a * 3 + k], 0);
        }
      }
    });
}

test("seeded ref_pos: deterministic, seed-dependent, a rigid motion per residue of the centred conformer", () => {
  const base = { sequence: "MKTVRQERLKX", smiles: "Oc1ccccc1" };
  const a = featuriseOwn({ ...base, seed: 3 });
  const b = featuriseOwn({ ...base, seed: 3 });
  const c = featuriseOwn({ ...base, seed: 4 });
  assert.deepEqual(a.features.ref_pos.data, b.features.ref_pos.data);
  assert.notDeepEqual(a.features.ref_pos.data, c.features.ref_pos.data);
  const pinned = featuriseOwn({ ...base, seed: 3, conformers: Array.from(a.conformerIndex), refPosAugment: "none" });
  const uid = a.features.ref_space_uid.data;
  const mask = a.features.atom_pad_mask.data;
  assert.ok(maxIntraResidueDelta(a.features.ref_pos.data, pinned.features.ref_pos.data, uid, mask, base.sequence.length) < 1e-5);
  assert.ok(a.conformerIndex.every((i) => i >= 0 && i < 10));
});

test("unknown residues are UNK with the five atoms N CA C O CB", () => {
  const r = featuriseOwn({ sequence: "AXB", smiles: "CO" });
  assert.deepEqual(Array.from(r.compact.resType.subarray(0, 3)), [2, 22, 22]);
  assert.equal(r.atoms, 5 + 5 + 5 + 2);
  const a2t = Array.from(r.compact.atomToToken);
  assert.deepEqual(a2t.slice(5, 10), [1, 1, 1, 1, 1]);
});

test("size limits follow GenFree unless allowExtended", () => {
  assert.throws(() => featuriseOwn({ sequence: "A".repeat(201), smiles: "CO" }), /allow-extended/);
  assert.ok(featuriseOwn({ sequence: "A".repeat(201), smiles: "CO", allowExtended: true }).receptorTokens === 201);
  assert.throws(() => featuriseOwn({ sequence: "AAA", smiles: "CO", ligandConformer: "etkdg" }), /not supported/);
});

test("ligandAtoms override sets the token order and the names", () => {
  const r = featuriseOwn({ sequence: "A", smiles: "ignored", ligandAtoms: { smiles: "CO", atomNames: ["C9", "O7"] } });
  assert.equal(r.ligandTokens, 2);
  const codes = r.compact.atomNameCodes;
  const first = 5; // after ALA's five atoms
  assert.deepEqual(Array.from(codes.subarray(first * 4, first * 4 + 4)), [35, 25, 0, 0]);
  assert.throws(() => featuriseOwn({ sequence: "A", smiles: "x", ligandAtoms: { smiles: "CO", atomNames: ["C1"] } }), /atom names/);
});

// ---- MSA rules ----

test("crc32 matches zlib", () => {
  assert.equal(crc32("123456789"), 0xcbf43926);
  assert.equal(crc32(""), 0);
});

test("A3M parsing: duplicates dropped, insertions counted, comments skipped", () => {
  const rows = parseA3m("#c\n>q\nMKT\n>a\nM-T\n>dup\nmkt\n>b\nMaaKT\n");
  assert.equal(rows.length, 3); // "mkt" repeats the query once upper-cased
  assert.deepEqual(Array.from(rows[2].residues), [14, 13, 18]); // MET LYS THR
  assert.deepEqual([...rows[2].deletions], [[1, 2]]);
  assert.throws(() => parseA3m(">q\nM*T\n"), /alphabet/);
});

test("deletions: only the query row keeps its counts (Boltz's deletion-array slicing)", () => {
  const msa = buildMsa([14, 13, 18], 1, ">q\nMaKT\n>a\nMbbKT\n", {});
  assert.equal(msa.rows, 2);
  assert.deepEqual(Array.from(msa.deletion), [0, 1, 0, 0, 0, 0, 0, 0]);
  const value = Math.fround((Math.PI / 2) * Math.atan(1 / 3));
  assert.ok(Math.abs(msa.deletionMean[1] - value / 2) < 1e-7);
  assert.ok(Math.abs(msa.deletionMeanAffinity[1] - value) < 1e-7);
});

test("MSA layout: ligand columns are UNK in the query row and gap elsewhere; profile over all rows", () => {
  const msa = buildMsa([14, 13], 2, ">q\nMK\n>a\nM-\n>b\nMF\n", {});
  assert.deepEqual(Array.from(msa.msa), [14, 13, 22, 22, 14, 1, 1, 1, 14, 15, 1, 1]);
  assert.deepEqual(Array.from(msa.paired), [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]);
  assert.equal(msa.profile[0 * 33 + 14], 1);
  assert.ok(Math.abs(msa.profile[1 * 33 + 1] - 1 / 3) < 1e-7);
  assert.ok(Math.abs(msa.profile[2 * 33 + 22] - 1 / 3) < 1e-7);
  assert.equal(msa.profileAffinity[2 * 33 + 22], 1);
});

test("MSA is capped at 8192 rows, not subsampled to 1024; maxRows lowers the cap", () => {
  const letters = "ACDEFGHIKLMNPQRSTVWY";
  const lines = [">q", "AAAAAAAA"];
  for (let i = 1; i < 9000; i += 1) {
    lines.push(`>s${i}`, [0, 1, 2, 3].map((k) => letters[Math.floor(i / 20 ** k) % 20]).join("") + "AAAA");
  }
  const text = lines.join("\n");
  const query = new Array(8).fill(2);
  assert.equal(buildMsa(query, 1, text).rows, 8192);
  assert.equal(buildMsa(query, 1, text, { maxRows: 100 }).rows, 100);
});

test("a query that does not match the input falls back to the query alone; UNK in the input is accepted", () => {
  const bad = buildMsa([14, 13, 18], 1, ">q\nMKS\n>a\nMKT\n");
  assert.equal(bad.rows, 1);
  assert.match(bad.notes[0], /does not match/);
  // input X at column 1 over an alignment query K: Boltz overwrites the query row with the input types
  const unk = buildMsa([14, 22, 18], 1, ">q\nMKT\n>a\nMKA\n");
  assert.equal(unk.rows, 2);
  assert.deepEqual(Array.from(unk.msa.subarray(0, 3)), [14, 22, 18]);
  // too long a query is not trimmed here
  assert.match(buildMsa([14, 13], 0, ">q\nMKT\n").notes[0], /trim/);
});

test("rows sharing a UniRef id are the paired block, ahead of the unpaired rows", () => {
  const text = ">q\nMKT\n>UniRef100_X\nMKS\n>UniRef100_Y\nMRT\n>UniRef100_X\nMQT\n";
  const msa = buildMsa([14, 13, 18], 0, text);
  // order: query, the two UniRef100_X rows (paired), then UniRef100_Y
  assert.equal(msa.rows, 4);
  assert.deepEqual(Array.from(msa.msa.subarray(3)), [14, 13, 17, 14, 7, 18, 14, 3, 18]);
  assert.deepEqual(Array.from(msa.paired), [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]);
});

// ---- atom windows ----

test("atom windows: 32 queries per block, a centred 128-key window, padding masked", () => {
  const atoms = 100;
  const toToken = Int32Array.from({ length: atoms }, (_, a) => Math.floor(a / 5));
  const w = atomWindows(toToken);
  assert.equal(w.subsets, 4);
  assert.equal(w.tokensToQueries.indices.length, 4 * QUERIES);
  assert.equal(w.queriesToKeys.indices.length, 4 * KEYS);
  // block 0: the window starts at atom -48, so atoms 0..79 sit in key slots 48..127 and slots 0..47 are padding
  assert.deepEqual([w.queriesToKeys.mask[47], w.queriesToKeys.mask[48], w.queriesToKeys.mask[127]], [0, 1, 1]);
  assert.equal(w.queriesToKeys.indices[48], 0);
  assert.equal(w.queriesToKeys.indices[127], 79);
  assert.equal(w.tokensToKeys.indices[127], toToken[79]);
  // block 1 starts at atom 32 - 48 = -16
  assert.equal(w.queriesToKeys.indices[KEYS + 16], 0);
  assert.equal(w.queriesToKeys.mask[KEYS + 15], 0);
  // last block: 4 real queries (96..99)
  const lastQueries = Array.from(w.tokensToQueries.mask.subarray(3 * QUERIES, 4 * QUERIES));
  assert.equal(lastQueries.filter((v) => v === 1).length, 4);
  assert.equal(w.tokensToQueries.indices[3 * QUERIES + 3], toToken[99]);
  // block 3 starts at atom 48: slots 0..51 hold atoms 48..99, the rest is padding
  assert.deepEqual([w.queriesToKeys.mask[3 * KEYS + 51], w.queriesToKeys.mask[3 * KEYS + 52]], [1, 0]);
  assert.equal(w.queriesToKeys.indices[3 * KEYS + 51], 99);
});

test("featuriseOwn carries window tables for its atoms", () => {
  const r = featuriseOwn({ sequence: "MKTV", smiles: "CO" });
  assert.equal(r.windows.subsets, Math.ceil(r.atoms / 32));
  assert.equal(r.windows.tokensToQueries.mask.reduce((s, v) => s + v, 0), r.atoms);
});
