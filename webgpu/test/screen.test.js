import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseYaml } from "../src/yaml-lite.js";
import { loadScreenConfig, parseDelimited, readLigands } from "../src/screen.js";

const dir = () => mkdtempSync(join(tmpdir(), "screen-"));

test("yaml-lite parses the GenFree example shape", () => {
  const cfg = parseYaml(`version: 1
target:
  sequence: MKT   # comment
  msa: auto
ligands:
  smiles:
    - CCO
    - "c1ccccc1"
options:
  seed: 20260718
  flag: true
  list: [1, 2]
`);
  assert.deepEqual(cfg, {
    version: 1, target: { sequence: "MKT", msa: "auto" }, ligands: { smiles: ["CCO", "c1ccccc1"] },
    options: { seed: 20260718, flag: true, list: [1, 2] },
  });
});

test("parseDelimited handles quotes and tabs", () => {
  assert.deepEqual(parseDelimited('id,smiles\na,"C,C"\n'), [["id", "smiles"], ["a", "C,C"]]);
  assert.deepEqual(parseDelimited("a\tb\n1\t2\n", "\t"), [["a", "b"], ["1", "2"]]);
});

test("readLigands: csv with custom columns, duplicate ids and missing columns are rejected", () => {
  const d = dir();
  writeFileSync(join(d, "l.csv"), "name,smi\nx,CCO\ny,CCN\n");
  assert.deepEqual(readLigands({ csv: "l.csv", id_column: "name", smiles_column: "smi" }, d).map((l) => [l.id, l.value]), [["x", "CCO"], ["y", "CCN"]]);
  assert.throws(() => readLigands({ csv: "l.csv" }, d), /columns/);
  writeFileSync(join(d, "dup.csv"), "id,smiles\na,CC\na,CCC\n");
  assert.throws(() => readLigands({ csv: "dup.csv" }, d), /duplicate id/);
  assert.throws(() => readLigands({ csv: "l.csv", smiles: ["C"] }, d), /exactly one/);
});

test("loadScreenConfig: options, msa and unsupported features", () => {
  const d = dir();
  writeFileSync(join(d, "l.csv"), "id,smiles\na,CCO\n");
  writeFileSync(join(d, "t.a3m"), ">q\nMKT\n");
  const write = (body) => { writeFileSync(join(d, "c.yaml"), body); return join(d, "c.yaml"); };
  const base = "target:\n  sequence: MKT\n  msa: ./t.a3m\nligands:\n  csv: ./l.csv\n";
  const cfg = loadScreenConfig(write(`${base}options:\n  seed: 7\n  affinity_recycling_steps: 1\n  ligand_reference_conformer: etkdg\n  num_workers: 4\n`));
  assert.equal(cfg.seed, 7); assert.equal(cfg.recycles, 1); assert.equal(cfg.conformer, "etkdg");
  assert.match(cfg.a3m, /^>q/);
  assert.throws(() => loadScreenConfig(write(`${base}options:\n  bogus: 1\n`)), /options.bogus/);
  assert.throws(() => loadScreenConfig(write("target:\n  sequence: MKT\n  crop:\n    reference: x.cif\nligands:\n  smiles: [CC]\n")), /crop/);
  const auto = loadScreenConfig(write("target:\n  sequence: MKT\n  msa: auto\nligands:\n  smiles: [CC]\n"));
  assert.equal(auto.a3m, null); assert.match(auto.notes[0], /MSA server/);
  const omitted = loadScreenConfig(write("target:\n  sequence: MKT\nligands:\n  smiles: [CC]\n"));
  assert.match(omitted.notes[0], /MSA server/);
  const empty = loadScreenConfig(write("target:\n  sequence: MKT\n  msa: empty\nligands:\n  smiles: [CC]\n"));
  assert.equal(empty.a3m, null); assert.deepEqual(empty.notes, []);
});

import { SCORE_COLUMNS, parseLigandText, scoreRow, scoresCsv } from "../src/screen-core.js";

test("parseLigandText: header with smiles column, id,smiles pairs, bare SMILES, tabs", () => {
  assert.deepEqual(parseLigandText("id,smiles\na,CCO\nb,CCN\n").map((l) => [l.id, l.value]), [["a", "CCO"], ["b", "CCN"]]);
  assert.deepEqual(parseLigandText("smiles,name\nCCO,x\n").map((l) => [l.id, l.value]), [["x", "CCO"]]);
  assert.deepEqual(parseLigandText("SMILES\nCCO\n").map((l) => [l.id, l.value]), [["ligand_1", "CCO"]]);
  assert.deepEqual(parseLigandText("a,CCO\nb,CCN").map((l) => [l.id, l.value]), [["a", "CCO"], ["b", "CCN"]]);
  assert.deepEqual(parseLigandText("CCO\nCCN\n").map((l) => [l.id, l.value]), [["ligand_1", "CCO"], ["ligand_2", "CCN"]]);
  assert.deepEqual(parseLigandText("id\tsmiles\na\tCCO\n").map((l) => [l.id, l.value]), [["a", "CCO"]]);
  assert.deepEqual(parseLigandText(""), []);
  assert.throws(() => parseLigandText("a,CCO\na,CCN"), /duplicate/);
  assert.throws(() => parseLigandText("id,smiles\na,\n"), /empty SMILES/);
});

test("scoreRow and scoresCsv use GenFree's columns and record_id/chunk format", () => {
  const row = scoreRow({ id: "phenol", kind: "smiles", value: "Oc1ccccc1" }, 1, { ligandTokens: 7, receptorTokens: 117, value: 1.5, probability: 0.25 }, 1000);
  assert.equal(row.record_id, "000002_phenol");
  assert.equal(row.chunk, "chunk_00000");
  const lines = scoresCsv([row], ["error"]).trim().split("\n");
  assert.equal(lines[0], [...SCORE_COLUMNS, "error"].join(","));
  assert.equal(lines[1], "000002_phenol,phenol,smiles,Oc1ccccc1,7,chunk_00000,complete_input,117,1.5,0.25,");
});
