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
