// Ligand atom names from the RDKit port in src/trunk/canonical.js. No GPU needed.
import { test } from "node:test";
import assert from "node:assert/strict";
import { ligandAtomsFromSmiles } from "../src/trunk/canonical.js";

test("ligand atom names follow RDKit canonical ranks (phenol)", () => {
  const r = ligandAtomsFromSmiles("Oc1ccccc1");
  assert.deepEqual(r.atomNames, ["O1", "C6", "C4", "C2", "C1", "C3", "C5"]);
  assert.ok(r.exact);
});
