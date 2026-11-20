import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync } from "node:fs";
import { TensorBundle } from "../src/bundle.js";
import { compare } from "../src/compare.js";
import { inputFromBundle } from "../src/trunk-input.js";

const dumpPath = new URL("../oracle-dumps/small-dump.json", import.meta.url).pathname;
test("masking the raw trunk z reproduces the oracle's z_affinity", { skip: !existsSync(dumpPath) && "no oracle dump" }, async () => {
  const dump = await TensorBundle.open(dumpPath);
  const input = inputFromBundle(dump);
  assert.equal(compare(input.zAffinity, dump.get("input/z_affinity")).maxAbs, 0);
});
