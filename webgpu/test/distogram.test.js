import { test } from "node:test";
import assert from "node:assert/strict";
import { affinityMasks, buildStructureFreeBins, STRUCTURE_FREE_MODES } from "../src/distogram.js";

const boundaries = Array.from({ length: 63 }, (_, i) => 2 + (20 * i) / 62);
const pad = [1, 1, 1, 1, 1, 0];
const molType = [0, 0, 0, 3, 3, 0];
const aff = [0, 0, 0, 1, 1, 0];

test("modes: structure-free set is exactly the five Python modes", () => {
  assert.deepEqual([...STRUCTURE_FREE_MODES].sort(), ["normal_random", "uniform_mean_embedding", "uniform_random", "zero_conditioning", "zero_distogram_features"]);
});

test("coordinate-based modes are rejected", () => {
  const { receptor, ligand } = affinityMasks(pad, molType, aff);
  for (const mode of ["control", "constant_far", "collapsed_coordinates"]) {
    assert.throws(() => buildStructureFreeBins({ boundaries, receptor, ligand, mode }), /not coordinate-free/);
  }
});

test("random bins are symmetric, in range, zero on the diagonal and on padding", () => {
  const { receptor, ligand } = affinityMasks(pad, molType, aff);
  for (const mode of ["uniform_random", "normal_random"]) {
    const { bins } = buildStructureFreeBins({ boundaries, receptor, ligand, mode, seed: 3 });
    const n = pad.length;
    for (let i = 0; i < n; i++) {
      assert.equal(bins[i * n + i], 0);
      assert.equal(bins[i * n + 5], 0, "padding column");
      for (let j = 0; j < n; j++) {
        assert.equal(bins[i * n + j], bins[j * n + i]);
        assert.ok(bins[i * n + j] <= 63);
      }
    }
  }
});

test("pool mask excludes the diagonal, pairformer mask keeps ligand-ligand diagonal", () => {
  const { crossPairMask, poolMask } = affinityMasks(pad, molType, aff);
  assert.equal(crossPairMask[3 * 6 + 3], 1);
  assert.equal(poolMask[3 * 6 + 3], 0);
  assert.equal(crossPairMask[0 * 6 + 1], 0, "receptor-receptor pairs are masked out");
});
