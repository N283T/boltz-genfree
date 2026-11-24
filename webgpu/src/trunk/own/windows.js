// The atom attention windows of Boltz-2's atom encoder/decoder (AtomAttentionEncoder, W = 32 queries, H = 128 keys),
// derived from `atom_to_token` instead of being stored in the batch.
//
// Atoms are the compact list in token order (our featuriser's order; padding atoms follow). Query block b holds atoms
// 32b .. 32b+31; its key window is atoms 32b - 48 .. 32b + 79 (centred, H = 128 slots). Slots outside 0..atoms-1 are
// padding (index 0, mask 0), as in Boltz and in LocalFold's boltz2 dialect (`paddedAtomKeys`), derived from
// shared/af3/featurise/featurise.js atomGathers (tokensToQueries / tokensToKeys / queriesToKeys).
//
//   tokensToQueries[b*32 + i]  token of query atom 32b+i            (mask: atom is real)
//   queriesToKeys[b*128 + k]   atom index of key slot k of block b  (mask: in range)
//   tokensToKeys[b*128 + k]    token of that key atom
// The pair-bias gathers of the encoder (ref_space_uid equality over query x key) follow from queriesToKeys.

export const QUERIES = 32;
export const KEYS = 128;

/**
 * @param {Int32Array} atomToToken token index per real atom (length = real atom count)
 * @returns {{subsets: number, queries: number, keys: number,
 *            tokensToQueries: {indices: Int32Array, mask: Float32Array},
 *            queriesToKeys: {indices: Int32Array, mask: Float32Array},
 *            tokensToKeys: {indices: Int32Array, mask: Float32Array}}}
 */
export function atomWindows(atomToToken) {
  const atoms = atomToToken.length;
  const subsets = Math.max(1, Math.ceil(atoms / QUERIES));
  const make = (count) => ({ indices: new Int32Array(count), mask: new Float32Array(count) });
  const tokensToQueries = make(subsets * QUERIES);
  const queriesToKeys = make(subsets * KEYS);
  const tokensToKeys = make(subsets * KEYS);
  for (let b = 0; b < subsets; b += 1) {
    for (let i = 0; i < QUERIES; i += 1) {
      const atom = b * QUERIES + i;
      if (atom >= atoms) continue;
      tokensToQueries.indices[b * QUERIES + i] = atomToToken[atom];
      tokensToQueries.mask[b * QUERIES + i] = 1;
    }
    const start = b * QUERIES - (KEYS - QUERIES) / 2;
    for (let k = 0; k < KEYS; k += 1) {
      const atom = start + k;
      if (atom < 0 || atom >= atoms) continue;
      queriesToKeys.indices[b * KEYS + k] = atom;
      queriesToKeys.mask[b * KEYS + k] = 1;
      tokensToKeys.indices[b * KEYS + k] = atomToToken[atom];
      tokensToKeys.mask[b * KEYS + k] = 1;
    }
  }
  return { subsets, queries: QUERIES, keys: KEYS, tokensToQueries, queriesToKeys, tokensToKeys };
}
