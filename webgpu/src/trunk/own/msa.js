// Boltz-2's MSA path for one protein chain plus one ligand chain, with the alignment given as A3M text:
//   parse_a3m (src/boltz/data/parse/a3m.py)  ->  construct_paired_msa / prepare_msa_arrays
//   (src/boltz/data/feature/featurizerv2.py) -> process_msa_features.
//
// Behaviour kept from Boltz (each checked against the oracle dumps, see test/own-featurise.test.js):
//   * comment lines skipped, a row repeating an earlier row (gaps removed, upper-cased, insertions kept) dropped,
//     at most `maxSeqs` rows kept, the query being the first row kept;
//   * the alignment covers the protein chain only: ligand columns hold the ligand token (UNK) in the query row
//     and the gap in every other row, and the ligand chain has no taxonomy;
//   * the paired block is absent unless two rows of the chain share a taxonomy key (UniRef header, hashed as in
//     Boltz); one chain means such rows only come from repeated UniRef ids;
//   * deletions: Boltz's `construct_paired_msa` re-slices its deletion array inside the row loop, so only the
//     query row keeps its deletion counts, every other row has none (`zeroDeletions` in LocalFold's boltz2
//     dialect). `has_deletion`, `deletion_value` and `deletion_mean` follow;
//   * the profile is the mean one-hot over every row kept, ligand columns included.
//
// Not implemented: trimming an alignment whose query is longer than the input (Boltz's _trim_msa_to_input); such an
// alignment falls back to the query alone with a note (GenFree crops the A3M itself before featurising).

import { NUM_TOKENS, TOKEN_GAP, TOKEN_MET, TOKEN_UNK, letterToTokenId } from "./boltz-const.js";

const CRC_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n += 1) {
    let c = n;
    for (let k = 0; k < 8; k += 1) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    table[n] = c >>> 0;
  }
  return table;
})();

/** zlib.crc32 of an ASCII/UTF-8 string. */
export function crc32(text) {
  let crc = 0xffffffff;
  for (const byte of new TextEncoder().encode(text)) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
  return (crc ^ 0xffffffff) >>> 0;
}

/**
 * Boltz's `_parse_a3m` without a taxonomy database.
 * @returns {{residues: Int32Array, deletions: Map<number, number>, taxonomy: number}[]}
 */
export function parseA3m(text, maxSeqs = 8192) {
  const rows = [];
  const seen = new Set();
  let taxonomy = -1;
  for (const raw of text.split("\n")) {
    const line = raw.trim().replaceAll("\0", "");
    if (line === "" || line.startsWith("#")) continue;
    if (line.startsWith(">")) {
      const header = line.split(/\s+/)[0];
      taxonomy = header.startsWith(">UniRef") ? (crc32(header.slice(1)) % (2 ** 31 - 2)) + 1 : -1;
      continue;
    }
    const key = line.replaceAll("-", "").toUpperCase();
    if (seen.has(key)) continue;
    seen.add(key);
    const residues = [];
    const deletions = new Map();
    let count = 0;
    for (const letter of line) {
      if (letter !== "-" && letter >= "a" && letter <= "z") { count += 1; continue; }
      const token = letterToTokenId(letter);
      if (token === undefined) throw new Error(`A3M residue ${JSON.stringify(letter)} is not in Boltz's protein alphabet`);
      if (count > 0) { deletions.set(residues.length, count); count = 0; }
      residues.push(token);
    }
    rows.push({ residues: Int32Array.from(residues), deletions, taxonomy });
    if (rows.length >= maxSeqs) break;
  }
  return rows;
}

/**
 * Check the first row against the input as `construct_paired_msa` does. Returns the rows to use, or null when
 * Boltz would replace the chain's MSA by a dummy (query only).
 */
function matchQuery(rows, inputTypes, notes) {
  if (rows.length === 0) { notes.push("A3M has no sequences: query only"); return null; }
  const first = rows[0].residues;
  if (first.length !== inputTypes.length) {
    notes.push(`A3M query has ${first.length} columns for ${inputTypes.length} residues: query only (Boltz would ${
      first.length > inputTypes.length ? "trim the alignment, which is not implemented here" : "also drop the MSA"})`);
    return null;
  }
  const mismatch = [];
  for (let i = 0; i < first.length; i += 1) if (first[i] !== inputTypes[i]) mismatch.push(i);
  if (mismatch.length > 0) {
    const allMetOverUnk = mismatch.every((i) => inputTypes[i] === TOKEN_MET && first[i] === TOKEN_UNK);
    const allUnk = mismatch.every((i) => inputTypes[i] === TOKEN_UNK);
    if (!(allMetOverUnk || allUnk)) {
      notes.push("A3M query does not match the input sequence: query only (as Boltz's dummy MSA)");
      return null;
    }
    const patched = { ...rows[0], residues: Int32Array.from(inputTypes) };
    return [patched, ...rows.slice(1)];
  }
  return rows;
}

/** Row order of Boltz's pairing for one chain: query, rows sharing a taxonomy ("paired"), then the rest in order. */
function pairingOrder(rows, maxPairs = 8192, maxTotal = 16384) {
  const groups = new Map();
  rows.forEach((row, index) => {
    if (row.taxonomy === -1) return;
    if (!groups.has(row.taxonomy)) groups.set(row.taxonomy, []);
    groups.get(row.taxonomy).push(index);
  });
  const pairedGroups = [...groups.values()].filter((members) => members.length > 1);
  const visited = new Set(pairedGroups.flat());
  const available = [];
  for (let i = 1; i < rows.length; i += 1) if (!visited.has(i)) available.push(i);
  const order = [{ seq: 0, paired: 1 }];
  outer: for (const members of pairedGroups) {
    for (const seq of members) {
      order.push({ seq, paired: 1 });
      if (order.length >= maxPairs) break outer;
    }
  }
  let next = 0;
  while (order.length < maxTotal && next < available.length) order.push({ seq: available[next++], paired: 0 });
  return order;
}

/**
 * The MSA features of one protein chain (`inputTypes`, token ids) followed by `ligandTokens` ligand tokens.
 *
 * @param {Int32Array|number[]} inputTypes protein token ids
 * @param {number} ligandTokens
 * @param {string|null} a3m
 * @param {{maxRows?: number, maxSeqs?: number}} [options] `maxRows` caps the rows of the result (default 8192)
 * @returns {{rows: number, tokens: number, msa: Int32Array, paired: Uint8Array, deletion: Int32Array,
 *            queryDeletion: Int32Array, profile: Float32Array, deletionMean: Float32Array,
 *            profileAffinity: Float32Array, deletionMeanAffinity: Float32Array, notes: string[]}}
 */
export function buildMsa(inputTypes, ligandTokens, a3m, { maxRows = 8192, maxSeqs = 8192 } = {}) {
  const notes = [];
  const n = inputTypes.length;
  const tokens = n + ligandTokens;
  let rows = null;
  if (a3m) rows = matchQuery(parseA3m(a3m, maxSeqs), inputTypes, notes);
  if (rows === null) rows = [{ residues: Int32Array.from(inputTypes), deletions: new Map(), taxonomy: -1 }];
  const order = pairingOrder(rows).slice(0, maxRows);
  const R = order.length;

  const msa = new Int32Array(R * tokens).fill(TOKEN_GAP);
  const paired = new Uint8Array(R * tokens);
  const deletion = new Int32Array(R * tokens);
  const queryDeletion = new Int32Array(tokens);
  for (const [position, count] of rows[0].deletions) queryDeletion[position] = count;
  order.forEach(({ seq, paired: flag }, r) => {
    const residues = rows[seq].residues;
    for (let t = 0; t < n; t += 1) {
      msa[r * tokens + t] = residues[t];
      paired[r * tokens + t] = flag;
      if (seq === 0) deletion[r * tokens + t] = queryDeletion[t];
    }
    if (r === 0) {
      for (let t = n; t < tokens; t += 1) { msa[t] = TOKEN_UNK; paired[t] = 1; }
    }
  });

  // process_msa_features: profile = mean one-hot, deletion = pi / 2 * atan(count / 3) (float32), per-column mean
  const profile = new Float32Array(tokens * NUM_TOKENS);
  const deletionMean = new Float32Array(tokens);
  for (let t = 0; t < tokens; t += 1) {
    const counts = new Int32Array(NUM_TOKENS);
    let sum = 0;
    for (let r = 0; r < R; r += 1) {
      counts[msa[r * tokens + t]] += 1;
      sum += Math.fround(Math.fround(Math.PI / 2) * Math.fround(Math.atan(Math.fround(deletion[r * tokens + t] / 3))));
    }
    for (let c = 0; c < NUM_TOKENS; c += 1) profile[t * NUM_TOKENS + c] = counts[c] / R;
    deletionMean[t] = sum / R;
  }
  // affinity=True with max_seqs_batch=1: the query row alone
  const profileAffinity = new Float32Array(tokens * NUM_TOKENS);
  const deletionMeanAffinity = new Float32Array(tokens);
  for (let t = 0; t < tokens; t += 1) {
    profileAffinity[t * NUM_TOKENS + msa[t]] = 1;
    deletionMeanAffinity[t] = Math.fround(Math.fround(Math.PI / 2) * Math.fround(Math.atan(Math.fround(queryDeletion[t] / 3))));
  }
  return { rows: R, tokens, msa, paired, deletion, queryDeletion, profile, deletionMean, profileAffinity, deletionMeanAffinity, notes };
}
