// Compare the pure-JS ligand atom canonicaliser against RDKit (PyTorch-side) over a corpus.
//
//   .venv/bin/python webgpu/tools/ligand_atoms_corpus.py --out corpus.json --ccd 4000
//   node webgpu/tools/check_canonical.mjs corpus.json [--show 5] [--bucket plain]
//
// Prints agreement per bucket (and for molecules flagged exact:true) and failure categories
// with examples. A molecule "agrees" when the canonical SMILES string and every atom name match;
// "names-only" ignores the SMILES string (useful for stereo molecules, where the string carries
// @ and / \\ marks this port does not write but the atom order and names can still agree).

import { readFileSync } from "node:fs";
import { ligandAtomsFromSmiles } from "../src/trunk/canonical.js";

const args = process.argv.slice(2);
const file = args.find((a) => !a.startsWith("--"));
if (!file) {
  console.error("usage: check_canonical.mjs corpus.json [--show N] [--bucket NAME]");
  process.exit(2);
}
const option = (name, fallback) => {
  const k = args.indexOf(`--${name}`);
  return k >= 0 ? args[k + 1] : fallback;
};
const show = Number(option("show", 3));
const onlyBucket = option("bucket", null);

const corpus = JSON.parse(readFileSync(file, "utf8"));
const stats = new Map();
const failures = new Map();
const bump = (bucket, key) => {
  if (!stats.has(bucket)) stats.set(bucket, { total: 0, agree: 0, bothError: 0, jsError: 0, pyError: 0, smilesDiff: 0, namesDiff: 0, exactTrue: 0, exactTrueAgree: 0, timeMs: 0 });
  stats.get(bucket)[key]++;
  return stats.get(bucket);
};

for (const item of corpus) {
  if (onlyBucket && item.bucket !== onlyBucket) continue;
  const buckets = [item.bucket, "ALL"];
  if (item.bucket === "plain") buckets.push(item.changed ? "plain (standardiser changed it)" : "plain (unchanged)");
  let got = null;
  let jsError = null;
  const t0 = performance.now();
  try {
    got = ligandAtomsFromSmiles(item.smiles);
  } catch (error) {
    jsError = error.message;
  }
  const dt = performance.now() - t0;
  const pyError = item.expected.error ?? null;
  let outcome;
  let detail = "";
  if (pyError && jsError) outcome = "bothError";
  else if (pyError) outcome = "pyError";
  else if (jsError) { outcome = "jsError"; detail = jsError; }
  else if (got.smiles !== item.expected.smiles) { outcome = "smilesDiff"; detail = `${got.smiles}  vs  ${item.expected.smiles}`; }
  else if (got.atomNames.join(" ") !== item.expected.atomNames.join(" ")) { outcome = "namesDiff"; }
  else outcome = "agree";
  const namesOnly = got && !pyError && got.atomNames.join(" ") === item.expected.atomNames.join(" ");
  for (const b of buckets) {
    const s = bump(b, "total");
    if (namesOnly) s.namesOnly = (s.namesOnly ?? 0) + 1;
    s.timeMs += dt;
    if (outcome === "agree" || outcome === "bothError") s.agree++;
    if (outcome !== "agree") s[outcome]++;
    if (got && got.exact) {
      s.exactTrue++;
      if (outcome === "agree") s.exactTrueAgree++;
    }
  }
  if (outcome !== "agree" && outcome !== "bothError") {
    const key = `${item.bucket} / ${outcome}${outcome === "jsError" ? `: ${detail.slice(0, 60)}` : ""}`;
    if (!failures.has(key)) failures.set(key, []);
    failures.get(key).push({ item, got, detail });
  }
}

const pct = (a, b) => (b ? `${((100 * a) / b).toFixed(2)}%` : "n/a");
console.log("bucket".padEnd(40), "n".padStart(6), "agree".padStart(8), "rate".padStart(8), "exact:true".padStart(11), "agree(exact)".padStart(13), "names-only".padStart(11), "ms/mol");
for (const [bucket, s] of [...stats.entries()].sort()) {
  console.log(
    bucket.padEnd(40), String(s.total).padStart(6), String(s.agree).padStart(8), pct(s.agree, s.total).padStart(8),
    String(s.exactTrue).padStart(11), pct(s.exactTrueAgree, s.exactTrue).padStart(13), pct(s.namesOnly ?? 0, s.total).padStart(11), (s.timeMs / s.total).toFixed(2),
  );
}
console.log("\nbreakdown (ALL): ", JSON.stringify(stats.get("ALL")));
console.log("\nfailure categories:");
for (const [key, list] of [...failures.entries()].sort((a, b) => b[1].length - a[1].length)) {
  console.log(`\n${list.length}\t${key}`);
  for (const { item, got, detail } of list.slice(0, show)) {
    console.log(`   in : ${item.smiles}`);
    console.log(`   py : ${item.expected.smiles}  ${item.expected.atomNames?.join(" ")}`);
    if (got) console.log(`   js : ${got.smiles}  ${got.atomNames.join(" ")}`);
    else console.log(`   js : ${detail}`);
  }
}
