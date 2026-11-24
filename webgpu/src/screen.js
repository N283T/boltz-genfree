// `node src/cli.js screen screen.yaml`: the WebGPU counterpart of `boltz-genfree screen`.
// Reads the same config (target.sequence / target.msa, ligands.csv|tsv|smiles|ccd, options.*),
// scores every ligand with the own chain and writes scores.csv (same columns as GenFree),
// screen_status.json and a resumable per-record log. A changed input is rejected on resume.
import { parseArgs } from "node:util";
import { createHash } from "node:crypto";
import { appendFileSync, existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { parseYaml } from "./yaml-lite.js";
import { TensorBundle } from "./bundle.js";

const here = (p) => resolve(new URL(`../${p}`, import.meta.url).pathname);

/** Rows of fields; quoted fields and "" escapes; delimiter configurable. */
export function parseDelimited(text, delimiter = ",") {
  const rows = [];
  let row = [], field = "", quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const c = text[i];
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i += 1; } else if (c === '"') quoted = false; else field += c;
    } else if (c === '"') quoted = true;
    else if (c === delimiter) { row.push(field); field = ""; }
    else if (c === "\n" || c === "\r") {
      if (c === "\r" && text[i + 1] === "\n") i += 1;
      row.push(field); field = "";
      if (row.some((x) => x !== "")) rows.push(row);
      row = [];
    } else field += c;
  }
  row.push(field);
  if (row.some((x) => x !== "")) rows.push(row);
  return rows;
}

const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

/** Ligand records [{id, kind: "smiles"|"ccd", value}] from the `ligands:` block. */
export function readLigands(ligands, baseDir) {
  if (!ligands || typeof ligands !== "object") throw new Error("config needs a `ligands:` block");
  const sources = ["csv", "tsv", "smiles", "ccd", "sdf"].filter((k) => ligands[k] !== undefined && ligands[k] !== null);
  if (sources.length !== 1) throw new Error(`ligands needs exactly one source (csv, tsv, smiles or ccd); got ${sources.length}`);
  const source = sources[0];
  if (source === "sdf") throw new Error("ligands.sdf is not supported by the WebGPU screen (SDF coordinates are not reused by GenFree either; convert to csv with SMILES)");
  if (source === "smiles" || source === "ccd") {
    const list = Array.isArray(ligands[source]) ? ligands[source] : [ligands[source]];
    return list.map((value, i) => ({ id: `${source}_${i + 1}`, kind: source, value: String(value) }));
  }
  const delimiter = ligands.delimiter ?? (source === "tsv" ? "\t" : ",");
  const [header, ...body] = parseDelimited(readFileSync(resolve(baseDir, ligands[source]), "utf8"), delimiter);
  const idColumn = ligands.id_column ?? "id";
  const smilesColumn = ligands.smiles_column ?? "smiles";
  const iId = header.indexOf(idColumn), iSmiles = header.indexOf(smilesColumn);
  if (iId < 0 || iSmiles < 0) throw new Error(`${ligands[source]}: columns ${JSON.stringify(idColumn)} and ${JSON.stringify(smilesColumn)} are required; header is ${JSON.stringify(header)}`);
  const seen = new Set();
  return body.map((r, n) => {
    const id = (r[iId] ?? "").trim(), value = (r[iSmiles] ?? "").trim();
    if (!id || !value) throw new Error(`${ligands[source]} row ${n + 2}: empty id or smiles`);
    if (seen.has(id)) throw new Error(`${ligands[source]}: duplicate id ${JSON.stringify(id)}`);
    seen.add(id);
    return { id, kind: "smiles", value };
  });
}

const SCORE_COLUMNS = ["record_id", "ligand_id", "ligand_kind", "ligand_value", "ligand_tokens", "chunk", "crop_budget", "receptor_tokens", "affinity_pred_value", "affinity_probability_binary"];
const IGNORED_OPTIONS = new Set(["devices", "num_workers", "affinity_batch_size", "no_kernels", "msa_server_url", "msa_server_username", "msa_server_password", "msa_pairing_strategy", "affinity_conditioner_mode"]);

export function loadScreenConfig(configPath) {
  const baseDir = dirname(resolve(configPath));
  const config = parseYaml(readFileSync(configPath, "utf8"));
  if (config.version !== undefined && config.version !== 1) throw new Error(`unsupported config version ${config.version}`);
  const target = config.target;
  if (!target?.sequence) throw new Error("config needs target.sequence");
  if (target.crop) throw new Error("target.crop (reference-complex cropping) is not supported by the WebGPU screen yet; use target.sequence without crop, or the Python `boltz-genfree screen`");
  const options = config.options ?? {};
  for (const key of Object.keys(options)) {
    if (!["seed", "affinity_recycling_steps", "ligand_reference_conformer", "allow_extended_full_input"].includes(key) && !IGNORED_OPTIONS.has(key)) {
      throw new Error(`options.${key} is not supported by the WebGPU screen`);
    }
  }
  if (options.affinity_conditioner_mode && options.affinity_conditioner_mode !== "original") throw new Error("options.affinity_conditioner_mode: only `original` is supported");
  if (options.affinity_checkpoint || options.affinity_adapter) throw new Error("options.affinity_checkpoint / affinity_adapter are not supported (the pack is fixed; pass --pack)");
  const notes = [];
  let a3m = null;
  const msa = target.msa ?? "auto";   // GenFree's default when the key is absent
  if (msa === "empty") a3m = null;
  else if (msa !== "auto") a3m = readFileSync(resolve(baseDir, String(msa)), "utf8");
  else notes.push("target.msa: auto needs the MSA server, which the WebGPU screen does not call; scored without an MSA (use msa: empty to say so explicitly, or give an A3M path)");
  return {
    baseDir, notes, a3m,
    sequence: String(target.sequence).replace(/\s+/g, ""),
    ligands: readLigands(config.ligands, baseDir),
    outputDir: resolve(baseDir, config.output_dir ?? "screen-output"),
    chunkSize: Number(config.chunk_size ?? 1000),
    seed: Number(options.seed ?? 0),
    recycles: Number(options.affinity_recycling_steps ?? 0),
    conformer: options.ligand_reference_conformer ?? "skip",
    allowExtended: Boolean(options.allow_extended_full_input),
  };
}

const fingerprint = (cfg) => createHash("sha256").update(JSON.stringify({
  sequence: cfg.sequence, a3m: cfg.a3m, seed: cfg.seed, recycles: cfg.recycles, conformer: cfg.conformer,
  allowExtended: cfg.allowExtended, ligands: cfg.ligands,
})).digest("hex");

function writeAtomic(path, text) {
  const tmp = `${path}.tmp`;
  writeFileSync(tmp, text);
  renameSync(tmp, path);
}

export async function runScreenCli(argv) {
  const { values, positionals } = parseArgs({
    args: argv, allowPositionals: true,
    options: {
      "prepare-only": { type: "boolean", default: false },
      "trunk-pack": { type: "string", default: here("packs/boltz2-trunk-own/boltz2-trunk-own.json") },
      pack: { type: "string", default: here("packs/genfree-affinity.json") },
    },
  });
  if (positionals.length !== 1) { console.error("usage: node src/cli.js screen <screen.yaml> [--prepare-only] [--trunk-pack ...] [--pack ...]"); process.exit(2); }
  let cfg;
  try { cfg = loadScreenConfig(positionals[0]); } catch (err) { console.error(`error: ${err.message}`); process.exit(2); }
  const print = (m) => console.error(m);
  for (const n of cfg.notes) print(`note: ${n}`);
  print(`target: ${cfg.sequence.length} residues, ${cfg.a3m ? "with MSA" : "no MSA"}; ${cfg.ligands.length} ligands; conformer ${cfg.conformer}; seed ${cfg.seed}; recycles ${cfg.recycles}`);
  if (values["prepare-only"]) { print("prepare-only: inputs are valid"); return; }

  mkdirSync(cfg.outputDir, { recursive: true });
  const statusPath = resolve(cfg.outputDir, "screen_status.json");
  const logPath = resolve(cfg.outputDir, ".webgpu-records.jsonl");
  const scoresPath = resolve(cfg.outputDir, "scores.csv");
  const print256 = fingerprint(cfg);
  if (existsSync(statusPath)) {
    const prev = JSON.parse(readFileSync(statusPath, "utf8"));
    if (prev.fingerprint !== print256) { console.error(`error: ${cfg.outputDir} holds a run with different inputs (target, MSA, ligands or options); use a new output_dir`); process.exit(2); }
  }
  const done = new Map();
  if (existsSync(logPath)) {
    for (const line of readFileSync(logPath, "utf8").split("\n")) if (line.trim()) { const r = JSON.parse(line); done.set(r.ligand_id, r); }
  }
  if (done.size) print(`resuming: ${done.size}/${cfg.ligands.length} already scored`);

  const flush = (completed) => {
    const rows = cfg.ligands.filter((l) => done.has(l.id)).map((l) => done.get(l.id));
    writeAtomic(scoresPath, `${[SCORE_COLUMNS.join(","), ...rows.map((r) => SCORE_COLUMNS.map((c) => csvCell(r[c])).join(","))].join("\n")}\n`);
    writeAtomic(statusPath, `${JSON.stringify({ schema_version: 1, fingerprint: print256, expected: cfg.ligands.length, completed: done.size, finished: completed, backend: "webgpu" }, null, 2)}\n`);
  };
  flush(done.size === cfg.ligands.length);

  const todo = cfg.ligands.filter((l) => !done.has(l.id));
  if (todo.length === 0) { print(`all ${cfg.ligands.length} ligands already scored: ${scoresPath}`); return; }

  if (!/^https?:\/\//.test(values["trunk-pack"]) && !existsSync(values["trunk-pack"])) {
    console.error(`error: no trunk pack at ${values["trunk-pack"]}: run tools/convert_boltz_trunk.py or pass --trunk-pack`);
    process.exit(2);
  }
  const affinityPack = await TensorBundle.open(values.pack);
  const { createNodeDevice } = await import("./device.js");
  const { createOwnChain } = await import("./trunk/own/chain.js");
  const { device } = await createNodeDevice();
  const chain = await createOwnChain({ trunkPack: values["trunk-pack"], affinityPack, device });
  const started = performance.now();
  try {
    for (const [n, lig] of todo.entries()) {
      let result;
      try {
        result = await chain.score(
          { sequence: cfg.sequence, a3m: cfg.a3m, smiles: lig.value, ...(lig.kind === "ccd" ? { ligandKind: "ccd" } : {}), ligandConformer: cfg.conformer },
          { recycles: cfg.recycles, seed: cfg.seed, allowExtended: cfg.allowExtended },
        );
      } catch (err) {
        flush(false);
        console.error(`error: ligand ${JSON.stringify(lig.id)}: ${err.message ?? err}\n(${done.size}/${cfg.ligands.length} scored; fix the input and re-run to resume)`);
        process.exitCode = 1;
        return;
      }
      const index = cfg.ligands.indexOf(lig);
      const row = {
        record_id: `${String(index + 1).padStart(6, "0")}_${lig.id}`, ligand_id: lig.id, ligand_kind: lig.kind, ligand_value: lig.value,
        ligand_tokens: result.ligandTokens, chunk: `chunk_${String(Math.floor(index / cfg.chunkSize)).padStart(5, "0")}`,
        crop_budget: "complete_input", receptor_tokens: result.receptorTokens,
        affinity_pred_value: result.value, affinity_probability_binary: result.probability,
      };
      appendFileSync(logPath, `${JSON.stringify(row)}\n`);
      done.set(lig.id, row);
      print(`[${done.size}/${cfg.ligands.length}] ${lig.id}  value ${result.value.toFixed(4)}  P ${result.probability.toFixed(4)}  (${Math.round(result.timings.totalMs)} ms)`);
      if ((n + 1) % 25 === 0) flush(false);
    }
    flush(true);
    print(`done: ${done.size} scores in ${Math.round((performance.now() - started) / 1000)} s -> ${scoresPath}`);
  } finally {
    await chain.destroy();
    device.destroy();
  }
}
