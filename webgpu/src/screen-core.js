// Pure helpers shared by the Node `screen` command (src/screen.js) and the browser page (web/screen.html):
// delimited-text parsing, ligand lists, and GenFree's scores.csv rows. No Node or DOM APIs here.

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

export const csvCell = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));

/** GenFree's scores.csv columns, in order. */
export const SCORE_COLUMNS = ["record_id", "ligand_id", "ligand_kind", "ligand_value", "ligand_tokens", "chunk", "crop_budget", "receptor_tokens", "affinity_pred_value", "affinity_probability_binary"];

/** One scores.csv row for ligand number `index` (0-based) of the library. */
export function scoreRow(lig, index, result, chunkSize = 1000) {
  return {
    record_id: `${String(index + 1).padStart(6, "0")}_${lig.id}`, ligand_id: lig.id, ligand_kind: lig.kind, ligand_value: lig.value,
    ligand_tokens: result.ligandTokens, chunk: `chunk_${String(Math.floor(index / chunkSize)).padStart(5, "0")}`,
    crop_budget: "complete_input", receptor_tokens: result.receptorTokens,
    affinity_pred_value: result.value, affinity_probability_binary: result.probability,
  };
}

/** scores.csv text from rows (objects keyed by SCORE_COLUMNS); `extra` adds trailing columns, e.g. ["error"]. */
export function scoresCsv(rows, extra = []) {
  const columns = [...SCORE_COLUMNS, ...extra];
  return `${[columns.join(","), ...rows.map((r) => columns.map((c) => csvCell(r[c] ?? "")).join(","))].join("\n")}\n`;
}

/**
 * Ligands pasted or loaded as text. Accepted shapes (first match wins):
 *   a header row naming a `smiles` column (and optionally `id`) -> those columns;
 *   two columns per row -> id, smiles;  one column per row -> SMILES with ids ligand_1, ligand_2, ...
 * The delimiter (comma, tab or semicolon) is detected from the first line. Returns [{id, kind: "smiles", value}].
 */
export function parseLigandText(text, { idColumn = "id", smilesColumn = "smiles" } = {}) {
  const first = text.split(/\r?\n/).find((l) => l.trim() !== "") ?? "";
  const delimiter = [",", "\t", ";"].find((d) => first.includes(d)) ?? ",";
  const rows = parseDelimited(text, delimiter).map((r) => r.map((x) => x.trim()));
  if (rows.length === 0) return [];
  const header = rows[0].map((h) => h.toLowerCase());
  const iSmiles = header.indexOf(smilesColumn.toLowerCase());
  let body, pick;
  if (iSmiles >= 0) {
    let iId = header.indexOf(idColumn.toLowerCase());
    if (iId < 0 && rows[0].length > 1) iId = iSmiles === 0 ? 1 : 0;
    body = rows.slice(1);
    pick = (r, n) => [iId >= 0 ? r[iId] : "", r[iSmiles], n];
  } else if (rows[0].length >= 2) {
    body = rows;
    pick = (r, n) => [r[0], r[1], n];
  } else {
    body = rows;
    pick = (r, n) => ["", r[0], n];
  }
  const seen = new Set();
  return body.map((r, n) => {
    const [rawId, value] = pick(r, n);
    const id = rawId || `ligand_${n + 1}`;
    if (!value) throw new Error(`ligand ${n + 1}: empty SMILES`);
    if (seen.has(id)) throw new Error(`duplicate ligand id ${JSON.stringify(id)}`);
    seen.add(id);
    return { id, kind: "smiles", value };
  });
}
