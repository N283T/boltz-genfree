// Minimal YAML reader for the screen config: nested mappings, block and inline lists,
// quoted / plain scalars, comments. Not a general YAML parser (no anchors, multi-line scalars, flow maps).

function scalar(text) {
  const t = text.trim();
  if (t === "" || t === "~" || t === "null") return null;
  if (t === "true") return true;
  if (t === "false") return false;
  if (/^-?\d+$/.test(t)) return Number(t);
  if (/^-?\d*\.\d+(e[+-]?\d+)?$/i.test(t)) return Number(t);
  if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) return t.slice(1, -1);
  if (t.startsWith("[") && t.endsWith("]")) {
    const inner = t.slice(1, -1).trim();
    return inner === "" ? [] : inner.split(",").map(scalar);
  }
  return t;
}

function stripComment(line) {
  let quote = null;
  for (let i = 0; i < line.length; i += 1) {
    const c = line[i];
    if (quote) { if (c === quote) quote = null; } else if (c === '"' || c === "'") quote = c;
    else if (c === "#" && (i === 0 || /\s/.test(line[i - 1]))) return line.slice(0, i);
  }
  return line;
}

export function parseYaml(text) {
  const lines = text.split(/\r?\n/).map(stripComment).map((l) => l.replace(/\s+$/, ""))
    .map((raw, n) => ({ raw, n: n + 1 })).filter((l) => l.raw.trim() !== "" && l.raw.trim() !== "---");
  let pos = 0;
  const indentOf = (l) => l.raw.match(/^ */)[0].length;

  function block(indent) {
    if (lines[pos].raw.trim().startsWith("- ") || lines[pos].raw.trim() === "-") return list(indent);
    return mapping(indent);
  }
  function list(indent) {
    const out = [];
    while (pos < lines.length && indentOf(lines[pos]) === indent && lines[pos].raw.trim().startsWith("-")) {
      out.push(scalar(lines[pos].raw.trim().slice(1)));
      pos += 1;
    }
    return out;
  }
  function mapping(indent) {
    const out = {};
    while (pos < lines.length && indentOf(lines[pos]) === indent) {
      const line = lines[pos];
      const m = line.raw.trim().match(/^([^:]+?):(?:\s+(.*))?$/);
      if (!m) throw new Error(`YAML line ${line.n}: expected "key: value", got ${JSON.stringify(line.raw.trim())}`);
      pos += 1;
      const rest = m[2];
      if (rest !== undefined && rest.trim() !== "") out[m[1].trim()] = scalar(rest);
      else if (pos < lines.length && indentOf(lines[pos]) > indent) out[m[1].trim()] = block(indentOf(lines[pos]));
      else out[m[1].trim()] = null;
    }
    if (pos < lines.length && indentOf(lines[pos]) > indent) throw new Error(`YAML line ${lines[pos].n}: unexpected indentation`);
    return out;
  }
  if (lines.length === 0) return {};
  return block(indentOf(lines[0]));
}
