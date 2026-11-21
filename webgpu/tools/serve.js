#!/usr/bin/env node
// Static server for the browser page: one origin serving webgpu/ at / and a LocalFold checkout
// (read-only, never modified) at /localfold/, so the page can import LocalFold's modules.
//   LOCALFOLD_DIR=/path/to/localfold node tools/serve.js [port]
import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize, resolve, sep } from "node:path";

const webgpuRoot = resolve(new URL("..", import.meta.url).pathname);
const localfoldRoot = process.env.LOCALFOLD_DIR ? resolve(process.env.LOCALFOLD_DIR) : null;
const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".html": "text/html", ".json": "application/json", ".wasm": "application/wasm", ".bin": "application/octet-stream" };

http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, "http://x");
    let root = webgpuRoot;
    let path = decodeURIComponent(url.pathname);
    if (path.startsWith("/localfold/")) {
      if (!localfoldRoot) throw new Error("LOCALFOLD_DIR not set");
      root = localfoldRoot;
      path = path.slice("/localfold".length);
    }
    const file = normalize(join(root, path === "/" ? "/web/index.html" : path));
    if (!file.startsWith(root + sep)) throw new Error("outside root");
    const bytes = await readFile(file);
    response.writeHead(200, { "content-type": types[extname(file)] ?? "application/octet-stream", "cache-control": "no-store" });
    response.end(bytes);
  } catch {
    response.writeHead(404);
    response.end("not found");
  }
}).listen(Number(process.argv[2] ?? 4180), "127.0.0.1", () => console.log(`http://127.0.0.1:${process.argv[2] ?? 4180}/`));
