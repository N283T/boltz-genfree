#!/usr/bin/env node
// Static server for the browser page: serves webgpu/ (page, sources, packs/) at / with HTTP Range support, so the page can
// open the 805 MB trunk pack lazily (OwnWeights.fromUrl(..., {lazy: true}) fetches one layer's byte range at a time).
//   node tools/serve.js [port]
import http from "node:http";
import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import { extname, join, normalize, resolve, sep } from "node:path";

const root = resolve(new URL("..", import.meta.url).pathname);
const types = { ".js": "text/javascript", ".mjs": "text/javascript", ".html": "text/html", ".json": "application/json", ".wasm": "application/wasm", ".bin": "application/octet-stream" };

http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, "http://x");
    const path = decodeURIComponent(url.pathname);
    const file = normalize(join(root, path === "/" ? "/web/index.html" : path));
    if (!file.startsWith(root + sep)) throw new Error("outside root");
    const info = await stat(file);
    const size = info.size;
    if (!info.isFile()) throw new Error("not a file");
    const headers = { "content-type": types[extname(file)] ?? "application/octet-stream", "cache-control": "no-store", "accept-ranges": "bytes" };
    const range = /^bytes=(\d*)-(\d*)$/.exec(request.headers.range ?? "");
    if (range && (range[1] !== "" || range[2] !== "")) {
      let start = range[1] === "" ? size - Number(range[2]) : Number(range[1]);
      let end = range[1] === "" || range[2] === "" ? size - 1 : Math.min(Number(range[2]), size - 1);
      start = Math.max(0, start);
      if (start > end || start >= size) {
        response.writeHead(416, { "content-range": `bytes */${size}` });
        response.end();
        return;
      }
      response.writeHead(206, { ...headers, "content-range": `bytes ${start}-${end}/${size}`, "content-length": end - start + 1 });
      createReadStream(file, { start, end }).pipe(response);
      return;
    }
    response.writeHead(200, { ...headers, "content-length": size });
    createReadStream(file).pipe(response);
  } catch {
    response.writeHead(404);
    response.end("not found");
  }
}).listen(Number(process.argv[2] ?? 4180), "127.0.0.1", () => console.log(`http://127.0.0.1:${process.argv[2] ?? 4180}/`));
