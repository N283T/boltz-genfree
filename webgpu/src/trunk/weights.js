// Opening a Boltz-2 trunk weight set (LocalFold's layout) and adapting it to GenFree.
//
// Two sources are supported, both in LocalFold's tensor-manifest layout:
//   - webgpu/packs/boltz2-trunk-f32/  written by tools/convert_trunk_weights.py from the very
//     checkpoint GenFree loads (boltz2_aff.ckpt); float32, 770 MiB. Used for parity.
//   - LocalFold's published `boltz2-int5` bundle (382 MB, int5 group-32 quantised, converted
//     from Boltz-2 by LocalFold's own tooling). Smaller, with ~4 % per-tensor weight round-off.
//
// `adaptWeights` applies what GenFree's route needs on top of either (see DESIGN.md):
//   - method conditioning row: GenFree calls the model with `override_method="other"` (class 4),
//     LocalFold's featuriser hard-wires class 1 (X-ray); row 1 receives row 4.
//   - template module: boltz2_aff.ckpt has none (`use_templates: False`), LocalFold always runs
//     it, even for an input with no template. Its output projection is zeroed so the term is
//     exactly zero.

const isUrl = (location) => /^https?:\/\//.test(location);

/** Serve a directory over http on 127.0.0.1 (Node only) so LocalFold's fetch-based store can read it. */
async function serveDirectory(directory) {
  const http = await import("node:http");
  const { readFile } = await import("node:fs/promises");
  const { join, normalize, sep } = await import("node:path");
  const root = normalize(directory);
  const server = http.createServer(async (request, response) => {
    try {
      const path = normalize(join(root, decodeURIComponent(new URL(request.url, "http://x").pathname)));
      if (!path.startsWith(root + sep)) throw new Error("outside root");
      const bytes = await readFile(path);
      response.writeHead(200, { "content-length": bytes.length, "cache-control": "no-store" });
      response.end(bytes);
    } catch {
      response.writeHead(404);
      response.end();
    }
  });
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  return { url: `http://127.0.0.1:${server.address().port}/`, close: () => new Promise((resolve) => server.close(resolve)) };
}

/**
 * @param {object} lf result of loadLocalFold
 * @param {string} location directory containing manifest.json (Node) or the URL of manifest.json / its directory
 * @returns {Promise<{store: object, close: () => Promise<void>}>}
 */
export async function openTrunkStore(lf, location, onProgress) {
  let manifestUrl = location;
  let close = async () => {};
  if (!isUrl(location)) {
    const served = await serveDirectory(location);
    manifestUrl = `${served.url}manifest.json`;
    close = served.close;
  } else if (!location.endsWith(".json")) {
    manifestUrl = `${location.replace(/\/$/, "")}/manifest.json`;
  }
  const store = await lf.openAf3Store(manifestUrl, null, onProgress);
  return { store, close };
}

export const METHOD_OTHER = 4;
const METHOD_XRAY = 1;

/** Apply GenFree's route to loaded weights. Returns new objects; the inputs are not modified. */
export function adaptWeights(trunk, targetFeat) {
  const sum = targetFeat.encoder.targetFeatSum;
  const channels = 384;
  const method = Float32Array.from(sum.method);
  method.copyWithin(METHOD_XRAY * channels, METHOD_OTHER * channels, (METHOD_OTHER + 1) * channels);
  const adaptedTargetFeat = {
    ...targetFeat,
    encoder: { ...targetFeat.encoder, targetFeatSum: { ...sum, method } },
  };
  const template = trunk.template;
  const adaptedTrunk = {
    ...trunk,
    template: { ...template, outputLinear: new Float32Array(template.outputLinear.length) },
  };
  return { trunk: adaptedTrunk, targetFeat: adaptedTargetFeat };
}

/** Load every weight the trunk and the input embedder read. */
export async function loadTrunkWeights(lf, store) {
  const depths = lf.trunkDepths(store);
  const [trunk, targetFeat] = await Promise.all([
    lf.trunkWeights(store, depths.pairformerBlocks, depths.msaBlocks),
    lf.targetFeatureWeights(store),
  ]);
  return { ...adaptWeights(trunk, targetFeat), depths };
}
