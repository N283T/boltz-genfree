// Access to a LocalFold checkout (https://github.com/sokrypton/localfold), which is
// referenced and never copied or modified. `root` is either a directory (Node) or the
// base URL under which the checkout is served (browser, or Node with an http URL).
//
// Only the pieces the Boltz-2 trunk needs are imported, lazily, by relative path.

const isUrl = (root) => /^https?:\/\//.test(root);

async function importFrom(root, relative) {
  if (isUrl(root)) return import(new URL(relative, root.endsWith("/") ? root : `${root}/`).href);
  const { pathToFileURL } = await import("node:url");
  const { join } = await import("node:path");
  return import(pathToFileURL(join(root, relative)).href);
}

/** @returns {Promise<LocalFold>} the LocalFold modules the trunk driver uses. */
export async function loadLocalFold(root) {
  if (!root) {
    throw new Error("LocalFold is required for the trunk: set LOCALFOLD_DIR (or pass --localfold) to a "
      + "checkout of https://github.com/sokrypton/localfold");
  }
  const [weights, diffusionWeights, batch, dialect, component, contact, trunk, fold, device, msaDraw] = await Promise.all([
    importFrom(root, "shared/af3/weights/weights.js"),
    importFrom(root, "shared/af3/weights/diffusion-weights.js"),
    importFrom(root, "shared/af3/featurise/batch.js"),
    importFrom(root, "shared/af3/dialect.js"),
    importFrom(root, "shared/chem/component.js"),
    importFrom(root, "shared/af3/featurise/contact-classes.js"),
    importFrom(root, "webgpu/af3/trunk/trunk.js"),
    importFrom(root, "webgpu/af3/fold.js"),
    importFrom(root, "webgpu/runtime/device.js"),
    importFrom(root, "shared/af3/msa-draw.js"),
  ]);
  return {
    root,
    openAf3Store: weights.openAf3Store,
    trunkDepths: weights.trunkDepths,
    trunkWeights: weights.trunkWeights,
    targetFeatureWeights: diffusionWeights.targetFeatureWeights,
    af3BatchFromA3m: batch.af3BatchFromA3m,
    dialectFor: dialect.dialectFor,
    featuriserDialect: dialect.featuriserDialect,
    smilesComponent: component.smilesComponent,
    af3ContactClasses: contact.af3ContactClasses,
    Af3TrunkGpu: trunk.Af3TrunkGpu,
    buildTargetFeat: fold.buildTargetFeat,
    defaultPairWeightPrecision: fold.defaultPairWeightPrecision,
    requestAlphaFoldDevice: device.requestAlphaFoldDevice,
    msaDrawRows: msaDraw.msaDrawRows,
  };
}
