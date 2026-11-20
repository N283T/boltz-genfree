/** Error statistics between a candidate and a reference tensor. */
export function compare(actual, expected) {
  if (actual.length !== expected.length) {
    throw new Error(`length mismatch: ${actual.length} vs ${expected.length}`);
  }
  let maxAbs = 0;
  let sumSq = 0;
  let refSq = 0;
  let maxRef = 0;
  for (let i = 0; i < actual.length; i++) {
    const d = actual[i] - expected[i];
    if (!Number.isFinite(actual[i])) return { maxAbs: Infinity, relRms: Infinity, maxRef: 0 };
    maxAbs = Math.max(maxAbs, Math.abs(d));
    sumSq += d * d;
    refSq += expected[i] * expected[i];
    maxRef = Math.max(maxRef, Math.abs(expected[i]));
  }
  return { maxAbs, relRms: Math.sqrt(sumSq / Math.max(refSq, 1e-30)), maxRef };
}
