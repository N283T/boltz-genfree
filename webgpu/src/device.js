// WebGPU device creation for Node and the browser.
//
// Node: uses the optional `webgpu` package (Dawn's node binding, the same one
// LocalFold uses). If LOCALFOLD_DIR points at a LocalFold checkout, the device
// is requested through LocalFold's own `requestAlphaFoldDevice`
// (webgpu/runtime/device.js) so that limits and features match its kernels;
// otherwise a plain device with the adapter's maximum buffer limits is used.
// Browser: call `deviceFromAdapter(await navigator.gpu.requestAdapter())`.
import { pathToFileURL } from "node:url";
import { join } from "node:path";

const LIMITS = [
  "maxStorageBufferBindingSize", "maxBufferSize", "maxComputeWorkgroupStorageSize",
  "maxComputeInvocationsPerWorkgroup", "maxComputeWorkgroupSizeX", "maxComputeWorkgroupsPerDimension",
];

export async function deviceFromAdapter(adapter, { localfoldDir } = {}) {
  if (adapter === null || adapter === undefined) throw new Error("no WebGPU adapter available");
  if (localfoldDir) {
    const mod = await import(pathToFileURL(join(localfoldDir, "webgpu/runtime/device.js")).href);
    return mod.requestAlphaFoldDevice(adapter);
  }
  const requiredLimits = {};
  for (const name of LIMITS) if (typeof adapter.limits?.[name] === "number") requiredLimits[name] = adapter.limits[name];
  return adapter.requestDevice({ requiredLimits });
}

// The Dawn instance must stay referenced for the life of the process: if it is
// garbage-collected while a callback is pending, dawn.node segfaults in
// InstanceBase::ProcessEvents (observed as exit code 139 mid-readback).
const keepAlive = [];

export async function createNodeDevice(options = {}) {
  let binding;
  try {
    binding = await import("webgpu");
  } catch (cause) {
    throw new Error(`Node WebGPU needs the \`webgpu\` package (npm install in webgpu/): ${cause.message.split("\n")[0]}`);
  }
  Object.assign(globalThis, binding.globals);
  const gpu = binding.create([]);
  keepAlive.push(gpu);
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  if (adapter === null) throw new Error("no WebGPU adapter: Dawn found no usable GPU on this machine");
  const device = await deviceFromAdapter(adapter, { localfoldDir: options.localfoldDir ?? process.env.LOCALFOLD_DIR });
  return { device, adapter, gpu };
}
