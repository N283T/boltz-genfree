// Minimal compute context: pipeline cache, buffers, uniform-parameter dispatch.
import { GROUP_WIDTH } from "./wgsl.js";

const STORAGE = 0x80 | 0x4 | 0x8; // STORAGE | COPY_SRC | COPY_DST (GPUBufferUsage constants)

export class GpuContext {
  constructor(device) {
    this.device = device;
    this.pipelines = new Map();
    this.encoder = null;
    this.transient = [];
    this.submitted = [];
    this.live = [];
  }

  pipeline(key, code) {
    let pipeline = this.pipelines.get(key);
    if (pipeline === undefined) {
      const module = this.device.createShaderModule({ label: key, code });
      pipeline = this.device.createComputePipeline({ label: key, layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelines.set(key, pipeline);
    }
    return pipeline;
  }

  /** A zero-initialised storage buffer. Scratch buffers are freed by `release()`. */
  buffer(label, floats, persistent = false) {
    const size = Math.max(16, floats * 4);
    const buffer = this.device.createBuffer({ label, size, usage: STORAGE });
    if (!persistent) this.live.push(buffer);
    return buffer;
  }

  upload(label, data, persistent = false) {
    const buffer = this.buffer(label, data.length, persistent);
    this.device.queue.writeBuffer(buffer, 0, data.buffer, data.byteOffset, data.byteLength);
    return buffer;
  }

  /**
   * Encode one dispatch. `bindings` are GPUBuffers or {buffer, offset, size}.
   * `params` fills the 8-word uniform block (numbers; Float32 via `{f32}`).
   */
  dispatch(key, code, bindings, params, groupsX, groupsY = 1) {
    if (groupsX === 0 || groupsY === 0) return;
    this.encoder ??= this.device.createCommandEncoder();
    const pipeline = this.pipeline(key, code);
    const uniform = this.device.createBuffer({ size: 32, usage: 0x40 | 0x8 }); // UNIFORM | COPY_DST
    const words = new Uint32Array(8);
    params.forEach((value, i) => {
      words[i] = typeof value === "object" ? new Uint32Array(Float32Array.of(value.f32).buffer)[0] : value;
    });
    this.device.queue.writeBuffer(uniform, 0, words);
    this.transient.push(uniform);
    const entries = bindings.map((resource, binding) => ({
      binding,
      resource: resource.buffer === undefined ? { buffer: resource } : resource,
    }));
    entries.push({ binding: bindings.length, resource: { buffer: uniform } });
    const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    const pass = this.encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(groupsX, groupsY);
    pass.end();
  }

  /** Dispatch `elements / 64` workgroups folded over x and y. */
  dispatch1d(key, code, bindings, params, elements) {
    const groups = Math.ceil(elements / 64);
    this.dispatch(key, code, bindings, params, Math.min(groups, GROUP_WIDTH), Math.ceil(groups / GROUP_WIDTH));
  }

  flush() {
    if (this.encoder === null) return;
    this.device.queue.submit([this.encoder.finish()]);
    this.encoder = null;
    // Uniform blocks are destroyed only after a readback or release has
    // synchronised with the queue; destroying them right after submit hung the
    // Dawn node binding intermittently.
    this.submitted.push(...this.transient);
    this.transient = [];
  }

  #reapSubmitted() {
    for (const buffer of this.submitted) buffer.destroy();
    this.submitted = [];
  }

  async read(buffer, floats, byteOffset = 0) {
    this.flush();
    const staging = this.device.createBuffer({ size: floats * 4, usage: 0x1 | 0x8 }); // MAP_READ | COPY_DST
    const encoder = this.device.createCommandEncoder();
    encoder.copyBufferToBuffer(buffer, byteOffset, staging, 0, floats * 4);
    this.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(1);
    const out = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    this.#reapSubmitted();
    return out;
  }

  /** Free scratch buffers once their work has been submitted. */
  async release() {
    this.flush();
    await this.device.queue.onSubmittedWorkDone();
    this.#reapSubmitted();
    for (const buffer of this.live) buffer.destroy();
    this.live = [];
  }
}
