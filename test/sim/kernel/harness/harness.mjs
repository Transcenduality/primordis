// Standalone WGSL kernel harness for PRIMORDIS-TASK-003.
//
// Loads the canonical kernel (../../../../lib/sim/kernel/primordis.wgsl) into a
// real WebGPU runtime and drives the three compute passes, so the *shader's
// behaviour* can be validated outside Flutter (the .wgsl is non-Dart and so is
// not covered by `flutter test`).
//
// This module is environment-agnostic plumbing (device + buffers + dispatch +
// readback). It has NO static `node:*` imports: it picks the WebGPU runtime and
// the source-loading strategy at runtime, so it imports unchanged in both
//
//   • Node.js  — `@kmamal/gpu` (Dawn/Tint); source read from disk, and
//   • a browser — `navigator.gpu`; source fetched over HTTP (the Naga path).
//
// `run.mjs` is the Node CLI that drives this module and holds the assertions.
// The bind-group/binding layout mirrors `kernel_source.dart` (compute group 0).

/** Canonical location of the kernel, resolved relative to this module. */
export const KERNEL_URL = new URL(
  '../../../../lib/sim/kernel/primordis.wgsl',
  import.meta.url,
)

export const WORKGROUP_SIZE = 256
export const MAX_BIN_PARTICLES = 512

/** True when running under Node.js (vs a browser). */
const isNode = typeof process !== 'undefined' && !!process.versions?.node

/** Resolve a WebGPU adapter source + the usage-flag enums for the host env. */
export async function getEnv() {
  if (!isNode) {
    return {
      source: navigator.gpu,
      BufferUsage: globalThis.GPUBufferUsage,
      MapMode: globalThis.GPUMapMode,
      destroy: () => {},
    }
  }
  const mod = (await import('@kmamal/gpu')).default
  const instance = mod.create([])
  return {
    source: instance,
    BufferUsage: mod.GPUBufferUsage,
    MapMode: mod.GPUMapMode,
    destroy: () => mod.destroy(instance),
  }
}

let _kernelCode = null

/**
 * Load the WGSL source (memoised). On Node it reads the file from disk; in a
 * browser it fetches `KERNEL_URL` over HTTP. Await this once before [createSim].
 */
export async function loadKernelSource() {
  if (_kernelCode != null) return _kernelCode
  if (isNode) {
    const { readFile } = await import('node:fs/promises')
    _kernelCode = await readFile(KERNEL_URL, 'utf8')
  } else {
    _kernelCode = await (await fetch(KERNEL_URL)).text()
  }
  return _kernelCode
}

/** The already-loaded source. Throws if [loadKernelSource] hasn't run yet. */
export function kernelCode() {
  if (_kernelCode == null) {
    throw new Error('call loadKernelSource() before createSim()')
  }
  return _kernelCode
}

/** Pack the 64-byte uniform block, matching SimMarshalling's slot order. */
export function packParams(p) {
  const buf = new ArrayBuffer(64)
  const dv = new DataView(buf)
  const f = (slot, v) => dv.setFloat32(slot * 4, v, true)
  const u = (slot, v) => dv.setUint32(slot * 4, v, true)
  f(0, p.attractionK)
  f(1, p.repulsionK)
  f(2, p.friction)
  f(3, p.dt)
  f(4, p.worldWidth)
  f(5, p.worldHeight)
  f(6, p.maxRadius)
  f(7, p.binSize)
  u(8, p.gridWidth)
  u(9, p.gridHeight)
  u(10, p.numParticles)
  u(11, p.numBins)
  u(12, p.typeCount)
  return buf
}

const ceilDiv = (a, b) => Math.floor((a + b - 1) / b)

/**
 * Build a simulation on `device` from `params` (uniform values) and `initial`
 * SoA buffers ({positions, velocities, types, forces, minDistances, radii}).
 * Returns handles to step the passes and read buffers back.
 */
export function createSim(device, env, params, initial) {
  const BU = env.BufferUsage
  const module = device.createShaderModule({ code: kernelCode() })

  const storage = (data) => {
    const b = device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: BU.STORAGE | BU.COPY_DST | BU.COPY_SRC,
      mappedAtCreation: true,
    })
    new data.constructor(b.getMappedRange()).set(data)
    b.unmap()
    return b
  }

  const numBins = params.numBins
  const positions = storage(initial.positions)
  const velocities = storage(initial.velocities)
  const types = storage(initial.types)
  const forces = storage(initial.forces)
  const minDistances = storage(initial.minDistances)
  const radii = storage(initial.radii)
  const binCounts = storage(new Uint32Array(numBins))
  const binParticles = storage(new Uint32Array(numBins * MAX_BIN_PARTICLES))

  const paramsBuf = device.createBuffer({ size: 64, usage: BU.UNIFORM | BU.COPY_DST })
  device.queue.writeBuffer(paramsBuf, 0, packParams(params))

  // Explicit group-0 layout with all nine compute bindings, shared by all three
  // pipelines (unused bindings per pass are allowed).
  const ro = 'read-only-storage'
  const rw = 'storage'
  const layout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: 4, buffer: { type: 'uniform' } },
      { binding: 1, visibility: 4, buffer: { type: rw } },
      { binding: 2, visibility: 4, buffer: { type: rw } },
      { binding: 3, visibility: 4, buffer: { type: ro } },
      { binding: 4, visibility: 4, buffer: { type: ro } },
      { binding: 5, visibility: 4, buffer: { type: ro } },
      { binding: 6, visibility: 4, buffer: { type: ro } },
      { binding: 7, visibility: 4, buffer: { type: rw } },
      { binding: 8, visibility: 4, buffer: { type: rw } },
    ],
  })
  const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] })

  const constants = { WORKGROUP_SIZE, MAX_BIN_PARTICLES }
  const pipe = (entryPoint) =>
    device.createComputePipeline({
      layout: pipelineLayout,
      compute: { module, entryPoint, constants },
    })
  const clearPipe = pipe('clearBins')
  const scatterPipe = pipe('scatterBins')
  const interactPipe = pipe('interact')

  const bindGroup = device.createBindGroup({
    layout,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: positions } },
      { binding: 2, resource: { buffer: velocities } },
      { binding: 3, resource: { buffer: types } },
      { binding: 4, resource: { buffer: forces } },
      { binding: 5, resource: { buffer: minDistances } },
      { binding: 6, resource: { buffer: radii } },
      { binding: 7, resource: { buffer: binCounts } },
      { binding: 8, resource: { buffer: binParticles } },
    ],
  })

  const nBinGroups = ceilDiv(numBins, WORKGROUP_SIZE)
  const nPartGroups = ceilDiv(params.numParticles, WORKGROUP_SIZE)

  function step(frames = 1) {
    for (let n = 0; n < frames; n++) {
      const enc = device.createCommandEncoder()
      const pass = enc.beginComputePass()
      pass.setPipeline(clearPipe)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(nBinGroups)
      pass.setPipeline(scatterPipe)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(nPartGroups)
      pass.setPipeline(interactPipe)
      pass.setBindGroup(0, bindGroup)
      pass.dispatchWorkgroups(nPartGroups)
      pass.end()
      device.queue.submit([enc.finish()])
    }
  }

  async function read(buffer, byteLength, Ctor) {
    const staging = device.createBuffer({
      size: byteLength,
      usage: BU.COPY_DST | BU.MAP_READ,
    })
    const enc = device.createCommandEncoder()
    enc.copyBufferToBuffer(buffer, 0, staging, 0, byteLength)
    device.queue.submit([enc.finish()])
    await staging.mapAsync(env.MapMode.READ)
    const out = new Ctor(staging.getMappedRange().slice(0))
    staging.unmap()
    staging.destroy()
    return out
  }

  return {
    step,
    readPositions: () => read(positions, initial.positions.byteLength, Float32Array),
    readVelocities: () =>
      read(velocities, initial.velocities.byteLength, Float32Array),
    readBinCounts: () => read(binCounts, numBins * 4, Uint32Array),
    readBinParticles: () =>
      read(binParticles, numBins * MAX_BIN_PARTICLES * 4, Uint32Array),
    destroy() {
      for (const b of [
        positions, velocities, types, forces, minDistances, radii,
        binCounts, binParticles, paramsBuf,
      ]) b.destroy()
    },
  }
}

/** Default world/grid params (the reference 1080x720, 96px bins, 11x7 grid). */
export function worldParams(overrides = {}) {
  return {
    attractionK: 1, repulsionK: 1, friction: 1, dt: 1,
    worldWidth: 1080, worldHeight: 720, maxRadius: 96, binSize: 96,
    gridWidth: 11, gridHeight: 7, numBins: 77,
    numParticles: 0, typeCount: 0,
    ...overrides,
  }
}
