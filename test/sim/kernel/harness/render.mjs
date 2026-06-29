// Render-path validation for the web WebGPU backend (PRIMORDIS-TASK-004).
//
// run.mjs (TASK-003) validates the three COMPUTE passes. This harness validates
// the part TASK-004 adds: the point-RENDER pipeline built from the same shared
// kernel (vs_main / fs_main, group 1 read-only views) and the bind-group wiring
// of the full four-pass frame. It mirrors `web_webgpu_backend.dart`: same
// bind-group layout (compute group 0 + render group 1), same point-list
// topology, same target-format wiring.
//
//   node render.mjs
//
// SCOPE / ENVIRONMENT LIMIT: @kmamal/gpu 0.2.x bundles a Dawn that gates ALL
// texture views behind the `allow_unsafe_apis` toggle (a "swizzle … feature not
// enabled" validation error on every `createView`), which the binding doesn't
// expose. So a render PASS — which needs a colour-attachment view — cannot be
// executed in Node here. What IS validated under Dawn/Tint: the render pipeline
// and the group-1 bind group BUILD from the shared kernel with no validation
// error (the shader stages compile, the layout/targets/topology are valid), and
// the combined compute loop runs clean. The actual point draw + present is
// validated in a real WebGPU browser (the Naga path) and by the Flutter web
// build — the venue the task's "browser integration smoke" targets.
//
// Naga (browser) parity is out of scope here (PRIMORDIS-TASK-017).

import {
  getEnv, kernelCode, loadKernelSource, packParams, worldParams,
  WORKGROUP_SIZE, MAX_BIN_PARTICLES,
} from './harness.mjs'

let passed = 0
let failed = 0
let skipped = 0
function check(name, cond, detail = '') {
  if (cond) { passed++; console.log(`  ✓ ${name}`) }
  else { failed++; console.log(`  ✗ ${name}${detail ? ` — ${detail}` : ''}`) }
}
function skip(name, why) {
  skipped++
  console.log(`  ⚠ SKIP ${name} — ${why}`)
}

const BUFFER_UNIFORM = 0x40
const BUFFER_COPY_DST = 0x08
const TEXTURE_RENDER_ATTACHMENT = 0x10
const TEXTURE_COPY_SRC = 0x01
const VISIBILITY_VERTEX = 0x1
const VISIBILITY_COMPUTE = 0x4

const ceilDiv = (a, b) => Math.floor((a + b - 1) / b)

// Offscreen target. 256-wide so any readback `bytesPerRow` (256×4 = 1024) is the
// required 256-byte multiple; clip space is normalized so the 1080×720 world
// still maps in correctly.
const TARGET = 256
const TARGET_FORMAT = 'rgba8unorm'

function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/**
 * Build the full four-pass sim (3 compute + render) from `params` and `initial`
 * SoA buffers, mirroring web_webgpu_backend.dart. `initial` adds `typeColors`
 * (RGBA per type) on top of run.mjs's compute inputs.
 */
function createRenderSim(device, env, params, initial) {
  const BU = env.BufferUsage
  const module = device.createShaderModule({ code: kernelCode() })

  const storage = (data) => {
    const b = device.createBuffer({
      size: Math.max(4, data.byteLength),
      usage: BU.STORAGE | BU.COPY_DST,
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
  const typeColors = storage(initial.typeColors)
  const binCounts = storage(new Uint32Array(numBins))
  const binParticles = storage(new Uint32Array(numBins * MAX_BIN_PARTICLES))
  const paramsBuf = device.createBuffer({ size: 64, usage: BUFFER_UNIFORM | BUFFER_COPY_DST })
  device.queue.writeBuffer(paramsBuf, 0, packParams(params))

  const ro = 'read-only-storage'
  const rw = 'storage'
  // Compute group 0 (visibility COMPUTE) — identical to run.mjs/createSim.
  const computeLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: VISIBILITY_COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VISIBILITY_COMPUTE, buffer: { type: rw } },
      { binding: 2, visibility: VISIBILITY_COMPUTE, buffer: { type: rw } },
      { binding: 3, visibility: VISIBILITY_COMPUTE, buffer: { type: ro } },
      { binding: 4, visibility: VISIBILITY_COMPUTE, buffer: { type: ro } },
      { binding: 5, visibility: VISIBILITY_COMPUTE, buffer: { type: ro } },
      { binding: 6, visibility: VISIBILITY_COMPUTE, buffer: { type: ro } },
      { binding: 7, visibility: VISIBILITY_COMPUTE, buffer: { type: rw } },
      { binding: 8, visibility: VISIBILITY_COMPUTE, buffer: { type: rw } },
    ],
  })
  // Render group 1 (visibility VERTEX) — uniform + read-only position/type/colour.
  const renderLayout = device.createBindGroupLayout({
    entries: [
      { binding: 0, visibility: VISIBILITY_VERTEX, buffer: { type: 'uniform' } },
      { binding: 1, visibility: VISIBILITY_VERTEX, buffer: { type: ro } },
      { binding: 2, visibility: VISIBILITY_VERTEX, buffer: { type: ro } },
      { binding: 3, visibility: VISIBILITY_VERTEX, buffer: { type: ro } },
    ],
  })

  const computePipelineLayout =
    device.createPipelineLayout({ bindGroupLayouts: [computeLayout] })
  const renderPipelineLayout =
    device.createPipelineLayout({ bindGroupLayouts: [computeLayout, renderLayout] })

  const constants = { WORKGROUP_SIZE, MAX_BIN_PARTICLES }
  const pipe = (entryPoint) =>
    device.createComputePipeline({
      layout: computePipelineLayout,
      compute: { module, entryPoint, constants },
    })
  const clearPipe = pipe('clearBins')
  const scatterPipe = pipe('scatterBins')
  const interactPipe = pipe('interact')

  // The new TASK-004 surface: render pipeline from the shared kernel's vs/fs.
  const renderPipe = device.createRenderPipeline({
    layout: renderPipelineLayout,
    vertex: { module, entryPoint: 'vs_main' },
    fragment: { module, entryPoint: 'fs_main', targets: [{ format: TARGET_FORMAT }] },
    primitive: { topology: 'point-list' },
  })

  const computeBindGroup = device.createBindGroup({
    layout: computeLayout,
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
  const renderBindGroup = device.createBindGroup({
    layout: renderLayout,
    entries: [
      { binding: 0, resource: { buffer: paramsBuf } },
      { binding: 1, resource: { buffer: positions } },
      { binding: 2, resource: { buffer: types } },
      { binding: 3, resource: { buffer: typeColors } },
    ],
  })

  const target = device.createTexture({
    size: { width: TARGET, height: TARGET },
    format: TARGET_FORMAT,
    usage: TEXTURE_RENDER_ATTACHMENT | TEXTURE_COPY_SRC,
  })

  const nBinGroups = ceilDiv(numBins, WORKGROUP_SIZE)
  const nPartGroups = ceilDiv(params.numParticles, WORKGROUP_SIZE)

  function encodeCompute(enc) {
    const cpass = enc.beginComputePass()
    cpass.setPipeline(clearPipe)
    cpass.setBindGroup(0, computeBindGroup)
    cpass.dispatchWorkgroups(nBinGroups)
    cpass.setPipeline(scatterPipe)
    cpass.setBindGroup(0, computeBindGroup)
    cpass.dispatchWorkgroups(nPartGroups)
    cpass.setPipeline(interactPipe)
    cpass.setBindGroup(0, computeBindGroup)
    cpass.dispatchWorkgroups(nPartGroups)
    cpass.end()
  }

  // Compute-only frame (the part Dawn-node can execute).
  function computeFrame(frames = 1) {
    for (let n = 0; n < frames; n++) {
      const enc = device.createCommandEncoder()
      encodeCompute(enc)
      device.queue.submit([enc.finish()])
    }
  }

  // Whether this Dawn build will hand back a texture view (see the SCOPE note).
  let _view = null
  async function viewSupported() {
    device.pushErrorScope('validation')
    let v = null
    try { v = target.createView() } catch (_) { /* fall through */ }
    const err = await device.popErrorScope()
    if (err || !v) return false
    _view = v
    return true
  }

  // Full four-pass frame (only runnable when views are supported).
  function renderFrame(frames = 1) {
    for (let n = 0; n < frames; n++) {
      const enc = device.createCommandEncoder()
      encodeCompute(enc)
      const rpass = enc.beginRenderPass({
        colorAttachments: [{
          view: _view,
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        }],
      })
      rpass.setPipeline(renderPipe)
      rpass.setBindGroup(0, computeBindGroup)
      rpass.setBindGroup(1, renderBindGroup)
      rpass.draw(params.numParticles)
      rpass.end()
      device.queue.submit([enc.finish()])
    }
  }

  async function readPixels() {
    const bytesPerRow = TARGET * 4
    const staging = device.createBuffer({
      size: bytesPerRow * TARGET,
      usage: BU.COPY_DST | BU.MAP_READ,
    })
    const enc = device.createCommandEncoder()
    enc.copyTextureToBuffer(
      { texture: target },
      { buffer: staging, bytesPerRow, rowsPerImage: TARGET },
      { width: TARGET, height: TARGET },
    )
    device.queue.submit([enc.finish()])
    await staging.mapAsync(env.MapMode.READ)
    const out = new Uint8Array(staging.getMappedRange().slice(0))
    staging.unmap()
    staging.destroy()
    return out
  }

  function destroy() {
    for (const b of [
      positions, velocities, types, forces, minDistances, radii, typeColors,
      binCounts, binParticles, paramsBuf,
    ]) b.destroy()
    target.destroy()
  }

  return { computeFrame, viewSupported, renderFrame, readPixels, destroy }
}

/** Count pixels the point render lit (alpha > 0), and any carrying colour. */
function litStats(pixels) {
  let lit = 0
  let coloured = 0
  for (let i = 0; i < pixels.length; i += 4) {
    if (pixels[i + 3] > 0) {
      lit++
      if (pixels[i] > 0 || pixels[i + 1] > 0 || pixels[i + 2] > 0) coloured++
    }
  }
  return { lit, coloured }
}

function seedParticles(N, T, rnd, p) {
  const positions = new Float32Array(N * 2)
  const velocities = new Float32Array(N * 2)
  const types = new Uint32Array(N)
  for (let i = 0; i < N; i++) {
    positions[i * 2] = rnd() * p.worldWidth
    positions[i * 2 + 1] = rnd() * p.worldHeight
    velocities[i * 2] = rnd() * 16 - 8
    velocities[i * 2 + 1] = rnd() * 16 - 8
    types[i] = Math.floor(rnd() * T)
  }
  const forces = new Float32Array(T * T)
  const minDistances = new Float32Array(T * T)
  const radii = new Float32Array(T * T)
  for (let k = 0; k < T * T; k++) {
    forces[k] = (0.1 + rnd() * 0.7) * (rnd() < 0.5 ? -1 : 1)
    minDistances[k] = 4 + rnd() * 8
    radii[k] = 20 + rnd() * 76
  }
  const typeColors = new Float32Array(T * 4)
  for (let t = 0; t < T; t++) {
    typeColors[t * 4] = 0.3 + 0.7 * rnd()
    typeColors[t * 4 + 1] = 0.3 + 0.7 * rnd()
    typeColors[t * 4 + 2] = 0.3 + 0.7 * rnd()
    typeColors[t * 4 + 3] = 1
  }
  return { positions, velocities, types, forces, minDistances, radii, typeColors }
}

async function main() {
  const env = await getEnv()
  const adapter = await env.source.requestAdapter()
  if (!adapter) { console.error('No WebGPU adapter available'); process.exit(2) }
  const device = await adapter.requestDevice()

  try {
    await loadKernelSource()

    // --- Render pipeline + group-1 bind group build from the shared kernel ---
    console.log('\n[render] render pipeline + bind groups build from the kernel')
    const N = 24000
    const T = 32
    const p = worldParams({
      numParticles: N, typeCount: T,
      attractionK: 32, repulsionK: 32, friction: 0.25, dt: 1 / 60,
    })
    device.pushErrorScope('validation')
    const sim = createRenderSim(device, env, p, seedParticles(N, T, mulberry32(2), p))
    const buildErr = await device.popErrorScope()
    check('render pipeline (vs_main/fs_main, point-list) + group-1 bind group ' +
      'build with no validation error', !buildErr, buildErr ? buildErr.message : '')

    const viewOk = await sim.viewSupported()

    if (viewOk) {
      // --- Real draw + pixel readback (Dawn build with working views) -------
      console.log('\n[render] 4-pass frame draws particles to a texture')
      device.pushErrorScope('validation')
      sim.renderFrame(60)
      const pixels = await sim.readPixels()
      const err = await device.popErrorScope()
      const { lit, coloured } = litStats(pixels)
      check('point render produced lit pixels', lit > 0, `${lit} lit`)
      check('lit pixels carry per-type colour (fs_main output)', coloured > 0,
        `${coloured} coloured`)
      check('no validation error over the full 4-pass run', !err,
        err ? err.message : '')
    } else {
      // --- Dawn-node can't make views; validate the runnable compute loop ----
      skip('point draw + pixel readback',
        '@kmamal/gpu (Dawn) gates texture views behind allow_unsafe_apis; the ' +
        'draw path is validated in a real WebGPU browser and the Flutter web build')
      console.log('\n[render] combined compute loop (clear→bin→interact), 24k/32')
      device.pushErrorScope('validation')
      const t0 = Date.now()
      sim.computeFrame(60)
      const err = await device.popErrorScope()
      const ms = Date.now() - t0
      check('combined compute loop runs with no validation error', !err,
        err ? err.message : '')
      console.log(`    ran 60 frames in ${ms} ms (${(ms / 60).toFixed(2)} ms/frame)`)
    }
    sim.destroy()
  } finally {
    device.destroy?.()
    env.destroy()
  }

  console.log(`\n${passed} passed, ${failed} failed, ${skipped} skipped`)
  process.exit(failed === 0 ? 0 : 1)
}

main().catch((e) => { console.error(e); process.exit(2) })
