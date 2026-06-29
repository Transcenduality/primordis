// Standalone validation suite for the Primordis WGSL kernel (PRIMORDIS-TASK-003).
//
// Runs the real kernel on a WebGPU runtime and asserts every behaviour the task
// calls out: a translator compile check, binning correctness, toroidal
// minimum-image, the three force regimes, matrix asymmetry, slider monotonicity,
// and a 24k/32-type smoke run. Exits non-zero on any failure.
//
//   node run.mjs          # full suite (Node + @kmamal/gpu / Dawn / Tint)
//   node run.mjs --smoke  # only the 24k smoke run
//
// Naga (wgpu/browser) translator parity is NOT asserted here — that cross-
// translator comparison is owned by PRIMORDIS-TASK-017. This harness validates
// the source under whatever runtime executes it (Tint, via Dawn, in Node).

import {
  getEnv, createSim, worldParams, loadKernelSource, MAX_BIN_PARTICLES,
} from './harness.mjs'

let passed = 0
let failed = 0
function check(name, cond, detail = '') {
  if (cond) { passed++; console.log(`  ✓ ${name}`) }
  else { failed++; console.log(`  ✗ ${name}${detail ? ` — ${detail}` : ''}`) }
}
const approx = (a, b, rel = 0.02, abs = 1e-7) =>
  Math.abs(a - b) <= Math.max(abs, rel * Math.abs(b))

// Deterministic PRNG so the smoke run is reproducible across machines.
function mulberry32(seed) {
  let a = seed >>> 0
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

const cellOf = (x, y, p) => {
  const cx = Math.min(Math.max(Math.floor(x / p.binSize), 0), p.gridWidth - 1)
  const cy = Math.min(Math.max(Math.floor(y / p.binSize), 0), p.gridHeight - 1)
  return cy * p.gridWidth + cx
}

async function main() {
  const onlySmoke = process.argv.includes('--smoke')
  const env = await getEnv()
  const adapter = await env.source.requestAdapter()
  if (!adapter) { console.error('No WebGPU adapter available'); process.exit(2) }
  const device = await adapter.requestDevice()
  device.addEventListener?.('uncapturederror', (e) =>
    check('no uncaptured device error', false, e.error?.message ?? 'error'))

  try {
    // Load the WGSL once (async: disk on Node, fetch in a browser); every
    // createSim() below reuses the memoised source synchronously.
    const code = await loadKernelSource()

    // --- Translator compile check (Tint, via Dawn) -------------------------
    console.log('\n[compile] Tint (Dawn) shader-module validation')
    const mod = device.createShaderModule({ code })
    const info = await mod.getCompilationInfo()
    const errors = info.messages.filter((m) => m.type === 'error')
    for (const m of info.messages) console.log(`    [${m.type}] ${m.message}`)
    check('kernel compiles under Tint with no errors', errors.length === 0,
      `${errors.length} error(s)`)
    console.log('    (Naga / wgpu parity is deferred to PRIMORDIS-TASK-017)')

    if (!onlySmoke) {
      await testBinning(device, env)
      await testMinImage(device, env)
      await testForceRegimes(device, env)
      await testAsymmetry(device, env)
      await testSliders(device, env)
    }
    await testSmoke(device, env)
  } finally {
    device.destroy?.()
    env.destroy()
  }

  console.log(`\n${passed} passed, ${failed} failed`)
  process.exit(failed === 0 ? 0 : 1)
}

// --- Binning correctness ---------------------------------------------------
async function testBinning(device, env) {
  console.log('\n[binning] scatter 24,000 particles into the 11x7 grid')
  const N = 24000
  const p = worldParams({ numParticles: N, typeCount: 1, dt: 0, friction: 1 })
  const rnd = mulberry32(12345)
  const positions = new Float32Array(N * 2)
  for (let i = 0; i < N; i++) {
    positions[i * 2] = rnd() * p.worldWidth
    positions[i * 2 + 1] = rnd() * p.worldHeight
  }
  const sim = createSim(device, env, p, {
    positions,
    velocities: new Float32Array(N * 2),
    types: new Uint32Array(N),
    forces: new Float32Array(1),
    minDistances: new Float32Array(1),
    radii: new Float32Array(1),
  })
  sim.step(1) // dt=0 so positions are unchanged after the frame
  const counts = await sim.readBinCounts()
  const index = await sim.readBinParticles()
  const pos = await sim.readPositions()

  // CPU re-bin from the (unchanged) positions.
  const cpu = new Uint32Array(p.numBins)
  for (let i = 0; i < N; i++) cpu[cellOf(pos[i * 2], pos[i * 2 + 1], p)]++

  let countsMatch = true
  let totalCount = 0
  for (let c = 0; c < p.numBins; c++) {
    if (counts[c] !== cpu[c]) countsMatch = false
    totalCount += counts[c]
  }
  check('every particle counted exactly once (sum == N)', totalCount === N,
    `${totalCount} != ${N}`)
  check('GPU bin counts equal a CPU re-bin', countsMatch)

  // Every written index entry maps back to its own cell.
  let written = 0
  let allCorrect = true
  let expectedWritten = 0
  for (let c = 0; c < p.numBins; c++) {
    const w = Math.min(counts[c], MAX_BIN_PARTICLES)
    expectedWritten += w
    for (let s = 0; s < w; s++) {
      const j = index[c * MAX_BIN_PARTICLES + s]
      written++
      if (j >= N || cellOf(pos[j * 2], pos[j * 2 + 1], p) !== c) allCorrect = false
    }
  }
  check('clamped bin counts == number of binned particles',
    written === expectedWritten, `${written} != ${expectedWritten}`)
  check("every binned particle's recorded cell matches its position", allCorrect)
  sim.destroy()
}

// --- Toroidal minimum-image ------------------------------------------------
async function testMinImage(device, env) {
  console.log('\n[min-image] interaction wraps across the world seams')
  // 1 type, repulsion within min_dist; v0=0, friction=1, small dt -> v == f*dt.
  const base = {
    velocities: new Float32Array(4),
    types: new Uint32Array([0, 0]),
    forces: new Float32Array([0.5]),
    minDistances: new Float32Array([20]),
    radii: new Float32Array([80]),
  }
  const dt = 1e-3
  // Repulsion magnitude at dist 10, min 20: |0.5|*5*(1-10/20) = 1.25.
  const expected = 1.25 * dt

  // x-seam: A near x=0, B near x=worldWidth.
  {
    const p = worldParams({ numParticles: 2, typeCount: 1, dt })
    const sim = createSim(device, env, p, {
      ...base, positions: new Float32Array([5, 360, 1075, 360]),
    })
    sim.step(1)
    const v = await sim.readVelocities()
    check('x-seam: A is pushed in +x (wrapped vector used)', v[0] > 0,
      `vx=${v[0]}`)
    check('x-seam: force magnitude matches the wrapped distance',
      approx(v[0], expected) && approx(v[1], 0, 0.02, 1e-5),
      `v=(${v[0]}, ${v[1]}) expected (${expected}, 0)`)
    sim.destroy()
  }
  // y-seam: A near y=0, B near y=worldHeight.
  {
    const p = worldParams({ numParticles: 2, typeCount: 1, dt })
    const sim = createSim(device, env, p, {
      ...base, positions: new Float32Array([540, 5, 540, 715]),
    })
    sim.step(1)
    const v = await sim.readVelocities()
    check('y-seam: A is pushed in +y (wrapped vector used)', v[1] > 0,
      `vy=${v[1]}`)
    check('y-seam: force magnitude matches the wrapped distance',
      approx(v[1], expected) && approx(v[0], 0, 0.02, 1e-5),
      `v=(${v[0]}, ${v[1]}) expected (0, ${expected})`)
    sim.destroy()
  }
}

// --- Force-regime boundaries ----------------------------------------------
async function testForceRegimes(device, env) {
  console.log('\n[force] three regimes for a 2-particle / 2-type setup')
  const dt = 1e-3
  // A=type0, B=type1. forces[A<-B]=forces[1]=0.5; forces[B<-A]=forces[2]=0 so B
  // stays put and A reads an unmoved neighbour (exact force extraction).
  const mk = (dist) => ({
    positions: new Float32Array([500, 360, 500 + dist, 360]),
    velocities: new Float32Array(4),
    types: new Uint32Array([0, 1]),
    forces: new Float32Array([0, 0.5, 0, 0]),
    minDistances: new Float32Array([20, 20, 20, 20]),
    radii: new Float32Array([80, 80, 80, 80]),
  })
  const run = async (dist) => {
    const p = worldParams({ numParticles: 2, typeCount: 2, dt })
    const sim = createSim(device, env, p, mk(dist))
    sim.step(1)
    const v = await sim.readVelocities()
    sim.destroy()
    return v[0] / dt // f_A.x
  }

  const fRep = await run(10) // dist<min: repulsion (negative x, toward -x)
  const expRep = -(0.5 * 5 * (1 - 10 / 20))
  check('dist < min_dist: 5x |force| repulsion', approx(fRep, expRep),
    `${fRep} vs ${expRep}`)

  const fAtt = await run(50) // min<=dist<rad: signed attraction
  const expAtt = 0.5 * (1 - 50 / 80)
  check('min_dist <= dist < radius: signed linear attraction',
    approx(fAtt, expAtt), `${fAtt} vs ${expAtt}`)

  const fZero = await run(90) // dist>=rad: no contribution
  check('dist >= radius: zero force', approx(fZero, 0, 0.02, 1e-4),
    `${fZero}`)
}

// --- Matrix asymmetry ------------------------------------------------------
async function testAsymmetry(device, env) {
  console.log('\n[asymmetry] forces[i][j] != forces[j][i] -> different forces')
  const dt = 1e-3
  const p = worldParams({ numParticles: 2, typeCount: 2, dt })
  // forces[A<-B] = forces[1] = 0.5 ; forces[B<-A] = forces[2] = 0.2.
  const sim = createSim(device, env, p, {
    positions: new Float32Array([500, 360, 550, 360]),
    velocities: new Float32Array(4),
    types: new Uint32Array([0, 1]),
    forces: new Float32Array([0, 0.5, 0.2, 0]),
    minDistances: new Float32Array([5, 5, 5, 5]),
    radii: new Float32Array([80, 80, 80, 80]),
  })
  sim.step(1)
  const v = await sim.readVelocities()
  sim.destroy()
  const fA = Math.abs(v[0] / dt) // magnitude on A (uses forces[0][1]=0.5)
  const fB = Math.abs(v[2] / dt) // magnitude on B (uses forces[1][0]=0.2)
  check('force on A differs from force on B', !approx(fA, fB),
    `|fA|=${fA} |fB|=${fB}`)
  check('magnitudes follow the i->j matrix entries (ratio 0.5:0.2)',
    approx(fA / fB, 0.5 / 0.2, 0.05), `ratio=${fA / fB}`)
}

// --- Slider monotonicity ---------------------------------------------------
async function testSliders(device, env) {
  console.log('\n[sliders] attraction K / repulsion K / drift respond monotonically')
  const dt = 1e-3
  const attract = async (K) => {
    const p = worldParams({ numParticles: 2, typeCount: 2, dt, attractionK: K })
    const sim = createSim(device, env, p, {
      positions: new Float32Array([500, 360, 550, 360]),
      velocities: new Float32Array(4),
      types: new Uint32Array([0, 1]),
      forces: new Float32Array([0, 0.5, 0, 0]),
      minDistances: new Float32Array([5, 5, 5, 5]),
      radii: new Float32Array([80, 80, 80, 80]),
    })
    sim.step(1)
    const v = await sim.readVelocities(); sim.destroy()
    return Math.abs(v[0])
  }
  const a1 = await attract(1), a2 = await attract(2), a4 = await attract(4)
  check('higher Attraction K -> stronger pull', a1 < a2 && a2 < a4,
    `${a1} < ${a2} < ${a4}`)
  check('attraction scales linearly with K', approx(a2 / a1, 2, 0.05) &&
    approx(a4 / a1, 4, 0.05), `x2=${a2 / a1} x4=${a4 / a1}`)

  const repulse = async (K) => {
    const p = worldParams({ numParticles: 2, typeCount: 2, dt, repulsionK: K })
    const sim = createSim(device, env, p, {
      positions: new Float32Array([500, 360, 510, 360]),
      velocities: new Float32Array(4),
      types: new Uint32Array([0, 1]),
      forces: new Float32Array([0, 0.5, 0, 0]),
      minDistances: new Float32Array([20, 20, 20, 20]),
      radii: new Float32Array([80, 80, 80, 80]),
    })
    sim.step(1)
    const v = await sim.readVelocities(); sim.destroy()
    return Math.abs(v[0])
  }
  const r1 = await repulse(1), r2 = await repulse(2), r4 = await repulse(4)
  check('higher Repulsion K -> stronger push', r1 < r2 && r2 < r4,
    `${r1} < ${r2} < ${r4}`)

  // Drift/friction: isolated particle, force-free, v *= friction.
  const drift = async (friction) => {
    const p = worldParams({ numParticles: 1, typeCount: 1, dt: 1, friction })
    const sim = createSim(device, env, p, {
      positions: new Float32Array([540, 360]),
      velocities: new Float32Array([10, 0]),
      types: new Uint32Array([0]),
      forces: new Float32Array([0]),
      minDistances: new Float32Array([1]),
      radii: new Float32Array([1]),
    })
    sim.step(1)
    const v = await sim.readVelocities(); sim.destroy()
    return v[0]
  }
  const d1 = await drift(0.1), d5 = await drift(0.5), d9 = await drift(0.9)
  check('higher drift retains more velocity (v *= friction)',
    d1 < d5 && d5 < d9, `${d1} < ${d5} < ${d9}`)
  check('retained velocity == v0 * friction', approx(d1, 1) && approx(d5, 5) &&
    approx(d9, 9), `${d1}, ${d5}, ${d9}`)
}

// --- 24k / 32-type smoke run ----------------------------------------------
async function testSmoke(device, env) {
  const FRAMES = 300
  console.log(`\n[smoke] 24,000 particles / 32 types for ${FRAMES} frames`)
  const N = 24000
  const T = 32
  const p = worldParams({
    numParticles: N, typeCount: T,
    attractionK: 32, repulsionK: 32, friction: 0.25, dt: 1 / 60,
  })
  const rnd = mulberry32(98765)
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
  const sim = createSim(device, env, p,
    { positions, velocities, types, forces, minDistances, radii })

  device.pushErrorScope?.('validation')
  const t0 = Date.now()
  sim.step(FRAMES)
  const pos = await sim.readPositions()
  const vel = await sim.readVelocities()
  const err = await device.popErrorScope?.()
  const ms = Date.now() - t0

  let nan = 0
  let oob = 0
  for (let i = 0; i < N; i++) {
    const x = pos[i * 2], y = pos[i * 2 + 1]
    const vx = vel[i * 2], vy = vel[i * 2 + 1]
    if (!Number.isFinite(x) || !Number.isFinite(y) ||
        !Number.isFinite(vx) || !Number.isFinite(vy)) nan++
    if (x < 0 || x >= p.worldWidth || y < 0 || y >= p.worldHeight) oob++
  }
  check('no NaN/Inf in positions or velocities after 300 frames', nan === 0,
    `${nan} bad particles`)
  check('all particles stay within the toroidal world', oob === 0,
    `${oob} out of bounds`)
  check('no device validation error during the run', !err,
    err ? err.message : '')
  console.log(`    ran ${FRAMES} frames in ${ms} ms (${(ms / FRAMES).toFixed(2)} ms/frame)`)
  sim.destroy()
}

main().catch((e) => { console.error(e); process.exit(2) })
