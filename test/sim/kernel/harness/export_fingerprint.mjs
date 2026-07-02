// Exports the standalone WGSL kernel's parity fingerprint: it loads the
// byte-identical seeded initial condition (from `tool/parity/export_seed.dart`),
// drives the canonical compute kernel on a real WebGPU device for the parity
// scenario's step budget, and dumps the raw SoA snapshot at each named
// checkpoint as JSON. The Dart parity suite ingests this file
// (`test/parity/wgsl_kernel_parity_test.dart`) and computes the SAME metrics it
// uses for the CPU tier, then compares under the *looser* cross-backend band.
//
// This is the TASK-003 -> TASK-009 handoff: the kernel is parity-checked
// independent of any Flutter platform backend, on the exact `.wgsl` source the
// GPU tiers ship.
//
// Regenerate the committed fixture (needs a WebGPU host, e.g. a Mac with the
// prebuilt Dawn @kmamal/gpu):
//
//   # 1. export the shared seed (byte-identical initial condition)
//   dart run tool/parity/export_seed.dart --seed 42 --particles 3000 --types 32 \
//     > test/sim/kernel/harness/seed.json
//   # 2. run the kernel and dump snapshots
//   cd test/sim/kernel/harness && node export_fingerprint.mjs
//   # -> writes ../../../parity/fixtures/wgsl_kernel_snapshots.json
//
// The scenario constants below MUST match ParityHarnessConfig.defaults() in
// test/parity/parity_harness_support.dart.

import { readFileSync, writeFileSync } from 'node:fs'
import { getEnv, createSim, worldParams } from './harness.mjs'

// --- Scenario (mirror ParityHarnessConfig.defaults()) ---
const SEED = 42
const PARTICLES = 3000
const TYPES = 32
const ATTRACTION_K = 32
const REPULSION_K = 32
const FRICTION = 0.25
const DT = 1 / 60
const TOTAL_STEPS = 240
const CHECKPOINTS = { early: 0, mid: 60, steady: 239 }

const SEED_PATH = new URL('./seed.json', import.meta.url)
const OUT_PATH = new URL(
  '../../../parity/fixtures/wgsl_kernel_snapshots.json',
  import.meta.url,
)

async function main() {
  let seedJson
  try {
    seedJson = JSON.parse(readFileSync(SEED_PATH, 'utf8'))
  } catch (e) {
    console.error(
      `Missing ${SEED_PATH.pathname}. First run:\n` +
        `  dart run tool/parity/export_seed.dart --seed ${SEED} ` +
        `--particles ${PARTICLES} --types ${TYPES} > ` +
        `test/sim/kernel/harness/seed.json`,
    )
    process.exit(1)
  }

  const initial = {
    positions: Float32Array.from(seedJson.positions),
    velocities: Float32Array.from(seedJson.velocities),
    types: Int32Array.from(seedJson.types),
    forces: Float32Array.from(seedJson.forces),
    minDistances: Float32Array.from(seedJson.minDistances),
    radii: Float32Array.from(seedJson.radii),
  }

  const params = worldParams({
    attractionK: ATTRACTION_K,
    repulsionK: REPULSION_K,
    friction: FRICTION,
    dt: DT,
    numParticles: PARTICLES,
    typeCount: TYPES,
  })

  const env = await getEnv()
  const sim = createSim(env.device, env, params, initial)

  const checkpoints = {}
  const byStep = new Map()
  for (const [name, step] of Object.entries(CHECKPOINTS)) {
    if (!byStep.has(step)) byStep.set(step, [])
    byStep.get(step).push(name)
  }

  async function snapshot() {
    const positions = await sim.readPositions()
    const velocities = await sim.readVelocities()
    return {
      particleCount: PARTICLES,
      positions: Array.from(positions),
      velocities: Array.from(velocities),
      types: Array.from(seedJson.types),
    }
  }

  // Checkpoint at step 0 captures the seeded state before any step.
  for (const name of byStep.get(0) ?? []) checkpoints[name] = await snapshot()
  for (let s = 1; s < TOTAL_STEPS; s++) {
    sim.step(1)
    for (const name of byStep.get(s) ?? []) checkpoints[name] = await snapshot()
  }

  sim.destroy()

  const out = {
    label: 'wgsl-kernel',
    seed: SEED,
    particleCount: PARTICLES,
    typeCount: TYPES,
    attractionK: ATTRACTION_K,
    repulsionK: REPULSION_K,
    friction: FRICTION,
    checkpoints,
  }
  writeFileSync(OUT_PATH, JSON.stringify(out))
  console.log(`Wrote ${OUT_PATH.pathname}`)
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})
