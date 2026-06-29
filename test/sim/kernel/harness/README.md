# Primordis WGSL kernel — standalone validation harness

Validates the canonical compute kernel
[`lib/sim/kernel/primordis.wgsl`](../../../../lib/sim/kernel/primordis.wgsl) by
running it on a **real WebGPU runtime**, outside Flutter. The `.wgsl` source is
not Dart, so it is not covered by `flutter test`; this harness is its test
(PRIMORDIS-TASK-003 → handed to the parity harness in PRIMORDIS-TASK-009).

The harness loads the kernel file directly (no copy) and drives the three
compute passes (clear → atomic-scatter bin → interaction+integrate), so it
always validates the exact source the GPU backends ship.

## What it checks

- **Translator compile check** — the kernel compiles under **Tint** (Dawn) with
  no errors. (Naga / wgpu cross-translator parity is owned by PRIMORDIS-TASK-017,
  not asserted here.)
- **Binning correctness** — 24,000 particles scattered into the 11×7 grid: every
  particle counted once, GPU bin counts equal a CPU re-bin, and every written
  index entry maps back to its own cell (clamped at `MAX_BIN_PARTICLES` = 512).
- **Toroidal minimum-image** — particles straddling the x and y seams interact
  via the wrapped (shorter) vector, with the expected force magnitude.
- **Force regimes** — a 2-particle / 2-type setup checked against hand-computed
  forces in all three regimes (5× `abs(force)` repulsion, signed linear
  attraction, zero beyond the radius).
- **Asymmetry** — `forces[i][j] != forces[j][i]` produces different forces on
  `i`-from-`j` vs `j`-from-`i`.
- **Slider response** — Attraction K, Repulsion K, and Drift/friction each move
  their term monotonically (and linearly, where expected).
- **24k smoke** — 24,000 particles / 32 types for 300 frames with no NaNs, no
  out-of-bounds positions, and no device validation error.

## Running it

### Node.js (Dawn / Tint)

```sh
cd test/sim/kernel/harness
npm install      # pulls @kmamal/gpu (prebuilt Dawn; x64 + arm-Mac)
npm test         # full suite
npm run smoke    # only the 24k smoke run
```

`@kmamal/gpu` is a dev-only Node dependency of this harness; it is **not** a
dependency of the Flutter app (`node_modules/` is gitignored here).

### Browser (Naga path / the web backend's runtime)

`harness.mjs` is environment-agnostic: it has no `node:*` imports, selects
`navigator.gpu` when not on Node, and fetches the WGSL over HTTP. So the same
buffer/dispatch plumbing can be reused in a WebGPU-capable browser to sanity-
check the **Naga** translator ahead of the formal Tint-vs-Naga parity work in
PRIMORDIS-TASK-017.

`run.mjs` itself is the **Node CLI** (it uses `process` for argv/exit), so a
browser run means a tiny page that imports `harness.mjs` and calls
`loadKernelSource()` + `createSim()` — not loading `run.mjs` directly. Serve the
repo over a secure context so `fetch(KERNEL_URL)` can read the asset, e.g.:

```js
import { getEnv, loadKernelSource, createSim, worldParams } from './harness.mjs'
await loadKernelSource()
const env = await getEnv()
const device = await (await env.source.requestAdapter()).requestDevice()
const sim = createSim(device, env, worldParams({ numParticles: 24000, typeCount: 32 }), initialBuffers)
sim.step(300)
```

## Notes

- Determinism is not a goal: the atomic scatter is intentionally racy and the
  interaction pass reads neighbour positions other invocations may be updating,
  exactly as the single-buffered reference does. Assertions are written to hold
  regardless of scatter order (e.g. force tests keep the neighbour stationary, or
  use a tiny `dt` so neighbour movement within a pass is negligible).
- The uniform-block packing in `harness.mjs` mirrors `SimMarshalling`
  (`lib/sim/sim_marshalling.dart`); the bind-group/binding indices mirror
  `KernelBindings` (`lib/sim/kernel/kernel_source.dart`).
