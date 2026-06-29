// Primordis — the single, canonical WGSL compute kernel.
//
// THE source of truth for the simulation physics (PRIMORDIS-ADR-003). This exact
// string is handed, unchanged, to:
//   • browser WebGPU via dart:js_interop   (web backend, PRIMORDIS-TASK-004), and
//   • Dawn/wgpu-over-Metal via dart:ffi     (macOS backend, PRIMORDIS-TASK-011).
// The optional Metal/MSL plugin (PRIMORDIS-TASK-013) must be a faithful 1:1
// transliteration of *this* file — it is a derived artefact, never a peer.
//
// This file is a faithful port of the GLSL `#version 430` compute shaders in
// `Primordis.py` (clear bin counts / atomic-scatter binning / interaction +
// Euler integrate), plus the point-render vertex/fragment stages. It contains NO
// backend-specific code: no device/pipeline creation, no JS-interop, no FFI. The
// host marshals data into the buffers declared below using the layout fixed in
// `buffer_layout.dart` and `sim_marshalling.dart`; dispatch geometry and the
// bind-group/binding map are declared once in `kernel_source.dart`.
//
// GLSL -> WGSL mapping (canonical, per ADR-003):
//   layout(std430) buffer        -> var<storage, read_write> / var<storage, read>
//   atomicAdd(bins[i], 1)        -> atomicAdd(&binCounts[i], 1u)   (returns old value)
//   reading the bin counters     -> atomicLoad(&binCounts[i])
//   zeroing the bin counters     -> atomicStore(&binCounts[i], 0u)
//   layout(local_size_x = N)     -> @workgroup_size(WORKGROUP_SIZE)
//   gl_GlobalInvocationID        -> @builtin(global_invocation_id)
//
// Atomics rules (the permanent authoring tax that keeps Tint and Naga in
// agreement): `binCounts` is `array<atomic<u32>>` and is touched ONLY through
// atomic builtins (atomicAdd / atomicLoad / atomicStore). It is never read or
// written through a non-atomic alias — that is undefined in WGSL and is exactly
// the class of bug Tint/Naga diverge on (validated in PRIMORDIS-TASK-017).
//
// Determinism is explicitly NOT a goal. The atomic-scatter binning is racy and
// the interaction pass reads neighbour positions that other invocations may be
// updating in the same pass — both faithful to the single-buffered reference.
// "Faithful" means visually/statistically equivalent (PRIMORDIS-TASK-009), never
// bit-exact.

// ---------------------------------------------------------------------------
// Pipeline-overridable constants.
//
// Defaults are the canonical values; both backends read the same numbers from
// `kernel_source.dart` (`kWorkgroupSize`, `kMaxBinParticles`) and may set them
// as pipeline-override constants, so dispatch geometry and the per-bin cap can
// never drift between host and shader. A unit test asserts the defaults here
// mirror the Dart constants.
// ---------------------------------------------------------------------------

override WORKGROUP_SIZE: u32 = 256u;

// Per-bin particle-index capacity. Over-cap particles are dropped from the bin
// index (they still exist and still move; they are simply invisible to
// neighbours that frame) — matching the reference's single-buffered behaviour.
override MAX_BIN_PARTICLES: u32 = 512u;

// ---------------------------------------------------------------------------
// Uniform block — mirrors `SimMarshalling` (sim_marshalling.dart): sixteen
// 4-byte slots, 64 bytes total, slot k at byte offset k*4. Field order and types
// MUST match the packed block the host uploads. Slots 13..15 are reserved pad.
// ---------------------------------------------------------------------------

struct Params {
  attractionK : f32,   // slot 0  — K_attraction multiplier
  repulsionK  : f32,   // slot 1  — K_repulsion multiplier
  friction    : f32,   // slot 2  — drift/friction (v *= friction)
  dt          : f32,   // slot 3  — per-tick delta time (seconds)
  worldWidth  : f32,   // slot 4
  worldHeight : f32,   // slot 5
  maxRadius   : f32,   // slot 6  — interaction cutoff
  binSize     : f32,   // slot 7  — == maxRadius
  gridWidth   : u32,   // slot 8  — grid columns
  gridHeight  : u32,   // slot 9  — grid rows
  numParticles: u32,   // slot 10
  numBins     : u32,   // slot 11 — == gridWidth * gridHeight
  typeCount   : u32,   // slot 12 — matrix side length
  _pad0       : u32,   // slot 13
  _pad1       : u32,   // slot 14
  _pad2       : u32,   // slot 15
}

// ---------------------------------------------------------------------------
// Compute bind group (group 0). Binding indices are declared in
// `kernel_source.dart` (kBindParams, kBindPositions, …) and must agree.
// ---------------------------------------------------------------------------

@group(0) @binding(0) var<uniform>                params       : Params;
@group(0) @binding(1) var<storage, read_write>    positions    : array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write>    velocities   : array<vec2<f32>>;
@group(0) @binding(3) var<storage, read>          types        : array<u32>;
@group(0) @binding(4) var<storage, read>          forces       : array<f32>;
@group(0) @binding(5) var<storage, read>          minDistances : array<f32>;
@group(0) @binding(6) var<storage, read>          radii        : array<f32>;
@group(0) @binding(7) var<storage, read_write>    binCounts    : array<atomic<u32>>;
@group(0) @binding(8) var<storage, read_write>    binParticles : array<u32>;

// Toroidal cell wrap — identical to the reference's `wrap(val, max)`:
// `(val + max) % max`. `val` is `cellIndex + {-1,0,1}` with `cellIndex >= 0`, so
// `val + max >= max - 1 >= 0` and the WGSL `%` (remainder, sign of dividend)
// behaves exactly like the GLSL `%` here.
fn wrapCell(val: i32, maxVal: i32) -> i32 {
  return (val + maxVal) % maxVal;
}

// ===========================================================================
// PASS 1 — clear: zero every bin counter before binning each frame.
// GLSL: clear_counts_shader (local_size_x = 256), `bin_counts[i] = 0`.
// ===========================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn clearBins(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i < params.numBins) {
    atomicStore(&binCounts[i], 0u);
  }
}

// ===========================================================================
// PASS 2 — bin: atomic-scatter every particle into the 11x7 toroidal grid.
// GLSL: binning_shader. `atomicAdd` returns the previous value, which is this
// particle's write offset within its bin. Over-cap writes are dropped.
// ===========================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn scatterBins(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numParticles) { return; }

  let p = positions[i];
  // int(p.x / bin_size): WGSL i32() truncates toward zero like GLSL int(); p >= 0
  // so this is floor. Clamp to the valid cell range exactly as the reference.
  var x = i32(p.x / params.binSize);
  var y = i32(p.y / params.binSize);
  x = clamp(x, 0, i32(params.gridWidth) - 1);
  y = clamp(y, 0, i32(params.gridHeight) - 1);
  let binIdx = u32(y) * params.gridWidth + u32(x);

  let offset = atomicAdd(&binCounts[binIdx], 1u);
  if (offset < MAX_BIN_PARTICLES) {
    binParticles[binIdx * MAX_BIN_PARTICLES + offset] = i;
  }
  // offset >= cap: particle is dropped from the bin index (no out-of-bounds
  // write), matching the reference.
}

// ===========================================================================
// PASS 3 — interaction + integrate: scan the 3x3 toroidal neighbour bins,
// apply the two-regime force model, Euler-integrate, and wrap.
// GLSL: interaction_shader.
//
// NOTE (faithful to the reference): the centre cell cx/cy is computed WITHOUT
// clamping — only the binning pass clamps. A particle in the [1056,1080) x-band
// has cx = 11 (one past the last column); `wrapCell` then folds its 3x3 scan to
// columns {10, 0, 1}. Preserve this; do not "fix" it by clamping cx/cy.
// ===========================================================================
@compute @workgroup_size(WORKGROUP_SIZE)
fn interact(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= params.numParticles) { return; }

  let p = positions[i];
  var v = velocities[i];
  var f = vec2<f32>(0.0, 0.0);
  let myType = types[i];

  let cx = i32(p.x / params.binSize);
  let cy = i32(p.y / params.binSize);
  let halfW = params.worldWidth * 0.5;
  let halfH = params.worldHeight * 0.5;
  let gw = i32(params.gridWidth);
  let gh = i32(params.gridHeight);

  for (var dx = -1; dx <= 1; dx = dx + 1) {
    let nx = wrapCell(cx + dx, gw);
    for (var dy = -1; dy <= 1; dy = dy + 1) {
      let ny = wrapCell(cy + dy, gh);
      let binIdx = u32(ny) * params.gridWidth + u32(nx);

      // Read the counter atomically, then cap at the per-bin storage so we only
      // ever read slots that were actually written (the reference relies on
      // undefined out-of-bounds reads for the overflow case; we make it
      // well-defined by reading exactly the capped, valid range).
      let count = min(atomicLoad(&binCounts[binIdx]), MAX_BIN_PARTICLES);
      for (var b = 0u; b < count; b = b + 1u) {
        let j = binParticles[binIdx * MAX_BIN_PARTICLES + b];
        if (j == i) { continue; }

        // Minimum-image displacement on the torus (fold any axis > world/2).
        var d = positions[j] - p;
        if (d.x > halfW) { d.x = d.x - params.worldWidth; }
        else if (d.x < -halfW) { d.x = d.x + params.worldWidth; }
        if (d.y > halfH) { d.y = d.y - params.worldHeight; }
        else if (d.y < -halfH) { d.y = d.y + params.worldHeight; }

        let dist = length(d);
        if (dist > params.maxRadius || dist < 0.1) { continue; }
        let dn = d / dist;

        let otherType = types[j];
        let idx = myType * params.typeCount + otherType;
        let mind = minDistances[idx];
        let rad = radii[idx];
        let forceStrength = forces[idx];

        if (dist < mind) {
          // Short-range repulsion: 5x weighted, magnitude from |force|, always
          // repulsive, linear falloff to the min distance.
          f = f - dn * abs(forceStrength) * 5.0 * (1.0 - dist / mind) * params.repulsionK;
        } else if (dist < rad) {
          // Signed attraction with linear falloff toward the radius.
          f = f + dn * forceStrength * (1.0 - dist / rad) * params.attractionK;
        }
        // dist >= rad: no contribution.
      }
    }
  }

  // Euler integrate — order matters (matches the reference trajectories).
  v = v + f * params.dt;
  v = v * params.friction;
  var np = p + v * params.dt;

  // Wrap both axes onto the torus.
  if (np.x < 0.0) { np.x = np.x + params.worldWidth; }
  else if (np.x >= params.worldWidth) { np.x = np.x - params.worldWidth; }
  if (np.y < 0.0) { np.y = np.y + params.worldHeight; }
  else if (np.y >= params.worldHeight) { np.y = np.y - params.worldHeight; }

  positions[i] = np;
  velocities[i] = v;
}

// ===========================================================================
// Point-render stages (vertex + fragment).
//
// Co-located here for convenience (ADR-003): they draw the particles as
// per-type-coloured points, mapping world space to clip space exactly as the
// reference vertex shader does. The *compositing* of this output (the present
// path, point size, any y-flip, blend state) is BACKEND-SPECIFIC and OUT OF
// SCOPE for this kernel — see PRIMORDIS-ADR-005 and TASK-005 / TASK-012.
//
// Render resources live in their own bind group (group 1) with read-only
// access: a vertex stage may not use a read_write storage buffer, so these are
// separate `read` views the host binds to the same physical position/type
// buffers used by the compute group.
// ===========================================================================

@group(1) @binding(0) var<uniform>           renderParams : Params;
@group(1) @binding(1) var<storage, read>     renderPositions : array<vec2<f32>>;
@group(1) @binding(2) var<storage, read>     renderTypes : array<u32>;
@group(1) @binding(3) var<storage, read>     typeColors : array<vec4<f32>>;

struct VsOut {
  @builtin(position) position : vec4<f32>,
  @location(0) color : vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> VsOut {
  let pos = renderPositions[vid];
  let t = renderTypes[vid];
  var out: VsOut;
  // World -> clip space, identical to the reference (no y-flip here; any flip is
  // a present-path concern).
  out.position = vec4<f32>(
    (pos.x / renderParams.worldWidth) * 2.0 - 1.0,
    (pos.y / renderParams.worldHeight) * 2.0 - 1.0,
    0.0,
    1.0,
  );
  out.color = typeColors[t].rgb;
  return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
  return vec4<f32>(in.color, 1.0);
}
