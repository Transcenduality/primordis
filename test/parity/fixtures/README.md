# Parity fixtures

Committed reference baselines the parity harness measures backends against. See
[`tool/parity/README.md`](../../../tool/parity/README.md) for the full harness.

Parity is **statistical, never bit-exact** — these fixtures are aggregate
statistics (a checkpoint time series of `FrameMetrics`), never particle
positions or pixels.

## Files

| File | Source | Regenerate with | Needs |
|------|--------|-----------------|-------|
| `cpu_reference_default.json` | Deterministic CPU tier, `ParityHarnessConfig.defaults()` | `dart run tool/parity/generate_cpu_reference.dart` | nothing (bit-stable) |
| `cpu_reference_highdrift.json` | Deterministic CPU tier, `ParityHarnessConfig.highDrift()` | same as above | nothing |
| `wgsl_kernel_snapshots.json` *(optional)* | Standalone WGSL kernel, raw SoA snapshots | `node test/sim/kernel/harness/export_fingerprint.mjs` | WebGPU host (Dawn) |
| `py_reference_snapshots.json` *(optional)* | Original `Primordis.py` shaders, raw SoA snapshots | `python3 tool/parity/capture_reference.py ...` | OpenGL 4.3 |

The two `cpu_reference_*.json` fixtures are the CI-checked baselines: the CPU
tier reproduces them exactly, and they are the reference the harness bands are
built around. The two optional `*_snapshots.json` fixtures let the WGSL kernel
and the Python reference be parity-checked when a GPU host is available; their
parity tests **skip** cleanly when the fixture is absent, so the pure-Dart CI job
stays green without a GPU.

## The scenario (single source of truth)

`ParityHarnessConfig.defaults()` in
[`test/parity/parity_harness_support.dart`](../parity_harness_support.dart)
defines the shared seed, particle/type counts, slider values, step budget, and
checkpoints. **Every** regeneration path (CPU, WGSL, Python) must use matching
constants; the seed's concrete initial arrays are exported once by
`tool/parity/export_seed.dart` and loaded by the non-Dart backends so all
backends start byte-identical.

## When to regenerate

Only when the shared seed/params or the metric definitions **intentionally**
change. Regenerate, commit the JSON, and review the diff — a large unexpected
diff means a metric or seeding change you did not intend.
