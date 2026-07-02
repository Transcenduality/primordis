/// Load-time adaptation of the canonical WGSL kernel for the minigpu
/// (Dawn/wgpu-over-Metal via `dart:ffi`) host — PRIMORDIS-TASK-801.
///
/// `lib/sim/kernel/primordis.wgsl` stays the single, verbatim source of truth
/// (PRIMORDIS-ADR-003). minigpu's public Dart API, however, imposes three
/// host-side constraints the browser WebGPU API does not (verified against
/// minigpu 1.5.x source, `compute_shader.cpp`):
///
/// 1. **Compute entry point is hardcoded to `main`** — one pipeline per
///    `ComputeShader`, entry `"main"`. The kernel declares three named passes
///    (`clearBins` / `scatterBins` / `interact`), so each pass gets its own
///    module with that pass renamed to `main`.
/// 2. **No uniform-buffer bindings** — `setUniformBuffer` exists only in the
///    C++ internals; every Dart-visible binding is a storage buffer. The
///    `Params` block is sixteen 4-byte scalars, so its uniform and storage
///    layouts are byte-identical; rebinding it as storage is safe.
/// 3. **Every buffer binding is laid out `read_write`** — the bind-group
///    layout builder hardcodes `WGPUBufferBindingType_Storage`, and WebGPU
///    validation requires the shader's declared access mode to match, so the
///    kernel's `var<storage, read>` views must become `read_write` here.
///    `std430` layout is unaffected; the kernel never writes those buffers.
/// 4. **≤ 8 storage buffers per compute stage** — with the `Params` uniform
///    forced to storage by (2), the kernel's nine compute bindings become
///    nine storage buffers, exceeding WebGPU's default
///    `maxStorageBuffersPerShaderStage` limit of 8 (minigpu requests the
///    device with default limits; verified empirically — Dawn rejects the
///    `interact` pipeline). The three read-only per-type-pair matrices
///    (`forces` / `minDistances` / `radii`, each `typeCount²` f32) are
///    therefore merged into ONE buffer at binding 4 with fixed offsets
///    `0 / n² / 2n²` (`n = params.typeCount`); bindings 5 and 6 are left
///    vacant so the remaining binding numbers stay identical to the
///    canonical kernel. The host uploads the concatenation
///    `[forces, minDistances, radii]` ([mergedMatrices]).
///
/// The transform below is deterministic string surgery on compute-group
/// (`@group(0)`) declarations, three indexed-read rewrites, and a single
/// entry-point rename. The physics expressions, pass structure, and atomics
/// are untouched. It is pure Dart, GPU-free, and unit-tested; if minigpu
/// later exposes entry points / uniform / read-only bindings upstream, this
/// adapter shrinks or disappears.
///
/// Render-group (`@group(1)`) declarations and the vertex/fragment stages are
/// left untouched: they are not statically referenced by any compute entry
/// point, so they impose no compute bind-group requirements.
library;

import 'dart:typed_data';

import 'package:primordis/sim/kernel/kernel_source.dart';

/// Matches a compute-group buffer declaration line, capturing the
/// address-space/access part, e.g. `var<uniform>` or `var<storage, read>`.
final RegExp _computeGroupVar = RegExp(
  r'(@group\(0\)\s*@binding\(\d+\)\s*)var<[^>]+>',
);

/// The binding slot carrying the merged `[forces, minDistances, radii]`
/// buffer (constraint 4 above) — reuses the canonical `forces` slot.
const int kMergedMatricesBinding = KernelBindings.forces;

/// Declaration for the merged matrices buffer (replaces the `forces`
/// declaration; the `minDistances` / `radii` declarations are removed and
/// their binding numbers left vacant).
const String _mergedMatricesDecl =
    '@group(0) @binding($kMergedMatricesBinding) '
    'var<storage, read_write> mergedMatrices : array<f32>;';

/// Matches the three canonical matrix declarations by buffer name.
RegExp _matrixDecl(int binding, String name) => RegExp(
      '@group\\(0\\)\\s*@binding\\($binding\\)\\s*var<[^>]+>\\s*'
      '$name\\s*:\\s*array<f32>;',
    );

/// Concatenates the three flattened matrices in the merged-buffer layout the
/// adapted kernel indexes: `forces` at 0, `minDistances` at n², `radii` at
/// 2n² (each of length n²).
Float32List mergedMatrices(
  Float32List forces,
  Float32List minDistances,
  Float32List radii,
) {
  assert(
    forces.length == minDistances.length && forces.length == radii.length,
    'matrix buffers must be equally sized '
    '(${forces.length}/${minDistances.length}/${radii.length})',
  );
  final out = Float32List(3 * forces.length);
  out.setAll(0, forces);
  out.setAll(forces.length, minDistances);
  out.setAll(2 * forces.length, radii);
  return out;
}

/// Derives the single-entry-point, storage-only module for [entryPoint] from
/// the canonical kernel [source].
///
/// [entryPoint] must be one of [KernelEntryPoints.computePasses]. Throws
/// [ArgumentError] if the entry point is unknown or absent from [source], and
/// [StateError] if [source] already declares `fn main` (which would collide
/// with the rename).
String adaptKernelForMinigpu(String source, String entryPoint) {
  if (!KernelEntryPoints.computePasses.contains(entryPoint)) {
    throw ArgumentError.value(
      entryPoint,
      'entryPoint',
      'not a compute pass (expected one of '
          '${KernelEntryPoints.computePasses})',
    );
  }
  if (RegExp(r'\bfn\s+main\s*\(').hasMatch(source)) {
    throw StateError(
      'kernel source already declares `fn main`; entry-point rename would '
      'collide',
    );
  }
  final entryPattern = RegExp('\\bfn\\s+$entryPoint\\s*\\(');
  if (!entryPattern.hasMatch(source)) {
    throw ArgumentError.value(
      entryPoint,
      'entryPoint',
      'not found in kernel source',
    );
  }

  // (4): merge the three matrix buffers into one at binding 4 so the compute
  // stage needs at most 8 storage buffers. Declarations first…
  var merged = source
      .replaceFirst(
        _matrixDecl(KernelBindings.forces, 'forces'),
        _mergedMatricesDecl,
      )
      .replaceFirst(_matrixDecl(KernelBindings.minDistances, 'minDistances'), '')
      .replaceFirst(_matrixDecl(KernelBindings.radii, 'radii'), '');
  for (final name in ['forces', 'minDistances', 'radii']) {
    if (RegExp('@binding\\(\\d+\\)\\s*var<[^>]+>\\s*$name\\b')
        .hasMatch(merged)) {
      throw StateError('matrix declaration `$name` was not merged');
    }
  }
  // …then the three indexed reads. `n² = params.typeCount * params.typeCount`
  // is u32 arithmetic, same as the index expressions.
  merged = merged
      .replaceAllMapped(
        RegExp(r'\bforces\[([^\]]+)\]'),
        (m) => 'mergedMatrices[${m[1]}]',
      )
      .replaceAllMapped(
        RegExp(r'\bminDistances\[([^\]]+)\]'),
        (m) =>
            'mergedMatrices[params.typeCount * params.typeCount + (${m[1]})]',
      )
      .replaceAllMapped(
        RegExp(r'\bradii\[([^\]]+)\]'),
        (m) =>
            'mergedMatrices[2u * params.typeCount * params.typeCount + '
            '(${m[1]})]',
      );

  // (1)+(2)+(3): every remaining @group(0) binding becomes read_write storage.
  final storageOnly = merged.replaceAllMapped(
    _computeGroupVar,
    (m) => '${m[1]}var<storage, read_write>',
  );

  // Rename the requested pass to `main` (minigpu's hardcoded entry point).
  return storageOnly.replaceFirst(entryPattern, 'fn main(');
}
