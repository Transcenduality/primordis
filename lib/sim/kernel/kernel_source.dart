import 'package:flutter/services.dart' show rootBundle;
import 'package:primordis/shared/constants/primordis_config.dart';

/// Thin accessor for the canonical WGSL kernel ([`primordis.wgsl`]) and the
/// dispatch-geometry / bind-group constants both GPU backends share.
///
/// The WGSL source itself is the single source of truth for the physics
/// ([PRIMORDIS-ADR-003]); it lives as a Flutter asset (one copy, no Dart
/// duplicate) and is loaded as a string via [loadKernelSource]. Everything a
/// backend needs to *drive* that source without re-deriving it — the workgroup
/// size, the per-bin cap, the entry-point names, and the bind-group/binding
/// indices — is declared here so the web ([PRIMORDIS-TASK-004]) and macOS
/// ([PRIMORDIS-TASK-011]) backends can never drift from each other or from the
/// shader.
///
/// This file is platform-neutral: it only references `package:flutter/services`
/// (`rootBundle`) and constants. It contains no device/pipeline creation, no
/// JS-interop, and no FFI — those live in the concrete backends behind the
/// `SimBackend` seam ([PRIMORDIS-ADR-001]).

/// Canonical kernel constants shared by every backend.
abstract final class KernelConfig {
  /// Asset key for the WGSL kernel (also the path under `lib/`, registered in
  /// `pubspec.yaml`). Loaded via [loadKernelSource].
  static const String assetPath = 'lib/sim/kernel/primordis.wgsl';

  /// `@workgroup_size(N)` chosen once for all three compute passes (mirrors the
  /// reference `COMPUTE_GROUP_SIZE`). Backends dispatch [computeWorkgroups]
  /// groups and may set the WGSL `WORKGROUP_SIZE` override to this value; a unit
  /// test asserts the shader's override default matches.
  static const int workgroupSize = 256;

  /// Per-bin particle-index capacity — the WGSL `MAX_BIN_PARTICLES` override.
  /// Mirrors [PrimordisConfig.maxBinParticles] so host buffer sizing and the
  /// shader's drop-on-overflow cap agree.
  static const int maxBinParticles = PrimordisConfig.maxBinParticles;
}

/// Names of the kernel's entry points, exactly as declared in `primordis.wgsl`.
///
/// Backends pass these as `entryPoint` when building pipelines. A unit test
/// asserts each name is actually present in the loaded source.
abstract final class KernelEntryPoints {
  /// Pass 1 — zero the bin counters.
  static const String clearBins = 'clearBins';

  /// Pass 2 — atomic-scatter particles into the spatial grid.
  static const String scatterBins = 'scatterBins';

  /// Pass 3 — neighbour interaction + Euler integrate + wrap.
  static const String interact = 'interact';

  /// Point-render vertex stage.
  static const String vertexMain = 'vs_main';

  /// Point-render fragment stage.
  static const String fragmentMain = 'fs_main';

  /// The three compute passes, in the order a frame dispatches them.
  static const List<String> computePasses = <String>[
    clearBins,
    scatterBins,
    interact,
  ];
}

/// Bind-group and binding indices, matching the `@group(g) @binding(b)`
/// attributes in `primordis.wgsl`.
///
/// Compute resources are in group [computeGroup]; the read-only render views are
/// in group [renderGroup]. Backends bind the same physical position/type buffers
/// to both groups (the compute group via read_write bindings, the render group
/// via read views — a vertex stage may not use a read_write storage buffer).
abstract final class KernelBindings {
  /// Bind group holding the compute pass resources.
  static const int computeGroup = 0;

  static const int params = 0;
  static const int positions = 1;
  static const int velocities = 2;
  static const int types = 3;
  static const int forces = 4;
  static const int minDistances = 5;
  static const int radii = 6;
  static const int binCounts = 7;
  static const int binParticles = 8;

  /// Bind group holding the read-only render resources.
  static const int renderGroup = 1;

  static const int renderParams = 0;
  static const int renderPositions = 1;
  static const int renderTypes = 2;
  static const int typeColors = 3;
}

/// Number of workgroups needed to cover [itemCount] invocations at
/// [workgroupSize] (`ceil(itemCount / workgroupSize)`).
///
/// Used for all three dispatches: the clear pass over `numBins`, the bin and
/// interaction passes over `numParticles`.
int computeWorkgroups(
  int itemCount, {
  int workgroupSize = KernelConfig.workgroupSize,
}) {
  assert(itemCount >= 0, 'itemCount must be non-negative, got $itemCount');
  assert(workgroupSize > 0, 'workgroupSize must be positive, got $workgroupSize');
  return (itemCount + workgroupSize - 1) ~/ workgroupSize;
}

String? _cachedSource;

/// Loads the WGSL kernel source string from the bundled asset (cached).
///
/// Async because asset loading is async (`rootBundle.loadString`). The result is
/// memoised so repeated backend/init calls don't re-read the asset. Call
/// [resetKernelSourceCache] in tests that need a fresh load.
Future<String> loadKernelSource() async =>
    _cachedSource ??= await rootBundle.loadString(KernelConfig.assetPath);

/// Clears the [loadKernelSource] cache (test seam).
void resetKernelSourceCache() => _cachedSource = null;
