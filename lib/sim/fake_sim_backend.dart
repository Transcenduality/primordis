import 'package:primordis/shared/constants/primordis_config.dart';
import 'package:primordis/sim/models/seeded_sim.dart';
import 'package:primordis/sim/models/sim_capabilities.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/sim_backend.dart';
import 'package:primordis/sim/sim_seeder.dart';

/// The names recorded in [FakeSimBackend.calls], so tests can assert ordering
/// without matching on raw strings.
abstract final class FakeSimCall {
  static const String init = 'init';
  static const String seed = 'seed';
  static const String setParams = 'setParams';
  static const String step = 'step';
  static const String present = 'present';
  static const String dispose = 'dispose';
}

/// A no-GPU [SimBackend] that records calls and advances trivial in-memory
/// state, so the UI, frame loop, and providers are testable in CI without
/// WebGPU/FFI/Metal ([PRIMORDIS-ADR-001]).
///
/// It is also the default backend the provider injects, so the app boots and the
/// frame loop runs before any real backend lands; backend selection
/// ([PRIMORDIS-TASK-007] / [PRIMORDIS-TASK-015]) overrides the provider with a
/// concrete GPU/CPU backend later.
///
/// To stay faithful to the real lifecycle, [seed] runs the shared [SimSeeder]
/// and exposes the deterministic result via [lastSeeded].
class FakeSimBackend implements SimBackend {
  /// Ordered log of lifecycle calls (see [FakeSimCall]).
  final List<String> calls = <String>[];

  bool _initialized = false;
  bool _disposed = false;
  int _initCount = 0;
  int _seedCount = 0;
  int _setParamsCount = 0;
  int _stepCount = 0;
  int _presentCount = 0;

  SimSeed? _lastSeed;
  SeededSim? _lastSeeded;
  SimParams? _lastParams;
  double _lastDt = 0;
  double _simulatedTime = 0;

  /// Whether [init] has run (and [dispose] has not).
  bool get isInitialized => _initialized;

  /// Whether [dispose] has run.
  bool get isDisposed => _disposed;

  int get initCount => _initCount;
  int get seedCount => _seedCount;
  int get setParamsCount => _setParamsCount;
  int get stepCount => _stepCount;
  int get presentCount => _presentCount;

  /// The last [SimSeed] passed to [seed].
  SimSeed? get lastSeed => _lastSeed;

  /// The deterministic [SeededSim] materialized by the last [seed] call.
  SeededSim? get lastSeeded => _lastSeeded;

  /// The last [SimParams] passed to [setParams].
  SimParams? get lastParams => _lastParams;

  /// The `dt` of the last [step].
  double get lastDt => _lastDt;

  /// Sum of all `dt`s stepped — the fake's only "physics" state.
  double get simulatedTime => _simulatedTime;

  @override
  SimBackendCapabilities get capabilities => const SimBackendCapabilities(
        isGpuAccelerated: false,
        maxParticles: PrimordisConfig.particleCount,
        defaultParticleCount: PrimordisConfig.particleCount,
        label: 'fake',
      );

  @override
  Future<void> init() async {
    calls.add(FakeSimCall.init);
    _initCount++;
    _initialized = true;
    _disposed = false;
  }

  @override
  Future<void> seed(SimSeed seed) async {
    calls.add(FakeSimCall.seed);
    _seedCount++;
    _lastSeed = seed;
    _lastSeeded = seedSimulation(seed);
  }

  @override
  void setParams(SimParams params) {
    calls.add(FakeSimCall.setParams);
    _setParamsCount++;
    _lastParams = params;
  }

  @override
  void step(double dt) {
    calls.add(FakeSimCall.step);
    _stepCount++;
    _lastDt = dt;
    _simulatedTime += dt;
  }

  @override
  void present() {
    calls.add(FakeSimCall.present);
    _presentCount++;
  }

  @override
  Future<void> dispose() async {
    calls.add(FakeSimCall.dispose);
    _initialized = false;
    _disposed = true;
  }
}
