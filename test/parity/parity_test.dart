import 'dart:convert';
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/cpu/counting_sort_binning.dart';
import 'package:primordis/sim/cpu/cpu_sim_step.dart';
import 'package:primordis/sim/cpu/particle_soa.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/type_matrix.dart';
import 'package:primordis/sim/parity/frame_metrics.dart';
import 'package:primordis/sim/parity/parity_bands.dart';
import 'package:primordis/sim/parity/parity_fingerprint.dart';
import 'package:primordis/sim/parity/parity_runner.dart';
import 'package:primordis/sim/sim_seeder.dart';

import 'parity_harness_support.dart';

/// The parity harness proper: it drives the deterministic CPU backend
/// ([PRIMORDIS-TASK-008]) from the shared seed and asserts its statistical
/// fingerprint falls within the committed reference tolerance bands, plus the
/// toroidal-correctness, matrix-orientation-lock, Drift-response, and
/// regression-detection ("teeth") guarantees the task requires.
///
/// Parity is **statistical, never bit-exact** ([PRIMORDIS-ADR-001]): no test
/// here asserts per-particle position or pixel equality.
ParityFingerprint _loadReference(String fixture) {
  final file = File('test/parity/fixtures/$fixture');
  return ParityFingerprint.fromJson(
    jsonDecode(file.readAsStringSync()) as Map<String, dynamic>,
  );
}

ParityFingerprint _runCpu(ParityHarnessConfig cfg) => runParity(
      backend: cfg.buildCpuBackend(),
      seed: cfg.seed,
      grid: cfg.grid,
      totalSteps: cfg.totalSteps,
      checkpoints: cfg.checkpoints,
      dt: cfg.dt,
      attractionK: cfg.attractionK,
      repulsionK: cfg.repulsionK,
      friction: cfg.friction,
    );

void main() {
  group('CPU backend vs committed reference fingerprint', () {
    test('default scenario is within the backend-vs-reference bands', () {
      final reference = _loadReference('cpu_reference_default.json');
      final observed = _runCpu(ParityHarnessConfig.defaults());
      final violations =
          observed.violationsAgainst(reference, ParityBands.vsReference);
      expect(
        violations,
        isEmpty,
        reason: 'CPU backend drifted out of the reference band:\n'
            '${violations.join('\n')}',
      );
    });

    test('high-Drift scenario is within the backend-vs-reference bands', () {
      final reference = _loadReference('cpu_reference_highdrift.json');
      final observed = _runCpu(ParityHarnessConfig.highDrift());
      final violations =
          observed.violationsAgainst(reference, ParityBands.vsReference);
      expect(violations, isEmpty, reason: violations.join('\n'));
    });

    test('the deterministic CPU tier reproduces its fingerprint exactly', () {
      // The counting-sort binning is bit-stable per seed+params, so two CPU
      // runs are *identical* — the one place a tight (near-exact) comparison is
      // legitimate (CPU-vs-CPU, never CPU-vs-GPU; [PRIMORDIS-TASK-009] notes).
      final a = _runCpu(ParityHarnessConfig.defaults());
      final b = _runCpu(ParityHarnessConfig.defaults());
      final violations = a.violationsAgainst(b, ParityBands.vsReference);
      expect(violations, isEmpty, reason: violations.join('\n'));
      // And the raw scalar metrics match to full precision.
      final sa = a.checkpoints['steady']!;
      final sb = b.checkpoints['steady']!;
      expect(sa.speed.mean, sb.speed.mean);
      expect(sa.occupancy.variance, sb.occupancy.variance);
      expect(sa.nearestNeighbour.mean, sb.nearestNeighbour.mean);
    });
  });

  group('Population conservation (conserved quantity, exact band)', () {
    test('no particle is lost through the toroidal wrap over the run', () {
      final observed = _runCpu(ParityHarnessConfig.defaults());
      for (final entry in observed.checkpoints.entries) {
        expect(
          entry.value.particleCount,
          ParityHarnessConfig.defaults().particleCount,
          reason: 'population changed at checkpoint ${entry.key}',
        );
      }
    });
  });

  group('Drift-slider behavioural response', () {
    test('higher Drift (lower friction) decays kinetic energy faster', () {
      // Same seed, same everything but friction: the low-retention run must
      // have *less* kinetic energy at the mid checkpoint. This is the
      // param-marshalling behavioural check every backend must reproduce.
      final normal = _runCpu(ParityHarnessConfig.defaults());
      final highDrift = _runCpu(ParityHarnessConfig.highDrift());
      final keNormal = normal.checkpoints['mid']!.kineticEnergy;
      final keHighDrift = highDrift.checkpoints['mid']!.kineticEnergy;
      expect(
        keHighDrift,
        lessThan(keNormal),
        reason: 'expected faster KE decay under higher Drift: '
            'highDrift=$keHighDrift !< normal=$keNormal',
      );
    });
  });

  group('Toroidal correctness (wrap-seam scenario)', () {
    // Build a frame with a uniform background plus one dense cluster, placed
    // either straddling the x=0/x=1080 seam or wholly in the interior. A
    // minimum-image-aware sim must treat both identically: the seam cluster is
    // ONE cluster (not two half-clusters torn by the seam), and it produces the
    // SAME edge-density profile a real interior cluster would — no *artificial*
    // accumulation attributable to the wrap itself.
    FrameMetrics buildFrame({required bool straddleSeam}) {
      final grid = ParityHarnessConfig.defaults().grid;
      const n = 3000;
      const clusterSize = 400;
      final positions = Float32List(n * 2);
      final velocities = Float32List(n * 2);
      final types = Int32List(n);
      var rngState = 777;
      double next(double scale) {
        rngState = (rngState * 1103515245 + 12345) & 0x7fffffff;
        return (rngState % 100000) / 100000.0 * scale;
      }

      // Dense cluster: a ~40px blob in an *edge* column either straddling the
      // x=0/x=1080 seam (centre 0, x wraps around) or wholly inside the last
      // column without wrapping (centre 1032, column 10). Both land in edge
      // bins, so a correct torus must give them comparable edge signatures — the
      // wrap must add no artificial accumulation. Only the seam-crossing (and
      // hence the wrap handling) differs.
      for (var i = 0; i < clusterSize; i++) {
        final localX = next(40) - 20; // -20..20 around the centre
        final centre = straddleSeam ? 0.0 : 1032.0;
        var x = centre + localX;
        if (x < 0) x += 1080;
        if (x >= 1080) x -= 1080;
        positions[i * 2] = x;
        positions[i * 2 + 1] = 360 + next(40) - 20;
        types[i] = i % 32;
      }
      // Uniform background fills the rest so interior bins are populated.
      for (var i = clusterSize; i < n; i++) {
        positions[i * 2] = next(1080);
        positions[i * 2 + 1] = next(720);
        types[i] = i % 32;
      }
      return FrameMetrics.from(
        positions: positions,
        velocities: velocities,
        types: types,
        particleCount: n,
        typeCount: 32,
        grid: grid,
      );
    }

    test('a cluster straddling the seam is counted as ONE cluster', () {
      final seam = buildFrame(straddleSeam: true);
      final interior = buildFrame(straddleSeam: false);
      // Toroidal connectivity must not split the seam cluster into two; both
      // frames have the same number of clusters (background + the one blob).
      expect(seam.clusterCount, interior.clusterCount,
          reason: 'seam cluster was torn into pieces by the wrap: '
              'seam=${seam.clusterCount} interior=${interior.clusterCount}');
    });

    test('a seam-crossing cluster produces no MORE edge accumulation than a '
        'non-wrapping edge cluster', () {
      final seam = buildFrame(straddleSeam: true);
      final edge = buildFrame(straddleSeam: false);
      // Both clusters hold the same number of particles in an edge column. A
      // *wrap bug* — double-counting a seam cluster, or piling it artificially
      // against the boundary — would make the seam case's edge excess exceed the
      // equivalent non-wrapping edge case. Correct minimum-image handling instead
      // spreads the seam blob across the two seam-adjacent columns, so the seam
      // case's peak edge density is at most the non-wrapping case's (never a
      // wrap-induced surplus). A small slack absorbs binning granularity.
      expect(
        seam.maxEdgeBinExcess,
        lessThanOrEqualTo(edge.maxEdgeBinExcess + 0.5),
        reason: 'seam cluster showed wrap-induced edge accumulation: '
            'seam=${seam.maxEdgeBinExcess} edge=${edge.maxEdgeBinExcess}',
      );
    });
  });

  group('Matrix orientation is locked ([my_type][other_type])', () {
    test('the CPU force law indexes forces[my_type * typeCount + other_type]',
        () {
      // Two particles, two types, close enough to attract. Type 0 is strongly
      // attracted to type 1 (forces[0*2+1] = +big), but type 1 is neutral to
      // type 0 (forces[1*2+0] = 0). If the orientation were transposed, the
      // FORCE would land on the wrong particle. We assert particle 0 (type 0)
      // accelerates toward particle 1, and particle 1 (type 1) does not.
      const typeCount = 2;
      final forces = TypeMatrix.fromRows(const <List<double>>[
        <double>[0.0, 0.8], // my_type 0 -> attracted to other_type 1
        <double>[0.0, 0.0], // my_type 1 -> neutral to everyone
      ]);
      // Radii large enough that the pair is in the attraction band, minDistance
      // small so it is not repulsion.
      final radii = TypeMatrix.fromRows(const <List<double>>[
        <double>[96.0, 96.0],
        <double>[96.0, 96.0],
      ]);
      final minDistances = TypeMatrix.fromRows(const <List<double>>[
        <double>[1.0, 1.0],
        <double>[1.0, 1.0],
      ]);
      final params = SimParams(
        forces: forces,
        minDistances: minDistances,
        radii: radii,
        attractionK: 1,
        repulsionK: 1,
        friction: 1, // no drift, so the force shows up cleanly in velocity
        particleCount: 2,
        typeCount: typeCount,
        // world/grid geometry left at the reference defaults (1080x720, 96px
        // bins, 11x7 grid).
      );
      final grid = GridGeometry(
        worldWidth: 1080,
        worldHeight: 720,
        binSize: 96,
      );
      final buffers = ParticleSoa(
        particleCount: 2,
        typeCount: typeCount,
        binCount: 77,
      );
      // Particle 0 at (500,360) type 0; particle 1 at (540,360) type 1.
      buffers.positions[0] = 500;
      buffers.positions[1] = 360;
      buffers.positions[2] = 540;
      buffers.positions[3] = 360;
      buffers.types[0] = 0;
      buffers.types[1] = 1;

      cpuSimStep(buffers, params, grid, 1);

      // Particle 0 (type 0, attracted to type 1) should gain +x velocity
      // toward particle 1; particle 1 (type 1, neutral) should not move.
      expect(buffers.velocities[0], greaterThan(0.0),
          reason: 'type-0 particle should accelerate toward type-1 neighbour');
      expect(buffers.velocities[2], closeTo(0.0, 1e-6),
          reason: 'type-1 particle is neutral; a transposed matrix would move '
              'it instead — orientation lock violated');
    });
  });

  group('The harness has teeth (detects a known regression)', () {
    test('removing attraction (all-repulsion physics) breaks the parity bands',
        () {
      // Reproduce the reference run but with a deliberately BROKEN backend that
      // makes every pair repulsive over a long range (forces all negative, wide
      // radii). With no attraction the particles never settle into the seeded
      // structure — their residual speed stays high and clustering flattens — so
      // `speed.mean` and `occupancy.variance` land well outside the reference
      // band. This is the "prove the harness has teeth" check ([TASK-009]).
      final reference = _loadReference('cpu_reference_default.json');
      final cfg = ParityHarnessConfig.defaults();
      final observed = runParity(
        backend: _AllRepulsionCpuBackend(cfg),
        seed: cfg.seed,
        grid: cfg.grid,
        totalSteps: cfg.totalSteps,
        checkpoints: cfg.checkpoints,
        dt: cfg.dt,
        attractionK: cfg.attractionK,
        repulsionK: cfg.repulsionK,
        friction: cfg.friction,
      );
      final violations =
          observed.violationsAgainst(reference, ParityBands.vsReference);
      expect(
        violations,
        isNotEmpty,
        reason: 'removing attraction must fail parity — the harness '
            'would be toothless otherwise',
      );
      // And the failure must be diagnostic: name a metric that actually moved.
      expect(
        violations.join('\n'),
        contains('speed.mean'),
        reason: 'expected the residual-speed metric to flag the regression',
      );
    });

    test('a transposed matrix is at least NOT bit-identical to the reference',
        () {
      // A weaker guard: transposing the (near-symmetric-in-distribution) random
      // matrices barely perturbs aggregate statistics, so it is NOT reliably
      // caught by the band check — which is *itself* a lesson the harness
      // encodes: aggregate parity cannot police per-pair orientation. That job
      // belongs to the dedicated orientation-lock unit test above. Here we only
      // confirm the transposed run diverges from the reference at all (some
      // metric moves), so a future bug that flips orientation AND shifts
      // aggregates is still visible.
      final reference = _loadReference('cpu_reference_default.json');
      final cfg = ParityHarnessConfig.defaults();
      final observed = runParity(
        backend: _TransposedForceCpuBackend(cfg),
        seed: cfg.seed,
        grid: cfg.grid,
        totalSteps: cfg.totalSteps,
        checkpoints: cfg.checkpoints,
        dt: cfg.dt,
        attractionK: cfg.attractionK,
        repulsionK: cfg.repulsionK,
        friction: cfg.friction,
      );
      final steadyRef = reference.checkpoints['steady']!;
      final steadyObs = observed.checkpoints['steady']!;
      // At least one aggregate must differ (they are not the same physics run).
      final diffs = FrameMetrics.compare(steadyRef, steadyObs);
      final anyMoved = diffs.any((d) => !d.contains('|Δ|=0.0000'));
      expect(anyMoved, isTrue,
          reason: 'transposed physics should perturb some aggregate metric');
    });
  });
}

/// A deliberately-broken CPU backend whose every pair is repulsive over a wide
/// radius (all forces negative, radii near max, minDistances near max). With no
/// attraction the particles never condense into the seeded structure — residual
/// speed stays high and occupancy variance flattens — a clear aggregate shift
/// the parity bands must catch. Used by the "teeth" test.
class _AllRepulsionCpuBackend implements ParityBackend {
  _AllRepulsionCpuBackend(this._cfg)
      : _grid = GridGeometry(
          worldWidth: _cfg.worldWidth.toDouble(),
          worldHeight: _cfg.worldHeight.toDouble(),
          binSize: _cfg.binSize,
        ),
        _buffers = ParticleSoa(
          particleCount: _cfg.particleCount,
          typeCount: _cfg.typeCount,
          binCount: _cfg.grid.binCount,
        );

  final ParityHarnessConfig _cfg;
  final GridGeometry _grid;
  final ParticleSoa _buffers;
  late SimParams _params;

  @override
  String get label => 'cpu-all-repulsion';

  @override
  int get particleCount => _cfg.particleCount;

  @override
  int get typeCount => _cfg.typeCount;

  @override
  void seed() {
    final seeded = seedSimulation(_cfg.simSeed);
    _buffers.loadFrom(seeded);
    final dim = seeded.forces.dimension;
    _params = _cfg.params.copyWith(
      // Every pair strongly repulsive.
      forces: TypeMatrix.generate(dim, (_, _) => -0.8),
      // Repulsion active out to near the interaction radius.
      minDistances: TypeMatrix.generate(dim, (_, _) => 90.0),
      radii: seeded.radii,
    );
  }

  @override
  void step(double dt) => cpuSimStep(_buffers, _params, _grid, dt);

  @override
  Float32List get positions => _buffers.positions;

  @override
  Float32List get velocities => _buffers.velocities;

  @override
  Int32List get types => _buffers.types;
}

/// A deliberately-broken CPU backend that transposes the three matrices,
/// inverting the locked `[my_type][other_type]` orientation. Retained for the
/// weaker "transposed run diverges at all" guard.
class _TransposedForceCpuBackend implements ParityBackend {
  _TransposedForceCpuBackend(this._cfg)
      : _grid = GridGeometry(
          worldWidth: _cfg.worldWidth.toDouble(),
          worldHeight: _cfg.worldHeight.toDouble(),
          binSize: _cfg.binSize,
        ),
        _buffers = ParticleSoa(
          particleCount: _cfg.particleCount,
          typeCount: _cfg.typeCount,
          binCount: _cfg.grid.binCount,
        );

  final ParityHarnessConfig _cfg;
  final GridGeometry _grid;
  final ParticleSoa _buffers;
  late SimParams _params;

  @override
  String get label => 'cpu-transposed';

  @override
  int get particleCount => _cfg.particleCount;

  @override
  int get typeCount => _cfg.typeCount;

  @override
  void seed() {
    final seeded = seedSimulation(_cfg.simSeed);
    _buffers.loadFrom(seeded);
    _params = _cfg.params.copyWith(
      forces: _transpose(seeded.forces),
      minDistances: _transpose(seeded.minDistances),
      radii: _transpose(seeded.radii),
    );
  }

  static TypeMatrix _transpose(TypeMatrix m) => TypeMatrix.generate(
        m.dimension,
        (row, col) => m.at(col, row),
      );

  @override
  void step(double dt) => cpuSimStep(_buffers, _params, _grid, dt);

  @override
  Float32List get positions => _buffers.positions;

  @override
  Float32List get velocities => _buffers.velocities;

  @override
  Int32List get types => _buffers.types;
}
