import 'dart:math' as math;

import 'package:primordis/sim/parity/frame_metrics.dart';

/// A tolerance band for a single scalar metric.
///
/// Parity is statistical: a backend passes when each metric lands *within a
/// band* of the reference, never on an exact value. A band is expressed as an
/// absolute tolerance plus a relative (fraction-of-reference) tolerance; the
/// effective allowance is the larger of the two, so a band works whether the
/// reference value is near zero (absolute dominates) or large (relative
/// dominates).
///
/// Bands are intentionally asymmetric in *tightness* across metrics — conserved
/// quantities (population) get a zero band, nondeterministic cluster geometry
/// gets a loose one. Every band in the harness is justified in a comment at its
/// definition site and committed as part of the golden baseline.
class Band {
  const Band({this.absolute = 0.0, this.relative = 0.0})
      : assert(absolute >= 0, 'absolute tolerance must be non-negative'),
        assert(relative >= 0, 'relative tolerance must be non-negative');

  /// An exact band: the observed value must equal the reference. Used for
  /// conserved quantities such as particle population.
  static const Band exact = Band();

  /// Absolute tolerance (same units as the metric).
  final double absolute;

  /// Relative tolerance as a fraction of the reference magnitude (0.1 = 10%).
  final double relative;

  /// The effective allowed deviation for a given [reference] value.
  double allowanceFor(double reference) =>
      math.max(absolute, relative * reference.abs());

  /// Whether [observed] is within this band of [reference].
  bool contains(double reference, double observed) =>
      (observed - reference).abs() <= allowanceFor(reference);

  @override
  String toString() =>
      'Band(abs=$absolute, rel=${(relative * 100).toStringAsFixed(1)}%)';
}

/// The complete set of per-metric bands used to judge one [FrameMetrics]
/// against a reference frame.
///
/// Two named presets are provided:
/// - [ParityBands.vsReference] — backend-vs-reference bands (tighter). The
///   deterministic CPU tier ([PRIMORDIS-TASK-008]) is measured against these.
/// - [ParityBands.crossBackend] — backend-vs-backend bands (looser),
///   acknowledging translator/atomics nondeterminism (Tint vs Naga;
///   [PRIMORDIS-TASK-017]) and the reference's own racy binning. The standalone
///   WGSL kernel and, later, the GPU backends are measured against these.
class ParityBands {
  const ParityBands({
    required this.particleCount,
    required this.clusterCount,
    required this.occupiedBinFraction,
    required this.occupancyMean,
    required this.occupancyVariance,
    required this.nearestNeighbourMean,
    required this.nearestNeighbourVariance,
    required this.segregationIndex,
    required this.speedMean,
    required this.speedVariance,
    required this.kineticEnergy,
    required this.maxEdgeBinExcess,
  });

  /// Backend-vs-reference bands: tight on conserved and low-variance
  /// quantities, moderate on cluster geometry.
  ///
  /// Rationale per band (see also the golden baseline docs):
  /// - `particleCount`: **exact** — population is conserved; a lost particle is
  ///   a wrap bug, never a tolerance question.
  /// - `occupancyMean`: **exact-ish** (tiny absolute) — it is
  ///   `particleCount / binCount` and so is fixed by conservation, independent
  ///   of dynamics; only float summation noise can move it.
  /// - `clusterCount`: ±2 clusters — connected-component count is sensitive to
  ///   the dense-bin threshold near boundaries, so a small absolute slack.
  /// - `occupiedBinFraction`: 10% relative — coarse spread measure, stable.
  /// - `occupancyVariance`, `nearestNeighbour*`, `segregationIndex`,
  ///   `speed*`, `kineticEnergy`: 25–35% relative — geometry-dependent
  ///   distribution shape; the reference binning is racy so even it varies
  ///   run-to-run, hence a generous band.
  /// - `maxEdgeBinExcess`: absolute 0.75 — a *correctness* guard, not a match
  ///   target; it must stay small (no edge pile-up) but need not match a
  ///   specific value.
  static const ParityBands vsReference = ParityBands(
    particleCount: Band.exact,
    clusterCount: Band(absolute: 2),
    occupiedBinFraction: Band(absolute: 0.05, relative: 0.10),
    occupancyMean: Band(absolute: 0.01, relative: 0.001),
    occupancyVariance: Band(absolute: 5, relative: 0.30),
    nearestNeighbourMean: Band(absolute: 1, relative: 0.25),
    nearestNeighbourVariance: Band(absolute: 5, relative: 0.35),
    segregationIndex: Band(absolute: 0.05, relative: 0.30),
    speedMean: Band(absolute: 0.5, relative: 0.25),
    speedVariance: Band(absolute: 2, relative: 0.35),
    kineticEnergy: Band(absolute: 2, relative: 0.35),
    maxEdgeBinExcess: Band(absolute: 0.75),
  );

  /// Backend-vs-backend bands: uniformly looser than [vsReference], because two
  /// nondeterministic GPU translators (Tint vs Naga) diverge more from each
  /// other than a deterministic CPU run diverges from a captured reference.
  /// Population is still exact (conservation is not translator-dependent).
  static const ParityBands crossBackend = ParityBands(
    particleCount: Band.exact,
    clusterCount: Band(absolute: 4),
    occupiedBinFraction: Band(absolute: 0.10, relative: 0.20),
    occupancyMean: Band(absolute: 0.05, relative: 0.01),
    occupancyVariance: Band(absolute: 10, relative: 0.50),
    nearestNeighbourMean: Band(absolute: 2, relative: 0.40),
    nearestNeighbourVariance: Band(absolute: 10, relative: 0.60),
    segregationIndex: Band(absolute: 0.10, relative: 0.50),
    speedMean: Band(absolute: 1, relative: 0.40),
    speedVariance: Band(absolute: 4, relative: 0.60),
    kineticEnergy: Band(absolute: 4, relative: 0.60),
    maxEdgeBinExcess: Band(absolute: 1.0),
  );

  final Band particleCount;
  final Band clusterCount;
  final Band occupiedBinFraction;
  final Band occupancyMean;
  final Band occupancyVariance;
  final Band nearestNeighbourMean;
  final Band nearestNeighbourVariance;
  final Band segregationIndex;
  final Band speedMean;
  final Band speedVariance;
  final Band kineticEnergy;
  final Band maxEdgeBinExcess;

  /// Checks [observed] against [reference] under these bands.
  ///
  /// Returns an empty list when every metric is within band; otherwise one
  /// diagnostic line per out-of-band metric naming the metric, the reference,
  /// the observed value, the deviation, and the allowed band — so a failure is
  /// immediately actionable ("which metric, which backend, observed vs band").
  List<String> violations(FrameMetrics reference, FrameMetrics observed) {
    final failures = <String>[];

    void check(String name, Band band, double ref, double obs) {
      if (!band.contains(ref, obs)) {
        final delta = (obs - ref).abs();
        final allowed = band.allowanceFor(ref);
        failures.add(
          '$name: observed ${obs.toStringAsFixed(4)} vs reference '
          '${ref.toStringAsFixed(4)} (|Δ|=${delta.toStringAsFixed(4)} > '
          'allowed ${allowed.toStringAsFixed(4)}; $band)',
        );
      }
    }

    check('particleCount', particleCount, reference.particleCount.toDouble(),
        observed.particleCount.toDouble());
    check('clusterCount', clusterCount, reference.clusterCount.toDouble(),
        observed.clusterCount.toDouble());
    check('occupiedBinFraction', occupiedBinFraction,
        reference.occupiedBinFraction, observed.occupiedBinFraction);
    check('occupancy.mean', occupancyMean, reference.occupancy.mean,
        observed.occupancy.mean);
    check('occupancy.variance', occupancyVariance, reference.occupancy.variance,
        observed.occupancy.variance);
    check('nearestNeighbour.mean', nearestNeighbourMean,
        reference.nearestNeighbour.mean, observed.nearestNeighbour.mean);
    check('nearestNeighbour.variance', nearestNeighbourVariance,
        reference.nearestNeighbour.variance,
        observed.nearestNeighbour.variance);
    check('segregationIndex', segregationIndex, reference.segregationIndex,
        observed.segregationIndex);
    check('speed.mean', speedMean, reference.speed.mean, observed.speed.mean);
    check('speed.variance', speedVariance, reference.speed.variance,
        observed.speed.variance);
    check('kineticEnergy', kineticEnergy, reference.kineticEnergy,
        observed.kineticEnergy);
    check('maxEdgeBinExcess', maxEdgeBinExcess, reference.maxEdgeBinExcess,
        observed.maxEdgeBinExcess);

    return failures;
  }
}
