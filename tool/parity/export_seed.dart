// Exports the deterministic seeded initial condition (positions, velocities,
// types, and the three matrices) as JSON, so *every* backend — the Dart CPU
// tier, the standalone WGSL kernel (Node/WebGPU), and later the Dawn/Metal FFI
// tier — starts from the **byte-identical** initial state the parity contract
// requires ([PRIMORDIS-TASK-009]).
//
// Because the Dart seeder uses `dart:math`'s `Random` and the reference uses
// NumPy's PRNG, the two cannot reproduce each other's draws. The parity harness
// sidesteps that by making ONE source of truth (this exporter) emit the concrete
// initial arrays; non-Dart backends load these rather than re-seeding. See the
// task's "Seed sharing" implementation note.
//
// Usage (matches the harness's default scenario):
//   dart run tool/parity/export_seed.dart \
//     --seed 42 --particles 3000 --types 32 > seed.json
//
// The emitted JSON schema (all flat number arrays):
//   { seed, particleCount, typeCount, worldWidth, worldHeight,
//     positions:[x,y,...], velocities:[vx,vy,...], types:[...],
//     forces:[...n*n...], minDistances:[...], radii:[...] }
//
// This file lives under tool/ (reference-only tooling, not shipped in the app).

import 'dart:convert';
import 'dart:io';

import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/sim_seeder.dart';

int _intArg(List<String> args, String name, int fallback) {
  final i = args.indexOf(name);
  if (i >= 0 && i + 1 < args.length) return int.parse(args[i + 1]);
  return fallback;
}

void main(List<String> args) {
  final seed = _intArg(args, '--seed', 42);
  final particleCount = _intArg(args, '--particles', 3000);
  final typeCount = _intArg(args, '--types', 32);

  final seeded = seedSimulation(
    SimSeed(seed: seed, particleCount: particleCount, typeCount: typeCount),
  );

  final json = <String, dynamic>{
    'seed': seed,
    'particleCount': seeded.particleCount,
    'typeCount': seeded.typeCount,
    'worldWidth': 1080,
    'worldHeight': 720,
    'positions': seeded.positions.toList(),
    'velocities': seeded.velocities.toList(),
    'types': seeded.types.toList(),
    'forces': seeded.forces.values.toList(),
    'minDistances': seeded.minDistances.values.toList(),
    'radii': seeded.radii.values.toList(),
  };

  stdout.writeln(const JsonEncoder.withIndent('  ').convert(json));
}
