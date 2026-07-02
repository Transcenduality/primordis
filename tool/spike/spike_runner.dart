// PRIMORDIS-TASK-801 spike core — see spike_main.dart for context.
//
// Drives the exact 3-pass atomic-binning kernel at the full reference workload
// (24,000 particles / 32 types) through minigpu (Dawn-over-Metal) and
// validates, via periodic CPU readback, that the simulation stays healthy:
// finite in-world positions/velocities and bin-count conservation
// (Σ raw binCounts == numParticles every frame, since over-cap particles still
// increment their bin's counter).

import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:minigpu/minigpu.dart';

import 'package:primordis/sim/ffi/wgsl_pass_adapter.dart';
import 'package:primordis/sim/kernel/kernel_source.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/models/sim_seed.dart';
import 'package:primordis/sim/sim_marshalling.dart';
import 'package:primordis/sim/sim_seeder.dart';

/// Fixed spike timestep (the app frame loop supplies wall-clock dt; the spike
/// wants reproducible integration, so it uses the nominal 60 Hz tick).
const double kSpikeDt = 1.0 / 60.0;

/// Outcome of a spike run.
class SpikeReport {
  SpikeReport({
    required this.framesRun,
    required this.framesRequested,
    required this.elapsed,
    required this.failures,
  });

  final int framesRun;
  final int framesRequested;
  final Duration elapsed;
  final List<String> failures;

  bool get passed => failures.isEmpty && framesRun >= framesRequested;

  double get fps => elapsed.inMicroseconds == 0
      ? 0
      : framesRun / (elapsed.inMicroseconds / 1e6);

  String summary() {
    final status = passed ? 'PASS' : 'FAIL';
    final b = StringBuffer()
      ..writeln('=== TASK-801 Dawn/minigpu spike: $status ===')
      ..writeln('frames: $framesRun/$framesRequested')
      ..writeln(
        'elapsed: ${elapsed.inMilliseconds} ms '
        '(${fps.toStringAsFixed(1)} fps incl. per-pass await + readback)',
      );
    for (final f in failures) {
      b.writeln('FAILURE: $f');
    }
    return b.toString();
  }
}

/// Runs the 3-pass kernel for [frames] frames at the full reference workload,
/// validating state every [logEvery] frames (and on the final frame).
Future<SpikeReport> runSpike({required int frames, int logEvery = 100}) async {
  final source = await loadKernelSource();
  final failures = <String>[];

  // Surface Dawn/minigpu native diagnostics (validation errors are otherwise
  // reported asynchronously and invisibly). 0 = debug.
  Minigpu.setLogCallback(
    (level, message) => debugPrint('[mgpu:$level] $message'),
    level: 0,
  );

  final gpu = Minigpu();
  await gpu.init();

  // --- Seed the full reference workload via the SHARED seeding path. ---
  final seeded = seedSimulation(const SimSeed());
  final params = SimParams(
    forces: seeded.forces,
    minDistances: seeded.minDistances,
    radii: seeded.radii,
  );
  final n = params.particleCount;
  final uniformBytes = packUniforms(params, kSpikeDt);

  // --- Buffers (slots mirror KernelBindings / the WGSL @binding indices). ---
  // createBuffer takes a BYTE size (mgpuCreateBuffer(byteSize, dataType));
  // Buffer.read/write take ELEMENT counts. Verified against minigpu 1.5.x
  // source — the pub.dev docs' examples predate this signature.
  Future<Buffer> f32(Float32List data) async {
    final buf = gpu.createBuffer(data.lengthInBytes, BufferDataType.float32);
    await buf.write(data, data.length);
    return buf;
  }

  Future<Buffer> u32(Uint32List data) async {
    final buf = gpu.createBuffer(data.lengthInBytes, BufferDataType.uint32);
    await buf.write(data, data.length, dataType: BufferDataType.uint32);
    return buf;
  }

  final paramsBuf = await u32(uniformBytes.buffer.asUint32List());
  final positions = await f32(seeded.positions);
  final velocities = await f32(seeded.velocities);
  final types = await u32(seeded.types);
  // The three matrices ride in ONE buffer (adapter constraint 4: ≤8 storage
  // buffers per stage under minigpu); bindings 5/6 stay vacant.
  final matrices = await f32(
    mergedMatrices(
      flattenMatrix(params.forces),
      flattenMatrix(params.minDistances),
      flattenMatrix(params.radii),
    ),
  );
  final binCounts = await u32(newBinCounts(params));
  final binParticles = await u32(newBinParticles(params));

  final slots = <int, Buffer>{
    KernelBindings.params: paramsBuf, // 0
    KernelBindings.positions: positions, // 1
    KernelBindings.velocities: velocities, // 2
    KernelBindings.types: types, // 3
    kMergedMatricesBinding: matrices, // 4 — [forces, minDistances, radii]
    KernelBindings.binCounts: binCounts, // 7
    KernelBindings.binParticles: binParticles, // 8
  };

  // --- One pipeline per pass (minigpu: entry point is always `main`). ---
  ComputeShader pass(String entryPoint) {
    final shader = gpu.createComputeShader()
      ..loadKernelString(adaptKernelForMinigpu(source, entryPoint));
    for (final entry in slots.entries) {
      shader.setBufferAtSlot(entry.key, entry.value);
    }
    return shader;
  }

  // --- Upload→readback round-trip canary (BEFORE any dispatch). ---
  // All-zero buffers would pass the finite/in-world checks below (0,0 is a
  // valid world position), so prove buffer I/O works at all: what we read back
  // must be what the seeder uploaded, not zeros.
  {
    final canary = Float32List(2 * n);
    await positions.read(canary, canary.length);
    var mismatches = 0;
    for (var i = 0; i < canary.length; i++) {
      if (canary[i] != seeded.positions[i]) mismatches++;
    }
    if (mismatches > 0) {
      failures.add(
        'round-trip canary: $mismatches/${canary.length} readback values '
        'differ from the uploaded seed (buffer I/O is broken; '
        'first uploaded=${seeded.positions[0]}, read=${canary[0]})',
      );
      return SpikeReport(
        framesRun: 0,
        framesRequested: frames,
        elapsed: Duration.zero,
        failures: failures,
      );
    }
    debugPrint('spike: upload→readback round-trip OK');
  }

  final clearBins = pass(KernelEntryPoints.clearBins);
  final scatterBins = pass(KernelEntryPoints.scatterBins);
  final interact = pass(KernelEntryPoints.interact);

  final binGroups = computeWorkgroups(params.binCount);
  final particleGroups = computeWorkgroups(n);

  // --- Readback scratch (reused; readback is spike-only, never app code). ---
  final posOut = Float32List(2 * n);
  final velOut = Float32List(2 * n);
  final countsOut = Uint32List(params.binCount);

  Future<void> validate(int frame) async {
    await positions.read(posOut, posOut.length);
    await velocities.read(velOut, velOut.length);
    await binCounts.read(
      countsOut,
      countsOut.length,
      dataType: BufferDataType.uint32,
    );

    var badPos = 0;
    var badVel = 0;
    final w = params.worldWidth.toDouble();
    final h = params.worldHeight.toDouble();
    for (var i = 0; i < n; i++) {
      final x = posOut[2 * i];
      final y = posOut[2 * i + 1];
      if (!x.isFinite || !y.isFinite || x < 0 || x >= w || y < 0 || y >= h) {
        badPos++;
      }
      if (!velOut[2 * i].isFinite || !velOut[2 * i + 1].isFinite) badVel++;
    }
    var countSum = 0;
    for (final c in countsOut) {
      countSum += c;
    }

    if (badPos > 0) {
      failures.add('frame $frame: $badPos NaN/Inf/out-of-world positions');
    }
    if (badVel > 0) {
      failures.add('frame $frame: $badVel NaN/Inf velocities');
    }
    if (countSum != n) {
      failures.add('frame $frame: Σ binCounts == $countSum, expected $n');
    }
    debugPrint(
      'spike frame $frame: badPos=$badPos badVel=$badVel binSum=$countSum',
    );
  }

  // --- The 1000-frame soak. ---
  final sw = Stopwatch()..start();
  var framesRun = 0;
  try {
    for (var frame = 1; frame <= frames; frame++) {
      await clearBins.dispatch(binGroups, 1, 1);
      await scatterBins.dispatch(particleGroups, 1, 1);
      await interact.dispatch(particleGroups, 1, 1);
      framesRun = frame;
      if (frame % logEvery == 0 || frame == frames) {
        await validate(frame);
        if (failures.isNotEmpty) break; // fail fast; state is already sick
      }
    }
  } catch (e) {
    failures.add('frame ${framesRun + 1}: dispatch/readback threw: $e');
  }
  sw.stop();

  for (final shader in [clearBins, scatterBins, interact]) {
    shader.destroy();
  }
  for (final buf in slots.values) {
    buf.destroy();
  }
  await gpu.destroy();

  return SpikeReport(
    framesRun: framesRun,
    framesRequested: frames,
    elapsed: sw.elapsed,
    failures: failures,
  );
}
