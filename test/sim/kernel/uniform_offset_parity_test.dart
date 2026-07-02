import 'dart:io';

import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/sim/sim_marshalling.dart';

/// Cross-checks the Dart-side uniform slot layout ([SimMarshalling]) against
/// the live WGSL `struct Params` text in `primordis.wgsl`
/// ([PRIMORDIS-TASK-006]).
///
/// [test/sim/sim_marshalling_test.dart] already asserts that `packUniforms`
/// writes each field at its documented byte offset — but that only proves the
/// Dart packer agrees with the Dart slot constants. It cannot catch the two
/// files drifting *together*: someone adding a WGSL struct field (or
/// reordering one) without updating `SimMarshalling` would silently shift
/// which byte offset a slider's `f32`/`u32` value lands at, corrupting the
/// live K values with no compile-time signal on either side.
///
/// This test closes that gap by parsing `struct Params { ... }` directly out
/// of the shader source and asserting its field order/types exactly match
/// [SimMarshalling]'s slot table, so a future field addition on either side
/// that isn't mirrored on the other fails CI immediately.
void main() {
  test('WGSL Params struct field order matches SimMarshalling slot offsets',
      () {
    final wgslPath = File('lib/sim/kernel/primordis.wgsl');
    expect(wgslPath.existsSync(), isTrue, reason: 'run from the package root');
    final source = wgslPath.readAsStringSync();

    final structMatch =
        RegExp(r'struct Params \{([^}]*)\}', dotAll: true).firstMatch(source);
    expect(structMatch, isNotNull, reason: 'struct Params not found in WGSL source');
    final body = structMatch!.group(1)!;

    // Field lines look like `  attractionK : f32,   // slot 0 ...`.
    final fieldPattern = RegExp(r'(\w+)\s*:\s*(f32|u32)\s*,');
    final fields = fieldPattern
        .allMatches(body)
        .map((m) => (m.group(1)!, m.group(2)!))
        .toList();

    // The expected order/types, indexed by SimMarshalling slot (source of
    // truth for the *Dart* side); the WGSL field at index k must be this slot.
    const expected = <(String, String)>[
      ('attractionK', 'f32'),
      ('repulsionK', 'f32'),
      ('friction', 'f32'),
      ('dt', 'f32'),
      ('worldWidth', 'f32'),
      ('worldHeight', 'f32'),
      ('maxRadius', 'f32'),
      ('binSize', 'f32'),
      ('gridWidth', 'u32'),
      ('gridHeight', 'u32'),
      ('numParticles', 'u32'),
      ('numBins', 'u32'),
      ('typeCount', 'u32'),
      // Slots 13..15 are explicit reserved padding on both sides.
      ('_pad0', 'u32'),
      ('_pad1', 'u32'),
      ('_pad2', 'u32'),
    ];

    expect(
      fields.length,
      SimMarshalling.uniformSlotCount,
      reason: 'WGSL Params field count must equal uniformSlotCount',
    );
    expect(
      fields.length,
      expected.length,
      reason: 'update this test file expected table alongside any struct change',
    );

    for (var slot = 0; slot < expected.length; slot++) {
      final (wgslName, wgslType) = fields[slot];
      final (wantName, wantType) = expected[slot];
      expect(
        wgslName,
        wantName,
        reason: 'slot $slot name mismatch: WGSL has "$wgslName", '
            'SimMarshalling expects "$wantName" at byte offset ${slot * 4}',
      );
      expect(
        wgslType,
        wantType,
        reason: 'slot $slot ("$wgslName") type mismatch: WGSL declares '
            '$wgslType, expected $wantType',
      );
    }

    // Every f32/u32 slot is exactly 4 bytes, so slot k's byte offset is
    // always k*4 — pin that arithmetic explicitly against the named slot
    // constants the live sliders write through.
    expect(SimMarshalling.slotAttractionK * 4, 0);
    expect(SimMarshalling.slotRepulsionK * 4, 4);
    expect(SimMarshalling.slotFriction * 4, 8);
    expect(SimMarshalling.slotDt * 4, 12);
    expect(SimMarshalling.slotWorldWidth * 4, 16);
    expect(SimMarshalling.slotWorldHeight * 4, 20);
    expect(SimMarshalling.slotMaxRadius * 4, 24);
    expect(SimMarshalling.slotBinSize * 4, 28);
    expect(SimMarshalling.slotGridWidth * 4, 32);
    expect(SimMarshalling.slotGridHeight * 4, 36);
    expect(SimMarshalling.slotNumParticles * 4, 40);
    expect(SimMarshalling.slotNumBins * 4, 44);
    expect(SimMarshalling.slotTypeCount * 4, 48);
  });
}
