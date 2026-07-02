import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/features/simulation/providers/sim_params_provider.dart';
import 'package:primordis/features/simulation/widgets/simulation_view.dart';
import 'package:primordis/sim/providers/sim_providers.dart';

/// Pumps [SimulationView] inside a transparent scaffold at a fixed surface size
/// whose aspect (1.5) matches the world, so the field fills the region with no
/// letterbox (scale 10/9). [taps] records forwarded field-pointer world coords.
/// [disableAnimations] simulates `prefers-reduced-motion` /
/// `MediaQuery.disableAnimations`. Returns the [ProviderContainer] backing the
/// pumped tree so tests can inspect provider state directly.
Future<ProviderContainer> _pump(
  WidgetTester tester,
  List<Offset> taps, {
  bool disableAnimations = false,
}) async {
  tester.view.devicePixelRatio = 1.0;
  tester.view.physicalSize = const Size(1200, 800);
  addTearDown(tester.view.reset);

  final container = ProviderContainer();
  addTearDown(container.dispose);

  await tester.pumpWidget(
    UncontrolledProviderScope(
      container: container,
      child: MediaQuery(
        data: MediaQueryData(disableAnimations: disableAnimations),
        child: MaterialApp(
          home: Scaffold(
            backgroundColor: Colors.transparent,
            body: SimulationView(onFieldPointer: taps.add),
          ),
        ),
      ),
    ),
  );
  // A bounded number of pumps (not pumpAndSettle): the view now runs a
  // continuous per-frame Ticker ([PRIMORDIS-TASK-006]), which never
  // "settles". A few frames is enough for layout and async backend bring-up.
  for (var i = 0; i < 5; i++) {
    await tester.pump(const Duration(milliseconds: 16));
  }
  return container;
}

void main() {
  testWidgets('overlays chrome over a transparent glass-pane', (tester) async {
    await _pump(tester, <Offset>[]);

    // Chrome renders (title + tagline + version).
    expect(find.text('Primordis'), findsOneWidget);
    expect(find.textContaining('Flutter Web + macOS'), findsOneWidget);

    // The glass-pane paints nothing opaque over the field: every ColoredBox in
    // the view's subtree is translucent (the chrome panel), so the simulation
    // behind shows through.
    final boxes = tester.widgetList<ColoredBox>(
      find.descendant(
        of: find.byType(SimulationView),
        matching: find.byType(ColoredBox),
      ),
    );
    expect(boxes, isNotEmpty);
    for (final box in boxes) {
      expect(box.color.a, lessThan(1.0), reason: 'glass-pane must stay see-through');
    }
  });

  testWidgets('forwards open-field pointers as world coordinates', (tester) async {
    final taps = <Offset>[];
    await _pump(tester, taps);

    // Region 1200×800, world 1080×720 → scale 10/9, no offset.
    // Local (1100,700) → world (990, 630).
    await tester.tapAt(const Offset(1100, 700));
    await tester.pump();

    expect(taps, hasLength(1));
    expect(taps.single.dx, closeTo(990, 0.5));
    expect(taps.single.dy, closeTo(630, 0.5));
  });

  testWidgets('pointers over the chrome are not forwarded to the field',
      (tester) async {
    final taps = <Offset>[];
    await _pump(tester, taps);

    // Tapping the chrome title must stay with Flutter — the field seam ignores
    // it (no double-dispatch).
    await tester.tap(find.text('Primordis'));
    await tester.pump();

    expect(taps, isEmpty);
  });

  group('reduced motion (PRIMORDIS-ADR-006)', () {
    testWidgets(
        'starts paused when MediaQuery.disableAnimations is true',
        (tester) async {
      final container =
          await _pump(tester, <Offset>[], disableAnimations: true);

      expect(container.read(runStateControllerProvider).isPaused, isTrue);
      expect(container.read(reducedMotionControllerProvider), isTrue);
      // The static-state affordance is surfaced and controls stay reachable.
      expect(find.text('Play'), findsOneWidget);
    });

    testWidgets('does not start paused when reduced motion is off',
        (tester) async {
      final container = await _pump(tester, <Offset>[]);

      expect(container.read(runStateControllerProvider).isPaused, isFalse);
      expect(container.read(reducedMotionControllerProvider), isFalse);
    });

    testWidgets(
        'controls remain operable (tappable) while paused for reduced motion',
        (tester) async {
      final container =
          await _pump(tester, <Offset>[], disableAnimations: true);

      // Play/Resume is reachable and flips the run state back on.
      await tester.tap(find.text('Play'));
      await tester.pump();

      expect(container.read(runStateControllerProvider).isPaused, isFalse);
    });
  });

  group('play/pause frame stepping', () {
    testWidgets('paused holds the frame counter; resume advances it',
        (tester) async {
      final container = await _pump(tester, <Offset>[]);
      // Let bring-up (init/seed) complete and a few frames step.
      for (var i = 0; i < 3; i++) {
        await tester.pump(const Duration(milliseconds: 16));
      }
      final framesBeforePause = container.read(runStateControllerProvider).frame;
      expect(framesBeforePause, greaterThan(0));

      await tester.tap(find.text('Pause'));
      await tester.pump();
      final frameAtPause = container.read(runStateControllerProvider).frame;

      for (var i = 0; i < 3; i++) {
        await tester.pump(const Duration(milliseconds: 16));
      }
      expect(
        container.read(runStateControllerProvider).frame,
        frameAtPause,
        reason: 'paused ticks must not advance the frame counter',
      );

      await tester.tap(find.text('Play'));
      await tester.pump();
      for (var i = 0; i < 3; i++) {
        await tester.pump(const Duration(milliseconds: 16));
      }
      expect(
        container.read(runStateControllerProvider).frame,
        greaterThan(frameAtPause),
        reason: 'resuming must advance the frame counter again',
      );
    });
  });
}
