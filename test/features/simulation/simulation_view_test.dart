import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/features/simulation/widgets/simulation_view.dart';

/// Pumps [SimulationView] inside a transparent scaffold at a fixed surface size
/// whose aspect (1.5) matches the world, so the field fills the region with no
/// letterbox (scale 10/9). [taps] records forwarded field-pointer world coords.
Future<void> _pump(WidgetTester tester, List<Offset> taps) async {
  tester.view.devicePixelRatio = 1.0;
  tester.view.physicalSize = const Size(1200, 800);
  addTearDown(tester.view.reset);

  await tester.pumpWidget(
    ProviderScope(
      child: MaterialApp(
        home: Scaffold(
          backgroundColor: Colors.transparent,
          body: SimulationView(onFieldPointer: taps.add),
        ),
      ),
    ),
  );
  await tester.pumpAndSettle();
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
}
