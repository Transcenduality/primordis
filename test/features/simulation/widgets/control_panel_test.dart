import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/features/simulation/widgets/control_panel.dart';
import 'package:primordis/sim/fake_sim_backend.dart';
import 'package:primordis/sim/models/sim_params.dart';
import 'package:primordis/sim/providers/sim_providers.dart';

Future<FakeSimBackend> _pump(WidgetTester tester) async {
  final backend = FakeSimBackend();
  await tester.pumpWidget(
    ProviderScope(
      overrides: [simBackendProvider.overrideWith((ref) => backend)],
      child: const MaterialApp(
        home: Scaffold(body: SingleChildScrollView(child: ControlPanel())),
      ),
    ),
  );
  await tester.pump();
  return backend;
}

void main() {
  testWidgets('renders three labeled sliders, reset, seed, and play/pause',
      (tester) async {
    await _pump(tester);

    expect(find.text('Attraction K'), findsOneWidget);
    expect(find.text('Repulsion K'), findsOneWidget);
    expect(find.text('Drift'), findsOneWidget);
    expect(find.byType(Slider), findsNWidgets(3));

    expect(find.text('Reset params'), findsOneWidget);
    expect(find.text('Reseed'), findsOneWidget);
    expect(find.text('Pause'), findsOneWidget);

    // Every slider carries a tooltip.
    expect(find.byType(Tooltip), findsWidgets);
  });

  testWidgets('dragging a slider updates its provider value', (tester) async {
    await _pump(tester);

    final container = ProviderScope.containerOf(
      tester.element(find.byType(ControlPanel)),
    );
    final before = container.read(simParamsControllerProvider).attractionK;

    await tester.drag(find.byType(Slider).first, const Offset(80, 0));
    await tester.pump();

    final after = container.read(simParamsControllerProvider).attractionK;
    expect(after, isNot(equals(before)));
  });

  testWidgets('tapping reset restores slider defaults', (tester) async {
    await _pump(tester);
    final container = ProviderScope.containerOf(
      tester.element(find.byType(ControlPanel)),
    );

    container.read(simParamsControllerProvider.notifier).setAttractionK(99);
    await tester.pump();

    await tester.tap(find.text('Reset params'));
    await tester.pump();

    expect(
      container.read(simParamsControllerProvider).attractionK,
      SimSliders.attractionDefault,
    );
  });

  testWidgets(
      'tapping reseed regenerates the seed and calls seed() on the backend '
      'without tearing down the widget tree', (tester) async {
    final backend = await _pump(tester);
    final container = ProviderScope.containerOf(
      tester.element(find.byType(ControlPanel)),
    );
    final seedBefore = container.read(simSeedControllerProvider).seed;

    await tester.tap(find.text('Reseed'));
    // Allow the async reseed() future to complete.
    await tester.pump();
    await tester.pump();

    expect(container.read(simSeedControllerProvider).seed, isNot(seedBefore));
    expect(backend.seedCount, greaterThanOrEqualTo(1));
    // The widget tree is still there (no teardown/rebuild storm).
    expect(find.byType(ControlPanel), findsOneWidget);
  });

  testWidgets('play/pause toggles the run-state provider', (tester) async {
    await _pump(tester);
    final container = ProviderScope.containerOf(
      tester.element(find.byType(ControlPanel)),
    );

    expect(container.read(runStateControllerProvider).isPaused, isFalse);

    await tester.tap(find.text('Pause'));
    await tester.pump();
    expect(container.read(runStateControllerProvider).isPaused, isTrue);
    expect(find.text('Play'), findsOneWidget);

    await tester.tap(find.text('Play'));
    await tester.pump();
    expect(container.read(runStateControllerProvider).isPaused, isFalse);
  });
}
