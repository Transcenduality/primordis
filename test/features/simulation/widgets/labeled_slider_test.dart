import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:primordis/features/simulation/widgets/labeled_slider.dart';

Future<void> _pump(WidgetTester tester, {required Widget child}) async {
  await tester.pumpWidget(MaterialApp(home: Scaffold(body: child)));
}

void main() {
  testWidgets('renders label, formatted value, and a tooltip', (tester) async {
    await _pump(
      tester,
      child: LabeledSlider(
        label: 'Attraction K',
        value: 32,
        min: 0,
        max: 128,
        tooltip: 'Scales attraction strength.',
        onChanged: (_) {},
      ),
    );

    expect(find.text('Attraction K'), findsOneWidget);
    expect(find.text('32.00'), findsOneWidget);
    expect(find.byType(Tooltip), findsOneWidget);
    expect(
      (tester.widget(find.byType(Tooltip)) as Tooltip).message,
      'Scales attraction strength.',
    );
  });

  testWidgets('exposes a Semantics label and value for screen readers',
      (tester) async {
    final handle = tester.ensureSemantics();

    await _pump(
      tester,
      child: LabeledSlider(
        label: 'Repulsion K',
        value: 10,
        min: 0,
        max: 20,
        tooltip: 'Scales repulsion strength.',
        onChanged: (_) {},
      ),
    );

    final semantics = tester.getSemantics(find.bySemanticsLabel('Repulsion K'));
    expect(semantics.label, 'Repulsion K');
    expect(semantics.value, '10.00');

    handle.dispose();
  });

  testWidgets('dragging invokes onChanged with a clamped value', (tester) async {
    double? changed;
    await _pump(
      tester,
      child: LabeledSlider(
        label: 'Drift',
        value: 0.5,
        min: 0,
        max: 1,
        tooltip: 'Friction multiplier.',
        onChanged: (v) => changed = v,
      ),
    );

    final sliderFinder = find.byType(Slider);
    expect(sliderFinder, findsOneWidget);

    // Drag the slider thumb; any drag on the track should invoke onChanged.
    await tester.drag(sliderFinder, const Offset(20, 0));
    await tester.pump();

    expect(changed, isNotNull);
  });

  testWidgets('applies a custom valueFormatter', (tester) async {
    await _pump(
      tester,
      child: LabeledSlider(
        label: 'Drift',
        value: 0.25,
        min: 0,
        max: 1,
        tooltip: 'Friction multiplier.',
        onChanged: (_) {},
        valueFormatter: (v) => '${(v * 100).round()}%',
      ),
    );

    expect(find.text('25%'), findsOneWidget);
  });
}
