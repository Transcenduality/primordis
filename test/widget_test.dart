import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:primordis/app/app.dart';

void main() {
  setUpAll(() {
    // Don't hit the network for fonts in tests (keeps pumps deterministic).
    GoogleFonts.config.allowRuntimeFetching = false;
  });

  testWidgets('app boots into a ProviderScope and renders the home route',
      (tester) async {
    await tester.pumpWidget(const ProviderScope(child: PrimordisApp()));
    await tester.pump();

    // ProviderScope is mounted at the root (Riverpod is wired in).
    expect(find.byType(ProviderScope), findsOneWidget);

    // The home route rendered via GoRouter.
    expect(find.text('Primordis'), findsOneWidget);
    expect(find.textContaining('Flutter Web + macOS'), findsOneWidget);
  });
}
