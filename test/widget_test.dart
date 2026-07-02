import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:primordis/app/app.dart';

void main() {
  setUpAll(() {
    // Disable network font fetching so pumpAndSettle() below doesn't wait on a
    // pending GoogleFonts request (which would otherwise time out in tests).
    GoogleFonts.config.allowRuntimeFetching = false;
  });

  testWidgets('app boots into a ProviderScope and renders the home route',
      (tester) async {
    await tester.pumpWidget(const ProviderScope(child: PrimordisApp()));
    // A bounded number of pumps (not pumpAndSettle): the simulation view now
    // runs a continuous per-frame Ticker ([PRIMORDIS-TASK-006]), which never
    // "settles". A handful of frames is enough for GoRouter's initial routing
    // and the async backend bring-up to complete deterministically.
    for (var i = 0; i < 5; i++) {
      await tester.pump(const Duration(milliseconds: 16));
    }

    // ProviderScope is mounted at the root (Riverpod is wired in).
    expect(find.byType(ProviderScope), findsOneWidget);

    // The home route rendered via GoRouter.
    expect(find.text('Primordis'), findsOneWidget);
    expect(find.textContaining('Flutter Web + macOS'), findsOneWidget);
  });
}
