import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/widgets.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:flutter_web_plugins/url_strategy.dart';
import 'package:primordis/app/app.dart';

void main() {
  if (kIsWeb) {
    // Path-based URLs on web (DGROUP_WEB-ADR-021). Guarded by kIsWeb because
    // the strategy uses web-only browser APIs and would throw on macOS.
    usePathUrlStrategy();
  }
  runApp(const ProviderScope(child: PrimordisApp()));
}
