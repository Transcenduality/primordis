import 'dart:io';

import 'package:flutter_test/flutter_test.dart';

/// `flutter build web --wasm` (Skwasm) forbids the legacy interop libraries
/// `dart:html` and `dart:js_util` ANYWHERE in the dependency tree
/// ([PRIMORDIS-ADR-007]). The authoritative enforcement is the `--wasm` build
/// itself (CI, [PRIMORDIS-TASK-010]); this fast guard catches a regression in
/// *our own* `lib/` source before it reaches that build — the web backend must
/// use `dart:js_interop` + `package:web` exclusively.
void main() {
  test('no dart:html / dart:js_util imports anywhere under lib/', () {
    final libDir = Directory('lib');
    expect(libDir.existsSync(), isTrue, reason: 'run from the package root');

    final forbidden = RegExp(
      r'''import\s+['"]dart:(html|js_util)['"]''',
    );
    final offenders = <String>[];

    for (final entity in libDir.listSync(recursive: true)) {
      if (entity is! File || !entity.path.endsWith('.dart')) continue;
      final lines = entity.readAsLinesSync();
      for (var i = 0; i < lines.length; i++) {
        if (forbidden.hasMatch(lines[i])) {
          offenders.add('${entity.path}:${i + 1}: ${lines[i].trim()}');
        }
      }
    }

    expect(
      offenders,
      isEmpty,
      reason: 'legacy interop is banned under --wasm:\n${offenders.join('\n')}',
    );
  });
}
