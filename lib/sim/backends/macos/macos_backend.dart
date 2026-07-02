/// Conditional-import facade for the macOS Dawn backend (PRIMORDIS-TASK-801),
/// mirroring the web facade (`../web/web_backend.dart`): importing this file
/// is safe on every platform; only native builds pull in `dart:ffi`/minigpu.
library;

export 'macos_backend_stub.dart'
    if (dart.library.io) 'macos_backend_io.dart';
