import 'package:flutter/material.dart';
import 'package:flutter_riverpod/flutter_riverpod.dart';
import 'package:primordis/app/router.dart';
import 'package:primordis/app/theme.dart';

/// Root application widget.
///
/// A `ConsumerWidget` (not `StatelessWidget`) so the Riverpod pattern is in
/// place from the start; routing goes through `MaterialApp.router` + GoRouter.
class PrimordisApp extends ConsumerWidget {
  const PrimordisApp({super.key});

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    return MaterialApp.router(
      title: 'Primordis',
      debugShowCheckedModeBanner: false,
      theme: buildPrimordisTheme(),
      routerConfig: appRouter,
    );
  }
}
