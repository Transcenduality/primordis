import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Builds the app's Material 3 theme.
///
/// Typography uses GoogleFonts (house standard — no hardcoded `fontFamily`).
/// A dark scheme suits the simulation's near-black canvas background.
ThemeData buildPrimordisTheme() {
  final base = ThemeData(
    useMaterial3: true,
    colorScheme: ColorScheme.fromSeed(
      seedColor: const Color(0xFF6750A4),
      brightness: Brightness.dark,
    ),
  );
  return base.copyWith(
    textTheme: GoogleFonts.interTextTheme(base.textTheme),
  );
}
