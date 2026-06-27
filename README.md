# 🧬 Primordis

- A digital emergence simulator where structure, motion, and proto-life arise from raw physics.

Primordis is a GPU-accelerated particle-based simulation where life-like behaviors—swimmers, caterpillars, and **multicellular organisms**—emerge from nothing but local interaction.

There are no hardcoded cells.  
No genetic logic.  
Just particles, forces, memory... and time.

---

## 🌌 What Happens Inside

- 💠 Particles of 32 types interact via attractive/repulsive force matrices
- 🔁 Local recursive loops form **stable arrangements**
- 🐛 Some clusters begin to move as **swimmers** and **caterpillars**
- 🧬 Eventually, collections of swimmers **cohere** into **multicellular structures** - Organized, persistent, and adaptive.

These aren't designed organisms.  
They **emerge**, evolve, and stabilize on their own.

---

## 📷 Screenshots

![1.3 Preview](Atlas/1.3_Preview.png)

---

## 🛠 Getting Started

1. Install Pygame, Moderngl and Numpy
'pip install pygame moderngl numpy'

2. Right click and run with Python. If this doesn't work, use the terminal, and navigate to the Primordis directory with 'cd (file location)' followed by 'python Primordis.py'

> `Primordis.py` is the **reference implementation**. It is kept as the parity
> baseline for the Flutter port (see `docs/` and PRIMORDIS-TASK-009).

---

## 🦋 Flutter app (Web + macOS)

Primordis is being ported to a single Flutter app that runs on **Flutter Web
(WASM)** and **native macOS**. See [`docs/`](docs/) for the full plan
(PRD, ADRs, tasks) and [`docs/standards-mcp-setup.md`](docs/standards-mcp-setup.md)
for the standards MCP wiring.

### Layout

```
lib/
  main.dart                    # usePathUrlStrategy (web) + ProviderScope + MaterialApp.router
  app/                         # PrimordisApp, GoRouter, Material 3 theme
  features/home/...            # screens (feature-first; DGROUP_WEB-ADR-019)
  shared/constants/            # PrimordisConfig (version + sim constants)
  sim/                         # SimBackend seam — GPU/FFI/compute lives here, behind the interface
```

> This is a **standalone repo** (the Flutter app lives at the repo root, not in
> a monorepo). All GPU/compute code is quarantined behind `SimBackend`
> (PRIMORDIS-ADR-001); the UI/state layers follow house Flutter standards
> (Riverpod, Freezed, GoRouter, Material 3, `package:lint`).

### Build & run

```bash
flutter pub get
dart run build_runner build --delete-conflicting-outputs   # Freezed / Riverpod codegen
flutter analyze                                             # zero-warning policy
flutter test
flutter run -d chrome                                       # web (dev)
flutter run -d macos                                        # native macOS (dev)
flutter build web --wasm                                    # Skwasm; CanvasKit/dart2js fallback
flutter build macos
```

> **Interop constraint:** the web build targets `flutter build web --wasm`, which
> forbids legacy interop (`dart:html` / `dart:js_util`) anywhere in the
> dependency tree. Use `dart:js_interop` + `package:web` only
> (PRIMORDIS-ADR-007).
