#!/usr/bin/env python3
"""Capture a parity reference fingerprint from the original ``Primordis.py``.

This runs the reference simulation's *exact* compute shaders headlessly on a
machine with an OpenGL 4.3 context, from the byte-identical seeded initial
condition emitted by ``tool/parity/export_seed.dart``, and exports a raw
per-checkpoint SoA snapshot in the schema the Dart parity harness ingests
(``test/parity/wgsl_kernel_parity_test.dart`` / ``raw_snapshot.dart``).

Why raw snapshots and not reduced metrics: the Dart side computes every metric
once (in ``lib/sim/parity/frame_metrics.dart``); re-implementing them in Python
would risk divergence and undermine the comparison. So Python only emits the raw
particle buffers; Dart reduces them with the shared metric code.

Parity is **statistical, never bit-exact** (PRIMORDIS-ADR-001): the reference
GPU binning is a single-buffered atomic scatter with a known race, so two runs
of the reference itself differ per-particle. The captured fingerprint is a
committed baseline; the Dart CPU tier is compared to it within tolerance bands,
and the reference run is compared to the CPU/GPU tiers within the *looser*
cross-backend band.

Prerequisites (NOT available in the pure-Dart CI job — run locally once and
commit the fixture):
    pip install numpy moderngl
    # plus an OpenGL 4.3 core context. On a headless box use EGL/OSMesa or a
    # virtual display; ``moderngl.create_standalone_context`` avoids pygame.

Usage:
    # 1. export the shared seed (byte-identical initial condition)
    dart run tool/parity/export_seed.dart --seed 42 --particles 3000 --types 32 \
        > /tmp/seed.json
    # 2. capture the reference fingerprint
    python3 tool/parity/capture_reference.py \
        --seed /tmp/seed.json \
        --out test/parity/fixtures/py_reference_snapshots.json \
        --attraction 32 --repulsion 32 --friction 0.25 \
        --steps 240 --checkpoints early=0,mid=60,steady=239

The scenario flags MUST match ``ParityHarnessConfig.defaults()`` in
``test/parity/parity_harness_support.dart``.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

# The reference binning/interaction compute-shader source, lifted verbatim from
# Primordis.py so this capture validates the SAME kernel the reference ships.
# (Kept inline so the capture is self-contained; if Primordis.py's shaders
# change, re-copy them here and regenerate the fixture.)
MAX_BIN_PARTICLES = 512
COMPUTE_GROUP_SIZE = 256


def _binning_shader(num_types: int) -> str:
    return f"""
    #version 430
    layout(local_size_x = {COMPUTE_GROUP_SIZE}) in;
    layout(std430, binding=0) buffer Positions {{ vec2 pos[]; }};
    layout(std430, binding=6) buffer BinCounts {{ uint bin_counts[]; }};
    layout(std430, binding=7) buffer BinParticles {{ uint bin_particles[]; }};
    uniform float bin_size;
    uniform int grid_width;
    uniform int grid_height;
    uniform int num_particles;
    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i >= num_particles) return;
        vec2 p = pos[i];
        int x = int(p.x / bin_size);
        int y = int(p.y / bin_size);
        x = clamp(x, 0, grid_width - 1);
        y = clamp(y, 0, grid_height - 1);
        int bin_idx = y * grid_width + x;
        uint offset = atomicAdd(bin_counts[bin_idx], 1);
        if (offset < {MAX_BIN_PARTICLES})
            bin_particles[bin_idx * {MAX_BIN_PARTICLES} + offset] = i;
    }}
    """


def _interaction_shader(num_types: int) -> str:
    return f"""
    #version 430
    layout(local_size_x = {COMPUTE_GROUP_SIZE}) in;
    layout(std430, binding=0) buffer Positions {{ vec2 pos[]; }};
    layout(std430, binding=1) buffer Velocities {{ vec2 vel[]; }};
    layout(std430, binding=2) buffer Types {{ int types[]; }};
    layout(std430, binding=3) readonly buffer Forces {{ float forces[]; }};
    layout(std430, binding=4) readonly buffer MinDistances {{ float min_distances[]; }};
    layout(std430, binding=5) readonly buffer Radii {{ float radii[]; }};
    layout(std430, binding=6) readonly buffer BinCounts {{ uint bin_counts[]; }};
    layout(std430, binding=7) readonly buffer BinParticles {{ uint bin_particles[]; }};
    uniform int num_particles;
    uniform float world_width;
    uniform float world_height;
    uniform float K_attraction;
    uniform float K_repulsion;
    uniform float friction;
    uniform float delta_time;
    uniform float max_radius;
    uniform float bin_size;
    uniform int grid_width;
    uniform int grid_height;
    int wrap(int val, int max_val) {{ return (val + max_val) % max_val; }}
    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i >= num_particles) return;
        vec2 p = pos[i];
        vec2 v = vel[i];
        vec2 f = vec2(0.0);
        int my_type = types[i];
        int cx = int(p.x / bin_size);
        int cy = int(p.y / bin_size);
        float half_world_width = world_width * 0.5;
        float half_world_height = world_height * 0.5;
        for (int dx = -1; dx <= 1; dx++) {{
            int nx = wrap(cx + dx, grid_width);
            for (int dy = -1; dy <= 1; dy++) {{
                int ny = wrap(cy + dy, grid_height);
                int bin_idx = ny * grid_width + nx;
                uint count = bin_counts[bin_idx];
                for (uint b = 0u; b < count; b++) {{
                    uint j = bin_particles[bin_idx * {MAX_BIN_PARTICLES} + b];
                    if (j == i) continue;
                    vec2 d = pos[j] - p;
                    if (d.x > half_world_width) d.x -= world_width;
                    else if (d.x < -half_world_width) d.x += world_width;
                    if (d.y > half_world_height) d.y -= world_height;
                    else if (d.y < -half_world_height) d.y += world_height;
                    float dist = length(d);
                    if (dist > max_radius || dist < 0.1) continue;
                    vec2 dn = d / dist;
                    int other_type = types[j];
                    int idx = my_type * {num_types} + other_type;
                    float mind = min_distances[idx];
                    float rad = radii[idx];
                    float force_strength = forces[idx];
                    if (dist < mind) {{
                        f -= dn * abs(force_strength) * 5.0 * (1.0 - dist / mind) * K_repulsion;
                    }} else if (dist < rad) {{
                        f += dn * force_strength * (1.0 - dist / rad) * K_attraction;
                    }}
                }}
            }}
        }}
        v += f * delta_time;
        v *= friction;
        p += v * delta_time;
        if (p.x < 0.0) p.x += world_width;
        else if (p.x >= world_width) p.x -= world_width;
        if (p.y < 0.0) p.y += world_height;
        else if (p.y >= world_height) p.y -= world_height;
        pos[i] = p;
        vel[i] = v;
    }}
    """


def _clear_shader() -> str:
    return """
    #version 430
    layout(local_size_x = 256) in;
    layout(std430, binding=6) buffer BinCounts { uint bin_counts[]; };
    uniform int num_bins;
    void main() {
        uint i = gl_GlobalInvocationID.x;
        if (i < num_bins) bin_counts[i] = 0;
    }
    """


def parse_checkpoints(spec: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for pair in spec.split(","):
        name, _, step = pair.partition("=")
        out[name.strip()] = int(step)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", required=True, help="path to export_seed.dart JSON")
    ap.add_argument("--out", required=True, help="output fixture path")
    ap.add_argument("--attraction", type=float, default=32.0)
    ap.add_argument("--repulsion", type=float, default=32.0)
    ap.add_argument("--friction", type=float, default=0.25)
    ap.add_argument("--dt", type=float, default=1.0 / 60.0)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--checkpoints", default="early=0,mid=60,steady=239")
    args = ap.parse_args()

    try:
        import moderngl
    except ImportError:
        print("moderngl is required: pip install moderngl numpy", file=sys.stderr)
        return 1

    with open(args.seed) as fh:
        seed = json.load(fh)

    num_particles = int(seed["particleCount"])
    num_types = int(seed["typeCount"])
    world_width = float(seed["worldWidth"])
    world_height = float(seed["worldHeight"])
    bin_size = 96.0
    grid_width = int(world_width // bin_size)
    grid_height = int(world_height // bin_size)
    num_bins = grid_width * grid_height

    positions = np.array(seed["positions"], dtype=np.float32).reshape(-1, 2)
    velocities = np.array(seed["velocities"], dtype=np.float32).reshape(-1, 2)
    types = np.array(seed["types"], dtype=np.int32)
    forces = np.array(seed["forces"], dtype=np.float32)
    min_distances = np.array(seed["minDistances"], dtype=np.float32)
    radii = np.array(seed["radii"], dtype=np.float32)

    ctx = moderngl.create_standalone_context(require=430)

    pos_buf = ctx.buffer(positions.tobytes(), dynamic=True)
    vel_buf = ctx.buffer(velocities.tobytes(), dynamic=True)
    type_buf = ctx.buffer(types.tobytes())
    forces_buf = ctx.buffer(forces.tobytes())
    min_dist_buf = ctx.buffer(min_distances.tobytes())
    radii_buf = ctx.buffer(radii.tobytes())
    bin_counts_buf = ctx.buffer(reserve=num_bins * 4, dynamic=True)
    bin_particles_buf = ctx.buffer(
        reserve=num_bins * MAX_BIN_PARTICLES * 4, dynamic=True
    )

    binning = ctx.compute_shader(_binning_shader(num_types))
    interaction = ctx.compute_shader(_interaction_shader(num_types))
    clear = ctx.compute_shader(_clear_shader())

    for i, buf in enumerate(
        [pos_buf, vel_buf, type_buf, forces_buf, min_dist_buf, radii_buf,
         bin_counts_buf, bin_particles_buf]
    ):
        buf.bind_to_storage_buffer(i)

    for name, value in [
        ("num_particles", num_particles),
        ("world_width", world_width),
        ("world_height", world_height),
        ("K_attraction", args.attraction),
        ("K_repulsion", args.repulsion),
        ("friction", args.friction),
        ("max_radius", 96.0),
        ("bin_size", bin_size),
        ("grid_width", grid_width),
        ("grid_height", grid_height),
    ]:
        if name in interaction:
            interaction[name].value = value
        if name in binning:
            binning[name].value = value
    clear["num_bins"].value = num_bins
    interaction["delta_time"].value = args.dt

    checkpoints = parse_checkpoints(args.checkpoints)
    by_step: dict[int, list[str]] = {}
    for name, step in checkpoints.items():
        by_step.setdefault(step, []).append(name)

    def snapshot() -> dict:
        pos = np.frombuffer(pos_buf.read(), dtype=np.float32)
        vel = np.frombuffer(vel_buf.read(), dtype=np.float32)
        return {
            "particleCount": num_particles,
            "positions": pos.tolist(),
            "velocities": vel.tolist(),
            "types": types.tolist(),
        }

    captured: dict[str, dict] = {}
    for name in by_step.get(0, []):
        captured[name] = snapshot()
    for s in range(1, args.steps):
        clear.run(group_x=(num_bins + 255) // 256)
        binning.run(
            group_x=(num_particles + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE
        )
        interaction.run(
            group_x=(num_particles + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE
        )
        for name in by_step.get(s, []):
            captured[name] = snapshot()

    out = {
        "label": "reference",
        "seed": int(seed["seed"]),
        "particleCount": num_particles,
        "typeCount": num_types,
        "attractionK": args.attraction,
        "repulsionK": args.repulsion,
        "friction": args.friction,
        "checkpoints": captured,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
