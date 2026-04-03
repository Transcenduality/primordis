import pygame
import moderngl
import numpy as np

# Display and world dimensions
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1080, 720
WORLD_WIDTH, WORLD_HEIGHT = 768, 480
NUM_TYPES = 32        # fewer types = clearer roles (membrane, core, linker, etc.)
NUM_PARTICLES = 16_000

# Binning parameters
MAX_RADIUS = 32
BIN_SIZE = MAX_RADIUS
GRID_WIDTH = WORLD_WIDTH // BIN_SIZE
GRID_HEIGHT = WORLD_HEIGHT // BIN_SIZE
NUM_BINS = GRID_WIDTH * GRID_HEIGHT
MAX_BIN_PARTICLES = 256
COMPUTE_GROUP_SIZE = 512

# Density field
FIELD_CELL = 16
FIELD_W = WORLD_WIDTH // FIELD_CELL
FIELD_H = WORLD_HEIGHT // FIELD_CELL
FIELD_CELLS = FIELD_W * FIELD_H


class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, initial_val, label):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self.val = initial_val
        self.label = label
        self.dragging = False
        self.handle_radius = height // 2

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            handle_x = self.rect.x + (self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.width
            handle_rect = pygame.Rect(handle_x - self.handle_radius, self.rect.y,
                                      self.handle_radius * 2, self.rect.height)
            if handle_rect.collidepoint(mouse_pos):
                self.dragging = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            mouse_x = pygame.mouse.get_pos()[0]
            relative_x = mouse_x - self.rect.x
            relative_x = max(0, min(self.rect.width, relative_x))
            self.val = self.min_val + (relative_x / self.rect.width) * (self.max_val - self.min_val)
            return True
        return False

    def draw(self, surface, font):
        pygame.draw.rect(surface, (100, 100, 100), self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)
        handle_x = int(self.rect.x + (self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        pygame.draw.circle(surface, (255, 255, 255), (handle_x, self.rect.centery), self.handle_radius)
        pygame.draw.circle(surface, (150, 150, 150), (handle_x, self.rect.centery), self.handle_radius, 2)
        text = font.render(f"{self.label}: {self.val:.3f}", True, (255, 255, 255))
        surface.blit(text, (self.rect.x, self.rect.y - 25))


def set_parameters():
    """Generate structured interaction rules that promote cellular organisms.

    Key ideas:
    - Gaussian kernel peaks (mu, sigma) control WHERE in the radius forces are strongest
    - Asymmetric force matrix creates roles: some types form shells, some form cores
    - Chain topology (0->1->2->...->0) ensures multi-type cooperation
    """
    N = NUM_TYPES

    # === FORCE STRENGTHS (attraction/repulsion) ===
    # Start with moderate self-attraction (cohesion within a type)
    forces = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        forces[i, i] = np.random.uniform(0.3, 0.7)  # self-attract

    # Create asymmetric chain: type i strongly attracts type (i+1)%N
    # and weakly attracts type (i+2)%N. This creates layered structures.
    for i in range(N):
        forces[i, (i + 1) % N] = np.random.uniform(0.4, 0.8)   # strong pull to next
        forces[(i + 1) % N, i] = np.random.uniform(0.1, 0.3)   # weaker pull back
        forces[i, (i + 2) % N] = np.random.uniform(0.1, 0.4)   # mild pull to +2
        forces[(i + 2) % N, i] = np.random.uniform(-0.2, 0.1)  # slight repel/weak pull back

    # Opposite types repel (creates separation into inner/outer layers)
    for i in range(N):
        opp = (i + N // 2) % N
        if opp != i:
            forces[i, opp] = np.random.uniform(-0.6, -0.2)
            forces[opp, i] = np.random.uniform(-0.6, -0.2)

    # === GAUSSIAN KERNEL PARAMETERS ===
    # mu: where in [0,1] normalized distance the force peaks
    # sigma: how wide the peak is
    # This replaces the linear ramp — creates stable equilibrium distances
    mu = np.random.uniform(0.3, 0.7, (N, N)).astype(np.float32)
    sigma = np.random.uniform(0.05, 0.2, (N, N)).astype(np.float32)

    # Self-interactions peak closer (tight clusters)
    for i in range(N):
        mu[i, i] = np.random.uniform(0.2, 0.4)
        sigma[i, i] = np.random.uniform(0.08, 0.15)

    # Chain neighbors peak at medium distance (shells)
    for i in range(N):
        j = (i + 1) % N
        mu[i, j] = np.random.uniform(0.4, 0.6)
        mu[j, i] = np.random.uniform(0.5, 0.7)

    # === MIN DISTANCES (hard repulsion zone) ===
    min_distances = np.random.uniform(3, 8, (N, N)).astype(np.float32)

    # === RADII ===
    radii = np.random.uniform(24, MAX_RADIUS, (N, N)).astype(np.float32)

    # === DENSITY HOMEOSTASIS: target local density per type ===
    # Each type has an ideal neighbor count — too many = repel, too few = attract more
    target_density = np.random.uniform(6, 18, N).astype(np.float32)

    # === FIELD AFFINITY (long-range) ===
    affinity = (forces * np.random.uniform(0.05, 0.25, (N, N))).astype(np.float32)

    return forces, min_distances, radii, mu, sigma, target_density, affinity


def type_colors():
    """Distinct, visually clear colors for each type."""
    # Use HSV with even spacing for maximum distinguishability
    colors = np.zeros((NUM_TYPES, 3), dtype=np.float32)
    for i in range(NUM_TYPES):
        h = i / NUM_TYPES
        # Convert HSV to RGB (s=0.9, v=0.95)
        s, v = 0.9, 0.95
        c = v * s
        x = c * (1 - abs((h * 6) % 2 - 1))
        m = v - c
        if h < 1/6:   r, g, b = c, x, 0
        elif h < 2/6: r, g, b = x, c, 0
        elif h < 3/6: r, g, b = 0, c, x
        elif h < 4/6: r, g, b = 0, x, c
        elif h < 5/6: r, g, b = x, 0, c
        else:          r, g, b = c, 0, x
        colors[i] = [r + m, g + m, b + m]
    return colors


def main():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    pygame.display.set_caption("Particle Life — Cellular Organisms")
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE

    ui_surface = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.SRCALPHA)
    font = pygame.font.Font(None, 24)

    hide_sliders = False
    padding = 10
    button_width, button_height = 120, 30
    button_x = 50
    button_y = DISPLAY_HEIGHT - button_height - padding
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    button_font = pygame.font.Font(None, 24)

    # Streamlined sliders — fewer, more impactful controls
    sliders = []
    k_slider         = Slider(  50,  50, 200, 20,   0.1, 128.0, 8,  "Force Strength")
    friction_slider  = Slider(  50, 100, 200, 20,  0.01,  0.99,  0.99, "Friction")
    repel_slider     = Slider(  50, 150, 200, 20,   1.0,  24.0,  24,  "Repulsion Hardness")
    k_field_slider   = Slider(  50, 200, 200, 20,   0.0,  32.0,  4.0,  "Field Strength")
    blur_slider      = Slider( 300,  50, 200, 20,   1,    12,    4,    "Field Range")
    homeo_slider     = Slider( 300, 100, 200, 20,   0.0,  32.0,  2.0,  "Homeostasis Strength")
    homeo_w_slider   = Slider( 300, 150, 200, 20,   0.5,  16.0,  16.0,  "Homeostasis Width")

    sliders.extend([
        k_slider, friction_slider, repel_slider,
        k_field_slider, blur_slider,
        homeo_slider, homeo_w_slider
    ])

    # Initialize particle data
    positions = np.random.rand(NUM_PARTICLES, 2).astype(np.float32)
    positions[:, 0] *= WORLD_WIDTH
    positions[:, 1] *= WORLD_HEIGHT
    velocities = np.zeros((NUM_PARTICLES, 2), dtype=np.float32)
    types = np.random.randint(0, NUM_TYPES, NUM_PARTICLES, dtype=np.int32)
    colors = type_colors()
    per_particle_colors = colors[types]

    forces, min_distances, radii, mu, sigma, target_density, affinity = set_parameters()

    # GPU buffers
    pos_buf = ctx.buffer(positions.tobytes(), dynamic=True)
    vel_buf = ctx.buffer(velocities.tobytes(), dynamic=True)
    type_buf = ctx.buffer(types.tobytes())
    color_buf = ctx.buffer(per_particle_colors.tobytes())
    forces_buf = ctx.buffer(forces.tobytes())
    mindist_buf = ctx.buffer(min_distances.tobytes())
    radii_buf = ctx.buffer(radii.tobytes())
    mu_buf = ctx.buffer(mu.tobytes())
    sigma_buf = ctx.buffer(sigma.tobytes())
    target_density_buf = ctx.buffer(target_density.tobytes())
    affinity_buf = ctx.buffer(affinity.tobytes())
    bincounts_buf = ctx.buffer(reserve=NUM_BINS * 4, dynamic=True)
    binparts_buf = ctx.buffer(reserve=NUM_BINS * MAX_BIN_PARTICLES * 4, dynamic=True)

    # Density field buffers
    field_size = NUM_TYPES * FIELD_CELLS
    field_buf_a = ctx.buffer(reserve=field_size * 4, dynamic=True)
    field_buf_b = ctx.buffer(reserve=field_size * 4, dynamic=True)
    field_buf_float = ctx.buffer(reserve=field_size * 4, dynamic=True)

    # Rendering shaders
    vertex_shader = f"""
    #version 430
    in vec2 in_pos;
    in vec3 in_color;
    out vec3 v_color;
    void main() {{
        v_color = in_color;
        gl_Position = vec4((in_pos.x / {WORLD_WIDTH}.0) * 2.0 - 1.0,
                           (in_pos.y / {WORLD_HEIGHT}.0) * 2.0 - 1.0,
                           0.0, 1.0);
        gl_PointSize = 6.0;
    }}
    """
    fragment_shader = """
    #version 430
    in vec3 v_color;
    out vec4 fragColor;

    void main() {
        vec2 uv = gl_PointCoord * 2.0 - 1.0;
        float d = length(uv);

        // sharper core
        float core = exp(-d * d * 10.0);

        // wide strong glow
        float glow = exp(-d * d * 2.0);

        // boost intensity HARD
        vec3 color = v_color * (core * 4.0 + glow * 1.0);

        // alpha still needed for shaping
        float alpha = clamp(core + glow, 0.0, 1.0);

        fragColor = vec4(color, alpha);
    }
    """
    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    vao = ctx.vertex_array(prog, [
        (pos_buf, '2f', 'in_pos'),
        (color_buf, '3f', 'in_color'),
    ])

    # === COMPUTE SHADERS ===

    bin_shader = ctx.compute_shader(f"""
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
        int b = y * grid_width + x;
        uint off = atomicAdd(bin_counts[b], 1);
        if (off < {MAX_BIN_PARTICLES})
            bin_particles[b * {MAX_BIN_PARTICLES} + off] = i;
    }}
    """)

    deposit_shader = ctx.compute_shader(f"""
    #version 430
    layout(local_size_x = {COMPUTE_GROUP_SIZE}) in;
    layout(std430, binding=0) readonly buffer Positions {{ vec2 pos[]; }};
    layout(std430, binding=2) readonly buffer Types {{ int types[]; }};
    layout(std430, binding=9) buffer FieldA {{ uint field_a[]; }};
    uniform int num_particles;
    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i >= num_particles) return;
        vec2 p = pos[i];
        int fx = clamp(int(p.x / {FIELD_CELL}.0), 0, {FIELD_W} - 1);
        int fy = clamp(int(p.y / {FIELD_CELL}.0), 0, {FIELD_H} - 1);
        int t = types[i];
        int idx = t * {FIELD_CELLS} + fy * {FIELD_W} + fx;
        atomicAdd(field_a[idx], 1u);
    }}
    """)

    uint_to_float_shader = ctx.compute_shader(f"""
    #version 430
    layout(local_size_x = 256) in;
    layout(std430, binding=9) readonly buffer FieldUint {{ uint field_uint[]; }};
    layout(std430, binding=12) writeonly buffer FieldFloat {{ float field_float[]; }};
    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i >= {field_size}) return;
        field_float[i] = float(field_uint[i]);
    }}
    """)

    blur_h_shader = ctx.compute_shader(f"""
    #version 430
    layout(local_size_x = 256) in;
    layout(std430, binding=12) readonly buffer FieldSrc {{ float src[]; }};
    layout(std430, binding=10) writeonly buffer FieldDst {{ float dst[]; }};
    void main() {{
        uint gid = gl_GlobalInvocationID.x;
        if (gid >= {field_size}) return;
        int lc = int(gid);
        int layer = lc / {FIELD_CELLS};
        int cell = lc - layer * {FIELD_CELLS};
        int y = cell / {FIELD_W};
        int x = cell - y * {FIELD_W};
        int base = layer * {FIELD_CELLS} + y * {FIELD_W};
        int xl = (x - 1 + {FIELD_W}) % {FIELD_W};
        int xr = (x + 1) % {FIELD_W};
        dst[lc] = src[base + xl] * 0.25 + src[base + x] * 0.5 + src[base + xr] * 0.25;
    }}
    """)

    blur_v_shader = ctx.compute_shader(f"""
    #version 430
    layout(local_size_x = 256) in;
    layout(std430, binding=10) readonly buffer FieldSrc {{ float src[]; }};
    layout(std430, binding=12) writeonly buffer FieldDst {{ float dst[]; }};
    void main() {{
        uint gid = gl_GlobalInvocationID.x;
        if (gid >= {field_size}) return;
        int lc = int(gid);
        int layer = lc / {FIELD_CELLS};
        int cell = lc - layer * {FIELD_CELLS};
        int y = cell / {FIELD_W};
        int x = cell - y * {FIELD_W};
        int base = layer * {FIELD_CELLS};
        int yu = ((y - 1 + {FIELD_H}) % {FIELD_H}) * {FIELD_W} + x;
        int yd = ((y + 1) % {FIELD_H}) * {FIELD_W} + x;
        int yc = y * {FIELD_W} + x;
        dst[lc] = src[base + yu] * 0.25 + src[base + yc] * 0.5 + src[base + yd] * 0.25;
    }}
    """)

    clear_field_shader = ctx.compute_shader(f"""
    #version 430
    layout(local_size_x = 256) in;
    layout(std430, binding=9) writeonly buffer FieldA {{ uint field_a[]; }};
    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i >= {field_size}) return;
        field_a[i] = 0u;
    }}
    """)

    # === MAIN INTERACTION SHADER ===
    # Key changes:
    # 1. Gaussian kernel instead of linear ramp — force peaks at specific distance
    # 2. Density homeostasis — local neighbor count regulates attraction/repulsion
    # 3. Field gradient for long-range sensing
    interact_shader = ctx.compute_shader(f"""
    #version 430
    layout(local_size_x = {COMPUTE_GROUP_SIZE}) in;
    layout(std430, binding=0) buffer Positions {{ vec2 pos[]; }};
    layout(std430, binding=1) buffer Velocities {{ vec2 vel[]; }};
    layout(std430, binding=2) buffer Types {{ int types[]; }};
    layout(std430, binding=3) readonly buffer Forces {{ float forces[]; }};
    layout(std430, binding=4) readonly buffer MinDistances {{ float min_distances[]; }};
    layout(std430, binding=5) readonly buffer Radii {{ float radii[]; }};
    layout(std430, binding=6) buffer BinCounts {{ uint bin_counts[]; }};
    layout(std430, binding=7) buffer BinParticles {{ uint bin_particles[]; }};
    layout(std430, binding=8) readonly buffer Mu {{ float mu[]; }};
    layout(std430, binding=12) readonly buffer Field {{ float field[]; }};
    layout(std430, binding=10) readonly buffer Sigma {{ float sig[]; }};
    layout(std430, binding=11) readonly buffer Affinity {{ float affinity[]; }};
    layout(std430, binding=13) readonly buffer TargetDensity {{ float target_density[]; }};

    uniform int num_particles;
    uniform float world_width;
    uniform float world_height;
    uniform float K;
    uniform float friction;
    uniform float delta_time;
    uniform float max_radius;
    uniform float bin_size;
    uniform int grid_width;
    uniform int grid_height;
    uniform float K_field;
    uniform float K_homeo;
    uniform float homeo_width;
    uniform float repel_hardness;

    int wrap(int v, int m) {{ return (v + m) % m; }}

    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i >= num_particles) return;
        vec2 p = pos[i];
        vec2 vel_ = vel[i];
        vec2 f = vec2(0.0);
        int my_t = types[i];

        int cx = int(p.x / bin_size);
        int cy = int(p.y / bin_size);
        float half_w = world_width * 0.5;
        float half_h = world_height * 0.5;

        // Per-type neighbor counts for homeostasis
        int type_counts[{NUM_TYPES}];
        for (int t = 0; t < {NUM_TYPES}; t++) type_counts[t] = 0;
        int total_n = 0;

        // === SHORT-RANGE: Gaussian kernel forces ===
        for (int dx=-1; dx<=1; dx++) {{
            for (int dy=-1; dy<=1; dy++) {{
                int nx = wrap(cx+dx, grid_width);
                int ny = wrap(cy+dy, grid_height);
                int b = ny*grid_width + nx;
                uint cnt = bin_counts[b];
                for (uint idx=0; idx<cnt; idx++) {{
                    uint j = bin_particles[b*{MAX_BIN_PARTICLES} + idx];
                    if (j == i) continue;
                    vec2 d = pos[j] - p;
                    if (d.x > half_w) d.x -= world_width;
                    else if (d.x < -half_w) d.x += world_width;
                    if (d.y > half_h) d.y -= world_height;
                    else if (d.y < -half_h) d.y += world_height;
                    float dist = length(d);
                    if (dist < 0.1 || dist > max_radius) continue;

                    int ot = types[j];
                    type_counts[ot]++;
                    total_n++;

                    int idx2 = my_t*{NUM_TYPES} + ot;
                    float mind = min_distances[idx2];
                    float rad = radii[idx2];
                    float fs = forces[idx2];
                    float m = mu[idx2];
                    float s = sig[idx2];
                    vec2 dn = d / dist;

                    // Zone 1: hard repulsion below min_distance
                    if (dist < mind) {{
                        f -= dn * repel_hardness * (1.0 - dist/mind);
                    }}
                    // Zone 2: Gaussian kernel — force peaks at dist/rad == mu
                    else {{
                        float r_norm = dist / rad;
                        float kernel = exp(-0.5 * pow((r_norm - m) / s, 2.0));
                        f += dn * fs * kernel * K;
                    }}
                }}
            }}
        }}

        // === DENSITY HOMEOSTASIS ===
        // Each particle adjusts based on how crowded its neighborhood is
        // compared to the target density for its type.
        // Too crowded -> push away from center of mass
        // Too sparse -> pull toward center of mass
        float my_target = target_density[my_t];
        float local_density = float(total_n);
        // Gaussian growth function: peaks at target, falls off on both sides
        float density_error = (local_density - my_target) / homeo_width;
        // Negative = too sparse (attract more), positive = too crowded (repel)
        // We apply this as a radial force toward/away from local COM
        if (total_n > 0) {{
            vec2 com = vec2(0.0);
            // Recompute COM (cheaper than storing during loop for GPU)
            // Actually, approximate from our position + average displacement
            // Use a simpler approach: push away from nearest neighbors when crowded
            float homeo_force = -tanh(density_error) * K_homeo;
            // When homeo_force > 0: attract (too sparse)
            // When homeo_force < 0: repel (too crowded)
            // Apply as velocity damping/amplification instead of directional force
            // This naturally regulates cluster density without needing COM
            vel_ *= (1.0 + homeo_force * delta_time);
        }}

        // === LONG-RANGE: density field gradient ===
        int fix = clamp(int(p.x / {FIELD_CELL}.0), 0, {FIELD_W} - 1);
        int fiy = clamp(int(p.y / {FIELD_CELL}.0), 0, {FIELD_H} - 1);

        vec2 field_force = vec2(0.0);
        for (int t = 0; t < {NUM_TYPES}; t++) {{
            float aff = affinity[my_t * {NUM_TYPES} + t];
            if (abs(aff) < 0.001) continue;
            int base = t * {FIELD_CELLS};
            int xl = (fix - 1 + {FIELD_W}) % {FIELD_W};
            int xr = (fix + 1) % {FIELD_W};
            int yu = ((fiy - 1 + {FIELD_H}) % {FIELD_H}) * {FIELD_W};
            int yd = ((fiy + 1) % {FIELD_H}) * {FIELD_W};
            float gx = field[base + fiy * {FIELD_W} + xr] - field[base + fiy * {FIELD_W} + xl];
            float gy = field[base + yd + fix] - field[base + yu + fix];
            field_force += vec2(gx, gy) * aff;
        }}
        f += field_force * K_field;

        // Integrate
        vel_ += f * delta_time;
        vel_ *= friction;
        float spd = length(vel_);
        if (spd > 48.0) vel_ *= 48.0 / spd;
        p += vel_ * delta_time;
        if (p.x < 0) p.x += world_width; else if (p.x >= world_width) p.x -= world_width;
        if (p.y < 0) p.y += world_height; else if (p.y >= world_height) p.y -= world_height;
        pos[i] = p;
        vel[i] = vel_;
    }}
    """)

    clear_shader = ctx.compute_shader(f"""
    #version 430
    layout(local_size_x=256) in;
    layout(std430,binding=6) buffer Bins{{uint bin_counts[];}};
    uniform int num_bins;
    void main(){{uint i=gl_GlobalInvocationID.x; if(i<num_bins)bin_counts[i]=0;}}
    """)

    # Bind buffers — careful with slot assignments
    pos_buf.bind_to_storage_buffer(0)
    vel_buf.bind_to_storage_buffer(1)
    type_buf.bind_to_storage_buffer(2)
    forces_buf.bind_to_storage_buffer(3)
    mindist_buf.bind_to_storage_buffer(4)
    radii_buf.bind_to_storage_buffer(5)
    bincounts_buf.bind_to_storage_buffer(6)
    binparts_buf.bind_to_storage_buffer(7)
    mu_buf.bind_to_storage_buffer(8)
    field_buf_a.bind_to_storage_buffer(9)       # uint deposit
    sigma_buf.bind_to_storage_buffer(10)        # NOTE: shared with blur ping-pong
    affinity_buf.bind_to_storage_buffer(11)
    field_buf_float.bind_to_storage_buffer(12)  # float field
    target_density_buf.bind_to_storage_buffer(13)

    # Set uniforms
    params = {
        'num_particles': NUM_PARTICLES,
        'world_width': float(WORLD_WIDTH),
        'world_height': float(WORLD_HEIGHT),
        'max_radius': float(MAX_RADIUS),
        'bin_size': float(BIN_SIZE),
        'grid_width': GRID_WIDTH,
        'grid_height': GRID_HEIGHT,
        'num_bins': NUM_BINS
    }
    for name, val in params.items():
        for shader in [bin_shader, interact_shader, clear_shader]:
            if name in shader:
                shader[name].value = val
    deposit_shader['num_particles'].value = NUM_PARTICLES

    def update_uniforms():
        interact_shader['K'].value = k_slider.val
        interact_shader['friction'].value = friction_slider.val
        interact_shader['repel_hardness'].value = repel_slider.val
        interact_shader['K_field'].value = k_field_slider.val
        interact_shader['K_homeo'].value = homeo_slider.val
        interact_shader['homeo_width'].value = homeo_w_slider.val

    update_uniforms()

    field_groups = (field_size + 255) // 256
    particle_groups = (NUM_PARTICLES + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE

    clock = pygame.time.Clock()
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    hide_sliders = not hide_sliders
                    continue
            if not hide_sliders:
                for s in sliders:
                    if s.handle_event(event):
                        update_uniforms()
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
                running = False

        ctx.clear(0.02, 0.02, 0.08, 1.0)

        # Clear bins + field
        clear_shader.run(group_x=(NUM_BINS + 255) // 256)
        clear_field_shader.run(group_x=field_groups)
        ctx.finish()

        # Bin + deposit
        bin_shader.run(group_x=particle_groups)
        deposit_shader.run(group_x=particle_groups)
        ctx.finish()

        # uint -> float
        uint_to_float_shader.run(group_x=field_groups)
        ctx.finish()

        # Blur passes (swap binding 10 to field_buf_b, then restore to sigma)
        blur_passes = max(1, int(blur_slider.val))
        field_buf_b.bind_to_storage_buffer(10)
        for _ in range(blur_passes):
            blur_h_shader.run(group_x=field_groups)
            ctx.finish()
            blur_v_shader.run(group_x=field_groups)
            ctx.finish()
        # Restore sigma to binding 10 for interact shader
        sigma_buf.bind_to_storage_buffer(10)

        # Main interaction
        interact_shader['delta_time'].value = dt
        interact_shader.run(group_x=particle_groups)
        vao.render(mode=moderngl.POINTS)

        # UI overlay
        ui_surface.fill((0, 0, 0, 0))
        btn_color = (180, 180, 180) if not hide_sliders else (100, 100, 100)
        pygame.draw.rect(ui_surface, btn_color, button_rect)
        label = "Hide Sliders" if not hide_sliders else "Show Sliders"
        txt_surf = button_font.render(label, True, (0, 0, 0))
        txt_rect = txt_surf.get_rect(center=button_rect.center)
        ui_surface.blit(txt_surf, txt_rect)
        if not hide_sliders:
            for s in sliders:
                s.draw(ui_surface, font)

        ui_data = pygame.image.tostring(ui_surface, 'RGBA')
        ui_tex = ctx.texture((DISPLAY_WIDTH, DISPLAY_HEIGHT), 4, ui_data)
        ui_tex.use(0)

        if not hasattr(main, 'ui_prog'):
            vs = '''
            #version 430
            in vec2 in_position;
            in vec2 in_texcoord;
            out vec2 v_texcoord;
            void main() {
                gl_Position = vec4(in_position, 0.0, 1.0);
                v_texcoord = in_texcoord;
            }
            '''
            fs = '''
            #version 430
            uniform sampler2D ui_texture;
            in vec2 v_texcoord;
            out vec4 fragColor;
            void main() {
                fragColor = texture(ui_texture, v_texcoord);
            }
            '''
            main.ui_prog = ctx.program(vertex_shader=vs, fragment_shader=fs)
            quad = np.array([
                -1, -1, 0, 1,
                 1, -1, 1, 1,
                -1,  1, 0, 0,
                 1,  1, 1, 0,
            ], dtype=np.float32)
            main.ui_vbo = ctx.buffer(quad.tobytes())
            main.ui_vao = ctx.vertex_array(
                main.ui_prog,
                [(main.ui_vbo, '2f 2f', 'in_position', 'in_texcoord')]
            )

        main.ui_prog['ui_texture'].value = 0
        main.ui_vao.render(mode=moderngl.TRIANGLE_STRIP)
        ui_tex.release()

        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
