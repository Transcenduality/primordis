import pygame
import moderngl
import numpy as np

DISPLAY_WIDTH, DISPLAY_HEIGHT = 1080, 720
WORLD_WIDTH, WORLD_HEIGHT = 1080, 720
NUM_PARTICLES = 16_000
ATTRACTION_K = 64.0          
RATIO_INIT = 1             
MAX_RADIUS = 128
COMPUTE_GROUP_SIZE = 256
BIN_SIZE = 64
SEARCH_RANGE = (MAX_RADIUS + BIN_SIZE - 1) // BIN_SIZE
GRID_WIDTH = (WORLD_WIDTH + BIN_SIZE - 1) // BIN_SIZE
GRID_HEIGHT = (WORLD_HEIGHT + BIN_SIZE - 1) // BIN_SIZE
NUM_BINS = GRID_WIDTH * GRID_HEIGHT
MAX_BIN_PARTICLES = 256

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
        handle_x = self.rect.x + (self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        pygame.draw.circle(surface, (255, 255, 255), 
                         (int(handle_x), self.rect.centery), self.handle_radius)
        pygame.draw.circle(surface, (150, 150, 150), 
                         (int(handle_x), self.rect.centery), self.handle_radius, 2)
        label_text = font.render(f"{self.label}: {self.val:.3f}", True, (255, 255, 255))
        surface.blit(label_text, (self.rect.x, self.rect.y - 25))

def main():
    # Camera state
    cam_x = WORLD_WIDTH / 2.0
    cam_y = WORLD_HEIGHT / 2.0
    zoom = 1.0
    PAN_SPEED = 300.0
    ZOOM_SPEED = 2.0

    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    ui_surface = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.SRCALPHA)
    font = pygame.font.Font(None, 24)
    
    ratio_slider = Slider(50, 50, 200, 20, 0.5, 0.99, RATIO_INIT, "Repulsion/Attraction Ratio")
    friction_slider = Slider(50, 100, 200, 20, 0.05, 0.99, 0.25, "Particle Drift Strength")
    sliders = [ratio_slider, friction_slider]

    # Initialize Particle Data
    positions = np.random.rand(NUM_PARTICLES, 2).astype(np.float32)
    positions[:, 0] *= WORLD_WIDTH
    positions[:, 1] *= WORLD_HEIGHT
    velocities = np.random.uniform(-8, 8, (NUM_PARTICLES, 2)).astype(np.float32)
    
    # Initialize genomes: 6-dimensional unit vectors representing genetic information
    raw_genomes = np.random.uniform(-1, 1, (NUM_PARTICLES, 6)).astype(np.float32)
    norms = np.linalg.norm(raw_genomes, axis=1, keepdims=True)
    genomes = raw_genomes / norms

    # Buffers
    pos_buffer = ctx.buffer(positions.tobytes(), dynamic=True)
    vel_buffer = ctx.buffer(velocities.tobytes(), dynamic=True)
    genome_buffer = ctx.buffer(genomes.tobytes(), dynamic=True)
    bin_counts_buf = ctx.buffer(reserve=NUM_BINS * 4, dynamic=True)
    bin_particles_buf = ctx.buffer(reserve=NUM_BINS * MAX_BIN_PARTICLES * 4, dynamic=True)

    # Shaders
    vertex_shader = f'''
    #version 430
    in vec2 in_pos;
    in vec4 in_genome;
    out vec3 v_color;
    uniform vec2 cam;
    uniform float zoom;
    uniform vec2 world;
    void main() {{
        // Shift genome from [-1, 1] to [0, 1] for smooth color transitions
        v_color = in_genome.xyz * 0.5 + 0.5; 
        
        vec2 delta = in_pos - cam;
        delta = delta - world * round(delta / world);
        gl_Position = vec4(delta.x * (2.0 * zoom / world.x),
                           delta.y * (2.0 * zoom / world.y),
                           0.0, 1.0);
        gl_PointSize = 3.0 * zoom;
    }}
    '''

    fragment_shader = '''
    #version 430
    in vec3 v_color;
    out vec4 fragColor;
    void main() {
        vec2 p = gl_PointCoord - vec2(0.5);
        float dist = length(p);
        if (dist > 0.5) discard;
        float alpha = 1.0 - smoothstep(0.45, 0.5, dist);
        fragColor = vec4(v_color, alpha);
    }
    '''

    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    vao = ctx.vertex_array(prog, [(pos_buffer, '2f', 'in_pos'), (genome_buffer, '4f 8x', 'in_genome')])

    binning_shader = ctx.compute_shader(f'''
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
        int x = clamp(int(p.x / bin_size), 0, grid_width - 1);
        int y = clamp(int(p.y / bin_size), 0, grid_height - 1);
        int bin_idx = y * grid_width + x;
        uint offset = atomicAdd(bin_counts[bin_idx], 1);
        if (offset < {MAX_BIN_PARTICLES})
            bin_particles[bin_idx * {MAX_BIN_PARTICLES} + offset] = i;
    }}
    ''')

    interaction_shader = ctx.compute_shader(f'''
    #version 430
    layout(local_size_x = {COMPUTE_GROUP_SIZE}) in;
    layout(std430, binding=0) buffer Positions {{ vec2 pos[]; }};
    layout(std430, binding=1) buffer Velocities {{ vec2 vel[]; }};
    layout(std430, binding=2) buffer Genomes {{ float genomes[]; }};
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
    float rand(vec2 co){{ return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453); }}

    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i >= num_particles) return;

        vec2 p = pos[i];
        vec2 v = vel[i];

        float my_genome[6];
        for(int g = 0; g < 6; g++) {{
            my_genome[g] = genomes[i * 6 + g];
        }}

        // ---- Accumulated genetic repulsion (divergence) ----
        float genetic_repulsion[6] = float[6](0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        const float divergence_strength = 0.1;   // tune this slider-like value

        vec2 f = vec2(0.0);
        int cx = int(p.x / bin_size);
        int cy = int(p.y / bin_size);
        float half_w = world_width * 0.5;
        float half_h = world_height * 0.5;

        const int sr = {SEARCH_RANGE};
        for (int dx = -sr; dx <= sr; dx++) {{
            int nx = wrap(cx + dx, grid_width);
            for (int dy = -sr; dy <= sr; dy++) {{
                int ny = wrap(cy + dy, grid_height);
                int bin_idx = ny * grid_width + nx;
                uint count = bin_counts[bin_idx];
                for (uint b = 0u; b < count; b++) {{
                    uint j = bin_particles[bin_idx * {MAX_BIN_PARTICLES} + b];
                    if (j == i) continue;

                    vec2 d = pos[j] - p;
                    if (d.x > half_w) d.x -= world_width;
                    else if (d.x < -half_w) d.x += world_width;
                    if (d.y > half_h) d.y -= world_height;
                    else if (d.y < -half_h) d.y += world_height;

                    float dist = length(d);
                    if (dist > max_radius || dist < 0.1) continue;
                    vec2 dn = d / dist;

                    float other_genome[6];
                    float compatibility = 0.0;
                    for(int g = 0; g < 6; g++) {{
                        other_genome[g] = genomes[j * 6 + g];
                        compatibility += my_genome[g] * other_genome[g];
                    }}

                    // Personal space (based on the 4th gene)
                    float mind = 8.0 + 2.0 * (other_genome[3] * 0.5 + 0.5);

                    if (dist < mind) {{
                        f -= dn * 8.0 * (1.0 - dist / mind) * K_repulsion;
                    }} else if (dist < max_radius) {{
                        f += dn * compatibility * (1.0 - dist / max_radius) * K_attraction;

                        // ---- NEW: genetic divergence instead of homogenisation ----
                        float similarity = compatibility;  // [-1, 1]
                        float threshold = 0.5;              // start diverging above this
                        if (similarity > threshold) {{
                            float force = (similarity - threshold) * divergence_strength;
                            for(int g = 0; g < 6; g++) {{
                                // Push my_genome AWAY from other_genome
                                genetic_repulsion[g] += (my_genome[g] - other_genome[g]) * force;
                            }}
                        }}
                    }}
                }}
            }}
        }}

        float max_force = 2000.0;
        if (length(f) > max_force) f = normalize(f) * max_force;

        v += f * delta_time;
        v *= friction;

        float max_vel = 150.0;
        if (length(v) > max_vel) v = normalize(v) * max_vel;

        p += v * delta_time;

        if (p.x < 0.0) p.x += world_width;
        else if (p.x >= world_width) p.x -= world_width;
        if (p.y < 0.0) p.y += world_height;
        else if (p.y >= world_height) p.y -= world_height;

        pos[i] = p;
        vel[i] = v;

        // Apply genetic repulsion, and re‑normalise
        float drift_speed = 0.5;   // same as before, now directly used
        float len_sq = 0.0;
        for(int g = 0; g < 6; g++) {{
            my_genome[g] += genetic_repulsion[g] * drift_speed * delta_time;
            len_sq += my_genome[g] * my_genome[g];
        }}

        float inv_len = inversesqrt(len_sq);
        for(int g = 0; g < 6; g++) {{
            genomes[i * 6 + g] = my_genome[g] * inv_len;
        }}
    }}
    ''')

    clear_counts_shader = ctx.compute_shader(f'''
    #version 430
    layout(local_size_x = 256) in;
    layout(std430, binding=6) buffer BinCounts {{ uint bin_counts[]; }};
    uniform int num_bins;
    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i < num_bins) bin_counts[i] = 0;
    }}
    ''')

    # Bind storage buffers
    buffers = [pos_buffer, vel_buffer, genome_buffer, None, None, None,
               bin_counts_buf, bin_particles_buf]
    for i, buf in enumerate(buffers):
        if buf is not None:
            buf.bind_to_storage_buffer(i)

    def update_forces():
        attraction = ATTRACTION_K
        repulsion = attraction * ratio_slider.val
        interaction_shader['K_attraction'].value = attraction
        interaction_shader['K_repulsion'].value = repulsion
        interaction_shader['friction'].value = friction_slider.val

    update_forces()

    for name, value in [
        ('num_particles', NUM_PARTICLES),
        ('world_width', float(WORLD_WIDTH)),
        ('world_height', float(WORLD_HEIGHT)),
        ('max_radius', float(MAX_RADIUS)),
        ('bin_size', float(BIN_SIZE)),
        ('grid_width', GRID_WIDTH),
        ('grid_height', GRID_HEIGHT),
    ]:
        interaction_shader[name].value = value
        if name in binning_shader:
            binning_shader[name].value = value

    clear_counts_shader['num_bins'].value = NUM_BINS
    prog['cam'] = (cam_x, cam_y)
    prog['zoom'] = zoom
    prog['world'] = (float(WORLD_WIDTH), float(WORLD_HEIGHT))

    ui_texture = ctx.texture((DISPLAY_WIDTH, DISPLAY_HEIGHT), 4, dtype='f1')
    ui_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    ui_vertex_shader = '''
    #version 430
    in vec2 in_position;
    in vec2 in_texcoord;
    out vec2 v_texcoord;
    void main() {
        gl_Position = vec4(in_position, 0.0, 1.0);
        v_texcoord = in_texcoord;
    }
    '''
    ui_fragment_shader = '''
    #version 430
    uniform sampler2D ui_texture;
    in vec2 v_texcoord;
    out vec4 fragColor;
    void main() {
        fragColor = texture(ui_texture, v_texcoord);
    }
    '''
    ui_prog = ctx.program(vertex_shader=ui_vertex_shader, fragment_shader=ui_fragment_shader)
    quad_vertices = np.array([
        -1.0, -1.0, 0.0, 1.0,
         1.0, -1.0, 1.0, 1.0,
        -1.0,  1.0, 0.0, 0.0,
         1.0,  1.0, 1.0, 0.0
    ], dtype=np.float32)
    ui_vbo = ctx.buffer(quad_vertices.tobytes())
    ui_vao = ctx.vertex_array(ui_prog, [(ui_vbo, '2f 2f', 'in_position', 'in_texcoord')])

    clock = pygame.time.Clock()
    sim_speed = 1
    running = True
    total_time = 0.0

    while running:
        dt = clock.tick(60) / 1000.0
        total_time += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]):
                running = False
            for slider in sliders:
                if slider.handle_event(event):
                    update_forces()

        # Camera controls
        keys = pygame.key.get_pressed()
        pan_world_speed = PAN_SPEED / zoom
        if keys[pygame.K_w]:    cam_y += pan_world_speed * dt
        if keys[pygame.K_s]:    cam_y -= pan_world_speed * dt
        if keys[pygame.K_a]:    cam_x -= pan_world_speed * dt
        if keys[pygame.K_d]:    cam_x += pan_world_speed * dt

        zoom_factor = 1.0 + ZOOM_SPEED * dt
        if keys[pygame.K_UP]:    zoom *= zoom_factor
        if keys[pygame.K_DOWN]:  zoom /= zoom_factor
        zoom = max(zoom, 1.0)

        prog['cam'] = (cam_x, cam_y)
        prog['zoom'] = zoom

        # Simulation Loop
        ctx.clear(0.02, 0.02, 0.08, 1.0)
        clear_counts_shader.run(group_x=(NUM_BINS + 255) // 256)
        binning_shader.run(group_x=(NUM_PARTICLES + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE)
        ctx.memory_barrier(barriers=moderngl.SHADER_STORAGE_BARRIER_BIT)
        
        interaction_shader['delta_time'].value = dt * sim_speed
        interaction_shader.run(group_x=(NUM_PARTICLES + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE)
        
        vao.render(moderngl.POINTS)

        # UI rendering
        ui_surface.fill((0, 0, 0, 0))
        for slider in sliders:
            slider.draw(ui_surface, font)
        ui_data = pygame.image.tostring(ui_surface, 'RGBA')
        ui_texture.write(ui_data)
        ui_texture.use(0)
        ui_prog['ui_texture'].value = 0
        ui_vao.render(moderngl.TRIANGLE_STRIP)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
