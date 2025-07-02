import pygame
import moderngl
import numpy as np

DISPLAY_WIDTH, DISPLAY_HEIGHT = 1920, 1080
WORLD_WIDTH, WORLD_HEIGHT = 1080, 720
NUM_TYPES = 32
NUM_PARTICLES = 16_000
K = 32
MAX_RADIUS = 64
COMPUTE_GROUP_SIZE = 512
BIN_SIZE = MAX_RADIUS
GRID_WIDTH = WORLD_WIDTH // BIN_SIZE
GRID_HEIGHT = WORLD_HEIGHT // BIN_SIZE
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
        # Draw slider track
        pygame.draw.rect(surface, (100, 100, 100), self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)
        
        # Draw handle
        handle_x = self.rect.x + (self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.width
        pygame.draw.circle(surface, (255, 255, 255), 
                         (int(handle_x), self.rect.centery), self.handle_radius)
        pygame.draw.circle(surface, (150, 150, 150), 
                         (int(handle_x), self.rect.centery), self.handle_radius, 2)
        
        # Draw label and value
        label_text = font.render(f"{self.label}: {self.val:.3f}", True, (255, 255, 255))
        surface.blit(label_text, (self.rect.x, self.rect.y - 25))

def set_parameters():
    forces = np.random.uniform(0.1, 0.8, (NUM_TYPES, NUM_TYPES)).astype(np.float32)
    mask = np.random.random((NUM_TYPES, NUM_TYPES)) < 0.5
    forces[mask] *= -1
    min_distances = np.random.uniform(4, 12, (NUM_TYPES, NUM_TYPES)).astype(np.float32)
    radii = np.random.uniform(20, MAX_RADIUS, (NUM_TYPES, NUM_TYPES)).astype(np.float32)
    return forces, min_distances, radii

def random_type_colors():
    return np.random.rand(NUM_TYPES, 3).astype(np.float32)

def main():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 4)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # Create UI surface for sliders
    ui_surface = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.SRCALPHA)
    font = pygame.font.Font(None, 24)
    
    # Create sliders
    k_slider = Slider(50, 50, 200, 20, 0.1, 128.0, K, "Interaction Strength")
    friction_slider = Slider(50, 120, 200, 20, 0.05, 0.99, 0.75, "Particle Drift Strength")
    sliders = [k_slider, friction_slider]

    positions = np.random.rand(NUM_PARTICLES, 2).astype(np.float32)
    positions[:, 0] *= WORLD_WIDTH
    positions[:, 1] *= WORLD_HEIGHT
    velocities = np.random.uniform(-8, 8, (NUM_PARTICLES, 2)).astype(np.float32)
    types = np.random.randint(0, NUM_TYPES, NUM_PARTICLES, dtype=np.int32)
    colors = random_type_colors()
    per_particle_colors = colors[types]

    forces, min_distances, radii = set_parameters()

    pos_buffer = ctx.buffer(positions.tobytes(), dynamic=True)
    vel_buffer = ctx.buffer(velocities.tobytes(), dynamic=True)
    type_buffer = ctx.buffer(types.tobytes())
    color_buffer = ctx.buffer(per_particle_colors.tobytes())

    forces_buf = ctx.buffer(forces.tobytes())
    min_dist_buf = ctx.buffer(min_distances.tobytes())
    radii_buf = ctx.buffer(radii.tobytes())

    bin_counts_buf = ctx.buffer(reserve=NUM_BINS * 4, dynamic=True)
    bin_particles_buf = ctx.buffer(reserve=NUM_BINS * MAX_BIN_PARTICLES * 4, dynamic=True)

    vertex_shader = f'''
    #version 430
    in vec2 in_pos;
    in vec3 in_color;
    out vec3 v_color;
    void main() {{
        v_color = in_color;
        gl_Position = vec4((in_pos.x / {WORLD_WIDTH}.0) * 2.0 - 1.0, (in_pos.y / {WORLD_HEIGHT}.0) * 2.0 - 1.0, 0.0, 1.0);
        gl_PointSize = 2.0;
    }}
    '''

    fragment_shader = '''
    #version 430
    in vec3 v_color;
    out vec4 fragColor;
    void main() {
        fragColor = vec4(v_color, 1.0);
    }
    '''

    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    cbo = color_buffer

    vao = ctx.vertex_array(prog, [(pos_buffer, '2f', 'in_pos'), (cbo, '3f', 'in_color')])

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
        int x = int(p.x / bin_size);
        int y = int(p.y / bin_size);
        x = clamp(x, 0, grid_width - 1);
        y = clamp(y, 0, grid_height - 1);
        int bin_idx = y * grid_width + x;

        uint offset = atomicAdd(bin_counts[bin_idx], 1);
        if (offset < {MAX_BIN_PARTICLES}) {{
            bin_particles[bin_idx * {MAX_BIN_PARTICLES} + offset] = i;
        }}
    }}
    ''')

    interaction_shader = ctx.compute_shader(f'''
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
    uniform float K;
    uniform float friction;
    uniform float delta_time;
    uniform float max_radius;
    uniform float bin_size;
    uniform int grid_width;
    uniform int grid_height;

    int wrap(int val, int max_val) {{
        return (val + max_val) % max_val;
    }}

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
                    int idx = my_type * {NUM_TYPES} + other_type;
                    float mind = min_distances[idx];
                    float rad = radii[idx];
                    float force_strength = forces[idx];

                    if (dist < mind) {{
                        f -= dn * abs(force_strength) * 5.0 * (1.0 - dist / mind) * K;
                    }} else if (dist < rad) {{
                        f += dn * force_strength * (1.0 - dist / rad) * K;
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
    ''')

    clear_counts_shader = ctx.compute_shader(f'''
    #version 430
    layout(local_size_x = 256) in;
    layout(std430, binding = 6) buffer BinCounts {{ uint bin_counts[]; }};
    uniform int num_bins;
    void main() {{
        uint i = gl_GlobalInvocationID.x;
        if (i < num_bins) bin_counts[i] = 0;
    }}
    ''')

    buffers = [pos_buffer, vel_buffer, type_buffer, forces_buf, min_dist_buf, radii_buf, bin_counts_buf, bin_particles_buf]
    for i, buf in enumerate(buffers):
        buf.bind_to_storage_buffer(i)

    for name, value in [
        ('num_particles', NUM_PARTICLES),
        ('world_width', float(WORLD_WIDTH)),
        ('world_height', float(WORLD_HEIGHT)),
        ('K', k_slider.val),
        ('friction', friction_slider.val),
        ('max_radius', float(MAX_RADIUS)),
        ('bin_size', float(BIN_SIZE)),
        ('grid_width', GRID_WIDTH),
        ('grid_height', GRID_HEIGHT),
    ]:
        interaction_shader[name].value = value
        if name in binning_shader:
            binning_shader[name].value = value

    clear_counts_shader['num_bins'].value = NUM_BINS

    clock = pygame.time.Clock()
    sim_speed = 1
    running = True

    while running:
        dt = clock.tick(60) / 1000.0
        
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key in [pygame.K_ESCAPE, pygame.K_q]):
                running = False
            
            # Handle slider events
            for slider in sliders:
                if slider.handle_event(event):
                    # Update shader uniforms when sliders change
                    interaction_shader['K'].value = k_slider.val
                    interaction_shader['friction'].value = friction_slider.val

        # Run simulation
        ctx.clear(0.02, 0.02, 0.08, 1.0)

        clear_counts_shader.run(group_x=(NUM_BINS + 255) // 256)
        binning_shader.run(group_x=(NUM_PARTICLES + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE)

        interaction_shader['delta_time'].value = dt * sim_speed
        interaction_shader.run(group_x=(NUM_PARTICLES + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE)

        vao.render(mode=moderngl.POINTS)
        
        # Draw UI overlay
        ui_surface.fill((0, 0, 0, 0))  # Clear with transparent
        for slider in sliders:
            slider.draw(ui_surface, font)
        
        # Convert UI surface to texture and render
        ui_data = pygame.image.tostring(ui_surface, 'RGBA')
        ui_texture = ctx.texture((DISPLAY_WIDTH, DISPLAY_HEIGHT), 4, ui_data)
        
        # Simple UI shader for overlay
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
        
        # Create UI rendering program if not exists
        if not hasattr(main, 'ui_prog'):
            main.ui_prog = ctx.program(vertex_shader=ui_vertex_shader, fragment_shader=ui_fragment_shader)
            # Create fullscreen quad
            quad_vertices = np.array([
                -1.0, -1.0, 0.0, 1.0,
                 1.0, -1.0, 1.0, 1.0,
                -1.0,  1.0, 0.0, 0.0,
                 1.0,  1.0, 1.0, 0.0
            ], dtype=np.float32)
            main.ui_vbo = ctx.buffer(quad_vertices.tobytes())
            main.ui_vao = ctx.vertex_array(main.ui_prog, [(main.ui_vbo, '2f 2f', 'in_position', 'in_texcoord')])
        
        # Render UI overlay
        ui_texture.use(0)
        main.ui_prog['ui_texture'].value = 0
        main.ui_vao.render(mode=moderngl.TRIANGLE_STRIP)
        
        ui_texture.release()
        
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()