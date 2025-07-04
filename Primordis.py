import pygame
import moderngl
import numpy as np

# Display and world dimensions
DISPLAY_WIDTH, DISPLAY_HEIGHT = 1920, 1080
WORLD_WIDTH, WORLD_HEIGHT = 1080, 720
NUM_TYPES = 32
NUM_PARTICLES = 16_000

# Binning parameters
MAX_RADIUS = 64
BIN_SIZE = MAX_RADIUS
GRID_WIDTH = WORLD_WIDTH // BIN_SIZE
GRID_HEIGHT = WORLD_HEIGHT // BIN_SIZE
NUM_BINS = GRID_WIDTH * GRID_HEIGHT
MAX_BIN_PARTICLES = 256
COMPUTE_GROUP_SIZE = 512

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
        # Draw track
        pygame.draw.rect(surface, (100, 100, 100), self.rect)
        pygame.draw.rect(surface, (200, 200, 200), self.rect, 2)
        # Draw handle
        handle_x = int(self.rect.x + (self.val - self.min_val) / (self.max_val - self.min_val) * self.rect.width)
        pygame.draw.circle(surface, (255, 255, 255), (handle_x, self.rect.centery), self.handle_radius)
        pygame.draw.circle(surface, (150, 150, 150), (handle_x, self.rect.centery), self.handle_radius, 2)
        # Draw label
        text = font.render(f"{self.label}: {self.val:.3f}", True, (255, 255, 255))
        surface.blit(text, (self.rect.x, self.rect.y - 25))


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
    pygame.display.set_caption("Particle Sim with Emergent Rewards")
    screen = pygame.display.set_mode((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.enable(moderngl.PROGRAM_POINT_SIZE)
    ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

    # UI overlay
    ui_surface = pygame.Surface((DISPLAY_WIDTH, DISPLAY_HEIGHT), pygame.SRCALPHA)
    font = pygame.font.Font(None, 24)

    # Hide-sliders toggle button
    hide_sliders = False
    padding = 10
    button_width, button_height = 120, 30
    button_x = 50
    button_y = DISPLAY_HEIGHT - button_height - padding
    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)
    button_font  = pygame.font.Font(None, 24)

    # Create sliders
    sliders = []
    k_slider         = Slider(50,  50, 200, 20, 0.1, 128.0, 32,  "Interaction Strength")
    friction_slider  = Slider(50, 100, 200, 20, 0.05, 0.99,  0.75,  "Particle Drift Strength")
    k_div_slider     = Slider(300,  50, 200, 20, 0.0,  64.0,  64,  "Diversity Strength")
    k_stab_slider    = Slider(300, 100, 200, 20, 0.0,  64.0,  2,   "Stability Strength")
    k_cluster_slider = Slider(300, 150, 200, 20, 0.0,  64.0,  64,  "Cluster Strength")
    s_min_slider     = Slider(550,  50, 200, 20, 1,    MAX_BIN_PARTICLES, 256, "Cluster Size Min")
    t_min_slider     = Slider(550, 100, 200, 20, 1,    NUM_TYPES,         32,  "Distinct Types Min")

    alpha1_slider    = Slider(800,  50, 200, 20, 0.5,  5.0,   2.5,  "α1 (multi small-scale)")
    alpha2_slider    = Slider(1050, 50, 200, 20, 0.5,  5.0,   5,    "α2 (multi mid-scale)")
    alpha3_slider    = Slider(800, 150, 200, 20, 0.5, 10.0,  10,   "α3 (multi large-scale)")
    k1_slider        = Slider(800, 100, 200, 20, 0.0, 10.0,  2.5,  "K1 (multi small-scale)")
    k2_slider        = Slider(1050,100,200, 20, 0.0, 10.0,  5,    "K2 (multi mid-scale)")
    k3_slider        = Slider(1050,150,200, 20, 0.0, 10.0,  10,   "K3 (multi large-scale)")

    sliders.extend([
        k_slider, friction_slider, k_div_slider, k_stab_slider, k_cluster_slider,
        s_min_slider, t_min_slider,
        alpha1_slider, k1_slider, alpha2_slider, k2_slider, alpha3_slider, k3_slider
    ])

    # Initialize particle data
    positions = np.random.rand(NUM_PARTICLES, 2).astype(np.float32)
    positions[:, 0] *= WORLD_WIDTH
    positions[:, 1] *= WORLD_HEIGHT
    velocities = np.random.uniform(-8, 8, (NUM_PARTICLES, 2)).astype(np.float32)
    types = np.random.randint(0, NUM_TYPES, NUM_PARTICLES, dtype=np.int32)
    colors = random_type_colors()
    per_particle_colors = colors[types]

    forces, min_distances, radii = set_parameters()

    # GPU buffers
    pos_buf = ctx.buffer(positions.tobytes(), dynamic=True)
    vel_buf = ctx.buffer(velocities.tobytes(), dynamic=True)
    type_buf = ctx.buffer(types.tobytes())
    color_buf = ctx.buffer(per_particle_colors.tobytes())
    forces_buf = ctx.buffer(forces.tobytes())
    mindist_buf = ctx.buffer(min_distances.tobytes())
    radii_buf = ctx.buffer(radii.tobytes())
    bincounts_buf = ctx.buffer(reserve=NUM_BINS * 4, dynamic=True)
    binparts_buf = ctx.buffer(reserve=NUM_BINS * MAX_BIN_PARTICLES * 4, dynamic=True)
    rewards_buf = ctx.buffer(reserve=NUM_PARTICLES * 4, dynamic=True)

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
        gl_PointSize = 2.0;
    }}
    """
    fragment_shader = """
    #version 430
    in vec3 v_color;
    out vec4 fragColor;
    void main() {
        fragColor = vec4(v_color, 1.0);
    }
    """
    prog = ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
    vao = ctx.vertex_array(prog, [
        (pos_buf, '2f', 'in_pos'),
        (color_buf, '3f', 'in_color'),
    ])

    # Compute shaders
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
    layout(std430, binding=8) buffer Rewards {{ float rewards[]; }};

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

    uniform float K_div;
    uniform float K_stab;
    uniform float K_cluster;
    uniform uint S_min;
    uniform uint T_min;
    uniform float alpha1;
    uniform float K1;
    uniform float alpha2;
    uniform float K2;
    uniform float alpha3;
    uniform float K3;

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

        int mask = 0;
        int distinct = 0;
        int total_n = 0;
        vec2 com = vec2(0.0);

        // neighbor loop
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
                    total_n++;
                    int ot = types[j];
                    int bit = 1<<ot;
                    if ((mask & bit) == 0) {{ mask |= bit; distinct++; }}
                    com += pos[j];
                    // base interaction
                    int idx2 = my_t*{NUM_TYPES} + ot;
                    float mind = min_distances[idx2];
                    float rad = radii[idx2];
                    float fs = forces[idx2];
                    vec2 dn = (d/dist);
                    if (dist < mind) f -= dn*abs(fs)*5.0*(1.0 - dist/mind)*K;
                    else if (dist < rad) f += dn*fs*(1.0 - dist/rad)*K;
                }}
            }}
        }}
        if (total_n>0) com /= float(total_n);

        float diversity = float(distinct)/float({NUM_TYPES});
        f += normalize(com-p)*K_div;

        float speed = length(vel_);
        float stability = max(0.0, 1.0 - speed/64.0);
        f += -vel_*stability*K_stab;

        float m1 = pow(diversity,alpha1)*K1;
        float m2 = pow(diversity,alpha2)*K2;
        float m3 = pow(diversity,alpha3)*K3;
        f += normalize(com-p)*(m1+m2+m3);

        uint bc = bin_counts[cx+cy*grid_width];
        if (bc>=S_min && uint(distinct)>=T_min)
            f += normalize(com-p)*(float(bc)/float({MAX_BIN_PARTICLES})*float(distinct)/float({NUM_TYPES}))*K_cluster;

        // integrate
        vel_ += f*delta_time;
        vel_ *= friction;
        p += vel_*delta_time;
        // wrap
        if (p.x<0) p.x+=world_width; else if(p.x>=world_width) p.x-=world_width;
        if (p.y<0) p.y+=world_height; else if(p.y>=world_height) p.y-=world_height;
        pos[i]=p; vel[i]=vel_;

        rewards[i]=diversity+stability+m1+m2+m3;
    }}
    """)
    clear_shader = ctx.compute_shader(f"""
    #version 430
    layout(local_size_x=256) in;
    layout(std430,binding=6) buffer Bins{{uint bin_counts[];}};
    uniform int num_bins;
    void main(){{uint i=gl_GlobalInvocationID.x; if(i<num_bins)bin_counts[i]=0;}}
    """)

    # Bind buffers
    for i, b in enumerate([pos_buf, vel_buf, type_buf, forces_buf,
                           mindist_buf, radii_buf, bincounts_buf, binparts_buf]):
        b.bind_to_storage_buffer(i)
    rewards_buf.bind_to_storage_buffer(8)

    # Set uniform parameters
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
        if name in bin_shader:
            bin_shader[name].value = val
        if name in interact_shader:
            interact_shader[name].value = val
        if name in clear_shader:
            clear_shader[name].value = val

    def update_uniforms():
        interact_shader['K'].value = k_slider.val
        interact_shader['friction'].value = friction_slider.val
        interact_shader['K_div'].value = k_div_slider.val
        interact_shader['K_stab'].value = k_stab_slider.val
        interact_shader['K_cluster'].value = k_cluster_slider.val
        interact_shader['S_min'].value = int(s_min_slider.val)
        interact_shader['T_min'].value = int(t_min_slider.val)
        interact_shader['alpha1'].value = alpha1_slider.val
        interact_shader['K1'].value = k1_slider.val
        interact_shader['alpha2'].value = alpha2_slider.val
        interact_shader['K2'].value = k2_slider.val
        interact_shader['alpha3'].value = alpha3_slider.val
        interact_shader['K3'].value = k3_slider.val

    update_uniforms()

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

        # Compute & render
        ctx.clear(0.02, 0.02, 0.08, 1.0)
        clear_shader.run(group_x=(NUM_BINS + 255) // 256)
        bin_shader.run(group_x=(NUM_PARTICLES + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE)
        interact_shader['delta_time'].value = dt
        interact_shader.run(group_x=(NUM_PARTICLES + COMPUTE_GROUP_SIZE - 1) // COMPUTE_GROUP_SIZE)
        vao.render(mode=moderngl.POINTS)

        # UI overlay
        ui_surface.fill((0, 0, 0, 0))

        # Draw button
        btn_color = (180, 180, 180) if not hide_sliders else (100, 100, 100)
        pygame.draw.rect(ui_surface, btn_color, button_rect)
        label = "Hide Sliders" if not hide_sliders else "Show Sliders"
        txt_surf = button_font.render(label, True, (0, 0, 0))
        txt_rect = txt_surf.get_rect(center=button_rect.center)
        ui_surface.blit(txt_surf, txt_rect)

        # Draw sliders if visible
        if not hide_sliders:
            for s in sliders:
                s.draw(ui_surface, font)

        # Blit UI onto GL texture and render
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
