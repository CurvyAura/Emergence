import moderngl_window as mglw
import moderngl
import numpy as np
import config


class ParticleLifeWindow(mglw.WindowConfig):
    """Basic window for our particle life simulation"""
    
    # Window configuration
    title = "Particle Life Simulation"
    window_size = (800, 600)
    resizable = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get the OpenGL context - this is what lets us talk to the GPU
        self.ctx = self.wnd.ctx
        
        print(f"OpenGL Version: {self.ctx.version_code}")
        print(f"GPU: {self.ctx.info['GL_RENDERER']}")
        
        # Set up our first particles
        self.setup_particles()
        
        # Print controls on startup
        print("\n" + "="*50)
        print("PARTICLE LIFE CONTROLS")
        print("="*50)
        print("1/2 - Decrease/Increase Force Factor")
        print("3/4 - Decrease/Increase R-Max")
        print("5/6 - Decrease/Increase Damping") 
        print("7/8 - Decrease/Increase Time Scale")
        print("9   - Toggle Boundaries (wrap/bounce)")
        print("Q/W - Decrease/Increase Number of Types")
        print("E/R - Decrease/Increase Min Distance")
        print("T/Y - Decrease/Increase Repulsion Strength")
        print("M   - Randomize Attraction Matrix")
        print("H   - Show this help")
        print("P   - Print current values")
        print("="*50 + "\n")
        
    def randomize_attraction_matrix(self):
        """Randomize the attraction matrix with new random values"""
        print(f"Randomizing attraction matrix for {self.num_types} types...")
        
        # Create completely new random attraction matrix
        self.attraction_matrix = np.random.uniform(
            *config.DEFAULT_ATTRACTION_RANGE, 
            (self.num_types, self.num_types)
        ).astype(np.float32)
        
        # Release the old buffer and create a new one
        if hasattr(self, 'attraction_buffer'):
            self.attraction_buffer.release()
        
        self.attraction_buffer = self.ctx.buffer(self.attraction_matrix.flatten().tobytes())
        
        # Rebind the new buffer to the compute shader
        self.attraction_buffer.bind_to_storage_buffer(1)
        
        print("New attraction matrix:")
        print(self.attraction_matrix)
        print("Press 'P' to see current values or experiment with the new behaviors!")

    def rebuild_attraction_matrix(self):
        """Rebuild attraction matrix when number of types changes"""
        # Create a new attraction matrix with appropriate size
        new_matrix = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        
        # Copy existing values where possible
        old_matrix = np.array(config.ATTRACTION_MATRIX, dtype=np.float32)
        min_size = min(old_matrix.shape[0], self.num_types)
        new_matrix[:min_size, :min_size] = old_matrix[:min_size, :min_size]
        
        # Fill new rows/columns with random values if types increased
        if self.num_types > old_matrix.shape[0]:
            for i in range(old_matrix.shape[0], self.num_types):
                for j in range(self.num_types):
                    new_matrix[i, j] = np.random.uniform(*config.DEFAULT_ATTRACTION_RANGE)
                    if i != j:  # Don't overwrite diagonal for existing types
                        new_matrix[j, i] = np.random.uniform(*config.DEFAULT_ATTRACTION_RANGE)
        
        self.attraction_matrix = new_matrix
        
        # Release the old buffer and create a new one with the correct size
        if hasattr(self, 'attraction_buffer'):
            self.attraction_buffer.release()
        
        self.attraction_buffer = self.ctx.buffer(self.attraction_matrix.flatten().tobytes())
        
        # Rebind the new buffer to the compute shader
        self.attraction_buffer.bind_to_storage_buffer(1)
        
        print(f"Rebuilt attraction matrix for {self.num_types} types:")
        print(self.attraction_matrix)
    
    def reassign_particle_types(self):
        """Reassign particle types when number of types changes"""
        # Read current particle data and make a writable copy
        buffer_data = np.frombuffer(self.particle_buffer.read(), dtype=np.float32)
        particle_data = buffer_data.reshape((-1, 8)).copy()  # Make writable copy
        
        # Get current types
        current_types = particle_data[:, 4].copy()
        
        # If we reduced the number of types, reassign out-of-range particles
        if self.num_types < len(np.unique(current_types)):
            # Reassign particles with types >= num_types
            out_of_range_mask = current_types >= self.num_types
            num_reassign = np.sum(out_of_range_mask)
            if num_reassign > 0:
                new_types = np.random.randint(0, self.num_types, num_reassign).astype(np.float32)
                current_types[out_of_range_mask] = new_types
                print(f"Reassigned {num_reassign} particles to valid type range (0-{self.num_types-1})")
        else:
            # If we increased types, randomly assign some particles to new types
            # This gives immediate visual feedback
            num_to_reassign = min(self.num_particles // 4, 1000)  # Reassign up to 1/4 of particles
            indices_to_reassign = np.random.choice(self.num_particles, num_to_reassign, replace=False)
            new_types = np.random.randint(0, self.num_types, num_to_reassign).astype(np.float32)
            current_types[indices_to_reassign] = new_types
            print(f"Reassigned {num_to_reassign} particles to include new types (0-{self.num_types-1})")
        
        # Update particle data
        particle_data[:, 4] = current_types
        
        # Write back to GPU
        self.particle_buffer.write(particle_data.tobytes())
        
        # Update type buffer for rendering
        self.type_buffer.write(current_types.tobytes())

    def setup_particles(self):
        """Create particles with GPU compute shader physics and life rules"""
        
        # Use configuration values
        self.num_particles = config.NUM_PARTICLES
        self.num_types = config.NUM_TYPES
        print(f"Creating {self.num_particles} particles with {self.num_types} types")
        print(f"Boundary mode: {'Wrapping' if config.WRAP_BOUNDARIES else 'Bouncing'}")
        
        # Create initial particle data using config ranges
        positions = np.random.uniform(
            -config.INITIAL_POSITION_RANGE, 
            config.INITIAL_POSITION_RANGE, 
            (self.num_particles, 2)
        ).astype(np.float32)
        
        velocities = np.random.uniform(
            -config.INITIAL_VELOCITY_RANGE, 
            config.INITIAL_VELOCITY_RANGE, 
            (self.num_particles, 2)
        ).astype(np.float32)
        
        # Assign random types to particles
        types = np.random.randint(0, self.num_types, self.num_particles).astype(np.float32)
        
        # Pack position, velocity, and type into array for the GPU
        # Format: [x, y, vx, vy, type, padding, padding, padding] for each particle
        particle_data = np.zeros((self.num_particles, 8), dtype=np.float32)  # Increased to 8 for alignment
        particle_data[:, 0:2] = positions  # x, y
        particle_data[:, 2:4] = velocities # vx, vy
        particle_data[:, 4] = types        # particle type
        
        # Use attraction matrix from config
        self.attraction_matrix = np.array(config.ATTRACTION_MATRIX, dtype=np.float32)
        
        print("Attraction matrix:")
        print(self.attraction_matrix)
        
        # Create compute shader for particle life physics
        compute_shader = """
        #version 430
        
        layout(local_size_x = 64) in;
        
        layout(std430, binding = 0) restrict buffer ParticleBuffer {
            float particles[];  // x, y, vx, vy, type, pad, pad, pad for each particle
        };
        
        layout(std430, binding = 1) restrict buffer AttractionBuffer {
            float attractions[];  // attraction matrix flattened
        };
        
        uniform float dt;
        uniform int num_particles;
        uniform int num_types;
        uniform float force_factor;
        uniform float min_distance;
        uniform float repulsion_strength;
        uniform float max_distance;
        uniform float damping;
        uniform bool wrap_boundaries;
        uniform float bounce_damping;
        
        void main() {
            uint i = gl_GlobalInvocationID.x;
            if (i >= num_particles) return;
            
            // Get current particle data (8 floats per particle)
            uint base_i = i * 8;
            vec2 pos_i = vec2(particles[base_i], particles[base_i + 1]);
            vec2 vel_i = vec2(particles[base_i + 2], particles[base_i + 3]);
            int type_i = int(particles[base_i + 4]);
            
            vec2 force = vec2(0.0);
            
            // Calculate forces from all other particles
            for (uint j = 0; j < num_particles; j++) {
                if (i == j) continue;
                
                uint base_j = j * 8;
                vec2 pos_j = vec2(particles[base_j], particles[base_j + 1]);
                int type_j = int(particles[base_j + 4]);
                
                // Calculate distance (considering wrapping if enabled)
                vec2 diff = pos_j - pos_i;
                
                if (wrap_boundaries) {
                    // Handle wrapping - find shortest distance considering wrap-around
                    if (abs(diff.x) > 1.0) {
                        diff.x = diff.x > 0.0 ? diff.x - 2.0 : diff.x + 2.0;
                    }
                    if (abs(diff.y) > 1.0) {
                        diff.y = diff.y > 0.0 ? diff.y - 2.0 : diff.y + 2.0;
                    }
                }
                
                float dist = length(diff);
                
                // Handle minimum distance with gentle repulsion
                if (dist < min_distance) {
                    if (dist > 0.001) {  // Avoid division by very small numbers
                        // Gentle repulsive force to separate overlapping particles
                        vec2 force_dir = normalize(diff);
                        // Use smoother repulsion curve: stronger when very close, gentler as distance increases
                        float overlap_ratio = (min_distance - dist) / min_distance;
                        float repulsion_magnitude = overlap_ratio * overlap_ratio;  // Quadratic for smoother response
                        force += force_dir * repulsion_magnitude * force_factor * repulsion_strength;
                    }
                    continue;  // Skip normal force calculation
                }
                
                // Skip if too far
                if (dist > max_distance) continue;
                
                // Get attraction value between these types
                float attraction = attractions[type_i * num_types + type_j];
                
                // Classic Jeffrey Ventrella Particle Life force function
                // F(r) = attraction * f(r/rmax) where f is the standard force curve
                float r_ratio = dist / max_distance;
                
                float force_magnitude;
                if (r_ratio < 0.3) {
                    // Repulsion zone - linear repulsion
                    force_magnitude = r_ratio / 0.3 - 1.0;
                } else if (r_ratio < 1.0) {
                    // Attraction zone - smooth attraction with peak around 0.5
                    float x = (r_ratio - 0.3) / 0.7;
                    force_magnitude = x * (1.0 - x) * 4.0; // Peak at x=0.5
                } else {
                    force_magnitude = 0.0;
                }
                
                // Apply attraction scaling and force factor
                force_magnitude *= attraction;
                
                // Calculate final force
                vec2 force_dir = normalize(diff);
                force += force_dir * force_magnitude * force_factor;
            }
            
            // Apply force to velocity with damping
            vel_i += force * dt;
            vel_i *= damping;
            
            // Update position
            pos_i += vel_i * dt;
            
            // Handle boundaries
            if (wrap_boundaries) {
                // Wrap around edges
                if (pos_i.x > 1.0) pos_i.x -= 2.0;
                if (pos_i.x < -1.0) pos_i.x += 2.0;
                if (pos_i.y > 1.0) pos_i.y -= 2.0;
                if (pos_i.y < -1.0) pos_i.y += 2.0;
            } else {
                // Bounce off walls
                if (pos_i.x <= -1.0 || pos_i.x >= 1.0) {
                    vel_i.x *= -bounce_damping;
                    pos_i.x = clamp(pos_i.x, -0.99, 0.99);
                }
                
                if (pos_i.y <= -1.0 || pos_i.y >= 1.0) {
                    vel_i.y *= -bounce_damping;
                    pos_i.y = clamp(pos_i.y, -0.99, 0.99);
                }
            }
            
            // Write back to buffer
            particles[base_i] = pos_i.x;
            particles[base_i + 1] = pos_i.y;
            particles[base_i + 2] = vel_i.x;
            particles[base_i + 3] = vel_i.y;
        }
        """
        
        # Create the compute program
        self.compute_program = self.ctx.compute_shader(compute_shader)
        
        # Create buffer for particle data (positions + velocities + types)
        self.particle_buffer = self.ctx.buffer(particle_data.tobytes())
        
        # Create buffer for attraction matrix
        self.attraction_buffer = self.ctx.buffer(self.attraction_matrix.flatten().tobytes())
        
        # Create separate buffers for rendering (positions and types)
        self.position_buffer = self.ctx.buffer(positions.tobytes())
        self.type_buffer = self.ctx.buffer(types.tobytes())
        
        # Bind buffers to compute shader
        self.particle_buffer.bind_to_storage_buffer(0)
        self.attraction_buffer.bind_to_storage_buffer(1)
        
        # Rendering shaders with color support
        vertex_shader = f"""
        #version 330
        in vec2 position;
        in float particle_type;
        out float type;
        
        void main() {{
            gl_Position = vec4(position, 0.0, 1.0);
            gl_PointSize = {config.PARTICLE_SIZE};
            type = particle_type;
        }}
        """
        
        fragment_shader = f"""
        #version 330
        in float type;
        out vec4 fragColor;
        
        void main() {{
            // Color particles based on type with dynamic colors
            int itype = int(type + 0.5);  // Round to nearest integer
            
            // Use a simple color generation for types beyond predefined colors
            if (itype == 0) {{
                fragColor = vec4(1.0, 0.3, 0.3, 1.0);  // Red
            }} else if (itype == 1) {{
                fragColor = vec4(0.3, 1.0, 0.3, 1.0);  // Green
            }} else if (itype == 2) {{
                fragColor = vec4(0.3, 0.3, 1.0, 1.0);  // Blue
            }} else if (itype == 3) {{
                fragColor = vec4(1.0, 1.0, 0.3, 1.0);  // Yellow
            }} else if (itype == 4) {{
                fragColor = vec4(1.0, 0.3, 1.0, 1.0);  // Magenta
            }} else if (itype == 5) {{
                fragColor = vec4(0.3, 1.0, 1.0, 1.0);  // Cyan
            }} else if (itype == 6) {{
                fragColor = vec4(1.0, 0.6, 0.3, 1.0);  // Orange
            }} else {{
                fragColor = vec4(0.8, 0.8, 0.8, 1.0);  // White/Gray
            }}
        }}
        """
        
        # Create rendering program
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        # Create vertex array for rendering (positions and types)
        self.vertex_array = self.ctx.vertex_array(
            self.program, 
            [
                (self.position_buffer, '2f', 'position'),
                (self.type_buffer, '1f', 'particle_type')
            ]
        )
        
    def on_render(self, time, frame_time):
        """This function is called every frame to draw everything"""
        
        # Update particle physics
        self.update_particles(frame_time)
        
        # Clear the screen with configurable background color
        bg = config.BACKGROUND_COLOR
        self.ctx.clear(bg[0], bg[1], bg[2], bg[3])
        
        # Draw our particles as points
        self.vertex_array.render(moderngl.POINTS)
        
    def update_particles(self, frame_time):
        """Update particle physics using GPU compute shader with life rules"""
        
        # Apply time scale to slow down the simulation
        scaled_frame_time = frame_time * config.TIME_SCALE
        
        # Set uniforms for the compute shader using config values
        self.compute_program['dt'] = scaled_frame_time
        self.compute_program['num_particles'] = self.num_particles
        self.compute_program['num_types'] = self.num_types
        self.compute_program['force_factor'] = config.FORCE_FACTOR
        self.compute_program['min_distance'] = config.MIN_DISTANCE
        self.compute_program['repulsion_strength'] = config.REPULSION_STRENGTH
        self.compute_program['max_distance'] = config.RMAX
        self.compute_program['damping'] = config.DAMPING
        self.compute_program['wrap_boundaries'] = config.WRAP_BOUNDARIES
        self.compute_program['bounce_damping'] = config.BOUNCE_DAMPING
        
        # Dispatch the compute shader
        work_groups = (self.num_particles + config.WORK_GROUP_SIZE - 1) // config.WORK_GROUP_SIZE
        self.compute_program.run(work_groups)
        
        # Memory barrier
        self.ctx.memory_barrier(moderngl.SHADER_STORAGE_BARRIER_BIT)
        
        # Read back particle data (now 8 floats per particle)
        particle_data = np.frombuffer(self.particle_buffer.read(), dtype=np.float32)
        particle_data = particle_data.reshape((-1, 8))
        
        # Extract positions and update position buffer
        positions = particle_data[:, 0:2]
        self.position_buffer.write(positions.tobytes())
        
    def on_key_event(self, key, action, modifiers):
        """Handle keyboard events to modify config values in real-time"""
        # Check if it's a key press (not release)
        if str(action) != "ACTION_PRESS":
            return
        
        # 1/2 - Force Factor
        if key == 49:  # '1' key
            old_val = config.FORCE_FACTOR
            config.FORCE_FACTOR = max(0.001, config.FORCE_FACTOR - 0.001)
            print(f"FORCE FACTOR: {old_val:.4f} -> {config.FORCE_FACTOR:.4f}")
        elif key == 50:  # '2' key
            old_val = config.FORCE_FACTOR
            config.FORCE_FACTOR = min(0.1, config.FORCE_FACTOR + 0.001)
            print(f"FORCE FACTOR: {old_val:.4f} -> {config.FORCE_FACTOR:.4f}")
            
        # 3/4 - R-Max
        elif key == 51:  # '3' key
            old_val = config.RMAX
            config.RMAX = max(0.01, config.RMAX - 0.01)
            print(f"R-MAX: {old_val:.3f} -> {config.RMAX:.3f}")
        elif key == 52:  # '4' key
            old_val = config.RMAX
            config.RMAX = min(0.5, config.RMAX + 0.01)
            print(f"R-MAX: {old_val:.3f} -> {config.RMAX:.3f}")
            
        # 5/6 - Damping
        elif key == 53:  # '5' key
            old_val = config.DAMPING
            config.DAMPING = max(0.8, config.DAMPING - 0.01)
            print(f"DAMPING: {old_val:.3f} -> {config.DAMPING:.3f}")
        elif key == 54:  # '6' key
            old_val = config.DAMPING
            config.DAMPING = min(1.0, config.DAMPING + 0.01)
            print(f"DAMPING: {old_val:.3f} -> {config.DAMPING:.3f}")
            
        # 7/8 - Time Scale
        elif key == 55:  # '7' key
            old_val = config.TIME_SCALE
            config.TIME_SCALE = max(0.01, config.TIME_SCALE - 0.05)
            print(f"TIME SCALE: {old_val:.2f} -> {config.TIME_SCALE:.2f}")
        elif key == 56:  # '8' key
            old_val = config.TIME_SCALE
            config.TIME_SCALE = min(2.0, config.TIME_SCALE + 0.05)
            print(f"TIME SCALE: {old_val:.2f} -> {config.TIME_SCALE:.2f}")
            
        # 9 - Toggle boundaries
        elif key == 57:  # '9' key
            old_val = config.WRAP_BOUNDARIES
            config.WRAP_BOUNDARIES = not config.WRAP_BOUNDARIES
            print(f"BOUNDARIES: {'WRAP' if old_val else 'BOUNCE'} -> {'WRAP' if config.WRAP_BOUNDARIES else 'BOUNCE'}")
            
        # Q/W - Number of Types
        elif key == 113:  # 'Q' key
            old_val = self.num_types
            self.num_types = max(config.MIN_TYPES, self.num_types - 1)
            if self.num_types != old_val:
                print(f"NUM TYPES: {old_val} -> {self.num_types}")
                self.rebuild_attraction_matrix()
                self.reassign_particle_types()
            else:
                print(f"NUM TYPES: {self.num_types} (minimum reached)")
        elif key == 119:  # 'W' key
            old_val = self.num_types
            self.num_types = min(config.MAX_TYPES, self.num_types + 1)
            if self.num_types != old_val:
                print(f"NUM TYPES: {old_val} -> {self.num_types}")
                self.rebuild_attraction_matrix()
                self.reassign_particle_types()
            else:
                print(f"NUM TYPES: {self.num_types} (maximum reached)")
            
        # E/R - Min Distance
        elif key == 101:  # 'E' key
            old_val = config.MIN_DISTANCE
            config.MIN_DISTANCE = max(0.005, config.MIN_DISTANCE - 0.002)
            print(f"MIN DISTANCE: {old_val:.3f} -> {config.MIN_DISTANCE:.3f}")
        elif key == 114:  # 'R' key
            old_val = config.MIN_DISTANCE
            config.MIN_DISTANCE = min(0.08, config.MIN_DISTANCE + 0.002)
            print(f"MIN DISTANCE: {old_val:.3f} -> {config.MIN_DISTANCE:.3f}")
            
        # T/Y - Repulsion Strength
        elif key == 116:  # 'T' key
            old_val = config.REPULSION_STRENGTH
            config.REPULSION_STRENGTH = max(0.5, config.REPULSION_STRENGTH - 0.2)
            print(f"REPULSION STRENGTH: {old_val:.1f} -> {config.REPULSION_STRENGTH:.1f}")
        elif key == 121:  # 'Y' key
            old_val = config.REPULSION_STRENGTH
            config.REPULSION_STRENGTH = min(10.0, config.REPULSION_STRENGTH + 0.2)
            print(f"REPULSION STRENGTH: {old_val:.1f} -> {config.REPULSION_STRENGTH:.1f}")
            
        # M - Randomize Attraction Matrix
        elif key == 109:  # 'M' key
            self.randomize_attraction_matrix()
            
        # H - Help (key 104)
        elif key == 104:  # 'H' key
            self.print_controls()
            
        # P - Print current values (key 112) 
        elif key == 112:  # 'P' key
            print("\nCURRENT VALUES:")
            print(f"Force Factor: {config.FORCE_FACTOR:.4f}")
            print(f"R-Max: {config.RMAX:.3f}")
            print(f"Damping: {config.DAMPING:.3f}")
            print(f"Time Scale: {config.TIME_SCALE:.2f}")
            print(f"Boundaries: {'WRAP' if config.WRAP_BOUNDARIES else 'BOUNCE'}")
            print(f"Number of Types: {self.num_types}")
            print(f"Min Distance: {config.MIN_DISTANCE:.3f}")
            print(f"Repulsion Strength: {config.REPULSION_STRENGTH:.1f}")
            print("")
        
        # Ignore unmapped keys silently
    
    def print_controls(self):
        """Print keyboard controls"""
        print("\n" + "="*50)
        print("PARTICLE LIFE CONTROLS")
        print("="*50)
        print("1/2 - Decrease/Increase Force Factor")
        print("3/4 - Decrease/Increase R-Max")
        print("5/6 - Decrease/Increase Damping") 
        print("7/8 - Decrease/Increase Time Scale")
        print("9   - Toggle Boundaries (wrap/bounce)")
        print("Q/W - Decrease/Increase Number of Types")
        print("E/R - Decrease/Increase Min Distance")
        print("T/Y - Decrease/Increase Repulsion Strength")
        print("M   - Randomize Attraction Matrix")
        print("H   - Show this help")
        print("P   - Print current values")
        print("="*50 + "\n")
        
    def on_close(self):
        """Clean up when window closes"""
        # Release GPU buffers
        if hasattr(self, 'particle_buffer'):
            self.particle_buffer.release()
        if hasattr(self, 'attraction_buffer'):
            self.attraction_buffer.release()
        if hasattr(self, 'position_buffer'):
            self.position_buffer.release()
        if hasattr(self, 'type_buffer'):
            self.type_buffer.release()
