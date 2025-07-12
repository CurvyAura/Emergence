"""
Particle Life Simulation - Core GPU-accelerated simulation engine
Features real-time parameter adjustment and emergent behavior exploration

Copyright (c) 2025 CurvyAura
Licensed under MIT License - see LICENSE file for details
"""

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
    vsync = False  # Disable VSync for maximum FPS
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Get the OpenGL context - this is what lets us talk to the GPU
        self.ctx = self.wnd.ctx
        
        print(f"OpenGL Version: {self.ctx.version_code}")
        print(f"GPU: {self.ctx.info['GL_RENDERER']}")
        
        # Enable blending for smooth circular particles with anti-aliasing
        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        
        # Set up our first particles
        self.setup_particles()
        
        # Mouse interaction state
        self.mouse_pressed = False
        self.mouse_pos = (0.0, 0.0)  # Normalized coordinates (-1 to 1)
        
        # Print controls on startup
        print("\n" + "="*50)
        print("PARTICLE LIFE CONTROLS - MOMENTUM EDITION")
        print("="*50)
        print("MOUSE:")
        print("Left Click + Drag - Push particles away from cursor")
        print("Mouse Wheel - Adjust mouse repulsion radius")
        print("")
        print("KEYBOARD:")
        print("1/2 - Decrease/Increase Force Factor")
        print("3/4 - Decrease/Increase R-Max")
        print("5/6 - Decrease/Increase Damping (fine-tuned for clusters)") 
        print("7/8 - Decrease/Increase Time Scale")
        print("9   - Toggle Boundaries (wrap/bounce)")
        print("Q/W - Decrease/Increase Number of Types")
        print("E/R - Decrease/Increase Min Distance (Hard Collision Radius)")
        print("T/Y - Decrease/Increase Mouse Repulsion Strength")
        print("C   - Cycle Colour Palettes (Classic/Neon/Pastel/Ocean/Fire/Monochrome/Purple)")
        print("M   - Randomize Attraction Matrix (try for new behaviors!)")
        print("SPACE - Reset/Disperse Particles (fresh start!)")
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

    def reset_particles(self):
        """Reset particle positions and velocities to random dispersed state"""
        print("🔄 Resetting particles - dispersing randomly...")
        
        # Generate new random positions and velocities
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
        
        # Read current particle data from GPU and make a writable copy
        buffer_data = np.frombuffer(self.particle_buffer.read(), dtype=np.float32)
        particle_data = buffer_data.reshape((self.num_particles, 8)).copy()  # Make writable copy
        
        # Update positions and velocities while keeping types
        particle_data[:, 0:2] = positions  # x, y
        particle_data[:, 2:4] = velocities # vx, vy
        # Keep particle_data[:, 4] (types) unchanged
        
        # Write updated data back to GPU
        self.particle_buffer.write(particle_data.tobytes())
        
        print("✨ Particles dispersed! Ready for new emergent behavior!")

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
        compute_shader = f"""
        #version 430
        
        layout(local_size_x = {config.WORK_GROUP_SIZE}) in;
        
        layout(std430, binding = 0) restrict buffer ParticleBuffer {{
            float particles[];  // x, y, vx, vy, type, pad, pad, pad for each particle
        }};
        
        layout(std430, binding = 1) restrict buffer AttractionBuffer {{
            float attractions[];  // attraction matrix flattened
        }};
        
        uniform float dt;
        uniform int num_particles;
        uniform int num_types;
        uniform float force_factor;
        uniform float min_distance;
        uniform float max_distance;
        uniform float damping;
        uniform bool wrap_boundaries;
        uniform float bounce_damping;
        uniform bool mouse_pressed;
        uniform vec2 mouse_pos;
        uniform float mouse_repulsion_strength;
        uniform float mouse_repulsion_radius;
        
        void main() {{
            uint i = gl_GlobalInvocationID.x;
            if (i >= num_particles) return;
            
            // Get current particle data (8 floats per particle)
            uint base_i = i * 8;
            vec2 pos_i = vec2(particles[base_i], particles[base_i + 1]);
            vec2 vel_i = vec2(particles[base_i + 2], particles[base_i + 3]);
            int type_i = int(particles[base_i + 4]);
            
            vec2 force = vec2(0.0);
            int neighbor_count = 0;  // Count nearby particles for adaptive damping
            
            // Calculate forces from all other particles
            for (uint j = 0; j < num_particles; j++) {{
                if (i == j) continue;
                
                uint base_j = j * 8;
                vec2 pos_j = vec2(particles[base_j], particles[base_j + 1]);
                vec2 vel_j = vec2(particles[base_j + 2], particles[base_j + 3]);
                int type_j = int(particles[base_j + 4]);
                
                // Calculate distance (considering wrapping if enabled)
                vec2 diff = pos_j - pos_i;
                
                if (wrap_boundaries) {{
                    // Handle wrapping - find shortest distance considering wrap-around
                    if (abs(diff.x) > 1.0) {{
                        diff.x = diff.x > 0.0 ? diff.x - 2.0 : diff.x + 2.0;
                    }}
                    if (abs(diff.y) > 1.0) {{
                        diff.y = diff.y > 0.0 ? diff.y - 2.0 : diff.y + 2.0;
                    }}
                }}
                
                float dist = length(diff);
                
                // Count neighbors for adaptive damping
                if (dist < max_distance) {{
                    neighbor_count++;
                }}
                
                // Skip if within minimum distance (collision resolution will handle this)
                if (dist < min_distance) {{
                    continue;  // Skip force calculation for overlapping particles
                }}
                
                // Skip if too far
                if (dist > max_distance) continue;
                
                // Get attraction value between these types
                float attraction = attractions[type_i * num_types + type_j];
                
                // Smooth exponential force function for stable ship formation
                float r_ratio = dist / max_distance;
                
                float force_magnitude;
                if (r_ratio < 0.15) {{
                    // Close range: slightly stronger repulsion to spread dense clumps
                    force_magnitude = -0.7 * (0.15 - r_ratio) / 0.15;
                }} else if (r_ratio < 1.0) {{
                    // Medium to long range: smooth exponential attraction
                    float decay_factor = exp(-r_ratio * 2.0);  // Gentler exponential decay
                    float smooth_factor = 1.0 / (1.0 + r_ratio * 2.0);  // Smoother than 1/r^2
                    force_magnitude = attraction * decay_factor * smooth_factor;
                }} else {{
                    force_magnitude = 0.0;
                }}
                
                // Calculate basic force
                vec2 force_dir = normalize(diff);
                vec2 basic_force = force_dir * force_magnitude * force_factor;
                
                // Add momentum transfer for cluster coherence
                // Strongly attracted particles should share momentum
                if (abs(attraction) > 0.2 && dist < max_distance * 0.8) {{
                    vec2 vel_diff = vel_j - vel_i;
                    vec2 momentum_transfer = vel_diff * 0.05 * abs(attraction);
                    basic_force += momentum_transfer;
                }}
                
                force += basic_force;
            }}
            
            // Mouse repulsion force
            if (mouse_pressed) {{
                vec2 mouse_diff = pos_i - mouse_pos;
                float mouse_dist = length(mouse_diff);
                
                if (mouse_dist < mouse_repulsion_radius && mouse_dist > 0.001) {{
                    // Calculate repulsion force that falls off with distance
                    float repulsion_factor = (mouse_repulsion_radius - mouse_dist) / mouse_repulsion_radius;
                    repulsion_factor = repulsion_factor * repulsion_factor; // Square for sharper falloff
                    
                    vec2 repulsion_dir = normalize(mouse_diff);
                    vec2 mouse_force = repulsion_dir * mouse_repulsion_strength * repulsion_factor;
                    
                    force += mouse_force;
                }}
            }}
            
            // Adaptive damping: clusters get less damping to maintain momentum
            float adaptive_damping = damping;
            if (neighbor_count > 5) {{
                // Reduce damping for particles in dense clusters
                float density_factor = min(float(neighbor_count) / 15.0, 0.7);
                adaptive_damping = mix(damping, 0.9995, density_factor);
            }}
            
            // Apply damping first to existing velocity, then add new force
            vel_i *= adaptive_damping;
            vel_i += force * dt;
            
            // Update position
            pos_i += vel_i * dt;
            
            // COLLISION RESOLUTION - Prevent overlapping entirely
            // Check against all other particles and correct position if overlapping
            vec2 total_correction = vec2(0.0);
            int collision_count = 0;
            
            for (uint j = 0; j < num_particles; j++) {{
                if (i == j) continue;
                
                uint base_j = j * 8;
                vec2 pos_j = vec2(particles[base_j], particles[base_j + 1]);
                
                // Calculate distance (considering wrapping if enabled)
                vec2 diff = pos_i - pos_j;
                
                if (wrap_boundaries) {{
                    // Handle wrapping for collision detection
                    if (abs(diff.x) > 1.0) {{
                        diff.x = diff.x > 0.0 ? diff.x - 2.0 : diff.x + 2.0;
                    }}
                    if (abs(diff.y) > 1.0) {{
                        diff.y = diff.y > 0.0 ? diff.y - 2.0 : diff.y + 2.0;
                    }}
                }}
                
                float dist = length(diff);
                
                // If particles are overlapping, accumulate correction
                if (dist < min_distance && dist > 0.001) {{
                    // Calculate overlap amount
                    float overlap = min_distance - dist;
                    
                    // Normalize the difference vector
                    vec2 separation_dir = diff / dist;
                    
                    // Add to total correction (smaller, more conservative movement)
                    total_correction += separation_dir * (overlap * 0.1);
                    collision_count++;
                }}
            }}
            
            // Apply accumulated correction with safety limits
            if (collision_count > 0) {{
                // Limit maximum correction per frame to prevent teleporting
                float correction_magnitude = length(total_correction);
                if (correction_magnitude > min_distance * 0.5) {{
                    total_correction = normalize(total_correction) * (min_distance * 0.5);
                }}
                
                pos_i += total_correction;
                
                // Dampen velocity only slightly to maintain natural motion
                vel_i *= 0.95;
            }}
            
            // Handle boundaries (after collision resolution)
            if (wrap_boundaries) {{
                // Wrap around edges
                if (pos_i.x > 1.0) pos_i.x -= 2.0;
                if (pos_i.x < -1.0) pos_i.x += 2.0;
                if (pos_i.y > 1.0) pos_i.y -= 2.0;
                if (pos_i.y < -1.0) pos_i.y += 2.0;
            }} else {{
                // Bounce off walls
                if (pos_i.x <= -1.0 || pos_i.x >= 1.0) {{
                    vel_i.x *= -bounce_damping;
                    pos_i.x = clamp(pos_i.x, -0.99, 0.99);
                }}
                
                if (pos_i.y <= -1.0 || pos_i.y >= 1.0) {{
                    vel_i.y *= -bounce_damping;
                    pos_i.y = clamp(pos_i.y, -0.99, 0.99);
                }}
            }}
            
            // Safety check: ensure particle position is reasonable
            // Clamp to slightly beyond normal boundaries to prevent disappearing
            pos_i.x = clamp(pos_i.x, -1.2, 1.2);
            pos_i.y = clamp(pos_i.y, -1.2, 1.2);
            
            // Write back to buffer
            particles[base_i] = pos_i.x;
            particles[base_i + 1] = pos_i.y;
            particles[base_i + 2] = vel_i.x;
            particles[base_i + 3] = vel_i.y;
        }}
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
        
        # Rendering shaders with colour support - using instanced quads for reliable particle rendering
        vertex_shader = f"""
        #version 330
        in vec2 position;
        in float particle_type;
        in vec2 quad_vertex;  // Corner of the quad (-1,-1 to 1,1)
        out float type;
        out vec2 quad_coord;  // Pass quad coordinates to fragment shader
        
        void main() {{
            // Convert particle size from pixels to normalized coordinates
            float size_x = {config.PARTICLE_SIZE} / 400.0;  // Assuming 800 width
            float size_y = {config.PARTICLE_SIZE} / 300.0;  // Assuming 600 height
            
            // Create a quad around the particle position
            vec2 offset = quad_vertex * vec2(size_x, size_y);
            gl_Position = vec4(position + offset, 0.0, 1.0);
            type = particle_type;
            quad_coord = quad_vertex;  // Pass the quad coordinates (-1 to 1)
        }}
        """
        
        fragment_shader = f"""
        #version 330
        in float type;
        in vec2 quad_coord;  // Quad coordinates from vertex shader
        out vec4 fragColor;
        
        uniform vec3 colours[8];  // Array of colours for up to 8 particle types
        
        void main() {{
            // Create circular particles by discarding pixels outside radius
            float distance_from_center = length(quad_coord);
            if (distance_from_center > 1.0) {{
                discard;  // Don't draw pixels outside the circle
            }}
            
            // Optional: Add anti-aliasing by fading the edges
            float alpha = 1.0 - smoothstep(0.8, 1.0, distance_from_center);
            
            // Colour particles based on type using dynamic colours
            int itype = int(type + 0.5);  // Round to nearest integer
            
            vec3 colour;
            if (itype >= 0 && itype < 8) {{
                colour = colours[itype];
            }} else {{
                colour = vec3(0.8, 0.8, 0.8);  // Default white/gray for out-of-range types
            }}
            
            fragColor = vec4(colour, alpha);
        }}
        """
        
        # Create rendering program
        self.program = self.ctx.program(
            vertex_shader=vertex_shader,
            fragment_shader=fragment_shader
        )
        
        # Create quad vertices for instanced rendering (each particle becomes a quad)
        quad_vertices = np.array([
            -1.0, -1.0,  # Bottom-left
             1.0, -1.0,  # Bottom-right
            -1.0,  1.0,  # Top-left
             1.0,  1.0,  # Top-right
        ], dtype=np.float32)
        
        quad_indices = np.array([0, 1, 2, 1, 3, 2], dtype=np.uint32)  # Two triangles
        
        self.quad_buffer = self.ctx.buffer(quad_vertices.tobytes())
        self.index_buffer = self.ctx.buffer(quad_indices.tobytes())
        
        # Create vertex array for instanced rendering
        self.vertex_array = self.ctx.vertex_array(
            self.program, 
            [
                (self.position_buffer, '2f/i', 'position'),      # Per instance (particle)
                (self.type_buffer, '1f/i', 'particle_type'),    # Per instance (particle)
                (self.quad_buffer, '2f', 'quad_vertex'),        # Per vertex (quad corner)
            ],
            self.index_buffer
        )
        
    def on_render(self, time, frame_time):
        """This function is called every frame to draw everything"""
        
        # Update particle physics
        self.update_particles(frame_time)
        
        # Update colour uniforms for current palette
        colours_array = []
        for i in range(8):
            if i < len(config.PARTICLE_COLOURS):
                colour = config.PARTICLE_COLOURS[i]
                colours_array.append((colour[0], colour[1], colour[2]))
            else:
                colours_array.append((0.8, 0.8, 0.8))  # Default grey
        
        # Set the colours array as a uniform (list of tuples)
        self.program['colours'] = colours_array
        
        # Clear the screen with configurable background colour
        bg = config.BACKGROUND_COLOR
        self.ctx.clear(bg[0], bg[1], bg[2], bg[3])
        
        # Draw our particles as instanced quads (much more reliable than points)
        self.vertex_array.render(instances=self.num_particles)
        
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
        self.compute_program['max_distance'] = config.RMAX
        self.compute_program['damping'] = config.DAMPING
        self.compute_program['wrap_boundaries'] = config.WRAP_BOUNDARIES
        self.compute_program['bounce_damping'] = config.BOUNCE_DAMPING
        self.compute_program['mouse_pressed'] = self.mouse_pressed
        self.compute_program['mouse_pos'] = self.mouse_pos
        self.compute_program['mouse_repulsion_strength'] = config.MOUSE_REPULSION_STRENGTH
        self.compute_program['mouse_repulsion_radius'] = config.MOUSE_REPULSION_RADIUS
        
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
            config.FORCE_FACTOR = min(2.0, config.FORCE_FACTOR + 0.001)
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
            config.DAMPING = max(0.98, config.DAMPING - 0.001)
            print(f"DAMPING: {old_val:.4f} -> {config.DAMPING:.4f}")
        elif key == 54:  # '6' key
            old_val = config.DAMPING
            config.DAMPING = min(1.0, config.DAMPING + 0.001)
            print(f"DAMPING: {old_val:.4f} -> {config.DAMPING:.4f}")
            
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
            
        # E/R - Min Distance (Hard Collision Radius)
        elif key == 101:  # 'E' key
            old_val = config.MIN_DISTANCE
            config.MIN_DISTANCE = max(0.005, config.MIN_DISTANCE - 0.002)
            print(f"MIN DISTANCE (Hard Collision Radius): {old_val:.3f} -> {config.MIN_DISTANCE:.3f}")
        elif key == 114:  # 'R' key
            old_val = config.MIN_DISTANCE
            config.MIN_DISTANCE = min(0.08, config.MIN_DISTANCE + 0.002)
            print(f"MIN DISTANCE (Hard Collision Radius): {old_val:.3f} -> {config.MIN_DISTANCE:.3f}")
            
        # T/Y - Mouse Repulsion Strength
        elif key == 116:  # 'T' key
            old_val = config.MOUSE_REPULSION_STRENGTH
            config.MOUSE_REPULSION_STRENGTH = max(0.1, config.MOUSE_REPULSION_STRENGTH - 0.1)
            print(f"MOUSE REPULSION STRENGTH: {old_val:.2f} -> {config.MOUSE_REPULSION_STRENGTH:.2f}")
        elif key == 121:  # 'Y' key
            old_val = config.MOUSE_REPULSION_STRENGTH
            config.MOUSE_REPULSION_STRENGTH = min(5.0, config.MOUSE_REPULSION_STRENGTH + 0.1)
            print(f"MOUSE REPULSION STRENGTH: {old_val:.2f} -> {config.MOUSE_REPULSION_STRENGTH:.2f}")
            
        # M - Randomize Attraction Matrix
        elif key == 109:  # 'M' key
            self.randomize_attraction_matrix()
            
        # C - Cycle Colour Palettes
        elif key == 99:  # 'C' key
            old_index = config.CURRENT_PALETTE_INDEX
            config.CURRENT_PALETTE_INDEX = (config.CURRENT_PALETTE_INDEX + 1) % len(config.COLOUR_PALETTES)
            config.PARTICLE_COLOURS = config.COLOUR_PALETTES[config.CURRENT_PALETTE_INDEX]
            
            palette_names = ["Classic", "Neon", "Pastel", "Ocean", "Fire", "Monochrome", "Purple"]
            old_name = palette_names[old_index] if old_index < len(palette_names) else f"Palette {old_index}"
            new_name = palette_names[config.CURRENT_PALETTE_INDEX] if config.CURRENT_PALETTE_INDEX < len(palette_names) else f"Palette {config.CURRENT_PALETTE_INDEX}"
            print(f"COLOUR PALETTE: {old_name} -> {new_name}")
            
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
            print(f"Mouse Repulsion Strength: {config.MOUSE_REPULSION_STRENGTH:.2f}")
            print(f"Mouse Repulsion Radius: {config.MOUSE_REPULSION_RADIUS:.3f}")
            print("")
            
        # SPACEBAR - Reset particles (key 32)
        elif key == 32:  # Spacebar key
            self.reset_particles()
        
        # Ignore unmapped keys silently
    
    def print_controls(self):
        """Print keyboard controls"""
        print("\n" + "="*50)
        print("PARTICLE LIFE CONTROLS")
        print("="*50)
        print("MOUSE:")
        print("Left Click + Drag - Push particles away from cursor")
        print("Mouse Wheel - Adjust mouse repulsion radius")
        print("")
        print("KEYBOARD:")
        print("1/2 - Decrease/Increase Force Factor")
        print("3/4 - Decrease/Increase R-Max")
        print("5/6 - Decrease/Increase Damping") 
        print("7/8 - Decrease/Increase Time Scale")
        print("9   - Toggle Boundaries (wrap/bounce)")
        print("Q/W - Decrease/Increase Number of Types")
        print("E/R - Decrease/Increase Min Distance (Hard Collision Radius)")
        print("T/Y - Decrease/Increase Mouse Repulsion Strength")
        print("C   - Cycle Colour Palettes")
        print("M   - Randomize Attraction Matrix")
        print("SPACE - Reset/Disperse Particles")
        print("H   - Show this help")
        print("P   - Print current values")
        print("="*50 + "\n")
    
    # Mouse event handlers using the correct method names
    def on_mouse_press_event(self, x, y, button):
        """Handle mouse button press events"""
        if button == 1:  # Left mouse button
            self.mouse_pressed = True
            # Convert screen coordinates to normalized coordinates (-1 to 1)
            width, height = self.wnd.size
            norm_x = (x / width) * 2.0 - 1.0
            norm_y = -((y / height) * 2.0 - 1.0)  # Flip Y axis
            self.mouse_pos = (norm_x, norm_y)

    def on_mouse_release_event(self, x: int, y: int, button: int):
        """Handle mouse button release events"""
        if button == 1:  # Left mouse button
            self.mouse_pressed = False

    def on_mouse_position_event(self, x, y, dx, dy):
        """Handle mouse movement"""
        if self.mouse_pressed:
            # Convert screen coordinates to normalized coordinates (-1 to 1)
            width, height = self.wnd.size
            norm_x = (x / width) * 2.0 - 1.0
            norm_y = -((y / height) * 2.0 - 1.0)  # Flip Y axis
            self.mouse_pos = (norm_x, norm_y)

    def on_mouse_drag_event(self, x, y, dx, dy):
        """Handle mouse drag"""
        if self.mouse_pressed:
            # Convert screen coordinates to normalized coordinates (-1 to 1)
            width, height = self.wnd.size
            norm_x = (x / width) * 2.0 - 1.0
            norm_y = -((y / height) * 2.0 - 1.0)  # Flip Y axis
            self.mouse_pos = (norm_x, norm_y)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        """Handle mouse wheel"""
        # Scroll up/down to change mouse repulsion radius
        old_radius = config.MOUSE_REPULSION_RADIUS
        if y_offset > 0:
            config.MOUSE_REPULSION_RADIUS = min(1.0, config.MOUSE_REPULSION_RADIUS + 0.05)
        else:
            config.MOUSE_REPULSION_RADIUS = max(0.1, config.MOUSE_REPULSION_RADIUS - 0.05)
        
        print(f"MOUSE REPULSION RADIUS: {old_radius:.3f} -> {config.MOUSE_REPULSION_RADIUS:.3f}")
        
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
