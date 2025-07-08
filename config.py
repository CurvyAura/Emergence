"""
Particle Life Simulation Configuration
Adjust these values to experiment with different behaviors!


DEFAULT VALUES:

Particle count: 20000
Force Factor: 0.5000
R-Max: 0.080
Damping: 0.900
Time Scale: 0.26
Boundaries: WRAP
Number of Types: 6

"""

# Simulation Parameters
NUM_PARTICLES = 25000     # Fewer particles like successful demos
NUM_TYPES = 6            # 4 types for clear patterns

# Physics Parameters
FORCE_FACTOR = 0.5       # Strong forces like successful implementations
MIN_DISTANCE = 0.03      # Standard minimum distance
REPULSION_STRENGTH = 2.0  # Multiplier for repulsion force when particles are too close
RMAX = 0.08               # Classic interaction radius
DAMPING = 0.950          # Increased damping for stability
TIME_SCALE = 0.20        # Slower time scale for more stability

# Boundary Behavior
WRAP_BOUNDARIES = True   # True = wrap around edges, False = bounce off walls
BOUNCE_DAMPING = 0.8    # Energy loss when bouncing (only used if WRAP_BOUNDARIES = False)

# Visual Parameters
PARTICLE_SIZE = 20.0     # Size of particles in pixels
BACKGROUND_COLOR = (0.0, 0.0, 0.0, 1.0)  # Background color (R, G, B, A)

# Particle Colors (R, G, B, A) - 4 colors only
PARTICLE_COLORS = [
    (1.0, 0.3, 0.3, 1.0),  # Red
    (0.3, 1.0, 0.3, 1.0),  # Green
    (0.3, 0.3, 1.0, 1.0),  # Blue
    (1.0, 1.0, 0.3, 1.0),  # Yellow
]

# Attraction Matrix - Classic particle life values that actually work
# Based on successful implementations
ATTRACTION_MATRIX = [
    # Red   Green Blue  Yellow
    [ 0.9,  0.5, -0.5,  0.7],  # Red
    [-0.5,  0.0,  0.5, -0.3],  # Green  
    [ 0.5, -0.5,  0.9,  0.5],  # Blue
    [-0.5,  0.5, -0.5,  0.9],  # Yellow
]

# Initial Conditions
INITIAL_POSITION_RANGE = 0.8  # Spread particles out more initially
INITIAL_VELOCITY_RANGE = 0.02  # Small random velocities

# Performance Settings
WORK_GROUP_SIZE = 64    # Number of particles per GPU work group

# Dynamic Type System
MIN_TYPES = 2           # Minimum number of particle types
MAX_TYPES = 8           # Maximum number of particle types
DEFAULT_ATTRACTION_RANGE = (-1.0, 1.0)  # Range for generating new attraction values
