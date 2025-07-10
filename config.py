"""
Particle Life Simulation Configuration
Adjust these values to experiment with different behaviors!

This is your control panel for the entire simulation!

DEFAULT VALUES (good starting points):
Particle count: 20000
Force Factor: 0.5000
R-Max: 0.080
Damping: 0.900
Time Scale: 0.26
Boundaries: WRAP
Number of Types: 6

"""

# =============================================================================
# SIMULATION SCALE - How big is your universe?
# =============================================================================
NUM_PARTICLES = 20000      # More particles = more complex patterns, but slower performance
NUM_TYPES = 6              # Different "species" of particles - each behaves differently

# =============================================================================
# PHYSICS ENGINE - The rules that govern how particles interact
# =============================================================================
FORCE_FACTOR = 0.10         # Master volume control for all forces (0.1 = gentle, 1.0 = chaotic)
MIN_DISTANCE = 0.008       # Hard collision radius - roughly matches visual particle size
REPULSION_STRENGTH = 2.0   # Legacy parameter (kept for compatibility but no longer used)
RMAX = 0.20                # Maximum interaction distance - particles ignore each other beyond this
DAMPING = 0.90            # Friction in the universe (0.9 = realistic, 0.99 = space-like)
TIME_SCALE = 0.20          # Speed of time itself (0.1 = slow motion, 0.5 = fast forward)

# =============================================================================
# WORLD BOUNDARIES - What happens at the edge of your universe?
# =============================================================================
WRAP_BOUNDARIES = True     # True = Pac-Man style wrap-around, False = bouncy walls
BOUNCE_DAMPING = 0.8       # Energy lost when hitting walls (only matters if bouncing)

# =============================================================================
# VISUAL APPEARANCE - How your particles look on screen
# =============================================================================
PARTICLE_SIZE = 1.0        # Size in pixels - make bigger if particles are too small to see
BACKGROUND_COLOR = (0.0, 0.0, 0.0, 1.0)  # Black space background (R, G, B, Alpha)


#=============================================================================


# DONT TOUCH PARTICLE COLOURS OR THE ATTRACTION MATRIX IN THE CONFIG (You can randomize/adjust these while running without issue)


# =============================================================================

# Particle Colors (R, G, B, A) - 4 colors only
PARTICLE_COLORS = [
    (1.0, 0.3, 0.3, 1.0),  # Red
    (0.3, 1.0, 0.3, 1.0),  # Green
    (0.3, 0.3, 1.0, 1.0),  # Blue
    (1.0, 1.0, 0.3, 1.0),  # Yellow
]

# =============================================================================
# THE ATTRACTION MATRIX - The heart of emergent behavior!
# =============================================================================
# This matrix defines how each particle type feels about every other type:
#   Positive numbers = attraction (they like each other)
#   Negative numbers = repulsion (they can't stand each other)  
#   Zero = indifference (they ignore each other)
# 
# Think of it like a compatibility chart for dating apps, but for particles!
# Small changes here can create completely different ecosystems.
ATTRACTION_MATRIX = [
    # Red   Green Blue  Yellow
    [ 0.9,  0.5, -0.5,  0.7],  # Red
    [-0.5,  0.0,  0.5, -0.3],  # Green  
    [ 0.5, -0.5,  0.9,  0.5],  # Blue
    [-0.5,  0.5, -0.5,  0.9],  # Yellow
]

# =============================================================================
# STARTING CONDITIONS - How your universe begins
# =============================================================================
INITIAL_POSITION_RANGE = 0.8  # How spread out particles start (0.1 = clustered, 1.0 = everywhere)
INITIAL_VELOCITY_RANGE = 0.02  # Initial random motion (0.0 = still, 0.1 = already moving)

# =============================================================================
# PERFORMANCE TUNING - Making it run smooth as butter
# =============================================================================
WORK_GROUP_SIZE = 128      # GPU optimization - bigger numbers = better performance on good GPUs
ENABLE_VSYNC = False       # Turn off frame rate limiting for maximum speed

# =============================================================================
# DYNAMIC EXPERIMENTATION - Live editing while the simulation runs
# =============================================================================
# These control the range of the Q/W keys for changing particle types on the fly
MIN_TYPES = 2              # Minimum number of particle species (2 = simple, boring)
MAX_TYPES = 8              # Maximum number of particle species (8 = complex, chaotic)
DEFAULT_ATTRACTION_RANGE = (-1.0, 1.0)  # Range for generating new random attractions
