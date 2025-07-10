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
# SIMULATION SCALE - How big is your universe?
# =============================================================================
NUM_PARTICLES = 15000       # Fewer particles for clearer ship visibility
NUM_TYPES = 4              # Classic 4-type setup for clear ship formation

# =============================================================================
# PHYSICS ENGINE - The rules that govern how particles interact
# =============================================================================
FORCE_FACTOR = 0.4         # Lower force for smoother, less jittery movement
MIN_DISTANCE = 0.008       # Slightly larger to reduce close-range instability
REPULSION_STRENGTH = 2.0   # Legacy parameter (kept for compatibility but no longer used)
RMAX = 0.08                # Smaller radius for more stable interactions
DAMPING = 0.998            # Higher damping for smoother, less erratic movement
TIME_SCALE = 0.2           # Slower time scale for more stable dynamics

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
# 
# Balanced 4x4 attraction matrix for moving ship creatures:
ATTRACTION_MATRIX = [
    # Red   Green Blue  Yellow
    [ 0.1,   0.7, -0.5,  0.3],  # Red: moderate chase Green, mild attraction to Yellow
    [-0.2,   0.1,  0.6, -0.4],  # Green: mild self-cohesion, moderate attraction to Blue
    [ 0.4,  -0.3,  0.1,  0.5],  # Blue: moderate attractions, mild self-cohesion
    [-0.3,   0.5, -0.6,  0.1],  # Yellow: moderate chase Green, mild self-cohesion
]

# =============================================================================
# STARTING CONDITIONS - How your universe begins
# =============================================================================
INITIAL_POSITION_RANGE = 0.8  # Spread for better ship formation
INITIAL_VELOCITY_RANGE = 0.08  # Higher initial velocity for directional movement

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
