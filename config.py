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
NUM_PARTICLES = 12000      # Sweet spot for complex patterns without excessive clumping
NUM_TYPES = 6              # More types = more complex interactions and behaviors

# =============================================================================
# PHYSICS ENGINE - The rules that govern how particles interact
# =============================================================================
FORCE_FACTOR = 0.1         # Much higher force for stronger cluster formation and movement
MIN_DISTANCE = 0.005       # Smaller collision radius to allow closer interactions
REPULSION_STRENGTH = 2.0   # Legacy parameter (kept for compatibility but no longer used)
RMAX = 0.035               # Larger radius for more long-range cluster coherence
DAMPING = 0.995            # Less damping to preserve cluster momentum and velocity
TIME_SCALE = 0.4           # Slightly higher time scale for more responsive dynamics

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
# Improved attraction matrix for dynamic ships and orbital behaviors:
ATTRACTION_MATRIX = [
    # Red   Green Blue  Yellow Magenta Cyan
    [ 0.1,   0.8, -0.9,  0.6,  -0.5,   0.4],  # Red: strongly chases Green, flees Blue
    [-0.4,   0.2,  0.9, -0.8,   0.7,  -0.3],  # Green: loves Blue, hates Yellow, likes Magenta  
    [ 0.7,  -0.6,  0.1,  0.8,  -0.9,   0.5],  # Blue: attracts Yellow, repels Magenta strongly
    [-0.5,   0.6, -0.7,  0.3,   0.9,  -0.4],  # Yellow: complex mixed relationships
    [ 0.9,  -0.3,  0.4, -0.6,   0.2,   0.8],  # Magenta: chases Red and Cyan, mixed others
    [-0.6,   0.5, -0.4,  0.7,  -0.8,   0.1],  # Cyan: creates orbital dynamics and ships
]

# =============================================================================
# STARTING CONDITIONS - How your universe begins
# =============================================================================
INITIAL_POSITION_RANGE = 0.9  # Spread particles widely to prevent initial clumping
INITIAL_VELOCITY_RANGE = 0.12  # Higher initial momentum for more dynamic cluster formation

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
