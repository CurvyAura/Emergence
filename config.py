"""
Particle Life Simulation Configuration
Adjust these values to experiment with different behaviors!

This is your control panel for the entire simulation!

Copyright (c) 2025 CurvyAura
Licensed under MIT License - see LICENSE file for details

DEFAULT VALUES (good starting points):
Particle count: 10000
Force Factor: 0.4000
R-Max: 0.08
Damping: 0.998
Time Scale: 0.20
Boundaries: WRAP
Number of Types: 4

"""

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
DAMPING = 0.798            # Higher damping for smoother, less erratic movement (0.998)
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

# =============================================================================
# MOUSE INTERACTION - Control particles with your cursor
# =============================================================================
MOUSE_REPULSION_STRENGTH = 5.0    # How strong the mouse push effect is
MOUSE_REPULSION_RADIUS = 0.25     # How far from cursor the effect reaches


#=============================================================================


# DONT TOUCH PARTICLE COLOURS OR THE ATTRACTION MATRIX IN THE CONFIG (You can randomize/adjust these while running without issue)


# =============================================================================

# Particle Colours (R, G, B, A) - Multiple palettes to cycle through
COLOUR_PALETTES = [
    # Classic Palette
    [
        (1.0, 0.3, 0.3, 1.0),  # Red
        (0.3, 1.0, 0.3, 1.0),  # Green
        (0.3, 0.3, 1.0, 1.0),  # Blue
        (1.0, 1.0, 0.3, 1.0),  # Yellow
        (1.0, 0.3, 1.0, 1.0),  # Magenta
        (0.3, 1.0, 1.0, 1.0),  # Cyan
        (1.0, 0.6, 0.3, 1.0),  # Orange
        (0.8, 0.8, 0.8, 1.0),  # White/Gray
    ],
    # Neon Palette
    [
        (1.0, 0.0, 0.5, 1.0),  # Hot Pink
        (0.0, 1.0, 0.0, 1.0),  # Bright Green
        (0.0, 0.5, 1.0, 1.0),  # Electric Blue
        (1.0, 1.0, 0.0, 1.0),  # Bright Yellow
        (1.0, 0.0, 1.0, 1.0),  # Bright Magenta
        (0.0, 1.0, 1.0, 1.0),  # Bright Cyan
        (1.0, 0.3, 0.0, 1.0),  # Bright Orange
        (0.5, 0.0, 1.0, 1.0),  # Purple
    ],
    # Pastel Palette
    [
        (1.0, 0.7, 0.7, 1.0),  # Light Pink
        (0.7, 1.0, 0.7, 1.0),  # Light Green
        (0.7, 0.7, 1.0, 1.0),  # Light Blue
        (1.0, 1.0, 0.7, 1.0),  # Light Yellow
        (1.0, 0.7, 1.0, 1.0),  # Light Magenta
        (0.7, 1.0, 1.0, 1.0),  # Light Cyan
        (1.0, 0.8, 0.6, 1.0),  # Peach
        (0.9, 0.9, 0.9, 1.0),  # Light Gray
    ],
    # Ocean Palette
    [
        (0.0, 0.4, 0.6, 1.0),  # Deep Blue
        (0.0, 0.7, 0.8, 1.0),  # Teal
        (0.2, 0.8, 0.6, 1.0),  # Sea Green
        (0.6, 0.9, 0.4, 1.0),  # Light Green
        (0.9, 0.9, 0.3, 1.0),  # Sand Yellow
        (0.8, 0.6, 0.4, 1.0),  # Brown
        (1.0, 0.8, 0.6, 1.0),  # Coral
        (0.9, 0.9, 0.9, 1.0),  # Foam White
    ],
    # Fire Palette
    [
        (1.0, 0.0, 0.0, 1.0),  # Bright Red
        (1.0, 0.3, 0.0, 1.0),  # Red-Orange
        (1.0, 0.6, 0.0, 1.0),  # Orange
        (1.0, 0.8, 0.0, 1.0),  # Yellow-Orange
        (1.0, 1.0, 0.2, 1.0),  # Yellow
        (0.8, 0.0, 0.0, 1.0),  # Dark Red
        (0.6, 0.0, 0.0, 1.0),  # Maroon
        (0.3, 0.0, 0.0, 1.0),  # Dark Maroon
    ],
    # Monochrome Palette (Blacks, Whites, Greys)
    [
        (1.0, 1.0, 1.0, 1.0),  # Pure White
        (0.9, 0.9, 0.9, 1.0),  # Light Grey
        (0.7, 0.7, 0.7, 1.0),  # Medium Light Grey
        (0.5, 0.5, 0.5, 1.0),  # Medium Grey
        (0.3, 0.3, 0.3, 1.0),  # Dark Grey
        (0.15, 0.15, 0.15, 1.0),  # Very Dark Grey
        (0.05, 0.05, 0.05, 1.0),  # Almost Black
        (0.0, 0.0, 0.0, 1.0),  # Pure Black
    ],
    # Purple Palette
    [
        (0.5, 0.0, 1.0, 1.0),  # Electric Purple
        (0.7, 0.2, 1.0, 1.0),  # Bright Purple
        (0.8, 0.4, 1.0, 1.0),  # Light Purple
        (1.0, 0.6, 1.0, 1.0),  # Pink-Purple
        (0.6, 0.0, 0.8, 1.0),  # Deep Purple
        (0.4, 0.0, 0.6, 1.0),  # Dark Purple
        (0.3, 0.0, 0.5, 1.0),  # Very Dark Purple
        (0.2, 0.0, 0.3, 1.0),  # Midnight Purple
    ],
]

# Current palette index and colours
CURRENT_PALETTE_INDEX = 0
PARTICLE_COLOURS = COLOUR_PALETTES[CURRENT_PALETTE_INDEX]

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
