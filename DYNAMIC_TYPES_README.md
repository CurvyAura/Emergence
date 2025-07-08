# Dynamic Particle Types Feature

## Overview
The particle life simulation now supports adjustable particle type count while the simulation is running. You can dynamically increase or decrease the number of particle types to see how it affects the emergent behaviors in real-time.

## Controls
- **Q**: Decrease number of particle types (minimum: 2)
- **W**: Increase number of particle types (maximum: 8)
- **M**: Randomize attraction matrix with new random values
- **P**: Print current values (including number of types)
- **M**: Randomize the entire attraction matrix with new random values

## How It Works

### Type Adjustment
When you change the number of types:

1. **Decreasing Types**: Particles with types outside the new range are randomly reassigned to valid types
2. **Increasing Types**: A portion of existing particles (up to 25%) are randomly reassigned to include the new types, providing immediate visual feedback

### Attraction Matrix Rebuilding
The attraction matrix is automatically rebuilt when types change:
- Existing type relationships are preserved
- New type relationships are generated with random values in the configured range (-1.0 to 1.0)
- The new matrix maintains the proper size for the current number of types

### Visual Changes
- Particles are colored based on their type using a predefined color scheme
- Types 0-7 have distinct colors: Red, Green, Blue, Yellow, Magenta, Cyan, Orange, and White/Gray
- Color changes are immediately visible when particle types are reassigned

## Configuration
You can adjust the following parameters in `config.py`:

```python
# Dynamic Type System
MIN_TYPES = 2           # Minimum number of particle types
MAX_TYPES = 8           # Maximum number of particle types
DEFAULT_ATTRACTION_RANGE = (-1.0, 1.0)  # Range for generating new attraction values
```

## Performance Notes
- The GPU compute shader automatically handles the new number of types
- Performance scales with the square of the number of types (due to the N×N attraction matrix)
- Maximum recommended types: 8 (for performance and visual clarity)

## Tips for Experimentation
1. Start with 2-3 types to see basic behaviors
2. Gradually increase to 4-6 types to observe more complex interactions
3. Try decreasing back to fewer types to see how the system simplifies
4. Use the 'P' key to monitor current settings and attraction matrix values
5. Press 'M' to randomize the matrix and discover new patterns

## Troubleshooting

### ValueError: assignment destination is read-only
If you encounter this error, it has been fixed in the current version. The issue was caused by trying to modify read-only numpy arrays created from GPU buffer data. The fix ensures we create writable copies of the data before modification.

### _moderngl.Error: out of range offset = 0 or size = X
This error occurred when changing particle types beyond 4 types. The issue was that the GPU buffer for the attraction matrix was created with a fixed size (for the original 4×4 matrix), but when expanding to 5×5 or larger matrices, the new data was too large for the original buffer.

**Fix**: The attraction buffer is now recreated with the appropriate size whenever the number of types changes, preventing buffer overflow errors.

### Particles Getting "Consumed" or Overlapping
This issue has been fixed by implementing a gentle repulsion system. Previously, the minimum distance only prevented force calculations but didn't actively separate overlapping particles.

**New Solution**:
- Particles closer than `MIN_DISTANCE` experience gentle repulsive forces
- **Quadratic falloff**: Repulsion decreases smoothly (not linearly) as distance increases
- **Reduced strength**: Default repulsion strength of 2.0 (vs aggressive 10.0)
- **Enhanced stability**: Higher damping (0.95) and slower time scale (0.20)

**Result**: Particles maintain proper separation without jittery or explosive behavior.

**Controls**:
- **E/R**: Decrease/Increase minimum distance (0.005-0.08)
- **T/Y**: Decrease/Increase repulsion strength (0.5-10.0)

This feature allows for real-time exploration of how particle type diversity affects the emergence of patterns, clustering behaviors, and system dynamics in the particle life simulation.

## New Features

### Matrix Randomization
- **M**: Randomize the entire attraction matrix with new random values
- Generates completely new behaviors and emergent patterns
- Works with any number of particle types (2-8)
- Values generated within the configured range (-1.0 to 1.0)
- Perfect for discovering interesting new particle life patterns
