# Emergence - Interactive Particle Life Simulation

A GPU-accelerated particle life simulation built with Python and ModernGL, featuring real-time parameter adjustment and dynamic emergent behavior exploration.

## Features

### 🎮 Interactive Controls
- **1/2**: Adjust Force Factor
- **3/4**: Adjust Interaction Radius (R-Max)
- **5/6**: Adjust Damping
- **7/8**: Adjust Time Scale
- **9**: Toggle Boundaries (wrap/bounce)
- **Q/W**: Change Number of Particle Types (2-8)
- **E/R**: Adjust Minimum Distance
- **T/Y**: Adjust Repulsion Strength
- **M**: Randomize Attraction Matrix
- **H**: Show Help
- **P**: Print Current Values

### 🔬 Advanced Features
- **Dynamic Type System**: Change particle types in real-time
- **Anti-Overlap Technology**: Prevents particle consumption with gentle repulsion
- **Matrix Randomization**: Discover new emergent behaviors instantly
- **GPU Acceleration**: Efficient compute shaders for smooth performance
- **Stable Physics**: Carefully tuned parameters for realistic behavior

### 🎯 Key Capabilities
- Up to 20,000 particles with 60+ FPS performance
- 8 different particle types with unique colors
- Real-time attraction matrix modification
- Emergent pattern discovery through randomization
- Boundary behavior switching (wrap-around vs bouncing)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/CurvyAura/Emergence.git
   cd Emergence
   ```

2. **Install dependencies:**
   ```bash
   pip install moderngl moderngl-window numpy
   ```

3. **Run the simulation:**
   ```bash
   python main.py
   ```

## How It Works

### Particle Life Algorithm
This simulation implements the "Particle Life" algorithm where particles of different types have varying attraction/repulsion relationships. The emergent behaviors arise from these simple rules:

1. **Attraction/Repulsion**: Each particle type has different relationships with others
2. **Distance-Based Forces**: Force strength varies with distance (repulsion when close, attraction at medium range)
3. **Minimum Distance**: Particles maintain separation to prevent overlap
4. **Boundary Conditions**: Particles can wrap around edges or bounce off walls

### GPU Acceleration
The simulation uses OpenGL compute shaders for:
- Parallel force calculations between all particle pairs
- Real-time position and velocity updates
- Efficient memory management for large particle counts

## Configuration

Key parameters can be adjusted in `config.py`:
- `NUM_PARTICLES`: Number of particles (default: 20,000)
- `NUM_TYPES`: Number of particle types (default: 6)
- `FORCE_FACTOR`: Overall force strength (default: 0.5)
- `MIN_DISTANCE`: Minimum separation distance (default: 0.03)
- `REPULSION_STRENGTH`: Anti-overlap force strength (default: 2.0)

## Discovering Emergent Behaviors

1. **Start Simple**: Begin with 2-3 particle types
2. **Adjust Parameters**: Use number keys to modify physics
3. **Add Complexity**: Increase types with Q/W keys
4. **Randomize**: Press M to generate new attraction matrices
5. **Experiment**: Try different combinations and observe patterns

## Technical Details

- **Language**: Python 3.8+
- **Graphics**: ModernGL (OpenGL 4.3+)
- **Compute**: GPU shaders for parallel processing
- **Performance**: 60+ FPS with 20,000 particles on modern GPUs

## File Structure

- `main.py`: Entry point
- `particle_life.py`: Core simulation logic and GPU shaders
- `config.py`: Configuration parameters

## Contributing

Feel free to contribute improvements, new features, or bug fixes! Some ideas:
- Additional particle interaction models
- Save/load system for interesting configurations
- Performance optimizations
- New visualization modes

## License

This project is open source. Feel free to use, modify, and distribute.

---

**Explore the fascinating world of emergent complexity!** 🌟
