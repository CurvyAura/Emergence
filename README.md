# Emergence - Interactive Particle Life Simulation

A GPU-accelerated particle life simulation built with Python and ModernGL, featuring real-time parameter adjustment and dynamic emergent behavior exploration.

### 🎮 Controls

**Mouse Interaction:**
- **Left Click + Drag**: Push particles away from cursor
- **Mouse Wheel**: Adjust mouse repulsion radius

**Keyboard:**
- **1/2**: Adjust Force Factor
- **3/4**: Adjust Interaction Radius (R-Max)
- **5/6**: Adjust Damping
- **7/8**: Adjust Time Scale
- **9**: Toggle Boundaries (wrap/bounce)
- **Spacebar**: Reset Particle Positions
- **Q/W**: Change Number of Particle Types (2-8)
- **E/R**: Adjust Minimum Distance
- **T/Y**: Adjust Mouse Repulsion Strength
- **C**: Cycle Colour Schemes
- **M**: Randomize Attraction Matrix
- **H**: Show Help
- **P**: Print Current Values

## 🚀 Quick Start

### Option 1: Download Executable 
(Not Recommended: Limited functionality, certain config values are unable to be tweaked i.e: particle count)
**[Download Emergence.exe](https://drive.google.com/file/d/1q5e6ffPFukoZbqiuyIuZ7MAdUv_aEKHU/view?usp=sharing)** - Single file, the simple way!

⚠️ **Security Notice**: Your antivirus may show warnings because this is an unsigned executable. This is normal for indie software. The complete source code is available below for verification.

**System Requirements:**
- Windows 10/11
- OpenGL 3.3+ compatible graphics card
- ~80MB disk space

Simply download and double-click to run!

### Option 2: Run from Source
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

### Particle "Life"
The emergent behaviors arise from these simple rules:

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

This Project was inspired by Jeffrey Ventrella.

Feel free to contribute improvements, new features, or bug fixes! Some ideas:
- Additional particle interaction models
- Save/load system for interesting configurations
- Performance optimizations
- New visualization modes

## Gallery

<img width="1647" height="962" alt="image" src="https://github.com/user-attachments/assets/662bbb0d-f854-459d-afbe-cdf81c16d347" />
<img width="1740" height="961" alt="image" src="https://github.com/user-attachments/assets/85df79bc-3241-4c49-b7bf-8b1bf8f976b2" />

## License

**MIT License**

Copyright (c) 2025 CurvyAura

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---
