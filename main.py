"""
Particle Life Simulation
A GPU-accelerated particle life simulation using ModernGL
"""

from particle_life import ParticleLifeWindow


def main():
    """Entry point for the particle life simulation"""
    print("Starting Particle Life Simulation...")
    ParticleLifeWindow.run()


if __name__ == "__main__":
    main()