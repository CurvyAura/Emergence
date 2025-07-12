"""
Particle Life Simulation
A GPU-accelerated particle life simulation using ModernGL

Copyright (c) 2025 CurvyAura
Licensed under MIT License - see LICENSE file for details
"""

import ctypes
import ctypes.wintypes
from particle_life import ParticleLifeWindow


def prevent_sleep():
    """Prevent Windows from going to sleep while the simulation runs"""
    # Windows constants for SetThreadExecutionState
    ES_CONTINUOUS = 0x80000000 # modifier flag to keep the state set
    ES_SYSTEM_REQUIRED = 0x00000001 # prevent the system from sleeping
    ES_DISPLAY_REQUIRED = 0x00000002 # prevent the display from sleeping

    # Prevent system sleep and keep display on
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED
    )
    print("💡 Sleep prevention activated - PC will stay awake during simulation")


def restore_sleep():
    """Restore normal sleep behavior when simulation ends"""
    ES_CONTINUOUS = 0x80000000
    
    # Reset to normal power management
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    print("😴 Sleep prevention disabled - PC can sleep normally again")


def main():
    """Entry point for the particle life simulation"""
    print("Starting Particle Life Simulation...")
    
    try:
        prevent_sleep()
        ParticleLifeWindow.run()
        
    finally:
        restore_sleep()


if __name__ == "__main__":
    main()