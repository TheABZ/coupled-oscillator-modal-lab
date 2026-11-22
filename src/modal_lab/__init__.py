"""Normal modes, coupled oscillations, and vibration control."""
from .diagnostics import energy, linear_frf, nonlinear_frf, power_spectrum, suppression_bands
from .linear_modes import assemble, describe_modes, normal_modes
from .nonlinear_sim import harmonic_base, harmonic_force, simulate
from .systems import (
    System, absorber_system, coupled_pendulums, critical_damping, critical_step,
    damped_oscillator, spring_mass_chain, symmetric_two_dof,
)

__all__ = [
    "System", "absorber_system", "assemble", "coupled_pendulums", "critical_damping",
    "critical_step", "damped_oscillator", "describe_modes", "energy", "harmonic_base",
    "harmonic_force", "linear_frf", "nonlinear_frf", "normal_modes", "power_spectrum",
    "simulate", "spring_mass_chain", "suppression_bands", "symmetric_two_dof",
]
