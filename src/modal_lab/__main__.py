"""Run the built-in modal analysis labs from the command line."""
import argparse
from pathlib import Path

import numpy as np

from .diagnostics import energy, linear_frf, nonlinear_frf, suppression_bands
from .linear_modes import describe_modes, normal_modes
from .nonlinear_sim import simulate
from .plotting import plot_beating, plot_damping, plot_frf, plot_modes
from .systems import (
    absorber_system, critical_damping, critical_step, damped_oscillator,
    spring_mass_chain, symmetric_two_dof,
)


def _beating():
    # This commensurate ratio makes the first envelope exchange an exact
    # swap of the initial displacement, with both velocities zero.
    coupling = ((21 / 20)**2 - 1) / 2
    system = spring_mass_chain(coupling=coupling)
    modes = normal_modes(system)
    low, high, beat = symmetric_two_dof(coupling=coupling)
    time = np.linspace(0, beat, 2501)
    response = simulate(system, time, q0=[1, 0])
    energies = energy(system, response)
    print(f"Analytical ω (rad/s): {low:.8f}, {high:.8f}")
    describe_modes(modes)
    print(f"Beat period: {beat:.4f} s; first exchange: {beat / 2:.4f} s")
    print(f"Numerical displacement at exchange: {response.displacement[time.size // 2]}")
    print(f"Maximum relative energy drift: {np.max(abs(energies.total / energies.total[0] - 1)):.2e}")
    return {"beating": plot_beating(response, energies), "beating-modes": plot_modes(modes)}


def _damping():
    time = np.linspace(0, 22, 801)
    systems = ("underdamped", "critical", "overdamped")
    responses = {label: simulate(damped_oscillator(label), time, force=lambda _: 1.0)
                 for label in systems}
    exact = critical_step(time)
    difference = responses["critical"].displacement[:, 0] - exact
    print(f"Analytical and model critical c: {critical_damping(1, 1):.4f}, "
          f"{damped_oscillator('critical').c[0]:.4f} N·s/m")
    print(f"Maximum numerical critical step error: {np.max(abs(difference)):.2e} m")
    return {"damping": plot_damping(responses, exact=(time, exact))}


def _resonance():
    omega = np.linspace(0, 2, 801)
    curves = {}
    for zeta in (0.05, 0.15, 0.35):
        response = linear_frf(damped_oscillator(zeta=zeta), omega)[:, 0]
        theory = np.sqrt(1 - 2 * zeta**2)
        peak = omega[np.argmax(abs(response))]
        print(f"ζ={zeta:.2f}: analytical peak {theory:.5f}, sampled peak {peak:.5f} rad/s")
        curves[f"ζ = {zeta}"] = omega, response
    return {"resonance": plot_frf(curves, ylabel="Receptance (m/N)")}


def _absorber():
    omega = np.linspace(0.65, 1.35, 31)
    load, damping = 0.1, 0.06
    uncontrolled = linear_frf(damped_oscillator(zeta=damping / 2), omega, load=load)[:, 0]
    tmd = linear_frf(absorber_system(), omega, load=[load, 0])[:, 0]
    nes = nonlinear_frf(absorber_system("nes"), omega, load=[load, 0])[:, 0]
    theory = load / abs(1 - omega**2 + 1j * damping * omega)
    print(f"Bare SDOF analytical amplitude check: max error {np.max(abs(abs(uncontrolled) - theory)):.2e} m")
    for label, curve in (("TMD", tmd), ("NES", nes)):
        bands = suppression_bands(omega, uncontrolled, curve)
        width = sum(upper - lower for lower, upper in bands)
        print(f"{label}: sampled peak {np.max(abs(curve)):.4g} m; ≥3 dB suppression width {width:.4f} rad/s")
    return {"absorber": plot_frf({"Uncontrolled": (omega, uncontrolled),
                                 "Linear TMD": (omega, tmd), "Cubic NES": (omega, nes)})}


def main(argv=None):
    """Run one named demo and save its figures."""
    demos = {"beating": _beating, "damping": _damping,
             "resonance": _resonance, "absorber": _absorber}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=demos, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs"), help="figure directory")
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt

    args.output.mkdir(parents=True, exist_ok=True)
    for name, figure in demos[args.demo]().items():
        path = args.output / f"{name}.png"
        try:
            figure.savefig(path, dpi=240)
        finally:
            plt.close(figure)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
