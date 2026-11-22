"""Report-ready plots for mode shapes, beating, damping, and FRFs."""
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .linear_modes import mode_nodes


def plot_modes(modes):
    """Plot normalized mode shapes with stationary locations marked."""
    count = modes.omega.size
    fig, axes = plt.subplots(count, 1, figsize=(7, max(2.4, 1.65 * count)),
                             sharex=True, squeeze=False, layout="constrained")
    positions = np.arange(1, modes.shapes.shape[0] + 1)
    for i, (axis, shape, omega) in enumerate(zip(axes[:, 0], modes.shapes.T, modes.omega), 1):
        amplitude = shape / np.abs(shape).max()
        axis.axhline(0, color="0.65", lw=0.8)
        axis.plot(positions, amplitude, "o-", color="#175c80")
        nodes = mode_nodes(shape)
        axis.scatter(nodes, np.zeros_like(nodes), marker="x", color="#a74732", zorder=3)
        axis.set(ylabel="Relative q", title=f"Mode {i} · {omega:.4g} rad/s", ylim=(-1.2, 1.2))
    axes[-1, 0].xaxis.set_major_locator(MaxNLocator(integer=True))
    axes[-1, 0].set_xlabel("Mass number")
    return fig


def plot_beating(response, energies):
    """Show displacement and per-mass mechanical energy over time."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, layout="constrained")
    for i in range(response.displacement.shape[1]):
        name = f"Mass {i + 1}"
        axes[0].plot(response.time, response.displacement[:, i], label=name, lw=1.15)
        axes[1].plot(response.time, energies.local[:, i], label=name)
    axes[0].set(ylabel="Displacement (m)")
    axes[1].set(ylabel="Local energy (J)", xlabel="Time (s)")
    for axis in axes:
        axis.legend(frameon=False)
    return fig


def plot_damping(responses, *, exact=None):
    """Compare named single-DOF step responses, optionally with an analytic curve."""
    fig, axis = plt.subplots(figsize=(7.2, 4), layout="constrained")
    for label, response in responses.items():
        axis.plot(response.time, response.displacement[:, 0], label=label)
    if exact is not None:
        time, values = exact
        axis.plot(time, values, "k--", lw=1, label="Critical analytical")
    axis.set(xlabel="Time (s)", ylabel="Displacement (m)")
    axis.legend(frameon=False)
    return fig


def plot_frf(curves, *, ylabel="Displacement amplitude (m)"):
    """Plot named (ω, complex amplitude) curves on a semilog scale."""
    fig, axis = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    for label, (omega, response) in curves.items():
        axis.semilogy(omega, np.abs(response), marker="." if "NES" in label else None, label=label)
    axis.set(xlabel="Drive angular frequency (rad/s)", ylabel=ylabel)
    axis.legend(frameon=False)
    axis.grid(alpha=0.2)
    return fig
