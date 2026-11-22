"""Assembly and mass-normalized conservative normal modes."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class Modes:
    """Angular frequencies, frequencies, periods, and column-wise mode shapes."""

    omega: np.ndarray
    hz: np.ndarray
    periods: np.ndarray
    shapes: np.ndarray


def assemble(system):
    """Return M = diag(m), C = Bᵀ diag(c) B and K = Bᵀ diag(k) B."""
    b = system.B
    return np.diag(system.masses), b.T @ (system.c[:, None] * b), b.T @ (system.k[:, None] * b)


def normal_modes(system):
    """Solve K v = ω² M v with Vᵀ M V = I, ignoring damping and cubic terms."""
    mass, _, stiffness = assemble(system)
    values, vectors = eigh(stiffness, mass)
    scale = np.max(np.abs(values))
    tolerance = np.finfo(float).eps * len(values) * scale * 8
    values[np.abs(values) <= tolerance] = 0
    if np.any(values < 0):
        raise ValueError("negative squared frequency indicates an unstable linear system")
    omega = np.sqrt(values)
    hz = omega / (2 * np.pi)
    periods = np.divide(1.0, hz, out=np.full_like(hz, np.inf), where=hz > 0)
    # Fix each arbitrary sign by making its largest component positive.
    anchors = vectors[np.argmax(abs(vectors), axis=0), np.arange(values.size)]
    vectors *= np.sign(anchors)
    return Modes(omega, hz, periods, vectors)


def mode_nodes(shape, *, tolerance=1e-8):
    """Return node locations in one-based DOF coordinates.

    Include stationary masses and zero crossings of the piecewise-linear
    line through adjacent masses. Locations between masses are illustrative.
    """
    shape = np.asarray(shape, dtype=float)
    active = np.abs(shape) > tolerance * np.max(np.abs(shape))
    crossings = np.flatnonzero(active[:-1] & active[1:] & (np.sign(shape[:-1]) != np.sign(shape[1:])))
    between = crossings + 1 + abs(shape[crossings]) / (abs(shape[crossings]) + abs(shape[crossings + 1]))
    return np.sort(np.r_[np.flatnonzero(~active) + 1, between])


def describe_modes(modes, *, file=None):
    """Print and return a text description of each undamped mode."""
    lines = []
    for i, (omega, frequency, shape) in enumerate(zip(modes.omega, modes.hz, modes.shapes.T), start=1):
        nodes = mode_nodes(shape)
        active = shape[np.abs(shape) > 1e-8 * np.max(np.abs(shape))]
        if omega == 0:
            motion = "zero-frequency rigid-body mode or unrestrained linear mechanism"
        elif np.any(active > 0) and np.any(active < 0):
            motion = "out-of-phase motion"
        elif active.size == 1:
            motion = "one moving mass"
        else:
            motion = "in-phase motion"
        locations = ", ".join(f"{node:.3g}" for node in nodes) if nodes.size else "none"
        lines.append(f"Mode {i}: {omega:.8g} rad/s ({frequency:.8g} Hz), {motion}. Nodes: {locations}.")
    summary = "\n".join(lines)
    print(summary, file=file)
    return summary
