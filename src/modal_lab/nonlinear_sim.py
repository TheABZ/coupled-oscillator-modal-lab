"""Time integration of damped, driven, and cubic mechanical systems."""
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Response:
    """Time samples and row-wise displacement and velocity vectors."""

    time: np.ndarray
    displacement: np.ndarray
    velocity: np.ndarray


def _vector(value, n, name):
    array = np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def harmonic_force(amplitude, omega, *, phase=0.0):
    """Return the force function F(t) = amplitude cos(ωt + phase)."""
    amplitude = np.asarray(amplitude, dtype=float).copy()
    if not np.isfinite(amplitude).all() or not np.isfinite([omega, phase]).all() or omega < 0:
        raise ValueError("harmonic forcing parameters must be finite with nonnegative frequency")

    def force(time):
        return amplitude * np.cos(omega * time + phase)

    return force


def harmonic_base(displacement, omega, *, phase=0.0):
    """Return base acceleration for y(t) = displacement cos(ωt + phase)."""
    return harmonic_force(-np.asarray(displacement, dtype=float) * omega**2, omega, phase=phase)


def simulate(system, time, *, q0=0.0, v0=0.0, force=None, base_acceleration=None,
             influence=1.0, method="DOP853", rtol=1e-10, atol=1e-12, max_step=np.inf):
    """Integrate M q̈ + Bᵀ(c B q̇ + k B q + k3 (B q)³) = F − M r ÿ.

    q is displacement relative to the base for base-driven runs. The
    influence vector r specifies how ground translation acts on each DOF.
    Force and base_acceleration are callables of time.
    """
    time = np.asarray(time, dtype=float)
    if (time.ndim != 1 or time.size < 2 or not np.isfinite(time).all()
            or np.any(np.diff(time) <= 0)):
        raise ValueError("time must be finite and strictly increasing with at least two samples")
    if method not in ("RK45", "DOP853"):
        raise ValueError("method must be 'RK45' or 'DOP853'")
    if not np.isfinite([rtol, atol]).all() or min(rtol, atol) <= 0 or not max_step > 0:
        raise ValueError("integration tolerances and max_step must be positive")
    n = system.ndof
    b = system.B
    y0 = np.r_[_vector(q0, n, "q0"), _vector(v0, n, "v0")]
    r = _vector(influence, n, "influence")

    def rhs(t, state):
        q, velocity = state[:n], state[n:]
        extension = b @ q
        link_force = system.k * extension + system.k3 * extension**3 + system.c * (b @ velocity)
        acceleration = -(b.T @ link_force) / system.masses
        if force is not None:
            acceleration += np.broadcast_to(force(t), (n,)) / system.masses
        if base_acceleration is not None:
            acceleration -= r * base_acceleration(t)
        return np.r_[velocity, acceleration]

    solved = solve_ivp(rhs, (time[0], time[-1]), y0, t_eval=time, method=method,
                       rtol=rtol, atol=atol, max_step=max_step)
    if not solved.success:
        raise RuntimeError(f"integration failed: {solved.message}")
    return Response(solved.t, solved.y[:n].T, solved.y[n:].T)
