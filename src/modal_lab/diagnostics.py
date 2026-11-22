"""Energy, frequency response, suppression bandwidth, and FFT diagnostics."""
from dataclasses import dataclass

import numpy as np
from scipy.fft import rfft, rfftfreq
from scipy.signal.windows import hann

from .linear_modes import assemble
from .nonlinear_sim import _vector, harmonic_force, simulate


@dataclass(frozen=True)
class Energy:
    """Per-DOF kinetic/local energy, per-element potential, and system totals."""

    kinetic: np.ndarray
    spring: np.ndarray
    local: np.ndarray
    total: np.ndarray
    dissipation_power: np.ndarray


def energy(system, response):
    """Compute T_i = m_i v_i²/2 and U_j = k_j d_j²/2 + k3_j d_j⁴/4.

    For visualization, allocate each spring's potential equally among
    its participating DOFs. Coupling energy has no unique local split.
    Dissipation power is Σ c_j ḋ_j²; mechanical energy excludes work
    and dissipated heat.
    """
    displacement, velocity = response.displacement, response.velocity
    extension = displacement @ system.B.T
    speed = velocity @ system.B.T
    kinetic = 0.5 * system.masses * velocity**2
    spring = 0.5 * system.k * extension**2 + 0.25 * system.k3 * extension**4
    participants = (system.B != 0).astype(float)
    weights = participants / participants.sum(axis=1, keepdims=True)
    local = kinetic + spring @ weights
    return Energy(kinetic, spring, local, local.sum(axis=1), speed**2 @ system.c)


def _frequencies(omega):
    omega = np.asarray(omega, dtype=float)
    if omega.ndim != 1 or not omega.size or not np.isfinite(omega).all() or np.any(omega < 0):
        raise ValueError("provide a nonempty array of finite nonnegative angular frequencies")
    return omega


def linear_frf(system, omega, *, load=1.0):
    """Return complex displacements for unit or prescribed harmonic load.

    Evaluate (K − ω² M + iω C) q̂ = F̂. Rows are frequencies, columns
    are DOFs. Use a load vector with one nonzero entry for point forcing.
    For unit base acceleration, use load = −masses * influence.
    """
    if np.any(system.k3):
        raise ValueError("linear_frf requires zero cubic stiffness")
    omega = _frequencies(omega)
    force = _vector(load, system.ndof, "load")
    mass, damping, stiffness = assemble(system)
    frequencies = omega[:, None, None]
    dynamic = stiffness - frequencies**2 * mass + 1j * frequencies * damping
    # The explicit right-hand side column works on NumPy 1.x and 2.x.
    rhs = np.broadcast_to(force, (omega.size, system.ndof))[..., None]
    return np.linalg.solve(dynamic, rhs)[..., 0]


def nonlinear_frf(system, omega, *, load=1.0, settle_cycles=44, measure_cycles=8,
                  samples_per_cycle=32, rtol=2e-7, atol=2e-9):
    """Return the fundamental response for each harmonic forcing frequency.

    Each run starts from rest, discards settle_cycles, and samples an
    integer number of final cycles. Return complex displacement at the
    drive frequency, which depends on load amplitude for nonlinear systems.
    A finite settling interval may retain transients. Multistable branches
    depend on initial conditions, and this independent sweep samples one.
    """
    omega = _frequencies(omega)
    if np.any(omega == 0):
        raise ValueError("time-domain sweeps require positive angular frequency")
    for value, name, minimum in ((settle_cycles, "settle_cycles", 0),
                                 (measure_cycles, "measure_cycles", 1),
                                 (samples_per_cycle, "samples_per_cycle", 8)):
        if not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    force = _vector(load, system.ndof, "load")
    result = np.empty((omega.size, system.ndof), dtype=complex)
    count = measure_cycles * samples_per_cycle
    for i, frequency in enumerate(omega):
        period = 2 * np.pi / frequency
        sample = period * (settle_cycles + np.arange(count) / samples_per_cycle)
        times = np.r_[0.0, sample] if settle_cycles else sample
        response = simulate(system, times, force=harmonic_force(force, frequency),
                            max_step=period / 16, rtol=rtol, atol=atol)
        phase = np.exp(-1j * frequency * sample)
        result[i] = (2 / count) * (phase @ response.displacement[-count:])
    return result


def suppression_bands(omega, reference, controlled, *, db=3.0):
    """Return contiguous angular-frequency intervals satisfying target attenuation.

    Interpolate crossings of |controlled| − 10^(−dB/20)|reference|.
    Endpoints are clipped to the sampled range. Bandwidth is the sum of
    interval widths, and the grid limits the resolution.
    """
    omega = _frequencies(omega)
    reference, controlled = np.abs(reference), np.abs(controlled)
    if (omega.size < 2 or np.any(np.diff(omega) <= 0)
            or reference.shape != omega.shape or controlled.shape != omega.shape
            or not np.isfinite(reference).all() or not np.isfinite(controlled).all()
            or np.any(reference <= 0) or not np.isfinite(db) or db < 0):
        raise ValueError("provide sorted frequencies, positive reference, finite response and nonnegative dB")
    delta = controlled - 10**(-db / 20) * reference
    inside = delta <= 0
    changes = np.diff(np.r_[False, inside, False].astype(int))
    bands = []
    for first, last in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        start, stop = omega[first], omega[last - 1]
        if first > 0:
            start = np.interp(0, delta[first - 1:first + 1][::-1], omega[first - 1:first + 1][::-1])
        if last < omega.size:
            stop = np.interp(0, delta[last - 1:last + 1], omega[last - 1:last + 1])
        bands.append((float(start), float(stop)))
    return bands


def power_spectrum(time, signal, *, window="hann", detrend=True):
    """Return frequencies in Hz and one-sided mean-square FFT bin powers.

    Input must be uniformly sampled. With the boxcar window, summed
    powers equal the mean square of the optionally demeaned signal.
    Hann scaling compensates for its mean-square window weight.
    """
    time, signal = np.asarray(time, dtype=float), np.asarray(signal, dtype=float)
    if (time.ndim != 1 or time.size < 3 or signal.shape != time.shape
            or not np.isfinite(time).all() or not np.isfinite(signal).all()):
        raise ValueError("provide matching finite vectors of at least three samples")
    spacing = np.diff(time)
    dt = spacing.mean()
    if dt <= 0 or not np.allclose(spacing, dt, rtol=1e-7, atol=0):
        raise ValueError("sample times must be uniformly spaced and increasing")
    if window == "hann":
        weights = hann(time.size, sym=False)
    elif window == "boxcar":
        weights = np.ones(time.size)
    else:
        raise ValueError("window must be 'hann' or 'boxcar'")
    centered = signal - signal.mean() if detrend else signal
    power = abs(rfft(centered * weights))**2 / (time.size * np.dot(weights, weights))
    power[1:(-1 if time.size % 2 == 0 else None)] *= 2
    return rfftfreq(time.size, dt), power
