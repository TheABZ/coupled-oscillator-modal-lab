"""Mechanical system builders and closed-form textbook references."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class System:
    """Lumped system with diagonal masses and one row of B per element.

    Element deformation is d = B q. For a translational spring between
    DOFs i and j, its row has +1 at i and −1 at j. A grounded element
    has just +1. Each element carries coefficients k, c and k3.
    """

    masses: np.ndarray
    B: np.ndarray
    k: np.ndarray
    c: np.ndarray
    k3: np.ndarray

    def __post_init__(self):
        arrays = {name: np.asarray(getattr(self, name), dtype=float).copy()
                  for name in ("masses", "B", "k", "c", "k3")}
        masses, b, k, c, k3 = arrays.values()
        if masses.ndim != 1 or not masses.size or np.any(masses <= 0):
            raise ValueError("provide a nonempty vector of positive masses")
        if b.ndim != 2 or b.shape[1] != masses.size:
            raise ValueError("B must have one column per degree of freedom")
        if any(a.shape != (b.shape[0],) for a in (k, c, k3)):
            raise ValueError("provide one k, c and k3 for each element")
        if any(not np.isfinite(a).all() for a in arrays.values()) or np.any(k < 0) or np.any(c < 0):
            raise ValueError("coefficients must be finite with nonnegative k and c")
        if np.any(np.all(b == 0, axis=1)):
            raise ValueError("each element must act on at least one coordinate")
        for name, array in arrays.items():
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    @property
    def ndof(self):
        return self.masses.size


def spring_mass_chain(n=2, *, mass=1.0, coupling=0.1, wall=1.0,
                      boundary="fixed-fixed", damping=0.0, cubic=0.0):
    """Build a chain with scalar or per-DOF mass and nearest-neighbor links.

    Fixed-free grounds the leftmost mass. Free-free has no grounded
    elements. Coupling, damping and cubic coefficients are scalar and
    shared by interior links. Wall selects the grounded linear stiffness.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if boundary not in ("fixed-fixed", "fixed-free", "free-free"):
        raise ValueError("unknown boundary condition")
    if not np.isfinite([coupling, wall, damping, cubic]).all() or min(coupling, wall, damping) < 0:
        raise ValueError("chain coefficients must be finite, with nonnegative linear stiffness and damping")
    identity = np.eye(n)
    rows = identity[:-1] - identity[1:]
    ends = {"fixed-fixed": [0, n - 1], "fixed-free": [0], "free-free": []}[boundary]
    b = np.vstack((rows, identity[ends]))
    return System(np.broadcast_to(mass, (n,)), b,
                  np.r_[np.full(n - 1, coupling), np.full(len(ends), wall)],
                  np.r_[np.full(n - 1, damping), np.zeros(len(ends))],
                  np.r_[np.full(n - 1, cubic), np.zeros(len(ends))])


def coupled_pendulums(n=2, *, mass=1.0, length=1.0, gravity=9.81, coupling=0.1):
    """Small-angle pendulums coupled by springs at the bobs.

    The coordinates are horizontal bob displacements q ≈ length·θ,
    so gravity contributes a grounded stiffness m g / length.
    """
    if not np.isfinite([length, gravity]).all() or min(length, gravity) <= 0:
        raise ValueError("length and gravity must be finite and positive")
    chain = spring_mass_chain(n, mass=mass, coupling=coupling, boundary="free-free")
    return System(chain.masses, np.vstack((chain.B, np.eye(n))),
                  np.r_[chain.k, chain.masses * gravity / length],
                  np.zeros(2 * n - 1), np.zeros(2 * n - 1))


def critical_damping(mass, stiffness):
    """Return c_crit = 2 sqrt(k m) for one grounded mass."""
    if not np.isfinite([mass, stiffness]).all() or min(mass, stiffness) <= 0:
        raise ValueError("mass and stiffness must be finite and positive")
    return 2 * np.sqrt(stiffness * mass)


def damped_oscillator(regime="underdamped", *, mass=1.0, stiffness=1.0, zeta=None):
    """Build one mass with viscous damping c = 2 ζ sqrt(k m)."""
    presets = {"underdamped": 0.25, "critical": 1.0, "critically_damped": 1.0,
               "overdamped": 2.0}
    if regime not in presets:
        raise ValueError("unknown damping regime")
    ratio = presets[regime] if zeta is None else zeta
    return System([mass], [[1]], [stiffness], [critical_damping(mass, stiffness) * ratio], [0])


def absorber_system(kind="tmd", *, mass=1.0, stiffness=1.0, damping=0.06,
                    attachment_mass=0.08, tune=None, attachment_damping=0.025,
                    cubic=3.0):
    """Build a grounded primary with a linear TMD or purely cubic NES.

    The TMD attachment has k_a = m_a ω_t², with ω_t defaulting to the
    primary undamped frequency. The NES attachment uses k1=0 and k3>0.
    Both attachment dampers act on relative velocity.
    """
    ccrit = critical_damping(mass, stiffness)
    if kind not in ("tmd", "nes"):
        raise ValueError("kind must be 'tmd' or 'nes'")
    if kind == "tmd":
        omega = ccrit / (2 * mass) if tune is None else tune
        if not np.isfinite(omega) or omega <= 0:
            raise ValueError("tune must be a positive angular frequency")
        attachment_k, attachment_k3 = attachment_mass * omega**2, 0.0
    else:
        if not np.isfinite(cubic) or cubic <= 0:
            raise ValueError("NES cubic stiffness must be finite and positive")
        attachment_k, attachment_k3 = 0.0, cubic
    return System([mass, attachment_mass], [[1, 0], [1, -1]],
                  [stiffness, attachment_k], [damping, attachment_damping],
                  [0, attachment_k3])


def symmetric_two_dof(mass=1.0, wall=1.0, coupling=0.1):
    """Return (ω_in, ω_out, beat period) for identical grounded masses.

    ω_in = sqrt(k/m), ω_out = sqrt((k+2κ)/m). The envelope repeats in
    magnitude after 2π/(ω_out−ω_in); the first exchange is half that.
    """
    critical_damping(mass, wall)
    if not np.isfinite(coupling) or coupling < 0:
        raise ValueError("coupling must be finite and nonnegative")
    low, high = np.sqrt(np.array([wall, wall + 2 * coupling]) / mass)
    gap = 2 * coupling / (mass * (high + low))
    return low, high, 2 * np.pi / gap if gap else np.inf


def critical_step(time, *, mass=1.0, stiffness=1.0, force=1.0):
    """Zero-initial-condition critical response to a constant force F.

    x(t) = (F/k)[1 − (1 + ω_n t) exp(−ω_n t)].
    """
    critical_damping(mass, stiffness)
    time = np.asarray(time, dtype=float)
    if np.any(time < 0) or not np.isfinite(time).all() or not np.isfinite(force):
        raise ValueError("time must be finite and nonnegative, and force finite")
    scaled = np.sqrt(stiffness / mass) * time
    return (force / stiffness) * (-np.expm1(-scaled) - scaled * np.exp(-scaled))
