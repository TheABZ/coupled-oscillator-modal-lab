"""Physics regressions and CLI smoke tests."""
import subprocess
import sys

import numpy as np
import pytest

from modal_lab import (
    absorber_system, assemble, coupled_pendulums, critical_damping, critical_step,
    damped_oscillator, describe_modes, energy, harmonic_base, harmonic_force, linear_frf,
    normal_modes, power_spectrum, simulate, spring_mass_chain, suppression_bands,
    symmetric_two_dof,
)


def test_two_dof_closed_form():
    params = {"mass": 2.0, "wall": 4.2, "coupling": 0.35}
    numeric = normal_modes(spring_mass_chain(**params))
    low, high, beat = symmetric_two_dof(**params)
    assert np.max(abs(numeric.omega - [low, high])) < 1e-10
    assert beat == pytest.approx(2 * np.pi / (high - low))
    text = describe_modes(numeric)
    assert "in-phase" in text and "out-of-phase" in text and "1.5" in text


@pytest.mark.parametrize("boundary", ["fixed-fixed", "fixed-free", "free-free"])
def test_mass_orthogonality(boundary):
    model = spring_mass_chain(6, mass=np.arange(1.0, 7.0), boundary=boundary)
    modes = normal_modes(model)
    mass, _, stiffness = assemble(model)
    np.testing.assert_allclose(modes.shapes.T @ mass @ modes.shapes, np.eye(6), atol=1e-13)
    np.testing.assert_allclose(stiffness @ modes.shapes, mass @ modes.shapes * modes.omega**2,
                               atol=1e-13)
    assert (modes.omega[0] == 0) == (boundary == "free-free")
    if boundary == "free-free":
        assert "rigid-body" in describe_modes(modes)


def test_pendulum_in_phase_mode():
    model = coupled_pendulums(mass=[1.0, 2.0], length=0.8)
    assert normal_modes(model).omega[0] == pytest.approx(np.sqrt(9.81 / 0.8))


def test_critical_step_and_zero_overshoot():
    model = damped_oscillator("critical", mass=1.7, stiffness=3.2)
    time = np.linspace(0, 16, 701)
    response = simulate(model, time, force=lambda _: 2.0, method="RK45")
    expected = critical_step(time, mass=1.7, stiffness=3.2, force=2.0)
    assert model.c[0] == pytest.approx(critical_damping(1.7, 3.2))
    np.testing.assert_allclose(response.displacement[:, 0], expected, atol=2e-10, rtol=0)
    assert np.min(np.diff(response.displacement[:, 0])) >= 0
    assert response.displacement[:, 0].max() < 2 / 3.2


@pytest.mark.parametrize("cubic", [0.0, 3.0])
def test_undamped_energy(cubic):
    model = spring_mass_chain(coupling=0.25, cubic=cubic)
    run = simulate(model, np.linspace(0, 35, 1501), q0=[0.7, -0.2], v0=[0.1, 0.4])
    energies = energy(model, run)
    drift = np.max(abs(energies.total / energies.total[0] - 1))
    assert drift < 1e-8
    assert np.max(abs(energies.total - energies.kinetic.sum(1) - energies.spring.sum(1))) < 1e-14
    assert np.all(energies.dissipation_power == 0)


def test_base_forcing_and_linear_frf():
    model = damped_oscillator("underdamped", mass=2.0, stiffness=3.0)
    omega = 1.2
    time = np.linspace(0, 14, 501)
    base = simulate(model, time, base_acceleration=harmonic_base(0.3, omega))
    force = simulate(model, time, force=harmonic_force(2 * 0.3 * omega**2, omega))
    np.testing.assert_allclose(base.displacement, force.displacement, atol=1e-12)
    tmd = absorber_system(attachment_damping=0.0)
    assert abs(linear_frf(tmd, [1.0], load=[1.0, 0])[0, 0]) < 1e-13


def test_power_and_interpolated_suppression_band():
    time = np.arange(1024) / 128
    signal = np.cos(2 * np.pi * 6 * time) + 0.2 * np.cos(2 * np.pi * 18 * time)
    hz, power = power_spectrum(time, signal, window="boxcar")
    assert hz[np.argmax(power)] == 6
    assert power[np.argmin(abs(hz - 18))] == pytest.approx(0.02)
    assert power.sum() == pytest.approx(np.mean(signal**2))
    assert suppression_bands([1, 2, 3], [1, 1, 1], [1, 0, 1], db=20*np.log10(2)) == [(1.5, 2.5)]


@pytest.mark.parametrize("demo", ["beating", "damping", "resonance", "absorber"])
def test_cli_demo(demo, tmp_path, monkeypatch):
    monkeypatch.setenv("MPLBACKEND", "Agg")
    proc = subprocess.run(
        [sys.executable, "-m", "modal_lab", "--demo", demo, "--output", str(tmp_path)],
        capture_output=True, text=True, timeout=120, check=True,
    )
    assert "Analytical" in proc.stdout or "analytical" in proc.stdout
    assert (tmp_path / f"{demo}.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
