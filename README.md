# Coupled Oscillator Modal Lab

A Python lab for normal modes, coupled motion, damping, resonance, and
vibration absorbers. It includes textbook reference solutions, reusable
numerical functions, three Jupyter labs, and four command-line demos.

## Quick start

From the repository root, use Python 3.10 or later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m modal_lab --demo beating
```

In PowerShell, activate with `.venv\Scripts\Activate.ps1`. For a runtime-only
installation, use `python -m pip install .`. Each demo prints a numerical
summary and writes 240 DPI PNGs to `outputs/` in your current directory.

| Command | What it shows |
| --- | --- |
| `python -m modal_lab --demo beating` | Two mode shapes and periodic exchange of motion and energy |
| `python -m modal_lab --demo damping` | Underdamped, critical, and overdamped force-step responses |
| `python -m modal_lab --demo resonance` | How damping shifts and limits the displacement resonance |
| `python -m modal_lab --demo absorber` | Uncontrolled, linear TMD, and cubic NES response under detuning |

Pass `--output path/to/figures` to change the destination. The absorber demo
integrates a separate time history at each drive frequency.

## Model a system in Python

```python
import numpy as np
from modal_lab import describe_modes, energy, normal_modes, simulate, spring_mass_chain
from modal_lab.plotting import plot_beating

system = spring_mass_chain(n=2, mass=1.0, wall=1.0, coupling=0.05)
modes = normal_modes(system)
describe_modes(modes)
response = simulate(system, np.linspace(0, 140, 2801), q0=[1.0, 0.0])
energies = energy(system, response)
plot_beating(response, energies).savefig("beating.png", dpi=240)
```

Use SI units for the examples. Coordinates are translational displacements
in metres, including horizontal bob displacement in the small-angle
pendulum builder. Array rows in a time history correspond to time samples,
and columns correspond to degrees of freedom.

## Governing equations

An element's deformation is $d_j=(Bq)_j$. A grounded element has one
nonzero entry in its row of $B$. A link between two coordinates takes
their difference. Element damping acts on deformation rate. For force
$F(t)$ and optional common base motion $y(t)$,

$$
M\ddot q+B^T\big[c\odot B\dot q+k\odot Bq
+k_3\odot(Bq)^3\big]=F(t)-Mr\ddot y(t).
$$

Here $r$ is the influence vector, $q$ is base-relative when using
base excitation, and powers and products inside the brackets are
elementwise. Let $D_c$ and $D_k$ be the diagonal matrices with entries
$c_j$ and $k_j$, respectively. The linear matrices are $C=B^T D_c B$ and
$K=B^T D_k B$. `simulate` converts the system to first order
and integrates it with SciPy RK45 or DOP853. An external harmonic force
has $F(t)=F_0\cos(\omega_d t)$, and a harmonic base **displacement**
supplied to `harmonic_base` is converted to **acceleration** before
integration.

### Normal modes and beating

The conservative linearized system solves
$Kv=\omega^2 Mv$ with $V^TMV=I$. Frequencies in Hz are $\omega/(2\pi)$.
A free-free chain can translate rigidly at zero frequency, with no
finite oscillation period. Nodes between masses are interpolated zero
crossings of the displayed discrete shape.

For two identical masses $m$ with wall springs $k$ and coupling $\kappa$,

$$
\omega_1=\sqrt{k/m},\qquad
\omega_2=\sqrt{(k+2\kappa)/m},\qquad
T_b=\frac{2\pi}{\omega_2-\omega_1}.
$$

Starting from $(q_1,q_2)=(A,0)$ at rest, the motions are
$A(\cos\omega_1t\pm\cos\omega_2t)/2$. The first envelope exchange is
at $T_b/2$. The demo chooses $\omega_2/\omega_1=21/20$ so displacement
and velocity swap exactly at this instant. A coupled spring can still
store energy there. The local-energy plot assigns half that spring's
energy to each mass.

### Damping and resonance

For one mass, $\zeta=c/c_{\rm crit}$ with $c_{\rm crit}=2\sqrt{km}$.
A constant force step $F$ from rest at critical damping gives
$x(t)=(F/k)[1-(1+\omega_nt)e^{-\omega_nt}]$.
The analytical frequency of the forced displacement peak is
$\omega_n\sqrt{1-2\zeta^2}$ for $\zeta<1/\sqrt 2$.
The linear complex response solves
$(K-\omega^2 M+i\omega C)\hat q=\hat F$.

### Energy and absorbers

$$
E=\sum_i \tfrac12 m_i\dot q_i^2
+\sum_j\left(\tfrac12 k_jd_j^2+\tfrac14 k_{3,j}d_j^4\right),
\qquad P_{\rm diss}=\sum_j c_j\dot d_j^2.
$$

With no drive and no damping, total energy should remain constant.
A TMD adds a linear attachment spring and a viscous damper. The NES
example uses a purely cubic attachment spring and relative damping.
Each nonlinear FRF point starts at rest, discards the initial cycles,
and extracts the complex fundamental over complete final cycles.
This response depends on forcing amplitude and may retain transients
or miss coexisting branches. Increase settling time and refine frequency
spacing to check a result. Suppression bands are interpolated threshold
intervals relative to a chosen baseline. A sampled 3 dB bandwidth is
the sum of their widths in rad/s.

## Project map

| Module | Use |
| --- | --- |
| `systems` | Build fixed/fixed, fixed/free, and free/free chains, pendulums, damped oscillators, TMDs and NES models. Look up analytical references. |
| `linear_modes` | Assemble mass, damping and stiffness matrices, solve the symmetric generalized eigenproblem, and describe mode phases and nodes. |
| `nonlinear_sim` | Integrate transients with external or base forcing and cubic links. |
| `diagnostics` | Track kinetic and spring energy, evaluate linear and nonlinear FRFs, measure suppression bands, and calculate one-sided FFT power. |
| `plotting` | Build Matplotlib figures for mode shapes, beating, damping and vibration control. |
| `__main__` | Run the four built-in labs and save their figures. |

Run `jupyter lab notebooks` to open the notebooks.

1. [Normal modes and beating](notebooks/01_undergrad_normal_modes_and_beating.ipynb)
2. [Critical damping and resonance](notebooks/02_critical_damping_and_resonance.ipynb)
3. [Nonlinear vibration absorber](notebooks/03_nonlinear_vibration_absorber.ipynb)

## Verification and publishing

Run `python -m pytest` to check mode frequencies to $10^{-10}$ rad/s,
mass orthogonality, the closed-form critical step, and undamped nonlinear
and linear relative energy drift below $10^{-8}$. Tests also execute
all four CLI demos. GitHub Actions tests and builds on Python 3.10–3.14.

Create an empty GitHub repository, then add it as your remote and push.

```bash
git remote add origin https://github.com/YOUR-ACCOUNT/coupled-oscillator-modal-lab.git
git push -u origin main
```

Released under the [MIT License](LICENSE).
