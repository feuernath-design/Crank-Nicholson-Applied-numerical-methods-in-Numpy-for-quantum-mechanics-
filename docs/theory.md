# Theory and numerical method

Atomic units are used throughout: $\hbar = m_e = e = 1$ (unit of time ≈ 24.19 as, unit of energy ≈ 27.21 eV, unit of field ≈ 5.14 × 10¹¹ V/m).

## 1. Model

The time-dependent Schrödinger equation (TDSE) for one electron in one dimension,

$$
i\,\partial_t \psi(x,t) = H(t)\,\psi(x,t), \qquad
H(t) = -\frac{1}{2}\frac{\partial^2}{\partial x^2} + V(x) + F(t)\,x ,
$$

is solved for two potentials:

| Potential | $V(x)$ | Use |
|---|---|---|
| Harmonic oscillator | $\tfrac12\omega^2x^2$ | validation: exact spectrum $E_n = \omega(n+\tfrac12)$, coherent states |
| Soft-core Coulomb | $-1/\sqrt{x^2+1}$ | 1D model atom (Javanainen, Eberly & Su 1988): $E_0 = -0.66986$, $E_1 = -0.27498$ a.u. at $\Delta x = 0.1$ (converged: $-0.66978$, $-0.27490$) |

The laser is treated in the dipole approximation, length gauge, with field $F(t) = F_0\, f(t)\sin(\omega t)$ and an envelope $f$ (constant, $\sin^2$ turn-on or trapezoid). The field amplitude is written $F_0$ to avoid confusion with the ground-state energy $E_0$.

## 2. Spatial discretisation

On a uniform grid $x_j = x_{\min} + j\,\Delta x$ with $\psi = 0$ outside the box, the 3-point stencil

$$
\psi''(x_j) \simeq \frac{\psi_{j-1} - 2\psi_j + \psi_{j+1}}{\Delta x^2} + O(\Delta x^2)
$$

turns $H$ into a real symmetric **tridiagonal** matrix with diagonal $1/\Delta x^2 + V(x_j) + F(t)x_j$ and constant off-diagonal $-1/(2\Delta x^2)$. Every operation (matrix–vector product, linear solve) is therefore $O(N)$.

Wavefunctions are normalised with the grid weight: $\sum_j |\psi_j|^2\,\Delta x = 1$, and $\langle \phi|\psi\rangle = \sum_j \phi_j^*\psi_j\,\Delta x$.

**Expected accuracy:** eigenvalues converge as $O(\Delta x^2)$ — verified in `scripts/01_spatial_convergence.py` (measured slope 2.003).

## 3. Stationary states: shifted inverse iteration

For a shift $\varepsilon$ close to an eigenvalue, iterate

$$
(H-\varepsilon)\,\psi_{k+1} = \phi_k, \qquad \phi_{k+1} = \psi_{k+1}/\lVert\psi_{k+1}\rVert .
$$

Expanding $\phi_0 = \sum_m c_m u_m$ on eigenvectors, $(H-\varepsilon)^{-k}\phi_0 = \sum_m c_m (E_m-\varepsilon)^{-k} u_m$: the eigenvector whose energy is closest to $\varepsilon$ dominates geometrically.

**Shift update.** For an exact eigenvector, $\langle\phi|(H-\varepsilon)^{-1}|\phi\rangle = 1/(E-\varepsilon)$, hence

$$
\varepsilon \leftarrow \varepsilon + \frac{1}{\langle\phi_k|\psi_{k+1}\rangle}
$$

lands on $E$. This makes the iteration converge very quickly in practice (7 iterations for the soft-Coulomb ground state, against 25 with a fixed shift). A few fixed-shift iterations are performed first so that the method locks onto the level nearest to the *initial* shift. Convergence is declared when the residual $\lVert H\phi - E\phi\rVert$ falls below $10^{-10}\max(1,|E|)$.

## 4. Time propagation: Crank–Nicolson

Averaging the explicit and implicit Euler steps gives

$$
\Big(1 + \tfrac{i\Delta t}{2} H_{n+1/2}\Big)\,\psi^{n+1} = \Big(1 - \tfrac{i\Delta t}{2} H_{n+1/2}\Big)\,\psi^{n},
\qquad H_{n+1/2} = H\big(t_n + \tfrac{\Delta t}{2}\big).
$$

### Unitarity

The one-step propagator is the **Cayley transform** $U = (1 + iA)^{-1}(1 - iA)$ with $A = \Delta t\,H/2$ Hermitian. Since $(1-iA)$ and $(1+iA)^{-1}$ commute and $(1-iA)^\dagger = 1+iA$,

$$
U^\dagger U = (1+iA)(1-iA)^{-1}(1+iA)^{-1}(1-iA) = 1 .
$$

The norm is conserved **exactly** (to round-off, ~10⁻¹⁴ observed) for any $\Delta t$ — the scheme is unconditionally stable. It is also time-reversible: $U(-\Delta t) = U(\Delta t)^{-1}$ (tested).

### Phase error

For an eigenstate $H u = E u$,

$$
U u = \frac{1 - iE\Delta t/2}{1 + iE\Delta t/2}\, u = e^{-2i\arctan(E\Delta t/2)}\, u ,
$$

so the numerical state evolves with the effective energy

$$
\bar E = \frac{2}{\Delta t}\arctan\frac{E\Delta t}{2} = E - \frac{E^3\Delta t^2}{12} + O(\Delta t^4).
$$

The scheme is **second order** in time, and the phase error grows like $E^3$: high-energy components are the first to be dephased. Note that $E$ is measured from the zero of the potential; shifting the energy origin changes only a global phase, which is unobservable, but it does change this phase error, so the energy reference matters when choosing $\Delta t$. `scripts/02_time_step_error.py` measures $|\bar E - E|$ by propagation and matches this closed form to the printed precision.

### Time-dependent Hamiltonian

Evaluating $H$ at the mid-point $t_n + \Delta t/2$ keeps the local error $O(\Delta t^3)$, hence the global error $O(\Delta t^2)$, also with a laser field (verified by a convergence test: measured order 2.00).

### Cost

$1 + \tfrac{i\Delta t}{2}H$ is tridiagonal: each step is one $O(N)$ solve (LAPACK `?gtsv` with partial pivoting, called by `scipy.linalg.solve_banded` for one sub- and one super-diagonal, in Python; the Thomas algorithm, without pivoting, in Fortran). The dense textbook version (`tdse1d.reference`) costs $O(N^3)$ per step and is only kept as a test reference.

## 5. Physics of the driven atom

### Sudden versus adiabatic turn-on

With $F(t) = F_0\sin(\omega t)$ switched on abruptly at $t = 0$, the sudden kick excites the atom's *free* response at the Bohr frequencies, which beats against the forced response at $\omega$ ($\omega \pm \omega_{10}$, $\omega_{10} = E_1 - E_0 = 0.395$ a.u.). A smooth $\sin^2$ turn-on over a few cycles removes this transient: the atom then follows the field-dressed (Floquet) state, and $|\langle\psi_0|\psi(t)\rangle|$ oscillates at $2\omega$ (period $\pi/\omega$) with an amplitude of order $F_0^2$, on top of a slow decay due to ionisation. In a two-level picture the excited-state amplitude is $c_1 \propto F_0\left[\frac{e^{i\omega t}}{\omega_{10}+\omega} - \frac{e^{-i\omega t}}{\omega_{10}-\omega}\right]$, so $|c_1|^2$ contains a term $\propto -\cos(2\omega t)/(\omega_{10}^2-\omega^2)$. Because the drive is above the first transition ($\omega > \omega_{10}$), this term is largest when the field crosses zero: the ground-state loss is in antiphase with $F(t)^2$ (correlation $-0.999$ in the simulation), whereas below resonance it would follow $F(t)^2$.

### One-photon ionisation (photoelectric effect)

All frequencies used ($\omega \ge 0.8$ a.u.) exceed the ionisation potential $I_p = 0.67$ a.u., so a single photon ionises the atom. First-order time-dependent perturbation theory predicts:

* **$1 - P_0 \propto F_0^2$** at weak field — measured log–log slope 2.0000 for $F_0 \le 0.01$; deviations appear at $F_0 \gtrsim 0.05$ (depletion of the ground state, higher orders);
* **$1 - P_0$ linear in the pulse duration** (Fermi's golden rule: constant rate $\Gamma$) — measured $\Gamma \approx 4.8\times10^{-6}$ a.u. at $F_0 = 0.01$, $\omega = 1$ (the negative intercept of the linear fit comes from the ramps, during which the field is weaker);
* an ionisation **rate that decreases with $\omega$** above threshold, following the photo-ionisation cross-section. The scan uses pulses with a fixed number of cycles, whose duration scales as $1/\omega$, so the rate is obtained by dividing the depletion by $T_{\rm eff} = \int f(t)^2\,dt$. At $\omega = 1$ this gives $4.9\times10^{-6}$ a.u., consistent with the duration scan; the rate drops by a factor of about 500 between $\omega = 0.8$ and $2.5$.

### Box size

The box $[-150, 150]$ a.u. is large enough for the ionised wavepacket not to reach the edges during the pulses studied. For longer or stronger pulses, `tdse1d.absorbing_mask` provides a $\cos^{1/8}$ absorber; the lost norm is then the ionisation probability.

## References

1. J. Crank and P. Nicolson, *A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type*, Proc. Camb. Phil. Soc. **43**, 50 (1947).
2. A. Goldberg, H. M. Schey and J. L. Schwartz, *Computer-generated motion pictures of one-dimensional quantum-mechanical transmission and reflection phenomena*, Am. J. Phys. **35**, 177 (1967).
3. J. Javanainen, J. H. Eberly and Q. Su, *Numerical simulations of multiphoton ionization and above-threshold electron spectra*, Phys. Rev. A **38**, 3430 (1988).
4. J. L. Krause, K. J. Schafer and K. C. Kulander, *Calculation of photoemission from atoms subject to intense laser fields*, Phys. Rev. A **45**, 4998 (1992).
