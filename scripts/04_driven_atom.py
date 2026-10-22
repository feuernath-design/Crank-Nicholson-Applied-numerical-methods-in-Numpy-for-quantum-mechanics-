"""1D soft-Coulomb atom in a laser field: sudden versus adiabatic turn-on.

H(t) = -1/2 d^2/dx^2 - 1/sqrt(1 + x^2) + E(t) x on [-150, 150] (dx = 0.1).
The ground state (E0 = -0.6699 a.u.) is obtained by inverse iteration and
propagated in E(t) = E0 f(t) sin(omega t) with E0 = 0.01, omega = 1 a.u.,
128 steps per optical cycle.

* Sudden turn-on (f = 1 from t = 0): the abrupt switch excites the free
  response of the atom, which beats at omega +/- omega_10
  (omega_10 = E1 - E0 = 0.395 a.u.) on top of the forced response.
* sin^2 turn-on over 4 cycles: the transient disappears and the overlap
  |<psi0|psi(t)>| becomes smooth and periodic with period pi/omega (the
  ground state is adiabatically polarised, a quadratic effect in E(t)),
  slowly decaying because of one-photon ionisation (omega > Ip = 0.67 a.u.).

The script also runs the numerical checks quoted in the README: norm
conservation, time-step convergence and the field-free reference.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, save

from tdse1d import (
    CrankNicolson,
    Grid,
    Hamiltonian1D,
    LaserField,
    constant_envelope,
    inverse_iteration,
    potentials,
    sin2_ramp_envelope,
)


def run(h, ground, field, dt, n_steps, record_every=1):
    grid = h.grid
    return CrankNicolson(h, dt, field=field).propagate(
        ground.state,
        n_steps,
        record_every=record_every,
        observables={
            "overlap": lambda psi: abs(grid.inner(ground.state, psi)),
            "norm": grid.norm2,
        },
    )


def main() -> None:
    args = parse_args(__doc__)
    grid = Grid.from_spacing(-150.0, 150.0, 0.1)
    h = Hamiltonian1D(grid, potentials.soft_coulomb(grid.x))
    ground = inverse_iteration(h, shift=-1.0)
    e1 = inverse_iteration(h, shift=-0.3).energy
    print(f"E0 = {ground.energy:.6f} a.u.  ({ground.iterations} iterations)")
    print(f"E1 = {e1:.6f} a.u.  -> omega_10 = {e1 - ground.energy:.4f} a.u.")

    e0, omega = 1e-2, 1.0
    period = 2 * np.pi / omega
    steps_per_cycle = 128
    dt = period / steps_per_cycle
    n_cycles = 6 if args.quick else 15
    n_ramp = 4

    sudden = run(
        h, ground, LaserField(e0, omega, constant_envelope()), dt, n_cycles * steps_per_cycle
    )
    ramped = run(
        h,
        ground,
        LaserField(e0, omega, sin2_ramp_envelope(n_ramp * period)),
        dt,
        (n_ramp + n_cycles) * steps_per_cycle,
    )

    # ------------------------------------------------------------- checks
    print("\nNumerical checks")
    print(f"  max |norm - 1| (sudden)        = {np.max(np.abs(sudden['norm'] - 1)):.2e}")
    print(f"  max |norm - 1| (sin^2 ramp)    = {np.max(np.abs(ramped['norm'] - 1)):.2e}")
    half = run(
        h,
        ground,
        LaserField(e0, omega, constant_envelope()),
        dt / 2,
        2 * n_cycles * steps_per_cycle,
        record_every=2,
    )
    print(
        f"  dt -> dt/2 changes the overlap by "
        f"{np.max(np.abs(half['overlap'] - sudden['overlap'])):.2e}"
    )
    free = run(h, ground, None, dt, n_cycles * steps_per_cycle)
    print(f"  field-free: max |1 - |<psi0|psi>|| = {np.max(np.abs(1 - free['overlap'])):.2e}")

    # ------------------------------------------------------------- figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=False)
    ax1.plot(sudden.times, sudden["overlap"], lw=1.2)
    ax1.set_title(r"Sudden turn-on: $E(t) = E_0 \sin(\omega t)$  ($E_0 = 0.01$, $\omega = 1$ a.u.)")
    ax1.set_ylabel(r"$|\langle\psi_0|\psi(t)\rangle|$")
    ax1.set_xlabel("t (a.u.)")

    ax2.plot(ramped.times, ramped["overlap"], lw=1.2, color="tab:red")
    ramp_label = r"$\sin^2$ turn-on (4 cycles)"
    ax2.axvspan(0, n_ramp * period, color="orange", alpha=0.12, label=ramp_label)
    for k in range(0, int(ramped.times[-1] / np.pi) + 1):
        ax2.axvline(k * np.pi / omega, color="gray", ls=":", lw=0.5)
    ax2.set_title(r"Adiabatic $\sin^2$ turn-on: smooth, $\pi/\omega$-periodic response")
    ax2.set_xlabel("t (a.u.)")
    ax2.set_ylabel(r"$|\langle\psi_0|\psi(t)\rangle|$")
    ax2.legend(loc="lower left")
    for ax in (ax1, ax2):
        ax.ticklabel_format(axis="y", useOffset=False)
    save(fig, args, "04_driven_atom.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
