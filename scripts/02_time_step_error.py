"""Temporal error of Crank-Nicolson: phase (energy) error of eigenstates.

An eigenstate phi_n of the *discrete* Hamiltonian is propagated with
Crank-Nicolson. Exactly, <phi_n|psi(T)> = exp(-i E_n T); numerically the
state evolves with E_eff = (2/dt) arctan(E_n dt/2). The measured energy error
|E_eff - E_n| is compared with this closed form and with its leading term
E_n^3 dt^2 / 12:

* left:  error versus dt for the ground state (expected slope 2),
* right: error versus E_n at fixed dt (expected ~ E_n^3).

Because the reference is the discrete eigenstate, the spatial error plays no
role: only the time discretisation is measured.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, save

from tdse1d import (
    CrankNicolson,
    Grid,
    Hamiltonian1D,
    cn_effective_energy,
    cn_energy_error_leading_order,
    inverse_iteration,
    loglog_slope,
    measure_energy_error,
    potentials,
)


def main() -> None:
    args = parse_args(__doc__)
    grid = Grid.symmetric(20.0, 400 if args.quick else 800)
    h = Hamiltonian1D(grid, potentials.harmonic(grid.x))
    total_time = 10.0

    # Eigenstates from inverse iteration with shifts n + 0.4 (n = 0..5)
    levels = [inverse_iteration(h, shift=n + 0.4) for n in range(6)]
    energies = np.array([lv.energy for lv in levels])

    # ---- error versus dt (ground state) ---------------------------------
    dts = np.array([0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001])
    if args.quick:
        dts = dts[:5]
    e0 = energies[0]
    err_dt = np.array(
        [measure_energy_error(CrankNicolson(h, dt), levels[0].state, e0, total_time) for dt in dts]
    )
    exact_dt = np.abs(cn_effective_energy(e0, dts) - e0)
    print(f"{'dt':>8} {'measured':>12} {'(2/dt)atan(E dt/2)':>20} {'E^3 dt^2/12':>12}")
    for dt, m, ex, lo in zip(
        dts, err_dt, exact_dt, cn_energy_error_leading_order(e0, dts), strict=True
    ):
        print(f"{dt:8.3f} {m:12.4e} {ex:20.4e} {lo:12.4e}")
    slope = loglog_slope(dts, err_dt)
    print(f"measured order in dt: {slope:.4f} (expected 2)\n")

    # ---- error versus energy (fixed dt) ---------------------------------
    dt = 0.05
    prop = CrankNicolson(h, dt)
    err_e = np.array([measure_energy_error(prop, lv.state, lv.energy, total_time) for lv in levels])
    print(f"{'n':>3} {'E_n':>10} {'measured':>12} {'predicted':>12}")
    for n, (e, m) in enumerate(zip(energies, err_e, strict=True)):
        print(f"{n:3d} {e:10.6f} {m:12.4e} {abs(cn_effective_energy(e, dt) - e):12.4e}")

    # ---- figure -----------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.loglog(dts, err_dt, "o", label="measured (propagation)")
    cayley = r"$\frac{2}{\Delta t}\arctan\frac{E\Delta t}{2} - E$"
    ax1.loglog(dts, exact_dt, "-", lw=1.2, label=cayley)
    ax1.loglog(
        dts, cn_energy_error_leading_order(e0, dts), "k--", lw=1, label=r"$E^3\Delta t^2/12$"
    )
    ax1.set_xlabel(r"$\Delta t$ (a.u.)")
    ax1.set_ylabel(r"$|\bar E_0 - E_0|$ (a.u.)")
    ax1.set_title(f"Ground state: error vs time step (slope {slope:.3f})")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    e_fine = np.linspace(energies[0], energies[-1], 200)
    ax2.plot(energies, err_e, "o", label="measured")
    ax2.plot(e_fine, np.abs(cn_effective_energy(e_fine, dt) - e_fine), "-", lw=1.2, label="exact")
    ax2.set_xlabel(r"$E_n$ (a.u.)")
    ax2.set_ylabel(r"$|\bar E_n - E_n|$ (a.u.)")
    ax2.set_title(rf"Error vs energy ($\Delta t = {dt}$): $\propto E_n^3$")
    ax2.legend()
    save(fig, args, "02_time_step_error.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
