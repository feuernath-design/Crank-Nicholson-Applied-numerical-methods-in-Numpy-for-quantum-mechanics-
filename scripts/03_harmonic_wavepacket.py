"""Coherent-state dynamics in a harmonic trap.

A Gaussian of width sigma = 1 displaced to x0 = 2 is the coherent state of the
omega = 1 oscillator: it oscillates rigidly, <x>(t) = x0 cos t (Ehrenfest),
without spreading. The script shows the space-time map of |psi(x, t)|^2, the
comparison with the classical trajectory and the norm conservation.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, save

from tdse1d import CrankNicolson, Grid, Hamiltonian1D, gaussian_wavepacket, potentials


def main() -> None:
    args = parse_args(__doc__)
    grid = Grid.symmetric(10.0, 401 if args.quick else 1001)
    h = Hamiltonian1D(grid, potentials.harmonic(grid.x, omega=1.0))
    x0, dt, t_final = 2.0, 0.01, (2 * np.pi if args.quick else 4 * np.pi)
    n_steps = round(t_final / dt)
    record_every = max(1, n_steps // 400)

    prop = CrankNicolson(h, dt)
    psi = gaussian_wavepacket(grid, x0=x0, sigma=1.0)
    times, mean_x, norms, densities = [], [], [], []
    for k in range(n_steps + 1):
        if k % record_every == 0 or k == n_steps:
            times.append(k * dt)
            mean_x.append(grid.expectation_x(psi))
            norms.append(grid.norm2(psi))
            densities.append(np.abs(psi) ** 2)
        if k < n_steps:
            psi = prop.step(psi, k * dt)
    times, mean_x, norms = np.array(times), np.array(mean_x), np.array(norms)
    densities = np.array(densities)

    classical = x0 * np.cos(times)
    print(f"max |<x> - x0 cos t|  = {np.max(np.abs(mean_x - classical)):.3e}")
    print(f"max |norm - 1|        = {np.max(np.abs(norms - 1)):.3e}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    ax = axes[0]
    im = ax.imshow(
        densities,
        aspect="auto",
        origin="lower",
        extent=[grid.x_min, grid.x_max, times[0], times[-1]],
        cmap="inferno",
    )
    ax.plot(mean_x, times, color="cyan", lw=1, label=r"$\langle x\rangle(t)$")
    ax.set_xlim(-5, 5)
    ax.set_xlabel("x (a.u.)")
    ax.set_ylabel("t (a.u.)")
    ax.set_title(r"$|\psi(x,t)|^2$")
    ax.legend(loc="upper right")
    ax.grid(False)
    fig.colorbar(im, ax=ax)

    axes[1].plot(times, mean_x, label="Crank-Nicolson")
    axes[1].plot(times, classical, "k--", lw=1, label=r"$x_0\cos t$")
    axes[1].set_xlabel("t (a.u.)")
    axes[1].set_ylabel(r"$\langle x\rangle$ (a.u.)")
    axes[1].set_title("Ehrenfest: classical trajectory")
    axes[1].legend()

    axes[2].plot(times, norms - 1.0)
    axes[2].set_xlabel("t (a.u.)")
    axes[2].set_ylabel(r"$\langle\psi|\psi\rangle - 1$")
    axes[2].set_title("Norm conservation (unitarity)")
    axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    save(fig, args, "03_harmonic_wavepacket.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
