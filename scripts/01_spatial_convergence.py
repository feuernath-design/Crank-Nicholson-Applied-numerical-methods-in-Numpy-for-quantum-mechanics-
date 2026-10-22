"""Spatial convergence: harmonic-oscillator ground state versus grid spacing.

The ground state of H = -1/2 d^2/dx^2 + x^2/2 on [-20, 20] is computed by
shifted inverse iteration for N = 100 ... 3200 grid points and compared with
the exact value E0 = 1/2. The 3-point Laplacian is second-order accurate, so
the error should scale as dx^2 (slope 2 in log-log).
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, save

from tdse1d import Grid, Hamiltonian1D, inverse_iteration, loglog_slope, potentials


def main() -> None:
    args = parse_args(__doc__)
    half_width = 20.0
    sizes = [100, 200, 400, 800] if args.quick else [100, 200, 400, 800, 1600, 3200]

    dxs, errors = [], []
    print(f"{'N':>6} {'dx':>10} {'E0 (numerical)':>18} {'|E0 - 1/2|':>12} {'iter':>5}")
    for n in sizes:
        grid = Grid.symmetric(half_width, n)
        h = Hamiltonian1D(grid, potentials.harmonic(grid.x))
        res = inverse_iteration(h, shift=0.4)
        err = abs(res.energy - 0.5)
        dxs.append(grid.dx)
        errors.append(err)
        print(f"{n:6d} {grid.dx:10.5f} {res.energy:18.12f} {err:12.3e} {res.iterations:5d}")

    dxs, errors = np.array(dxs), np.array(errors)
    slope = loglog_slope(dxs, errors)
    print(f"\nmeasured order: {slope:.4f}  (expected 2)")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.loglog(dxs, errors, "o-", label="inverse iteration")
    ax.loglog(dxs, errors[-1] * (dxs / dxs[-1]) ** 2, "k--", lw=1, label=r"$\propto \Delta x^2$")
    ax.set_xlabel(r"grid spacing $\Delta x$ (a.u.)")
    ax.set_ylabel(r"$|E_0^{\rm num} - 1/2|$ (a.u.)")
    ax.set_title(f"Spatial convergence (harmonic oscillator), slope = {slope:.3f}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save(fig, args, "01_spatial_convergence.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
