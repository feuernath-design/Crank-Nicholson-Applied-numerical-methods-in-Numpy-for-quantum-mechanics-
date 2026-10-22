"""Cross-check and timing: Fortran solver versus Python package.

Builds ``fortran/`` with make (gfortran), runs the soft-Coulomb example
(ground state + sin^2-ramped field, 19 optical cycles, 2432 steps) with both
implementations, and reports the maximum difference of |<psi0|psi(t)>|.

Timing: the comparison run records observables at every step. The solvers are
then timed separately *without* observables or output (Fortran with
record_every = n_steps, Python without observables), best of 3 runs, so that
both timings measure the propagation only.
"""

import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from _common import ROOT, parse_args, save

from tdse1d import (
    CrankNicolson,
    Grid,
    Hamiltonian1D,
    LaserField,
    inverse_iteration,
    potentials,
    sin2_ramp_envelope,
)


def main() -> None:
    args = parse_args(__doc__)
    if shutil.which("gfortran") is None:
        raise SystemExit("gfortran not found: install it to run this comparison")
    fortran_dir = ROOT / "fortran"
    subprocess.run(["make", "-C", str(fortran_dir)], check=True)

    omega, e0, n_ramp = 1.0, 1e-2, 4
    period = 2 * np.pi / omega
    dt = period / 128
    n_steps = (n_ramp + (4 if args.quick else 15)) * 128

    def run_fortran(tmp: Path, record_every: int) -> tuple[np.ndarray, float]:
        csv = tmp / "out.csv"
        nml = tmp / "config.nml"
        nml.write_text(
            "&config\n"
            "  potential = 'softcoulomb', initial = 'ground'\n"
            "  x_min = -150.0, x_max = 150.0, nx = 3001\n"
            f"  omega = {omega!r}, e0 = {e0!r}, n_ramp = {float(n_ramp)!r}\n"
            f"  dt = {dt!r}, n_steps = {n_steps}, record_every = {record_every}\n"
            f"  output = '{csv}'\n/\n"
        )
        out = subprocess.run(
            [str(fortran_dir / "build" / "cn_propagate"), str(nml)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        seconds = float(out.split("propagation time=")[1].split()[0])
        return np.loadtxt(csv, delimiter=",", skiprows=1), seconds

    with tempfile.TemporaryDirectory() as tmp:
        fortran, _ = run_fortran(Path(tmp), record_every=1)
        t_fortran = min(run_fortran(Path(tmp), record_every=n_steps)[1] for _ in range(3))

    grid = Grid.from_spacing(-150.0, 150.0, 0.1)
    h = Hamiltonian1D(grid, potentials.soft_coulomb(grid.x))
    ground = inverse_iteration(h, shift=-1.0)
    field = LaserField(e0, omega, sin2_ramp_envelope(n_ramp * period))
    prop = CrankNicolson(h, dt, field=field)
    res = prop.propagate(
        ground.state,
        n_steps,
        observables={"overlap": lambda psi: abs(grid.inner(ground.state, psi))},
    )
    t_python = np.inf
    for _ in range(3):
        start = time.perf_counter()
        prop.propagate(ground.state, n_steps)
        t_python = min(t_python, time.perf_counter() - start)

    diff = np.max(np.abs(fortran[:, 3] - res["overlap"]))
    print(f"max |overlap_Fortran - overlap_Python| = {diff:.2e}")
    print(
        f"propagation only ({n_steps} steps, N = {grid.n}, best of 3): "
        f"Fortran {t_fortran:.3f} s, Python/SciPy {t_python:.3f} s "
        f"(ratio {t_python / t_fortran:.2f})"
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(res.times, res["overlap"], label="Python (NumPy/SciPy)")
    ax1.plot(fortran[:, 0], fortran[:, 3], "--", label="Fortran 2008")
    ax1.set_ylabel(r"$|\langle\psi_0|\psi(t)\rangle|$")
    ax1.ticklabel_format(axis="y", useOffset=False)
    ax1.legend()
    ax1.set_title("Same physics, two implementations")
    ax2.semilogy(res.times[1:], np.abs(fortran[1:, 3] - res["overlap"][1:]) + 1e-17)
    ax2.set_xlabel("t (a.u.)")
    ax2.set_ylabel("|difference|")
    save(fig, args, "06_fortran_vs_python.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
