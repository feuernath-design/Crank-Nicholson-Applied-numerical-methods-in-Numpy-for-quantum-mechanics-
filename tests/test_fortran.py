"""Cross-validation of the Fortran solver against the Python package.

Skipped automatically when ``gfortran`` is not installed.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tdse1d import (
    CrankNicolson,
    Grid,
    Hamiltonian1D,
    LaserField,
    gaussian_wavepacket,
    inverse_iteration,
    potentials,
    sin2_ramp_envelope,
)

FORTRAN_DIR = Path(__file__).resolve().parents[1] / "fortran"

pytestmark = pytest.mark.skipif(shutil.which("gfortran") is None, reason="gfortran not found")


@pytest.fixture(scope="module")
def executable(tmp_path_factory):
    out = tmp_path_factory.mktemp("fortran") / "cn_propagate"
    sources = [FORTRAN_DIR / "src" / "tdse_cn.f90", FORTRAN_DIR / "src" / "cn_propagate.f90"]
    flags = ["-std=f2008", "-O0", "-fcheck=all", "-Wall", "-Werror", "-ffpe-trap=invalid,zero"]
    subprocess.run(
        ["gfortran", *flags, "-J", str(out.parent), *map(str, sources), "-o", str(out)],
        check=True,
    )
    return out


def run_fortran(executable, tmp_path, **params):
    lines = []
    for key, value in params.items():
        lines.append(f"  {key} = '{value}'" if isinstance(value, str) else f"  {key} = {value!r}")
    csv = tmp_path / "out.csv"
    nml = tmp_path / "config.nml"
    nml.write_text("&config\n" + "\n".join(lines) + f"\n  output = '{csv}'\n/\n")
    proc = subprocess.run([str(executable), str(nml)], check=True, capture_output=True, text=True)
    return np.loadtxt(csv, delimiter=",", skiprows=1), proc.stdout


def test_driven_soft_coulomb_matches_python(executable, tmp_path):
    omega, e0, n_ramp, n_steps = 1.0, 0.02, 2.0, 640
    period = 2 * np.pi / omega
    dt = period / 128
    data, stdout = run_fortran(
        executable,
        tmp_path,
        potential="softcoulomb",
        initial="ground",
        x_min=-100.0,
        x_max=100.0,
        nx=1001,
        omega=omega,
        e0=e0,
        n_ramp=n_ramp,
        dt=dt,
        n_steps=n_steps,
        record_every=1,
    )
    grid = Grid(-100.0, 100.0, 1001)
    h = Hamiltonian1D(grid, potentials.soft_coulomb(grid.x))
    ground = inverse_iteration(h, shift=-1.0)
    e_fortran = float(stdout.split("E_ground=")[1].split()[0])
    assert e_fortran == pytest.approx(ground.energy, abs=1e-12)

    field = LaserField(e0, omega, sin2_ramp_envelope(n_ramp * period))
    res = CrankNicolson(h, dt, field=field).propagate(
        ground.state,
        n_steps,
        observables={
            "norm": grid.norm2,
            "x": grid.expectation_x,
            "overlap": lambda psi: abs(grid.inner(ground.state, psi)),
        },
    )
    np.testing.assert_allclose(data[:, 0], res.times, atol=1e-12)
    np.testing.assert_allclose(data[:, 1], res["norm"], atol=1e-12)
    np.testing.assert_allclose(data[:, 2], res["x"], atol=1e-10)
    np.testing.assert_allclose(data[:, 3], res["overlap"], atol=1e-10)


def test_harmonic_wavepacket_matches_python(executable, tmp_path):
    data, _ = run_fortran(
        executable,
        tmp_path,
        potential="harmonic",
        initial="gaussian",
        x_min=-20.0,
        x_max=20.0,
        nx=801,
        x0=2.0,
        sigma=1.0,
        e0=0.0,
        dt=0.01,
        n_steps=500,
        record_every=10,
    )
    grid = Grid(-20.0, 20.0, 801)
    h = Hamiltonian1D(grid, potentials.harmonic(grid.x))
    res = CrankNicolson(h, 0.01).propagate(
        gaussian_wavepacket(grid, x0=2.0),
        500,
        record_every=10,
        observables={"x": grid.expectation_x},
    )
    np.testing.assert_allclose(data[:, 2], res["x"], atol=1e-10)
    assert np.max(np.abs(data[:, 1] - 1.0)) < 1e-12
