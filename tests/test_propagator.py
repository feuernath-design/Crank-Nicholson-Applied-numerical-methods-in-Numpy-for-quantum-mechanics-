import numpy as np
import pytest

from tdse1d import (
    CrankNicolson,
    Grid,
    Hamiltonian1D,
    LaserField,
    absorbing_mask,
    cn_effective_energy,
    gaussian_wavepacket,
    inverse_iteration,
    potentials,
    sin2_ramp_envelope,
)
from tdse1d.reference import dense_cn_matrices, dense_cn_step


@pytest.fixture(scope="module")
def ho():
    g = Grid.symmetric(20.0, 401)
    return Hamiltonian1D(g, potentials.harmonic(g.x))


@pytest.fixture(scope="module")
def atom():
    g = Grid.from_spacing(-100.0, 100.0, 0.2)
    h = Hamiltonian1D(g, potentials.soft_coulomb(g.x))
    return h, inverse_iteration(h, shift=-1.0)


def test_banded_matches_dense_reference(ho):
    """O(N) banded solver == pedagogical dense np.linalg.solve version."""
    dt = 0.05
    psi = gaussian_wavepacket(ho.grid, x0=1.5, k0=0.3)
    u_plus, u_minus = dense_cn_matrices(ho, dt)
    prop = CrankNicolson(ho, dt)
    a, b = psi.copy(), psi.copy()
    for _ in range(50):
        a = prop.step(a)
        b = dense_cn_step(b, u_plus, u_minus)
    np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)


@pytest.mark.parametrize("dt", [0.001, 0.05, 1.0, 10.0])
def test_norm_conserved_for_any_dt(ho, dt):
    """Cayley form => unitary, even for huge time steps."""
    psi0 = gaussian_wavepacket(ho.grid, x0=2.0, k0=1.0)
    res = CrankNicolson(ho, dt).propagate(psi0, 200, observables={"norm": ho.grid.norm2})
    assert np.max(np.abs(res["norm"] - 1.0)) < 1e-12


def test_norm_conserved_with_time_dependent_field(atom):
    h, ground = atom
    field = LaserField(0.05, 1.0, sin2_ramp_envelope(2 * np.pi))
    res = CrankNicolson(h, 2 * np.pi / 128, field=field).propagate(
        ground.state, 640, observables={"norm": h.grid.norm2}
    )
    assert np.max(np.abs(res["norm"] - 1.0)) < 1e-12


def test_eigenstate_phase_is_exact_cayley_phase(ho):
    """<phi|psi(T)> = exp(-i E_eff T) with E_eff = 2/dt arctan(E dt/2), to round-off."""
    energies, states = ho.eigh(4)
    dt, n_steps = 0.1, 300
    for e, phi in zip(energies, states.T, strict=True):
        psi = CrankNicolson(ho, dt).propagate(phi, n_steps).psi
        expected = np.exp(-1j * cn_effective_energy(e, dt) * n_steps * dt)
        assert ho.grid.inner(phi, psi) == pytest.approx(expected, abs=1e-11)


def test_time_reversibility(atom):
    """U(-dt) U(dt) = 1, also with a field (mid-point rule is symmetric)."""
    h, ground = atom
    field = LaserField(0.05, 1.0)
    dt, n = 0.05, 400
    psi0 = ground.state.astype(complex)
    forward = CrankNicolson(h, dt, field=field).propagate(psi0, n)
    backward = CrankNicolson(h, -dt, field=field).propagate(forward.psi, n, t0=forward.t)
    np.testing.assert_allclose(backward.psi, psi0, rtol=0, atol=1e-11)


def test_coherent_state_follows_classical_trajectory():
    g = Grid.symmetric(20.0, 1601)
    h = Hamiltonian1D(g, potentials.harmonic(g.x))
    x0 = 2.0
    res = CrankNicolson(h, 0.005).propagate(
        gaussian_wavepacket(g, x0=x0), 2000, observables={"x": g.expectation_x}
    )
    np.testing.assert_allclose(res["x"], x0 * np.cos(res.times), atol=5e-3)


def test_second_order_convergence_with_field(atom):
    """Global error of the driven propagation is O(dt^2) (mid-point rule)."""
    h, ground = atom
    field = LaserField(0.05, 1.0, sin2_ramp_envelope(2 * np.pi))
    t_final = 4 * np.pi

    def run(n_steps):
        return CrankNicolson(h, t_final / n_steps, field=field).propagate(ground.state, n_steps).psi

    ref = run(4096)
    errs = [np.sqrt(h.grid.norm2(run(n) - ref)) for n in (64, 128, 256)]
    order = np.log2(np.array(errs[:-1]) / np.array(errs[1:]))
    np.testing.assert_allclose(order, 2.0, atol=0.05)


def test_records_and_times(ho):
    psi0 = gaussian_wavepacket(ho.grid)
    res = CrankNicolson(ho, 0.1).propagate(
        psi0, 10, t0=1.0, record_every=4, observables={"norm": ho.grid.norm2}
    )
    np.testing.assert_allclose(res.times, [1.0, 1.4, 1.8, 2.0])
    assert res.t == pytest.approx(2.0)
    assert res["norm"].dtype == np.float64


def test_absorbing_mask_removes_outgoing_flux():
    g = Grid.symmetric(50.0, 1001)
    h = Hamiltonian1D(g, potentials.free(g.x))
    mask = absorbing_mask(g, width=10.0)
    assert mask.max() == 1.0
    assert mask[0] < 0.01 and mask[-1] < 0.01  # cos(pi/2)**(1/8) ~ 0.009 in floating point
    assert np.all(mask[np.abs(g.x) <= 40.0] == 1.0)
    psi0 = gaussian_wavepacket(g, x0=0.0, sigma=2.0, k0=2.0)  # moving right
    res = CrankNicolson(h, 0.05, mask=mask).propagate(psi0, 1000, observables={"n": g.norm2})
    assert res["n"][-1] < 1e-3  # packet absorbed instead of reflected
    assert np.all(np.diff(res["n"]) <= 1e-15)  # norm can only decrease


@pytest.mark.parametrize("bad", [0.0, np.inf, np.nan])
def test_invalid_dt(ho, bad):
    with pytest.raises(ValueError):
        CrankNicolson(ho, bad)
