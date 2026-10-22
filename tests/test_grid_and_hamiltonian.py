import numpy as np
import pytest

from tdse1d import Grid, Hamiltonian1D, gaussian_wavepacket, potentials


class TestGrid:
    def test_from_spacing(self):
        g = Grid.from_spacing(-150.0, 150.0, 0.1)
        assert g.n == 3001
        assert g.dx == pytest.approx(0.1, rel=1e-12)
        assert g.x[0] == -150.0 and g.x[-1] == 150.0

    def test_from_spacing_rejects_non_integer_ratio(self):
        with pytest.raises(ValueError):
            Grid.from_spacing(0.0, 1.0, 0.3)

    @pytest.mark.parametrize("args", [(0.0, 1.0, 2), (1.0, 0.0, 10)])
    def test_invalid_grid(self, args):
        with pytest.raises(ValueError):
            Grid(*args)

    def test_normalisation_and_inner_product(self):
        g = Grid.symmetric(20.0, 801)
        psi = g.normalize(np.exp(-(g.x**2)))
        assert g.norm2(psi) == pytest.approx(1.0, abs=1e-14)
        assert g.inner(psi, psi) == pytest.approx(1.0, abs=1e-14)
        assert g.inner(psi, 1j * psi) == pytest.approx(1j, abs=1e-14)  # antilinear in bra

    def test_gaussian_wavepacket_mean_position(self):
        g = Grid.symmetric(20.0, 801)
        psi = gaussian_wavepacket(g, x0=2.0, sigma=1.0, k0=0.5)
        assert g.norm2(psi) == pytest.approx(1.0, abs=1e-14)
        assert g.expectation_x(psi) == pytest.approx(2.0, abs=1e-10)


class TestHamiltonian:
    @pytest.fixture
    def h(self):
        g = Grid.symmetric(10.0, 201)
        return Hamiltonian1D(g, potentials.harmonic(g.x))

    def test_matrix_elements(self, h):
        dx = h.grid.dx
        assert h.offdiag == pytest.approx(-0.5 / dx**2)
        np.testing.assert_allclose(h.diag, 1 / dx**2 + 0.5 * h.grid.x**2)

    def test_dense_is_symmetric_tridiagonal(self, h):
        dense = h.to_dense()
        np.testing.assert_array_equal(dense, dense.T)
        assert np.count_nonzero(np.triu(dense, 2)) == 0

    def test_apply_matches_dense(self, h):
        rng = np.random.default_rng(1)
        psi = rng.normal(size=h.n) + 1j * rng.normal(size=h.n)
        np.testing.assert_allclose(h.apply(psi), h.to_dense() @ psi, rtol=0, atol=1e-10)

    def test_shape_validation(self):
        g = Grid.symmetric(1.0, 10)
        with pytest.raises(ValueError):
            Hamiltonian1D(g, np.zeros(9))

    def test_harmonic_spectrum_second_order_in_dx(self):
        """E_n -> n + 1/2 and the discretisation error decreases as dx^2."""
        errors = []
        for n_points in (401, 801, 1601):
            g = Grid.symmetric(20.0, n_points)
            energies, _ = Hamiltonian1D(g, potentials.harmonic(g.x)).eigh(6)
            err = np.abs(energies - (np.arange(6) + 0.5))
            errors.append(err)
            assert np.all(err < 0.02)
        ratios = np.array(errors[:-1]) / np.array(errors[1:])
        # halving dx divides the error by 4
        np.testing.assert_allclose(ratios, 4.0, rtol=0.01)

    def test_eigh_states_are_grid_normalised(self, h):
        _, states = h.eigh(3)
        for j in range(3):
            assert h.grid.norm2(states[:, j]) == pytest.approx(1.0, abs=1e-12)


def test_mass_dependence_of_spectrum():
    """V = m w^2 x^2 / 2 gives E_n = w (n + 1/2) for any mass m."""
    g = Grid.symmetric(15.0, 3001)
    for mass in (0.5, 2.0):
        h = Hamiltonian1D(g, potentials.harmonic(g.x, omega=1.3, mass=mass), mass=mass)
        energies, _ = h.eigh(3)
        np.testing.assert_allclose(energies, 1.3 * (np.arange(3) + 0.5), atol=2e-4)
