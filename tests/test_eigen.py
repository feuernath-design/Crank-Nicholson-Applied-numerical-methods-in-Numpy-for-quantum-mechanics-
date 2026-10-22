import numpy as np
import pytest

from tdse1d import Grid, Hamiltonian1D, inverse_iteration, potentials


@pytest.fixture(scope="module")
def harmonic():
    g = Grid.symmetric(20.0, 800)
    return Hamiltonian1D(g, potentials.harmonic(g.x))


@pytest.fixture(scope="module")
def soft_coulomb():
    g = Grid.from_spacing(-150.0, 150.0, 0.1)
    return Hamiltonian1D(g, potentials.soft_coulomb(g.x))


@pytest.mark.parametrize("n", range(6))
def test_inverse_iteration_matches_lapack(harmonic, n):
    """Shifts n + 0.4 select the n-th level (setup of the internship study)."""
    ref_e, ref_v = harmonic.eigh(6)
    res = inverse_iteration(harmonic, shift=n + 0.4)
    assert res.converged
    assert res.energy == pytest.approx(ref_e[n], abs=1e-11)
    # same state up to a global sign (the sign of an odd state is arbitrary)
    overlap = harmonic.grid.inner(ref_v[:, n], res.state).real
    assert abs(overlap) == pytest.approx(1.0, abs=1e-12)
    np.testing.assert_allclose(res.state, np.sign(overlap) * ref_v[:, n], atol=1e-7)
    assert harmonic.grid.norm2(res.state) == pytest.approx(1.0, abs=1e-12)


def test_soft_coulomb_ground_state(soft_coulomb):
    res = inverse_iteration(soft_coulomb, shift=-1.0)
    assert res.converged
    assert res.energy == pytest.approx(-0.66986, abs=1e-5)
    assert res.energy == pytest.approx(soft_coulomb.eigh(1)[0][0], abs=1e-12)
    # ground state is nodeless and even
    assert np.all(res.state > -1e-12)
    np.testing.assert_allclose(res.state, res.state[::-1], atol=1e-8)


def test_fixed_shift_converges_to_nearest_level(harmonic):
    """Without the shift update the iteration is plain shifted inverse iteration."""
    res = inverse_iteration(harmonic, shift=2.3, update_shift=False, max_iter=2000)
    assert res.converged
    assert res.energy == pytest.approx(2.5, abs=2e-3)


def test_shift_update_accelerates(soft_coulomb):
    slow = inverse_iteration(soft_coulomb, shift=-1.0, update_shift=False, max_iter=2000)
    fast = inverse_iteration(soft_coulomb, shift=-1.0)
    assert fast.converged and slow.converged
    assert fast.iterations < slow.iterations


def test_non_convergence_warns(harmonic):
    with pytest.warns(RuntimeWarning):
        res = inverse_iteration(harmonic, shift=0.4, max_iter=1, update_shift=False)
    assert not res.converged
