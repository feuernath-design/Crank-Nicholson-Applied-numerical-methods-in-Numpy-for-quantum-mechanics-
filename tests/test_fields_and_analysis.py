import numpy as np
import pytest

from tdse1d import (
    CrankNicolson,
    Grid,
    Hamiltonian1D,
    LaserField,
    cn_effective_energy,
    cn_energy_error_leading_order,
    constant_envelope,
    loglog_slope,
    measure_energy_error,
    potentials,
    sin2_ramp_envelope,
    trapezoid_envelope,
)


class TestEnvelopes:
    def test_constant(self):
        np.testing.assert_array_equal(constant_envelope()(np.array([0.0, 5.0])), [1.0, 1.0])

    def test_sin2_ramp(self):
        f = sin2_ramp_envelope(4.0)
        np.testing.assert_allclose(f(np.array([0.0, 2.0, 4.0, 10.0])), [0.0, 0.5, 1.0, 1.0])
        np.testing.assert_array_equal(sin2_ramp_envelope(0.0)(np.array([0.0, 1.0])), [1.0, 1.0])

    def test_trapezoid(self):
        f = trapezoid_envelope(2.0, 3.0, 4.0)
        t = np.array([-1.0, 0.0, 1.0, 2.0, 4.0, 5.0, 7.0, 9.0, 12.0])
        np.testing.assert_allclose(f(t), [0, 0, 0.5, 1, 1, 1, 0.5, 0, 0])

    def test_envelopes_are_continuous(self):
        t = np.linspace(-1, 12, 130001)
        for f in (sin2_ramp_envelope(3.0), trapezoid_envelope(2.0, 3.0, 4.0)):
            assert np.max(np.abs(np.diff(f(t)))) < 1e-3

    def test_laser_field(self):
        field = LaserField(0.1, 2.0)
        assert field.period == pytest.approx(np.pi)
        assert field(np.pi / 4) == pytest.approx(0.1)
        assert isinstance(field(0.3), float)
        assert field(np.array([0.0, np.pi / 4])).shape == (2,)


class TestAnalysis:
    def test_effective_energy_expansion(self):
        e, dt = 1.3, 1e-2
        exact = cn_effective_energy(e, dt)
        assert abs(exact - e) == pytest.approx(cn_energy_error_leading_order(e, dt), rel=1e-3)

    def test_loglog_slope(self):
        x = np.logspace(-3, 0, 10)
        assert loglog_slope(x, 3 * x**2) == pytest.approx(2.0)
        with pytest.raises(ValueError):
            loglog_slope([1.0, -1.0], [1.0, 1.0])

    def test_measured_error_equals_cayley_prediction(self):
        g = Grid.symmetric(20.0, 400)
        h = Hamiltonian1D(g, potentials.harmonic(g.x))
        energies, states = h.eigh(6)
        for dt in (0.05, 0.2, 0.5):
            prop = CrankNicolson(h, dt)
            for e, phi in zip(energies[[0, 5]], states.T[[0, 5]], strict=True):
                measured = measure_energy_error(prop, phi, e, total_time=10.0)
                predicted = abs(cn_effective_energy(e, dt) - e)
                assert measured == pytest.approx(predicted, rel=1e-6, abs=1e-13)

    def test_measure_rejects_driven_propagator(self):
        g = Grid.symmetric(5.0, 50)
        h = Hamiltonian1D(g, potentials.harmonic(g.x))
        with pytest.raises(ValueError):
            measure_energy_error(CrankNicolson(h, 0.1, field=LaserField(0.1, 1.0)), g.x, 0.5, 1.0)
