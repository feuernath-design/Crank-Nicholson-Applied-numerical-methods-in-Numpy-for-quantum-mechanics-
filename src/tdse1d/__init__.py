"""tdse1d: Crank-Nicolson solver for the 1D time-dependent Schrodinger equation.

Atomic units throughout (hbar = m_e = e = 1).

Typical use::

    import numpy as np
    from tdse1d import Grid, Hamiltonian1D, CrankNicolson, inverse_iteration, potentials

    grid = Grid.from_spacing(-150.0, 150.0, 0.1)
    h0 = Hamiltonian1D(grid, potentials.soft_coulomb(grid.x))
    ground = inverse_iteration(h0, shift=-1.0)          # E0 = -0.6699 a.u.
    prop = CrankNicolson(h0, dt=0.05)
    result = prop.propagate(ground.state, n_steps=1000)
"""

from . import potentials
from .analysis import (
    cn_effective_energy,
    cn_energy_error_leading_order,
    loglog_slope,
    measure_energy_error,
)
from .eigen import EigenPair, inverse_iteration
from .fields import (
    LaserField,
    constant_envelope,
    sin2_ramp_envelope,
    trapezoid_envelope,
)
from .grid import Grid
from .hamiltonian import Hamiltonian1D
from .propagator import CrankNicolson, PropagationResult, absorbing_mask
from .states import gaussian_wavepacket

__version__ = "1.0.0"

__all__ = [
    "CrankNicolson",
    "EigenPair",
    "Grid",
    "Hamiltonian1D",
    "LaserField",
    "PropagationResult",
    "__version__",
    "absorbing_mask",
    "cn_effective_energy",
    "cn_energy_error_leading_order",
    "constant_envelope",
    "gaussian_wavepacket",
    "inverse_iteration",
    "loglog_slope",
    "measure_energy_error",
    "potentials",
    "sin2_ramp_envelope",
    "trapezoid_envelope",
]
