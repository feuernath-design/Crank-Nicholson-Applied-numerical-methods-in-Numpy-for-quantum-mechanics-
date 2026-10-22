"""Model potentials (atomic units: hbar = m_e = e = 1)."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["free", "harmonic", "soft_coulomb"]


def harmonic(x: ArrayLike, omega: float = 1.0, mass: float = 1.0) -> NDArray[np.float64]:
    """Harmonic oscillator ``V(x) = m omega^2 x^2 / 2``.

    Exact spectrum: ``E_n = omega (n + 1/2)``.
    """
    x = np.asarray(x, dtype=float)
    return 0.5 * mass * omega**2 * x**2


def soft_coulomb(x: ArrayLike, softening: float = 1.0, charge: float = 1.0) -> NDArray[np.float64]:
    """Soft-core Coulomb potential ``V(x) = -Z / sqrt(x^2 + a^2)``.

    The standard 1D model atom of strong-field physics (Javanainen, Eberly &
    Su, Phys. Rev. A 38, 3430 (1988)). With ``a = 1`` and ``Z = 1`` the ground
    state energy is about -0.6698 a.u. and the first excited state about
    -0.2749 a.u.
    """
    if softening <= 0:
        raise ValueError("softening parameter must be positive")
    x = np.asarray(x, dtype=float)
    return -charge / np.sqrt(x**2 + softening**2)


def free(x: ArrayLike) -> NDArray[np.float64]:
    """Free particle, ``V(x) = 0``."""
    return np.zeros_like(np.asarray(x, dtype=float))
