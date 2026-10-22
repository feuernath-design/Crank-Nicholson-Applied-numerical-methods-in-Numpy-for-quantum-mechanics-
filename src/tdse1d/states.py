"""Initial wavefunctions."""

from __future__ import annotations

import numpy as np

from .grid import Grid

__all__ = ["gaussian_wavepacket"]


def gaussian_wavepacket(
    grid: Grid, x0: float = 0.0, sigma: float = 1.0, k0: float = 0.0
) -> np.ndarray:
    """Normalised Gaussian ``exp(-(x - x0)^2 / (2 sigma^2) + i k0 x)``.

    With ``sigma = 1/sqrt(omega)`` this is a coherent state of the harmonic
    oscillator of frequency ``omega``: it oscillates rigidly with
    ``<x>(t) = x0 cos(omega t)`` without spreading.
    """
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    x = grid.x
    psi = np.exp(-((x - x0) ** 2) / (2.0 * sigma**2) + 1j * k0 * x)
    return grid.normalize(psi)
