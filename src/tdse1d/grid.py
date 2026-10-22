"""Uniform 1D spatial grid with grid-weighted inner products.

All wavefunctions in this package are stored as complex vectors sampled on a
uniform grid ``x_j = x_min + j * dx`` and normalised such that

    sum_j |psi_j|^2 * dx = 1,

i.e. the discrete analogue of  integral |psi(x)|^2 dx = 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Grid"]


@dataclass(frozen=True)
class Grid:
    """Uniform grid on ``[x_min, x_max]`` with ``n`` points (both ends included).

    The wavefunction is implicitly zero just outside the grid (Dirichlet
    boundary conditions), which is what the 3-point finite-difference
    Laplacian in :mod:`tdse1d.hamiltonian` assumes.
    """

    x_min: float
    x_max: float
    n: int

    def __post_init__(self) -> None:
        if self.n < 3:
            raise ValueError(f"a grid needs at least 3 points, got n={self.n}")
        if not self.x_max > self.x_min:
            raise ValueError("x_max must be strictly larger than x_min")

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_spacing(cls, x_min: float, x_max: float, dx: float) -> Grid:
        """Build a grid with spacing ``dx`` (``x_max - x_min`` must be a multiple of ``dx``)."""
        if dx <= 0:
            raise ValueError("dx must be positive")
        n_intervals = (x_max - x_min) / dx
        n_round = round(n_intervals)
        if abs(n_intervals - n_round) > 1e-9 * max(1.0, n_intervals):
            raise ValueError(f"(x_max - x_min) / dx = {n_intervals} is not an integer")
        return cls(x_min, x_max, int(n_round) + 1)

    @classmethod
    def symmetric(cls, half_width: float, n: int) -> Grid:
        """Grid on ``[-half_width, +half_width]`` with ``n`` points."""
        return cls(-half_width, half_width, n)

    # ------------------------------------------------------------------ #
    # Geometry
    # ------------------------------------------------------------------ #
    @property
    def x(self) -> NDArray[np.float64]:
        return np.linspace(self.x_min, self.x_max, self.n)

    @property
    def dx(self) -> float:
        return (self.x_max - self.x_min) / (self.n - 1)

    # ------------------------------------------------------------------ #
    # Grid-weighted linear algebra
    # ------------------------------------------------------------------ #
    def inner(self, bra: ArrayLike, ket: ArrayLike) -> complex:
        """``<bra|ket> = sum_j conj(bra_j) ket_j dx``."""
        return complex(np.vdot(bra, ket) * self.dx)

    def norm2(self, psi: ArrayLike) -> float:
        """``<psi|psi>``."""
        psi = np.asarray(psi)
        return float(np.real(np.vdot(psi, psi)) * self.dx)

    def normalize(self, psi: ArrayLike) -> NDArray[np.complex128]:
        """Return ``psi / sqrt(<psi|psi>)`` as a complex array."""
        psi = np.asarray(psi, dtype=np.complex128)
        nrm2 = self.norm2(psi)
        if nrm2 == 0.0:
            raise ValueError("cannot normalise the zero vector")
        return psi / np.sqrt(nrm2)

    def expectation_x(self, psi: ArrayLike) -> float:
        """``<psi|x|psi>`` (assumes ``psi`` is normalised)."""
        psi = np.asarray(psi)
        return float(np.sum(self.x * np.abs(psi) ** 2) * self.dx)
