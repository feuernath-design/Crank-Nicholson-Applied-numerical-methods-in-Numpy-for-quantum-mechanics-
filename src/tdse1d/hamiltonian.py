r"""Finite-difference Hamiltonian ``H = -1/(2m) d^2/dx^2 + V(x)``.

The second derivative is discretised with the 3-point stencil

    psi''(x_j) ~ (psi_{j-1} - 2 psi_j + psi_{j+1}) / dx^2,

which is second-order accurate (error O(dx^2)) and turns ``H`` into a real
symmetric **tridiagonal** matrix:

    H[j, j]   = 1 / (m dx^2) + V(x_j)
    H[j, j+1] = H[j+1, j] = -1 / (2 m dx^2)

Only the two diagonals are stored, so every operation in the package is
O(N) in memory and time. A dense version is available for testing and
teaching (:meth:`Hamiltonian1D.to_dense`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import eigh_tridiagonal

from .grid import Grid

__all__ = ["Hamiltonian1D"]


@dataclass(frozen=True)
class Hamiltonian1D:
    """Tridiagonal Hamiltonian on a :class:`~tdse1d.grid.Grid`.

    Parameters
    ----------
    grid:
        Spatial grid.
    potential:
        Potential sampled on ``grid.x`` (array of length ``grid.n``).
    mass:
        Particle mass in atomic units (default: electron, ``1``).
    """

    grid: Grid
    potential: NDArray[np.float64]
    mass: float = 1.0
    _diag: NDArray[np.float64] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        pot = np.asarray(self.potential, dtype=float)
        if pot.shape != (self.grid.n,):
            raise ValueError(f"potential has shape {pot.shape}, expected ({self.grid.n},)")
        if self.mass <= 0:
            raise ValueError("mass must be positive")
        object.__setattr__(self, "potential", pot)
        object.__setattr__(self, "_diag", 1.0 / (self.mass * self.grid.dx**2) + pot)

    # ------------------------------------------------------------------ #
    # Matrix elements
    # ------------------------------------------------------------------ #
    @property
    def diag(self) -> NDArray[np.float64]:
        """Main diagonal ``1/(m dx^2) + V(x_j)``."""
        return self._diag

    @property
    def offdiag(self) -> float:
        """Constant off-diagonal element ``-1/(2 m dx^2)``."""
        return -0.5 / (self.mass * self.grid.dx**2)

    @property
    def n(self) -> int:
        return self.grid.n

    # ------------------------------------------------------------------ #
    # Linear algebra
    # ------------------------------------------------------------------ #
    def apply(self, psi: ArrayLike, extra_diag: ArrayLike | None = None) -> NDArray:
        """Return ``(H + diag(extra_diag)) @ psi`` in O(N)."""
        psi = np.asarray(psi)
        d = self._diag if extra_diag is None else self._diag + np.asarray(extra_diag)
        out = d * psi
        out[:-1] += self.offdiag * psi[1:]
        out[1:] += self.offdiag * psi[:-1]
        return out

    def expectation(self, psi: ArrayLike) -> float:
        """Energy ``<psi|H|psi> / <psi|psi>`` (grid-weighted)."""
        psi = np.asarray(psi)
        num = np.vdot(psi, self.apply(psi))
        den = np.vdot(psi, psi)
        return float(np.real(num / den))

    def to_dense(self) -> NDArray[np.float64]:
        """Dense ``N x N`` matrix (for tests and small grids only)."""
        off = np.full(self.n - 1, self.offdiag)
        return np.diag(self._diag) + np.diag(off, 1) + np.diag(off, -1)

    def eigh(self, k: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Lowest ``k`` eigenpairs with LAPACK (reference solver).

        Returns ``(energies, states)`` with ``states[:, n]`` grid-normalised
        and with a positive value at its largest-magnitude point (fixes the
        arbitrary sign).
        """
        if not 1 <= k <= self.n:
            raise ValueError(f"k must be in [1, {self.n}]")
        energies, vecs = eigh_tridiagonal(
            self._diag,
            np.full(self.n - 1, self.offdiag),
            select="i",
            select_range=(0, k - 1),
        )
        vecs = vecs / np.sqrt(self.grid.dx)
        for j in range(k):
            if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
                vecs[:, j] *= -1
        return energies, vecs
