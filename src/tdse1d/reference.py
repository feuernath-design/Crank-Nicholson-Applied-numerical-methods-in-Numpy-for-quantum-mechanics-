"""Dense-matrix Crank-Nicolson: the pedagogical version.

This is the "textbook" implementation used at the start of the internship:
``H`` is stored as a full ``N x N`` matrix and every step calls
``numpy.linalg.solve``. It costs O(N^3) per step (O(N^2) memory) and is only
kept as an independent reference for the test-suite and for teaching.
Use :class:`tdse1d.propagator.CrankNicolson` for real computations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .hamiltonian import Hamiltonian1D

__all__ = ["dense_cn_matrices", "dense_cn_step"]


def dense_cn_matrices(
    hamiltonian: Hamiltonian1D, dt: float
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Return ``(U_plus, U_minus) = (1 + i dt H/2, 1 - i dt H/2)`` as dense matrices."""
    h = hamiltonian.to_dense().astype(np.complex128)
    eye = np.eye(hamiltonian.n, dtype=np.complex128)
    return eye + 0.5j * dt * h, eye - 0.5j * dt * h


def dense_cn_step(
    psi: ArrayLike, u_plus: NDArray[np.complex128], u_minus: NDArray[np.complex128]
) -> NDArray[np.complex128]:
    """One step: solve ``U_plus psi_new = U_minus psi``."""
    return np.linalg.solve(u_plus, u_minus @ np.asarray(psi, dtype=np.complex128))
