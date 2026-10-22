r"""Stationary states by shifted inverse iteration.

Given a shift ``epsilon`` close to an eigenvalue ``E_n`` of ``H``, repeatedly
solving

    (H - epsilon) psi_{k+1} = phi_k,       phi_{k+1} = psi_{k+1} / ||psi_{k+1}||

amplifies the component of ``phi`` along the eigenvector whose eigenvalue is
closest to ``epsilon`` by a factor ``|E_m - epsilon| / |E_n - epsilon|`` per
iteration. Each solve is a tridiagonal system, i.e. O(N).

Convergence is accelerated by updating the shift with

    epsilon <- epsilon + 1 / <phi_k | psi_{k+1}>

(for an exact eigenvector ``<phi|(H - eps)^{-1}|phi> = 1/(E - eps)`` so the
update lands on ``E``). The first few iterations use a fixed shift so that
the iteration locks onto the eigenvalue nearest to the *initial* shift
before the (much faster) shift update kicks in.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import LinAlgError, solve_banded

from .hamiltonian import Hamiltonian1D

__all__ = ["EigenPair", "inverse_iteration"]


@dataclass(frozen=True)
class EigenPair:
    """Result of :func:`inverse_iteration`."""

    energy: float
    state: NDArray[np.float64]  #: grid-normalised, sign fixed (largest lobe > 0)
    iterations: int
    residual: float  #: ``||H phi - E phi|| / ||phi||`` (Euclidean)
    converged: bool


def inverse_iteration(
    hamiltonian: Hamiltonian1D,
    shift: float,
    *,
    tol: float = 1e-10,
    max_iter: int = 500,
    fixed_shift_iterations: int = 3,
    update_shift: bool = True,
    initial: ArrayLike | None = None,
    seed: int | None = 0,
) -> EigenPair:
    """Eigenpair of ``hamiltonian`` whose energy is closest to ``shift``.

    Parameters
    ----------
    hamiltonian:
        Tridiagonal Hamiltonian.
    shift:
        Initial guess for the energy.
    tol:
        Stop when the relative residual ``||H phi - E phi|| / max(1, |E|)``
        falls below ``tol``.
    max_iter:
        Maximum number of linear solves.
    fixed_shift_iterations:
        Number of iterations performed with the initial shift before the
        shift update is switched on (ignored if ``update_shift`` is False).
    update_shift:
        Use the accelerated shift update described in the module docstring.
    initial:
        Starting vector. Defaults to a random vector (reproducible through
        ``seed``) so that states of any parity can be reached.
    seed:
        Seed of the random starting vector.
    """
    h = hamiltonian
    n = h.n
    if initial is None:
        phi = np.random.default_rng(seed).random(n)
    else:
        phi = np.array(initial, dtype=float)
        if phi.shape != (n,):
            raise ValueError(f"initial vector must have shape ({n},)")
    phi /= np.linalg.norm(phi)

    ab = np.empty((3, n))
    ab[0, 1:] = h.offdiag
    ab[2, :-1] = h.offdiag
    ab[0, 0] = ab[2, -1] = 0.0

    eps = float(shift)
    energy = h.expectation(phi)
    residual = np.inf
    converged = False
    it = 0
    for it in range(1, max_iter + 1):
        ab[1] = h.diag - eps
        try:
            psi = solve_banded((1, 1), ab, phi, check_finite=False)
        except LinAlgError:
            # The shift is (numerically) an exact eigenvalue: phi already is
            # the eigenvector to machine precision.
            converged = True
            break

        if update_shift and it > fixed_shift_iterations:
            overlap = phi @ psi
            if overlap != 0.0 and np.isfinite(overlap):
                eps += 1.0 / overlap

        psi /= np.linalg.norm(psi)
        if phi @ psi < 0.0:  # remove the sign flip that occurs when eps > E
            psi = -psi
        phi = psi

        energy = float(phi @ h.apply(phi))  # Rayleigh quotient, ||phi|| = 1
        residual = float(np.linalg.norm(h.apply(phi) - energy * phi))
        if residual < tol * max(1.0, abs(energy)):
            converged = True
            break

    if not converged:
        warnings.warn(
            f"inverse iteration did not converge in {max_iter} iterations "
            f"(residual {residual:.2e})",
            RuntimeWarning,
            stacklevel=2,
        )

    energy = float(phi @ h.apply(phi))
    residual = float(np.linalg.norm(h.apply(phi) - energy * phi))
    state = phi / np.sqrt(h.grid.dx)  # grid normalisation: sum |phi|^2 dx = 1
    if state[np.argmax(np.abs(state))] < 0:
        state = -state
    return EigenPair(energy, state, it, residual, converged)
