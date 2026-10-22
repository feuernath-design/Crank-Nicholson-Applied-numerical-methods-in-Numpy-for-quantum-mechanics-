r"""Error analysis of the Crank-Nicolson scheme.

For an eigenstate ``H phi = E phi`` the Crank-Nicolson step multiplies
``phi`` by

    (1 - i E dt/2) / (1 + i E dt/2) = exp(-2 i arctan(E dt / 2)),

a pure phase (unitarity). The numerical evolution is therefore that of a
state with the *effective* energy

    E_eff(dt) = (2 / dt) arctan(E dt / 2) = E - E^3 dt^2 / 12 + O(dt^4),

so the phase (energy) error is ``|E_eff - E| ~ E^3 dt^2 / 12``: second order
in ``dt`` and growing like ``E^3``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .propagator import CrankNicolson

__all__ = [
    "cn_effective_energy",
    "cn_energy_error_leading_order",
    "loglog_slope",
    "measure_energy_error",
]


def cn_effective_energy(energy: ArrayLike, dt: ArrayLike) -> NDArray[np.float64]:
    """Exact effective energy ``(2/dt) arctan(E dt/2)`` of the CN propagator."""
    energy = np.asarray(energy, dtype=float)
    dt = np.asarray(dt, dtype=float)
    return 2.0 / dt * np.arctan(0.5 * energy * dt)


def cn_energy_error_leading_order(energy: ArrayLike, dt: ArrayLike) -> NDArray[np.float64]:
    """Leading-order phase error ``|E|^3 dt^2 / 12``."""
    energy = np.asarray(energy, dtype=float)
    dt = np.asarray(dt, dtype=float)
    return np.abs(energy) ** 3 * dt**2 / 12.0


def measure_energy_error(
    propagator: CrankNicolson, state: ArrayLike, energy: float, total_time: float
) -> float:
    r"""Numerically measured ``|E_eff - E|`` for an eigenstate.

    The state is propagated up to ``total_time`` and, at every step, the
    phase advance of ``c(t) = <phi|psi(t)>`` is compared with the exact
    advance ``-E dt``. Each per-step discrepancy is tiny (<< pi), so summing
    them never suffers from 2 pi ambiguities even when the total phase winds
    many times. The accumulated phase error divided by the elapsed time is the
    energy error.
    """
    if propagator.field is not None:
        raise ValueError("the energy error is defined for a field-free propagator")
    dt = propagator.dt
    n_steps = round(total_time / dt)
    if n_steps < 1:
        raise ValueError("total_time must be at least one time step")
    grid = propagator.h.grid
    phi = np.asarray(state, dtype=np.complex128)
    psi = phi.copy()
    c_prev = grid.inner(phi, psi)
    exact_step_phase = np.exp(1j * energy * dt)
    accumulated = 0.0
    for _ in range(n_steps):
        psi = propagator.step(psi)
        c = grid.inner(phi, psi)
        accumulated += np.angle(c * np.conj(c_prev) * exact_step_phase)
        c_prev = c
    return abs(accumulated) / (n_steps * dt)


def loglog_slope(x: ArrayLike, y: ArrayLike) -> float:
    """Least-squares slope of ``log y`` versus ``log x`` (convergence order)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.any(x <= 0) or np.any(y <= 0):
        raise ValueError("log-log slope needs strictly positive data")
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])
