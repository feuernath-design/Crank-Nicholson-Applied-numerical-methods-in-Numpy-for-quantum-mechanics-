r"""Laser fields in the dipole approximation (length gauge).

The interaction with a linearly polarised field ``E(t)`` adds

    V_int(x, t) = E(t) * x

to the Hamiltonian (electron charge -1 in atomic units, sign convention of the
original internship code). The field is

    E(t) = E0 * f(t) * sin(omega * t + phase),

where ``f`` is a dimensionless envelope with values in ``[0, 1]``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "Envelope",
    "LaserField",
    "constant_envelope",
    "sin2_ramp_envelope",
    "trapezoid_envelope",
]

Envelope = Callable[[NDArray[np.float64]], NDArray[np.float64]]


def constant_envelope() -> Envelope:
    """``f(t) = 1``: field switched on suddenly at ``t = 0``."""

    def f(t: ArrayLike) -> NDArray[np.float64]:
        return np.ones_like(np.asarray(t, dtype=float))

    return f


def sin2_ramp_envelope(t_ramp: float) -> Envelope:
    """Smooth turn-on ``f(t) = sin^2(pi t / (2 t_ramp))`` for ``t < t_ramp``, then 1.

    ``t_ramp = 0`` reproduces the sudden switch-on.
    """
    if t_ramp < 0:
        raise ValueError("t_ramp must be non-negative")

    def f(t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=float)
        if t_ramp == 0:
            return np.ones_like(t)
        ramp = np.sin(np.pi * np.clip(t, 0.0, t_ramp) / (2.0 * t_ramp)) ** 2
        return np.where(t < t_ramp, ramp, 1.0)

    return f


def trapezoid_envelope(t_up: float, t_flat: float, t_down: float) -> Envelope:
    """Linear ramp up (``t_up``), plateau (``t_flat``), linear ramp down (``t_down``), then 0."""
    if min(t_up, t_flat, t_down) < 0:
        raise ValueError("durations must be non-negative")
    t1, t2 = t_up, t_up + t_flat
    t3 = t2 + t_down

    def f(t: ArrayLike) -> NDArray[np.float64]:
        t = np.asarray(t, dtype=float)
        up = t / t_up if t_up > 0 else np.ones_like(t)
        down = (t3 - t) / t_down if t_down > 0 else np.zeros_like(t)
        return np.select(
            [t < 0, t < t1, t < t2, t < t3],
            [np.zeros_like(t), up, np.ones_like(t), down],
            default=0.0,
        )

    return f


@dataclass(frozen=True)
class LaserField:
    """``E(t) = amplitude * envelope(t) * sin(omega t + phase)``."""

    amplitude: float
    omega: float
    envelope: Envelope = constant_envelope()  # noqa: RUF009 - immutable callable
    phase: float = 0.0

    @property
    def period(self) -> float:
        return 2.0 * np.pi / self.omega

    def __call__(self, t: ArrayLike) -> NDArray[np.float64] | float:
        t_arr = np.asarray(t, dtype=float)
        value = self.amplitude * self.envelope(t_arr) * np.sin(self.omega * t_arr + self.phase)
        return float(value) if value.ndim == 0 else value
