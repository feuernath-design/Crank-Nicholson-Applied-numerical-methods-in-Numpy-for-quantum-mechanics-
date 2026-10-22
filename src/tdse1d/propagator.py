r"""Crank-Nicolson time propagation of the 1D TDSE.

The TDSE ``i d/dt psi = H(t) psi`` is advanced from ``t`` to ``t + dt`` with

    (1 + i dt/2 H(t + dt/2)) psi(t + dt) = (1 - i dt/2 H(t + dt/2)) psi(t).

Properties (see ``docs/theory.md``):

* **Unitary.** The one-step propagator is the Cayley transform
  ``U = (1 + i dt H/2)^{-1} (1 - i dt H/2)``, which is exactly unitary for
  Hermitian ``H``: the norm is conserved to machine precision whatever ``dt``.
* **Second order.** Evaluating ``H`` at the mid-point ``t + dt/2`` keeps the
  local error O(dt^3), hence a global error O(dt^2), also for a
  time-dependent field.
* **Unconditionally stable** and **time-reversible** (``U(-dt) = U(dt)^{-1}``).
* **O(N) per step.** ``H`` is tridiagonal, so the linear system is solved with
  LAPACK's banded solver (``scipy.linalg.solve_banded``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import solve_banded

from .grid import Grid
from .hamiltonian import Hamiltonian1D

__all__ = ["CrankNicolson", "PropagationResult", "absorbing_mask"]

Observable = Callable[[NDArray[np.complex128]], complex | float]


def absorbing_mask(grid: Grid, width: float, power: float = 0.125) -> NDArray[np.float64]:
    r"""``cos^power`` mask that damps the wavefunction within ``width`` of each edge.

    Multiplying ``psi`` by this mask after every step removes outgoing
    (ionised) flux before it reflects on the box edges. The norm is then no
    longer conserved: ``1 - <psi|psi>`` is the probability absorbed at the
    boundaries. ``power = 1/8`` is the usual choice (Krause, Schafer &
    Kulander, Phys. Rev. A 45, 4998 (1992)).
    """
    if width <= 0 or 2 * width >= grid.x_max - grid.x_min:
        raise ValueError("width must be positive and smaller than half the box")
    x = grid.x
    dist = np.minimum(x - grid.x_min, grid.x_max - x)  # distance to nearest edge
    mask = np.ones_like(x)
    inside = dist < width
    mask[inside] = np.cos(0.5 * np.pi * (width - dist[inside]) / width) ** power
    return mask


@dataclass
class PropagationResult:
    """Output of :meth:`CrankNicolson.propagate`."""

    psi: NDArray[np.complex128]  #: final wavefunction
    t: float  #: final time
    times: NDArray[np.float64]  #: times at which observables were recorded
    records: dict[str, NDArray] = field(default_factory=dict)

    def __getitem__(self, name: str) -> NDArray:
        return self.records[name]


class CrankNicolson:
    """Crank-Nicolson propagator for ``H(t) = H0 + E(t) x``.

    Parameters
    ----------
    hamiltonian:
        Field-free Hamiltonian ``H0``.
    dt:
        Time step (may be negative to propagate backwards).
    field:
        Optional callable ``E(t)`` (e.g. :class:`tdse1d.fields.LaserField`).
        The coupling ``E(t) x`` is evaluated at the mid-point of each step.
    mask:
        Optional absorbing mask applied after every step (see
        :func:`absorbing_mask`).
    """

    def __init__(
        self,
        hamiltonian: Hamiltonian1D,
        dt: float,
        field: Callable[[float], float] | None = None,
        mask: ArrayLike | None = None,
    ) -> None:
        if dt == 0 or not np.isfinite(dt):
            raise ValueError("dt must be finite and non-zero")
        self.h = hamiltonian
        self.dt = float(dt)
        self.field = field
        self.mask = None if mask is None else np.asarray(mask, dtype=float)
        if self.mask is not None and self.mask.shape != (hamiltonian.n,):
            raise ValueError("mask must have the same length as the grid")
        self._x = hamiltonian.grid.x
        self._a = 0.5j * self.dt  # i dt / 2
        self._a_off = self._a * hamiltonian.offdiag
        self._ab = np.zeros((3, hamiltonian.n), dtype=np.complex128)

    # ------------------------------------------------------------------ #
    def diagonal_at(self, t: float) -> NDArray[np.float64]:
        """Main diagonal of ``H(t)``."""
        if self.field is None:
            return self.h.diag
        return self.h.diag + float(self.field(t)) * self._x

    def step(self, psi: ArrayLike, t: float = 0.0) -> NDArray[np.complex128]:
        """Advance ``psi`` from ``t`` to ``t + dt``."""
        psi = np.asarray(psi, dtype=np.complex128)
        d = self.diagonal_at(t + 0.5 * self.dt)
        a, a_off = self._a, self._a_off

        # Right-hand side b = (1 - i dt/2 H) psi
        b = (1.0 - a * d) * psi
        b[:-1] -= a_off * psi[1:]
        b[1:] -= a_off * psi[:-1]

        # Banded storage of (1 + i dt/2 H); rebuilt every step because
        # solve_banded is allowed to overwrite it.
        ab = self._ab
        ab[0, 0] = 0.0
        ab[0, 1:] = a_off
        ab[1] = 1.0 + a * d
        ab[2, :-1] = a_off
        ab[2, -1] = 0.0
        out = solve_banded((1, 1), ab, b, overwrite_ab=True, overwrite_b=True, check_finite=False)
        if self.mask is not None:
            out *= self.mask
        return out

    def propagate(
        self,
        psi0: ArrayLike,
        n_steps: int,
        *,
        t0: float = 0.0,
        observables: Mapping[str, Observable] | None = None,
        record_every: int = 1,
    ) -> PropagationResult:
        """Propagate ``psi0`` for ``n_steps`` steps.

        ``observables`` maps names to functions ``f(psi)``; each is evaluated
        on the initial state and then every ``record_every`` steps (the final
        state is always recorded).
        """
        if n_steps < 0:
            raise ValueError("n_steps must be non-negative")
        if record_every < 1:
            raise ValueError("record_every must be >= 1")
        observables = dict(observables or {})
        psi = np.array(psi0, dtype=np.complex128)
        if psi.shape != (self.h.n,):
            raise ValueError(f"psi0 must have shape ({self.h.n},)")

        record_steps = list(range(0, n_steps + 1, record_every))
        if record_steps[-1] != n_steps:
            record_steps.append(n_steps)
        n_rec = len(record_steps)
        times = t0 + self.dt * np.asarray(record_steps, dtype=float)
        records = {name: np.empty(n_rec, dtype=np.complex128) for name in observables}

        def record(slot: int) -> None:
            for name, fn in observables.items():
                records[name][slot] = fn(psi)

        record(0)
        slot = 1
        for k in range(1, n_steps + 1):
            # t computed from the step index: no accumulated rounding error
            psi = self.step(psi, t0 + (k - 1) * self.dt)
            if slot < n_rec and k == record_steps[slot]:
                record(slot)
                slot += 1

        # Return real arrays when an observable is real-valued
        for name, arr in records.items():
            if np.all(arr.imag == 0):
                records[name] = arr.real.copy()
        return PropagationResult(psi, t0 + n_steps * self.dt, times, records)
