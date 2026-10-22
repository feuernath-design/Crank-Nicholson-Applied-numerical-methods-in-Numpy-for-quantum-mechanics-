"""Photo-ionisation of the 1D soft-Coulomb atom: parameter scans.

Field F(t) = F0 f(t) sin(omega t) with a trapezoidal envelope f (4-cycle
linear ramp up, plateau, 4-cycle ramp down), followed by 2 field-free cycles.
The population left in the ground state at the end, P0 = |<psi0|psi_final>|^2,
is studied as a function of

1. time (trapezoid, F0 = 0.01, omega = 1),
2. the frequency omega in [0.8, 2.5] a.u. (all above the ionisation
   potential Ip = 0.67 a.u.: one-photon ionisation). The pulse has a fixed
   number of cycles, so its duration scales as 1/omega; the depletion is
   therefore also shown per unit time, (1 - P0) / T_eff with
   T_eff = integral f(t)^2 dt = (n_flat + (n_up + n_down)/3) * 2 pi / omega,
   which isolates the ionisation rate (proportional to the cross-section),
3. the amplitude F0 in [1e-4, 0.2] (omega = 1): first-order perturbation
   theory predicts 1 - P0 ~ F0^2 (slope 2 in log-log),
4. the pulse duration (plateau length): Fermi's golden rule predicts a
   constant ionisation rate, i.e. 1 - P0 linear in the duration.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import parse_args, save

from tdse1d import (
    CrankNicolson,
    Grid,
    Hamiltonian1D,
    LaserField,
    inverse_iteration,
    loglog_slope,
    potentials,
    trapezoid_envelope,
)

STEPS_PER_CYCLE = 128


def propagate(h, ground, f0, omega, n_up=4, n_flat=8, n_down=4, n_free=2, history=False):
    """Return (P0_final, max|norm-1|[, times, overlap])."""
    grid = h.grid
    period = 2 * np.pi / omega
    envelope = trapezoid_envelope(n_up * period, n_flat * period, n_down * period)
    field = LaserField(f0, omega, envelope)
    n_steps = (n_up + n_flat + n_down + n_free) * STEPS_PER_CYCLE
    observables = {"norm": grid.norm2}
    if history:
        observables["overlap"] = lambda psi: abs(grid.inner(ground.state, psi))
    res = CrankNicolson(h, period / STEPS_PER_CYCLE, field=field).propagate(
        ground.state, n_steps, observables=observables, record_every=1 if history else n_steps
    )
    p0 = abs(grid.inner(ground.state, res.psi)) ** 2
    norm_err = float(np.max(np.abs(res["norm"] - 1)))
    if history:
        return p0, norm_err, res.times, res["overlap"]
    return p0, norm_err


def main() -> None:
    args = parse_args(__doc__)
    grid = Grid.from_spacing(-150.0, 150.0, 0.1)
    h = Hamiltonian1D(grid, potentials.soft_coulomb(grid.x))
    ground = inverse_iteration(h, shift=-1.0)
    print(f"ground state energy: {ground.energy:.6f} a.u.")

    # ---------------------------------------------------- 1) time trace
    period = 2 * np.pi
    p0, nerr, times, overlap = propagate(h, ground, 1e-2, 1.0, history=True)
    print(f"[trapezoid omega=1, F0=0.01] P0 = {p0:.8f}, max|norm-1| = {nerr:.1e}")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(times, overlap)
    ax.axvspan(0, 4 * period, color="orange", alpha=0.12, label="ramp up (4 cycles)")
    ax.axvspan(12 * period, 16 * period, color="red", alpha=0.10, label="ramp down (4 cycles)")
    ax.axvspan(16 * period, 18 * period, color="green", alpha=0.10, label="field off (2 cycles)")
    ax.set_xlabel("t (a.u.)")
    ax.set_ylabel(r"$|\langle\psi_0|\psi(t)\rangle|$")
    ax.set_title(r"Trapezoidal pulse: $F_0 = 0.01$, $\omega = 1$ a.u.")
    ax.ticklabel_format(axis="y", useOffset=False)
    ax.legend()
    save(fig, args, "05a_trapezoid_overlap.png")

    # ---------------------------------------------------- 2) frequency scan
    freqs = np.arange(0.8, 2.5001, 0.25 if args.quick else 0.05)
    p0_w = np.array([propagate(h, ground, 1e-2, w)[0] for w in freqs])
    # Effective duration integral f^2 dt of the trapezoid (4 + 8 + 4 cycles)
    t_eff = (8 + (4 + 4) / 3) * 2 * np.pi / freqs
    rate_w = (1 - p0_w) / t_eff
    for w, p, r in zip(freqs, p0_w, rate_w, strict=True):
        print(f"  omega = {w:.2f}  P0 = {p:.8f}  rate = {r:.3e} a.u.")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(freqs, p0_w, "o-", ms=4)
    ax1.set_xlabel(r"$\omega$ (a.u.)")
    ax1.set_ylabel(r"$P_0 = |\langle\psi_0|\psi_{\rm final}\rangle|^2$")
    ax1.set_title("Population after a 16-cycle pulse")
    ax1.ticklabel_format(axis="y", useOffset=False)
    ax2.semilogy(freqs, rate_w, "o-", ms=4, color="tab:red")
    ax2.set_xlabel(r"$\omega$ (a.u.)")
    ax2.set_ylabel(r"$(1 - P_0)\,/\,T_{\rm eff}$ (a.u.)")
    ax2.set_title("Ionisation rate (depletion per unit time)")
    ax2.grid(True, which="both", alpha=0.3)
    fig.suptitle(r"$F_0 = 0.01$ a.u., $I_p = 0.67$ a.u.")
    save(fig, args, "05b_frequency_scan.png")

    # ---------------------------------------------------- 3) amplitude scan
    mantissas = (1, 5) if args.quick else (1, 2, 5, 7)
    amplitudes = sorted(
        {m * d for d in (1e-4, 1e-3, 1e-2, 1e-1) for m in mantissas if m * d <= 0.2} | {0.2}
    )
    amplitudes = np.array(amplitudes)
    depletion = np.array([1 - propagate(h, ground, f0, 1.0)[0] for f0 in amplitudes])
    weak = amplitudes <= 1e-2
    slope = loglog_slope(amplitudes[weak], depletion[weak])
    print(f"  slope of 1 - P0 vs F0 (F0 <= 0.01): {slope:.4f} (perturbation theory: 2)")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.semilogx(amplitudes, 1 - depletion, "o-", ms=4)
    ax1.set_xlabel(r"$F_0$ (a.u.)")
    ax1.set_ylabel(r"$P_0$")
    ax1.set_title("Final population vs amplitude")
    ax2.loglog(amplitudes, depletion, "o-", ms=4, label=r"$1 - P_0$")
    ax2.loglog(
        amplitudes,
        depletion[0] * (amplitudes / amplitudes[0]) ** 2,
        "k--",
        lw=1,
        label=r"$\propto F_0^2$",
    )
    ax2.set_xlabel(r"$F_0$ (a.u.)")
    ax2.set_ylabel(r"$1 - P_0$")
    ax2.set_title(f"Depletion (log-log), weak-field slope = {slope:.3f}")
    ax2.legend()
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle(r"$\omega = 1$ a.u.")
    save(fig, args, "05c_amplitude_scan.png")

    # ---------------------------------------------------- 4) duration scan
    plateaus = np.array([0, 4, 12] if args.quick else [0, 2, 4, 8, 12, 16, 20, 28, 40])
    durations = (4 + plateaus + 4) * period
    depl_d = np.array([1 - propagate(h, ground, 1e-2, 1.0, n_flat=int(p))[0] for p in plateaus])
    rate, intercept = np.polyfit(durations, depl_d, 1)
    resid = np.max(np.abs(np.polyval([rate, intercept], durations) - depl_d) / depl_d)
    print(f"  linear fit: 1 - P0 = {rate:.3e} * T + {intercept:.2e}  (max rel. dev. {resid:.1e})")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(durations, depl_d, "o", ms=5, label="simulation")
    ax.plot(
        durations,
        np.polyval([rate, intercept], durations),
        "k--",
        lw=1,
        label=rf"linear fit, rate $\Gamma$ = {rate:.2e} a.u.",
    )
    ax.set_xlabel("total field duration (a.u.)")
    ax.set_ylabel(r"$1 - P_0$")
    ax.set_title(r"Depletion vs pulse duration ($\omega = 1$, $F_0 = 0.01$): Fermi golden rule")
    ax.legend()
    save(fig, args, "05d_duration_scan.png")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
