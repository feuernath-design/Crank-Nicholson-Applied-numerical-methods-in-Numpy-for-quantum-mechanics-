"""Shared helpers for the figure scripts (command-line options, plot style)."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTDIR = ROOT / "figures"


def parse_args(description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_OUTDIR, help="output folder for figures"
    )
    parser.add_argument(
        "--quick", action="store_true", help="coarse/short run (smoke test, used by CI)"
    )
    parser.add_argument("--show", action="store_true", help="open the figures interactively")
    args = parser.parse_args()
    if not args.show:
        matplotlib.use("Agg")
    args.outdir.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 150,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )
    return args


def save(fig, args: argparse.Namespace, name: str) -> None:
    path = args.outdir / name
    fig.tight_layout()
    fig.savefig(path)
    print(f"figure saved: {path}")
