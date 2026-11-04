"""Shared matplotlib style for every paper figure.

Colours come from the validated reference palette (dataviz skill): the first categorical
slots pass the colour-blind checks as a set. Group colours are fixed by entity, never by rank.
"""
from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

from src.load import FIGURES

GROUP_COLORS = {"functional": "#2a78d6", "genre": "#eb6834"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three slots: validated all-pairs
NEUTRAL = "#a3a29c"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
SEQ_CMAP = "Blues"  # sequential = one hue, light -> dark


def apply() -> None:
    mpl.rcParams.update({
        "figure.dpi": 150,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "font.size": 7,
        "axes.titlesize": 7.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 6.5,
        "axes.labelcolor": TEXT_2,
        "axes.edgecolor": NEUTRAL,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.frameon": False,
        "legend.fontsize": 6,
        "lines.linewidth": 2,
        "pdf.fonttype": 42,
    })


def save(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png")

# Main dichotomy of the paper (validated pair: slots 1 and 2)
GHOST = "#eb6834"
REAL = "#2a78d6"
