"""Figure 2 (single column): the rule passes checks it never saw.

Share of names flagged ghost-like in four groups the rule knows nothing about, against the
share among all names on the mood playlists.

    python -m src.figures.fig2_convergence
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures import style
from src.figures.style import GHOST, NEUTRAL, REAL, TEXT, TEXT_2
from src.load import PROCESSED

ROWS = [
    ("Firefly Entertainment", "Firefly Entertainment\n(company in the press reports)"),
    ("Epidemic Sound", "Epidemic Sound's own names\n(credited composers)"),
    ("Classical majors", "Classical majors\n(DG, Decca, Sony, Warner)"),
    ("Genre-playlist artists", "Artists on the genre\nplaylists (control)"),
]


def main() -> None:
    style.apply()
    ref = pd.read_csv(PROCESSED / "convergence_reference_sets.csv").set_index("name")
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    base = df.loc[df.group == "functional", "ghost"].mean()

    fig, ax = plt.subplots(figsize=(2.5, 1.75))
    y = np.arange(len(ROWS))[::-1]
    for yi, (key, lab) in zip(y, ROWS):
        s, n = ref.loc[key, "share_ghost"], int(ref.loc[key, "n_artists"])
        ax.plot([0, 1], [yi, yi], color="#ecebe7", lw=5, solid_capstyle="round", zorder=1)
        ax.scatter([s], [yi], s=46, color=GHOST if s > 0.5 else REAL, zorder=3, lw=0.6, ec="white")
        ax.text(s + 0.045, yi, f"{s:.0%}  (n = {n})", va="center", ha="left", fontsize=6.5, color=TEXT)
    ax.axvline(base, color=TEXT_2, lw=0.7, ls=(0, (2, 2)), zorder=2)
    ax.text(base - 0.02, len(ROWS) - 0.45, f"all names on mood playlists: {base:.0%}", ha="right", va="bottom",
            fontsize=6.5, color=TEXT_2)
    ax.set_yticks(y, [lab for _, lab in ROWS], fontsize=6.5)
    ax.set_xlim(-0.03, 1.3)
    ax.set_ylim(-0.6, len(ROWS) - 0.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"], fontsize=6.5)
    ax.set_xlabel("Share flagged ghost-like", fontsize=6.5)
    ax.xaxis.set_label_coords(0.53 / 1.33, -0.2)
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_bounds(0, 1)
    ax.tick_params(axis="y", length=0)
    style.save(fig, "fig2_convergence")


if __name__ == "__main__":
    main()
