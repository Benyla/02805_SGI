"""Figure 2 (single column): the rule passes checks it never saw.

Share of names flagged ghost-like in four groups the rule knows nothing about, grouped by
what the flag *should* say about them, against the share among all names on the mood playlists.

    python -m src.figures.fig2_convergence
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from src.figures import style
from src.figures.style import GHOST, REAL, TEXT, TEXT_2
from src.load import PROCESSED

# (key in reference table, row label, expected answer)
ROWS = [
    ("Firefly Entertainment", "Firefly Entertainment", "ghost"),
    ("Genre-playlist artists", "Genre-playlist artists", "real"),
    ("Epidemic Sound", "Epidemic Sound", "real"),
    ("Classical majors", "Classical major labels", "real"),
]


def main() -> None:
    style.apply()
    ref = pd.read_csv(PROCESSED / "convergence_reference_sets.csv").set_index("name")
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    base = df.loc[df.group == "functional", "ghost"].mean()

    fig, ax = plt.subplots(figsize=(2.6, 1.95))
    # y positions with a gap between the two groups
    y = {0: 4.3, 1: 2.6, 2: 1.6, 3: 0.6}
    for i, (key, lab, exp) in enumerate(ROWS):
        s, n = ref.loc[key, "share_ghost"], int(ref.loc[key, "n_artists"])
        col = GHOST if exp == "ghost" else REAL
        ax.barh(y[i], s, height=0.62, color=col, lw=0, zorder=2)
        ax.text(s + 0.02, y[i], f"{s:.0%}", va="center", ha="left", fontsize=7, color=TEXT,
                fontweight="bold", zorder=3)
        ax.text(-0.02, y[i], f"{lab}\n{n} names", va="center", ha="right", fontsize=6.3, color=TEXT,
                linespacing=1.1)
    # group headers
    ax.text(-0.02, 5.15, "Should be flagged", ha="right", va="center", fontsize=6.5, color=GHOST,
            fontweight="bold")
    ax.text(-0.02, 3.45, "Should not be flagged", ha="right", va="center", fontsize=6.5, color=REAL,
            fontweight="bold")
    # reference line
    ax.axvline(base, color=TEXT_2, lw=0.7, ls=(0, (2, 2)), zorder=1)
    ax.text(base + 0.015, 5.15, f"All mood-playlist\nnames ({base:.0%})", ha="left", va="center",
            fontsize=6, color=TEXT_2, linespacing=1.1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.05, 5.6)
    ax.set_yticks([])
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"], fontsize=6.5)
    ax.set_xlabel("Share of names the rule flags as ghost-like", fontsize=6.5)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    style.save(fig, "fig2_convergence")


if __name__ == "__main__":
    main()
