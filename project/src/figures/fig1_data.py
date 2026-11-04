"""Figure 1: the data as three linked layers (schematic; every count is computed).

Top layer: playlists. Middle: artist names. Bottom: ISRC houses.
Edges: placement (playlist - name), "Fans also like" (name -> name), ownership (name - house).

    python -m src.figures.fig1_data
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.figures import style
from src.figures.style import GHOST, REAL, TEXT, TEXT_2
from src.load import PROCESSED, SEED, load

GENRE_C = "#a3a29c"


def main() -> None:
    style.apply()
    rng = np.random.default_rng(SEED)
    pl, t, ar = load("playlists"), load("tracks"), load("artists")
    pt, fal = load("playlist_tracks"), load("fans_also_like_edges")
    own = json.loads((PROCESSED / "ownership.json").read_text())
    rec = json.loads((PROCESSED / "recommendation.json").read_text())

    fig, ax = plt.subplots(figsize=(7.08, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 40)
    ax.axis("off")

    yP, yA, yH = 34, 19.5, 5
    # layer bands
    for y, lab in [(yP, "Playlists"), (yA, "Artist names"), (yH, "ISRC houses")]:
        ax.add_patch(FancyBboxPatch((1, y - 4.2), 62, 8.4, boxstyle="round,pad=0,rounding_size=2",
                                    fc="#f4f3f0", ec="none", zorder=0))
        ax.text(2.2, y + 3.0, lab, fontsize=6.5, fontweight="bold", color=TEXT_2, va="center")

    # playlists: 9 functional + 2 genre shown
    px = np.r_[np.linspace(8, 46, 9), [53, 58.5]]
    pc = ["functional"] * 9 + ["genre"] * 2
    # names: ghost cluster around houses, real names elsewhere
    n_g, n_r = 26, 12
    ax_g = np.sort(rng.uniform(6, 44, n_g))
    ax_r = np.sort(rng.uniform(40, 61, n_r))
    ax_r[:4] = rng.uniform(10, 40, 4)  # real artists on functional playlists (e.g. classical)
    names_x = np.r_[ax_g, ax_r]
    names_y = yA + rng.uniform(-2.4, 2.4, len(names_x))
    is_g = np.r_[np.ones(n_g, bool), np.zeros(n_r, bool)]
    # houses: 3 big ghost houses, several small labels
    hx = np.array([12, 25, 38, 47, 52, 56, 60])
    hsize = np.array([95, 80, 70, 14, 10, 12, 9])
    # placement edges
    for i, x in enumerate(names_x):
        cand = np.where(np.array(pc) == ("genre" if (not is_g[i] and x > 50) else "functional"))[0]
        j = cand[np.argmin(np.abs(px[cand] - x) + rng.uniform(0, 8, len(cand)))]
        ax.plot([x, px[j]], [names_y[i], yP], color="#c9c8c2", lw=0.45, zorder=1)
    # ownership edges
    for i, x in enumerate(names_x):
        k = np.argmin(np.abs(hx[:3] - x)) if is_g[i] else 3 + np.argmin(np.abs(hx[3:] - x))
        ax.plot([x, hx[k]], [names_y[i], yH], color="#c9c8c2", lw=0.45, zorder=1)
    # Fans-also-like arcs among names
    for _ in range(16):
        i, j = rng.choice(len(names_x), 2, replace=False)
        if abs(names_x[i] - names_x[j]) > 14:
            continue
        ax.add_patch(FancyArrowPatch((names_x[i], names_y[i]), (names_x[j], names_y[j]),
                                     connectionstyle="arc3,rad=0.5", arrowstyle="-|>,head_length=1.6,head_width=0.9",
                                     color="#6f6e69", lw=0.45, zorder=2))
    ax.scatter(px, [yP] * len(px), s=46, marker="s",
               c=["#3d3c38" if c == "functional" else GENRE_C for c in pc], zorder=3, lw=0)
    ax.scatter(names_x[is_g], names_y[is_g], s=13, c=GHOST, zorder=3, lw=0.4, ec="white")
    ax.scatter(names_x[~is_g], names_y[~is_g], s=13, c=REAL, zorder=3, lw=0.4, ec="white")
    ax.scatter(hx, [yH] * len(hx), s=hsize, c="#1f1f1e", zorder=3, lw=0.6, ec="white")

    # edge-type labels
    lab_kw = dict(fontsize=5.6, color=TEXT, ha="left", va="center",
                  bbox=dict(fc="white", ec="#c9c8c2", lw=0.5, boxstyle="round,pad=0.25"))
    ax.text(48.5, (yP + yA) / 2 + 0.3, "placement: name is on playlist", **lab_kw)
    ax.text(16.5, yA + 5.6, "Fans also like: name points to name", **lab_kw)
    ax.text(48.5, (yA + yH) / 2 - 0.3, "ownership: house registered the name's tracks", **lab_kw)

    # right-hand column: what each layer holds
    col = 79.5
    blocks = [
        (yP + 1.0, f"{len(pl)} Spotify editorial playlists",
         f"{(pl.group == 'functional').sum()} functional (sleep, focus, chill, jazz…)\n"
         f"{(pl.group == 'genre').sum()} genre controls · {len(pt):,} slots · descriptions"),
        (yA + 2.2, f"{len(ar):,} names · {len(t):,} tracks",
         "identity: verified, bio, links, gallery\n"
         f"demand: followers, listeners, cities · {len(fal):,} 'Fans\nalso like' edges · titles, bios, covers"),
        (yH + 1.0, f"{own['n_houses']} ISRC houses",
         "who registered the recording; from the ISRC\n(matched on Deezer) · plus credit Source, (P) line"),
    ]
    for y, title, sub in blocks:
        ax.text(col - 12.5, y + 2.6, title, fontsize=6.4, fontweight="bold", color=TEXT, va="center")
        ax.text(col - 12.5, y + 0.9, sub, fontsize=5.3, color=TEXT_2, va="top", linespacing=1.3)

    # legend
    ax.scatter([], [], s=13, c=GHOST, label="ghost-like name")
    ax.scatter([], [], s=13, c=REAL, label="real artist")
    ax.scatter([], [], s=30, marker="s", c="#3d3c38", label="functional playlist")
    ax.scatter([], [], s=30, marker="s", c=GENRE_C, label="genre playlist")
    ax.legend(loc="lower center", bbox_to_anchor=(0.32, -0.13), ncol=4, fontsize=5.8,
              handletextpad=0.2, columnspacing=1.2)
    ax.text(1, 39.5, "Schematic (counts are real)", fontsize=5.2, color=TEXT_2, style="italic")
    style.save(fig, "fig1_data")


if __name__ == "__main__":
    main()
