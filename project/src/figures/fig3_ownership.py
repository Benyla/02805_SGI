"""Figure 3: hundreds of artist names, eight houses.

A  Ownership network. Each dark node is a house (the ISRC registrant of a name's tracks); each
   dot around it is an artist name whose tracks it registered. Disc area ~ number of names.
   Houses with one or two names are not drawn (counted in a note).
B  Who registered the tracks on the two kinds of playlist: each kind's eight largest houses,
   all other houses, and tracks without a code. No ghost label involved.

    python -m src.figures.fig3_ownership
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src.figures import style
from src.figures.houses import company, display_names
from src.figures.netlayout import pack_discs, vogel
from src.figures.style import GHOST, NEUTRAL, REAL, TEXT, TEXT_2
from src.load import PROCESSED

HUB = "#1f1f1e"
MIN_NAMES = 3  # smaller houses are counted, not drawn


def panel_network(ax) -> None:
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    dom = df.registrant_dom.dropna()
    members = dom.groupby(dom).apply(lambda s: list(s.index))
    members = members.loc[members.map(len).sort_values(ascending=False).index]
    small = members[members.map(len) < MIN_NAMES]
    members = members[members.map(len) >= MIN_NAMES]
    houses = members.index.tolist()
    sizes = members.map(len).to_numpy()

    radii = np.sqrt(sizes + 1.6 ** 2) + 0.4
    pads = np.where(sizes >= 20, 3.0, 0.9)
    centers = pack_discs(radii + pads / 2, pad=0.6, step=0.03)
    hpos = dict(zip(houses, centers))

    name_xy, name_ghost = {}, {}
    for h, names in members.items():
        g = df.ghost.reindex(names).fillna(False).astype(bool).sort_values(ascending=False)
        for n, q in zip(g.index, vogel(len(g)) + hpos[h]):
            name_xy[n], name_ghost[n] = q, bool(g[n])
    segs = [(hpos[dom[n]], xy) for n, xy in name_xy.items()]
    ax.add_collection(LineCollection(segs, colors="#d6d5cf", linewidths=0.12, zorder=1))
    xy = np.array(list(name_xy.values()))
    g = np.array([name_ghost[n] for n in name_xy])
    ax.scatter(xy[~g, 0], xy[~g, 1], s=1.6, c=REAL, lw=0, zorder=2, rasterized=True)
    ax.scatter(xy[g, 0], xy[g, 1], s=1.6, c=GHOST, lw=0, zorder=2, rasterized=True)
    c = np.array([hpos[h] for h in houses])
    ax.scatter(c[:, 0], c[:, 1], s=3 + 0.1 * sizes, c=HUB, lw=0.4, ec="white", zorder=3)

    # the eight largest houses (same names as in Fig. 1), labelled in two outer columns
    comp, disp = company(), display_names()
    allxy = np.vstack([xy, c])
    xmin, xmax = allxy[:, 0].min(), allxy[:, 0].max()
    idx_all = [houses.index(h) for h in disp]
    cx = np.median([centers[i][0] for i in idx_all])
    for side in ("left", "right"):
        idx = [i for i in idx_all if (centers[i][0] < cx) == (side == "left")]
        idx.sort(key=lambda i: -centers[i][1])
        ys = [centers[i][1] for i in idx]
        for k in range(1, len(ys)):
            ys[k] = min(ys[k], ys[k - 1] - 16.0)
        shift = (np.mean([centers[i][1] for i in idx]) - np.mean(ys)) if idx else 0
        ys = [y + shift for y in ys]
        lx = xmin - 3 if side == "left" else xmax + 3
        for i, ly in zip(idx, ys):
            h = houses[i]
            x, y = centers[i]
            ax.plot([x, lx], [y, ly], color=TEXT_2, lw=0.5, zorder=4)
            ax.scatter([x], [y], s=7, c="white", zorder=4, lw=0)
            ax.text(lx + (-1 if side == "left" else 1), ly, f"{disp[h]} · {sizes[i]} names\n{comp[h]}",
                    ha="right" if side == "left" else "left", va="center", fontsize=6.5,
                    color=TEXT, linespacing=1.15, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_ylim(allxy[:, 1].min() - 4, allxy[:, 1].max() + 4)
    ax.axis("off")
    ax.set_title("A  Hundreds of artist names, eight houses", loc="left")
    ax.scatter([], [], s=16, c=HUB, label="house (registered the recordings)")
    ax.scatter([], [], s=9, c=GHOST, label="ghost-like name")
    ax.scatter([], [], s=9, c=REAL, label="real artist")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=6.5, handletextpad=0.1,
              columnspacing=1.0)
    ax.text(0.5, -0.09, f"Each name sits around the house that registered most of its tracks. "
            f"Not drawn: {len(small)} houses with one or two names ({small.map(len).sum()} names).",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.5, color=TEXT_2)


def panel_bars(ax) -> None:
    lf = json.loads((PROCESSED / "ownership.json").read_text())["label_free"]
    cols = [("functional", "Mood\nplaylists"), ("genre", "Genre\nplaylists")]
    parts = [("its 8 largest houses", HUB, "white"), ("all other houses", NEUTRAL, TEXT),
             ("no recording code", "#e4e3df", TEXT_2)]
    for x, (k, _) in enumerate(cols):
        d = lf[k]
        vals = [d["top8_share_of_all"], d["with_code"] - d["top8_share_of_all"], 1 - d["with_code"]]
        bottom = 1.0
        for v, (lab, fc, tc) in zip(vals, parts):
            ax.bar(x, v, bottom=bottom - v, width=0.62, color=fc, ec="white", lw=0.6)
            if v > 0.05:
                ax.text(x, bottom - v / 2, f"{v:.0%}", ha="center", va="center", fontsize=6.5, color=tc)
            bottom -= v
    for lab, fc, _ in parts:
        ax.bar(0, 0, color=fc, label=lab)
    ax.set_xticks(range(len(cols)), [lab for _, lab in cols], fontsize=6.5)
    ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"], fontsize=6.5)
    ax.set_ylim(0, 1)
    ax.set_xlim(-0.5, 1.5)
    ax.grid(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), fontsize=6.5, handlelength=1.0,
              handletextpad=0.4, labelspacing=0.3)
    ax.set_title("B  Share of slots,\nby who registered\nthe track", loc="left")


def main() -> None:
    style.apply()
    fig = plt.figure(figsize=(7.08, 3.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.4, 0.62], wspace=0.55)
    panel_network(fig.add_subplot(gs[0, 0]))
    panel_bars(fig.add_subplot(gs[0, 1]))
    style.save(fig, "fig3_ownership")


if __name__ == "__main__":
    main()
