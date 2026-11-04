"""Figure 2: many faces, few hands.

A  Ownership network. Every dark node is an ISRC registrant ("house"); every dot around it is
   an artist name whose tracks it registered (dominant registrant). Disc area ~ number of
   names. Faint arcs: names that also have tracks registered by a second house.
B  House size seen from each name (CCDF), ghost-like vs real.
C  Cumulative share of functional-playlist slots covered by the k largest houses.

    python -m src.figures.fig2_ownership
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.path import Path
from matplotlib.patches import PathPatch

from src.common import placement
from src.features.production import track_houses
from src.figures import style
from src.figures.netlayout import pack_discs, vogel
from src.figures.style import GHOST, REAL, TEXT, TEXT_2
from src.load import PROCESSED

HUB = "#1f1f1e"


def house_label(t: pd.DataFrame) -> dict[str, str]:
    out = {}
    for reg, d in t.dropna(subset=["registrant"]).groupby("registrant"):
        s = d.loc[~d.self_source, "source_raw"].value_counts()
        out[reg] = s.index[0] if len(s) and s.iloc[0] >= 0.15 * len(d) else "names label themselves"
    return out


def panel_network(ax) -> None:
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    B = pd.read_csv(PROCESSED / "ownership_edges.csv", dtype={"primary_artist": str, "registrant": str})
    t = track_houses()
    lab = house_label(t)
    # dominant house per name (most tracks), restricted to names in B
    dom = B.sort_values("n_tracks", ascending=False).drop_duplicates("primary_artist")
    dom = dom.set_index("primary_artist").registrant
    members = dom.groupby(dom).apply(lambda s: list(s.index))
    members = members.loc[members.map(len).sort_values(ascending=False).index]
    houses = members.index.tolist()
    sizes = members.map(len).to_numpy()

    spacing = 1.0
    radii = spacing * np.sqrt(sizes + 1.6 ** 2) + 0.4
    n_lab = 10
    pads = np.where(sizes >= 20, 3.0, 0.9)
    centers = pack_discs(radii + pads / 2, pad=0.6, step=0.03)
    hpos = dict(zip(houses, centers))

    name_xy, name_ghost = {}, {}
    for h, names in members.items():
        p = df.p_ghost.reindex(names).fillna(0).sort_values(ascending=False)
        xy = vogel(len(p), spacing) + hpos[h]
        for n, q in zip(p.index, xy):
            name_xy[n] = q
            name_ghost[n] = bool(df.ghost.get(n, False))

    # spokes
    segs = [(hpos[dom[n]], xy) for n, xy in name_xy.items()]
    ax.add_collection(LineCollection(segs, colors="#d6d5cf", linewidths=0.12, zorder=1))
    # cross-house arcs
    sec = B.merge(dom.rename("dom"), left_on="primary_artist", right_index=True)
    sec = sec[sec.registrant != sec.dom]
    for n, h in zip(sec.primary_artist, sec.registrant):
        if h not in hpos:  # a house that is only ever secondary has no disc
            continue
        a, b = name_xy[n], hpos[h]
        mid = (a + b) / 2 + np.array([-(b - a)[1], (b - a)[0]]) * 0.18
        ax.add_patch(PathPatch(Path([a, mid, b], [Path.MOVETO, Path.CURVE3, Path.CURVE3]),
                               fc="none", ec="#3d3c38", lw=0.22, alpha=0.18, zorder=1))
    xy = np.array(list(name_xy.values()))
    g = np.array([name_ghost[n] for n in name_xy])
    ax.scatter(xy[~g, 0], xy[~g, 1], s=1.5, c=REAL, lw=0, zorder=2, rasterized=True)
    ax.scatter(xy[g, 0], xy[g, 1], s=1.5, c=GHOST, lw=0, zorder=2, rasterized=True)
    c = np.array([hpos[h] for h in houses])
    ax.scatter(c[:, 0], c[:, 1], s=3 + 0.1 * sizes, c=HUB, lw=0.4, ec="white", zorder=3)
    # labels outside the cloud, in a left and a right column, with leader lines
    allxy = np.vstack([xy, c])
    xmin, xmax = allxy[:, 0].min(), allxy[:, 0].max()
    cx = np.median(allxy[:, 0])
    top = list(range(n_lab))
    for side in ("left", "right"):
        idx = [i for i in top if (centers[i][0] < cx) == (side == "left")]
        idx.sort(key=lambda i: -centers[i][1])
        ys = [centers[i][1] for i in idx]
        gap = 17.0  # data units between label rows
        for k in range(1, len(ys)):  # push down to keep a minimum gap
            ys[k] = min(ys[k], ys[k - 1] - gap)
        shift = (np.mean([centers[i][1] for i in idx]) - np.mean(ys)) if idx else 0
        ys = [y + shift for y in ys]
        lx = xmin - 4 if side == "left" else xmax + 4
        for i, ly in zip(idx, ys):
            h = houses[i]
            x, y = centers[i]
            ex = x - radii[i] if side == "left" else x + radii[i]
            ax.plot([x, ex, lx], [y, y, ly], color=TEXT_2, lw=0.45, zorder=4)
            ax.scatter([x], [y], s=6, c="white", zorder=4, lw=0)
            name = lab[h] if len(lab[h]) <= 24 else lab[h][:23] + "…"
            ax.text(lx + (-1 if side == "left" else 1), ly, f"{h} · {sizes[i]} names\n{name}",
                    ha="right" if side == "left" else "left", va="center", fontsize=5.3,
                    color=TEXT, linespacing=1.2, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(xmin - 62, xmax + 62)
    ax.set_ylim(allxy[:, 1].min() - 4, allxy[:, 1].max() + 4)
    ax.axis("off")
    ax.set_title("A  Who registered the recordings", loc="left")
    ax.scatter([], [], s=9, c=GHOST, label="ghost-like name")
    ax.scatter([], [], s=9, c=REAL, label="real artist")
    ax.scatter([], [], s=16, c=HUB, label="ISRC house")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=5.8, handletextpad=0.1)


def panel_ccdf(ax) -> None:
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    own = json.loads((PROCESSED / "ownership.json").read_text())
    s = df.dropna(subset=["registrant_house_size"])
    for lab, m, c in [("real artists", ~s.ghost, REAL), ("ghost-like names", s.ghost, GHOST)]:
        x = np.sort(s.loc[m, "registrant_house_size"].to_numpy())
        y = 1 - np.arange(len(x)) / len(x)
        ax.step(x, y, where="post", color=c, lw=1.6, label=f"{lab} ({len(x):,})")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.03)
    ax.set_xlabel("Names registered by the same house")
    ax.set_ylabel("Share of names in a\nhouse at least this big")
    ax.set_title("B  House size, seen from a name", loc="left")
    ax.legend(loc="lower left", fontsize=5.6, bbox_to_anchor=(0.04, 0.0), handlelength=1.2)
    ax.annotate(f"median {own['median_house_ghost']:.0f}", xy=(own["median_house_ghost"], 0.5),
                xytext=(18, 0.62), fontsize=5.6, color=TEXT_2,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.5))
    ax.annotate(f"median {own['median_house_real']:.0f}", xy=(own["median_house_real"], 0.5),
                xytext=(1.6, 0.22), fontsize=5.6, color=TEXT_2,
                arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.5))


def panel_concentration(ax) -> None:
    own = json.loads((PROCESSED / "ownership.json").read_text())
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    pa = placement()
    pa = pa[pa.is_primary & (pa.group == "functional")]
    for lab, want, c in [("real slots", False, REAL), ("ghost-like slots", True, GHOST)]:
        reg = pa.artist_id.map(df.registrant_dom)[pa.artist_id.map(df.ghost) == want].dropna()
        vc = reg.value_counts().to_numpy()
        cum = np.cumsum(vc) / vc.sum()
        ax.plot(np.arange(1, len(cum) + 1), cum, color=c, lw=1.6, label=lab, drawstyle="steps-post")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 400)
    ax.set_ylim(0, 1.03)
    ax.axvline(8, color=TEXT_2, lw=0.6, ls=":")
    ax.text(14, 0.08, f"8 houses hold\n{own['top8_share_of_ghost_slots']:.0%} of ghost-like slots",
            fontsize=5.6, color=TEXT_2)
    ax.set_xlabel("Number of houses, largest first")
    ax.set_ylabel("Cumulative share of\nfunctional playlist slots")
    ax.set_title("C  Few hands", loc="left")
    ax.legend(loc="lower right", fontsize=5.6, bbox_to_anchor=(1.0, 0.24), handlelength=1.2)


def main() -> None:
    style.apply()
    fig = plt.figure(figsize=(7.08, 3.7))
    gs = fig.add_gridspec(2, 2, width_ratios=[2.5, 1.0], wspace=0.34, hspace=0.62)
    panel_network(fig.add_subplot(gs[:, 0]))
    panel_ccdf(fig.add_subplot(gs[0, 1]))
    panel_concentration(fig.add_subplot(gs[1, 1]))
    style.save(fig, "fig2_ownership")


if __name__ == "__main__":
    main()
