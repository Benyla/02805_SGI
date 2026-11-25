"""Figure 4: where the recommender puts ghost-like names.

A  "Fans also like" network among the dataset artists (undirected view, giant component,
   Fruchterman-Reingold). Colour = label; text = the playlist most common in each large
   Louvain community.
B  Observed / expected edge counts under two nulls (degree-preserving; degree + target's home
   playlist preserved). 1,000 samples each.
C  NMI between Louvain communities and three partitions, vs a label-shuffle null.

    python -m src.figures.fig4_recommender
"""
from __future__ import annotations

import json
import random

import igraph as ig
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src.common import placement
from src.figures import style
from src.figures.style import GHOST, GRID, NEUTRAL, REAL, TEXT, TEXT_2
from src.load import PROCESSED, SEED, load


def group_in_a_box(g: ig.Graph, cvec: np.ndarray) -> np.ndarray:
    """Each community laid out on its own (FR), scaled to radius ~ sqrt(size); community
    centres from an FR layout of the community graph, then pushed apart until discs
    (plus room for a label below) do not overlap."""
    ig.set_random_number_generator(random.Random(SEED))
    comms = pd.Series(range(len(cvec))).groupby(cvec).apply(list)
    local, radius = {}, {}
    for c, nodes in comms.items():
        if len(nodes) == 1:
            local[c] = np.zeros((1, 2))
        else:
            sub = g.induced_subgraph(nodes)
            q = np.array(sub.layout("fr", niter=800).coords)
            q -= np.median(q, 0)
            d = np.hypot(q[:, 0], q[:, 1])
            r80 = max(np.percentile(d, 80), 1e-9)
            # radial squash: every node ends up inside the disc of radius 0.55*sqrt(n)
            scale = 0.55 * np.sqrt(len(nodes)) * np.tanh(d / r80) / np.maximum(d, 1e-9)
            q = q * scale[:, None]
            local[c] = q
        radius[c] = 0.55 * np.sqrt(len(nodes)) + 0.4
    ids = list(comms.index)
    cix = {c: i for i, c in enumerate(ids)}
    E = np.array(g.get_edgelist())
    a, b = cvec[E[:, 0]], cvec[E[:, 1]]
    w = pd.Series(1, index=pd.MultiIndex.from_arrays([np.minimum(a, b), np.maximum(a, b)])).groupby(level=[0, 1]).sum()
    w = w[[i != j for i, j in w.index]]
    cg = ig.Graph(n=len(ids), edges=[(cix[i], cix[j]) for i, j in w.index])
    P = np.array(cg.layout("fr", weights=np.log1p(w.to_numpy()).tolist(), niter=3000).coords)
    r = np.array([radius[c] for c in ids])
    lab_pad = np.zeros_like(r)
    P = P / np.median(np.hypot(*(P - P.mean(0)).T)) * r.sum() / 6
    for _ in range(4000):  # relaxation: push overlapping discs apart, pull gently to centre
        D = P[:, None, :] - P[None, :, :]
        dist = np.hypot(D[..., 0], D[..., 1]) + 1e-9
        need = r[:, None] + r[None, :] + 1.6 + np.maximum(lab_pad[:, None], lab_pad[None, :]) * \
            (np.abs(D[..., 1]) > np.abs(D[..., 0]))
        over = np.clip(need - dist, 0, None)
        np.fill_diagonal(over, 0)
        if over.max() < 1e-3:
            break
        P += 0.5 * (over[..., None] * D / dist[..., None]).sum(1) / 2
        P -= 0.002 * (P - P.mean(0))
    xy = np.zeros((len(cvec), 2))
    for c, nodes in comms.items():
        xy[nodes] = local[c] + P[cix[c]]
    return xy


def panel_network(ax) -> None:
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    e = load("fans_also_like_edges")
    e = e[e.source_artist_id.isin(df.index) & e.target_artist_id.isin(df.index)]
    ids = sorted(set(e.source_artist_id) | set(e.target_artist_id))
    ix = {a: i for i, a in enumerate(ids)}
    g = ig.Graph(n=len(ids), edges=list(zip(e.source_artist_id.map(ix), e.target_artist_id.map(ix))),
                 directed=False).simplify()
    g.vs["name"] = ids
    giant = g.components().giant()
    names = giant.vs["name"]
    comm = pd.read_csv(PROCESSED / "fal_communities.csv", index_col=0).community
    cvec = comm.reindex(names).to_numpy()
    path = PROCESSED / "fig4_layout.npy"
    if path.exists() and len(np.load(path)) == giant.vcount():
        xy = np.load(path)
    else:
        xy = group_in_a_box(giant, cvec)
        np.save(path, xy)
    ghost = df.ghost.reindex(names).to_numpy(bool)
    E = np.array(giant.get_edgelist())
    segs = np.stack([xy[E[:, 0]], xy[E[:, 1]]], 1)
    ax.add_collection(LineCollection(segs, colors="#9f9e98", linewidths=0.1, alpha=0.12,
                                     zorder=1, rasterized=True))
    order = np.argsort(ghost)  # real first, ghosts on top
    ax.scatter(xy[order, 0], xy[order, 1], s=2.2, c=np.where(ghost[order], GHOST, REAL), lw=0,
               zorder=2, rasterized=True)
    # label the largest communities in two outer columns with leader lines
    pa = placement()
    home = pa.groupby("artist_id").name.agg(lambda s: s.value_counts().index[0])
    lab = pd.DataFrame({"c": cvec, "x": xy[:, 0], "y": xy[:, 1], "g": ghost,
                        "home": home.reindex(names).to_numpy()})
    # label the largest clusters directly on top of them (white halo keeps text readable)
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    pad = 0.015 * (xmax - xmin)
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_aspect("equal", adjustable="box")
    # number the largest clusters on the network and explain the numbers in a key below
    key = []
    # the 8 largest clusters, numbered by share of ghost-like names (highest first)
    top8 = lab.c.value_counts().index[:8]
    top8 = sorted(top8, key=lambda c: -lab.loc[lab.c == c, "g"].mean())
    for k, c in enumerate(top8, start=1):
        d = lab[lab.c == c]
        name = d.home.value_counts().index[0].strip()
        ax.scatter([d.x.median()], [d.y.median()], s=58, facecolor="white", edgecolor=TEXT, lw=0.7, zorder=5)
        ax.text(d.x.median(), d.y.median(), str(k), fontsize=6, fontweight="bold", ha="center", va="center",
                color=TEXT, zorder=6)
        key.append(f"$\\bf{{{k}}}$  {name} ({d.g.mean():.0%})")
    for k, line in enumerate(key):
        col, row = divmod(k, 4)
        ax.text(-0.02 + 0.7 * col, -0.06 - 0.034 * row, line, transform=ax.transAxes, ha="left", va="top",
                fontsize=6.1, color=TEXT)
    ax.axis("off")
    ax.scatter([], [], s=9, c=GHOST, label="ghost-like name")
    ax.scatter([], [], s=9, c=REAL, label="real artist")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.03), ncol=2, fontsize=6.5, handletextpad=0.1)
    ax.set_title("A  Each cluster of similar artists is a playlist", loc="left")
    ax.text(0.5, -0.205, "Nodes are artist names, edges are 'Fans also like' links (largest connected group).\n"
            "Each cluster is named after the most common home playlist of its names (the playlist\n"
            "where a name has the most slots). % = share of ghost-like names in the cluster.",
            transform=ax.transAxes, ha="center", va="top", fontsize=6.2, color=TEXT_2, linespacing=1.25)


def panel_nulls(ax) -> None:
    r = json.loads((PROCESSED / "recommendation.json").read_text())["nulls"]
    rows = [("Real artist recommends\na ghost-like name", "flows", 1),
            ("Ghost-like name\nrecommends a real artist", "flows", 2),
            ("Two names from the same\nhouse recommend each other", "same", None)]
    y = np.array([2.3, 1.15, 0.0])  # gap before the "same house" row
    lo = min(min(r[n]["flows_null_range"][i][0] for n in ("degree", "playlist") for i in (1, 2)),
             min(r[n]["same_null_range"][0] for n in ("degree", "playlist")))
    hi = max(max(r[n]["flows_null_range"][i][1] for n in ("degree", "playlist") for i in (1, 2)),
             max(r[n]["same_null_range"][1] for n in ("degree", "playlist")))
    # The 95% range of the shuffles is only a few percent wide, too narrow to draw on this
    # axis, so it is stated in words next to the 1x line instead of shown as a band.
    chance = max(abs(lo - 1), abs(hi - 1))
    ax.axvline(1, color=TEXT_2, lw=0.6, zorder=1)
    for (lab, kind, i), yi in zip(rows, y):
        for null, fc, dy in [("degree", "white", 0.2), ("playlist", TEXT, -0.2)]:
            v = r[null]["flows_ratio"][i] if kind == "flows" else r[null]["same_ratio"]
            ax.plot([1, v], [yi + dy, yi + dy], color=NEUTRAL, lw=0.8, zorder=2)
            ax.scatter([v], [yi + dy], s=22, facecolor=fc, edgecolor=TEXT, lw=0.8, zorder=3)
            # value labels never sit between points: right of the point if above 1x,
            # otherwise just right of the 1x line (the stem leads the eye back to the point)
            ax.text(v * 1.07 if v > 1 else hi * 1.05, yi + dy, f"{v:.2f}×", va="center", ha="left",
                    fontsize=6.5, color=TEXT)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 5.0)
    ax.set_xticks([0.5, 1, 2, 4], ["0.5×", "1×", "2×", "4×"], fontsize=6.5)
    ax.set_xlabel("Observed links ÷ links expected by chance", fontsize=6.5)
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks(y, [lab for lab, _, _ in rows], fontsize=6.5)
    ax.set_ylim(-1.95, 3.1)
    ax.grid(False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    # direction guide above the plot
    ax.text(0.94, 2.95, "$\\leftarrow$ rarer than chance", ha="right", va="center", fontsize=6.3, color=TEXT_2)
    ax.text(1.06, 2.95, "more common than chance $\\rightarrow$", ha="left", va="center", fontsize=6.3, color=TEXT_2)
    ax.scatter([], [], s=22, facecolor="white", edgecolor=TEXT, lw=0.8, label="vs shuffle 1")
    ax.scatter([], [], s=22, facecolor=TEXT, edgecolor=TEXT, lw=0.8, label="vs shuffle 2")
    ax.text(0.93, -1.25, f"Chance varies only\n±{chance:.0%} around 1×,\ntoo little to show", ha="right",
            va="center", fontsize=6, color=TEXT_2, linespacing=1.15)
    # compact legend in the empty lower-right corner, below every data row (as in panel D)
    ax.legend(loc="lower right", bbox_to_anchor=(1.03, -0.02), ncol=1, fontsize=6.2, handletextpad=0.4,
              labelspacing=0.3, borderaxespad=0.2)
    ax.set_title("C  Which recommendations happen more than chance", loc="left")


def panel_explainer(ax) -> None:
    """Three tiny diagrams: observed links, shuffle 1 (any target), shuffle 2 (same playlist only)."""
    ax.set_xlim(0, 9.3)
    ax.set_ylim(-0.12, 1.12)
    ax.axis("off")
    # two playlists, three names each (orange = ghost-like, blue = real)
    pts = np.array([[0.12, 0.75], [0.38, 0.4], [0.12, 0.08], [0.72, 0.78], [0.95, 0.42], [0.7, 0.08]])
    col = [GHOST] * 3 + [REAL] * 3
    obs = [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3), (1, 4)]
    # Both shuffles keep every name's out-degree (sources fixed) AND in-degree (targets are a
    # permutation of the observed targets: node 4 is hit twice, every other node once).
    # Shuffle 2 also keeps each link pointing into the same playlist (A targets {0,1,2},
    # B targets {3,4,4,5} are only permuted among themselves). No self-links.
    shuf1 = [(0, 4), (1, 5), (2, 1), (3, 0), (4, 2), (5, 3), (1, 4)]
    shuf2 = [(0, 2), (1, 0), (2, 1), (3, 4), (4, 3), (5, 4), (1, 5)]
    for E in (shuf1, shuf2):  # guard: same in- and out-degrees as the observed network
        assert sorted(s for s, _ in E) == sorted(s for s, _ in obs)
        assert sorted(t for _, t in E) == sorted(t for _, t in obs)
    heads = ["The real network", "Shuffle 1 (any name)", "Shuffle 2 (same playlist)"]
    texts = ["The actual 'Fans also like'\nlinks. Names mostly link to\nnames on their own playlist.",
             "Each name gives and receives\nas many links as before, but\nwho links to whom is random.\nThis is pure chance.",
             "As shuffle 1, but every link\nstill points into the same\nplaylist as before. This is\nwhat the playlists alone\nexplain."]
    for k, (E, head, txt) in enumerate(zip([obs, shuf1, shuf2], heads, texts)):
        ox = k * 3.15
        for x0 in (0.0, 0.6):
            ax.add_patch(plt.Rectangle((ox + x0 - 0.02, -0.06), 0.47, 1.0, fc="#f4f3f0", ec="none", zorder=0))
        for i, j in E:
            ax.annotate("", xy=pts[j] + [ox, 0], xytext=pts[i] + [ox, 0],
                        arrowprops=dict(arrowstyle="-|>", color="#6f6e69", lw=0.6, shrinkA=3, shrinkB=3,
                                        mutation_scale=5, connectionstyle="arc3,rad=0.15"))
        ax.scatter(pts[:, 0] + ox, pts[:, 1], s=14, c=col, zorder=3, lw=0.4, ec="white")
        ax.text(ox + 0.2, 0.97, "Playlist A", fontsize=5.8, color=TEXT_2, ha="center", va="bottom")
        ax.text(ox + 0.82, 0.97, "Playlist B", fontsize=5.8, color=TEXT_2, ha="center", va="bottom")
        ax.text(ox + 1.12, 1.0, head, ha="left", va="top", fontsize=6.5, color=TEXT, fontweight="bold")
        ax.text(ox + 1.12, 0.8, txt, ha="left", va="top", fontsize=6.3, color=TEXT, linespacing=1.2)
    ax.set_title("B  The two shuffles that define 'chance' in C", loc="left", pad=12)


def panel_nmi(ax) -> None:
    c = json.loads((PROCESSED / "recommendation.json").read_text())["communities"]
    items = list(c["nmi"].items())
    y = np.arange(len(items))[::-1]
    obs = [v["nmi"] for _, v in items]
    nul = [v["null_mean"] for _, v in items]
    h = 0.32
    ax.barh(y + h / 2 + 0.02, obs, color="#3d3c38", height=h, lw=0, label="real grouping")
    ax.barh(y - h / 2 - 0.02, nul, color="#c9c8c2", height=h, lw=0, label="shuffled (chance)")
    for yi, o, n in zip(y, obs, nul):
        ax.text(o + 0.015, yi + h / 2 + 0.02, f"{o:.2f}", va="center", fontsize=6.5, color=TEXT,
                fontweight="bold")
        ax.text(n + 0.015, yi - h / 2 - 0.02, f"{n:.2f}", va="center", fontsize=6, color=TEXT_2)
    ax.set_yticks(y, ["Home playlist", "House", "Ghost-like flag"], fontsize=6.5)
    ax.set_xlim(0, 0.8)
    ax.set_ylim(-1.15, 2.45)  # empty strip at the bottom holds the legend
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8])
    ax.set_xlabel("NMI between the clusters in A and each grouping\n(0 means no overlap, 1 means identical)",
                  fontsize=6.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower right", fontsize=6.2, bbox_to_anchor=(1.03, -0.02), labelspacing=0.3,
              borderaxespad=0.2)
    ax.set_title("D  NMI shows clusters follow playlists far more than houses", loc="left")


def main() -> None:
    style.apply()
    fig = plt.figure(figsize=(7.08, 5.7))
    # nested grids: the right column gets its own (tight) spacing between C and D, while the
    # gap above B leaves room for the key and caption under A
    outer = fig.add_gridspec(2, 1, height_ratios=[1.75, 0.4], hspace=0.62)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.7, 1.0], wspace=0.6)
    right = top[1].subgridspec(2, 1, height_ratios=[1.15, 0.85], hspace=0.62)
    panel_network(fig.add_subplot(top[0]))
    panel_nulls(fig.add_subplot(right[0]))
    panel_nmi(fig.add_subplot(right[1]))
    panel_explainer(fig.add_subplot(outer[1]))
    style.save(fig, "fig4_recommender")


if __name__ == "__main__":
    main()
