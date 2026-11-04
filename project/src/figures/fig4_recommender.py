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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from src.common import placement
from src.figures import style
from src.figures.style import GHOST, NEUTRAL, REAL, TEXT, TEXT_2
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
    ax.add_collection(LineCollection(segs, colors="#9f9e98", linewidths=0.08, alpha=0.12,
                                     zorder=1, rasterized=True))
    order = np.argsort(ghost)  # real first, ghosts on top
    ax.scatter(xy[order, 0], xy[order, 1], s=1.3, c=np.where(ghost[order], GHOST, REAL), lw=0,
               zorder=2, rasterized=True)
    # label the largest communities in two outer columns with leader lines
    pa = placement()
    home = pa.groupby("artist_id").name.agg(lambda s: s.value_counts().index[0])
    lab = pd.DataFrame({"c": cvec, "x": xy[:, 0], "y": xy[:, 1], "g": ghost,
                        "home": home.reindex(names).to_numpy()})
    stats = []
    for c in lab.c.value_counts().index[:10]:
        d = lab[lab.c == c]
        top = d.home.value_counts()
        txt = top.index[0] if top.iloc[0] / len(d) > 0.35 else f"{top.index[0]} +"
        stats.append((d.x.median(), d.y.median(), f"{txt}\n{len(d)} names · {d.g.mean():.0%} ghost-like"))
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    cx = np.median(xy[:, 0])
    span = xy[:, 1].max() - xy[:, 1].min()
    for side in ("left", "right"):
        S = sorted([t for t in stats if (t[0] < cx) == (side == "left")], key=lambda t: -t[1])
        ys = [t[1] for t in S]
        for k in range(1, len(ys)):
            ys[k] = min(ys[k], ys[k - 1] - span * 0.155)
        if S:
            shift = np.mean([t[1] for t in S]) - np.mean(ys)
            ys = [y + shift for y in ys]
        lx = xmin - span * 0.04 if side == "left" else xmax + span * 0.04
        for (x, y, txt), ly in zip(S, ys):
            ax.plot([x, lx], [y, ly], color=TEXT_2, lw=0.45, zorder=3)
            ax.text(lx + (-0.5 if side == "left" else 0.5), ly, txt, fontsize=4.8,
                    ha="right" if side == "left" else "left", va="center", color=TEXT, zorder=4,
                    linespacing=1.15)
    ax.set_xlim(xmin - span * 0.36, xmax + span * 0.36)
    ax.set_aspect("equal", adjustable="datalim")
    ax.axis("off")
    ax.scatter([], [], s=9, c=GHOST, label="ghost-like name")
    ax.scatter([], [], s=9, c=REAL, label="real artist")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=5.8, handletextpad=0.1)
    ax.set_title("A  'Fans also like' among the 5,024 names", loc="left")


def panel_nulls(ax) -> None:
    r = json.loads((PROCESSED / "recommendation.json").read_text())["nulls"]
    rows = [("real to ghost-like", 1), ("ghost-like to real", 2), ("ghost-like to ghost-like", 3),
            ("real to real", 0)]
    labels = [lab for lab, _ in rows] + ["same ISRC house"]
    y = np.arange(len(labels))[::-1]
    for k, (null, marker, fc, dy) in enumerate([("degree", "o", "white", 0.13),
                                                ("playlist", "o", TEXT, -0.13)]):
        vals = [r[null]["flows_ratio"][i] for _, i in rows] + [r[null]["same_ratio"]]
        zs = [r[null]["flows_z"][i] for _, i in rows] + [r[null]["same_z"]]
        ax.scatter(vals, y + dy, s=16, marker=marker, facecolor=fc, edgecolor=TEXT, lw=0.8, zorder=3,
                   label={"degree": "null: degrees kept", "playlist": "null: degrees + playlists kept"}[null])
        for v, yy in zip(vals, y + dy):
            ax.plot([1, v], [yy, yy], color=NEUTRAL, lw=0.7, zorder=2)
    ax.axvline(1, color=TEXT_2, lw=0.6)
    ax.set_xscale("log")
    ax.set_xlim(0.42, 3.6)
    ax.set_xticks([0.5, 1, 2], ["0.5×", "1×", "2×"])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_yticks(y, labels, fontsize=5.8)
    ax.set_ylim(-0.6, len(labels) - 0.4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Observed / expected edges")
    ax.text(0.46, -0.45, "fewer than chance", fontsize=5, color=TEXT_2)
    ax.text(3.4, -0.45, "more than chance", fontsize=5, color=TEXT_2, ha="right")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, 1.13), fontsize=5.3, ncol=2,
              handletextpad=0.1, columnspacing=0.6)
    ax.set_title("B  Who is recommended next to whom", loc="left", pad=14)


def panel_nmi(ax) -> None:
    c = json.loads((PROCESSED / "recommendation.json").read_text())["communities"]
    items = list(c["nmi"].items())
    y = np.arange(len(items))[::-1]
    obs = [v["nmi"] for _, v in items]
    nul = [v["null_mean"] for _, v in items]
    ax.barh(y, obs, color="#3d3c38", height=0.55, label="observed")
    ax.barh(y, nul, color="white", height=0.55, ec="#3d3c38", lw=0.6, hatch="//////", label="shuffled null")
    for yi, o in zip(y, obs):
        ax.text(o + 0.02, yi, f"{o:.2f}", va="center", fontsize=5.6, color=TEXT)
    ax.set_yticks(y, [k for k, _ in items], fontsize=5.8)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("NMI with recommender communities")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=5.3)
    ax.set_title(f"C  Communities follow playlists", loc="left")
    ax.text(0.98, 0.6, f"Louvain Q = {c['modularity']:.2f}\n(rewired: {c['modularity_null_mean']:.2f})",
            transform=ax.transAxes, ha="right", va="center", fontsize=5.3, color=TEXT_2)


def main() -> None:
    style.apply()
    fig = plt.figure(figsize=(7.08, 3.05))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.45, 1.0],
                          wspace=0.75, hspace=0.75)
    panel_network(fig.add_subplot(gs[:, 0]))
    panel_nulls(fig.add_subplot(gs[0, 1]))
    panel_nmi(fig.add_subplot(gs[1, 1]))
    style.save(fig, "fig4_recommender")


if __name__ == "__main__":
    main()
