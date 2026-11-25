"""Figures for version 2 of the 3-slide talk (slides/build_deck_v2.js).

Same data as slides.py, restyled for the deck rules in not-ai-decks-guide: one accent colour
(orange = ghost-like) for the point, grey for context, near-black for text, Arial only,
regular and bold. Drawn at the size they get on the slide, so font sizes are what people see.
Output: slides/figures_v2/*.png (300 dpi).

    python -m src.figures.slides_v2
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patheffects as pe
from matplotlib.collections import LineCollection

from src.figures.fig1_data import slots
from src.figures.houses import company, display_names
from src.figures.netlayout import pack_discs, vogel
from src.load import PROCESSED

OUT = Path(__file__).resolve().parents[2] / "slides" / "figures_v2"
ACCENT, INK, GREY, LIGHT, FAINT = "#e0592a", "#1f2328", "#77766f", "#c9c8c2", "#ecebe7"
BODY, SMALL = 16, 12  # the deck's body and caption sizes


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": BODY, "text.color": INK, "axes.edgecolor": GREY, "axes.labelcolor": GREY,
        "xtick.color": GREY, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"  wrote slides/figures_v2/{name}.png")


def s1_slots() -> None:
    """Share of slots: ghost-like (accent) vs real (grey), mood vs genre playlists."""
    pa = slots()
    share = pa.groupby("group").label.value_counts(normalize=True).unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(7.6, 2.7))
    fig.subplots_adjust(left=0.25, right=0.99, top=0.98, bottom=0.02)
    rows = [("functional", "Mood playlists\n45 playlists"), ("genre", "Genre playlists\n5, as a control")]
    for y, (g, lab) in zip([1, 0], rows):
        gh = share.loc[g, "ghost"]
        ax.barh(y, gh, color=ACCENT, height=0.66, lw=0)
        ax.barh(y, 1 - gh, left=gh, color=LIGHT, height=0.66, lw=0)
        if gh > 0.2:
            ax.text(0.03, y, f"{gh:.0%} ghost-like", ha="left", va="center", color="white", fontsize=22,
                    weight="bold")
            ax.text(0.97, y, f"{1 - gh:.0%} real", ha="right", va="center", color=INK, fontsize=BODY)
        else:
            ax.text(gh + 0.03, y, f"{gh:.0%} ghost-like, {1 - gh:.0%} real", ha="left", va="center",
                    color=INK, fontsize=BODY)
        ax.text(-0.03, y, lab, ha="right", va="center", fontsize=BODY, color=INK, linespacing=1.25)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 1.5)
    ax.set_axis_off()
    save(fig, "s1_slots")


def s2_houses() -> None:
    """Names around the house that registered their recordings; ghost-like names in the accent."""
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    dom = df.registrant_dom.dropna()
    members = dom.groupby(dom).apply(lambda s: list(s.index))
    members = members.loc[members.map(len).sort_values(ascending=False).index]
    members = members[members.map(len) >= 3]
    houses = members.index.tolist()
    sizes = members.map(len).to_numpy()
    radii = np.sqrt(sizes + 1.6 ** 2) + 0.4
    pads = np.where(sizes >= 20, 3.0, 0.9)
    centers = pack_discs(radii + pads / 2, pad=0.6, step=0.03)
    hpos = dict(zip(houses, centers))

    fig, ax = plt.subplots(figsize=(7.7, 4.7))
    fig.subplots_adjust(0, 0, 1, 1)
    segs, pts, gh = [], [], []
    for h, names in members.items():
        g = df.ghost.reindex(names).fillna(False).astype(bool).sort_values(ascending=False)
        for n, q in zip(g.index, vogel(len(g)) + hpos[h]):
            segs.append((hpos[h], q))
            pts.append(q)
            gh.append(bool(g[n]))
    pts, gh = np.array(pts), np.array(gh)
    ax.add_collection(LineCollection(segs, colors=FAINT, linewidths=0.25, zorder=1))
    ax.scatter(pts[~gh, 0], pts[~gh, 1], s=5, c=LIGHT, lw=0, zorder=2)
    ax.scatter(pts[gh, 0], pts[gh, 1], s=5, c=ACCENT, lw=0, zorder=2)
    c = np.array(centers)
    ax.scatter(c[:, 0], c[:, 1], s=8 + 0.15 * sizes, c=INK, lw=0.6, ec="white", zorder=3)

    comp, disp = company(), display_names()
    press = {"Overtone Studios": "Overtone Studios", "Firefly Entertainment AB": "Firefly"}
    for h in disp:
        i = houses.index(h)
        x, y = centers[i]
        lab = f"{sizes[i]} names"
        if comp[h] in press:
            lab += f"\n{press[comp[h]]}"
        ax.text(x, y - radii[i] * 0.42, lab, ha="center", va="center", fontsize=SMALL, weight="bold",
                color=INK, linespacing=1.1, zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")])
    allxy = np.vstack([pts, c])
    ax.set_xlim(allxy[:, 0].min() - 2, allxy[:, 0].max() + 2)
    ax.set_ylim(allxy[:, 1].min() - 2, allxy[:, 1].max() + 2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    save(fig, "s2_houses")


def s3_links() -> None:
    """'Fans also like' links between real and ghost-like names, against two kinds of chance."""
    r = json.loads((PROCESSED / "recommendation.json").read_text())["nulls"]
    rows = [("Real artist links\nto a ghost-like name", 1), ("Ghost-like name\nlinks to a real artist", 2)]
    nulls = [("degree", "Compared with random links", LIGHT, INK),
             ("playlist", "Compared with random links inside the same playlist", INK, "white")]
    fig, ax = plt.subplots(figsize=(6.3, 3.3))
    fig.subplots_adjust(left=0.37, right=0.97, top=0.74, bottom=0.03)
    for yi, (lab, i) in zip([1, 0], rows):
        for k, (null, _, col, tcol) in enumerate(nulls):
            v = r[null]["flows_ratio"][i]
            yy = yi + (0.18 if k == 0 else -0.18)
            ax.barh(yy, v, height=0.32, color=col, lw=0)
            ax.text(v - 0.03, yy, f"{v:.2f}×", va="center", ha="right", fontsize=SMALL, weight="bold", color=tcol)
        ax.text(-0.03, yi, lab, ha="right", va="center", fontsize=SMALL, linespacing=1.2)
    ax.axvline(1, color=GREY, lw=1.2, ls=(0, (3, 2)))
    ax.text(1, 1.42, "1× = chance", ha="center", va="bottom", fontsize=SMALL, color=GREY)
    for _, lab, col, _ in nulls:
        ax.scatter([], [], marker="s", s=110, color=col, label=lab)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.6, 1.1), ncol=1, frameon=False, fontsize=SMALL,
              handletextpad=0.3, labelspacing=0.3, borderaxespad=0)
    ax.set_xlim(0, 1.12)
    ax.set_ylim(-0.42, 1.42)
    ax.set_axis_off()
    save(fig, "s3_links")


def s3_auc() -> None:
    """How well each thing a listener sees predicts the flag (ROC AUC); only the bio stands out."""
    t = json.loads((PROCESSED / "text.json").read_text())
    v = json.loads((PROCESSED / "vision.json").read_text())
    a = json.loads((PROCESSED / "audio.json").read_text())
    # text rows (TF-IDF + logistic regression) are named as such; sound and covers are not NLP
    rows = [("Sound, audio features", a["auc_audio"]["auc"]), ("Cover art, image model", v["auc_cover"]["auc"]),
            ("Artist name, TF-IDF", t["auc_name"]["auc"]), ("Track titles, TF-IDF", t["auc_titles"]["auc"]),
            ("Bio, TF-IDF", t["auc_bio"]["auc"])]
    fig, ax = plt.subplots(figsize=(5.4, 2.75))
    fig.subplots_adjust(left=0.4, right=0.93, top=0.9, bottom=0.22)
    ax.axvspan(0.45, 0.66, color=FAINT, lw=0, zorder=0)
    y = np.arange(len(rows))[::-1] * 0.85
    ax.text(0.555, y[0] + 0.6, "close to guessing", ha="center", va="center", fontsize=SMALL, color=GREY)
    for yi, (lab, val) in zip(y, rows):
        key = lab.startswith("Bio")
        col = INK if key else "#9a9993"
        ax.plot([0.5, val], [yi, yi], color=col, lw=2.2, zorder=2, solid_capstyle="butt")
        ax.scatter([val], [yi], s=70, color=col, zorder=3, lw=0)
        ax.text(val + 0.02, yi, f"{val:.2f}", va="center", fontsize=SMALL, color=col,
                weight="bold" if key else None)
    ax.set_yticks(y, [lab for lab, _ in rows], fontsize=SMALL)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.5, y[0] + 0.85)
    ax.set_xticks([0.5, 0.75, 1.0], ["0.5\nguessing", "0.75", "1.0\nperfect"], fontsize=SMALL)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3)
    ax.spines["left"].set_visible(False)
    save(fig, "s3_auc")


def main() -> None:
    setup()
    s1_slots()
    s2_houses()
    s3_links()
    s3_auc()


if __name__ == "__main__":
    main()
