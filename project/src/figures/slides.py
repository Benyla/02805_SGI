"""Figures for the 3-slide talk, designed for slides rather than the paper.

Each figure shows one idea, with as little text as possible, and is drawn at the exact size it
gets on the 13.33 x 7.5 in slide, so the font sizes below are the sizes the audience sees.
Output: slides/figures/*.png (300 dpi).

    python -m src.figures.slides
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

OUT = Path(__file__).resolve().parents[2] / "slides" / "figures"
GHOST, REAL, INK, MUTED, FAINT, HUB = "#eb6834", "#2a78d6", "#1f2328", "#6b6a66", "#e6e5e1", "#1f1f1e"
NAVY = "#1c2434"


def setup() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 14, "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    })


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png", dpi=300, transparent=True)  # sits on tinted cards
    plt.close(fig)
    print(f"  wrote slides/figures/{name}.png")


# ------------------------------------------------------------------ slide 1
def s1_slots() -> None:
    """Two 100% bars: ghost-like vs real share of slots, mood vs genre playlists."""
    pa = slots()
    share = pa.groupby("group").label.value_counts(normalize=True).unstack().fillna(0)
    fig, ax = plt.subplots(figsize=(3.5, 1.6))
    fig.subplots_adjust(left=0.30, right=0.99, top=0.98, bottom=0.04)
    rows = [("functional", "Mood\nplaylists"), ("genre", "Genre\n(control)")]
    for y, (g, lab) in zip([1, 0], rows):
        gh = share.loc[g, "ghost"]
        ax.barh(y, gh, color=GHOST, height=0.62)
        ax.barh(y, 1 - gh, left=gh, color=REAL, height=0.62)
        if gh > 0.2:
            ax.text(gh / 2, y, f"{gh:.0%}", ha="center", va="center", color="white", fontsize=17, weight="bold")
        ax.text(gh + (1 - gh) / 2, y, f"{1 - gh:.0%}", ha="center", va="center", color="white", fontsize=15,
                weight="bold")
        ax.text(-0.04, y, lab, ha="right", va="center", fontsize=13, color=INK, linespacing=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 1.45)
    ax.set_axis_off()
    save(fig, "s1_slots")


def s1_check() -> None:
    """The flag on two groups it never saw: a leaked ghost catalogue and known real artists."""
    ref = pd.read_csv(PROCESSED / "convergence_reference_sets.csv").set_index("name")
    rows = [("Firefly Entertainment", "Leaked ghost\ncatalogue", GHOST),
            ("Genre-playlist artists", "Known real\nartists", REAL)]
    fig, ax = plt.subplots(figsize=(3.5, 1.45))
    fig.subplots_adjust(left=0.36, right=0.86, top=0.98, bottom=0.04)
    for y, (key, lab, c) in zip([1, 0], rows):
        s = ref.loc[key, "share_ghost"]
        ax.barh(y, 1, color="white", height=0.5)  # the card behind is grey
        ax.barh(y, s, color=c, height=0.5)
        ax.text(s + 0.03, y, f"{s:.0%}", va="center", ha="left", fontsize=15, weight="bold", color=c)
        ax.text(-0.05, y, lab, ha="right", va="center", fontsize=13, linespacing=1.0)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 1.45)
    ax.set_axis_off()
    save(fig, "s1_check")


# ------------------------------------------------------------------ slide 2
def s2_houses() -> None:
    """Names around the house that registered their recordings; the eight largest labelled by size."""
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

    fig, ax = plt.subplots(figsize=(7.6, 4.75))
    fig.subplots_adjust(0, 0, 1, 1)
    segs, pts, gh = [], [], []
    for h, names in members.items():
        g = df.ghost.reindex(names).fillna(False).astype(bool).sort_values(ascending=False)
        for n, q in zip(g.index, vogel(len(g)) + hpos[h]):
            segs.append((hpos[h], q))
            pts.append(q)
            gh.append(bool(g[n]))
    pts, gh = np.array(pts), np.array(gh)
    ax.add_collection(LineCollection(segs, colors="#dddcd6", linewidths=0.25, zorder=1))
    ax.scatter(pts[~gh, 0], pts[~gh, 1], s=5, c=REAL, lw=0, zorder=2)
    ax.scatter(pts[gh, 0], pts[gh, 1], s=5, c=GHOST, lw=0, zorder=2)
    c = np.array(centers)
    ax.scatter(c[:, 0], c[:, 1], s=8 + 0.15 * sizes, c=HUB, lw=0.6, ec="white", zorder=3)

    # the eight largest houses: number of names; the two companies the press named also by name
    comp, disp = company(), display_names()
    press = {"Overtone Studios": "Overtone Studios", "Firefly Entertainment AB": "Firefly"}
    for h in disp:
        i = houses.index(h)
        x, y = centers[i]
        lab = f"{sizes[i]} names"
        if comp[h] in press:
            lab += f"\n{press[comp[h]]}"
        ax.text(x, y - radii[i] * 0.42, lab, ha="center", va="center", fontsize=12.5, weight="bold",
                color=INK, linespacing=1.1, zorder=5,
                path_effects=[pe.withStroke(linewidth=4, foreground="white")])
    allxy = np.vstack([pts, c])
    ax.set_xlim(allxy[:, 0].min() - 2, allxy[:, 0].max() + 2)
    ax.set_ylim(allxy[:, 1].min() - 2, allxy[:, 1].max() + 2)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_axis_off()
    save(fig, "s2_houses")


# ------------------------------------------------------------------ slide 3
def s3_auc() -> None:
    """How well each thing a listener sees predicts the flag (ROC AUC)."""
    t = json.loads((PROCESSED / "text.json").read_text())
    v = json.loads((PROCESSED / "vision.json").read_text())
    a = json.loads((PROCESSED / "audio.json").read_text())
    rows = [("Sound", a["auc_audio"]["auc"]), ("Artist name", t["auc_name"]["auc"]),
            ("Track titles", t["auc_titles"]["auc"]), ("Cover art", v["auc_cover"]["auc"]),
            ("Bio", t["auc_bio"]["auc"])]
    fig, ax = plt.subplots(figsize=(4.9, 2.25))
    fig.subplots_adjust(left=0.25, right=0.91, top=0.97, bottom=0.2)
    ax.axvspan(0.45, 0.66, color=FAINT, lw=0, zorder=0)
    y = np.arange(len(rows))[::-1] * 0.85
    ax.text(0.555, y[0] + 0.62, "close to guessing", ha="center", va="center", fontsize=12, color=MUTED,
            style="italic")
    for yi, (lab, r) in zip(y, rows):
        strong = r > 0.7
        col = NAVY if strong else "#8d8c87"
        ax.plot([0.5, r], [yi, yi], color=col, lw=2.2, zorder=2, solid_capstyle="butt")
        ax.scatter([r], [yi], s=70, color=col, zorder=3, lw=0)
        ax.text(r + 0.02, yi, f"{r:.2f}", va="center", fontsize=13, color=col, weight="bold" if strong else None)
    ax.set_yticks(y, [lab for lab, _ in rows], fontsize=13)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.5, y[0] + 0.85)
    ax.set_xticks([0.5, 0.75, 1.0], ["0.5\nguessing", "0.75", "1.0\nperfect"], fontsize=11)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=3)
    ax.spines["left"].set_visible(False)
    save(fig, "s3_auc")


def s3_links() -> None:
    """'Fans also like' links between real and ghost-like names, against two kinds of chance."""
    r = json.loads((PROCESSED / "recommendation.json").read_text())["nulls"]
    rows = [("Real → ghost-like", 1), ("Ghost-like → real", 2)]
    nulls = [("degree", "vs random links", "#b9b8b2"),
             ("playlist", "vs random links inside the same playlist", NAVY)]
    fig, ax = plt.subplots(figsize=(4.9, 2.25))
    fig.subplots_adjust(left=0.36, right=0.97, top=0.72, bottom=0.04)
    for yi, (lab, i) in zip([1, 0], rows):
        for k, (null, _, col) in enumerate(nulls):
            v = r[null]["flows_ratio"][i]
            yy = yi + (0.17 if k == 0 else -0.17)
            ax.barh(yy, v, height=0.3, color=col)
            ax.text(v - 0.03, yy, f"{v:.2f}×", va="center", ha="right", fontsize=13, weight="bold",
                    color="white" if k else INK)
        ax.text(-0.03, yi, lab, ha="right", va="center", fontsize=13)
    ax.axvline(1, color=INK, lw=1.2)
    ax.text(1, 1.36, "chance", ha="center", va="bottom", fontsize=12, color=INK)
    for _, lab, col in nulls:
        ax.scatter([], [], marker="s", s=90, color=col, label=lab)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.55, 1.04), ncol=1, frameon=False, fontsize=12,
              handletextpad=0.3, labelspacing=0.25, borderaxespad=0)
    ax.set_xlim(0, 1.15)
    ax.set_ylim(-0.4, 1.4)
    ax.set_axis_off()
    save(fig, "s3_links")


def main() -> None:
    setup()
    for f in sorted(OUT.glob("*.png")) if OUT.exists() else []:
        f.unlink()  # drop figures from older versions of this script
    s1_slots()
    s1_check()
    s2_houses()
    s3_auc()
    s3_links()


if __name__ == "__main__":
    main()
