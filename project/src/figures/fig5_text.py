"""Figure 5 (single column): what a listener sees.

A  How well one kind of visible content predicts the ghost-like flag (cross-validated ROC AUC,
   functional-playlist names). Grey band: the same classifier on shuffled labels (95% range).
B  Words most typical of ghost-like and of real bios (weighted log-odds, informative prior).

    python -m src.figures.fig5_text
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures import style
from src.figures.style import GHOST, REAL, TEXT, TEXT_2
from src.load import PROCESSED


def panel_auc(ax) -> None:
    t = json.loads((PROCESSED / "text.json").read_text())
    v = json.loads((PROCESSED / "vision.json").read_text())
    a = json.loads((PROCESSED / "audio.json").read_text())
    rows = [("Artist name", t["auc_name"]), ("Track titles", t["auc_titles"]),
            ("Sound (30-s previews)", a["auc_audio"]), ("Cover art", v["auc_cover"]),
            ("Bio (names that have one)", t["auc_bio"])]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, r) in zip(y, rows):
        lo, hi = r["null_mean"] - 1.96 * r["null_sd"], r["null_mean"] + 1.96 * r["null_sd"]
        ax.plot([lo, hi], [yi, yi], color="#d6d5cf", lw=6, solid_capstyle="butt", zorder=1)
        ax.scatter([r["auc"]], [yi], s=30, color=TEXT, zorder=3, lw=0)
        ax.text(r["auc"] + 0.015, yi, f"{r['auc']:.2f}", va="center", fontsize=6.5, color=TEXT)
    ax.set_yticks(y, [lab for lab, _ in rows], fontsize=6.5)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xticks([0.5, 0.75, 1.0], ["0.5\nguessing", "0.75", "1.0\nperfect"], fontsize=6.5)
    ax.set_xlabel("How well it predicts the flag (ROC AUC)", fontsize=6.5)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    b = t["bio_presence"]
    ax.text(0.5, -0.3, f"Grey: the same test with shuffled labels (chance).\nOnly {b['ghost']:.0%} of ghost-like names have a bio "
            f"({b['real']:.0%} of real artists).", transform=ax.transAxes, ha="center", va="top", fontsize=6.5,
            color=TEXT_2)
    ax.set_title("A  Names, titles and sound say little;\n    bios and covers say more", loc="left")


def panel_words(ax) -> None:
    lo = pd.read_csv(PROCESSED / "bio_log_odds.csv")
    t = json.loads((PROCESSED / "text.json").read_text())
    g = lo[lo.z > 0].nlargest(10, "z").word.tolist()
    r = lo[lo.z < 0].nsmallest(10, "z").word.tolist()
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(-10.6, 1.2)
    ax.text(0.25, 0.4, "Real bios", ha="center", fontsize=7, fontweight="bold", color=REAL)
    ax.text(0.75, 0.4, "Ghost-like bios", ha="center", fontsize=7, fontweight="bold", color=GHOST)
    for k, (wr, wg) in enumerate(zip(r, g)):
        ax.text(0.25, -k - 0.6, wr, ha="center", fontsize=6.5, color=TEXT)
        ax.text(0.75, -k - 0.6, wg, ha="center", fontsize=6.5, color=TEXT)
    ent, tok = t["bio_any_entity"], t["bio_median_tokens"]
    ax.text(0.5, -10.4, f"Names a person, place, date or organisation: real {ent['False']:.0%}, "
            f"ghost-like {ent['True']:.0%}\nMedian length: real {tok['False']:.0f} words, "
            f"ghost-like {tok['True']:.0f} words", ha="center", va="top", fontsize=6.5, color=TEXT_2)
    ax.set_title("B  Ghost bios describe a mood; real bios a career", loc="left")


def main() -> None:
    style.apply()
    fig = plt.figure(figsize=(6.6, 2.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.9], wspace=0.25)
    panel_auc(fig.add_subplot(gs[0]))
    panel_words(fig.add_subplot(gs[1]))
    style.save(fig, "fig5_text")


if __name__ == "__main__":
    main()
