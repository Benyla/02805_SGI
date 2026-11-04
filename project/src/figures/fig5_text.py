"""Figure 5: what a listener sees.

A  Cross-validated ROC AUC for predicting the ghost-like label from one kind of visible
   content at a time (functional-playlist artists). Hatched = shuffled-label null.
B  Words that separate ghost-like from real bios (weighted log-odds, informative Dirichlet
   prior); and how often a bio names a person, place, date or organisation (spaCy NER).

    python -m src.figures.fig5_text
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures import style
from src.figures.style import GHOST, NEUTRAL, REAL, TEXT, TEXT_2
from src.load import PROCESSED


def panel_auc(ax) -> None:
    t = json.loads((PROCESSED / "text.json").read_text())
    v = json.loads((PROCESSED / "vision.json").read_text())
    a = json.loads((PROCESSED / "audio.json").read_text())
    rows = [("Artist name", t["auc_name"]), ("Track titles", t["auc_titles"]),
            ("Sound (30 s previews)*", a["auc_audio"]), ("Cover art (CLIP)*", v["auc_cover"]),
            ("Bio text (if any)", t["auc_bio"])]
    y = np.arange(len(rows))[::-1]
    auc = [r["auc"] for _, r in rows]
    nul = [r["null_mean"] + 1.96 * r["null_sd"] for _, r in rows]
    ax.barh(y, [x - 0.5 for x in auc], left=0.5, color="#3d3c38", height=0.58)
    for yi, n in zip(y, nul):
        ax.add_patch(plt.Rectangle((0.5, yi - 0.29), n - 0.5, 0.58, fc="white", ec=TEXT_2, lw=0.5,
                                   hatch="//////", zorder=3, alpha=0.9))
    for yi, x, n, (_, r) in zip(y, auc, nul, rows):
        ax.text(max(x, n) + 0.008, yi, f"{x:.2f}  (n={r['n']:,})", va="center", fontsize=5.4, color=TEXT)
    ax.set_yticks(y, [lab for lab, _ in rows], fontsize=5.9)
    ax.set_xlim(0.45, 1.0)
    ax.axvline(0.5, color=TEXT_2, lw=0.6)
    ax.set_xlabel("ROC AUC (0.5 = guessing, 1 = perfect)")
    ax.grid(axis="y", visible=False)
    ax.set_title("A  Can you see a ghost?", loc="left")


def panel_words(ax) -> None:
    lo = pd.read_csv(PROCESSED / "bio_log_odds.csv")
    t = json.loads((PROCESSED / "text.json").read_text())
    g = lo[lo.z > 0].nlargest(10, "z")
    r = lo[lo.z < 0].nsmallest(10, "z")
    for k, (w, z) in enumerate(zip(g.word, g.z)):
        ax.barh(-k, z, color=GHOST, height=0.62)
        ax.text(z + 0.4, -k, w, va="center", fontsize=5.6, color=TEXT)
    for k, (w, z) in enumerate(zip(r.word, r.z)):
        ax.barh(-k, z, color=REAL, height=0.62)
        ax.text(z - 0.4, -k, w, va="center", ha="right", fontsize=5.6, color=TEXT)
    ax.axvline(0, color=TEXT_2, lw=0.6)
    m = max(g.z.max(), -r.z.min()) * 1.55
    ax.set_xlim(-m, m)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Log-odds z-score")
    ax.text(-m * 0.97, 0.9, "real bios", color=REAL, fontsize=6.2, fontweight="bold", va="bottom")
    ax.text(m * 0.97, 0.9, "ghost-like bios", color=GHOST, fontsize=6.2, fontweight="bold",
            va="bottom", ha="right")
    ent = t["bio_any_entity"]
    tok = t["bio_median_tokens"]
    ax.text(0.5, -0.36, f"Bio names a person, place, date or organisation: real {ent['False']:.0%}, "
            f"ghost-like {ent['True']:.0%}\nMedian length: real {tok['False']:.0f} words, "
            f"ghost-like {tok['True']:.0f} words",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.4, color=TEXT_2)
    ax.set_ylim(-10, 1.8)
    ax.set_title("B  What the bios say", loc="left")


def main() -> None:
    style.apply()
    fig, axes = plt.subplots(2, 1, figsize=(3.42, 4.2), gridspec_kw={"height_ratios": [1, 1.45],
                                                                   "hspace": 0.55})
    panel_auc(axes[0])
    panel_words(axes[1])
    style.save(fig, "fig5_text")


if __name__ == "__main__":
    main()
