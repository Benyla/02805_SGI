"""Figure 2: three independent views agree, and where the ghost-like names sit.

A  Functional-playlist artists binned by how many of 5 identity markers are missing; mean
   production and demand score per bin (95% bootstrap CI). Inset: Spearman rho (functional only).
B  Share labelled ghost-like in reference groups that were not used to build the label.
C  Ghost-like share of each playlist's slots (dot area ~ followers).

    python -m src.figures.fig2_convergence
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.figures import style
from src.figures.style import GHOST, NEUTRAL, REAL, TEXT, TEXT_2
from src.load import PROCESSED, SEED

PROD_C, DEM_C = "#1baf7a", "#6250d6"  # aqua, violet: two series distinct from ghost/real


def boot_ci(x: np.ndarray, rng, n: int = 1000) -> tuple[float, float]:
    m = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def panel_a(ax, df, rng) -> None:
    f = df[df.group == "functional"].copy()
    f["k"] = (f.identity_score * 5).round().astype(int)
    for col, c, lab, dx in [("production_z", PROD_C, "production score", -0.08),
                            ("demand_z", DEM_C, "demand score", 0.08)]:
        g = f.groupby("k")[col]
        mu = g.mean()
        ci = np.array([boot_ci(v.dropna().to_numpy(), rng) for _, v in g])
        x = mu.index.to_numpy() + dx
        ax.fill_between(x, ci[:, 0], ci[:, 1], color=c, alpha=0.18, lw=0)
        ax.plot(x, mu, "-o", color=c, ms=3.2, lw=1.6, label=lab, mec="white", mew=0.5)
    n = f.k.value_counts().sort_index()
    ax.set_xticks(range(6), [str(k) for k in range(6)])
    ax.set_xlabel("Identity markers missing (of 5)")
    ax.set_ylabel("Mean score (z units)")
    ax.axhline(0, color=NEUTRAL, lw=0.5)
    ax.legend(loc="upper left", fontsize=5.8)
    ax.set_title("A  Three views agree", loc="left")


def panel_b(ax, df) -> None:
    ref = pd.read_csv(PROCESSED / "convergence_reference_sets.csv")
    ref["label"] = ref.name.str.replace("all 5 genre playlists", "Genre-playlist artists")
    order = ["Firefly Entertainment", "Epidemic Sound", "Deutsche Grammophon", "Decca",
             "Sony Classical", "Warner Classics", "Genre-playlist artists"]
    ref = ref.set_index("label").loc[order].reset_index()
    ref["label"] = ref.label.replace({"Firefly Entertainment": "Firefly Ent.",
                                      "Deutsche Grammophon": "Deutsche Gramm.",
                                      "Genre-playlist artists": "Genre playlists"})
    y = np.arange(len(ref))[::-1]
    colors = [GHOST if s > 0.5 else REAL for s in ref.share_ghost]
    ax.barh(y, ref.share_ghost, color=colors, height=0.62)
    for yi, s, n in zip(y, ref.share_ghost, ref.n_artists):
        ax.text(s + 0.02, yi, f"{s:.0%}  (n={n})", va="center", fontsize=5.4, color=TEXT_2)
    ax.set_yticks(y, ref.label, fontsize=5.8)
    ax.set_xlim(0, 1.5)
    ax.set_xticks([0, 0.5, 1], ["0%", "50%", "100%"])
    ax.set_xlabel("Labelled ghost-like")
    ax.grid(axis="y", visible=False)
    # bracket the groups
    for (y0, y1), lab in [((y[1], y[0]), "named suppliers"), ((y[5], y[2]), "classical majors\non the same playlists")]:
        ax.annotate("", xy=(1.48, y0 - 0.3), xytext=(1.48, y1 + 0.3),
                    arrowprops=dict(arrowstyle="-", color=NEUTRAL, lw=0.6))
        ax.text(1.46, (y0 + y1) / 2 - 0.25, lab, ha="right", va="center", fontsize=4.9, color=TEXT_2,
                rotation=0)
    ax.set_title("B  Checks the label never saw", loc="left")


def panel_c(ax) -> None:
    p = pd.read_csv(PROCESSED / "placement_playlists.csv")
    rng = np.random.default_rng(SEED)
    p["y"] = np.where(p.group == "genre", 0, 1) + rng.uniform(-0.22, 0.22, len(p))
    s = 6 + 60 * np.sqrt(p.followers / p.followers.max())
    for g, c in [("functional", "#3d3c38"), ("genre", NEUTRAL)]:
        m = p.group == g
        ax.scatter(p.ghost_share[m], p.y[m], s=s[m], c=c, alpha=0.75, lw=0.5, ec="white")
    ax.set_yticks([0, 1], ["genre\n(5)", "functional\n(45)"], fontsize=5.8)
    ax.set_ylim(-0.45, 1.7)
    ax.set_xlim(-0.04, 1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Playlist slots held by ghost-like names")
    ax.grid(axis="y", visible=False)
    # hand-placed callouts (text x, text y) so they never collide
    for name, tx, ty, ha in [("Calming Classical", 0.0, 1.62, "left"), ("lofi beats", 0.46, 0.62, "center"),
                             ("Sleep", 0.80, 0.55, "center"), ("Women of Ambient", 0.02, 0.38, "left"),
                             ("Jazz for Sleep", 1.0, 1.36, "right")]:
        r = p[p.name == name]
        if len(r):
            r = r.iloc[0]
            ax.annotate(f"{name} {r.ghost_share:.0%}", xy=(r.ghost_share, r.y), xytext=(tx, ty),
                        fontsize=5.0, color=TEXT, ha=ha, va="center",
                        arrowprops=dict(arrowstyle="-", color=TEXT_2, lw=0.4, shrinkB=2))
    f = p[p.group == "functional"]
    w = np.average(f.ghost_share, weights=f.n_tracks)
    ax.axvline(w, color=GHOST, lw=0.8, ls="--")
    ax.text(w + 0.01, 0.18, f"all functional\nslots: {w:.0%}", fontsize=5.2, color=GHOST,
            ha="left", va="center")
    ax.set_title("C  Where they are placed", loc="left")


def main() -> None:
    style.apply()
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    fig, axes = plt.subplots(1, 3, figsize=(7.08, 2.35), gridspec_kw={"width_ratios": [1.05, 1.0, 1.05],
                                                                      "wspace": 0.62})
    panel_a(axes[0], df, rng)
    panel_b(axes[1], df)
    panel_c(axes[2])
    style.save(fig, "fig2_convergence")


if __name__ == "__main__":
    main()
