"""Figure 1: where the playlist slots go. Every band is a count from the data.

Left: the 50 playlists, grouped by kind. Middle: whether the slot's (primary) artist name is
ghost-like. Right: who registered the slot's recording (ISRC registrant, the "house"): the
eight largest houses, all other houses, or no code (track not found on Deezer).
Band width = number of playlist slots.

    python -m src.figures.fig1_data
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path

from src.common import placement
from src.features.production import track_houses
from src.figures import style
from src.figures.houses import company, display_names, top_houses
from src.figures.style import GHOST, NEUTRAL, REAL, TEXT, TEXT_2
from src.load import PROCESSED, load

# Playlist kinds, by playlist name. Chosen by what the playlist is for, not by its content.
KINDS = {
    "Sleep": ["Sleep", "Deep Sleep", "Guitar for Sleep", "goodnight.", "Sleepy Piano",
              "Floating Through Space", "Ambient Dreams", "Dreamy Forest Music", "Jazz for Sleep"],
    "Focus and study": ["Deep Focus", "Ambient Study", "Focus Piano", "Reading Soundtrack", "Coffee Beats",
                        "Music for Concentration", "Intense Studying", "Piano in the Background",
                        "lofi beats", "Chill Instrumental Beats", "chillhop", "Workday Lounge",
                        "Jazz for Study"],
    "Calm and relax": ["Calm", "calm vibes", "Stress Relief", "Peaceful Meditation", "Yoga & Meditation",
                       "Chill Out Music", "Ambient Relaxation", "Spa Treatment", "Soft Lounge",
                       "Calming Acoustic", "Sweater Weather Instrumentals", "rainy day piano",
                       "Relaxing Piano Covers", "mentally i'm here"],
    "Jazz": ["Jazz in the Background", "Chilled Jazz", "Cozy Coffee Shop Jazz", "Date Night Jazz"],
    "Classical and ambient": ["Calming Classical", "Women of Classical", "Atmospheric Piano",
                              "Ambient Essentials", "Women of Ambient"],
}
GENRE = "Genre playlists (control)"
OTHER, NOCODE = "All other houses", "No recording code"
HOUSE_C = "#3d3c38"


def slots() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    pl = load("playlists").set_index("playlist_id")
    kind = {n.strip(): k for k, names in KINDS.items() for n in names}
    pa = placement()
    pa = pa[pa.is_primary].copy()
    pa["kind"] = [GENRE if g == "genre" else kind[n.strip()] for g, n in zip(pa.group, pa.name)]
    pa["label"] = np.where(pa.artist_id.map(df.ghost).fillna(False).astype(bool), "ghost", "real")
    reg = pa.track_id.map(track_houses().set_index("track_id").registrant)
    names = display_names()
    pa["house"] = np.where(reg.isna(), NOCODE, reg.map(names).fillna(OTHER))
    assert len(pa.name.unique()) == len(pl), "every playlist must have a kind"
    return pa


def stack(order: list[str], sizes: pd.Series, gap: float, top: float = 1.0) -> dict[str, list[float]]:
    """Node -> [y_top, y_bottom], stacked from the top with a gap between nodes."""
    out, y = {}, top
    for k in order:
        h = sizes.get(k, 0)
        out[k] = [y, y - h]
        y -= h + gap
    return out


def band(ax, x0, y0, x1, y1, h, color, alpha) -> None:
    """Smooth band from (x0, y0..y0-h) to (x1, y1..y1-h)."""
    xm = (x0 + x1) / 2
    verts = [(x0, y0), (xm, y0), (xm, y1), (x1, y1), (x1, y1 - h), (xm, y1 - h), (xm, y0 - h), (x0, y0 - h),
             (x0, y0)]
    codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4,
             Path.CURVE4, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), fc=color, ec="white", lw=0.35, alpha=alpha, zorder=1))


def main() -> None:
    style.apply()
    pa = slots()
    N = len(pa)
    scale = 0.82 / N  # share of the vertical space taken by bars; the rest is gaps
    cols = {
        "kind": (list(KINDS) + [GENRE], 0.022),
        "label": (["ghost", "real"], 0.03),
        "house": ([f"House {k}" for k in range(1, 9)] + [OTHER, NOCODE], 0.017),
    }
    X = {"kind": 0.25, "label": 0.46, "house": 0.715}
    W = 0.012
    pos = {}
    for c, (order, gap) in cols.items():
        sizes = pa[c].value_counts() * scale
        total = sizes.sum() + gap * (len(order) - 1)
        pos[c] = stack(order, sizes, gap, top=0.5 + total / 2)

    fig, ax = plt.subplots(figsize=(7.08, 3.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.04, 1.04)
    ax.axis("off")

    lab_col = {"ghost": GHOST, "real": REAL}
    # bands: kind -> label (coloured by label), then label -> house (coloured by label)
    for a, b, col_by in [("kind", "label", "label"), ("label", "house", "label")]:
        flows = pa.groupby([a, b]).size() * scale
        out_y = {k: v[0] for k, v in pos[a].items()}
        in_y = {k: v[0] for k, v in pos[b].items()}
        for (s, t) in sorted(flows.index, key=lambda st: (cols[a][0].index(st[0]), cols[b][0].index(st[1]))):
            h = flows[(s, t)]
            # sorted source-first, so bands arrive at each target in source order
            band(ax, X[a] + W, out_y[s], X[b], in_y[t], h, lab_col[t if col_by == b else s], 0.8)
            out_y[s] -= h
            in_y[t] -= h
    # nodes
    for c, (order, _) in cols.items():
        for k in order:
            yt, yb = pos[c][k]
            fc = lab_col.get(k, HOUSE_C if c != "kind" else "#6f6e69")
            if k in (OTHER, NOCODE):
                fc = NEUTRAL
            ax.add_patch(Rectangle((X[c], yb), W, yt - yb, fc=fc, ec="none", zorder=2))

    fs = 6.5
    # left labels
    n_kind = pa.groupby("kind").size()
    n_pl = pa.groupby("kind").name.nunique()
    for k in cols["kind"][0]:
        yt, yb = pos["kind"][k]
        ax.text(X["kind"] - 0.008, (yt + yb) / 2, f"{k}\n{n_pl[k]} playlists · {n_kind[k]:,} slots",
                ha="right", va="center", fontsize=fs, color=TEXT, linespacing=1.2)
    # middle labels
    n_lab = pa.groupby("label").size()
    mood = pa[pa.group == "functional"].label.value_counts(normalize=True)
    for k, txt in [("ghost", "Ghost-like names"), ("real", "Real artists")]:
        yt, yb = pos["label"][k]
        ax.text(X["label"] + W + 0.008, yt - 0.012,
                f"{txt}\n{n_lab[k]:,} slots\n{mood[k]:.0%} of mood-playlist slots",
                ha="left", va="top", fontsize=fs, color=TEXT, linespacing=1.25,
                bbox=dict(fc="white", ec=lab_col[k], lw=0.8, boxstyle="round,pad=0.35"), zorder=4)
    # right labels
    comp = company()
    names = {v: k for k, v in display_names().items()}
    n_h = pa.groupby("house").size()
    for k in cols["house"][0]:
        yt, yb = pos["house"][k]
        if k.startswith("House"):
            co = comp[names[k]]
            txt = f"{k}  {names[k]} · {co}  ({n_h[k]:,})"
        else:
            txt = f"{k}  ({n_h[k]:,})"
        ax.text(X["house"] + W + 0.008, (yt + yb) / 2, txt, ha="left", va="center", fontsize=fs,
                color=TEXT if k.startswith("House") else TEXT_2)

    # column heads
    heads = [("kind", "Playlist", "50 Spotify editorial playlists"),
             ("label", "Artist name on the slot", ""),
             ("house", "Who registered the recording", "the house, from the track's ISRC code")]
    for c, h1, h2 in heads:
        x = X[c] + W / 2
        ha = "center"
        if c == "kind":
            x, ha = X[c] + W, "right"
        if c == "house":
            x, ha = X[c], "left"
        ax.text(x, 1.035, h1, ha=ha, va="bottom", fontsize=7, fontweight="bold", color=TEXT)
        ax.text(x, 1.035, h2, ha=ha, va="top", fontsize=fs, color=TEXT_2)
    # colour key under the middle heading: what orange and blue mean, used in every figure
    xm = X["label"] + W / 2
    for y, col, lab in [(1.013, GHOST, "ghost-like name"), (0.983, REAL, "real artist")]:
        ax.add_patch(Rectangle((xm - 0.075, y - 0.011), 0.011, 0.022, fc=col, ec="none", clip_on=False))
        ax.text(xm - 0.06, y, lab, ha="left", va="center", fontsize=fs, color=TEXT)
    ax.text(0.5, -0.045, "Band width = number of playlist slots (9,004). Also collected: each name's profile, bio and "
            "'Fans also like' list; each track's credits, cover art and 30-second preview.",
            ha="center", va="top", fontsize=fs, color=TEXT_2)
    style.save(fig, "fig1_data")


if __name__ == "__main__":
    main()
