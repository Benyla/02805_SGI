"""Phase 1: descriptives and sanity checks.

Writes figures/descriptives_*.pdf|png, data/processed/descriptives_top_*.csv and
notes/descriptives.md (every number in the note is computed here).

    python -m src.figures.descriptives
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import mannwhitneyu

from src.common import artist_group, placement, pline_owner, track_group
from src.figures import style
from src.figures.style import GROUP_COLORS, TEXT, TEXT_2
from src.load import PROCESSED, PROJECT, load, split_list

GROUPS = ["functional", "genre"]


# ---------------------------------------------------------------- per playlist
def playlist_table(pl: pd.DataFrame, pa: pd.DataFrame) -> pd.DataFrame:
    per = pa.groupby("playlist_id").agg(n_slots=("track_id", "nunique"),
                                        n_artists=("artist_id", "nunique"))
    out = pl.set_index("playlist_id")[["name", "group", "category", "followers"]].join(per)
    out["artists_per_track"] = out.n_artists / out.n_slots
    return out.sort_values(["group", "n_slots"], ascending=[True, True])


def fig_playlists(tab: pd.DataFrame) -> None:
    tab = tab.sort_values(["group", "n_slots"], ascending=[False, True])
    y = np.arange(len(tab))
    colors = tab.group.map(GROUP_COLORS)
    fig, axes = plt.subplots(1, 2, figsize=(7, 7.2), sharey=True)
    for ax, col, title in [(axes[0], "n_slots", "Tracks per playlist"),
                           (axes[1], "n_artists", "Distinct artists per playlist")]:
        ax.barh(y, tab[col], color=colors, height=0.72)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.margins(y=0.01)
    axes[0].set_yticks(y, tab.name, fontsize=6)
    axes[1].legend(handles=[Patch(color=GROUP_COLORS[g], label=g) for g in GROUPS],
                   loc="lower right")
    style.save(fig, "descriptives_playlists")


# ---------------------------------------------------------------- overlap
def jaccard_matrix(pa: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    sets = pa.groupby("playlist_id").artist_id.agg(set)
    ids = list(sets.index)
    m = np.zeros((len(ids), len(ids)))
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            m[i, j] = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
    names = pl.set_index("playlist_id").name
    return pd.DataFrame(m, index=ids, columns=ids).rename(index=names, columns=names)


def fig_jaccard(J: pd.DataFrame, group_of: pd.Series) -> None:
    # order: functional block then genre block, each by average-linkage clustering
    order = []
    for g in GROUPS:
        sub = J.loc[group_of[group_of == g].index, group_of[group_of == g].index]
        if len(sub) > 2:
            Z = linkage(squareform(1 - sub.values, checks=False), "average")
            order += list(sub.index[leaves_list(Z)])
        else:
            order += list(sub.index)
    J = J.loc[order, order]
    off = J.values.copy()
    np.fill_diagonal(off, np.nan)
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    im = ax.imshow(off, cmap=style.SEQ_CMAP, vmin=0, vmax=np.nanpercentile(off, 99.5))
    ax.set_xticks(range(len(J)), J.columns, rotation=90, fontsize=5)
    ax.set_yticks(range(len(J)), J.index, fontsize=5)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(GROUP_COLORS[group_of[lbl.get_text()]])
    nf = int((group_of == "functional").sum())
    for v in (nf - 0.5,):
        ax.axhline(v, color=TEXT, lw=0.8)
        ax.axvline(v, color=TEXT, lw=0.8)
    ax.grid(False)
    ax.set_title("Artist overlap between playlists (Jaccard); labels coloured by group")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Jaccard similarity of artist sets")
    style.save(fig, "descriptives_jaccard")


# ---------------------------------------------------------------- distributions
def ecdf(ax, x, color, label):
    x = np.sort(np.asarray(x, dtype=float))
    x = x[np.isfinite(x) & (x > 0)]
    ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, lw=1.6,
            label=f"{label} (n={len(x):,})")


def fig_distributions(A: pd.DataFrame, T: pd.DataFrame) -> None:
    panels = [
        (A, "monthly_listeners", "Monthly listeners (artist)"),
        (A, "followers", "Followers (artist)"),
        (A, "followers_per_listener", "Followers per monthly listener (artist)"),
        (T.assign(duration_min=T.duration_ms / 60000), "duration_min", "Track duration (minutes)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7, 5))
    for ax, (df, col, title) in zip(axes.flat, panels):
        for g in GROUPS:
            ecdf(ax, df.loc[df.grp == g, col], GROUP_COLORS[g], g)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_ylabel("Share at or below")
        med = df.groupby("grp")[col].median()
        ax.text(0.98, 0.04, "medians: " + " / ".join(f"{med[g]:.3g}" for g in GROUPS),
                transform=ax.transAxes, ha="right", fontsize=6.5, color=TEXT_2)
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    style.save(fig, "descriptives_distributions")


# ---------------------------------------------------------------- production houses
def house_table(T: pd.DataFrame, key: str) -> pd.DataFrame:
    x = T[[key, "track_id", "artist_list", "grp"]].dropna(subset=[key]).explode("artist_list")
    g = x.groupby(key)
    tab = pd.DataFrame({
        "n_tracks": g.track_id.nunique(),
        "n_artist_names": g.artist_list.nunique(),
        "share_functional_tracks": g.apply(
            lambda d: (d.drop_duplicates("track_id").grp == "functional").mean()),
    })
    tab[["n_tracks", "n_artist_names"]] = tab[["n_tracks", "n_artist_names"]].astype(int)
    return tab.sort_values(["n_artist_names", "n_tracks"], ascending=False)


def fig_houses(tables: dict[str, pd.DataFrame], n: int = 30) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9, 6.6), gridspec_kw={"wspace": 0.95})
    for ax, (title, tab) in zip(axes, tables.items()):
        top = tab.head(n).iloc[::-1]
        y = np.arange(len(top))
        majority = np.where(top.share_functional_tracks >= 0.5, "functional", "genre")
        ax.barh(y, top.n_artist_names, color=[GROUP_COLORS[m] for m in majority], height=0.72)
        labels = [k if len(k) <= 30 else k[:28] + "…" for k in top.index]
        ax.set_yticks(y, labels, fontsize=5.5)
        for yi, (names, tracks) in enumerate(zip(top.n_artist_names, top.n_tracks)):
            ax.text(names, yi, f" {tracks}", va="center", fontsize=5, color=TEXT_2)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Distinct artist names (label: tracks)")
        ax.grid(axis="y", visible=False)
        ax.margins(y=0.01, x=0.15)
    fig.legend(handles=[Patch(color=GROUP_COLORS[g], label=f"mostly {g} tracks")
                        for g in GROUPS], loc="upper right", ncol=2, bbox_to_anchor=(0.98, 0.985))
    fig.suptitle("Top 30 production houses by number of artist names", x=0.01, ha="left",
                 fontsize=9, fontweight="bold")
    style.save(fig, "descriptives_top_houses")


# ---------------------------------------------------------------- main
def main() -> None:
    style.apply()
    pl, t, ar = load("playlists"), load("tracks"), load("artists")
    pa = placement()
    tg, ag = track_group(), artist_group()

    T = t.join(tg.rename("grp"), on="track_id")
    T["artist_list"] = split_list(T.artist_ids)
    T = T.merge(load("track_sources"), on="track_id", how="left")
    T["pline_owner"] = pline_owner(T.p_line)
    A = ar.join(ag.rename("grp"), on="artist_id")

    # --- per playlist
    ptab = playlist_table(pl, pa)
    fig_playlists(ptab)
    J = jaccard_matrix(pa, pl)
    gname = pl.set_index("name").group
    fig_jaccard(J, gname)
    iu = np.triu_indices(len(J), 1)
    pairs = pd.DataFrame({"a": J.index[iu[0]], "b": J.columns[iu[1]], "j": J.values[iu]})
    pairs["kind"] = [
        "-".join(sorted([gname[a], gname[b]])) for a, b in zip(pairs.a, pairs.b)
    ]
    jac_by_kind = pairs.groupby("kind").j.agg(["mean", "max", "count"])
    top_pairs = pairs.sort_values("j", ascending=False).head(5)

    # --- distributions
    fig_distributions(A, T)

    # --- houses
    houses = {
        "Credit Source": house_table(T, "source"),
        "(P) line owner": house_table(T, "pline_owner"),
        "ISRC registrant (Deezer-matched)": house_table(T, "isrc_registrant"),
    }
    fig_houses(houses)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for (title, tab), slug in zip(houses.items(), ["source", "pline_owner", "isrc_registrant"]):
        tab.to_csv(PROCESSED / f"descriptives_top_{slug}.csv")

    # --- headline numbers
    alias_only = T.groupby("grp").apply(lambda d: (d.n_credited_non_artists == 0).mean())
    fpl_artist = A.groupby("grp").followers_per_listener.median()
    slots = pa[pa.is_primary].join(ar.set_index("artist_id").followers_per_listener, on="artist_id")
    fpl_slot = slots.groupby("group").followers_per_listener.median()
    mw = mannwhitneyu(A.loc[A.grp == "functional", "followers_per_listener"].dropna(),
                      A.loc[A.grp == "genre", "followers_per_listener"].dropna())
    deezer = T.groupby("grp").deezer_match.mean()
    verified = A.groupby("grp").verified.mean()
    has_bio = A.assign(hb=A.bio_len.fillna(0) > 0).groupby("grp").hb.mean()
    dur = T.groupby("grp").duration_ms.median() / 1000
    hs = houses["Credit Source"]

    def house_size(grp: str, key: str = "source", tab: pd.DataFrame | None = None) -> str:
        """Mean / median / p90 of artist names in the house of a track in this group."""
        tab = hs if tab is None else tab
        s = T.loc[T.grp == grp, key].map(tab.n_artist_names).dropna()
        return f"{s.mean():.1f} / {s.median():.0f} / {s.quantile(.9):.0f}"

    self_labelled = (T.source.str.casefold()
                     == T.artist_names.str.split(";").str[0].str.strip().str.casefold()).groupby(T.grp).mean()
    T["deezer_attempted"] = T.deezer_match.notna()
    dz_att = T.groupby("grp").deezer_attempted.mean()
    pl_cov = (load("playlist_tracks").merge(T[["track_id", "deezer_attempted"]])
              .groupby("playlist_id").deezer_attempted.mean())
    n_pl_zero = int((pl_cov < 0.1).sum())

    lines = [
        "# Phase 1: descriptives",
        "",
        "Generated by `python -m src.figures.descriptives`. Do not edit by hand.",
        "",
        "## Coverage",
        "",
        f"- Playlists: {len(pl)} ({(pl.group == 'functional').sum()} functional, "
        f"{(pl.group == 'genre').sum()} genre).",
        f"- Tracks: {len(t):,} ({(tg == 'functional').sum():,} functional only, "
        f"{(tg == 'genre').sum():,} genre only, {(tg == 'both').sum()} on both).",
        f"- Artists: {len(ar):,} ({(ag == 'functional').sum():,} functional only, "
        f"{(ag == 'genre').sum():,} genre only, {(ag == 'both').sum()} on both).",
        f"- Median tracks per playlist: functional {ptab[ptab.group == 'functional'].n_slots.median():.0f}, "
        f"genre {ptab[ptab.group == 'genre'].n_slots.median():.0f}. Median distinct artists per track: "
        f"functional {ptab[ptab.group == 'functional'].artists_per_track.median():.2f}, "
        f"genre {ptab[ptab.group == 'genre'].artists_per_track.median():.2f}.",
        "",
        "## Check against context.md §4",
        "",
        "| Quantity | context.md | Reproduced |",
        "|---|---|---|",
        f"| Alias-only credits, functional tracks | 81% | {alias_only['functional']:.1%} |",
        f"| Alias-only credits, genre tracks | 5% | {alias_only['genre']:.1%} |",
        f"| Median followers/listener, functional | 0.0016 | {fpl_slot['functional']:.5f} per playlist slot (primary artist); "
        f"{fpl_artist['functional']:.5f} per distinct artist |",
        f"| Median followers/listener, genre | 0.227 | {fpl_slot['genre']:.3f} per playlist slot (primary artist); "
        f"{fpl_artist['genre']:.3f} per distinct artist |",
        "",
        "The context.md medians are over playlist slots using each track's primary artist, so "
        "artists placed many times weigh more. Per distinct artist the gap is "
        f"{fpl_artist['genre'] / fpl_artist['functional']:.0f}x "
        f"(Mann-Whitney U p = {mw.pvalue:.1e}). Use the per-artist version in the paper.",
        "",
        "## Other contrasts (functional vs genre)",
        "",
        "| | functional | genre |",
        "|---|---|---|",
        f"| Deezer match rate | {deezer['functional']:.1%} | {deezer['genre']:.1%} |",
        f"| Verified artists | {verified['functional']:.1%} | {verified['genre']:.1%} |",
        f"| Artists with a bio | {has_bio['functional']:.1%} | {has_bio['genre']:.1%} |",
        f"| Median track duration (s) | {dur['functional']:.0f} | {dur['genre']:.0f} |",
        f"| Median monthly listeners | {A[A.grp == 'functional'].monthly_listeners.median():,.0f} | "
        f"{A[A.grp == 'genre'].monthly_listeners.median():,.0f} |",
        f"| Track's Source is the artist's own name | {self_labelled['functional']:.1%} | "
        f"{self_labelled['genre']:.1%} |",
        f"| Artist names in the track's Source house (mean / median / p90) | {house_size('functional')} | "
        f"{house_size('genre')} |",
        f"| Same, (P) line owner | {house_size('functional', 'pline_owner', houses['(P) line owner'])} | "
        f"{house_size('genre', 'pline_owner', houses['(P) line owner'])} |",
        f"| Same, ISRC registrant (Deezer-matched tracks only) | "
        f"{house_size('functional', 'isrc_registrant', houses['ISRC registrant (Deezer-matched)'])} | "
        f"{house_size('genre', 'isrc_registrant', houses['ISRC registrant (Deezer-matched)'])} |",
        "",
        "House size is bimodal on functional playlists: about half the tracks name the alias itself "
        "as Source (a one-name house), the rest sit in houses with dozens to hundreds of names. "
        "Report the tail (mean, p90, CCDF), not the median.",
        "",
        "## Deezer coverage",
        "",
        f"Deezer matching was attempted for {T.deezer_attempted.sum():,} of {len(T):,} tracks "
        f"({dz_att['functional']:.0%} of functional, {dz_att['genre']:.0%} of genre; "
        f"{n_pl_zero} playlists under 10% coverage). Match rate: {T.deezer_match.mean():.1%} overall, "
        f"{deezer['functional']:.1%} functional vs {deezer['genre']:.1%} genre. "
        "(Data v2, 2026-11-04 19:23, re-ran the enrichment that had stopped partway in v1.)",
        "",
        "## Playlist overlap (Jaccard of artist sets)",
        "",
        jac_by_kind.round(3).to_markdown() if hasattr(jac_by_kind, "to_markdown") else
        jac_by_kind.round(3).to_string(),
        "",
        "Most similar pairs: " + "; ".join(
            f"{r.a} / {r.b} ({r.j:.2f})" for r in top_pairs.itertuples()) + ".",
        "",
        "## Production houses",
        "",
        "Top 10 by distinct artist names (full tables in `data/processed/descriptives_top_*.csv`). "
        "Company names only; see the ethics note in context.md.",
        "",
    ]
    for title, tab in houses.items():
        top = tab.head(10)
        lines += [f"**{title}:** " + ", ".join(
            f"{r.Index} ({r.n_artist_names} names / {r.n_tracks} tracks)" for r in top.itertuples()) + ".", ""]
    lines += [
        "Caveat: classical labels (DG, Decca, Warner Classics) rank high because classical tracks "
        "list composer and performers as artists, so they have more names than tracks. Ghost-style "
        "houses have fewer names than tracks but a large number of single-name aliases. Phase 3 "
        "should count primary-artist names, or use names per track, to separate the two.",
        "",
        "Known gap: the same company can appear under several spellings "
        "(e.g. `ScoreAScore` and `Songs of Score A Score` in p_line, `Score a Score` as Source). "
        "Phase 3 needs entity resolution across the three fields before building the ownership graph.",
        "",
        "## Figures",
        "",
        "- `figures/descriptives_playlists.pdf`",
        "- `figures/descriptives_jaccard.pdf`",
        "- `figures/descriptives_distributions.pdf`",
        "- `figures/descriptives_top_houses.pdf`",
    ]
    notes = PROJECT / "notes"
    notes.mkdir(exist_ok=True)
    (notes / "descriptives.md").write_text("\n".join(lines) + "\n")
    print("  wrote notes/descriptives.md")


if __name__ == "__main__":
    main()
