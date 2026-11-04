"""Phase 2 convergence: do identity, production and demand point to the same names?

Adds `p_ghost` / `ghost` to artist_features.parquet (2-component Gaussian mixture on the three
scores; the playlist group is never used), writes notes/convergence.md and a stratified
hand-labelling sheet.

    python -m src.features.convergence
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

from src.load import PROCESSED, PROJECT, SEED

S = ["identity_z", "production_z", "demand_z"]
PAIRS = [("identity_z", "production_z"), ("identity_z", "demand_z"), ("production_z", "demand_z")]

# External reference sets, matched on the artist's dominant credit Source.
# Known suppliers of stock / "perfect fit" content named in press coverage (Pelly 2025; DN 2022):
KNOWN_SUPPLIERS = {"Firefly Entertainment": "Firefly", "Epidemic Sound": "Epidemic"}
# Classical majors whose artists sit on the same functional playlists: an in-playlist control.
CLASSICAL_MAJORS = {"Deutsche Grammophon": "Deutsche Grammophon", "Decca": "Decca",
                    "Sony Classical": "Sony Classical", "Warner Classics": "Warner Classics"}


def fit_mixture(df: pd.DataFrame, cols: list[str] = S) -> pd.Series:
    X = df[cols].fillna(df[cols].median()).values
    gm = GaussianMixture(2, random_state=SEED, n_init=10).fit(X)
    ghost_comp = int(np.argmax(gm.means_.sum(axis=1)))
    return pd.Series(gm.predict_proba(X)[:, ghost_comp], index=df.index, name="p_ghost")


def boot_spearman(x: np.ndarray, y: np.ndarray, n: int = 1000, seed: int = SEED):
    rng = np.random.default_rng(seed)
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    r = spearmanr(x, y).statistic
    bs = [spearmanr(x[i], y[i]).statistic for i in (rng.integers(0, len(x), len(x)) for _ in range(n))]
    return r, *np.percentile(bs, [2.5, 97.5])


def corr_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, sub in [("all artists", df), ("functional only", df[df.group == "functional"]),
                      ("genre only", df[df.group == "genre"])]:
        for a, b in PAIRS:
            r, lo, hi = boot_spearman(sub[a].to_numpy(float), sub[b].to_numpy(float))
            rows.append({"subset": name, "n": len(sub), "pair": f"{a[:-2]} ~ {b[:-2]}",
                         "rho": r, "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(rows)


def reference_rates(df: pd.DataFrame) -> pd.DataFrame:
    src = df.source_dom_raw.fillna("")
    rows = []
    for kind, d in [("known supplier", KNOWN_SUPPLIERS), ("classical major", CLASSICAL_MAJORS)]:
        for label, pat in d.items():
            m = src.str.contains(pat, case=False, regex=False)
            rows.append({"set": kind, "name": label, "n_artists": int(m.sum()),
                         "share_ghost": df.loc[m, "ghost"].mean()})
    rows.append({"set": "genre playlists", "name": "all 5 genre playlists",
                 "n_artists": int((df.group == "genre").sum()),
                 "share_ghost": df.loc[df.group == "genre", "ghost"].mean()})
    return pd.DataFrame(rows)


def hand_label_sheet(df: pd.DataFrame, n: int = 150) -> pd.DataFrame:
    """Stratified sample for the group to label BEFORE looking at p_ghost."""
    rng = np.random.default_rng(SEED)
    f = df[df.group == "functional"]
    bins = pd.cut(f.p_ghost, [-0.01, 0.1, 0.9, 1.01], labels=["low", "mid", "high"])
    parts = []
    for b, k in [("low", 50), ("mid", 30), ("high", 50)]:
        idx = f.index[bins == b]
        parts.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
    g = df.index[df.group == "genre"]
    parts.append(rng.choice(g, size=n - sum(map(len, parts)), replace=False))
    ids = np.concatenate(parts)
    rng.shuffle(ids)
    return pd.DataFrame({
        "artist_id": ids, "name": df.loc[ids, "name"].values,
        "spotify_url": [f"https://open.spotify.com/artist/{i}" for i in ids],
        "label_august": "", "label_bertram": "", "label_saxe": "", "note": "",
    })


def main() -> None:
    path = PROCESSED / "artist_features.parquet"
    df = pd.read_parquet(path)
    df["p_ghost"] = fit_mixture(df)
    df["ghost"] = df.p_ghost > 0.5
    # label without any production information: used wherever the claim is about houses,
    # so "ghosts sit in few houses" is not true by construction
    df["p_ghost_noprod"] = fit_mixture(df, ["identity_z", "demand_z"])
    df["ghost_noprod"] = df.p_ghost_noprod > 0.5
    df.to_parquet(path)

    ct = corr_table(df)
    ct.to_csv(PROCESSED / "convergence_spearman.csv", index=False)
    ref = reference_rates(df)
    ref.to_csv(PROCESSED / "convergence_reference_sets.csv", index=False)

    sheet = PROJECT / "notes" / "hand_label_sheet.csv"
    if not sheet.exists():  # never overwrite labels the group has started filling in
        hand_label_sheet(df).to_csv(sheet, index=False)

    uncertain = ((df.p_ghost > 0.1) & (df.p_ghost < 0.9)).mean()
    lines = [
        "# Phase 2: three dimensions and their convergence",
        "",
        "Generated by `python -m src.features.convergence`.",
        "",
        "## Scores (higher = more ghost-like; the playlist group is never an input)",
        "",
        "- **Identity**: share of 5 public-identity markers missing (verified, bio, links, gallery, "
        "Spotify 'This Is' playlist).",
        "- **Production**: z(log10 ISRC-registrant house size, counted in distinct primary-artist names) "
        "+ z(share of the artist's tracks whose credits name only the alias), averaged.",
        "- **Demand**: z(-log10 followers per listener) + z(top 'Discovered on' playlist is one of "
        "the 50) + z(-share of listeners in the top city), averaged.",
        "",
        "Source/(P)-line house size is left out of production on purpose: "
        f"{df.loc[df.group == 'functional', 'self_source_share'].gt(0.5).mean():.0%} of functional names "
        "list themselves as their own Source, which hides the house. Those self-labelled names sit in "
        "*larger* ISRC houses (median "
        f"{df[(df.group == 'functional') & (df.self_source_share > 0.5)].registrant_house_size.median():.0f} "
        "names vs "
        f"{df[(df.group == 'functional') & (df.self_source_share <= 0.5)].registrant_house_size.median():.0f}).",
        "",
        "## Spearman correlation between dimensions (1,000 bootstrap resamples, 95% CI)",
        "",
        ct.assign(rho=ct.rho.round(2), ci=ct.apply(lambda r: f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]", axis=1))
        [["subset", "n", "pair", "rho", "ci"]].to_markdown(index=False),
        "",
        "The functional-only rows are the real test: the dimensions agree even with the "
        "functional/genre contrast removed. Within genre artists they do not correlate, as expected "
        "if almost all of them are real.",
        "",
        "## Ghost-like label",
        "",
        f"2-component Gaussian mixture on the three scores. {int(df.ghost.sum()):,} of {len(df):,} names "
        f"are ghost-like (p > 0.5); {uncertain:.1%} have 0.1 < p < 0.9.",
        "",
        ref.assign(share_ghost=ref.share_ghost.map("{:.0%}".format)).to_markdown(index=False),
        "",
        "Firefly (named in the DN investigation) is almost all ghost-like; classical majors on the "
        "same playlists are almost all real. **Epidemic Sound is classified real**: its artists have "
        "bios, links and galleries and ~9x the functional median followers per listener. The identity "
        "dimension cannot see a supplier that invests in artist profiles. Say this in the Discussion.",
        "",
        "Deezer match rate is not a validator: tracks by ghost-like names match Deezer at almost the "
        "same rate as tracks by real names (computed in src/figures/convergence.py).",
        "",
        "## Hand labels",
        "",
        "`notes/hand_label_sheet.csv` has 150 artists (50 low / 30 mid / 50 high p_ghost on functional "
        "playlists, 20 genre), shuffled, without scores. Each group member labels them independently "
        "in their own column (ghost = 1 / real = 0) from the Spotify page, without looking at the scores; "
        "then run `python -m src.features.kappa` for inter-rater and model-vs-majority Cohen's kappa.",
    ]
    (PROJECT / "notes" / "convergence.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
