"""Phase 3.1 + 3.2: ownership (names -> houses) and placement (playlists -> names).

Ownership: bipartite graph of artist names and ISRC registrants ("houses"); an edge means
the name has a track registered by that house. House size = degree of the house node,
counted in primary-artist names.

Placement: share of playlist slots held by ghost-like names, per playlist.

    python -m src.networks.ownership
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.common import placement
from src.features.production import track_houses
from src.load import PROCESSED, load


def main() -> None:
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    t = track_houses()
    t["ghost"] = t.primary_artist.map(df.ghost)

    # ---- ownership: bipartite edges name -- registrant (primary artists, Deezer-matched tracks)
    B = t.dropna(subset=["registrant"]).groupby(["primary_artist", "registrant"]).size().rename("n_tracks")
    B = B.reset_index()
    B.to_csv(PROCESSED / "ownership_edges.csv", index=False)
    size = B.groupby("registrant").primary_artist.nunique().sort_values(ascending=False)
    ghost_names = set(df.index[df.ghost])
    B["ghost"] = B.primary_artist.isin(ghost_names)
    # names in each house split by label
    by_house = B.groupby("registrant").agg(n_names=("primary_artist", "nunique"),
                                           n_ghost=("ghost", "sum"))
    by_house["country"] = by_house.index.str[:2]
    by_house.sort_values("n_names", ascending=False).to_csv(PROCESSED / "ownership_houses.csv")

    # house size seen from each name (the CCDF in Fig. 1)
    name_house = df.registrant_house_size.dropna()
    g_sizes = name_house[df.ghost.reindex(name_house.index)].to_numpy()
    r_sizes = name_house[~df.ghost.reindex(name_house.index)].to_numpy()

    # ---- placement
    pa = placement()
    pa = pa[pa.is_primary].copy()
    pa["ghost"] = pa.artist_id.map(df.ghost)
    pa["registrant"] = pa.artist_id.map(df.registrant_dom)
    fslots = pa[pa.group == "functional"]
    gslots = fslots[fslots.ghost == True]  # noqa: E712
    top_houses = gslots.registrant.value_counts()
    k8 = top_houses.head(8)
    share_top8 = float(k8.sum() / gslots.registrant.notna().sum())

    pl = load("playlists").set_index("playlist_id")
    per_pl = (pa.groupby("playlist_id").ghost.mean().rename("ghost_share").to_frame()
              .join(pl[["name", "group", "category", "followers", "n_tracks"]]))
    per_pl.to_csv(PROCESSED / "placement_playlists.csv")
    f = per_pl[per_pl.group == "functional"]
    rho = spearmanr(f.ghost_share, f.followers)

    # plays: lifetime playcount of tracks on functional playlists, by primary-artist label
    tp = t[t.track_id.isin(fslots.track_id)]
    plays = tp.groupby("ghost").playcount.sum()

    out = {
        "n_houses": int(len(size)),
        "n_names_with_registrant": int(df.registrant_dom.notna().sum()),
        "ghost_names": int(df.ghost.sum()),
        "ghost_names_with_registrant": int(df.loc[df.ghost, "registrant_dom"].notna().sum()),
        "ghost_distinct_houses": int(df.loc[df.ghost, "registrant_dom"].nunique()),
        "top8_houses": k8.index.tolist(),
        "top8_share_of_ghost_slots": share_top8,
        "top8_swedish": int(sum(h.startswith("SE") for h in k8.index)),
        "se_share_ghost_names": float(df.loc[df.ghost, "registrant_dom"].dropna().str.startswith("SE").mean()),
        "se_share_real_names": float(df.loc[~df.ghost, "registrant_dom"].dropna().str.startswith("SE").mean()),
        "median_house_ghost": float(np.median(g_sizes)), "median_house_real": float(np.median(r_sizes)),
        "functional_slot_ghost_share": float(fslots.ghost.mean()),
        "genre_slot_ghost_share": float(pa[pa.group == "genre"].ghost.mean()),
        "rho_ghostshare_followers": rho.statistic, "rho_p": rho.pvalue,
        "plays_functional_ghost": int(plays.get(True, 0)), "plays_functional_real": int(plays.get(False, 0)),
        "self_source_share_ghost": float(df.loc[df.ghost, "self_source_share"].gt(0.5).mean()),
        "self_source_share_real_functional": float(
            df.loc[(~df.ghost) & (df.group == "functional"), "self_source_share"].gt(0.5).mean()),
        "self_source_house_median": float(df.loc[df.self_source_share > 0.5, "registrant_house_size"].median()),
    }
    # robustness: same concentration with the production-free label
    r = df.loc[df.ghost_noprod, "registrant_dom"].dropna()
    vc = r.value_counts()
    out["noprod"] = {
        "ghost_names": int(df.ghost_noprod.sum()), "with_registrant": int(len(r)),
        "agreement_with_full_label": float((df.ghost_noprod == df.ghost).mean()),
        "distinct_houses": int(r.nunique()), "top8_share_names": float(vc.head(8).sum() / len(r)),
        "top10_share_names": float(vc.head(10).sum() / len(r)),
        "se_share": float(r.str.startswith("SE").mean()),
    }
    (PROCESSED / "ownership.json").write_text(json.dumps(out, indent=1, default=float))
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
