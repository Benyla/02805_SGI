"""Build data/processed/artist_features.parquet: one row per artist, three dimension scores.

Each score is oriented so that higher = more ghost-like. The functional/genre playlist label
is NOT used in any score (it is kept only as a column for validation).

    python -m src.features.build
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import artist_group
from src.features.demand import artist_demand
from src.features.identity import artist_identity
from src.features.production import artist_production
from src.load import PROCESSED, load


def z(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def build() -> pd.DataFrame:
    a = load("artists").set_index("artist_id")[["name", "n_playlists"]]
    df = a.join(artist_identity()).join(artist_production()).join(artist_demand())
    df["group"] = artist_group().reindex(df.index)

    # identity: already a 0-1 share of missing markers
    df["identity_z"] = z(df.identity_score)
    # production: the ISRC registrant's house is big + credits name only the alias.
    # The Source/(P)-line house size is NOT used: half of the functional names list themselves as
    # their own label (self_source_share), which hides the house; see notes/convergence.md.
    prod = pd.concat([z(df.log_registrant_house), z(df.alias_only_share)], axis=1)
    df["production_z"] = prod.mean(axis=1, skipna=True)
    # demand: few followers per listener, depends on one of our playlists, listeners not
    # concentrated in any home city
    dem = pd.concat([z(-df.log_fpl), z(df.top_is_ours), z(-df.top_city_share)], axis=1)
    df["demand_z"] = dem.mean(axis=1, skipna=True)
    return df


if __name__ == "__main__":
    df = build()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "artist_features.parquet")
    print(df[["identity_z", "production_z", "demand_z"]].describe().round(2))
    print(df.groupby("group")[["identity_z", "production_z", "demand_z"]].median().round(2))
