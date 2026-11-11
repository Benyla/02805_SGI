"""Build data/processed/artist_features.parquet: one row per artist with the identity markers,
the production fields (house, credits) and the demand fields (followers, listeners, cities).

The functional/genre playlist group is kept only as a column for validation; it is never
used to flag a name.

    python -m src.features.build
"""
from __future__ import annotations

from src.common import artist_group
from src.features.demand import artist_demand
from src.features.identity import artist_identity
from src.features.production import artist_production
from src.load import PROCESSED, load


def build():
    a = load("artists").set_index("artist_id")[["name", "n_playlists"]]
    df = a.join(artist_identity()).join(artist_production()).join(artist_demand())
    df["group"] = artist_group().reindex(df.index)
    return df


if __name__ == "__main__":
    df = build()
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PROCESSED / "artist_features.parquet")
    print(df.groupby("group")[["identity_score", "registrant_house_size", "log_fpl"]].median().round(2))
