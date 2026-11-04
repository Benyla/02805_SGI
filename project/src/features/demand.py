"""Demand dimension: do streams come from fans or from placement?

  log_fpl:        log10 of (followers + 1) / (monthly listeners + 1). Fans follow; playlist
                  listeners do not.
  top_is_ours:    the artist's #1 "Discovered on" playlist is one of the 50 dataset playlists
                  (the artist depends on one editorial placement).
  top_city_share: share of monthly listeners in the #1 city (used negated). A fan base is
                  concentrated somewhere; playlist listeners are spread around the world.
  n_countries:    distinct countries among the top-5 cities (kept as a column, not in the score:
                  a 1-5 count is too coarse).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.load import load


def artist_demand() -> pd.DataFrame:
    a = load("artists").set_index("artist_id")
    d = load("discovered_on_edges")
    ours = set(load("playlists").playlist_id)
    top = d[d["rank"] == 1].drop_duplicates("artist_id").set_index("artist_id")
    tc = load("artist_top_cities")
    out = pd.DataFrame(index=a.index)
    out["monthly_listeners"] = a.monthly_listeners.astype(float)
    out["followers"] = a.followers.astype(float)
    out["log_fpl"] = np.log10((a.followers.astype(float) + 1) / (a.monthly_listeners.astype(float) + 1))
    out["top_is_ours"] = top.playlist_id.isin(ours).reindex(a.index).astype(float)
    out["n_countries"] = tc.groupby("artist_id").country.nunique().reindex(a.index).astype(float)
    out["top_city_share"] = a.top_city_share
    return out
