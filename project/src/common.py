"""Derived keys used across phases: group membership and normalised company names."""
from __future__ import annotations

import re

import pandas as pd

from src.load import load, split_list


def placement() -> pd.DataFrame:
    """One row per (playlist slot, artist): playlist_tracks exploded on artist_ids, with group."""
    pt = load("playlist_tracks").merge(
        load("playlists")[["playlist_id", "name", "group", "category"]], on="playlist_id"
    )
    pt["artist_id"] = split_list(pt.artist_ids)
    pt["artist_rank"] = pt.artist_id.map(lambda l: list(range(len(l))))
    pt = pt.explode(["artist_id", "artist_rank"]).reset_index(drop=True)
    pt["is_primary"] = pt.artist_rank == 0
    return pt


def _group_label(s: pd.Series) -> str:
    u = set(s)
    return "both" if len(u) > 1 else next(iter(u))


def track_group() -> pd.Series:
    """track_id -> 'functional' | 'genre' | 'both' by the playlists the track is on."""
    pt = load("playlist_tracks").merge(load("playlists")[["playlist_id", "group"]])
    return pt.groupby("track_id").group.agg(_group_label).astype("string")


def artist_group() -> pd.Series:
    """artist_id -> 'functional' | 'genre' | 'both' by the playlists the artist appears on."""
    return placement().groupby("artist_id").group.agg(_group_label).astype("string")


_PLINE_PREFIX = re.compile(r"^\s*(?:℗|\(p\)|©|\(c\)|p)?\s*(?:(?:19|20)\d\d\s*[,/-]?\s*)+", re.I)


def pline_owner(p_line: pd.Series) -> pd.Series:
    """Strip the (P)/© marker and leading year(s) from a p_line: '2025 Chill Palm' -> 'Chill Palm'."""
    out = p_line.fillna("").str.replace(_PLINE_PREFIX, "", regex=True).str.strip(" .,")
    return out.replace("", pd.NA).astype("string")

