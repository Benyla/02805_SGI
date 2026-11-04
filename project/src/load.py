"""Load the Spotify 50-playlist snapshot (2026-11-04, DK market) as pandas DataFrames.

Usage:
    from src.load import load_all
    d = load_all()          # dict of DataFrames, keyed by table name
    d["tracks"].head()

Run `python -m src.load` from `project/` to check the joins.

Conventions:
- All IDs are kept as strings (bare Spotify IDs, not URIs).
- Multi-valued columns (`artist_ids`, `artist_names`, ...) are "; "-separated in the CSVs.
  `split_list()` turns them into Python lists.
- `n_pilot_playlists` is renamed to `n_playlists`: the number of the 50 dataset playlists
  an item appears on (the old name is a leftover from the pilot crawl).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
PROCESSED = PROJECT / "data" / "processed"
FIGURES = PROJECT / "figures"

SEP = "; "
SEED = 2805

_BOOL = "boolean"
_STR = "string"

# Explicit dtypes per table. Columns not listed are left to pandas' inference (numeric).
DTYPES: dict[str, dict[str, str]] = {
    "playlists": {
        "playlist_id": _STR, "name": _STR, "group": _STR, "category": _STR,
        "all_categories": _STR, "owner": _STR, "followers": "Int64", "n_tracks": "Int64",
        "description": _STR, "cover_url": _STR, "cover_file": _STR,
    },
    "playlist_tracks": {
        "playlist_id": _STR, "position": "Int64", "track_id": _STR, "artist_ids": _STR,
    },
    "tracks": {
        "track_id": _STR, "name": _STR, "album_id": _STR, "artist_ids": _STR,
        "artist_names": _STR, "n_artists": "Int64", "duration_ms": "Int64",
        "playcount": "Int64", "n_pilot_playlists": "Int64", "label": _STR, "p_line": _STR,
        "n_credits": "Int64", "credited_non_artists": _STR, "n_credited_non_artists": "Int64",
        "sources": _STR, "deezer_match": _BOOL, "deezer_id": _STR, "cover_file": _STR,
        "isrc": _STR, "isrc_country": _STR, "isrc_registrant": _STR, "isrc_year": _STR,
        "deezer_preview": _STR, "deezer_contributors": _STR, "deezer_title_exact": _BOOL,
    },
    "albums": {
        "album_id": _STR, "name": _STR, "type": _STR, "date_precision": _STR, "label": _STR,
        "p_line": _STR, "c_line": _STR, "n_tracks": "Int64", "artist_ids": _STR,
        "total_plays": "Int64", "cover_url": _STR, "cover_file": _STR, "missing": _STR,
    },
    "album_tracks": {
        "album_id": _STR, "track_id": _STR, "track_number": "Int64", "name": _STR,
        "playcount": "Int64", "duration_ms": "Int64",
    },
    "artists": {
        "artist_id": _STR, "name": _STR, "verified": _BOOL, "registered": _BOOL,
        "ai_persona": _STR, "monthly_listeners": "Int64", "followers": "Int64",
        "world_rank": "Int64", "n_top_cities": "Int64", "top_city": _STR,
        "top_countries": _STR, "bio_len": "Int64", "n_links": "Int64", "links": _STR,
        "has_avatar": _BOOL, "avatar_url": _STR, "n_gallery": "Int64", "n_albums": "Int64",
        "n_singles": "Int64", "n_compilations": "Int64", "related_total": "Int64",
        "n_discovered_on": "Int64", "n_pilot_playlists": "Int64", "bio": _STR, "missing": _STR,
    },
    "artist_top_cities": {
        "artist_id": _STR, "rank": "Int64", "city": _STR, "country": _STR, "listeners": "Int64",
    },
    "credits": {
        "track_id": _STR, "name": _STR, "role": _STR, "role_group": _STR,
        "contributor_artist_id": _STR,
    },
    "track_sources": {"track_id": _STR, "source": _STR},
    "fans_also_like_edges": {
        "source_artist_id": _STR, "target_artist_id": _STR, "target_name": _STR, "rank": "Int64",
    },
    "discovered_on_edges": {
        "artist_id": _STR, "playlist_id": _STR, "playlist_name": _STR, "playlist_owner": _STR,
        "rank": "Int64",
    },
}

DATES = {
    "playlists": ["snapshot_date"],
    "playlist_tracks": ["added_at", "snapshot_date"],
    "tracks": ["release_date"],
    "albums": ["release_date"],
}

RENAME = {"n_pilot_playlists": "n_playlists"}


def split_list(s: pd.Series) -> pd.Series:
    """Split a "; "-separated string column into lists (NA -> empty list)."""
    return s.fillna("").map(lambda x: [v for v in x.split(SEP) if v] if x else [])


@lru_cache(maxsize=None)
def load(name: str) -> pd.DataFrame:
    path = RAW / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing. Unzip data/raw/archives/spotify_50_playlists.zip into data/raw/."
        )
    df = pd.read_csv(path, dtype=DTYPES.get(name), keep_default_na=True)
    for col in DATES.get(name, []):
        if col in df:
            # albums with date_precision=YEAR have "YYYY"; coerce keeps them as Jan 1.
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=(col == "added_at"))
    return df.rename(columns=RENAME)


def load_all() -> dict[str, pd.DataFrame]:
    return {name: load(name).copy() for name in DTYPES}


def check_joins(d: dict[str, pd.DataFrame] | None = None) -> dict[str, int]:
    """Acceptance check from agent_plan Phase 0. Returns counts of dangling keys."""
    d = d or load_all()
    t, al, ar, pt = d["tracks"], d["albums"], d["artists"], d["playlist_tracks"]
    track_ids, album_ids, artist_ids = set(t.track_id), set(al.album_id), set(ar.artist_id)

    track_artists = {a for lst in split_list(t.artist_ids) for a in lst}
    pt_artists = {a for lst in split_list(pt.artist_ids) for a in lst}
    out = {
        "playlist_tracks.track_id not in tracks": int((~pt.track_id.isin(track_ids)).sum()),
        "playlist_tracks.playlist_id not in playlists":
            int((~pt.playlist_id.isin(set(d["playlists"].playlist_id))).sum()),
        "tracks.album_id not in albums": int((~t.album_id.isin(album_ids)).sum()),
        "tracks.artist_ids not in artists (distinct)": len(track_artists - artist_ids),
        "playlist_tracks.artist_ids not in artists (distinct)": len(pt_artists - artist_ids),
        "track_sources.track_id not in tracks":
            int((~d["track_sources"].track_id.isin(track_ids)).sum()),
        "credits.track_id not in tracks": int((~d["credits"].track_id.isin(track_ids)).sum()),
        "tracks without a track_sources row": len(track_ids - set(d["track_sources"].track_id)),
        "album_tracks.album_id not in albums":
            int((~d["album_tracks"].album_id.isin(album_ids)).sum()),
        "fans_also_like targets outside artists (rows)":
            int((~d["fans_also_like_edges"].target_artist_id.isin(artist_ids)).sum()),
    }
    return out


if __name__ == "__main__":
    d = load_all()
    print("Rows per table:")
    for k, v in d.items():
        print(f"  {k:<22} {len(v):>8,}")
    print("\nJoin checks (0 = clean for the first five):")
    for k, v in check_joins(d).items():
        print(f"  {k:<55} {v:>8,}")
