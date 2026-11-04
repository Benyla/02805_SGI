"""Identity dimension: is the name an identifiable person or band?

Five binary markers of a public identity, each 1 if *missing*:
  not verified, no bio, no external links, no gallery images, no Spotify "This Is" playlist
  (Spotify only builds "This Is <artist>" for established artists; taken from discovered_on).
identity_score = share of the five markers that are missing (0 = full identity, 1 = none).
"""
from __future__ import annotations

import pandas as pd

from src.load import load

MARKERS = ["no_verified", "no_bio", "no_links", "no_gallery", "no_this_is"]


def artist_identity() -> pd.DataFrame:
    a = load("artists").set_index("artist_id")
    d = load("discovered_on_edges")
    this_is = (d.assign(t=d.playlist_name.str.startswith("This Is ").fillna(False).astype(bool)
                         & d.playlist_owner.eq("Spotify").fillna(False).astype(bool))
               .groupby("artist_id").t.any())
    out = pd.DataFrame(index=a.index)
    out["no_verified"] = ~a.verified.fillna(False)
    out["no_bio"] = a.bio_len.fillna(0) == 0
    out["no_links"] = a.n_links.fillna(0) == 0
    out["no_gallery"] = a.n_gallery.fillna(0) == 0
    out["no_this_is"] = ~this_is.reindex(a.index).fillna(False).astype(bool)
    out = out.astype(float)
    out["identity_score"] = out[MARKERS].mean(axis=1)
    out["bio_len"] = a.bio_len
    return out
