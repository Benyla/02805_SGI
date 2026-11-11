"""Production dimension: who makes the catalogue behind an artist name.

Three company fields per track ("houses"):
  - source:      the credits dialog's Source (label or distributor), lightly normalised
  - pline_owner: the (P) line without year/marker
  - registrant:  ISRC country + registrant code (first 5 ISRC chars), e.g. "SE69Z".
                 Note: context.md's `isrc_registrant` is only chars 3-5, which collides
                 across countries; we use the 5-char key.

House size = number of distinct *primary* artist names on tracks carrying that key.
(Primary only, so classical tracks crediting Bach + three performers do not inflate
the label's name count.) For the registrant, the paper's definition is stricter: the number
of names whose *dominant* registrant it is (see artist_production).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import pline_owner
from src.load import load, split_list

HOUSE_KEYS = ["source", "pline_owner", "registrant"]


def norm_company(s: pd.Series) -> pd.Series:
    """'Score a Score' and 'ScoreAScore' -> 'scoreascore'. Keeps distinct names distinct."""
    return (s.str.casefold().str.replace(r"[^\w]+", "", regex=True)
            .replace("", pd.NA).astype("string"))


def track_houses() -> pd.DataFrame:
    t = load("tracks").merge(load("track_sources"), on="track_id", how="left")
    t["primary_artist"] = split_list(t.artist_ids).str[0]
    t["primary_name"] = t.artist_names.str.split(";").str[0].str.strip()
    t["source_raw"] = t.source
    t["source"] = norm_company(t.source)
    t["pline_owner_raw"] = pline_owner(t.p_line)
    t["pline_owner"] = norm_company(t.pline_owner_raw)
    t["registrant"] = t.isrc.str[:5].astype("string")
    t["self_source"] = t.source == norm_company(t.primary_name)
    t["alias_only"] = t.n_credited_non_artists.fillna(0) == 0
    return t


def house_sizes(t: pd.DataFrame) -> dict[str, pd.Series]:
    return {k: t.dropna(subset=[k]).groupby(k).primary_artist.nunique() for k in HOUSE_KEYS}


def _mode(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if len(s) else pd.NA


def artist_production() -> pd.DataFrame:
    t = track_houses()
    sizes = house_sizes(t)
    # every artist credited on a track (not only primary) gets that track's houses
    x = t.assign(artist_id=split_list(t.artist_ids)).explode("artist_id")
    g = x.groupby("artist_id")
    out = pd.DataFrame({
        "n_tracks": g.track_id.nunique(),
        "alias_only_share": g.alias_only.mean(),
        "self_source_share": g.self_source.mean(),
    })
    for k in HOUSE_KEYS:
        dom = g[k].agg(_mode)
        out[f"{k}_dom"] = dom
        out[f"{k}_house_size"] = dom.map(sizes[k]).astype("float")
    # The registrant ("house") uses ONE definition everywhere in the paper: a name's house is
    # the registrant of most of the tracks where it is the primary artist, and a house's size
    # is the number of names whose house it is.
    p = t.dropna(subset=["registrant"]).groupby(["primary_artist", "registrant"]).size()
    p = p.reset_index(name="n").sort_values(["n", "registrant"], ascending=[False, True])
    dom = p.drop_duplicates("primary_artist").set_index("primary_artist").registrant
    out["registrant_dom"] = dom.reindex(out.index).astype("string")
    out["registrant_house_size"] = out.registrant_dom.map(dom.value_counts()).astype("float")
    out["source_dom_raw"] = g.source_raw.agg(_mode)
    out["pline_dom_raw"] = g.pline_owner_raw.agg(_mode)
    # largest house the name belongs to, over the three fields
    out["max_house_size"] = out[[f"{k}_house_size" for k in HOUSE_KEYS]].max(axis=1)
    out["log_registrant_house"] = np.log10(out.registrant_house_size)
    out["log_label_house"] = np.log10(out[["source_house_size", "pline_owner_house_size"]].max(axis=1))
    return out
