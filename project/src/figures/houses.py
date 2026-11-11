"""The eight largest houses, named the same way in every figure.

Ranked by the number of functional-playlist slots whose track they registered (no ghost label
involved). "House 1" is the largest. The company is the most common credit Source on the
house's tracks, ignoring names that list themselves as their own label; if no company covers
at least 15% of the tracks, the house shows "no company in credits".
"""
from __future__ import annotations

import json

import pandas as pd

from src.features.production import track_houses
from src.load import PROCESSED


def top_houses() -> list[str]:
    own = json.loads((PROCESSED / "ownership.json").read_text())
    return own["label_free"]["functional"]["top8"]


def company(t: pd.DataFrame | None = None) -> dict[str, str]:
    t = track_houses() if t is None else t
    out = {}
    for reg, d in t.dropna(subset=["registrant"]).groupby("registrant"):
        s = d.loc[~d.self_source, "source_raw"].value_counts()
        out[reg] = s.index[0] if len(s) and s.iloc[0] >= 0.15 * len(d) else "no company in credits"
    return out


def display_names() -> dict[str, str]:
    return {h: f"House {k}" for k, h in enumerate(top_houses(), start=1)}
