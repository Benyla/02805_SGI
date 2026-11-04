"""Phase 6 (exploratory): audio features from Deezer's official 30 s previews.

Sample: for every functional playlist with both kinds, up to K ghost-like and K real tracks
(primary artist label), Deezer-matched only. Preview URLs expire, so each one is refreshed
via api.deezer.com/track/{id} first. Rate: <= 4 requests/s. Audio stays local (gitignored).

    python -m src.features.audio            # download + features
Output: data/processed/audio_features.parquet
"""
from __future__ import annotations

import json
import time
import urllib.request

import numpy as np
import pandas as pd

from src.load import PROCESSED, RAW, SEED, load, split_list
from src.common import placement

K = 8
PREV = RAW / "previews"
UA = {"User-Agent": "dtu-02805-course-project/1.0 (research; low rate)"}


def sample_tracks() -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    t = load("tracks")
    t = t[t.deezer_match.fillna(False)]
    pa = placement()
    pa = pa[pa.is_primary & (pa.group == "functional")].merge(t[["track_id", "deezer_id"]], on="track_id")
    pa["ghost"] = pa.artist_id.map(df.ghost)
    rng = np.random.default_rng(SEED)
    rows = []
    for pid, d in pa.groupby("playlist_id"):
        g, r = d[d.ghost == True], d[d.ghost == False]  # noqa: E712
        k = min(K, len(g), len(r))
        if k == 0:
            continue
        for part in (g, r):
            rows.append(part.iloc[rng.choice(len(part), k, replace=False)])
    s = pd.concat(rows).drop_duplicates("track_id")
    return s[["track_id", "deezer_id", "artist_id", "playlist_id", "ghost"]]


def _get(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
        return r.read()


def download(s: pd.DataFrame) -> None:
    PREV.mkdir(parents=True, exist_ok=True)
    for i, (tid, did) in enumerate(zip(s.track_id, s.deezer_id)):
        out = PREV / f"{tid}.mp3"
        if out.exists():
            continue
        try:
            meta = json.loads(_get(f"https://api.deezer.com/track/{did}"))
            time.sleep(0.25)
            url = meta.get("preview")
            if url:
                out.write_bytes(_get(url))
                time.sleep(0.25)
        except Exception as e:  # keep going; a missing preview is just a missing row
            print(f"  {tid}: {e}")
        if i % 50 == 0:
            print(f"  {i}/{len(s)}", flush=True)


def features(path) -> dict:
    import librosa

    y, sr = librosa.load(path, sr=22050, mono=True)
    if len(y) < sr * 5:
        return {}
    rms = librosa.feature.rms(y=y)[0]
    db = librosa.amplitude_to_db(rms + 1e-9)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    flat = librosa.feature.spectral_flatness(y=y)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr).mean(1)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    return {
        "tempo": float(np.atleast_1d(tempo)[0]),
        "rms_db_mean": float(db.mean()),
        "dyn_range_db": float(np.percentile(db, 95) - np.percentile(db, 5)),
        "centroid_hz": float(cent.mean()),
        "flatness": float(flat.mean()),
        "onset_mean": float(onset.mean()),
        "chroma_peak": float(chroma.max() / chroma.sum()),
        **{f"mfcc{i}": float(v) for i, v in enumerate(mfcc.mean(1))},
        **{f"mfccsd{i}": float(v) for i, v in enumerate(mfcc.std(1))},
    }


def main() -> None:
    s = sample_tracks()
    print(f"sample: {len(s)} tracks ({int(s.ghost.sum())} ghost-like) from {s.playlist_id.nunique()} playlists")
    download(s)
    rows = []
    for r in s.itertuples():
        p = PREV / f"{r.track_id}.mp3"
        if p.exists():
            f = features(p)
            if f:
                rows.append({"track_id": r.track_id, "artist_id": r.artist_id,
                             "playlist_id": r.playlist_id, "ghost": r.ghost, **f})
    out = pd.DataFrame(rows)
    out.to_parquet(PROCESSED / "audio_features.parquet")
    print(f"features for {len(out)} tracks")


if __name__ == "__main__":
    main()
