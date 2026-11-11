"""Phase 6 analysis (exploratory): can the sound of a 30 s preview tell ghost-like from real
tracks on the same functional playlists?  Logistic regression on librosa summary features,
5-fold CV ROC AUC, shuffled-label null.

    python -m src.features.audio_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.load import PROCESSED, SEED


def main() -> None:
    a = pd.read_parquet(PROCESSED / "audio_features.parquet")
    # the sample was drawn with an earlier label; always score it with the current one
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    a["ghost"] = a.artist_id.map(df.ghost).astype(bool)
    X = a.drop(columns=["track_id", "artist_id", "playlist_id", "ghost"]).to_numpy(float)
    y = a.ghost.astype(int).to_numpy()
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.3))
    auc = roc_auc_score(y, cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1])
    rng = np.random.default_rng(SEED)
    null = [roc_auc_score(ys, cross_val_predict(clf, X, ys, cv=cv, method="predict_proba")[:, 1])
            for ys in (rng.permutation(y) for _ in range(20))]
    feats = ["tempo", "rms_db_mean", "dyn_range_db", "centroid_hz", "chroma_peak"]
    out = {
        "auc_audio": {"auc": auc, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
                      "n": int(len(y)), "n_ghost": int(y.sum()), "n_playlists": int(a.playlist_id.nunique())},
        "medians": {f: {"ghost": float(a.loc[a.ghost == True, f].median()),  # noqa: E712
                        "real": float(a.loc[a.ghost == False, f].median()),  # noqa: E712
                        "p": float(mannwhitneyu(a.loc[a.ghost == True, f], a.loc[a.ghost == False, f]).pvalue)}  # noqa: E712
                    for f in feats},
    }
    (PROCESSED / "audio.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
