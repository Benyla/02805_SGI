"""Phase 5 analysis (exploratory): what covers say about ghost-like names and houses.

  - CLIP cover AUC: logistic regression on each functional artist's mean cover embedding,
    5-fold CV, predicting the ghost-like label (null: shuffled labels).
  - House style: for each cover, its most similar cover by a *different* name; share from the
    same ISRC house vs a label-shuffle null.
  - Literalness: CLIP cosine between a track title ("album cover for the song <title>") and its
    own cover, ghost vs real.
  - pHash reuse: near-identical covers (Hamming <= 6) under different names.

    python -m src.features.vision_analysis
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.load import PROCESSED, SEED, load, split_list


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    ids = (PROCESSED / "cover_ids.txt").read_text().split("\n")
    E = np.load(PROCESSED / "cover_clip.npy")
    ix = {a: i for i, a in enumerate(ids)}
    al = load("albums")
    al["pa"] = split_list(al.artist_ids).str[0]
    al = al[al.album_id.isin(ix)].reset_index(drop=True)
    al["ghost"] = al.pa.map(df.ghost)
    al["reg"] = al.pa.map(df.registrant_dom).astype(object)
    out: dict = {}

    # AUC
    f = df[df.group == "functional"]
    g = al[al.pa.isin(f.index)].groupby("pa").album_id.apply(list)
    X = np.stack([E[[ix[a] for a in l]].mean(0) for l in g])
    y = f.loc[g.index].ghost.to_numpy(int)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    auc = roc_auc_score(y, cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1])
    null = [roc_auc_score(ys, cross_val_predict(clf, X, ys, cv=cv, method="predict_proba")[:, 1])
            for ys in (rng.permutation(y) for _ in range(10))]
    out["auc_cover"] = {"auc": auc, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
                        "n": int(len(y)), "n_ghost": int(y.sum())}

    # house style
    A = al[al.reg.notna()].reset_index(drop=True)
    M = E[[ix[a] for a in A.album_id]]
    S = M @ M.T
    S[A.pa.values[:, None] == A.pa.values[None, :]] = -1
    nn = S.argmax(1)
    reg = A.reg.to_numpy(str)
    obs = float((reg == reg[nn]).mean())
    null = [float((r == r[nn]).mean()) for r in (rng.permutation(reg) for _ in range(500))]
    out["cover_nn_same_house"] = {"observed": obs, "null_mean": float(np.mean(null)),
                                  "null_sd": float(np.std(null)), "n_covers": int(len(A))}

    # literalness
    T = np.load(PROCESSED / "title_clip.npy")
    tids = (PROCESSED / "title_track_ids.txt").read_text().split("\n")
    t = load("tracks").set_index("track_id").loc[tids]
    has = t.album_id.isin(ix).to_numpy()
    lit = np.full(len(t), np.nan)
    lit[has] = (T[has] * E[[ix[a] for a in t.album_id[has]]]).sum(1)
    t["lit"] = lit
    t["ghost"] = split_list(t.artist_ids).str[0].map(df.ghost).values
    out["literalness_median"] = t.groupby("ghost").lit.median().rename(str).to_dict()

    # pHash reuse
    m = pd.read_parquet(PROCESSED / "covers_meta.parquet").dropna(subset=["phash"])
    m["pa"] = split_list(load("albums").set_index("album_id").artist_ids.reindex(m.album_id)).str[0].values
    h = np.array([int(x, 16) for x in m.phash], dtype=np.uint64)
    bits = np.unpackbits(h.view(np.uint8).reshape(-1, 8), axis=1).astype(bool)
    pairs = []
    for i in range(0, len(bits), 500):
        D = (bits[i:i + 500, None, :] != bits[None, :, :]).sum(2)
        a, b = np.where(D <= 6)
        k = a + i < b
        pairs += list(zip(a[k] + i, b[k]))
    pa = m.pa.to_numpy(str)
    P = [(i, j) for i, j in pairs if pa[i] != pa[j]]
    out["phash_pairs_across_names"] = len(P)

    (PROCESSED / "vision.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
