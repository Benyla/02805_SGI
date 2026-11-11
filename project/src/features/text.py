"""Phase 4: text. What can the words a listener sees tell us about ghost-like names?

Within functional-playlist artists only (so the playlist contrast cannot do the work):
  1. Classifiers (TF-IDF + logistic regression, 5-fold CV, ROC AUC) predicting the ghost-like
     label from (a) the artist name (char 2-4-grams), (b) the artist's track titles, (c) the bio
     (artists with a bio only). Null: the same pipeline on shuffled labels (20 runs).
  2. Bio named entities (spaCy en_core_web_sm): PERSON, GPE/LOC, DATE, ORG per bio. Real bios
     tell a life story (places, years, collaborators); template bios describe a mood.
  3. Distinctive bio words: weighted log-odds with an informative Dirichlet prior
     (Monroe, Colaresi & Quinn 2008), ghost-like vs real bios.
  4. Name form: share of two-word "Firstname Lastname" names.
  5. Title reuse: share of tracks whose (normalised) title is also used by another name.

    python -m src.features.text
"""
from __future__ import annotations

import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from src.load import PROCESSED, SEED, load, split_list

PERSON_NAME = r"[A-ZÀ-Þ][a-zà-ÿ]+ [A-ZÀ-Þ][a-zà-ÿ]+"


def _auc(docs: list[str], y: np.ndarray, n_null: int = 20, **vec) -> dict:
    X = TfidfVectorizer(sublinear_tf=True, min_df=2, **vec).fit_transform(docs)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    clf = LogisticRegression(max_iter=3000, C=1.0)
    p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(n_null):
        ys = rng.permutation(y)
        null.append(roc_auc_score(ys, cross_val_predict(clf, X, ys, cv=cv, method="predict_proba")[:, 1]))
    return {"auc": roc_auc_score(y, p), "null_mean": float(np.mean(null)), "null_sd": float(np.std(null)),
            "n": int(len(y)), "n_ghost": int(y.sum())}


def clean_bio(bio: str, name: str) -> str:
    t = re.sub(r"<[^>]+>", " ", bio)
    t = re.sub(re.escape(name), " ", t, flags=re.I)
    return re.sub(r"\s+", " ", t).strip()


def log_odds(a: list[str], b: list[str], top: int = 15) -> pd.DataFrame:
    """Monroe et al. (2008) weighted log-odds with informative Dirichlet prior; z > 0 favours a."""
    cv = CountVectorizer(stop_words="english", token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", min_df=3)
    X = cv.fit_transform(a + b)
    ya = np.asarray(X[: len(a)].sum(0)).ravel()
    yb = np.asarray(X[len(a):].sum(0)).ravel()
    prior = ya + yb + 0.01
    a0, na, nb = prior.sum(), ya.sum(), yb.sum()
    d = (np.log((ya + prior) / (na + a0 - ya - prior)) - np.log((yb + prior) / (nb + a0 - yb - prior)))
    var = 1 / (ya + prior) + 1 / (yb + prior)
    z = d / np.sqrt(var)
    words = np.array(cv.get_feature_names_out())
    df = pd.DataFrame({"word": words, "z": z, "n_ghost": ya, "n_real": yb})
    return pd.concat([df.nlargest(top, "z"), df.nsmallest(top, "z")])


def main() -> None:
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    f = df[df.group == "functional"].copy()
    a = load("artists").set_index("artist_id")
    t = load("tracks")
    t["pa"] = split_list(t.artist_ids).str[0]
    out: dict = {}

    # 1. classifiers
    y = f.ghost.to_numpy(int)
    out["auc_name"] = _auc(f.name.tolist(), y, analyzer="char_wb", ngram_range=(2, 4))
    titles = t.groupby("pa").name.apply(" ".join).reindex(f.index).fillna("")
    out["auc_titles"] = _auc(titles.tolist(), y, ngram_range=(1, 2))
    bio = a.bio.reindex(f.index).fillna("")
    has = bio.str.len() > 40
    fb = f[has]
    bios = [clean_bio(b, n) for b, n in zip(bio[has], fb.name)]
    out["auc_bio"] = _auc(bios, fb.ghost.to_numpy(int), ngram_range=(1, 2))
    # baseline: how far does bio length alone get? (shorter = more ghost-like)
    lens = np.array([len(b.split()) for b in bios])
    out["auc_bio_length"] = float(roc_auc_score(fb.ghost.to_numpy(int), -lens))
    out["bio_presence"] = {"ghost": float(has[f.ghost].mean()), "real": float(has[~f.ghost].mean())}

    # 2. named entities in bios
    import spacy

    nlp = spacy.load("en_core_web_sm", disable=["parser", "lemmatizer"])
    ents = []
    for doc in nlp.pipe(bios, batch_size=64):
        c = Counter(e.label_ for e in doc.ents)
        ents.append({"PERSON": c["PERSON"], "PLACE": c["GPE"] + c["LOC"], "DATE": c["DATE"],
                     "ORG": c["ORG"], "n_tokens": len(doc)})
    E = pd.DataFrame(ents, index=fb.index)
    E["ghost"] = fb.ghost.values
    E["any_entity"] = E[["PERSON", "PLACE", "DATE", "ORG"]].sum(1) > 0
    E.to_csv(PROCESSED / "bio_entities.csv")
    per100 = E.groupby("ghost")[["PERSON", "PLACE", "DATE", "ORG"]].sum().div(
        E.groupby("ghost").n_tokens.sum(), axis=0) * 100
    out["bio_entities_per_100_tokens"] = {str(k): v for k, v in per100.round(3).to_dict("index").items()}
    out["bio_any_entity"] = E.groupby("ghost").any_entity.mean().rename(str).to_dict()
    out["bio_median_tokens"] = E.groupby("ghost").n_tokens.median().rename(str).to_dict()

    # 3. distinctive words
    lo = log_odds([b for b, g in zip(bios, fb.ghost) if g], [b for b, g in zip(bios, fb.ghost) if not g])
    lo.to_csv(PROCESSED / "bio_log_odds.csv", index=False)
    out["bio_words_ghost"] = lo[lo.z > 0].word.tolist()
    out["bio_words_real"] = lo[lo.z < 0].word.tolist()

    # 4. name form
    pl = df.name.str.fullmatch(PERSON_NAME).fillna(False)
    out["person_like_name"] = {"ghost": float(pl[df.ghost].mean()),
                               "real_functional": float(pl[(~df.ghost) & (df.group == "functional")].mean()),
                               "genre": float(pl[df.group == "genre"].mean())}

    # 5. title reuse
    t["title"] = t.name.str.replace(r"\s*[\(\[-].*$", "", regex=True).str.lower().str.strip()
    t["ghost"] = t.pa.map(df.ghost)
    reuse = t.title.map(t.groupby("title").pa.nunique()) > 1
    out["title_reused_by_other_name"] = reuse.groupby(t.ghost).mean().rename(str).to_dict()
    out["title_words_mean"] = t.title.str.split().str.len().groupby(t.ghost).mean().rename(str).to_dict()

    (PROCESSED / "text.json").write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
