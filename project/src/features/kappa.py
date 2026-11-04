"""Agreement between hand labels and the ghost-like classifier.

Reads notes/hand_label_sheet.csv (columns label_<name>, values 0/1, blank = not labelled yet).
Prints pairwise Cohen's kappa between labellers and between the majority label and the model.

    python -m src.features.kappa
"""
from __future__ import annotations

from itertools import combinations

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from src.load import PROCESSED, PROJECT


def main() -> None:
    sheet = pd.read_csv(PROJECT / "notes" / "hand_label_sheet.csv", dtype={"artist_id": str})
    cols = [c for c in sheet.columns if c.startswith("label_")]
    lab = sheet[cols].apply(pd.to_numeric, errors="coerce")
    done = [c for c in cols if lab[c].notna().sum() > 0]
    if not done:
        print("No labels yet. Fill in the label_<name> columns with 0 (real) / 1 (ghost).")
        return
    for a, b in combinations(done, 2):
        m = lab[a].notna() & lab[b].notna()
        print(f"{a} vs {b}: kappa = {cohen_kappa_score(lab.loc[m, a], lab.loc[m, b]):.2f} (n={m.sum()})")
    majority = (lab[done].mean(axis=1) >= 0.5).astype(int).where(lab[done].notna().any(axis=1))
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    model = sheet.artist_id.map(df.ghost.astype(int))
    m = majority.notna()
    print(f"majority vs model: kappa = {cohen_kappa_score(majority[m], model[m]):.2f} (n={m.sum()}), "
          f"accuracy = {(majority[m] == model[m]).mean():.0%}")


if __name__ == "__main__":
    main()
