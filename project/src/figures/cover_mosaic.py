"""Front-page art: a 50 x 50 mosaic of album covers from the dataset, ordered by colour.

2,500 covers sampled with a fixed seed. Colourful covers are sorted by dominant hue into 50
columns (each column sorted by brightness); near-grey covers fill the last columns. Output is
gitignored (derived from copyrighted artwork) and only used on the paper's front page.

    python -m src.figures.cover_mosaic
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from src.load import PROJECT, RAW, PROCESSED, SEED

N, TILE = 50, 56


def main() -> None:
    m = pd.read_parquet(PROCESSED / "covers_meta.parquet").dropna(subset=["dom_hue"])
    m = m.sample(N * N, random_state=SEED)
    grey = m.sat_mean < 0.14
    col = m[~grey].copy()
    gry = m[grey].copy()
    # hue order starting at red-orange so the gradient begins warm
    col["h"] = (col.dom_hue - 0.97) % 1.0
    col = col.sort_values("h")
    gry = gry.sort_values("val", ascending=False)
    ordered = pd.concat([col, gry])
    cols = [ordered.iloc[i * N:(i + 1) * N].sort_values("val", ascending=False) for i in range(N)]
    canvas = Image.new("RGB", (N * TILE, N * TILE), "white")
    for i, c in enumerate(cols):
        for j, aid in enumerate(c.album_id):
            im = Image.open(RAW / "covers" / f"{aid}.jpg").convert("RGB").resize((TILE, TILE), Image.LANCZOS)
            canvas.paste(im, (i * TILE, j * TILE))
    out = PROJECT / "paper" / "cover_mosaic.jpg"
    canvas.save(out, quality=88)
    print(f"  wrote {out.relative_to(PROJECT)} ({len(col)} colourful, {len(gry)} grey covers)")


if __name__ == "__main__":
    main()
