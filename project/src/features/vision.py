"""Phase 5 (exploratory): cover-art features.

- pHash (64-bit perceptual hash) per album cover, to find exact/near reuse across names.
- CLIP image embeddings (open_clip ViT-B-32, laion2b_s34b_b79k), L2-normalised.
- CLIP text embeddings of each track title, for the title-cover "literalness" score.
- Mean RGB/HSV per cover (used for the front-page mosaic ordering).

Outputs in data/processed/: covers_meta.parquet, cover_clip.npy (+ cover_ids.txt),
title_clip.npy (+ title_track_ids.txt).

    python -m src.features.vision
"""
from __future__ import annotations

import colorsys

import numpy as np
import pandas as pd
from PIL import Image

from src.load import PROCESSED, RAW, load

MODEL, PRETRAINED = "ViT-B-32", "laion2b_s34b_b79k"


def hashes_and_colors(album_ids: list[str]) -> pd.DataFrame:
    import imagehash

    rows = []
    for aid in album_ids:
        p = RAW / "covers" / f"{aid}.jpg"
        try:
            im = Image.open(p).convert("RGB")
        except (FileNotFoundError, OSError):
            rows.append({"album_id": aid})
            continue
        small = np.asarray(im.resize((32, 32)), dtype=float) / 255
        r, g, b = small.reshape(-1, 3).mean(0)
        h, s, v = colorsys.rgb_to_hsv(r, g, b)
        # dominant hue weighted by saturation, more faithful than the hue of the mean colour
        hsv = np.array([colorsys.rgb_to_hsv(*px) for px in small.reshape(-1, 3)])
        w = hsv[:, 1] * hsv[:, 2] + 1e-6
        ang = 2 * np.pi * hsv[:, 0]
        dom_h = (np.arctan2((w * np.sin(ang)).sum(), (w * np.cos(ang)).sum()) / (2 * np.pi)) % 1
        rows.append({"album_id": aid, "phash": str(imagehash.phash(im)), "r": r, "g": g, "b": b,
                     "hue": h, "sat": s, "val": v, "dom_hue": dom_h, "sat_mean": hsv[:, 1].mean()})
    return pd.DataFrame(rows)


def clip_embeddings(album_ids: list[str], titles: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    import open_clip
    import torch

    torch.manual_seed(0)
    dev = "cpu"  # MPS crashed with an NSException on this machine
    torch.set_num_threads(8)
    model, _, pre = open_clip.create_model_and_transforms(MODEL, pretrained=PRETRAINED, device=dev)
    tok = open_clip.get_tokenizer(MODEL)
    model.eval()
    img_out = []
    with torch.no_grad():
        for i in range(0, len(album_ids), 128):
            batch = torch.stack([pre(Image.open(RAW / "covers" / f"{a}.jpg").convert("RGB"))
                                 for a in album_ids[i:i + 128]]).to(dev)
            e = model.encode_image(batch)
            img_out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
            print(f"  images {i + len(batch)}/{len(album_ids)}", flush=True)
        txt_out = []
        tl = titles.tolist()
        for i in range(0, len(tl), 512):
            e = model.encode_text(tok([f"album cover for the song {x}" for x in tl[i:i + 512]]).to(dev))
            txt_out.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
    return np.concatenate(img_out).astype(np.float32), np.concatenate(txt_out).astype(np.float32)


def main() -> None:
    al = load("albums")
    ids = [a for a in al.album_id if (RAW / "covers" / f"{a}.jpg").exists()]
    print(f"{len(ids)} covers on disk")
    if not (PROCESSED / "covers_meta.parquet").exists():
        hashes_and_colors(ids).to_parquet(PROCESSED / "covers_meta.parquet")
        print("  wrote covers_meta.parquet")
    t = load("tracks")
    img, txt = clip_embeddings(ids, t.name.fillna(""))
    np.save(PROCESSED / "cover_clip.npy", img)
    (PROCESSED / "cover_ids.txt").write_text("\n".join(ids))
    np.save(PROCESSED / "title_clip.npy", txt)
    (PROCESSED / "title_track_ids.txt").write_text("\n".join(t.track_id))
    print("  wrote CLIP embeddings")


if __name__ == "__main__":
    main()
