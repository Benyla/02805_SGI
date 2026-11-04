"""Layout helpers for network figures.

sunflower_houses: each house is drawn as a disc; its names sit on a Vogel (sunflower) spiral
around the house node, so disc area is proportional to the number of names. Discs are then
packed greedily on an outward spiral, largest first. Deterministic: no force layout, no seed.

component_layouts + pack: per-component Fruchterman-Reingold, then circle packing.
"""
from __future__ import annotations

import random

import igraph as ig
import numpy as np

from src.load import SEED

GOLDEN = np.pi * (3 - np.sqrt(5))


def vogel(n: int, spacing: float = 1.0, r0: float = 1.6) -> np.ndarray:
    """n points on a sunflower spiral, leaving a hole of radius ~r0 for the hub."""
    k = np.arange(n)
    r = spacing * np.sqrt(k + r0 ** 2)
    th = k * GOLDEN
    return np.c_[r * np.cos(th), r * np.sin(th)]


def pack_discs(radii: np.ndarray, pad: float = 0.8, step: float = 0.04,
               extra_top: np.ndarray | None = None) -> np.ndarray:
    """Greedy circle packing on an outward spiral; radii in descending order recommended.
    extra_top[i] > 0 reserves label room below disc i (its footprint is stretched downward)."""
    n = len(radii)
    extra = np.zeros(n) if extra_top is None else np.asarray(extra_top, float)
    centers = np.zeros((n, 2))
    # candidate positions along the spiral, generated once
    ts = np.arange(0, 4000, step)
    cand = np.c_[0.25 * ts * np.cos(ts), 0.25 * ts * np.sin(ts) * 0.8]
    start = 0
    for i in range(1, n):
        C = centers[:i]
        # footprint: a disc, plus a label box below for labelled discs (as a second disc)
        for k in range(start, len(cand)):
            c = cand[k]
            d = np.hypot(C[:, 0] - c[0], C[:, 1] - c[1])
            if np.any(d < radii[:i] + radii[i] + pad):
                continue
            # label boxes: keep labelled discs' label area (below them) free, and own label area
            lab_c = C - np.c_[np.zeros(i), radii[:i] + extra[:i] / 2]
            if np.any((extra[:i] > 0) & (np.hypot(lab_c[:, 0] - c[0], lab_c[:, 1] - c[1])
                                         < radii[i] + extra[:i] / 2 + pad)):
                continue
            if extra[i] > 0:
                own = c - np.array([0, radii[i] + extra[i] / 2])
                if np.any(np.hypot(C[:, 0] - own[0], C[:, 1] - own[1]) < radii[:i] + extra[i] / 2 + pad):
                    continue
            centers[i] = c
            # small discs never fit closer than the previous small disc: start from there
            if radii[i] <= 1.5:
                start = max(0, k - 200)
            break
    return centers


def component_layouts(g: ig.Graph, niter: int = 1500) -> list[tuple[list[int], np.ndarray]]:
    random.seed(SEED)
    ig.set_random_number_generator(random.Random(SEED))
    out = []
    for comp in g.components():
        if len(comp) == 1:
            xy = np.zeros((1, 2))
        else:
            sub = g.induced_subgraph(comp)
            xy = np.array(sub.layout("fr", niter=niter).coords)
            xy -= np.median(xy, 0)
            r = np.percentile(np.sqrt((xy ** 2).sum(1)), 95)
            xy = xy / max(r, 1e-9) * np.sqrt(len(comp))
        out.append((comp, xy))
    return out


def pack(comps: list[tuple[list[int], np.ndarray]], n: int, pad: float = 0.6) -> np.ndarray:
    comps = sorted(comps, key=lambda c: -len(c[0]))
    radii = np.array([max(np.sqrt((xy ** 2).sum(1)).max(), 0.5) for _, xy in comps])
    centers = pack_discs(radii, pad=pad)
    pos = np.zeros((n, 2))
    for (nodes, xy), c in zip(comps, centers):
        pos[nodes] = xy + c
    return pos
