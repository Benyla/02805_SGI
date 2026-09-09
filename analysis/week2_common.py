#!/usr/bin/env python3
"""Shared helpers for the week-2 model / null-model scripts."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DATA"
OUT = ROOT / "docs" / "assets" / "week2"

# Dark theme matching the site
BG = "#0d1117"
FG = "#e6edf3"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
ORANGE = "#f0883e"
PINK = "#f778ba"
PURPLE = "#a371f7"
MUTED = "#8b949e"
GRID = "#21262d"

SEED = 2805


def apply_dark_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": FG,
            "text.color": FG,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.alpha": 0.5,
            "legend.facecolor": "#161b22",
            "legend.edgecolor": GRID,
            "legend.labelcolor": FG,
            "font.family": "sans-serif",
            "font.size": 10,
            "savefig.facecolor": BG,
        }
    )


def load_marvel() -> tuple[nx.DiGraph, dict[str, str]]:
    """Directed Marvel graph with all 303 nodes, plus node_id -> display name."""
    nodes = pd.read_csv(DATA / "week1_nodes.tsv", sep="\t", comment="#")
    edges = pd.read_csv(
        DATA / "week1_edges.tsv", sep="\t", comment="#",
        names=["source", "target"], header=None,
    )
    g = nx.DiGraph()
    g.add_nodes_from(nodes["node_id"])
    g.add_edges_from(zip(edges["source"], edges["target"]))
    names = dict(zip(nodes["node_id"], nodes["name"]))
    return g, names


def undirected(g: nx.DiGraph) -> nx.Graph:
    """Collapse to simple undirected (A->B or B->A becomes one link)."""
    return nx.Graph(g)


def giant(g: nx.Graph) -> nx.Graph:
    comp = max(nx.connected_components(g), key=len)
    return g.subgraph(comp).copy()


def short_name(name: str) -> str:
    return name.split("(")[0].strip()


def ccdf(degrees) -> tuple[np.ndarray, np.ndarray]:
    """Complementary CDF: P(K >= k) at every observed k. No binning, no choices."""
    d = np.sort(np.asarray([x for x in degrees if x > 0]))
    ks = np.unique(d)
    n = len(d)
    p = np.array([(d >= k).sum() / n for k in ks])
    return ks, p


def degree_histogram(degrees) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(degrees)
    ks = np.array(sorted(counts.keys()))
    ps = np.array([counts[k] for k in ks], dtype=float)
    return ks, ps / ps.sum()


def log_binned(degrees, n_bins: int = 14) -> tuple[np.ndarray, np.ndarray]:
    """Logarithmically binned P(k) -- the week-1 recipe, tail noise flattened."""
    d = np.asarray([x for x in degrees if x > 0], dtype=float)
    bins = np.logspace(np.log10(d.min()), np.log10(d.max() + 1), n_bins)
    counts, edges = np.histogram(d, bins=bins)
    widths = np.diff(edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    dens = counts / widths / len(d)
    keep = counts > 0
    return centers[keep], dens[keep]


def hill_alpha(degrees, kmin: int) -> tuple[float, int]:
    """Maximum-likelihood power-law exponent for a discrete tail (Clauset eq. 3.7)."""
    d = np.asarray([x for x in degrees if x >= kmin], dtype=float)
    if len(d) < 2:
        return float("nan"), len(d)
    alpha = 1.0 + len(d) / np.sum(np.log(d / (kmin - 0.5)))
    return alpha, len(d)


def savefig(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")
    return path
