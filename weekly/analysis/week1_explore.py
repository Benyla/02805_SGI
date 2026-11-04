#!/usr/bin/env python3
"""Week 1 Marvel Wikipedia network exploration — figures for GitHub Pages."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repo root
DATA = ROOT / "weekly" / "data"
OUT = ROOT / "docs" / "assets" / "week1"

# Dark theme matching the site
BG = "#0d1117"
FG = "#e6edf3"
ACCENT = "#58a6ff"
GREEN = "#3fb950"
MUTED = "#8b949e"
GRID = "#21262d"


def apply_dark_style():
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
            "font.family": "sans-serif",
            "font.size": 10,
        }
    )


def load_graph() -> tuple[nx.DiGraph, pd.DataFrame]:
    nodes = pd.read_csv(DATA / "week1_nodes.tsv", sep="\t", comment="#")
    edges = pd.read_csv(
        DATA / "week1_edges.tsv",
        sep="\t",
        comment="#",
        names=["source", "target"],
        header=None,
    )
    g = nx.DiGraph()
    g.add_nodes_from(nodes["node_id"])
    g.add_edges_from(zip(edges["source"], edges["target"]))
    return g, nodes


def name_map(nodes: pd.DataFrame) -> dict[str, str]:
    return dict(zip(nodes["node_id"], nodes["name"]))


def sanity_checks(g: nx.DiGraph, names: dict[str, str]) -> dict:
    n = g.number_of_nodes()
    m_dir = g.number_of_edges()
    gu = g.to_undirected()
    m_undir = gu.number_of_edges()
    avg_k = 2 * m_undir / n

    in_deg = dict(g.in_degree())
    out_deg = dict(g.out_degree())

    top_in = sorted(in_deg.items(), key=lambda x: -x[1])[:5]
    top_out = sorted(out_deg.items(), key=lambda x: -x[1])[:5]

    print("=== Sanity checks ===")
    print(f"Directed: n={n}, m={m_dir}")
    print(f"Undirected collapse: m={m_undir}, ⟨k⟩={avg_k:.2f}")
    print("\nTop 5 in-degree:")
    for node, d in top_in:
        print(f"  {names.get(node, node)}: {d}")
    print("\nTop 5 out-degree:")
    for node, d in top_out:
        print(f"  {names.get(node, node)}: {d}")

    isolates = [node for node in g.nodes() if g.degree(node) == 0]
    print(f"\nIsolates (zero total degree): {len(isolates)}")

    components = sorted(nx.weakly_connected_components(g), key=len, reverse=True)
    sizes = [len(c) for c in components]
    print(f"Weakly connected components: {sizes}")

    return {
        "n": n,
        "m_dir": m_dir,
        "m_undir": m_undir,
        "avg_k": avg_k,
        "in_deg": in_deg,
        "out_deg": out_deg,
        "isolates": isolates,
        "component_sizes": sizes,
        "components": components,
        "top_in": top_in,
        "top_out": top_out,
    }


def degree_histogram(degrees: list[int]) -> tuple[np.ndarray, np.ndarray]:
    counts = Counter(degrees)
    ks = np.array(sorted(counts.keys()))
    ps = np.array([counts[k] for k in ks])
    return ks, ps


def plot_degree_distributions(stats: dict, out_path: Path):
    in_vals = list(stats["in_deg"].values())
    out_vals = list(stats["out_deg"].values())

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle("Marvel Wikipedia — degree distributions", color=FG, fontsize=14, y=0.98)

    for ax, vals, title in [
        (axes[0, 0], in_vals, "In-degree (linear)"),
        (axes[0, 1], out_vals, "Out-degree (linear)"),
        (axes[1, 0], in_vals, "In-degree (log–log)"),
        (axes[1, 1], out_vals, "Out-degree (log–log)"),
    ]:
        k, p = degree_histogram(vals)
        u = k + 1  # course convention: k+1 so isolates survive log axis
        loglog = ax is axes[1, 0] or ax is axes[1, 1]
        if loglog:
            ax.loglog(u, p, "o", color=ACCENT, markersize=5, alpha=0.85)
            ax.set_xlabel("k + 1")
            ax.set_ylabel("P(k)")
        else:
            ax.bar(u, p, width=0.8, color=ACCENT, alpha=0.85, edgecolor=GRID)
            ax.set_xlabel("k + 1")
            ax.set_ylabel("count")
        ax.set_title(title, color=FG, pad=8)
        ax.grid(True, which="both" if loglog else "major", linestyle="--", linewidth=0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_in_vs_out_degree(stats: dict, names: dict[str, str], out_path: Path):
    """Scatter: in-degree vs out-degree — fame vs curation for every hero."""
    in_deg = stats["in_deg"]
    out_deg = stats["out_deg"]
    isolates = set(stats["isolates"])

    xs, ys, cols = [], [], []
    for node in in_deg:
        xs.append(in_deg[node])
        ys.append(out_deg[node])
        cols.append(MUTED if node in isolates else ACCENT)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(xs, ys, c=cols, alpha=0.65, s=36, edgecolors=GRID, linewidths=0.4)

    # Label key outliers
    label_nodes = {n for n, _ in stats["top_in"][:3]} | {n for n, _ in stats["top_out"][:3]}
    for node in label_nodes:
        short = names.get(node, node).split("(")[0].strip()
        ax.annotate(
            short,
            (in_deg[node], out_deg[node]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
            color=FG,
            bbox=dict(boxstyle="round,pad=0.25", facecolor=BG, edgecolor=GRID, alpha=0.9),
        )

    ax.set_xlabel("In-degree (links received — fame)")
    ax.set_ylabel("Out-degree (links written — curation)")
    ax.set_title("Fame vs curation — every hero in the network", color=FG, pad=12)
    ax.grid(True, linestyle="--", alpha=0.4)

    # Legend for isolates
    ax.scatter([], [], c=MUTED, label=f"Isolates ({len(isolates)})")
    ax.scatter([], [], c=ACCENT, label="Connected heroes")
    ax.legend(loc="lower right", facecolor=BG, edgecolor=GRID, labelcolor=FG)

    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def plot_giant_component(g: nx.DiGraph, stats: dict, names: dict[str, str], out_path: Path):
    giant = max(stats["components"], key=len)
    sub = g.subgraph(giant).copy()
    in_deg = dict(sub.in_degree())

    # Size by in-degree; color by log(in+1)
    sizes = [300 + 80 * np.log1p(in_deg[n]) for n in sub.nodes()]
    colors = [in_deg[n] for n in sub.nodes()]

    pos = nx.spring_layout(sub, seed=42, k=0.35, iterations=60)

    fig, ax = plt.subplots(figsize=(12, 12))
    nx.draw_networkx_edges(sub, pos, ax=ax, edge_color=GRID, alpha=0.15, width=0.4, arrows=True, arrowsize=6)
    nodes = nx.draw_networkx_nodes(
        sub,
        pos,
        ax=ax,
        node_size=sizes,
        node_color=colors,
        cmap=plt.cm.plasma,
        alpha=0.92,
        linewidths=0.5,
        edgecolors=FG,
    )

    # Label top 8 in-degree hubs only
    top_hubs = sorted(in_deg.items(), key=lambda x: -x[1])[:8]
    labels = {n: names.get(n, n).split("(")[0].strip() for n, _ in top_hubs}
    nx.draw_networkx_labels(
        sub,
        pos,
        labels=labels,
        ax=ax,
        font_size=7,
        font_color=FG,
        bbox=dict(boxstyle="round,pad=0.2", facecolor=BG, edgecolor=GRID, alpha=0.8),
    )

    cbar = plt.colorbar(nodes, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("In-degree", color=FG)
    cbar.ax.yaxis.set_tick_params(color=MUTED)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color=MUTED)

    ax.set_title(
        f"Giant component ({sub.number_of_nodes()} heroes) — node size ∝ in-degree",
        color=FG,
        fontsize=13,
        pad=12,
    )
    ax.axis("off")
    fig.savefig(out_path, dpi=150, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    apply_dark_style()
    OUT.mkdir(parents=True, exist_ok=True)

    g, nodes_df = load_graph()
    names = name_map(nodes_df)
    stats = sanity_checks(g, names)

    plot_degree_distributions(stats, OUT / "degree_distributions.png")
    plot_in_vs_out_degree(stats, names, OUT / "in_vs_out_degree.png")
    plot_giant_component(g, stats, names, OUT / "network_giant_component.png")

    print("\nDone — figures in docs/assets/week1/")


if __name__ == "__main__":
    main()
