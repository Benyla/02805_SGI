#!/usr/bin/env python3
"""Week 1 Marvel Wikipedia network exploration — figures for GitHub Pages."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "DATA"
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


def plot_hub_table(stats: dict, names: dict[str, str], out_path: Path):
    top_n = 10
    top_in = stats["top_in"][:top_n] if len(stats["top_in"]) >= top_n else stats["top_in"]
    # recompute full top 10 out
    top_out = sorted(stats["out_deg"].items(), key=lambda x: -x[1])[:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")

    col1 = ["Rank", "In-degree leader", "Links in"] + [
        f"{i + 1}" for i in range(top_n)
    ]
    col2 = [""] * 3 + [names.get(n, n) for n, _ in top_in]
    col3 = [""] * 3 + [str(d) for _, d in top_in]
    col4 = ["", "Out-degree leader", "Links out"] + [
        f"{i + 1}" for i in range(top_n)
    ]
    col5 = [""] * 3 + [names.get(n, n) for n, _ in top_out]
    col6 = [""] * 3 + [str(d) for _, d in top_out]

    rows = list(zip(col1, col2, col3, col4, col5, col6))
    table = ax.table(
        cellText=rows,
        loc="center",
        cellLoc="left",
        colWidths=[0.06, 0.22, 0.08, 0.06, 0.22, 0.08],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)

    for (row, col), cell in table.get_celld().items():
        cell.set_facecolor(BG)
        cell.set_edgecolor(GRID)
        cell.set_text_props(color=FG)
        if row < 3:
            cell.set_text_props(weight="bold", color=ACCENT)

    ax.set_title(
        "Fame vs curation — different heroes lead in- and out-degree",
        color=FG,
        fontsize=12,
        pad=20,
    )
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


def plot_islands_and_isolates(
    g: nx.DiGraph, stats: dict, names: dict[str, str], out_path: Path
):
    isolates = stats["isolates"]
    components = stats["components"]
    non_giant = [c for c in components if len(c) > 1]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Left: component size distribution
    sizes = stats["component_sizes"]
    ax0 = axes[0]
    labels = []
    for i, s in enumerate(sizes):
        if i == 0:
            labels.append(f"Giant\n({s})")
        elif s == 1:
            labels.append("Isolate")
        else:
            labels.append(f"Island\n({s})")
    colors_bar = [ACCENT if i == 0 else GREEN if s == 1 else MUTED for i, s in enumerate(sizes)]
    ax0.bar(range(len(sizes)), sizes, color=colors_bar, edgecolor=GRID)
    ax0.set_xticks(range(len(sizes)))
    ax0.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    ax0.set_ylabel("Component size")
    ax0.set_title("Connected components", color=FG)
    ax0.grid(axis="y", linestyle="--", alpha=0.5)

    # Right: isolate names (scrollable text block as bar of count + annotation)
    ax1 = axes[1]
    ax1.axis("off")
    isolate_lines = [f"• {names.get(n, n)}" for n in sorted(isolates, key=lambda x: names.get(x, x))]
    island_lines = []
    for comp in non_giant:
        members = ", ".join(names.get(n, n) for n in sorted(comp, key=lambda x: names.get(x, x)))
        island_lines.append(f"Island ({len(comp)}): {members}")

    text = f"17 isolates — zero links in either direction:\n\n"
    text += "\n".join(isolate_lines[:10])
    if len(isolate_lines) > 10:
        text += f"\n… and {len(isolate_lines) - 10} more"
    if island_lines:
        text += "\n\nSmall islands outside the giant component:\n\n"
        text += "\n".join(island_lines)
    else:
        text += "\n\nNo multi-node islands — only the giant component plus isolates."

    ax1.text(
        0.02,
        0.98,
        text,
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        color=FG,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor=GRID, alpha=0.5, edgecolor=GRID),
    )
    ax1.set_title("Isolates & islands", color=FG)

    fig.suptitle("Outside the giant component", color=FG, fontsize=13, y=1.02)
    plt.tight_layout()
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
    plot_hub_table(stats, names, OUT / "hub_table.png")
    plot_giant_component(g, stats, names, OUT / "network_giant_component.png")
    plot_islands_and_isolates(g, stats, names, OUT / "islands_and_isolates.png")

    print("\nDone — figures in docs/assets/week1/")


if __name__ == "__main__":
    main()
