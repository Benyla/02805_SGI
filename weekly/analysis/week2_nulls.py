#!/usr/bin/env python3
"""Week 2, part 2 -- the null models.

Covers exercise 2.8 (compared to what -- properly), 2.9 (the friendship paradox
as a null-model exercise), and the extra work for 2.11: a shuffle battery over
ten different network statistics, and a grown-from-scratch Marvel.

Figures land in docs/assets/week2/, numbers in weekly/analysis/week2_results_nulls.json,
and the data for the interactive shuffle widget in docs/assets/week2/marvel_gc.json.
"""

from __future__ import annotations

import json
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from week2_common import (
    ACCENT, FG, GREEN, MUTED, ORANGE, PINK, PURPLE, ROOT, SEED,
    apply_dark_style, ccdf, giant, load_marvel, savefig, short_name, undirected,
)

RESULTS: dict = {}
N_NULL = 100          # replicates per null model
SWAP_FACTOR = 10      # double_edge_swap count, as a multiple of m


# --------------------------------------------------------------------------
# the two roads to a degree-preserving null
# --------------------------------------------------------------------------

def config_model(seq: list[int], seed: int) -> tuple[nx.Graph, int]:
    """Configuration model, simplified. Returns (graph, links destroyed)."""
    mg = nx.configuration_model(seq, seed=seed)
    before = mg.number_of_edges()
    g = nx.Graph(mg)                                   # collapses parallel links
    g.remove_edges_from(nx.selfloop_edges(g))          # drops self-loops
    return g, before - g.number_of_edges()


def swapped(g: nx.Graph, seed: int, factor: int = SWAP_FACTOR) -> nx.Graph:
    """Degree-preserving double edge swaps, starting from the real network."""
    h = g.copy()
    nswap = factor * h.number_of_edges()
    nx.double_edge_swap(h, nswap=nswap, max_tries=nswap * 40, seed=seed)
    return h


# --------------------------------------------------------------------------
# 2.8.4 + 2.8.5 -- clustering, two nulls, two roads
# --------------------------------------------------------------------------

def ex_2_8_45(gc: nx.Graph):
    print("\n[2.8.4/2.8.5] clustering under three nulls, on the giant component")
    # configuration_model relabels to 0..n-1 in the order of the sequence we hand it
    node_order = [v for v, _ in gc.degree()]
    seq = [gc.degree(v) for v in node_order]
    n, m = gc.number_of_nodes(), gc.number_of_edges()
    real_c = nx.average_clustering(gc)

    rng = np.random.default_rng(SEED)
    cfg_c, cfg_lost, gnm_c, swp_c = [], [], [], []
    deficit_by_degree: dict[int, list[int]] = {}

    for _ in range(N_NULL):
        s = int(rng.integers(1e9))
        h, lost = config_model(seq, s)
        cfg_c.append(nx.average_clustering(h))
        cfg_lost.append(lost)
        # whose links went missing? group the shortfall by original degree
        for i, d0 in enumerate(seq):
            deficit_by_degree.setdefault(d0, []).append(d0 - h.degree(i))
        gnm_c.append(nx.average_clustering(nx.gnm_random_graph(n, m, seed=s)))
        swp_c.append(nx.average_clustering(swapped(gc, s)))

    cfg_c, gnm_c, swp_c = map(np.array, (cfg_c, gnm_c, swp_c))

    def summary(vals: np.ndarray) -> dict:
        z = (real_c - vals.mean()) / vals.std(ddof=1)
        p = (1 + int((vals >= real_c).sum())) / (len(vals) + 1)
        return {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                "z": float(z), "p_one_sided": float(p), "max": float(vals.max())}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.hist(gnm_c, bins=18, color=GREEN, alpha=0.75, label="G(n,m) — density only")
    ax.hist(cfg_c, bins=18, color=ORANGE, alpha=0.75, label="configuration model")
    ax.hist(swp_c, bins=18, color=ACCENT, alpha=0.75, label="degree-preserving swaps")
    ax.axvline(real_c, color=PINK, lw=2.5, label=f"real Marvel: {real_c:.3f}")
    ax.set_xlabel("average clustering $C$")
    ax.set_ylabel(f"count out of {N_NULL}")
    ax.set_title("Three nulls, and the real value is off all three charts",
                 color=FG, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", lw=0.4, alpha=0.4)

    ax = axes[1]
    ax.hist(cfg_c, bins=18, color=ORANGE, alpha=0.8, label="configuration model")
    ax.hist(swp_c, bins=18, color=ACCENT, alpha=0.8, label="double edge swaps")
    ax.set_xlabel("average clustering $C$")
    ax.set_ylabel(f"count out of {N_NULL}")
    ax.set_title("Zoom: two roads that are supposed to lead\nto the same null, and do not",
                 color=FG, fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, ls="--", lw=0.4, alpha=0.4)
    savefig(fig, "clustering_nulls.png")

    # Where did the gap come from? The configuration model drops links, and it
    # drops them where the stubs pile up: on the hubs.
    degs = sorted(deficit_by_degree)
    mean_def = np.array([np.mean(deficit_by_degree[d]) for d in degs])
    degs_arr = np.array(degs, dtype=float)
    frac_lost = mean_def / degs_arr

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(degs_arr, frac_lost, "o", color=ORANGE, ms=6, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("original degree of the hero (log scale)")
    ax.set_ylabel("share of that hero's links lost\nto self-loops and doubles")
    ax.set_title("The configuration model taxes the hubs.\n"
                 "That is the whole 0.11-vs-0.15 story.", color=FG, fontsize=11)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)
    savefig(fig, "config_model_tax.png")

    top_def = sorted(deficit_by_degree, reverse=True)[:5]
    RESULTS["ex_2_8_45"] = {
        "real_clustering": float(real_c),
        "gnm": summary(gnm_c), "configuration": summary(cfg_c), "swap": summary(swp_c),
        "links_lost_mean": float(np.mean(cfg_lost)),
        "links_lost_pct": float(100 * np.mean(cfg_lost) / m),
        "m": m, "n": n,
        "hub_loss_pct": {int(d): float(100 * np.mean(deficit_by_degree[d]) / d)
                         for d in top_def},
        "low_degree_loss_pct": {int(d): float(100 * np.mean(deficit_by_degree[d]) / d)
                                for d in degs[:5]},
    }
    print(f"  real C={real_c:.4f}")
    for lab, v in [("G(n,m)", gnm_c), ("config", cfg_c), ("swap", swp_c)]:
        s = summary(v)
        print(f"    {lab:>8}: {s['mean']:.4f} +/- {s['sd']:.4f}  z={s['z']:+.1f}  "
              f"p={s['p_one_sided']:.3f}")
    print(f"  configuration model loses {np.mean(cfg_lost):.1f} links "
          f"({100 * np.mean(cfg_lost) / m:.1f}% of {m})")


# --------------------------------------------------------------------------
# 2.8.6 + the 2.11 shuffle battery -- which numbers survive a shuffle?
# --------------------------------------------------------------------------

def stat_pack(g: nx.Graph) -> dict:
    gc = giant(g)
    deg = np.array([d for _, d in g.degree()])
    return {
        "Average clustering C": nx.average_clustering(g),
        "Transitivity": nx.transitivity(g),
        "Triangles": sum(nx.triangles(g).values()) / 3,
        "Degree assortativity": nx.degree_assortativity_coefficient(g),
        "Mean distance (GC)": nx.average_shortest_path_length(gc),
        "Diameter (GC)": nx.diameter(gc),
        "Islands (components)": nx.number_connected_components(g),
        "Giant component size": gc.number_of_nodes(),
        "Isolated heroes": int((deg == 0).sum()),
        "Top hub's share of links": deg.max() / g.number_of_edges(),
    }


INVARIANT = {"Isolated heroes", "Top hub's share of links"}


def ex_shuffle_battery(u: nx.Graph):
    print("\n[2.8.6 + 2.11] the shuffle battery")
    real = stat_pack(u)
    rng = np.random.default_rng(SEED)

    swap_rows, gnm_rows = [], []
    n, m = u.number_of_nodes(), u.number_of_edges()
    for i in range(N_NULL):
        s = int(rng.integers(1e9))
        swap_rows.append(stat_pack(swapped(u, s)))
        gnm_rows.append(stat_pack(nx.gnm_random_graph(n, m, seed=s)))
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{N_NULL}")

    def collect(rows, key):
        return np.array([r[key] for r in rows], dtype=float)

    table = {}
    for key in real:
        row = {"real": float(real[key]), "invariant_by_construction": key in INVARIANT}
        for lab, rows in [("swap", swap_rows), ("gnm", gnm_rows)]:
            v = collect(rows, key)
            sd = v.std(ddof=1)
            z = (real[key] - v.mean()) / sd if sd > 1e-12 else 0.0
            p_hi = (1 + int((v >= real[key]).sum())) / (len(v) + 1)
            p_lo = (1 + int((v <= real[key]).sum())) / (len(v) + 1)
            row[lab] = {"mean": float(v.mean()), "sd": float(sd), "z": float(z),
                        "p": float(min(p_hi, p_lo) * 2)}
        table[key] = row

    order = sorted(real, key=lambda k: -abs(table[k]["swap"]["z"]))
    ys = np.arange(len(order))
    zs_swap = [table[k]["swap"]["z"] for k in order]
    zs_gnm = [table[k]["gnm"]["z"] for k in order]

    fig, ax = plt.subplots(figsize=(11, 6.8))
    ax.barh(ys + 0.2, zs_gnm, height=0.38, color=GREEN, alpha=0.7,
            label="vs G(n,m) — only n and m preserved")
    ax.barh(ys - 0.2, zs_swap, height=0.38, color=ACCENT,
            label="vs degree-preserving shuffle")
    # symlog keeps the |z|<2 band readable while z=150 still fits on the page
    ax.set_xscale("symlog", linthresh=2)
    ax.set_xlim(-400, 400)
    ax.axvspan(-2, 2, color=MUTED, alpha=0.18)
    ax.axvline(0, color=MUTED, lw=1)
    for i, k in enumerate(order):
        if k in INVARIANT:
            ax.text(2.6, i - 0.2, "frozen: the shuffle cannot touch this",
                    color=MUTED, fontsize=8, va="center", style="italic")
        else:
            z = table[k]["swap"]["z"]
            ax.text(z * 1.35 if z > 0 else z * 1.35, i - 0.2, f"{z:+.1f}",
                    color=FG, fontsize=8, va="center",
                    ha="left" if z > 0 else "right")
    ax.set_yticks(ys)
    ax.set_yticklabels(order)
    ax.invert_yaxis()
    ax.set_xlabel("z-score of the real Marvel value against the null   "
                  "(symlog axis; grey band is $|z|<2$)")
    ax.set_title(f"Ten numbers, two nulls, {N_NULL} shuffles each.\n"
                 "Green is the easy test. Blue is the honest one: what is left "
                 "once the degrees are given away for free?",
                 color=FG, fontsize=11)
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, axis="x", ls="--", lw=0.4, alpha=0.35)
    savefig(fig, "shuffle_battery.png")

    RESULTS["shuffle_battery"] = table
    for k in order:
        t = table[k]
        print(f"  {k:>26}: real {t['real']:>9.3f}  swap z={t['swap']['z']:+8.2f}  "
              f"gnm z={t['gnm']['z']:+9.2f}")
    return table


def ex_reciprocity(g_dir: nx.DiGraph):
    """2.8.2 -- the reciprocity claim, given a null that actually deserves the name."""
    print("\n[2.8.2] reciprocity against a degree-preserving directed null")
    real = nx.reciprocity(g_dir)
    n, m = g_dir.number_of_nodes(), g_dir.number_of_edges()
    rng = np.random.default_rng(SEED)

    swap_r, rnd_r = [], []
    for i in range(N_NULL):
        s = int(rng.integers(1e9))
        h = g_dir.copy()
        try:
            nx.directed_edge_swap(h, nswap=3 * m, max_tries=200 * m, seed=s)
        except nx.NetworkXAlgorithmError:
            pass
        swap_r.append(nx.reciprocity(h))
        rnd_r.append(nx.reciprocity(nx.gnm_random_graph(n, m, seed=s, directed=True)))
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{N_NULL}")
    swap_r, rnd_r = np.array(swap_r), np.array(rnd_r)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(rnd_r, bins=16, color=GREEN, alpha=0.75,
            label=f"random directed G(n,m): {rnd_r.mean():.3f}")
    ax.hist(swap_r, bins=16, color=ACCENT, alpha=0.8,
            label=f"in/out-degree-preserving swaps: {swap_r.mean():.3f}")
    ax.axvline(real, color=PINK, lw=2.5, label=f"real Marvel: {real:.3f}")
    ax.set_xlabel("reciprocity — share of links that are answered")
    ax.set_ylabel(f"count out of {N_NULL}")
    ax.set_title("Reciprocity: the hubs explain a chunk of it, but nowhere near all",
                 color=FG, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, ls="--", lw=0.4, alpha=0.4)
    savefig(fig, "reciprocity_null.png")

    RESULTS["reciprocity"] = {
        "real": float(real),
        "gnm_mean": float(rnd_r.mean()), "gnm_sd": float(rnd_r.std(ddof=1)),
        "swap_mean": float(swap_r.mean()), "swap_sd": float(swap_r.std(ddof=1)),
        "ratio_vs_gnm": float(real / rnd_r.mean()),
        "ratio_vs_swap": float(real / swap_r.mean()),
        "z_vs_swap": float((real - swap_r.mean()) / swap_r.std(ddof=1)),
    }
    print(f"  real {real:.3f}; gnm {rnd_r.mean():.4f} ({real / rnd_r.mean():.1f}x); "
          f"swap {swap_r.mean():.4f} ({real / swap_r.mean():.1f}x)")


# --------------------------------------------------------------------------
# 2.9 -- the friendship paradox
# --------------------------------------------------------------------------

def paradox(g: nx.Graph, n_samples: int = 1000, seed: int = SEED) -> dict:
    """The survey the paradox describes: pick a person, ask about a random friend.

    Reported three ways, because they are three different numbers and the
    difference is the exercise:
      * the sampled survey (what item 2 asks for),
      * its exact expectation, averaged over every (person, friend) pair,
      * <k> + var/<k>, which is the mean degree at a random *edge endpoint* --
        a hub-weighted sample, and therefore always the largest of the three.
    """
    rng = random.Random(seed)
    people = [v for v in g if g.degree(v) > 0]
    deg = dict(g.degree())

    mine, theirs, beaten = [], [], 0
    for _ in range(n_samples):
        v = rng.choice(people)
        f = rng.choice(list(g[v]))
        mine.append(deg[v]); theirs.append(deg[f])
        beaten += deg[f] >= deg[v]

    # exact versions -- no sampling error at all
    per_person_friend_mean = [np.mean([deg[w] for w in g[v]]) for v in people]
    per_person_beaten = [sum(deg[w] >= deg[v] for w in g[v]) / deg[v] for v in people]
    kk = np.array([deg[v] for v in people], dtype=float)

    return {
        "sampled_mean_own": float(np.mean(mine)),
        "sampled_mean_friend": float(np.mean(theirs)),
        "sampled_fraction_beaten": beaten / n_samples,
        "mean_own": float(kk.mean()),
        "mean_friend": float(np.mean(per_person_friend_mean)),
        "fraction_friend_at_least_as_popular": float(np.mean(per_person_beaten)),
        "variance": float(kk.var()),
        "edge_endpoint_formula": float(kk.mean() + kk.var() / kk.mean()),
        "n_with_friends": len(people),
        "n_samples": n_samples,
    }


def ex_2_9(u: nx.Graph, names: dict[str, str]):
    print("\n[2.9] the friendship paradox")
    n, m = u.number_of_nodes(), u.number_of_edges()
    ba = nx.barabasi_albert_graph(5000, 1, seed=SEED)
    rows = {
        "Marvel (real)": paradox(u),
        "Marvel, degree-preserving shuffle": paradox(swapped(u, SEED)),
        "Marvel-sized random G(n,m)": paradox(nx.gnm_random_graph(n, m, seed=SEED)),
        "BA, n=5000": paradox(ba),
        "BA-sized random G(n,m)": paradox(
            nx.gnm_random_graph(5000, ba.number_of_edges(), seed=SEED)),
    }
    for k, v in rows.items():
        print(f"  {k:>34}: you {v['mean_own']:.2f}  friend {v['mean_friend']:.2f} "
              f"(edge-endpoint {v['edge_endpoint_formula']:.2f})  "
              f"beaten {100 * v['fraction_friend_at_least_as_popular']:.0f}%")

    # --- who actually drives it? Exact probability that hero w is the answer
    # when a random hero is asked about a random friend:
    #     p(w) = (1/N_active) * sum over neighbours v of w of 1/k_v
    deg = dict(u.degree())
    active = [v for v in u if deg[v] > 0]
    pick = {w: sum(1.0 / deg[v] for v in u[w]) / len(active) for w in u}
    drivers = sorted(deg, key=lambda v: -deg[v])[:10]

    # --- and who is nobody's inferior? Local maxima of degree.
    local_max = [v for v in u if deg[v] > 0 and all(deg[w] <= deg[v] for w in u[v])]
    strict_max = [v for v in u if deg[v] > 0 and all(deg[w] < deg[v] for w in u[v])]
    # how many heroes name you as their most popular friend?
    crown = {v: 0 for v in u}
    for v in u:
        if deg[v] == 0:
            continue
        best = max(u[v], key=lambda w: deg[w])
        crown[best] += 1
    kings = sorted(crown, key=lambda v: -crown[v])[:8]
    worse_than_friends = sum(
        1 for v in u if deg[v] > 0
        and deg[v] < np.mean([deg[w] for w in u[v]]))
    n_active = sum(1 for v in u if deg[v] > 0)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    ax = axes[0]
    labels = list(rows)
    x = np.arange(len(labels))
    own = [rows[k]["mean_own"] for k in labels]
    fr = [rows[k]["mean_friend"] for k in labels]
    ax.bar(x - 0.2, own, 0.4, color=MUTED, label="your own degree")
    ax.bar(x + 0.2, fr, 0.4, color=ACCENT, label="your random friend's degree")
    ax.plot(x + 0.2, [rows[k]["edge_endpoint_formula"] for k in labels], "_",
            color=PINK, ms=18, mew=2.5,
            label="$\\langle k \\rangle + \\sigma^2/\\langle k \\rangle$ (edge-endpoint sampling)")
    for i, k in enumerate(labels):
        ax.text(i + 0.2, fr[i] + 0.8, f"×{fr[i] / own[i]:.1f}", ha="center",
                color=PINK, fontsize=9)
    short_labels = ["Marvel", "Marvel\nshuffled", "Marvel-size\nrandom",
                    "BA\nn=5000", "BA-size\nrandom"]
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=9)
    ax.set_ylim(0, max(fr) * 1.35)
    ax.set_ylabel("mean degree")
    ax.set_title("Your friends have more friends than you.\n"
                 "Everywhere. Including in the shuffle.", color=FG, fontsize=11)
    ax.legend(fontsize=8, loc="upper center", ncol=1)
    ax.grid(True, axis="y", ls="--", lw=0.4, alpha=0.4)

    ax = axes[1]
    top = sorted(deg, key=lambda v: -deg[v])[:12]
    share = [100 * pick[v] for v in top]
    ax.barh(range(len(top)), share, color=ORANGE, alpha=0.9)
    for i, v in enumerate(top):
        ax.text(share[i] + 0.05, i, f"crowns {crown[v]}", va="center",
                color=MUTED, fontsize=8)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([short_name(names[v]) for v in top], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("exact chance of being the 'friend' in a random survey answer (%)")
    ax.set_xlim(0, max(share) * 1.35)
    ax.set_title("The heroes who make you feel unpopular.\n"
                 "'Crowns' = heroes whose most popular friend is this one.",
                 color=FG, fontsize=11)
    ax.grid(True, axis="x", ls="--", lw=0.4, alpha=0.4)
    savefig(fig, "friendship_paradox.png")

    RESULTS["ex_2_9"] = {
        "rows": rows,
        "top_drivers": [{"name": names[v], "degree": deg[v],
                         "pick_probability_pct": 100 * pick[v],
                         "crowns": crown[v]} for v in drivers],
        "kings": [{"name": names[v], "crowns": crown[v], "degree": deg[v]}
                  for v in kings],
        "n_local_maxima": len(local_max),
        "local_maxima": [{"name": names[v], "degree": deg[v]} for v in
                         sorted(local_max, key=lambda v: -deg[v])],
        "n_strict_local_maxima": len(strict_max),
        "n_active": n_active,
        "worse_than_friends": worse_than_friends,
        "worse_than_friends_pct": 100 * worse_than_friends / n_active,
    }
    print(f"  local degree maxima (nobody out-populars them): {len(local_max)} of {n_active}")
    print("   ", [short_name(names[v]) for v in sorted(local_max, key=lambda v: -deg[v])])
    print(f"  heroes less popular than their friends' average: "
          f"{worse_than_friends}/{n_active} = {100 * worse_than_friends / n_active:.0f}%")


# --------------------------------------------------------------------------
# 2.11 opener -- grow your own Marvel
# --------------------------------------------------------------------------

def ex_grow_marvel(u: nx.Graph, names: dict[str, str]):
    print("\n[2.11] grow your own Marvel")
    n, m = u.number_of_nodes(), u.number_of_edges()
    m_ba = round(m / n)                       # each newcomer brings <k>/2 links
    ba = nx.barabasi_albert_graph(n, m_ba, seed=SEED)

    def row(g):
        gc = giant(g)
        d = np.array([x for _, x in g.degree()])
        return {
            "n": g.number_of_nodes(), "m": g.number_of_edges(),
            "k_max": int(d.max()), "k_min": int(d.min()),
            "mean_degree": float(d.mean()),
            "clustering": float(nx.average_clustering(g)),
            "transitivity": float(nx.transitivity(g)),
            "mean_distance": float(nx.average_shortest_path_length(gc)),
            "diameter": int(nx.diameter(gc)),
            "assortativity": float(nx.degree_assortativity_coefficient(g)),
            "isolates": int((d == 0).sum()),
            "components": nx.number_connected_components(g),
            "gc_size": gc.number_of_nodes(),
        }

    real_row, ba_row = row(u), row(ba)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    d_real = np.array([d for _, d in u.degree()])
    d_ba = np.array([d for _, d in ba.degree()])
    for d, lab, col, mk in [(d_real, f"real Marvel ($k_{{max}}$={d_real.max()})", ACCENT, "o"),
                            (d_ba, f"BA, n=303, m={m_ba} ($k_{{max}}$={d_ba.max()})", ORANGE, "s")]:
        k, p = ccdf(d)
        ax.loglog(k, p, mk, color=col, ms=6, alpha=0.85, label=lab)
    ax.set_xlabel("$k$"); ax.set_ylabel("$P(K \\geq k)$")
    ax.set_title("Same n, same m, and the tail is close —\nright up until Spider-Man",
                 color=FG, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)

    # Everything on one dimensionless axis: how far off is the model, in %?
    ax = axes[1]
    keys = [("mean_distance", "mean distance"), ("diameter", "diameter"),
            ("assortativity", "assortativity"), ("transitivity", "transitivity"),
            ("clustering", "clustering C"), ("k_max", "biggest hub $k_{max}$")]
    err = [100 * (ba_row[k] - real_row[k]) / abs(real_row[k]) for k, _ in keys]
    order = np.argsort([abs(e) for e in err])
    ys = np.arange(len(keys))
    ax.barh(ys, [err[i] for i in order],
            color=[GREEN if abs(err[i]) < 20 else ORANGE for i in order], alpha=0.9)
    for j, i in enumerate(order):
        e = err[i]
        ax.text(e + (3 if e >= 0 else -3), j,
                f"{real_row[keys[i][0]]:.6g} → {ba_row[keys[i][0]]:.6g}",
                va="center", ha="left" if e >= 0 else "right", color=FG, fontsize=8)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(ys); ax.set_yticklabels([keys[i][1] for i in order], fontsize=9)
    ax.set_xlim(-125, 60)
    ax.set_xlabel("model minus reality, as a % of the real value")
    ax.set_title(f"BA(303, {m_ba}) vs reality.\n"
                 "Distances: nailed. Triangles and hubs: not close.\n"
                 f"Not shown, because the model cannot produce them at all: "
                 f"{real_row['isolates']} isolates, {real_row['components']} islands.",
                 color=FG, fontsize=10)
    ax.grid(True, axis="x", ls="--", lw=0.4, alpha=0.4)
    savefig(fig, "grow_your_marvel.png")

    RESULTS["grow_marvel"] = {"m_ba": m_ba, "real": real_row, "ba": ba_row}
    for k in real_row:
        print(f"  {k:>15}: real {real_row[k]:>9.3f}   BA {ba_row[k]:>9.3f}")


# --------------------------------------------------------------------------
# data for the interactive shuffle widget on the site
# --------------------------------------------------------------------------

def export_widget_data(u: nx.Graph, names: dict[str, str]):
    gc = giant(u)
    idx = {v: i for i, v in enumerate(sorted(gc.nodes(), key=lambda v: -gc.degree(v)))}
    payload = {
        "names": [short_name(names[v]) for v in sorted(idx, key=idx.get)],
        "edges": [[idx[a], idx[b]] for a, b in gc.edges()],
        "real": {
            "clustering": nx.average_clustering(gc),
            "transitivity": nx.transitivity(gc),
            "triangles": sum(nx.triangles(gc).values()) / 3,
            "assortativity": nx.degree_assortativity_coefficient(gc),
        },
    }
    blob = json.dumps(payload, separators=(",", ":"))
    # A plain <script> beats fetch() here: the page then works from file:// too.
    path = ROOT / "docs" / "assets" / "week2" / "marvel_gc.js"
    path.write_text("window.MARVEL_GC = " + blob + ";\n", encoding="utf-8")
    print(f"\n  wrote {path.relative_to(ROOT)} "
          f"({len(payload['names'])} nodes, {len(payload['edges'])} edges, "
          f"{path.stat().st_size // 1024} kB)")


def main(only: set[str] | None = None):
    apply_dark_style()
    g_dir, names = load_marvel()
    u = undirected(g_dir)
    gc = giant(u)

    sections = {
        "clustering": lambda: ex_2_8_45(gc),
        "battery": lambda: ex_shuffle_battery(u),
        "reciprocity": lambda: ex_reciprocity(g_dir),
        "paradox": lambda: ex_2_9(u, names),
        "grow": lambda: ex_grow_marvel(u, names),
        "widget": lambda: export_widget_data(u, names),
    }
    for key, fn in sections.items():
        if only is None or key in only:
            fn()

    path = ROOT / "weekly" / "analysis" / "week2_results_nulls.json"
    if only is None:
        path.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
        print(f"\nWrote {path.relative_to(ROOT)}")
    else:
        # partial run: merge into whatever is already on disk
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        old.update(RESULTS)
        path.write_text(json.dumps(old, indent=2), encoding="utf-8")
        print(f"\nMerged {sorted(RESULTS)} into {path.relative_to(ROOT)}")


if __name__ == "__main__":
    import sys
    main(set(sys.argv[1:]) or None)
