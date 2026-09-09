#!/usr/bin/env python3
"""Week 2, part 1 -- the models.

Covers exercises 2.2 (random networks as code), 2.4 (Watts-Strogatz),
2.5b (preferential attachment, built and validated) and 2.6 (is it a power law?).

Every figure lands in docs/assets/week2/; every number lands in
analysis/week2_results_models.json so the write-up never hard-codes a value.
"""

from __future__ import annotations

import json
import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy import optimize, stats
from scipy.sparse.csgraph import connected_components as sp_components
from scipy.sparse.csgraph import shortest_path

from week2_common import (
    ACCENT, FG, GREEN, MUTED, ORANGE, PINK, ROOT, SEED,
    apply_dark_style, ccdf, degree_histogram, giant, hill_alpha, load_marvel,
    log_binned, savefig, undirected,
)

RESULTS: dict = {}


# --------------------------------------------------------------------------
# fast distances -- the pure-python BFS in networkx is the bottleneck in 2.4
# --------------------------------------------------------------------------

def mean_distance(g: nx.Graph) -> float:
    """Mean shortest path over the giant component, via scipy sparse BFS."""
    gc = giant(g)
    a = nx.to_scipy_sparse_array(gc, format="csr")
    d = shortest_path(a, method="D", unweighted=True, directed=False)
    n = d.shape[0]
    return float(d.sum() / (n * (n - 1)))


def giant_fraction(g: nx.Graph) -> float:
    a = nx.to_scipy_sparse_array(g, format="csr")
    _, labels = sp_components(a, directed=False)
    return float(np.bincount(labels).max() / g.number_of_nodes())


# --------------------------------------------------------------------------
# 2.2.1 -- G(n,p), Poisson, and the normal limit
# --------------------------------------------------------------------------

def ex_2_2_1():
    print("\n[2.2.1] G(n,p) degree distribution vs Poisson")
    n = 5000
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    out = {}

    for ax, kbar in zip(axes, [10, 100]):
        p = kbar / (n - 1)
        g = nx.gnp_random_graph(n, p, seed=SEED)
        deg = np.array([d for _, d in g.degree()])
        ks = np.arange(deg.min(), deg.max() + 1)
        ax.bar(ks, np.bincount(deg, minlength=ks.max() + 1)[ks] / n,
               width=0.85, color=ACCENT, alpha=0.7, label="G(n,p) sample")
        ax.plot(ks, stats.poisson.pmf(ks, deg.mean()), color=ORANGE, lw=2,
                label=f"Poisson($\\lambda$={deg.mean():.1f})")
        if kbar == 100:
            ax.plot(ks, stats.norm.pdf(ks, deg.mean(), np.sqrt(deg.mean())),
                    color=GREEN, lw=2, ls="--",
                    label="Normal($\\mu$=$\\sigma^2$=100)")
        # what degree does Poisson say the largest of n draws should reach?
        kmax_pred = int(stats.poisson.ppf(1 - 1 / n, kbar))
        ax.axvline(deg.max(), color=PINK, ls=":", lw=1.5,
                   label=f"observed $k_{{max}}$={deg.max()} (Poisson: {kmax_pred})")
        ax.set_xlabel("degree $k$")
        ax.set_ylabel("$P(k)$")
        ax.set_title(f"n=5000, p={p:.5f}, $\\langle k \\rangle$={kbar}", color=FG)
        ax.legend(fontsize=8)
        ax.grid(True, ls="--", lw=0.4, alpha=0.4)
        out[f"kbar_{kbar}"] = {
            "p": p, "mean_degree": float(deg.mean()), "var_degree": float(deg.var()),
            "k_max": int(deg.max()), "k_min": int(deg.min()),
            "k_max_poisson_predicted": kmax_pred,
        }
        print(f"  <k>={kbar}: mean={deg.mean():.2f} var={deg.var():.2f} "
              f"kmax={deg.max()} (Poisson predicts {kmax_pred})")

    fig.suptitle(
        "Random networks are boring on purpose: every degree sits within a hair of the mean",
        color=FG, fontsize=12)
    savefig(fig, "random_poisson.png")
    RESULTS["ex_2_2_1"] = out


# --------------------------------------------------------------------------
# 2.2.2 -- the phase transition at <k> = 1
# --------------------------------------------------------------------------

def ex_2_2_2():
    print("\n[2.2.2] giant-component phase transition")
    n, reps = 1000, 20
    kbars = np.round(np.arange(0.2, 4.01, 0.1), 2)
    means, sds = [], []
    rng = np.random.default_rng(SEED)
    for kbar in kbars:
        fr = [giant_fraction(nx.gnp_random_graph(n, kbar / (n - 1),
                                                 seed=int(rng.integers(1e9))))
              for _ in range(reps)]
        means.append(np.mean(fr))
        sds.append(np.std(fr))
    means, sds = np.array(means), np.array(sds)

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.errorbar(kbars, means, yerr=sds, fmt="o-", color=ACCENT, ecolor=MUTED,
                capsize=3, ms=4, lw=1.6, label=f"mean of {reps} runs $\\pm$ 1 s.d.")
    ax.axvline(1.0, color=PINK, ls="--", lw=1.5, label="$\\langle k \\rangle = 1$")
    ax.axvline(np.log(n), color=GREEN, ls=":", lw=1.5,
               label=f"$\\ln n$ = {np.log(n):.1f} (last isolate goes)")
    ax.set_xlabel("$\\langle k \\rangle$")
    ax.set_ylabel("giant component / n")
    ax.set_title(f"Phase transition, G(n,p) with n={n}", color=FG)
    ax.legend(loc="lower right")
    ax.grid(True, ls="--", lw=0.4, alpha=0.4)
    savefig(fig, "phase_transition.png")

    at1 = float(means[np.argmin(np.abs(kbars - 1.0))])
    idx = int(np.argmax(means > 0.5))
    RESULTS["ex_2_2_2"] = {
        "n": n, "reps": reps,
        "giant_fraction_at_k1": at1,
        "sd_at_k1": float(sds[np.argmin(np.abs(kbars - 1.0))]),
        "kbar_where_half_the_network_joins": float(kbars[idx]),
        "giant_fraction_at_k4": float(means[-1]),
        "ln_n": float(np.log(n)),
    }
    print(f"  S at <k>=1: {at1:.3f};  crosses 0.5 at <k>={kbars[idx]};  "
          f"S at <k>=4: {means[-1]:.3f}")


# --------------------------------------------------------------------------
# 2.2.3 -- what a path looks like in a barely-connected network
# --------------------------------------------------------------------------

def ex_2_2_3():
    print("\n[2.2.3] two steps away, near the transition")
    g = giant(nx.gnp_random_graph(200, 1.5 / 199, seed=7))
    rng = random.Random(SEED)
    src = rng.choice(sorted(g.nodes()))
    lengths = nx.single_source_shortest_path_length(g, src)
    one = {v for v, d in lengths.items() if d == 1}
    two = {v for v, d in lengths.items() if d == 2}

    colors, sizes = [], []
    for v in g.nodes():
        if v == src:
            colors.append("#000000"); sizes.append(240)
        elif v in two:
            colors.append(PINK); sizes.append(120)
        elif v in one:
            colors.append(ORANGE); sizes.append(80)
        else:
            colors.append("#9fd0ff"); sizes.append(45)

    pos = nx.spring_layout(g, seed=3, k=0.5, iterations=120)
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=MUTED, alpha=0.45, width=0.8)
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=colors, node_size=sizes,
                           edgecolors=FG, linewidths=0.6)
    ax.set_title(
        f"G(200, $\\langle k \\rangle$=1.5): the giant component is "
        f"{g.number_of_nodes()} nodes of spindly tree.\n"
        f"black = a random node, orange = 1 step away ({len(one)}), "
        f"pink = exactly 2 steps ({len(two)})",
        color=FG, fontsize=11)
    ax.axis("off")
    savefig(fig, "two_steps_away.png")

    RESULTS["ex_2_2_3"] = {
        "gc_size": g.number_of_nodes(), "gc_edges": g.number_of_edges(),
        "n_one_step": len(one), "n_two_steps": len(two),
        "eccentricity_of_source": nx.eccentricity(g, src),
        "diameter": nx.diameter(g),
        "mean_distance": mean_distance(g),
        "n_triangles": sum(nx.triangles(g).values()) // 3,
    }
    print(f"  GC={g.number_of_nodes()} nodes, 1 step: {len(one)}, 2 steps: {len(two)}, "
          f"diameter {nx.diameter(g)}")


# --------------------------------------------------------------------------
# 2.2.4 / 2.3.5 -- the comparison table
# --------------------------------------------------------------------------

def describe(u: nx.Graph) -> dict:
    gc = giant(u)
    return {
        "n": u.number_of_nodes(), "m": u.number_of_edges(),
        "mean_degree": 2 * u.number_of_edges() / u.number_of_nodes(),
        "gc_size": gc.number_of_nodes(),
        "mean_distance": mean_distance(u),
        "diameter": nx.diameter(gc),
        "isolates": sum(1 for _, d in u.degree() if d == 0),
        "k_max": max(d for _, d in u.degree()),
        "clustering": nx.average_clustering(u),
        "transitivity": nx.transitivity(u),
        "assortativity": nx.degree_assortativity_coefficient(u),
    }


def ex_2_2_4(u: nx.Graph):
    print("\n[2.2.4 + 2.3.5] real vs random, same n and m")
    real = describe(u)
    n, m = u.number_of_nodes(), u.number_of_edges()

    reps = 100
    rng = np.random.default_rng(SEED)
    rows = [describe(nx.gnm_random_graph(n, m, seed=int(rng.integers(1e9))))
            for _ in range(reps)]
    rnd = {k: float(np.mean([r[k] for r in rows])) for k in real}
    rnd_sd = {k: float(np.std([r[k] for r in rows])) for k in real}

    RESULTS["comparison_table"] = {
        "real": {k: float(v) for k, v in real.items()},
        "random_mean": rnd, "random_sd": rnd_sd, "random_reps": reps,
        "expected_isolates_formula": float(n * np.exp(-real["mean_degree"])),
    }
    for k in real:
        print(f"  {k:>15}: real {real[k]:>9.3f}   random {rnd[k]:>9.3f} +/- {rnd_sd[k]:.3f}")


# --------------------------------------------------------------------------
# 2.4 -- Watts-Strogatz
# --------------------------------------------------------------------------

def ex_2_4():
    print("\n[2.4] Watts-Strogatz sweep")
    n, k, reps = 500, 4, 50
    qs = [0.0, 0.0005, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1.0]
    rng = np.random.default_rng(SEED)
    cm, cs, dm, ds = [], [], [], []
    for q in qs:
        cc, dd = [], []
        for _ in range(reps):
            g = nx.watts_strogatz_graph(n, k, q, seed=int(rng.integers(1e9)))
            cc.append(nx.average_clustering(g))
            dd.append(mean_distance(g))
        cm.append(np.mean(cc)); cs.append(np.std(cc))
        dm.append(np.mean(dd)); ds.append(np.std(dd))
        print(f"  q={q:<7} C={cm[-1]:.4f}  <d>={dm[-1]:.3f}")
    cm, cs, dm, ds = map(np.array, (cm, cs, dm, ds))
    c0, d0 = cm[0], dm[0]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    qq = np.array(qs)
    mask = qq > 0
    ax.errorbar(qq[mask], (cm / c0)[mask], yerr=(cs / c0)[mask], fmt="o-",
                color=ACCENT, ecolor=MUTED, capsize=3, ms=5,
                label="$C(q)/C(0)$  — clustering")
    ax.errorbar(qq[mask], (dm / d0)[mask], yerr=(ds / d0)[mask], fmt="s-",
                color=ORANGE, ecolor=MUTED, capsize=3, ms=5,
                label="$\\langle d \\rangle(q)/\\langle d \\rangle(0)$  — distance")
    ax.set_xscale("log")
    ax.set_xlabel("rewiring probability $q$  (log scale)")
    ax.set_ylabel("value relative to the ring lattice, $q=0$")
    ax.set_ylim(0, 1.08)
    ax.set_title(f"Watts–Strogatz, n={n}, $\\langle k \\rangle$={k}, "
                 f"{reps} runs per point", color=FG)
    ax.axvspan(0.003, 0.05, color=GREEN, alpha=0.08)
    ax.text(0.012, 0.62,
            "the small-world window:\ndistance already collapsed,\nclustering barely scratched",
            color=GREEN, fontsize=9, ha="center")
    ax.legend(loc="lower left")
    ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)
    savefig(fig, "watts_strogatz.png")

    ws1 = nx.watts_strogatz_graph(n, k, 1.0, seed=1)
    RESULTS["ex_2_4"] = {
        "n": n, "k": k, "reps": reps, "q": qs,
        "C_mean": cm.tolist(), "C_sd": cs.tolist(),
        "d_mean": dm.tolist(), "d_sd": ds.tolist(),
        "C0": float(c0), "d0": float(d0),
        "q1_min_degree": int(min(d for _, d in ws1.degree())),
        "q1_clustering": float(nx.average_clustering(ws1)),
        "gnm_same_min_degree": int(min(d for _, d in nx.gnm_random_graph(
            n, ws1.number_of_edges(), seed=1).degree())),
    }


# --------------------------------------------------------------------------
# 2.5b -- preferential attachment, built by hand
# --------------------------------------------------------------------------

def grow(n_target: int, seed: int = SEED, preferential: bool = True):
    """Grow a network one node at a time. Returns (graph, log of target degrees).

    The endpoint list holds every edge endpoint, so drawing from it uniformly
    is exactly Pi(k_i) = k_i / sum_j k_j. That is the whole model.
    """
    rng = random.Random(seed)
    g = nx.Graph()
    g.add_edge(0, 1)
    endpoints = [0, 1]        # every edge contributes both of its nodes
    chosen_degrees = []       # degree of the target at the moment it was picked
    for new in range(2, n_target):
        target = rng.choice(endpoints) if preferential else rng.randrange(new)
        chosen_degrees.append(g.degree(target))
        g.add_edge(new, target)
        endpoints.extend([new, target])
    return g, chosen_degrees


def ex_2_5b_draw():
    print("\n[2.5b.1] preferential attachment at n=100")
    g, _ = grow(100, seed=11)
    deg = dict(g.degree())
    pos = nx.spring_layout(g, seed=5, k=0.42, iterations=150)
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=MUTED, alpha=0.4, width=0.7)
    nodes = nx.draw_networkx_nodes(
        g, pos, ax=ax, node_size=[35 + 26 * deg[v] for v in g],
        node_color=[deg[v] for v in g], cmap=plt.cm.plasma,
        edgecolors=FG, linewidths=0.5)
    top = sorted(deg, key=lambda v: -deg[v])[:5]
    nx.draw_networkx_labels(g, pos, {v: f"#{v}" for v in top}, ax=ax,
                            font_size=8, font_color=FG)
    cb = plt.colorbar(nodes, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label("degree", color=FG)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=MUTED)
    ax.set_title("100 nodes of pure preferential attachment. The labels are arrival\n"
                 "numbers — the hubs are the nodes that showed up early.",
                 color=FG, fontsize=11)
    ax.axis("off")
    savefig(fig, "ba_n100.png")
    RESULTS["ex_2_5b_1"] = {
        "n": g.number_of_nodes(),
        "top_hubs": [{"arrival_index": int(v), "degree": int(deg[v])} for v in top],
    }
    print("  biggest hubs (arrival index, degree):", [(v, deg[v]) for v in top])


def ex_2_5b_main():
    print("\n[2.5b.2-5] preferential attachment at n=5000, validated")
    n = 5000
    ba, chosen = grow(n, seed=SEED, preferential=True)
    uni, _ = grow(n, seed=SEED + 1, preferential=False)
    nxba = nx.barabasi_albert_graph(n, 1, seed=SEED)

    d_ba = np.array([d for _, d in ba.degree()])
    d_uni = np.array([d for _, d in uni.degree()])
    d_nx = np.array([d for _, d in nxba.degree()])

    alpha, ntail = hill_alpha(d_ba, kmin=3)
    # BA's exact P(k) = 4/(k(k+1)(k+2)) is only k^-3 asymptotically, so the MLE
    # has to be pushed up the tail before it recognises the exponent it was born
    # with. Worth reporting rather than hiding.
    hill_sweep = {km: hill_alpha(d_ba, kmin=km)[0] for km in (2, 3, 5, 10, 20, 40)}
    print("  Hill gamma vs kmin:", {k: round(v, 2) for k, v in hill_sweep.items()})

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    ax = axes[0]
    c, p = log_binned(d_ba)
    ax.loglog(c, p, "o", color=ACCENT, ms=7, alpha=0.9, label="our 20 lines")
    c, p = log_binned(d_nx)
    ax.loglog(c, p, "s", color=ORANGE, ms=6, alpha=0.8, mfc="none",
              label="nx.barabasi_albert_graph(5000, 1)")
    ks = np.logspace(0, np.log10(d_ba.max()), 50)
    ax.loglog(ks, 2.0 * ks ** -3.0, color=PINK, ls="--", lw=1.5, label="slope $-3$")
    ax.set_xlabel("$k$"); ax.set_ylabel("$P(k)$, log-binned")
    ax.set_title(f"(a) our loops vs networkx\nMLE $\\gamma$={alpha:.2f} on $k\\geq3$",
                 color=FG, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)

    # Validation (b): draws really do land in proportion to degree.
    ax = axes[1]
    cnt = np.bincount(np.array(chosen))
    share_observed = cnt / cnt.sum()
    kk = np.arange(len(cnt))
    pk = np.bincount(d_ba, minlength=len(cnt))[:len(cnt)] / n
    theory = kk * pk / d_ba.mean()
    keep = (share_observed > 0) & (theory > 0)
    solid = keep & (cnt >= 5)          # the noisy end is thin counts, not a bad model
    ax.loglog(kk[keep], share_observed[keep], "o", color=GREEN, ms=6,
              label="observed attachment events")
    ax.loglog(kk[solid], theory[solid], "-", color=PINK, lw=1.6,
              label="$k\\,P(k)/\\langle k \\rangle$ (where $\\geq$5 events)")
    ax.set_xlabel("degree of the node that got chosen")
    ax.set_ylabel("share of attachment events")
    ax.set_title("(b) the draws really are $\\propto k$", color=FG, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)

    ax = axes[2]
    for d, lab, col, mk in [(d_ba, "preferential attachment", ACCENT, "o"),
                            (d_uni, "growth only (uniform target)", ORANGE, "s")]:
        k2, p2 = ccdf(d)
        ax.loglog(k2, p2, mk, color=col, ms=5, alpha=0.85,
                  label=f"{lab}, $k_{{max}}$={d.max()}")
    ax.set_xlabel("$k$"); ax.set_ylabel("$P(K \\geq k)$")
    ax.set_title("(c) growth is not the ingredient.\nPreference is.", color=FG, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)

    savefig(fig, "ba_validation.png")

    # 2.5b.5 -- uniform growth vs G(n,m)
    m_uni = uni.number_of_edges()
    rnd = nx.gnm_random_graph(n, m_uni, seed=SEED)
    d_rnd = np.array([d for _, d in rnd.degree()])
    comp_uni = nx.number_connected_components(uni)
    comp_rnd = nx.number_connected_components(rnd)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    ax = axes[0]
    for d, lab, col in [
        (d_uni, f"uniform growth — {comp_uni} component, min degree {d_uni.min()}", ORANGE),
        (d_rnd, f"G(n,m), same n and m — {comp_rnd} components, {(d_rnd == 0).sum()} isolates", ACCENT),
    ]:
        k2, p2 = degree_histogram(d)
        ax.semilogy(k2, p2, "o-", color=col, ms=4, lw=1, alpha=0.85, label=lab)
    ax.set_xlabel("$k$"); ax.set_ylabel("$P(k)$, log y")
    ax.set_title("Same n, same m, no preference in either —\nand still not the same network",
                 color=FG, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, ls="--", lw=0.4, alpha=0.4)

    ax = axes[1]
    for d, lab, col, mk in [(d_ba, "preferential attachment", ACCENT, "o"),
                            (d_uni, "uniform growth", ORANGE, "s"),
                            (d_rnd, "G(n,m)", GREEN, "^")]:
        k2, p2 = ccdf(d)
        ax.loglog(k2, p2, mk, color=col, ms=5, alpha=0.8,
                  label=f"{lab} ($k_{{max}}$={d.max()})")
    ax.set_xlabel("$k$"); ax.set_ylabel("$P(K \\geq k)$")
    ax.set_title("Three CCDFs, one axis", color=FG, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)
    savefig(fig, "growth_vs_preference.png")

    RESULTS["ex_2_5b"] = {
        "n": n,
        "pa": {"k_max": int(d_ba.max()), "k_min": int(d_ba.min()),
               "mean": float(d_ba.mean()), "gamma_mle_kmin3": float(alpha),
               "n_tail": int(ntail),
               "gamma_mle_by_kmin": {str(k): float(v) for k, v in hill_sweep.items()}},
        "nx_ba": {"k_max": int(d_nx.max()), "k_min": int(d_nx.min())},
        "uniform_growth": {"k_max": int(d_uni.max()), "k_min": int(d_uni.min()),
                           "m": int(m_uni), "components": comp_uni},
        "gnm": {"k_max": int(d_rnd.max()), "k_min": int(d_rnd.min()),
                "components": comp_rnd, "isolates": int((d_rnd == 0).sum())},
    }
    print(f"  PA:   kmax={d_ba.max()} kmin={d_ba.min()} gamma_MLE={alpha:.2f}")
    print(f"  nxBA: kmax={d_nx.max()}")
    print(f"  uni:  kmax={d_uni.max()} components={comp_uni}")
    print(f"  gnm:  kmax={d_rnd.max()} components={comp_rnd} isolates={(d_rnd == 0).sum()}")
    return d_ba


# --------------------------------------------------------------------------
# 2.6 -- is the Marvel in-degree a power law?
# --------------------------------------------------------------------------

def fit_powerlaw(x: np.ndarray) -> dict:
    """Clauset-style: scan k_min, MLE alpha at each, keep the smallest KS."""
    x = np.sort(x[x > 0].astype(float))
    best = None
    for kmin in np.unique(x):
        if kmin < 2:
            continue
        tail = x[x >= kmin]
        if len(tail) < 20:
            break
        a = 1.0 + len(tail) / np.sum(np.log(tail / (kmin - 0.5)))
        emp = np.arange(len(tail), 0, -1) / len(tail)
        theo = (tail / kmin) ** (1 - a)
        ks = float(np.abs(emp - theo).max())
        if best is None or ks < best["ks"]:
            best = {"kmin": float(kmin), "alpha": float(a), "ks": ks,
                    "n_tail": int(len(tail))}
    return best


def lognormal_vs_powerlaw(x: np.ndarray, kmin: float) -> dict:
    """Vuong-style likelihood ratio on the same tail. |stat| < ~1.96 means a tie."""
    tail = np.sort(x[x >= kmin].astype(float))
    n = len(tail)
    a = 1.0 + n / np.sum(np.log(tail / (kmin - 0.5)))
    ll_pl = np.log((a - 1) / kmin) - a * np.log(tail / kmin)

    def neg_ll(params):
        mu, log_s = params
        s = np.exp(log_s)
        surv = stats.norm.sf((np.log(kmin) - mu) / s)
        if surv <= 0:
            return 1e12
        z = (np.log(tail) - mu) / s
        return -np.sum(-np.log(tail * s * np.sqrt(2 * np.pi)) - 0.5 * z ** 2
                       - np.log(surv))

    res = optimize.minimize(neg_ll,
                            [np.log(tail).mean(), np.log(np.log(tail).std() + 1e-9)],
                            method="Nelder-Mead")
    mu, s = res.x[0], np.exp(res.x[1])
    z = (np.log(tail) - mu) / s
    surv = stats.norm.sf((np.log(kmin) - mu) / s)
    ll_ln = (-np.log(tail * s * np.sqrt(2 * np.pi)) - 0.5 * z ** 2 - np.log(surv))

    diff = ll_pl - ll_ln
    R = float(diff.sum())
    sd = float(diff.std(ddof=1))
    stat = R / (np.sqrt(n) * sd) if sd > 0 else 0.0
    return {"R": R, "normalised_R": float(stat),
            "p_value": float(2 * stats.norm.sf(abs(stat))), "n_tail": int(n),
            "lognormal_mu": float(mu), "lognormal_sigma": float(s), "alpha": float(a)}


def ex_2_6(g_dir: nx.DiGraph, d_ba: np.ndarray):
    print("\n[2.6] is the Marvel in-degree a power law?")
    ind = np.array([d for _, d in g_dir.in_degree()])
    n_zero = int((ind == 0).sum())

    fit = fit_powerlaw(ind)
    lr = lognormal_vs_powerlaw(ind, fit["kmin"])
    k, p = ccdf(ind)
    naive = float(np.polyfit(np.log10(k), np.log10(p), 1)[0])

    n_rnd = 5000
    rnd = nx.gnm_random_graph(n_rnd, n_rnd * 5, seed=SEED)
    d_rnd = np.array([d for _, d in rnd.degree()])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    kb, pb = log_binned(ind)
    ax.loglog(kb, pb, "s", color=ORANGE, ms=7, label="binned $P(k)$ (week-1 recipe)")
    ax.loglog(k, p, "o", color=ACCENT, ms=5, alpha=0.8,
              label="CCDF $P(K \\geq k)$ — no binning choices")
    kk = np.logspace(np.log10(fit["kmin"]), np.log10(ind.max()), 40)
    tail_frac = fit["n_tail"] / (ind > 0).sum()
    ax.loglog(kk, tail_frac * (kk / fit["kmin"]) ** (1 - fit["alpha"]), color=PINK, lw=2,
              label=(f"MLE fit: $\\gamma$={fit['alpha']:.2f}, "
                     f"$k_{{min}}$={fit['kmin']:.0f} ({fit['n_tail']} heroes)"))
    ax.set_xlabel("in-degree $k$"); ax.set_ylabel("probability")
    ax.set_title(f"Marvel in-degree: the {(ind > 0).sum()} heroes with at least one incoming link",
                 color=FG, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)

    # The three networks have wildly different densities, so compare shapes:
    # rescale every degree by its own mean. Now only the tail shape is left.
    ax = axes[1]
    kk2, pp2 = ccdf(ind)
    mean_in = ind.mean()
    ax.loglog(kk2 / mean_in, pp2, "o", color=ACCENT, ms=5,
              label=f"Marvel in-degree (n=303, $\\langle k \\rangle$={mean_in:.1f})")
    for d, lab, col in [(d_ba, "BA, n=5000", ORANGE),
                        (d_rnd, "random, n=5000", GREEN)]:
        a2, b2 = ccdf(d)
        ax.loglog(a2 / d.mean(), b2, "-", color=col, lw=2, alpha=0.9,
                  label=f"{lab} ($\\langle k \\rangle$={d.mean():.1f})")
    ax.axvline(fit["kmin"] / mean_in, color=PINK, ls=":", lw=1.2,
               label=f"$k_{{min}}$ = {fit['kmin']:.0f}")
    ax.set_xlabel("$k / \\langle k \\rangle$")
    ax.set_ylabel("$P(K \\geq k)$")
    ax.set_title("Shape only: every degree divided by its own mean",
                 color=FG, fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, which="both", ls="--", lw=0.4, alpha=0.4)
    savefig(fig, "marvel_ccdf.png")

    RESULTS["ex_2_6"] = {
        "n_zero_in_degree": n_zero,
        "naive_full_range_ccdf_slope": naive,
        "naive_gamma": float(1 - naive),
        "fit": fit,
        "lognormal_test": lr,
        "ccdf_k": k.tolist(), "ccdf_p": p.tolist(),
    }
    print(f"  naive full-range CCDF slope {naive:.2f}  ->  gamma ~ {1 - naive:.2f}")
    print(f"  MLE fit: kmin={fit['kmin']:.0f} gamma={fit['alpha']:.2f} "
          f"KS={fit['ks']:.3f} n_tail={fit['n_tail']}")
    print(f"  power law vs log-normal: R={lr['R']:.2f} "
          f"normalised={lr['normalised_R']:.2f} p={lr['p_value']:.2f}")


def main():
    apply_dark_style()
    g_dir, _ = load_marvel()
    u = undirected(g_dir)

    ex_2_2_1()
    ex_2_2_2()
    ex_2_2_3()
    ex_2_2_4(u)
    ex_2_4()
    ex_2_5b_draw()
    d_ba = ex_2_5b_main()
    ex_2_6(g_dir, d_ba)

    path = ROOT / "analysis" / "week2_results_models.json"
    path.write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    print(f"\nWrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
