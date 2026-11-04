"""Phase 3.3: recommendation audit on the directed "Fans also like" network.

Nodes: the 5,024 dataset artists. Edges: source -> target, both in the dataset.
Labels: ghost-like / real (Phase 2), dominant ISRC registrant ("house"), home playlist.

Two nulls, 1,000 samples each:
  degree   - shuffle edge targets: every node keeps its out- and in-degree (directed
             configuration model, stub matching).
  playlist - shuffle targets only among edges whose target has the same home playlist:
             keeps degrees AND which playlists recommendations point into.

    python -m src.networks.recommendation
"""
from __future__ import annotations

import json
import random

import igraph as ig
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score as nmi

from src.common import placement
from src.load import PROCESSED, SEED, load

N_NULL = 1000


def edges() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    e = load("fans_also_like_edges")
    e = e[e.source_artist_id.isin(df.index)]
    return e, df


def home_playlist() -> pd.Series:
    pa = placement()
    return pa.groupby("artist_id").playlist_id.agg(lambda s: s.value_counts().index[0])


def _flows(s: np.ndarray, t: np.ndarray) -> np.ndarray:
    """[real->real, real->ghost, ghost->real, ghost->ghost]"""
    return np.array([np.sum(~s & ~t), np.sum(~s & t), np.sum(s & ~t), np.sum(s & t)])


def _same(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.sum((a == b) & (a != "NA")))


def null_tests(E: pd.DataFrame, df: pd.DataFrame, home: pd.Series, rng) -> dict:
    sg = E.source_artist_id.map(df.ghost).to_numpy(bool)
    tg = E.target_artist_id.map(df.ghost).to_numpy(bool)
    sr = E.source_artist_id.map(df.registrant_dom).astype(object).fillna("NA").to_numpy(str)
    tr = E.target_artist_id.map(df.registrant_dom).astype(object).fillna("NA").to_numpy(str)
    th = E.target_artist_id.map(home).astype(object).fillna("NA").to_numpy(str)
    strata = [np.where(th == h)[0] for h in np.unique(th)]

    obs_flow, obs_same = _flows(sg, tg), _same(sr, tr)
    out = {"observed": {"flows": obs_flow.tolist(), "same_house": obs_same}}
    for name in ("degree", "playlist"):
        flows, same = [], []
        for _ in range(N_NULL):
            if name == "degree":
                p = rng.permutation(len(E))
            else:
                p = np.arange(len(E))
                for idx in strata:
                    p[idx] = idx[rng.permutation(len(idx))]
            flows.append(_flows(sg, tg[p]))
            same.append(_same(sr, tr[p]))
        flows, same = np.array(flows), np.array(same)
        out[name] = {
            "flows_mean": flows.mean(0).tolist(), "flows_sd": flows.std(0).tolist(),
            "flows_ratio": (obs_flow / flows.mean(0)).tolist(),
            "flows_z": ((obs_flow - flows.mean(0)) / flows.std(0)).tolist(),
            "same_mean": float(same.mean()), "same_sd": float(same.std()),
            "same_ratio": obs_same / same.mean(), "same_z": (obs_same - same.mean()) / same.std(),
            "same_p": float((np.sum(same >= obs_same) + 1) / (N_NULL + 1)),
            "same_samples": same.tolist(),
            "flows_samples_ratio": (flows / flows.mean(0)).tolist()[:N_NULL],
        }
    return out


def communities(E: pd.DataFrame, df: pd.DataFrame, home: pd.Series, rng) -> dict:
    ids = sorted(set(E.source_artist_id) | set(E.target_artist_id))
    ix = {a: i for i, a in enumerate(ids)}
    g = ig.Graph(n=len(ids), edges=list(zip(E.source_artist_id.map(ix), E.target_artist_id.map(ix))),
                 directed=False).simplify()
    ig.set_random_number_generator(random.Random(SEED))
    part = g.community_multilevel()
    q_null = []
    for _ in range(50):
        h = g.copy()
        h.rewire(n=10 * h.ecount())
        q_null.append(h.community_multilevel().modularity)
    memb = pd.Series(part.membership, index=ids, name="community")
    A = memb.to_frame().join(df[["registrant_dom", "ghost"]]).join(home.rename("home"))
    B = A.dropna(subset=["registrant_dom"])
    lab = {
        "home playlist": (A.community, A.home.astype(str)),
        "ISRC house": (B.community, B.registrant_dom.astype(str)),
        "ghost-like label": (A.community, A.ghost.astype(str)),
    }
    res = {}
    for k, (c, l) in lab.items():
        obs = nmi(c, l)
        null = [nmi(c, rng.permutation(l.to_numpy())) for _ in range(100)]
        res[k] = {"nmi": obs, "null_mean": float(np.mean(null)), "null_sd": float(np.std(null))}
    memb.to_csv(PROCESSED / "fal_communities.csv")
    return {"n_nodes": g.vcount(), "n_edges_undirected": g.ecount(),
            "n_communities": len(part), "modularity": part.modularity,
            "modularity_null_mean": float(np.mean(q_null)), "modularity_null_sd": float(np.std(q_null)),
            "nmi": res}


def main() -> None:
    rng = np.random.default_rng(SEED)
    e, df = edges()
    home = home_playlist()
    in_set = e.target_artist_id.isin(df.index)
    E = e[in_set].reset_index(drop=True)
    sg_all = e.source_artist_id.map(df.ghost)
    real_src = e[~sg_all.astype(bool)]
    exposure = {
        "share_in_set_by_source": e.assign(i=in_set).groupby(sg_all).i.mean().rename(str).to_dict(),
        # of all "Fans also like" slots on real artists' pages, the share that is a ghost-like name
        "real_slots_to_ghost": float(real_src.target_artist_id.map(df.ghost).fillna(False).astype(bool).mean()),
        "ghost_slots_to_ghost": float(e[sg_all.astype(bool)].target_artist_id.map(df.ghost)
                                      .fillna(False).astype(bool).mean()),
    }
    out = {"n_edges_all": len(e), "n_edges_in_set": len(E), "exposure": exposure,
           "nulls": null_tests(E, df, home, rng), "communities": communities(E, df, home, rng)}
    (PROCESSED / "recommendation.json").write_text(json.dumps(out, indent=1, default=float))
    o = out["nulls"]
    print("flows [rr, rg, gr, gg]", o["observed"]["flows"])
    for k in ("degree", "playlist"):
        print(f"  {k:8s} ratio", np.round(o[k]["flows_ratio"], 2), " z", np.round(o[k]["flows_z"], 1),
              f" same-house x{o[k]['same_ratio']:.2f} z={o[k]['same_z']:.1f}")
    c = out["communities"]
    print(f"Louvain Q={c['modularity']:.3f} (null {c['modularity_null_mean']:.3f}±{c['modularity_null_sd']:.3f}),"
          f" {c['n_communities']} communities")
    for k, v in c["nmi"].items():
        print(f"  NMI with {k}: {v['nmi']:.2f} (null {v['null_mean']:.2f})")
    print("exposure", exposure)


if __name__ == "__main__":
    main()
