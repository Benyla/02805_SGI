"""Write paper/numbers.tex: every number quoted in the paper as a LaTeX macro, read from the
result files. No number in the paper is typed by hand.

    python -m src.paper_numbers
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common import artist_group, placement
from src.load import PROCESSED, PROJECT, load


def pct(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}\\%"


def num(x: float) -> str:
    return f"{x:,.0f}".replace(",", "{,}")


def main() -> None:
    J = lambda f: json.loads((PROCESSED / f).read_text())  # noqa: E731
    own, rec, txt, vis, aud = (J("ownership.json"), J("recommendation.json"), J("text.json"),
                               J("vision.json"), J("audio.json"))
    df = pd.read_parquet(PROCESSED / "artist_features.parquet")
    ct = pd.read_csv(PROCESSED / "convergence_spearman.csv")
    ref = pd.read_csv(PROCESSED / "convergence_reference_sets.csv").set_index("name")
    pl, t, ar = load("playlists"), load("tracks"), load("artists")
    pt, fal, dis, cr = load("playlist_tracks"), load("fans_also_like_edges"), load("discovered_on_edges"), load("credits")
    ag = artist_group()
    A = ar.set_index("artist_id").join(ag.rename("g"))
    fpl = A.groupby("g").followers_per_listener.median()
    tg = pt.merge(pl[["playlist_id", "group"]]).groupby("track_id").group.first()
    T = t.set_index("track_id").join(tg.rename("g"))
    alias = T.groupby("g").n_credited_non_artists.apply(lambda s: (s.fillna(0) == 0).mean())

    m: dict[str, str] = {}
    m["nPlaylists"] = str(len(pl))
    m["nFunctional"] = str((pl.group == "functional").sum())
    m["nGenre"] = str((pl.group == "genre").sum())
    m["nSlots"] = num(len(pt))
    m["nTracks"] = num(len(t))
    m["nNames"] = num(len(ar))
    m["nFal"] = num(len(fal))
    m["nFalIn"] = num(rec["n_edges_in_set"])
    m["nDiscovered"] = num(len(dis))
    m["nCredits"] = num(len(cr))
    m["nCovers"] = num(load("albums").cover_file.notna().sum())
    m["nAlbums"] = num(len(load("albums")))
    m["nHouses"] = num(own["n_houses"])
    m["deezerMatch"] = pct(t.deezer_match.mean())
    m["deezerMatchF"] = pct(T[T.g == "functional"].deezer_match.mean())
    m["deezerMatchG"] = pct(T[T.g == "genre"].deezer_match.mean())
    m["nWithRegistrant"] = num(own["n_names_with_registrant"])
    m["fplF"] = f"{fpl['functional']:.4f}"
    m["fplG"] = f"{fpl['genre']:.2f}"
    m["aliasOnlyF"] = pct(alias["functional"])
    m["aliasOnlyG"] = pct(alias["genre"])
    m["verifiedF"] = pct(A[A.g == "functional"].verified.mean())
    m["verifiedG"] = pct(A[A.g == "genre"].verified.mean())

    # convergence
    for sub, tag in [("functional only", "F"), ("genre only", "G"), ("all artists", "All")]:
        c = ct[ct.subset == sub].reset_index(drop=True)
        for i, pair in enumerate(["IP", "ID", "PD"]):
            m[f"rho{pair}{tag}"] = f"{c.rho[i]:.2f}"
            m[f"rho{pair}{tag}CI"] = f"[{c.ci_lo[i]:.2f}, {c.ci_hi[i]:.2f}]"
    m["nGhost"] = num(df.ghost.sum())
    m["nGhostNoprod"] = num(df.ghost_noprod.sum())
    m["labelAgreement"] = pct(own["noprod"]["agreement_with_full_label"])
    m["uncertainShare"] = pct(((df.p_ghost > 0.1) & (df.p_ghost < 0.9)).mean(), 1)
    m["refFirefly"] = pct(ref.loc["Firefly Entertainment", "share_ghost"])
    m["refEpidemic"] = pct(ref.loc["Epidemic Sound", "share_ghost"])
    m["refGenre"] = pct(ref.loc["all 5 genre playlists", "share_ghost"])
    maj = ref[ref.set == "classical major"]
    m["refMajorsMax"] = pct(maj.share_ghost.max())
    m["nMajors"] = num(maj.n_artists.sum())
    m["nFirefly"] = str(int(ref.loc["Firefly Entertainment", "n_artists"]))
    m["nEpidemic"] = str(int(ref.loc["Epidemic Sound", "n_artists"]))
    ep = df[df.source_dom_raw.fillna("").str.contains("Epidemic", regex=False)]
    f = df[df.group == "functional"]
    m["epidemicFplRatio"] = f"{10 ** ep.log_fpl.median() / 10 ** f.log_fpl.median():.0f}"

    # placement
    m["slotGhostF"] = pct(own["functional_slot_ghost_share"])
    m["slotGhostFNoprod"] = pct(own["noprod"]["functional_slot_ghost_share"])
    m["slotGhostG"] = pct(own["genre_slot_ghost_share"])
    m["playsGhost"] = f"{own['plays_functional_ghost'] / 1e9:.1f}"
    m["playsReal"] = f"{own['plays_functional_real'] / 1e9:.1f}"
    m["rhoFollowers"] = f"{own['rho_ghostshare_followers']:.2f}"
    m["rhoFollowersP"] = f"{own['rho_p']:.2f}"
    pp = pd.read_csv(PROCESSED / "placement_playlists.csv")
    m["nPlaylistsUnderTen"] = str(int(((pp.group == "functional") & (pp.ghost_share < 0.1)).sum()))

    # ownership
    n = own["noprod"]
    m["npNames"] = num(n["with_registrant"])
    m["npHouses"] = str(n["distinct_houses"])
    m["npTopEight"] = pct(n["top8_share_names"])
    m["npTopTen"] = pct(n["top10_share_names"])
    m["npSE"] = pct(n["se_share"])
    m["npMedGhost"] = num(n["median_house_ghost"])
    m["npMedReal"] = num(n["median_house_real"])
    m["npTopEightSlots"] = pct(n["top8_share_of_ghost_slots"])
    m["seReal"] = pct(own["se_share_real_names"])
    m["fullGhostWithReg"] = num(own["ghost_names_with_registrant"])
    m["fullGhostHouses"] = str(own["ghost_distinct_houses"])
    m["selfSourceGhost"] = pct(own["self_source_share_ghost"])
    m["selfSourceReal"] = pct(own["self_source_share_real_functional"])
    m["selfSourceHouse"] = num(own["self_source_house_median"])
    B = pd.read_csv(PROCESSED / "ownership_edges.csv", dtype=str)
    m["nMultiHouse"] = num((B.groupby("primary_artist").registrant.nunique() > 1).sum())

    # recommendation
    nl = rec["nulls"]
    names = ["RR", "RG", "GR", "GG"]
    for null, tag in [("degree", "Deg"), ("playlist", "Pl")]:
        for i, k in enumerate(names):
            m[f"flow{k}{tag}"] = f"{nl[null]['flows_ratio'][i]:.2f}"
            m[f"z{k}{tag}"] = f"{abs(nl[null]['flows_z'][i]):.0f}"
        m[f"same{tag}"] = f"{nl[null]['same_ratio']:.2f}"
        m[f"zSame{tag}"] = f"{nl[null]['same_z']:.0f}"
    m["sameObs"] = num(nl["observed"]["same_house"])
    m["sameExpDeg"] = num(nl["degree"]["same_mean"])
    m["sameExpPl"] = num(nl["playlist"]["same_mean"])
    m["realSlotsToGhost"] = pct(rec["exposure"]["real_slots_to_ghost"])
    m["inSetReal"] = pct(rec["exposure"]["share_in_set_by_source"]["False"])
    c = rec["communities"]
    m["louvainQ"] = f"{c['modularity']:.2f}"
    m["louvainQnull"] = f"{c['modularity_null_mean']:.2f}"
    m["louvainQnullSd"] = f"{c['modularity_null_sd']:.3f}"
    m["nComm"] = str(c["n_communities"])
    m["nGiant"] = num(c["n_nodes"])
    m["nmiHome"] = f"{c['nmi']['home playlist']['nmi']:.2f}"
    m["nmiHomeNull"] = f"{c['nmi']['home playlist']['null_mean']:.2f}"
    m["nmiHouse"] = f"{c['nmi']['ISRC house']['nmi']:.2f}"
    m["nmiHouseNull"] = f"{c['nmi']['ISRC house']['null_mean']:.2f}"
    m["nmiLabel"] = f"{c['nmi']['ghost-like label']['nmi']:.2f}"

    # text / vision / audio
    for k, v in [("Name", txt["auc_name"]), ("Titles", txt["auc_titles"]), ("Bio", txt["auc_bio"]),
                 ("Cover", vis["auc_cover"]), ("Audio", aud["auc_audio"])]:
        m[f"auc{k}"] = f"{v['auc']:.2f}"
        m[f"auc{k}Null"] = f"{v['null_mean']:.2f}"
        m[f"n{k}"] = num(v["n"])
    m["nAudioPlaylists"] = str(aud["auc_audio"]["n_playlists"])
    m["bioGhost"] = pct(txt["bio_presence"]["ghost"])
    m["bioReal"] = pct(txt["bio_presence"]["real"])
    m["entGhost"] = pct(txt["bio_any_entity"]["True"])
    m["entReal"] = pct(txt["bio_any_entity"]["False"])
    m["tokGhost"] = num(txt["bio_median_tokens"]["True"])
    m["tokReal"] = num(txt["bio_median_tokens"]["False"])
    m["personGhost"] = pct(txt["person_like_name"]["ghost"])
    m["personRealF"] = pct(txt["person_like_name"]["real_functional"])
    m["personGenre"] = pct(txt["person_like_name"]["genre"])
    m["coverSameHouse"] = pct(vis["cover_nn_same_house"]["observed"])
    m["coverSameHouseNull"] = pct(vis["cover_nn_same_house"]["null_mean"])
    m["phashPairs"] = str(vis["phash_pairs_across_names"])
    m["titleReuseGhost"] = pct(txt["title_reused_by_other_name"]["True"])
    m["titleReuseReal"] = pct(txt["title_reused_by_other_name"]["False"])

    out = ["% Generated by `python -m src.paper_numbers`. Do not edit by hand."]
    m = {k: ("0.00" if v == "-0.00" else v) for k, v in m.items()}
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (PROJECT / "paper" / "numbers.tex").write_text("\n".join(out) + "\n")
    print(f"wrote paper/numbers.tex ({len(m)} macros)")


if __name__ == "__main__":
    main()
