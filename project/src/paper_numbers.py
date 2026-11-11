"""Write paper/numbers.tex: every number quoted in the paper as a LaTeX macro, read from the
result files. No number in the paper is typed by hand.

    python -m src.paper_numbers
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.common import artist_group
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
    ref = pd.read_csv(PROCESSED / "convergence_reference_sets.csv").set_index("name")
    prof = pd.read_csv(PROCESSED / "convergence_profile.csv").set_index("group")
    pl, t, ar = load("playlists"), load("tracks"), load("artists")
    pt, fal = load("playlist_tracks"), load("fans_also_like_edges")
    A = ar.set_index("artist_id").join(artist_group().rename("g"))
    fpl = A.groupby("g").followers_per_listener.median()
    tg = pt.merge(pl[["playlist_id", "group"]]).groupby("track_id").group.first()
    T = t.set_index("track_id").join(tg.rename("g"))
    alias = T.groupby("g").n_credited_non_artists.apply(lambda s: (s.fillna(0) == 0).mean())
    f = df[df.group == "functional"]

    m: dict[str, str] = {}
    # ---- data
    m["nPlaylists"] = str(len(pl))
    m["nFunctional"] = str((pl.group == "functional").sum())
    m["nGenre"] = str((pl.group == "genre").sum())
    m["nSlots"] = num(len(pt))
    m["nTracks"] = num(len(t))
    m["nNames"] = num(len(ar))
    m["nFal"] = num(len(fal))
    m["nFalIn"] = num(rec["n_edges_in_set"])
    m["nCovers"] = num(load("albums").cover_file.notna().sum())
    m["nHouses"] = num(own["n_houses"])
    m["nWithRegistrant"] = num(own["n_names_with_registrant"])
    m["deezerMatch"] = pct(t.deezer_match.mean())
    m["deezerMatchF"] = pct(T[T.g == "functional"].deezer_match.mean())
    m["deezerMatchG"] = pct(T[T.g == "genre"].deezer_match.mean())
    m["fplF"] = f"{fpl['functional']:.4f}"
    m["fplG"] = f"{fpl['genre']:.2f}"
    m["aliasOnlyF"] = pct(alias["functional"])
    m["aliasOnlyG"] = pct(alias["genre"])
    m["verifiedF"] = pct(A[A.g == "functional"].verified.mean())
    m["verifiedG"] = pct(A[A.g == "genre"].verified.mean())

    # ---- the flag and its checks
    m["nGhost"] = num(df.ghost.sum())
    m["ghostShareNamesF"] = pct(f.ghost.mean())
    for key, tag in [("Firefly Entertainment", "Firefly"), ("Epidemic Sound", "Epidemic"),
                     ("Classical majors", "Majors"), ("Genre-playlist artists", "GenreNames")]:
        m[f"ref{tag}"] = pct(ref.loc[key, "share_ghost"])
        m[f"n{tag}"] = num(ref.loc[key, "n_artists"])
    m["nGenreFlagged"] = str(own["genre_names_flagged"])
    ep = df[df.source_dom_raw.fillna("").str.contains("Epidemic", regex=False)]
    m["epidemicFplRatio"] = f"{10 ** ep.log_fpl.median() / 10 ** f.log_fpl.median():.0f}"
    m["epidemicBio"] = pct(1 - ep.no_bio.mean())
    for g, tag in [("flagged", "G"), ("not flagged", "R")]:
        p = prof.loc[g]
        m[f"prof{tag}N"] = num(p.n)
        m[f"prof{tag}House"] = num(p.house_size_median)
        m[f"prof{tag}Fpl"] = f"{p.followers_per_1000_listeners:.1f}" if tag == "G" else \
            f"{p.followers_per_1000_listeners:.0f}"
        m[f"prof{tag}Alias"] = pct(p.alias_only_tracks)
        m[f"prof{tag}Self"] = pct(p.self_labelled)

    # ---- placement
    m["slotGhostF"] = pct(own["functional_slot_ghost_share"])
    m["slotGhostG"] = pct(own["genre_slot_ghost_share"])
    m["playsGhost"] = f"{own['plays_functional_ghost'] / 1e9:.1f}"
    m["playsReal"] = f"{own['plays_functional_real'] / 1e9:.1f}"

    # ---- ownership
    m["ghostWithReg"] = num(own["ghost_names_with_registrant"])
    m["ghostHouses"] = str(own["ghost_distinct_houses"])
    m["topEightGhostNames"] = pct(own["top8_share_of_ghost_names"])
    m["topEightGhostSlots"] = pct(own["top8_share_of_ghost_slots"])
    m["topEightSE"] = str(own["ghost_top8_swedish"])
    m["seGhost"] = pct(own["se_share_ghost_names"])
    m["seReal"] = pct(own["se_share_real_names"])
    m["medHouseGhost"] = num(own["median_house_ghost"])
    m["medHouseReal"] = num(own["median_house_real"])
    m["selfSourceGhost"] = pct(own["self_source_share_ghost"])
    m["selfSourceReal"] = pct(own["self_source_share_real_functional"])
    m["selfSourceHouse"] = num(own["self_source_house_median"])
    m["nMultiHouse"] = num(own["n_multi_house"])
    lf = own["label_free"]
    m["lfTopEightCodedF"] = pct(lf["functional"]["top8_share_of_coded"])
    m["lfTopEightAllF"] = pct(lf["functional"]["top8_share_of_all"])
    m["lfNoCodeF"] = pct(1 - lf["functional"]["with_code"])
    m["lfTopEightCodedG"] = pct(lf["genre"]["top8_share_of_coded"])
    m["lfHousesF"] = num(lf["functional"]["n_houses"])
    m["lfHousesG"] = num(lf["genre"]["n_houses"])

    # ---- recommendation
    nl = rec["nulls"]
    for null, tag in [("degree", "Deg"), ("playlist", "Pl")]:
        for i, k in enumerate(["RR", "RG", "GR", "GG"]):
            m[f"flow{k}{tag}"] = f"{nl[null]['flows_ratio'][i]:.2f}"
        m[f"same{tag}"] = f"{nl[null]['same_ratio']:.2f}"
    rng = [abs(v - 1) for null in ("degree", "playlist")
           for v in np.ravel(nl[null]["flows_null_range"]).tolist() + nl[null]["same_null_range"]]
    m["nullRange"] = pct(max(rng))
    m["sameObs"] = num(nl["observed"]["same_house"])
    m["sameExpPl"] = num(nl["playlist"]["same_mean"])
    ex = rec["exposure"]
    m["realSlotsToGhost"] = pct(ex["real_slots_to_ghost"])
    m["ghostSlotsToGhost"] = pct(ex["ghost_slots_to_ghost"])
    m["inSetReal"] = pct(ex["share_in_set_by_source"]["False"])
    m["inSetGhost"] = pct(ex["share_in_set_by_source"]["True"])
    c = rec["communities"]
    m["louvainQ"] = f"{c['modularity']:.2f}"
    m["louvainQnull"] = f"{c['modularity_null_mean']:.2f}"
    m["nComm"] = str(c["n_communities"])
    m["nGiant"] = num(c["n_giant"])
    m["nmiHome"] = f"{c['nmi']['home playlist']['nmi']:.2f}"
    m["nmiHomeNull"] = f"{c['nmi']['home playlist']['null_mean']:.2f}"
    m["nmiHouse"] = f"{c['nmi']['ISRC house']['nmi']:.2f}"
    m["nmiHouseNull"] = f"{c['nmi']['ISRC house']['null_mean']:.2f}"
    m["nmiLabel"] = f"{c['nmi']['ghost-like label']['nmi']:.2f}"

    # ---- what a listener sees
    for k, v in [("Name", txt["auc_name"]), ("Titles", txt["auc_titles"]), ("Bio", txt["auc_bio"]),
                 ("Cover", vis["auc_cover"]), ("Audio", aud["auc_audio"])]:
        m[f"auc{k}"] = f"{v['auc']:.2f}"
        m[f"n{k}"] = num(v["n"])
    m["aucBioLength"] = f"{txt['auc_bio_length']:.2f}"
    m["nAudioPlaylists"] = str(aud["auc_audio"]["n_playlists"])
    m["bioGhost"] = pct(txt["bio_presence"]["ghost"])
    m["bioReal"] = pct(txt["bio_presence"]["real"])
    m["entGhost"] = pct(txt["bio_any_entity"]["True"])
    m["entReal"] = pct(txt["bio_any_entity"]["False"])
    m["tokGhost"] = num(txt["bio_median_tokens"]["True"])
    m["tokReal"] = num(txt["bio_median_tokens"]["False"])
    m["personGhost"] = pct(txt["person_like_name"]["ghost"])
    m["personRealF"] = pct(txt["person_like_name"]["real_functional"])
    m["coverSameHouse"] = pct(vis["cover_nn_same_house"]["observed"])
    m["coverSameHouseNull"] = pct(vis["cover_nn_same_house"]["null_mean"])

    out = ["% Generated by `python -m src.paper_numbers`. Do not edit by hand."]
    m = {k: ("0.00" if v == "-0.00" else v) for k, v in m.items()}
    out += [f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in m.items()]
    (PROJECT / "paper" / "numbers.tex").write_text("\n".join(out) + "\n")
    print(f"wrote paper/numbers.tex ({len(m)} macros)")


if __name__ == "__main__":
    main()
