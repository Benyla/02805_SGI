# Many faces, few hands

How credit networks reveal Spotify's ghost artists, and where the recommender puts them.
Final project for 02805 Social Graphs and Interactions.

- [`context.md`](context.md): the research question, how the data was collected, and what each file holds. **Read this first.**
- [`agent_plan.md`](agent_plan.md): the phased work plan.
- [`notes/`](notes/): results notes per phase.

## Layout

```
data/raw/            CSVs from spotify_50_playlists.zip (committed); covers/, playlist_covers/,
                     dataset.json and archives/ are gitignored
data/processed/      features produced by scripts
src/load.py          one loader for every table, with join checks
src/features/        identity, production, demand, text, vision, audio
src/networks/        ownership, placement, recommendation, similarity
src/figures/         one script per figure
notebooks/           exploration only; nothing the paper depends on
paper/               main.tex, sections/, refs.bib
figures/             output referenced by the paper
```

## Setup

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r project/requirements.txt
cd project
# Raw data (only on a fresh clone: the CSVs are committed, covers are not)
#   put spotify_50_playlists.zip and spotify_covers_640_part0{1..6}_of_06.tar in data/raw/archives/
unzip -qj data/raw/archives/spotify_50_playlists.zip -d data/raw
for f in data/raw/archives/*.tar; do tar xf "$f" -C data/raw; done
```

## Run order

All scripts run from `project/` as modules. Seeds are fixed.

| Step | Command | Output |
|---|---|---|
| 0 | `python -m src.load` | row counts, join checks |
| 1 | `python -m src.figures.descriptives` | `notes/descriptives.md` |
| 2 | `python -m src.features.build` then `python -m src.features.convergence` | `artist_features.parquet` (with the `ghost` flag), `notes/convergence.md` |
| 3 | `python -m src.networks.ownership`, `python -m src.networks.recommendation` | `ownership.json`, `recommendation.json` |
| 4 | `python -m src.features.text` | `text.json` (needs `python -m spacy download en_core_web_sm`) |
| 5 | `python -m src.features.vision`, `python -m src.features.vision_analysis` | CLIP embeddings, `vision.json` (exploratory) |
| 6 | `python -m src.features.audio`, `python -m src.features.audio_analysis` | Deezer previews (gitignored), `audio.json` (exploratory) |
| figs | `python -m src.figures.fig1_data` … `fig5_text`, `cover_mosaic` | `figures/`, `paper/cover_mosaic.jpg` |

The ghost-like flag is one rule: a name lacks at least 4 of 5 identity markers (`src/features/convergence.py`). The previous version of the paper is kept in `paper/archive/`.
| paper | `paper/build.sh` | `paper/report.pdf` = front page + 6-page PNAS paper |

Every number in the paper is a macro in `paper/numbers.tex`, written by `python -m src.paper_numbers`.

## Ethics

Production clusters are reported in aggregate (companies and Sources), never as profiles of named individuals. See `context.md` §2.
