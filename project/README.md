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
# Raw data (only needed on a fresh clone: the CSVs are committed, the covers are not)
#   put spotify_50_playlists.zip and spotify_covers_640_part0{1..6}_of_06.tar in data/raw/archives/
unzip -qj data/raw/archives/spotify_50_playlists.zip -d data/raw
for f in data/raw/archives/*.tar; do tar xf "$f" -C data/raw; done
python -m src.load                    # row counts and join checks
python -m src.figures.descriptives    # phase 1 figures and notes/descriptives.md
```

All scripts are run from `project/` as modules (`python -m src...`).

## Ethics

Production clusters are reported in aggregate (companies and Sources), never as profiles of named individuals. See `context.md` §2.
