# 02805 Social Graphs and Interactions

Course work for [02805 Social Graphs and Interactions](https://sunelehmann.com/socialgraphs2026-web/). The repo has two separate parts:

| Folder | What |
|---|---|
| [`project/`](project/) | Final project: *Many faces, few hands*, about ghost artists on Spotify. See [`project/README.md`](project/README.md). |
| [`weekly/`](weekly/) | Weekly exercises (Marvel Wikipedia superhero network). |
| [`docs/`](docs/) | GitHub Pages site with the weekly data stories. It stays at the repo root because Pages only serves from `/` or `/docs`. |

## Site

https://benyla.github.io/02805_SGI/ (Settings → Pages → branch `main`, folder `/docs`).

## Weekly exercises

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r weekly/requirements.txt
python weekly/analysis/week1_explore.py   # writes figures to docs/assets/week1/
```

The frozen week 1 snapshot is in [`weekly/data/`](weekly/data/): 303 Marvel superheroes and 1,784 directed wiki links.
