# 02805 Network Stories

Our group's weekly data stories from the Marvel Wikipedia superhero network — built for [02805 Social Graphs and Interactions](https://sunelehmann.com/socialgraphs2026-web/).

## Site

After enabling GitHub Pages (Settings → Pages → branch `main`, folder `/docs`), the site lives at:

**https://benyla.github.io/02805_SGI/**

## Re-run Week 1 analysis

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python analysis/week1_explore.py
```

Figures are written to `docs/assets/week1/`.

## Data

Frozen week-1 snapshot in [`DATA/`](DATA/): 303 Marvel superheroes, 1,784 directed wiki-links.
