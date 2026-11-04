# Agent plan: from dataset to paper

Read `context.md` first: it covers what the data is and how it was collected. This file says what to do, in order. Work one phase at a time, commit after each one, and show the user the key numbers and figures before moving on.

## Ground rules

- **Never commit** `data/raw/covers/`, `data/raw/playlist_covers/`, audio, or `dataset.json`. Add them to `.gitignore` (or Git LFS if the user wants them versioned). The CSVs can be committed.
- **Reproducible:** every figure and table is produced by a script in `src/` that reads from `data/`. No hand-edited numbers in the paper. Fix random seeds.
- **Report in aggregate:** report production clusters (companies, Sources), not profiles of named individuals. Figures label companies, not people.
- **LLM and vision features are measurement instruments.** Fix the prompt, model version and temperature (0). Hand-label at least 100 items *before* looking at model output. Report Cohen's kappa for each feature, and drop any feature with poor agreement.
- **Every network statistic needs a null model** (degree-preserving rewiring, ≥1,000 samples). Report z-scores or empirical p-values.
- Don't call Spotify from Claude Code unless the user asks. The dataset is a fixed snapshot.

## Phase 0: Repo setup (≈15 min)

1. Create this layout:
   ```
   data/raw/            # unzip spotify_50_playlists.zip here; untar covers here
   data/processed/      # parquet/CSV features produced by scripts
   src/load.py          # one loader returning pandas DataFrames for all tables
   src/features/        # identity.py, production.py, demand.py, text.py, vision.py, audio.py
   src/networks/        # ownership.py, placement.py, recommendation.py, similarity.py
   src/figures/         # one script per figure
   notebooks/           # exploration only, nothing the paper depends on
   paper/               # main.tex, sections/, refs.bib
   figures/             # PDF/PNG output referenced by paper
   ```
2. `requirements.txt`: pandas, pyarrow, numpy, scipy, networkx, python-igraph or graph-tool (optional), scikit-learn, matplotlib, seaborn, statsmodels; later open_clip_torch, imagehash, pillow, librosa, transformers, bertopic.
3. `src/load.py`: read every CSV with explicit dtypes (keep IDs as strings), parse dates, and check joins. **Acceptance:** every `playlist_tracks.track_id` is in `tracks`, every `tracks.album_id` is in `albums`, and every artist id in `tracks.artist_ids` is in `artists`. Print the counts.
4. Rename `n_pilot_playlists` to `n_playlists` in the loader. See context.md for why.

## Phase 1: Descriptives and sanity checks (≈1 h)

- Tracks and artists per playlist, split functional vs genre. Overlap between playlists (Jaccard).
- Distributions of `monthly_listeners`, `followers`, `followers_per_listener` and `duration_ms`, for functional vs genre (log scales).
- The top 30 Sources, p_line owners and ISRC registrants by number of tracks and number of artist names.
- **Output:** `figures/descriptives_*.pdf` and a short `notes/descriptives.md` with the headline numbers. Check them against context.md §4. They should roughly reproduce 0.0016 vs 0.227 followers/listener and 81% alias-only credits.

## Phase 2: The three dimensions, per artist (the core)

Build one table, `data/processed/artist_features.parquet`, with one row per artist.

**Identity** (`src/features/identity.py`):
- `has_bio`, `bio_len`, `n_links`, `verified`, `has_avatar`, `n_gallery`
- `top_city_share` and the number of distinct countries in the top cities (geographic spread)
- Later, from Phase 4/5: name n-gram "generatedness", bio named-entity count, avatar has-face/AI-look
- Optional: MusicBrainz/Wikidata match (search by name; record `mb_found`)

**Production** (`src/features/production.py`):
- For each artist: their dominant Source, p_line owner and ISRC registrant
- **Production-house size:** how many distinct artist names share that Source / p_line / registrant. This is the key feature: ghosts sit in houses with many names.
- `alias_only_credits_share`: the share of the artist's tracks whose credits contain only the artist's own name
- Release cadence from `albums.release_date`: median gap between releases and its regularity (coefficient of variation)

**Demand** (`src/features/demand.py`):
- `followers_per_listener`
- Catalogue play concentration: Gini of `album_tracks.playcount` across all the artist's tracks, plus the share of plays on tracks that are on the 50 playlists
- Playlist dependence: whether the top "Discovered on" playlist is Spotify-owned, and whether it's one of our 50
- `deezer_match` rate over the artist's tracks

**Convergence (Figure 2):**
- Turn each dimension into a single score (z-scored mean of its features, or the first principal component). Avoid circularity: don't use the functional/genre label in any score.
- Plot the scores pairwise, coloured by functional vs genre playlist membership. Report Spearman correlations between dimensions, with bootstrap CIs.
- Optional: fit a latent-class or Gaussian mixture model on the three scores, then label "ghost-like" clusters. Validate on the genre playlists (expected real) and on the ground-truth names (DN list / Epidemic Sound roster) if collected.
- **Acceptance:** a figure, a correlation table, and one paragraph of interpretation.

## Phase 3: Networks (`src/networks/`)

1. **Ownership (Figure 1):** a bipartite graph of artist names and companies (Source, p_line owner and ISRC registrant as three edge types). Project it onto companies and onto artists. Headline: the degree distribution of companies (names per company) compared with the genre-playlist control. Use log-log and CCDF plots. Compute the NMI between the communities found in the artist projection (Louvain/Leiden on shared company) and the p_line owners.
2. **Placement:** a bipartite graph of playlists and artists (weighted by number of tracks). The ghost share per playlist, compared between functional and genre and against playlist description type. Test whether playlists are nested (NODF against a null).
3. **Recommendation audit (Figure 3):** the directed `fans_also_like_edges`, restricted to artists in the dataset (and optionally showing out-of-set targets as one aggregate node). Label ghost/real from Phase 2 (or by Source house size as a simpler first pass). Compute:
   - Assortativity by label
   - Conductance of the ghost set
   - The flow matrix (real→ghost, ghost→real, ghost→ghost, real→real) **against a degree-preserving directed null**
   - Funnel test: real→ghost edges more than expected while ghost→real edges are fewer
   A negative result with a convincing null is still a finding.
4. **Similarity (after Phases 4–6):** kNN graphs on CLIP cover embeddings, title embeddings and audio embeddings. Check whether their communities line up with the production houses (NMI/ARI).

## Phase 4: Text (`src/features/text.py`)

- **Track titles:** TF-IDF distinctiveness (genericness), the number of exact titles reused across different artists (an extra edge type), nature/mood word share, labMT happiness, BERTopic topics compared with the production houses.
- **Artist names:** a character n-gram language model trained on genre-playlist artists, then score functional artist names; clusters of name embeddings to find naming templates.
- **Bios:** presence/length, near-duplicate bios (embedding cosine > 0.9), named-entity count (spaCy). Real bios name places, years and collaborators.
- **Playlist descriptions:** classify each as functional vs genre language. Test ghost share against description type, which is Pelly's core claim.

## Phase 5: Vision (`src/features/vision.py`)

- Untar the covers into `data/raw/`. Note that `covers/` holds one image per album.
- Compute pHash for every cover and find exact or near reuse across different artists (Hamming distance ≤ 6).
- Compute CLIP image embeddings, build a kNN graph and detect communities (design templates).
- Compute the **literalness score**: CLIP similarity between each track title and its own cover. Compare ghost vs real.
- A vision LLM labels each cover as photo/illustration, face present, looks like stock imagery, text on cover, minimalism and looks AI-generated. **Hand-label 150 covers first**, then report kappa per feature.
- Avatars: download from `artists.avatar_url` only if needed, and gitignore them. Features: is there a face, does it look like a real photo, is it reused across names.

## Phase 6: Audio (optional, `src/features/audio.py`)

- Refresh the preview URLs via `api.deezer.com/track/{deezer_id}` and download the 30-second MP3s to `data/raw/previews/` (gitignored). Rate-limit to 8 requests/s or less.
- Use librosa for tempo, key, loudness/RMS dynamics and spectral centroid; use CLAP for embeddings.
- Narrowness: do ghost tracks sit in a tighter BPM or dynamics band than real tracks on the same playlists?
- Build an audio kNN graph and look for near-identical tracks under different names. That's the strongest possible fingerprint.

## Phase 7: Paper (`paper/`)

- Six pages. Sections: Introduction (PFC, Pelly, DN, Deezer AI share), Data & Methods (collection, ethics, market, snapshot date, feature definitions, null models, kappa protocol), Results (Fig. 1 ownership, Fig. 2 convergence, Fig. 3 recommender), Discussion (limitations: single snapshot, DK market, Deezer matching by name, Source ≠ composer), Conclusion.
- Build with `latexmk -pdf main.tex`. Figures come only from `figures/`.
- Put the sources from the brief in `refs.bib` (Pelly/Harper's 2025, MBW on DN/Firefly, Digital Music News, Spotify API change 2024-11-27, Deezer newsroom 2026-07-21, Inspect Element).

## Suggested first session prompt

> Read context.md and agent_plan.md. Do Phase 0 and Phase 1. Stop and show me the descriptive numbers and figures before continuing.
