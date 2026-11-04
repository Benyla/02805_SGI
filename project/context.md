# Project context: Ghost Artists on Spotify

Course 02805 Social Graphs and Interactions. Working title: *"Many faces, few hands: how credit networks reveal Spotify's ghost artists, and where the recommender puts them."*

Read this file first. `agent_plan.md` says what to do next.

---

## 1. The research question

Spotify fills mood and background playlists (Focus, Sleep, Chill, Ambient…) with cheap stock music released under many invented artist names ("Perfect Fit Content", PFC). Known suppliers include Epidemic Sound and Firefly Entertainment. The brief defines a **ghost artist** on three independently measurable dimensions:

| Dimension | Meaning | Measured with (in this dataset) |
|---|---|---|
| **Identity** | The name is not an identifiable person or band | bio presence and length, external links, verified flag, avatar, top-city spread, name and title text features, cover/avatar vision features, MusicBrainz/Wikidata (not collected yet) |
| **Production** | The catalogue is made by companies or people behind many names | credit **Source** (`track_sources.csv`), (P) line and label (`albums.csv`), ISRC registrant (`tracks.csv`), credited writers (`credits.csv`) |
| **Demand** | Streams come from placement, not from fans | `followers_per_listener`, play concentration across the catalogue (Gini), dependence on a single playlist (`discovered_on_edges.csv`) |

**The validation strategy:** if the three dimensions point to the same names, that convergence is the second method the grading rubric asks for.

Figures planned in the brief:
1. The credits/ownership network: hundreds of names collapsing onto a few companies, compared with the real-artist control group.
2. The three dimensions plotted against each other (convergence), with Cohen's kappa reported for every LLM or vision feature.
3. The recommendation flow matrix ("Fans also like") against a degree-preserving null, plus displacement rates (only once daily snapshots exist).

The original brief is `ghost_artists_project_brief.pdf`, in the project or repo.

---

## 2. How the data was collected

- **Date:** 2026-11-04, a single snapshot. Market: **Denmark** (set by IP). Anonymous web-player session, no login.
- **Why not the official Web API:** since 27 Nov 2024, new apps can't read editorial playlists, related artists, audio features or recommendations. We also checked: batch endpoints return 403, follower and popularity fields have been removed, and editorial playlists return 404.
- **Source:** Spotify's private web-player GraphQL API, `POST https://api-partner.spotify.com/pathfinder/v2/query` with persisted-query hashes. The crawl ran inside an open `open.spotify.com` tab, reusing the web player's own `authorization` and `client-token` headers and refreshing them automatically when they expired. Operations and hashes as of 2026-11-04 (they rotate every few weeks):
  - `fetchPlaylist` `21ea11a6ac75a5f3e01fd365d29c18e0a4a8b7dc09f89bbb5d73910190c6acaf`
  - `getAlbum` `0b6ce3511e1b02e4a40fa95440f992acbe85080eb308af9194322d708fc1914f`
  - `queryArtistOverview` `9f8134ef565e78621f1e1793555bd6633c5ac144ae0f89604ed3ae3f80b3c8e6`
  - `queryTrackCreditsGroupedModal` `f135fb9be58a72d041ab5d214d817021a272405d883860468e2627afb01a3ca9`
  - `browseAll` / `browsePage` (category pages, used to enumerate the playlists)
- **Rate:** about 10 requests per second. Roughly 25k requests in total, with no 429 errors.
- **Deezer enrichment:** the public API (`api.deezer.com`) matched tracks by artist and title search, then checked duration. It returns ISRC, BPM, gain, rank and a 30-second preview URL. About 76% of tracks match. **A missing Deezer match is itself a possible signal**, since Deezer demonetises and demotes functional/noise catalogue.
- **Ethics, for Methods:** Spotify's terms forbid scraping, so the crawl was kept small. No audio was downloaded from Spotify; audio analysis must use only Deezer's official previews, processed locally. Some ghost names belong to real working musicians, so production clusters are reported **in aggregate**, never as profiles of individuals.

### Playlist selection
- All Spotify-owned playlists listed on the DK Browse page: 2,622. Of these, 1,693 are `format=editorial` and 850 are `isAlgotorial=true` (algorithmically filled per user).
- **Decision: editorial only, non-algotorial.** These are hand-curated and the same for every listener, so snapshots are reproducible.
- **50 playlists:** 45 **functional** (Focus, Sleep, Chill, Instrumental, Background, Nature, Wellness, mood Jazz/Classical, e.g. Deep Focus, Intense Studying, Sleep, lofi beats, Peaceful Meditation) and 5 **genre** controls with known real artists (ALT NOW, Rap UK, Indie Twang, Desi Indie, Country Latino). The `group` column in `playlists.csv` holds this split.

---

## 3. The files

The data lives in `spotify_50_playlists.zip` (CSV, UTF-8; IDs are bare Spotify IDs, not URIs). The covers are in `spotify_covers_640_partXX_of_06.tar`; unpacking gives `covers/<album_id>.jpg` and `playlist_covers/<playlist_id>.jpg`.

| File | Rows | Key | Notes |
|---|---|---|---|
| `playlists.csv` | 50 | playlist_id | group (functional/genre), category, followers, n_tracks, description, cover_file |
| `playlist_tracks.csv` | 9,004 | playlist_id + position | added_at, track_id, artist_ids. **The placement layer.** One snapshot (2026-11-04) |
| `tracks.csv` | 8,093 | track_id | duration, playcount, label, p_line, credit summary (`sources`, `n_credited_non_artists`), cover_file, Deezer fields (`isrc`, `isrc_registrant` = ISRC chars 3–5, `bpm`, `gain`, `deezer_preview`, `deezer_match`) |
| `albums.csv` | 8,056 | album_id | release_date, label, p_line, c_line, total_plays, play_gini (within album), cover_url/cover_file |
| `album_tracks.csv` | 14,990 | album_id + track_id | Up to 50 tracks per album with playcount. Use it for catalogue concentration and release cadence |
| `artists.csv` | 5,024 | artist_id | monthly_listeners, followers, followers_per_listener, verified, ai_persona, top_city/top_city_share, bio, bio_len, links, n_links, has_avatar, avatar_url, discography counts, related_total, n_discovered_on |
| `artist_top_cities.csv` | 25,084 | artist_id + rank | Top 5 cities with listener counts |
| `credits.csv` | 28,933 | track_id | Every credited name, role and role_group |
| `track_sources.csv` | 8,087 | track_id | The credits dialog's "Source" (label or distributor). **The strongest production signal** |
| `fans_also_like_edges.csv` | 100,380 | source→target | Directed, with rank (up to 20 per artist). Targets are often outside the dataset |
| `discovered_on_edges.csv` | 71,442 | artist→playlist | Playlists where the artist's listeners found them, with owner |
| `dataset.json` | | | All of the above in one file |

Column quirks:
- `n_pilot_playlists` (in tracks and artists) means **the number of the 50 dataset playlists** the item appears on. The name is a leftover from the pilot.
- `catalog_play_gini_partial` uses only albums that appear in the dataset, not the artist's full discography.
- `bpm` is 0 for most Deezer tracks, so compute tempo from the previews instead.
- `deezer_preview` URLs are signed and expire. Re-request `https://api.deezer.com/track/{deezer_id}` to get a fresh one before downloading.
- `ai_persona` was null for every artist when collected. Note it, don't rely on it.

---

## 4. What we already know (from the pilot and the full crawl)

- **Credits rarely name humans on functional tracks.** On functional playlists, 81% of tracks credit only the alias itself (vs 5% on genre playlists). This is the brief's "kill condition": writer credits are weak, so build the production layer from **Source + (P) line + ISRC registrant**.
- Top credit Sources: Chill Palm (450 tracks), Overtone Studios (373), Score a Score (308), Chill Lemon (163), Calm Lands Music (113), **Firefly Entertainment AB (96)** (the company in the Dagens Nyheter investigation), Azure Noon, Spirit Of Hule, Gunn-R Rotation, Chill Caterpillar.
- Median followers per monthly listener: **0.0016** for artists on functional playlists vs **0.227** on genre playlists (more than 100× higher). Deutsche Grammophon artists on the functional playlists sit around 0.16, so real classical artists act as an in-playlist control.
- Some Sources are missing from Deezer entirely (Score a Score: 100% unmatched in the pilot).

---

## 5. Not collected yet (possible extensions)

- **Daily snapshots** of the 50 playlists, needed for the displacement/churn analysis. This needs a Playwright script run daily (`fetchPlaylist` only, about 100 requests per day).
- **MusicBrainz / Wikidata lookups** for the identity dimension. Both are public APIs; rate-limit MusicBrainz to 1 request/s and send a User-Agent.
- **Artist avatars** as image files (the URLs are in `artists.csv`).
- **Deezer preview audio** for tempo, loudness and CLAP embeddings.
- **Ground-truth names**: Epidemic Sound's roster, names published by DN and in Pelly's reporting.
- The full discography per artist (`queryArtistDiscographyAll`), if the release-cadence analysis needs more than the albums in the dataset.
