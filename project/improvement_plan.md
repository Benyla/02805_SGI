# Improvement plan

Second version, 11 November 2026. The aim of this version is a report that is easier to follow: fewer methods, one label, figures that can be read at first glance. Nothing here adds an analysis unless it replaces something more complicated.

All numbers were recomputed from the committed data.

## 1. The short version

1. **Replace the mixture model with one plain rule.** A name is ghost-like if it lacks at least 4 of 5 identity markers. This removes the mixture model, the z-scores, the correlations and the second label. See section 3.
2. **Tell the story in three steps:** who is on the playlists, who makes the music, where the recommender puts it. See section 4.
3. **Replace Figure 1** with a flow diagram drawn from the real data. Simplify the other figures. See section 5.
4. **Fix the errors in section 2.**
5. **Restructure the text** so each term is defined before it is used and each section opens with its finding. See section 6.

## 2. Errors to fix

| # | Where | Problem | Fix |
|---|---|---|---|
| E1 | Fig. 4 caption, `\nGiant` | "largest connected component, 4,827 names". 4,827 is the number of names with at least one in-set link. The largest component has **4,620** names. `paper_numbers.py` line 137 writes `n_nodes` into `nGiant`. | Store the giant size in `recommendation.json` and use it. |
| E2 | Fig. 4A title | "among the 5,024 names", but the panel draws the giant component only. | Drop the count from the title. |
| E3 | Abstract and Results | "Of 2,333 ghost-like names found without production data" reads as if 2,333 is all of them. 2,333 are the flagged names that have a recording code. | Say "of the flagged names with a recording code". |
| E4 | Results, "none of the genre slots" | Not exactly true. A few featured artists on *Desi Indie* are flagged and are clearly real. | Give the actual small figure and say these are false positives. |
| E5 | Results, Epidemic Sound | "One known supplier is missed" is the wrong framing. Epidemic's own names have bios, links, credited composers and 11× the followers per listener. By the paper's own definition they are not ghosts. | Say that Epidemic supplies this kind of music under real, credited composers, so its own names are correctly not flagged. |
| E6 | Abstract | "keeps them apart (0.62×)" uses the degree null only. The paper's own conclusion is that playlists explain most of the gap. | Quote the playlist-controlled figure, or both. |
| E7 | Fig. 3 and text | Three house sizes are in use for the same house (SE69Z: 409, 368, 358), and two counts of names with a code (3,719 and 3,446). | Use one definition everywhere: the dominant house of each primary name. |
| E8 | Results, "seven of them Swedish" | The code prefix shows which national agency issued the code, not where the company is. | Write "Swedish-issued codes" once. |
| E9 | Results and Methods | z-scores of 89 and 33 treat links as independent. They are not. | Report the ratio and the null's range. Drop the z-scores. |
| E10 | `main.tex` | Author contributions are still a TODO. | Fill in. |

**Remove everything about hand labels.** This is the TODO comment in the Limitations section of `main.tex`, the "Hand labels" section of `README.md`, `notes/hand_label_sheet.csv`, `src/features/kappa.py`, and the part of `convergence.py` that writes the sheet and the note.

## 3. Simplify the method

### 3.1 One label, stated as a rule

**Now:** a two-component Gaussian mixture on three z-scores, plus a second mixture without production for the ownership figure. Two labels, one model the reader has to trust.

**Proposed:** *a name is ghost-like if it lacks at least 4 of these 5: verified badge, bio, external link, gallery photo, "This Is" playlist.*

Why this is the right simplification:

- **It is already what the paper uses.** The production-free label behind Fig. 3 and the abstract's house claim differs from this rule on 26 of 5,024 names.
- **It removes the circularity.** The label uses no production or listening data, so "these names come from few houses" and "these names have no fans" become findings, not inputs.
- **It removes an artefact.** The current mixture collapses onto one value of the production score and calls 373 names with no identity markers at all "real".
- **It is one sentence.** No model, no posterior, no threshold to justify.

How the rule does on the reference groups:

| Group | Flagged by the rule | Current mixture |
|---|---|---|
| Firefly names | 90% | 94% |
| Epidemic Sound's own names | 10% | 0% |
| Classical majors | 18% | 1% |
| Genre-playlist names | 14 of 493 | 3 of 493 |
| Functional playlist slots | 75% | 65% |
| Flagged names with a code: share in 8 largest houses | 80% | (uses second label) |

**The cost, stated openly:** about one in six classical-major names has no profile either (session soloists, ensembles, "Traditional"). That is the rule's false-positive rate and the paper should say so. It works against the main finding, because those names sit in many small houses.

**Consequence for the headline:** two-thirds of slots becomes three-quarters. Rerun the pipeline and let `paper_numbers.py` update the rest.

### 3.2 Replace "three scores agree" with three plain comparisons

**Now:** z-scored composite scores, Spearman correlations with bootstrap intervals, pooled and by group. This is also the paper's weakest claim: most of the correlation is between playlists, and within a playlist it nearly disappears (production against demand: 0.36 pooled, 0.08 with playlist means removed).

**Proposed:** drop the scores and the correlations. Show what flagged and unflagged names look like on the same playlists, in raw units:

| Among functional-playlist names | Flagged | Not flagged |
|---|---|---|
| Other names in the same house (median) | 232 | 9 |
| Followers per 1,000 monthly listeners (median) | 1.4 | 25 |
| Tracks that credit no person | 86% | 45% |
| Name is listed as its own label | 55% | 13% |

A reader understands this table in one pass, and it carries the same message: names without an identity are also mass-produced and have no fans.

### 3.3 Other things to cut or shrink

| Item | Action | Why |
|---|---|---|
| The two-label distinction | Cut | One rule replaces both. |
| "Demand score", "production score", "identity score" as named quantities | Cut | Use the raw quantities in 3.2. |
| z-scores for the null models | Cut | See E9. |
| Louvain modularity against rewired networks | One sentence in Methods | True but adds little. Any "20 similar artists" network is highly modular. The comparison with playlists is the result. |
| Fig. 3B and 3C (two log-scale cumulative curves) | Replace with one plain panel | See 5.3. These are the hardest panels in the paper to read. |
| Fig. 2A (score against score) | Cut | Replaced by the table in 3.2. |
| Named-entity recognition | Keep as one sentence | It is simple and already written. |
| Covers and audio | Keep as two rows in one panel, plus one sentence each in the data section | They support "a listener cannot tell" and readers asked where they went. |

### 3.4 Things I checked that need no change

These came out fine. They are worth at most one sentence each in the paper.

- **Deezer matching is accurate.** 99.4% exact titles, the artist always matches, 99% within 2 seconds. The weakness is coverage (72% of functional tracks against 94% of genre tracks), which the paper already states.
- **Names in more than one house are normal.** 225 names. For ghost-like names it mostly links large houses to each other (76 names share SE5IB and SE69Z), which supports the existing sentence that a house is an upper bound on the number of operators.
- **The blue dots inside the big houses in Fig. 3A are not real artists.** They are names from the same houses that happen to have a written bio. One caption sentence covers it.
- **The recommender results are robust.** They hold among functional names only and under a stricter null. No new analysis needs to go in the paper.

## 4. The story

The fair criticism is that "a few houses supply the music" repeats the journalism and that the paper has too many findings. The answer is a three-step story with one figure per step:

1. **Who is on the playlists.** Three-quarters of mood-playlist slots go to names with no public identity. (New Fig. 1.)
2. **Who makes the music.** The credits say hundreds of self-released artists. The recording codes say eight houses. This is the part the journalism needed leaks for: the public credits hide it and the codes reveal it. (Fig. 3.)
3. **Where the recommender puts them.** No funnel from real artists to ghosts. The "similar artists" map reproduces Spotify's own playlists, and names from one house stay together beyond that. This is the new, network-science result and deserves the most space. (Fig. 4.)

Then two short supporting pieces: the label passes outside checks, and a listener cannot tell.

One label-free sentence strengthens step 2 at no cost in complexity: **eight houses registered the tracks in 72% of functional slots with a code, against 22% for the eight largest houses on genre playlists.**

One wording change for step 3: write "names from the same house are recommended together more than playlists explain" and leave the cause open. Shared composers and a shared sound could produce this without the recommender "grouping" anything.

## 5. Figures

Rules for every figure:

- **The title states the finding**, so the figure can be understood without the caption.
- **Every mark is data and has one meaning.** No decorative or invented elements.
- **Orange is ghost-like and blue is real, everywhere.** Keep this; it is the best thing about the current set.
- **No text under 6.5 pt.** Labels are 5.3–5.8 pt now.
- **Use the panel area.** Fig. 3A and 4A spend about half their space on empty margin.
- **Avoid log axes and cumulative curves** where a bar or a count says the same thing.

The custom network drawings are the strength of the paper. Keep them and make them larger.

Proposed set: three full-width figures for the three steps of the story, and two small single-column figures for the supporting pieces.

### 5.1 Figure 1: replace with a flow diagram of the real data

The current figure cannot be understood because it is not data: `fig1_data.py` places the nodes and links with random numbers. Only the counts in the right-hand column are real.

**Replacement.** Three columns joined by bands. Band width is the number of playlist slots.

- **Left: playlists**, grouped into about six kinds (sleep, focus, chill, jazz, classical and ambient, genre controls).
- **Middle: names**, ghost-like (orange) or real (blue).
- **Right: houses**: the eight largest, "all other houses", and "no recording code".

Title: "Three-quarters of mood-playlist slots go to names without an identity, and most of those come from eight houses."

What it does: introduces the three layers of the data, shows the headline, and shows the 28% of slots without a code instead of hiding them. Add one line under each column with the real counts (50 playlists, 5,024 names, 486 houses) and one line saying what else is attached to names (bios, 8,056 covers, 624 audio previews).

The "Fans also like" links do not belong in this figure. Introduce them in Fig. 4.

### 5.2 Figure 2 (small): the label passes outside checks

One panel, single column. Dots on a 0–100% axis, one row per reference group, with the number of names beside each: Firefly, Epidemic Sound, the four classical majors, genre playlists.

- Dots, not bars. The current 0% bars are invisible, so five of seven rows look empty.
- Title: "Names from the company in the press reports are flagged; real artists mostly are not."
- Put the table from 3.2 next to it in the text.

Cut panel A (see 3.2). Cut panel C: the new Fig. 1 shows placement by playlist kind, and the vertical scatter in C is random jitter that readers try to interpret. Mention *Calming Classical* and *Women of Ambient* in the text as the exceptions.

### 5.3 Figure 3: who registered the recordings

**Panel A, the network.** Make it the full figure width.

- Shrink or crop the outer ring of tiny houses. It takes most of the area.
- Say what a dot, a disc and a dark centre are in a small key inside the panel, not only in the caption.
- Replace "names label themselves" with "no company in the credits".
- Give each of the eight houses a short plain label ("House 1 … House 8") with the code in smaller type. Codes like SE5W3 mean nothing at first glance.
- Add one caption sentence about the blue dots (see 3.4).
- Title: "Hundreds of artist names, eight houses."

**Replace panels B and C with one plain panel.** Two horizontal bars: functional playlists and genre playlists. Each bar is split into the eight largest houses, all other houses, and no code. The reader sees at once that eight segments fill most of the functional bar and almost none of the genre bar. No log axis, no cumulative curve, no label needed.

### 5.4 Figure 4: where the recommender puts them

**Panel A, the network.** Keep the orange and blue colouring for consistency.

- Title: "Each cluster of similar artists is a playlist."
- Remove the unexplained "+" after playlist names. Write "Sleep and 3 similar playlists" or just the main playlist.
- Label fewer clusters, in larger type.
- Fix the count (E1, E2).

**Panel B, the key result.** It is the smallest panel now. Make it the largest.

- Plain row labels: "real artist's page lists a ghost", "ghost's page lists a real artist", "both from the same house".
- Show the null's range as a grey bar behind each point, in place of z-scores.
- Set the "same house" row apart with a gap. It is a different kind of quantity.
- **Add a three-step explainer of the two nulls** above the panel, as small drawings with one line each:
  1. Observed: the real lists.
  2. Shuffle 1: every page keeps its number of links, but who is listed is random.
  3. Shuffle 2: the same, but a link may only be swapped for another link into the same playlist.

**Panel C.** Keep, three bars with their shuffled baseline. Title as now. Remove the modularity note from the panel.

### 5.5 Figure 5 (small): what a listener sees

**Panel A.** One dot per kind of content on an axis from 0.5 to 1, with the shuffled range as a grey band. The current bars start at 0.5 with a hatched bar laid on top, and the sound bar runs backwards.

- Label the two ends of the axis in words: "no better than guessing" and "perfect".
- Title: "Only the bio gives a ghost away, and most ghosts have none."
- State that only one in four ghost-like names has a bio, inside the panel.

**Panel B.** Keep the two word lists. Draw both sides to the same visual weight, or drop the bars and show two columns of words. The real-bio bars are shorter only because there are fewer words to count, which looks like a weaker effect.

## 6. Text and communication

### 6.1 Structure

- **Results start on page 1 because nothing separates them from the introduction.** Add a short section before Results, "How we flag a ghost-like name", with the rule from 3.1 and the definition of a house. Five or six sentences. With the simpler method this fits easily.
- **Open each subsection with its conclusion.** The recommender section should start: "The recommender does not push listeners from real artists to ghosts. Its map of similar artists mostly reproduces Spotify's own playlists."
- **One finding per paragraph.** The ownership paragraph currently mixes self-labelling, concentration, Sweden, Overtone and multi-house names.
- **Follow the three steps in section 4** in the abstract, the Results and the Discussion, in the same order.

### 6.2 Things a first-time reader trips on

- **"House"** is defined once and used about 40 times. Repeat a short reminder at the start of the ownership section: "a house is whoever registered the recording".
- **The abstract carries too many numbers.** One number per claim. No ρ, no z, one ratio.
- **Keep the "What this captures and misses" paragraphs.** They work. Hold each to two sentences.
- **Say "flagged" or "ghost-like" consistently**, and "real" only for names that are not flagged. Avoid "ghost" as a bare noun for a person.

### 6.3 Explanations to add

- **The two null models**, in plain words, as in 5.4. Then the technical name once in Methods.
- **Covers and audio clips.** One sentence each in the data section: how many, where from, what they are used for.
- **How the 50 playlists were chosen.** By hand, to cover the mood categories, plus five genre playlists as a sanity check. Say that the 75% figure is for these playlists and would change with another selection.
- **Epidemic Sound**, as in E5.

### 6.4 The text-analysis section

Keep it short and honest. Three points to change:

- **Bio length alone nearly matches the classifier** (0.80 against 0.87). Say that ghost bios are mainly short and generic, which is simpler and true.
- **The real bios are largely classical** (*orchestra*, *conductor*, *symphony*). Say so in one clause, so the word list is not over-read.
- **Covers are not uninformative** (0.71). The abstract's "names, titles and sound do not give a ghost away; bios do" should mention covers or leave the list out.

One cheap addition that ties the text part to the network part, using a number already computed: a cover's nearest neighbour comes from the same house 15% of the time, against 6% by chance. Lead the paragraph with it.

### 6.5 Limits to state in one sentence each

- **Market.** These are Spotify's global editorial playlists seen from Denmark. Whether track lists differ by market is untested. The Swedish codes match the press reports about Swedish companies.
- **House size is counted inside the sample**: "232 names" means 232 names on these 50 playlists.
- **Coverage.** 28% of functional slots have no code. One supplier, Score a Score, is invisible to the house analysis for this reason.
- **The rule's false positives**, as in 3.1.

## 7. Order of work

1. Switch to the rule label (3.1), remove the mixture, scores, correlations and hand-label code (2, 3). Rerun the pipeline.
2. Fix E1–E10.
3. Build the new Fig. 1. Rebuild Figs. 2 and 5 as small figures. Revise Figs. 3 and 4.
4. Restructure the text (6.1), add the missing explanations (6.3).
5. Rewrite the abstract and significance statement last, from the final numbers.
