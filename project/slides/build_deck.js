// Build the 3-slide talk: slides/ghost_artists_talk.pptx (10 minutes, ~3 min per slide).
// Each slide tells one step of the story left to right (or top to bottom), joined by thin arrows.
// Figures come from `python -m src.figures.slides`; numbers match paper/numbers.tex.
//
//   npm install pptxgenjs        (once, anywhere on NODE_PATH)
//   node slides/build_deck.js    (from project/)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const FIG = path.join(__dirname, "figures");
const OUT = path.join(__dirname, "ghost_artists_talk.pptx");

// The paper's colours: orange = ghost-like, blue = real artist.
const INK = "1F2328", NAVY = "1C2434", MUTED = "6B6A66", ARROW = "9A9994";
const CARD = "F4F5F7", ANSWER = "EAF0F8", GHOST = "EB6834", REAL = "2A78D6";
const W = 13.333, M = 0.5;
const HEAD = "Cambria", BODY = "Arial";

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Many faces, few hands";
pres.subject = "Ghost artists on Spotify: 02805 Social Graphs and Interactions";
pres.author = "August Holm Jørgensen, Bertram Nyvold Larsen, Saxe Brøten";
pres.theme = { headFontFace: HEAD, bodyFontFace: BODY };

pres.defineSlideMaster({
  title: "STORY",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "kicker", type: "body", x: M, y: 0.3, w: 9, h: 0.3, fontSize: 12, bold: true,
        color: GHOST, charSpacing: 1, margin: 0, valign: "top", fontFace: BODY }, text: "" } },
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.6, w: W - 2 * M, h: 0.65, fontSize: 28,
        bold: true, color: NAVY, fontFace: HEAD, margin: 0, valign: "top", align: "left" }, text: "" } },
    { text: { text: "Many faces, few hands  ·  Jørgensen, Larsen & Brøten  ·  02805 Social Graphs and Interactions, DTU",
        options: { x: M, y: 7.05, w: 10, h: 0.25, fontSize: 10, color: MUTED, margin: 0, fontFace: BODY } } },
  ],
  slideNumber: { x: W - M - 1.0, y: 7.05, w: 1.0, h: 0.25, fontSize: 10, color: MUTED, align: "right", margin: 0 },
});

function newSlide(section, kicker, title) {
  pres.addSection({ title: section });
  const s = pres.addSlide({ masterName: "STORY", sectionTitle: section });
  s.addText(kicker.toUpperCase(), { placeholder: "kicker" });
  s.addText(title, { placeholder: "title" });
  return s;
}
function text(s, t, o) {
  s.addText(t, { isTextBox: true, margin: 0, color: INK, fontSize: 15, valign: "top", fontFace: BODY, ...o });
}
function card(s, x, y, w, h, fill = CARD, name = "card") {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { type: "none" },
    objectName: name });
}
function arrow(s, x1, y1, x2, y2) {
  s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1), h: Math.abs(y2 - y1),
    line: { color: ARROW, width: 1.5, endArrowType: "triangle" }, objectName: "arrow" });
}
// Numbered step: a small navy circle with the number, then a heading beside it.
function step(s, n, label, x, y, w) {
  s.addText(String(n), { shape: pres.shapes.OVAL, x, y, w: 0.42, h: 0.42, fill: { color: NAVY }, color: "FFFFFF",
    fontSize: 15, bold: true, align: "center", valign: "middle", margin: 0, fontFace: BODY, objectName: `step ${n}` });
  text(s, label, { x: x + 0.55, y: y + 0.02, w: w - 0.55, h: 0.4, fontSize: 19, bold: true, color: NAVY, fontFace: HEAD,
    valign: "middle" });
}
function fig(s, name, x, y, w, alt) {
  const file = path.join(FIG, name), p = pngSize(file), h = (w * p.h) / p.w;
  s.addImage({ path: file, x, y, w, h, altText: alt, objectName: name.replace(".png", "") });
  return h;
}

// ================================================================ slide 1
{
  const s = newSlide("Data and the flag", "1  ·  Data and the ghost-like flag",
    "Three in four mood-playlist slots go to names with no public identity");
  const y0 = 1.6, h = 5.2, cw = 3.75, gap = 0.54;
  const xs = [M, M + cw + gap, M + 2 * (cw + gap)];
  xs.forEach((x, i) => card(s, x, y0, cw, h, CARD, `step card ${i + 1}`));
  arrow(s, xs[0] + cw + 0.08, y0 + 0.51, xs[1] - 0.08, y0 + 0.51);
  arrow(s, xs[1] + cw + 0.08, y0 + 0.51, xs[2] - 0.08, y0 + 0.51);
  const pad = 0.25;

  // 1 collect
  let x = xs[0] + pad;
  const w = cw - 2 * pad;
  step(s, 1, "Collect the playlists", x, y0 + 0.3, w);
  [["50", "editorial playlists", "45 mood (sleep, focus, calm, jazz) + 5 genre playlists as a control"],
   ["9,004", "playlist slots", "each with its track, credits and recording code"],
   ["5,024", "artist names", "each with its public profile and “Fans also like” list"]].forEach(([n, lab, sub], i) => {
    const y = y0 + 0.95 + i * 1.2;
    text(s, [{ text: n + "  ", options: { fontSize: 30, bold: true, color: NAVY, fontFace: HEAD } },
             { text: lab, options: { fontSize: 16, bold: true } }], { x, y, w, h: 0.55, valign: "bottom" });
    text(s, sub, { x, y: y + 0.58, w, h: 0.55, fontSize: 13, color: MUTED });
  });
  text(s, "One snapshot, 4 November 2026", { x, y: y0 + h - 0.45, w, h: 0.3, fontSize: 13, color: MUTED, italic: true });

  // 2 flag
  x = xs[1] + pad;
  step(s, 2, "Flag ghost-like names", x, y0 + 0.3, w);
  text(s, "Ghost-like if at least 4 of these 5 are missing:", { x, y: y0 + 0.9, w, h: 0.5, fontSize: 14 });
  ["Verified badge", "Bio", "External link", "Gallery photo", "“This Is” playlist"].forEach((m, i) => {
    const y = y0 + 1.42 + i * 0.36;
    s.addShape(pres.shapes.OVAL, { x, y: y + 0.07, w: 0.16, h: 0.16, fill: { color: "FFFFFF" },
      line: { color: MUTED, width: 1 }, objectName: `marker ${i + 1}` });
    text(s, m, { x: x + 0.28, y, w: w - 0.28, h: 0.3, fontSize: 14, valign: "middle" });
  });
  text(s, "Tested on groups it never saw:", { x, y: y0 + 3.4, w, h: 0.3, fontSize: 14, bold: true });
  fig(s, "s1_check.png", xs[1] + 0.12, y0 + 3.75, cw - 0.24, "Share flagged: 90% of a leaked ghost catalogue, 3% of known real artists");

  // 3 count
  x = xs[2] + pad;
  step(s, 3, "Count the slots", x, y0 + 0.3, w);
  text(s, "75%", { x, y: y0 + 0.95, w, h: 1.0, fontSize: 66, bold: true, color: GHOST, fontFace: HEAD });
  text(s, "of mood-playlist slots go to ghost-like names", { x, y: y0 + 2.0, w, h: 0.65, fontSize: 16, bold: true });
  fig(s, "s1_slots.png", xs[2] + 0.12, y0 + 2.95, cw - 0.24, "Share of slots: mood playlists 75% ghost-like, genre playlists 1%");
  text(s, [{ text: "■ ", options: { color: GHOST } }, { text: "ghost-like    " },
           { text: "■ ", options: { color: REAL } }, { text: "real artist" }],
       { x, y: y0 + 4.6, w, h: 0.3, fontSize: 13, color: MUTED });

  s.addNotes(
`~3 min. Story: collect → flag → count.

Why we care: playlists like Sleep or Deep Focus are among the most followed on Spotify. Journalists report that much of this music comes from a few production companies under invented artist names, but that reporting rests on leaked lists. We ask what public data alone can show.

1 Collect. One snapshot through the web player on 4 November 2026 (the official API no longer serves editorial playlists). 45 mood playlists plus 5 genre playlists with known real artists as a control. 9,004 slots, 5,024 artist names. For every name its public profile and its "Fans also like" list; for every track its credits, cover, preview, and, via Deezer, its recording code (ISRC).

2 Flag. A name is ghost-like if at least 4 of 5 public markers are missing: verified badge, bio, external link, gallery photo, a "This Is" playlist. It is one sentence anyone can check, and it uses nothing we test later: not who made the music, not listening, not recommendations.
Does it find the right names? It flags 90% of the names released by Firefly Entertainment, the company in the Dagens Nyheter leak, and only 3% of the artists on the genre playlists. (Its main error: 18% of classical-label names, session ensembles without profiles. That error works against our results.)

3 Count. Ghost-like names hold 75% of mood-playlist slots, against 1% on the genre playlists. Next question: who is behind them?`);
}

// ================================================================ slide 2
{
  const s = newSlide("Network analysis", "2  ·  Network analysis: who makes the music?",
    "Credits show hundreds of artists; recording codes show eight houses");
  const x = M, w = 3.85, ch = 1.45, g = 0.45, y0 = 1.6, pad = 0.22;
  const ys = [y0, y0 + ch + g, y0 + 2 * (ch + g)];
  ys.forEach((y, i) => card(s, x, y, w, ch, i === 2 ? ANSWER : CARD, `flow card ${i + 1}`));
  arrow(s, x + w / 2, ys[0] + ch + 0.06, x + w / 2, ys[1] - 0.06);
  arrow(s, x + w / 2, ys[1] + ch + 0.06, x + w / 2, ys[2] - 0.06);

  const tx = x + pad, tw = w - 2 * pad;
  text(s, "The credits say: self-released", { x: tx, y: ys[0] + 0.15, w: tw, h: 0.35, fontSize: 16, bold: true, color: NAVY });
  text(s, [{ text: "Artist  ", options: { color: MUTED } }, { text: "<name>", options: { italic: true, breakLine: true } },
           { text: "Label   ", options: { color: MUTED } }, { text: "<the same name>", options: { italic: true } }],
       { x: tx, y: ys[0] + 0.52, w: tw, h: 0.5, fontSize: 13 });
  text(s, [{ text: "55%", options: { bold: true, color: GHOST } }, { text: " of ghost-like names (real: 13%)" }],
       { x: tx, y: ys[0] + 1.05, w: tw, h: 0.3, fontSize: 13 });

  text(s, "The recording code says who", { x: tx, y: ys[1] + 0.15, w: tw, h: 0.35, fontSize: 16, bold: true, color: NAVY });
  text(s, [{ text: "ISRC  ", options: { color: MUTED } }, { text: "SE69Z", options: { bold: true, color: GHOST } },
           { text: " 26 00123" }], { x: tx, y: ys[1] + 0.52, w: tw, h: 0.35, fontSize: 16, fontFace: "Courier New" });
  text(s, "First five characters = who registered the recording: the house",
       { x: tx, y: ys[1] + 0.9, w: tw, h: 0.5, fontSize: 13 });

  text(s, "Link each name to its house", { x: tx, y: ys[2] + 0.15, w: tw, h: 0.35, fontSize: 16, bold: true, color: NAVY });
  text(s, "A bipartite network: names on one side, the houses that registered their recordings on the other",
       { x: tx, y: ys[2] + 0.55, w: tw, h: 0.8, fontSize: 13 });

  // the network, fed by the third card
  const nx = 4.95, nw = W - M - nx;
  arrow(s, x + w + 0.08, ys[2] + ch / 2, nx - 0.02, ys[2] + ch / 2);
  text(s, [{ text: "● ", options: { color: GHOST } }, { text: "ghost-like name    " },
           { text: "● ", options: { color: REAL } }, { text: "real artist    " },
           { text: "● ", options: { color: "1F1F1E" } }, { text: "house that registered them" }],
       { x: nx, y: y0, w: nw, h: 0.3, fontSize: 13, color: MUTED, align: "center" });
  const fw2 = nw * 0.86;
  fig(s, "s2_houses.png", nx + (nw - fw2) / 2, y0 + 0.35, fw2, "Ownership network: artist names around the house that registered their recordings");
  text(s, [{ text: "81%", options: { bold: true, color: GHOST, fontSize: 26, fontFace: HEAD } },
           { text: " of ghost-like names come from just 8 houses (7 Swedish)", options: { bold: true } }],
       { x: nx, y: 6.3, w: nw, h: 0.55, fontSize: 17, valign: "middle", align: "center" });

  s.addNotes(
`~3 min. Story: credits → recording code → network.

The credits. On a ghost-like name's tracks, 55% list the artist's own name as the label, against 13% of the real artists on the same playlists. To a listener, each looks like an independent, self-released musician.

The recording code. Every recording has an ISRC; its first five characters identify who registered it (country + registrant). We call that the house. Example: SE69Z is the largest house in our data.

The network. Link every name to the house that registered most of its tracks: a bipartite network. Each dark dot is a house, the dots around it are its names; orange = ghost-like, blue = real.

What it shows. Real artists sit in many small houses. Ghost-like names pile up in a few huge ones: of the 2,311 ghost-like names with a code, 81% come from just eight houses with 194 to 377 names each. Seven of the eight use Swedish-issued codes, which fits the press reports. One is Firefly Entertainment, two belong to Overtone Studios (owned by Epidemic Sound), and for four no company appears anywhere in the credits.

This does not depend on our flag: counting every slot by its own track's code, the eight largest houses registered 72% of coded mood-playlist slots, against 22% on genre playlists.

Caveat: a house is whoever registered the recording, a label or distributor, not necessarily the composer; 28% of mood slots have no code because the Deezer match failed.`);
}

// ================================================================ slide 3
{
  const s = newSlide("NLP and recommender", "3  ·  NLP and the recommender",
    "A listener can hardly tell, and the recommender mirrors the playlists");
  const qx = M, qw = 3.1, fx = 4.0, fw = 4.9, ax = 9.3, aw = W - M - ax, rh = 2.3, pad = 0.22;
  const rows = [
    { y: 1.5, n: 1, q: "Could a listener tell?",
      how: "For each thing a listener sees, we train a classifier (TF-IDF, CLIP, audio features) and test it on names it has not seen.",
      fig: "s3_auc.png", alt: "How well each kind of content predicts the flag (ROC AUC)",
      a: [{ text: "Name, titles and sound:", options: { bold: true } }, { text: " close to guessing.", options: { breakLine: true } },
          { text: "Only the bio gives it away, and just ", options: {} }, { text: "22%", options: { bold: true, color: GHOST } },
          { text: " of ghost-like names have one.", options: {} }] },
    { y: 3.95, n: 2, q: "Does the recommender push listeners to ghosts?",
      how: "We compare the “Fans also like” links between real and ghost-like names with random links.",
      fig: "s3_links.png", alt: "Fans-also-like links between real and ghost-like names, divided by chance",
      a: [{ text: "No.", options: { bold: true } }, { text: " Both directions are rarer than chance.", options: { breakLine: true } },
          { text: "Keep the playlists fixed and most of the gap disappears: ", options: {} },
          { text: "the links follow the playlists.", options: { bold: true } }] },
  ];
  rows.forEach((r) => {
    card(s, qx, r.y, qw, rh, CARD, `question ${r.n}`);
    card(s, ax, r.y, aw, rh, ANSWER, `answer ${r.n}`);
    text(s, r.q, { x: qx + pad, y: r.y + 0.18, w: qw - 2 * pad, h: 0.75, fontSize: 17, bold: true, color: NAVY, fontFace: HEAD });
    text(s, r.how, { x: qx + pad, y: r.y + 0.98, w: qw - 2 * pad, h: 1.2, fontSize: 13, color: MUTED });
    fig(s, r.fig, fx, r.y + (rh - 2.25) / 2, fw, r.alt);
    arrow(s, qx + qw + 0.06, r.y + rh / 2, fx - 0.04, r.y + rh / 2);
    arrow(s, fx + fw + 0.04, r.y + rh / 2, ax - 0.06, r.y + rh / 2);
    text(s, r.a, { x: ax + pad, y: r.y + 0.2, w: aw - 2 * pad, h: rh - 0.4, fontSize: 15, valign: "middle", paraSpaceAfter: 6 });
  });
  text(s, [{ text: "So: ", options: { bold: true, color: GHOST } },
           { text: "the playlists, not the recommender, put ghost artists in front of listeners, and recording codes show who is behind them.",
             options: { bold: true, color: NAVY } }],
       { x: M, y: 6.4, w: W - 2 * M, h: 0.5, fontSize: 15, valign: "middle" });

  s.addNotes(
`~3 min. Two questions, each: question → result → answer.

Could a listener tell? For each thing a listener sees we trained a simple classifier to predict the flag, using mood-playlist names only, and scored it on names it had not seen (ROC AUC: 0.5 is guessing, 1 is perfect).
- Sound 0.52: nothing. On the same playlist, ghost tracks sound like real ones (tempo, loudness, brightness, timbre from 624 previews).
- Artist names 0.61, track titles 0.63: close to guessing. 58% of ghost-like names are a "Firstname Lastname", exactly as many as real artists: they are built to look like people.
- Covers 0.75: covers carry a house style.
- Bios 0.89, the NLP result: ghost bios describe a mood (quiet, calm, stillness), real bios a career (album, released, debut); named entities appear in 88% of real bios but 43% of ghost bios. But only 22% of ghost-like names have a bio at all (91% of real).

Does the recommender push listeners to ghosts? Every artist page lists up to 20 "Fans also like" artists. We compared the links with random links that keep each page's number of links (grey), and with random links that may only point into the same playlist (dark).
We expected a funnel from real artists to ghosts. There is none: real → ghost is 0.48× chance and ghost → real 0.56×, both reduced. Inside the same playlist the gap almost closes (0.82× and 0.86×): listeners who found a name on Sleep also listen to other Sleep names. Louvain clusters match the home playlist (NMI 0.70) far more than the house (0.32). Same-house names are linked a bit more than playlists alone explain (1.37×).

Conclusion: ghost-like names get their listeners from the playlists (68% were found mainly through one of our editorial playlists), not from the recommender. And recording codes are a cheap, public audit tool for who is behind a playlist.`);
}

// pptxgenjs writes folder entries into the zip, which some cloud services fail to open; drop them.
pres.writeFile({ fileName: OUT }).then((f) => {
  require("child_process").execFileSync("zip", ["-dq", f, "*/"]);
  console.log("wrote", f);
});
