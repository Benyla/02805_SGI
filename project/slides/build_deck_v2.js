// Version 2 of the 3-slide talk, following not-ai-decks-guide: slides/ghost_artists_talk_v2.pptx
// Takeaway titles, one dominant chart per slide, one typeface (Arial, regular and bold),
// three sizes (28 title, 16 body, 12 caption), white background, one accent (orange = ghost-like).
// Figures come from `python -m src.figures.slides_v2`; numbers match paper/numbers.tex.
//
//   npm install pptxgenjs        (once, anywhere on NODE_PATH)
//   node slides/build_deck_v2.js (from project/)
const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");

const FIG = path.join(__dirname, "figures_v2");
const OUT = path.join(__dirname, "ghost_artists_talk_v2.pptx");

const INK = "1F2328", GREY = "6B6A66", RULE = "D9D8D3";
const W = 13.333, M = 0.6, FONT = "Arial";
const TITLE = 28, BODY = 16, SMALL = 12;

function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.title = "Many faces, few hands";
pres.author = "August Holm Jørgensen, Bertram Nyvold Larsen, Saxe Brøten";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

pres.defineSlideMaster({
  title: "PLAIN",
  background: { color: "FFFFFF" },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: M, y: 0.45, w: W - 2 * M, h: 1.0, fontSize: TITLE,
        bold: true, color: INK, fontFace: FONT, margin: 0, valign: "top", align: "left" }, text: "" } },
    { text: { text: "Jørgensen, Larsen and Brøten, 02805 Social Graphs and Interactions, DTU",
        options: { x: M, y: 7.0, w: 9, h: 0.25, fontSize: SMALL, color: GREY, margin: 0, fontFace: FONT } } },
  ],
  slideNumber: { x: W - M - 1.0, y: 7.0, w: 1.0, h: 0.25, fontSize: SMALL, color: GREY, align: "right", margin: 0,
    fontFace: FONT },
});

function slide(title) {
  const s = pres.addSlide({ masterName: "PLAIN" });
  s.addText(title, { placeholder: "title" });
  return s;
}
function text(s, t, o) {
  s.addText(t, { isTextBox: true, margin: 0, color: INK, fontSize: BODY, valign: "top", fontFace: FONT, align: "left",
    paraSpaceAfter: 4, ...o });
}
// a short paragraph with a bold lead sentence, for the side columns
function para(lead, body) {
  return [{ text: lead + " ", options: { bold: true } }, { text: body, options: { breakLine: true } }];
}
function fig(s, name, x, y, w, alt) {
  const file = path.join(FIG, name), p = pngSize(file), h = (w * p.h) / p.w;
  s.addImage({ path: file, x, y, w, h, altText: alt });
  return h;
}

// ================================================================ slide 1
{
  const s = slide("Ghost-like names hold 75% of the slots on Spotify's mood playlists");
  const fw = 7.6, fy = 1.95;
  const fh = fig(s, "s1_slots.png", M, fy, fw, "Share of playlist slots: mood playlists 75% ghost-like, genre playlists 1%");
  text(s, "Share of playlist slots, Spotify web player, 4 November 2026",
       { x: M + 1.9, y: fy + fh + 0.1, w: fw - 1.9, h: 0.3, fontSize: SMALL, color: GREY });

  const cx = 8.9, cw = W - M - cx;
  text(s, [
    ...para("We saved 50 playlists.", "45 mood playlists like Sleep and Deep Focus, plus 5 genre playlists of known artists. That is 8,093 tracks by 5,024 artist names."),
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    ...para("We flag a name as ghost-like", "when at least 4 of 5 public profile signs are missing. The signs are a verified badge, a bio, an external link, gallery photos and a \"This Is\" playlist."),
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    ...para("The flag works on groups we already know.", "It catches 90% of the names released by Firefly Entertainment and 3% of the artists on the genre playlists."),
  ], { x: cx, y: 1.95, w: cw, h: 4.6 });

  s.addNotes(
`About 1 minute (about 140 words).

Playlists like Sleep and Deep Focus are some of the most followed on Spotify, and one slot can bring millions of streams. Journalists have reported that companies fill them with cheap music under made-up artist names. But that rests on leaks. We asked if public data alone can show it.

On 4 November we saved 45 mood playlists, plus 5 genre playlists of known artists as a control. That gave us 5,024 artist names.

We call a name ghost-like when its profile is almost empty. At least 4 of 5 signs are missing, like a verified badge or a bio. We checked the rule on groups we know. It flags 90% of Firefly Entertainment's names and only 3% of real artists.

The result is in the chart. Ghost-like names hold 75% of the mood-playlist slots, and 1% on the genre playlists. So who is behind them?`);
}

// ================================================================ slide 2
{
  const s = slide("81% of the ghost-like names we could trace come from just eight record houses");
  const lx = M, lw = 3.75;
  text(s, [
    ...para("The credits look self-released.", "55% of ghost-like names list their own name as the label. Among real artists it is 13%."),
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    ...para("The recording code tells who.", "Every recording has an ISRC code. Its first five characters name who registered it. We call that the house."),
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    ...para("We got the codes from Deezer.", "We matched each track by artist, title and length. 72% of mood-playlist tracks matched."),
    { text: " ", options: { breakLine: true, fontSize: 8 } },
    ...para("Seven of the eight houses are Swedish.", "One is Firefly Entertainment and two belong to Overtone Studios."),
  ], { x: lx, y: 1.75, w: lw, h: 4.9 });

  const fx = 4.85, fw = W - M - fx;
  const fh = fig(s, "s2_houses.png", fx, 1.6, fw, "Network of artist names around the house that registered their recordings");
  text(s, "Each black dot is a house. The dots around it are the names it registered, orange for ghost-like and grey for real artists. Houses with at least 3 names.",
       { x: fx, y: 1.6 + fh + 0.05, w: fw, h: 0.45, fontSize: SMALL, color: GREY });

  s.addNotes(
`About 1 minute (about 140 words).

So who is behind these names? The credits don't tell. 55% of ghost-like names list themselves as their own label, so each looks like an independent musician.

But every recording has an ISRC code, and its first five characters say who registered it. We call that the house. Spotify hides the code, so we matched each track on Deezer and got it there.

Then we built a bipartite network. Each name links to the house that registered its tracks. Real artists, in grey, spread over many small houses. Ghost-like names pile up in a few huge ones. 81% of the ghost-like names we could trace come from just eight houses, with 194 to 377 names each. Seven are Swedish. One is Firefly Entertainment and two belong to Overtone Studios.

One caveat. A house is whoever registered the recording, not always the composer.`);
}

// ================================================================ slide 3
{
  const s = slide("The recommender just follows the playlists, and only a bio gives a ghost-like name away");
  const lx = M, lw = 6.3, rx = 7.35, rw = W - M - rx, y0 = 1.75;
  text(s, [{ text: "The recommender follows the playlists.", options: { bold: true } },
           { text: " \"Fans also like\" links between the two groups are rarer than chance. Inside the same playlist the gap almost closes." }],
       { x: lx, y: y0, w: lw - 0.3, h: 1.1 });
  fig(s, "s3_links.png", lx, y0 + 1.05, lw * 0.92, "Fans also like links between real and ghost-like names, divided by chance");

  s.addShape(pres.shapes.LINE, { x: rx - 0.4, y: y0, w: 0, h: 4.2, line: { color: RULE, width: 0.75 } });
  text(s, [{ text: "NLP spots ghost-like names only from their bio.", options: { bold: true } },
           { text: " We turned each text into TF-IDF word weights and trained a classifier on them. Only 22% of ghost-like names have a bio." }],
       { x: rx, y: y0, w: rw, h: 1.1 });
  const ah = fig(s, "s3_auc.png", rx, y0 + 1.05, rw, "How well each thing a listener sees predicts the flag, ROC AUC");
  text(s, "Ghost bios use words like quiet, calm, stillness and sleep. Real bios use album, released, debut and concert.",
       { x: rx, y: y0 + 1.05 + ah + 0.02, w: rw, h: 0.45, fontSize: SMALL, color: GREY });

  text(s, "Recording codes are public, so anyone can check who is behind a playlist.",
       { x: M, y: 6.35, w: W - 2 * M, h: 0.4, bold: true });

  s.addNotes(
`About 1 minute (about 140 words).

Does the recommender push listeners to ghosts? We compared the "Fans also like" links with random links. Real artists link to ghost-like names at only half the chance rate. When the random links must stay inside the same playlist, the gap almost closes. So the recommender just follows the playlists.

Could a listener tell? Here we used NLP. We turned each bio, artist name and track title into TF-IDF word weights and trained a classifier to spot ghost-like names. Names and titles are close to guessing. The bio works well, with an AUC of 0.89. Ghost bios talk about a mood, like quiet and calm. Real bios talk about a career, like album and debut. But only 22% of ghost-like names have a bio.

So the playlists, not the recommender, bring the listeners. And recording codes are public, so anyone can check who is behind a playlist.`);
}

// pptxgenjs writes folder entries into the zip, which some cloud services fail to open; drop them.
pres.writeFile({ fileName: OUT }).then((f) => {
  require("child_process").execFileSync("zip", ["-dq", f, "*/"]);
  console.log("wrote", f);
});
