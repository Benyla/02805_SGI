/* The Shuffle Machine — a degree-preserving permutation test you can run yourself.
 *
 * Data comes from marvel_gc.js (window.MARVEL_GC): the 277-hero giant component
 * of the Marvel Wikipedia network, collapsed to undirected.
 *
 * Everything here runs on the real edge list in your browser. No server, no
 * pre-computed answers — press the button and the null distribution is built
 * in front of you, one degree-preserving shuffle at a time.
 */
(function () {
  "use strict";

  var root = document.getElementById("shuffle-machine");
  if (!root || !window.MARVEL_GC) return;

  var DATA = window.MARVEL_GC;
  var N = DATA.names.length;
  var M = DATA.edges.length;
  var SWAPS_PER_RUN = 10 * M; // the usual rule of thumb: ten swaps per link

  // ---------------------------------------------------------------- graph ---

  function buildAdj(edgeList) {
    var adj = new Array(N);
    for (var i = 0; i < N; i++) adj[i] = new Set();
    for (var e = 0; e < edgeList.length; e++) {
      adj[edgeList[e][0]].add(edgeList[e][1]);
      adj[edgeList[e][1]].add(edgeList[e][0]);
    }
    return adj;
  }

  var baseEdges = DATA.edges;
  var baseAdj = buildAdj(baseEdges);
  var baseDeg = new Int32Array(N);
  for (var i = 0; i < N; i++) baseDeg[i] = baseAdj[i].size;

  // Degree never changes, so the hub set never changes either. That makes
  // "are the hubs wired to each other?" a question the shuffle can answer.
  var HUB_COUNT = 20;
  var hubRank = Array.from({ length: N }, function (_, i) { return i; })
    .sort(function (a, b) { return baseDeg[b] - baseDeg[a]; });
  var isHub = new Uint8Array(N);
  for (var h = 0; h < HUB_COUNT; h++) isHub[hubRank[h]] = 1;

  // ------------------------------------------------------------ statistics ---

  /* One pass over the edges gives triangles, per-node neighbour links, and
     therefore transitivity, average clustering and Spider-Man's triangles. */
  function measure(edgeList, adj) {
    var neighbourLinks = new Float64Array(N);
    var closedPaths = 0;
    var e, u, v, s, l, w, it;

    for (e = 0; e < edgeList.length; e++) {
      u = edgeList[e][0]; v = edgeList[e][1];
      if (adj[u].size <= adj[v].size) { s = u; l = v; } else { s = v; l = u; }
      it = adj[s].values();
      for (var r = it.next(); !r.done; r = it.next()) {
        w = r.value;
        if (w !== l && adj[l].has(w)) {
          closedPaths++;          // this edge sits in a triangle with w
          neighbourLinks[w]++;    // ... which is an edge among w's neighbours
        }
      }
    }

    var triangles = closedPaths / 3;
    var triads = 0, clusterSum = 0, clusterCount = 0, k, pairs;
    for (var n = 0; n < N; n++) {
      k = adj[n].size;
      if (k < 2) continue;
      pairs = (k * (k - 1)) / 2;
      triads += pairs;
      clusterSum += neighbourLinks[n] / pairs;
      clusterCount++;
    }

    // Assortativity: Pearson r over the degrees at the two ends of each edge.
    var sJK = 0, sSum = 0, sSq = 0;
    for (e = 0; e < edgeList.length; e++) {
      var j = adj[edgeList[e][0]].size, kk = adj[edgeList[e][1]].size;
      sJK += j * kk;
      sSum += (j + kk) / 2;
      sSq += (j * j + kk * kk) / 2;
    }
    var mm = edgeList.length;
    var meanSum = sSum / mm;
    var assort = (sJK / mm - meanSum * meanSum) / (sSq / mm - meanSum * meanSum);

    var hubHub = 0;
    for (e = 0; e < edgeList.length; e++) {
      if (isHub[edgeList[e][0]] && isHub[edgeList[e][1]]) hubHub++;
    }

    return {
      transitivity: (3 * triangles) / triads,
      clustering: clusterSum / clusterCount,
      triangles: triangles,
      assortativity: assort,
      hubHub: hubHub,
      spidey: neighbourLinks[0],
      hubShare: adj[0].size / mm
    };
  }

  var STATS = [
    { key: "transitivity", label: "Transitivity (global clustering)", fmt: 3,
      blurb: "Of all the paths of length two, what share are closed into a triangle?" },
    { key: "clustering", label: "Average clustering C", fmt: 3,
      blurb: "The week-1 headline number, averaged over heroes." },
    { key: "triangles", label: "Number of triangles", fmt: 0,
      blurb: "Three heroes who all link to each other." },
    { key: "assortativity", label: "Degree assortativity", fmt: 3,
      blurb: "Do hubs link to hubs (positive) or to nobodies (negative)?" },
    { key: "hubHub", label: "Links among the top 20 hubs", fmt: 0,
      blurb: "The shuffle keeps everyone's degree, so the top 20 stay the top 20. Do they still hang out together?" },
    { key: "spidey", label: "Triangles through Spider-Man", fmt: 0,
      blurb: "How embedded is the biggest hub — is he a bridge or a neighbourhood?" },
    { key: "hubShare", label: "Spider-Man's share of all links", fmt: 4,
      blurb: "A control. The shuffle cannot move this by even one link — watch it not move." }
  ];

  var REAL = measure(baseEdges, baseAdj);

  // ---------------------------------------------------------------- swaps ---

  function shuffledCopy(nSwaps) {
    var edgeList = new Array(M);
    for (var e = 0; e < M; e++) edgeList[e] = [baseEdges[e][0], baseEdges[e][1]];
    var adj = buildAdj(edgeList);

    var done = 0, attempts = 0, maxAttempts = nSwaps * 25;
    while (done < nSwaps && attempts < maxAttempts) {
      attempts++;
      var i = (Math.random() * M) | 0, j = (Math.random() * M) | 0;
      if (i === j) continue;
      var u = edgeList[i][0], v = edgeList[i][1];
      var x, y;
      if (Math.random() < 0.5) { x = edgeList[j][0]; y = edgeList[j][1]; }
      else { x = edgeList[j][1]; y = edgeList[j][0]; }
      // no self-loops, no doubling an existing link
      if (u === x || u === y || v === x || v === y) continue;
      if (adj[u].has(x) || adj[v].has(y)) continue;
      adj[u].delete(v); adj[v].delete(u);
      adj[x].delete(y); adj[y].delete(x);
      adj[u].add(x); adj[x].add(u);
      adj[v].add(y); adj[y].add(v);
      edgeList[i] = [u, x];
      edgeList[j] = [v, y];
      done++;
    }
    return { edges: edgeList, adj: adj, swaps: done };
  }

  function degreesIntact(adj) {
    for (var n = 0; n < N; n++) if (adj[n].size !== baseDeg[n]) return false;
    return true;
  }

  // ------------------------------------------------------------------- ui ---

  var samples = [];      // every shuffle we have run, as full stat objects
  var current = STATS[0];
  var running = false;
  var degreeCheckOk = true;

  var select = root.querySelector("[data-role=stat]");
  var canvas = root.querySelector("[data-role=canvas]");
  var runBtn = root.querySelector("[data-role=run]");
  var resetBtn = root.querySelector("[data-role=reset]");
  var readout = root.querySelector("[data-role=readout]");
  var blurb = root.querySelector("[data-role=blurb]");
  var status = root.querySelector("[data-role=status]");
  var ctx = canvas.getContext("2d");

  STATS.forEach(function (s, i) {
    var o = document.createElement("option");
    o.value = String(i);
    o.textContent = s.label;
    select.appendChild(o);
  });

  function css(name, fallback) {
    var v = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return v || fallback;
  }

  function fmt(v, digits) {
    if (!isFinite(v)) return "—";
    return digits === 0 ? Math.round(v).toLocaleString("en-US") : v.toFixed(digits);
  }

  function draw() {
    var dpr = window.devicePixelRatio || 1;
    var w = canvas.clientWidth, h = canvas.clientHeight;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    var accent = css("--accent", "#58a6ff");
    var pink = "#f778ba";
    var muted = css("--text-muted", "#8b949e");
    var border = css("--border", "#21262d");

    var real = REAL[current.key];
    var vals = samples.map(function (s) { return s[current.key]; });

    var padL = 46, padR = 18, padT = 16, padB = 34;
    var plotW = w - padL - padR, plotH = h - padT - padB;

    // Axis range always contains the real value, so you can see how far out it is.
    var lo = real, hi = real;
    for (var i = 0; i < vals.length; i++) {
      if (vals[i] < lo) lo = vals[i];
      if (vals[i] > hi) hi = vals[i];
    }
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    var pad = (hi - lo) * 0.12;
    lo -= pad; hi += pad;

    var nBins = 26;
    var counts = new Array(nBins).fill(0);
    for (i = 0; i < vals.length; i++) {
      var b = Math.floor(((vals[i] - lo) / (hi - lo)) * nBins);
      if (b < 0) b = 0; if (b >= nBins) b = nBins - 1;
      counts[b]++;
    }
    var peak = Math.max(1, Math.max.apply(null, counts));

    ctx.strokeStyle = border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT);
    ctx.lineTo(padL, padT + plotH);
    ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    var bw = plotW / nBins;
    ctx.fillStyle = accent;
    for (i = 0; i < nBins; i++) {
      if (!counts[i]) continue;
      var bh = (counts[i] / peak) * plotH;
      ctx.fillRect(padL + i * bw + 0.5, padT + plotH - bh, Math.max(1, bw - 1), bh);
    }

    // the real network
    var rx = padL + ((real - lo) / (hi - lo)) * plotW;
    ctx.strokeStyle = pink;
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.moveTo(rx, padT - 4);
    ctx.lineTo(rx, padT + plotH);
    ctx.stroke();
    ctx.fillStyle = pink;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = rx > padL + plotW * 0.6 ? "right" : "left";
    ctx.fillText("real Marvel", rx + (rx > padL + plotW * 0.6 ? -6 : 6), padT + 8);

    ctx.fillStyle = muted;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.fillText(fmt(lo, current.fmt), padL, h - 12);
    ctx.textAlign = "right";
    ctx.fillText(fmt(hi, current.fmt), padL + plotW, h - 12);
    ctx.textAlign = "center";
    ctx.fillText(current.label, padL + plotW / 2, h - 12);
    ctx.textAlign = "right";
    ctx.fillText(String(peak), padL - 6, padT + 10);
    ctx.fillText("0", padL - 6, padT + plotH);
  }

  function updateReadout() {
    var real = REAL[current.key];
    var vals = samples.map(function (s) { return s[current.key]; });
    var rows = [["Real Marvel", fmt(real, current.fmt)]];

    if (vals.length >= 2) {
      var mean = vals.reduce(function (a, b) { return a + b; }, 0) / vals.length;
      var varSum = vals.reduce(function (a, b) { return a + (b - mean) * (b - mean); }, 0);
      var sd = Math.sqrt(varSum / (vals.length - 1));
      var z = sd > 1e-12 ? (real - mean) / sd : 0;
      var atLeast = vals.filter(function (v) { return v >= real; }).length;
      var atMost = vals.filter(function (v) { return v <= real; }).length;
      var p = (2 * (Math.min(atLeast, atMost) + 1)) / (vals.length + 1);
      var frozen = sd <= 1e-12;
      rows.push(["Shuffled mean", fmt(mean, current.fmt) + " ± " + fmt(sd, Math.max(current.fmt, 2))]);
      rows.push(["z-score", frozen ? "undefined (no spread)" : (z >= 0 ? "+" : "") + z.toFixed(2)]);
      if (!frozen) {
        rows.push(["Empirical p", p > 1 ? "1.0"
          : (p <= 2 / (vals.length + 1)
            ? "< " + (2 / (vals.length + 1)).toFixed(3)
            : p.toFixed(3))]);
      }
      rows.push(["Verdict", frozen
        ? "frozen — a function of the degree sequence alone, so the shuffle cannot move it"
        : Math.abs(z) < 2
          ? "survives the shuffle — the degrees already explain it"
          : "dies in the shuffle — this is real structure"]);
    } else {
      rows.push(["Shuffles so far", String(vals.length)]);
      rows.push(["", "Run some shuffles to build the null."]);
    }

    readout.innerHTML = rows.map(function (r) {
      return "<div class=\"sm-row\"><span>" + r[0] + "</span><strong>" + r[1] + "</strong></div>";
    }).join("");
    blurb.textContent = current.blurb;
    status.textContent = samples.length + " shuffles · " +
      SWAPS_PER_RUN.toLocaleString("en-US") + " edge swaps each · degrees " +
      (degreeCheckOk ? "verified unchanged" : "CHANGED (bug!)");
  }

  function runBatch(count) {
    if (running) return;
    running = true;
    runBtn.disabled = true;
    var left = count;

    function step() {
      var t0 = performance.now();
      // keep the frame under ~120 ms so the page never feels stuck
      while (left > 0 && performance.now() - t0 < 120) {
        var s = shuffledCopy(SWAPS_PER_RUN);
        if (samples.length === 0) degreeCheckOk = degreesIntact(s.adj);
        samples.push(measure(s.edges, s.adj));
        left--;
      }
      draw();
      updateReadout();
      runBtn.textContent = left > 0 ? "shuffling… (" + left + " to go)" : "Run 25 more shuffles";
      if (left > 0) {
        requestAnimationFrame(step);
      } else {
        running = false;
        runBtn.disabled = false;
      }
    }
    requestAnimationFrame(step);
  }

  select.addEventListener("change", function () {
    current = STATS[Number(select.value)];
    draw();
    updateReadout();
  });
  runBtn.addEventListener("click", function () { runBatch(25); });
  resetBtn.addEventListener("click", function () {
    samples = [];
    draw();
    updateReadout();
  });
  window.addEventListener("resize", draw);

  draw();
  updateReadout();
  runBatch(25); // give the reader something to look at straight away
})();
