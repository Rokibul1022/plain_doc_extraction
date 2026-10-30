/**
 * PlateScan front end: image and video plate scanning against /api/plate/*.
 * Vanilla JS; all server text is written with textContent (never innerHTML).
 */
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const el = (tag, cls, text) => {
    const n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  };

  function toast(msg, type = "info", ms = 3500) {
    const t = el("div", `toast ${type}`, msg);
    $("toastContainer").appendChild(t);
    setTimeout(() => t.remove(), ms);
  }

  // ---------- theme + capabilities
  const theme = localStorage.getItem("docscan_theme") || "dark";
  document.documentElement.setAttribute("data-theme", theme);
  $("themeBtn").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("docscan_theme", next);
  });

  fetch("/api/plate/capabilities").then((r) => r.json()).then((c) => {
    $("detDot").className = "status-dot" + (c.learned_detector ? "" : " inactive");
    $("detText").textContent = c.learned_detector ? "AI detector" : "Classic detector";
    $("ocrDot").className = "status-dot" + (c.ocr ? "" : " inactive");
    $("ocrText").textContent = c.ocr ? "OCR ready" : "OCR offline";
  }).catch(() => {
    $("detText").textContent = "Offline";
    $("ocrText").textContent = "Offline";
  });

  // ---------- tabs
  const MODES = { image: "imageMode", video: "videoMode", opencv: "opencvMode", merge: "mergeMode" };
  const TABS = { image: "tabImage", video: "tabVideo", opencv: "tabOpencv", merge: "tabMerge" };
  function setMode(mode) {
    Object.keys(MODES).forEach((m) => {
      $(MODES[m]).style.display = m === mode ? "" : "none";
      $(TABS[m]).classList.toggle("active", m === mode);
    });
  }
  $("tabMerge").addEventListener("click", () => setMode("merge"));
  $("tabOpencv").addEventListener("click", () => setMode("opencv"));
  $("tabImage").addEventListener("click", () => setMode("image"));
  $("tabVideo").addEventListener("click", () => setMode("video"));

  // ---------- rendering helpers
  const VERDICT_TEXT = { readable: "Readable", uncertain: "Uncertain", unreadable: "Unreadable" };

  function plateTextNode(reading) {
    const box = el("div", "plate-text");
    if (reading.script === "bengali") box.classList.add("bn");
    if (!reading.text) {
      box.classList.add("empty");
      box.textContent = "No text could be read";
      return box;
    }
    [...reading.text].forEach((ch, i) => {
      const c = (reading.char_confidence || [])[i];
      const span = el("span", "ch " + (c === undefined ? "" : c >= 0.9 ? "hi" : c >= 0.6 ? "mid" : "lo"), ch);
      span.title = c === undefined ? "" : `confidence ${(c * 100).toFixed(0)}%`;
      box.appendChild(span);
    });
    return box;
  }

  function chips(plate) {
    const wrap = el("div", "chips");
    (plate.quality.issues || []).forEach((i) => wrap.appendChild(el("span", "chip warn", i.replace(/_/g, " "))));
    (plate.steps || []).filter((s) => !/^contrast|^sharpen/.test(s)).forEach((s) =>
      wrap.appendChild(el("span", "chip fix", s.replace(/\(.*\)/, ""))));
    return wrap;
  }

  function plateCard(plate, label) {
    const card = el("div", "plate-card");
    const head = el("div", "plate-head");
    head.appendChild(el("strong", "", label));
    head.appendChild(el("span", `verdict ${plate.verdict}`, VERDICT_TEXT[plate.verdict] || plate.verdict));
    card.appendChild(head);
    card.appendChild(plateTextNode(plate.reading));

    const rel = plate.reading.reliability;
    if (rel !== null && rel !== undefined) {
      const row = el("div", "rel-row");
      row.appendChild(el("span", "", "Reliability"));
      const bar = el("div", "rel-bar");
      const fill = el("div");
      fill.style.width = `${Math.round(rel * 100)}%`;
      bar.appendChild(fill);
      row.appendChild(bar);
      row.appendChild(el("span", "", `${Math.round(rel * 100)}%`));
      card.appendChild(row);
    }
    card.appendChild(chips(plate));

    // image views
    const names = [["document", "Plain document"], ["typed", "Typed (best guess)"], ["scan", "Cleaned scan"], ["crop", "Original"], ["fused", "Fused"],
                   ["restored", "Restored"], ["readable", "Plain text"]]
      .filter(([k]) => plate.images && plate.images[k]);
    const tabs = el("div", "view-tabs");
    const stage = el("div", "view-stage");
    const img = el("img");
    stage.appendChild(img);
    let current = null;
    const show = (k) => {
      current = k;
      img.src = plate.images[k];
      img.alt = k;
      [...tabs.children].forEach((b) => b.classList.toggle("active", b.dataset.k === k));
    };
    names.forEach(([k, n]) => {
      const b = el("button", "view-tab", n);
      b.dataset.k = k;
      b.addEventListener("click", () => show(k));
      tabs.appendChild(b);
    });
    card.appendChild(tabs);
    card.appendChild(stage);
    if (names.length) show(plate.images.document ? "document" : plate.images.readable ? "readable" : names[0][0]);

    const qa = plate.quality_after, q = plate.quality;
    if (q) {
      const m = q.metrics || {};
      card.appendChild(el("div", "quality-line",
        `Plate ${Math.round(m.width_px || 0)} px wide · quality ${Math.round(q.score)} (${q.level})` +
        (qa ? ` → ${Math.round(qa.score)} after restoration` : "")));
    }
    if (plate.frame_readings && plate.frame_readings.length) {
      card.appendChild(el("div", "quality-line", `Seen in ${plate.frames_seen} frames; reads from the best frames:`));
      const fr = el("div", "frame-reads");
      plate.frame_readings.forEach(([f, t, c]) => {
        const chip = el("span", "chip");
        chip.appendChild(document.createTextNode(`#${f} `));
        chip.appendChild(el("b", "", t));
        chip.appendChild(document.createTextNode(` ${Math.round(c * 100)}%`));
        fr.appendChild(chip);
      });
      card.appendChild(fr);
    }

    const actions = el("div", "card-actions");
    const copy = el("button", "btn btn-secondary btn-sm", "Copy text");
    copy.disabled = !plate.reading.text;
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(plate.reading.text).then(() => toast("Plate text copied", "success"),
        () => toast("Clipboard not available", "error"));
    });
    const dl = el("button", "btn btn-secondary btn-sm", "Download view");
    dl.addEventListener("click", () => {
      if (!current) return;
      const a = el("a");
      a.href = plate.images[current];
      a.download = `plate_${(plate.reading.text || "unknown")}_${current}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    });
    actions.appendChild(copy);
    actions.appendChild(dl);
    if (plate.steps && plate.steps.length) {
      const d = el("details", "steps");
      d.appendChild(el("summary", "", "Processing steps"));
      d.appendChild(el("div", "", plate.steps.join(" → ")));
      card.appendChild(actions);
      card.appendChild(d);
    } else {
      card.appendChild(actions);
    }
    return card;
  }

  // ---------- image mode
  function wireDrop(zone, input, button, onFiles) {
    button.addEventListener("click", (e) => { e.stopPropagation(); input.click(); });
    zone.addEventListener("click", () => input.click());
    input.addEventListener("change", (e) => { if (e.target.files.length) onFiles([...e.target.files]); e.target.value = ""; });
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      if (e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]);
    });
  }

  function readCard(d) {
    const card = el("div", "plate-card");
    const head = el("div", "plate-head");
    head.appendChild(el("strong", "", d.model === "merged" ? "Merged reader" + (d.mode === "relief" ? " · embossed view" : "") : d.mode === "relief" ? "Embossed plate reader" : "Plate reader"));
    head.appendChild(el("span", `verdict ${d.verdict}`, VERDICT_TEXT[d.verdict] || d.verdict));
    card.appendChild(head);
    const top = d.letter ? `${d.top}-${d.letter}` : d.top;
    card.appendChild(el("div", "plate-text bn" + (top ? "" : " empty"), top || "top row not recognised"));
    card.appendChild(el("div", "plate-text bn", d.number));
    const chipsEl = el("div", "chips");
    chipsEl.appendChild(el("span", "chip fix", `${d.mode} reader`));
    if (d.model === "merged") {
      chipsEl.appendChild(el("span", "chip fix", `${d.device && d.device.startsWith("cuda") ? "GPU" : "CPU"} · ${d.views} views`));
      if (d.retried) chipsEl.appendChild(el("span", "chip warn", "re-read with more views (first read was not readable)"));
    }
    chipsEl.appendChild(el("span", "chip fix", `corners: ${d.corner_method}`));
    card.appendChild(chipsEl);
    const names = [["result", "Result"], ["page", d.mode === "relief" ? "Relief outlines" : "Plain page"], ["straight", "Straightened"], ["original", "Original + corners"]];
    const tabs = el("div", "view-tabs");
    const stage = el("div", "view-stage");
    const img = el("img");
    stage.appendChild(img);
    const show = (k) => {
      img.src = d.images[k];
      img.alt = k;
      [...tabs.children].forEach((b) => b.classList.toggle("active", b.dataset.k === k));
    };
    names.forEach(([k, n]) => {
      const b = el("button", "view-tab", n);
      b.dataset.k = k;
      b.addEventListener("click", () => show(k));
      tabs.appendChild(b);
    });
    card.appendChild(tabs);
    card.appendChild(stage);
    show("result");
    card.appendChild(el("div", "quality-line", `${d.width}×${d.height} · ${d.elapsed_s}s`));
    return card;
  }

  async function scanImage(file, api = "/api/plate/read", resultsId = "imageResults", reliefBox = "embossedRead") {
    const block = el("div");
    block.appendChild(el("div", "file-title", file.name));
    const status = el("div", "no-plates");
    status.appendChild(el("span", "spinner-inline"));
    status.appendChild(document.createTextNode("Reading…"));
    block.appendChild(status);
    $(resultsId).prepend(block);

    const fd = new FormData();
    fd.append("file", file);
    fd.append("mode", $(reliefBox).checked ? "relief" : "auto");
    try {
      const res = await fetch(api, { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
      const data = await res.json();
      status.remove();
      block.appendChild(readCard(data));
      if (data.model === "merged") addMergeRow(file.name, data);
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
      toast(err.message, "error");
    }
  }

  document.querySelector("#dropImage .cv-opt").addEventListener("click", (e) => e.stopPropagation());
  wireDrop($("dropImage"), $("fileImage"), $("pickImage"), async (files) => {
    $("dropImage").classList.add("compact");
    for (const f of files) await scanImage(f);
  });

  // ---------- merged-reader mode (same card, /api/merge/read)
  const mergeRows = [];
  function addMergeRow(name, d) {
    mergeRows.push([name, d.top, d.letter, d.digits, d.number, d.verdict, d.mode, d.retried ? "yes" : "no", d.device, d.elapsed_s]);
    $("mergeTools").style.display = "";
    $("mergeCount").textContent = `${mergeRows.length} plate(s) read`;
  }
  $("mergeCsv").addEventListener("click", () => {
    const q = (v) => `"${String(v).replace(/"/g, '""')}"`;
    const head = ["file", "city", "letter", "digits", "number", "verdict", "reader", "re-read", "device", "seconds"];
    const blob = new Blob(["\ufeff" + [head, ...mergeRows].map((r) => r.map(q).join(",")).join("\r\n")], { type: "text/csv;charset=utf-8" });
    const a = el("a");
    a.href = URL.createObjectURL(blob);
    a.download = "ocr_merge_results.csv";
    document.body.appendChild(a);
    a.click();
    a.remove();
  });
  document.querySelector("#dropMerge .cv-opt").addEventListener("click", (e) => e.stopPropagation());
  wireDrop($("dropMerge"), $("fileMerge"), $("pickMerge"), async (files) => {
    $("dropMerge").classList.add("compact");
    for (const f of files) await scanImage(f, "/api/merge/read", "mergeResults", "mergeRelief");
  });
  fetch("/api/merge/capabilities").then((r) => r.json()).then((c) => {
    $("mergeInfo").textContent = c.available
      ? `${c.model_mb} MB model on ${c.device} · ${c.views} views per crop` + (c.retry_views ? `, doubtful plates re-read with ${c.retry_views}` : "") + ` · class priors ${c.priors ? "on" : "off"}`
      : "model not found (ocr_merge/plate_net.onnx).";
  }).catch(() => { $("mergeInfo").textContent = "server offline."; });

  // ---------- video mode
  async function scanVideo(file) {
    $("videoResults").textContent = "";
    $("videoProgress").style.display = "";
    $("progFill").style.width = "0%";
    $("progPct").textContent = "0%";
    $("progText").textContent = `Uploading ${file.name}…`;
    const fd = new FormData();
    fd.append("file", file);
    const q = new URLSearchParams({ fps: $("fpsSel").value });
    if ($("secSel").value) q.set("max_seconds", $("secSel").value);
    try {
      const up = await fetch(`/api/plate/video?${q}`, { method: "POST", body: fd });
      if (!up.ok) throw new Error((await up.json()).detail || `HTTP ${up.status}`);
      const { job_id } = await up.json();
      $("progText").textContent = "Tracking plates and reading frames…";
      for (;;) {
        await new Promise((r) => setTimeout(r, 1000));
        const job = await (await fetch(`/api/plate/job/${job_id}`)).json();
        const pct = Math.round((job.progress || 0) * 100);
        $("progFill").style.width = `${pct}%`;
        $("progPct").textContent = `${pct}%`;
        if (job.status === "error") throw new Error(job.error || "Video analysis failed");
        if (job.status === "done") { renderVideo(job.result, file.name); break; }
      }
    } catch (err) {
      toast(err.message, "error", 6000);
      $("progText").textContent = `Error: ${err.message}`;
      return;
    }
    $("videoProgress").style.display = "none";
  }

  function renderVideo(result, name) {
    const box = $("videoResults");
    box.appendChild(el("div", "file-title",
      `${name} — ${result.plates.length} plate(s), ${result.frames_processed} frames analysed in ${result.elapsed_s}s`));
    if (!result.plates.length) {
      box.appendChild(el("div", "no-plates", "No plate was tracked in this video."));
      return;
    }
    const col = el("div", "plates-col");
    result.plates.forEach((p) => col.appendChild(plateCard(p,
      `Vehicle ${p.track_id} · best frame #${p.best_frame} (${p.best_time_s}s)`)));
    box.appendChild(col);
  }

  wireDrop($("dropVideo"), $("fileVideo"), $("pickVideo"), (files) => scanVideo(files[0]));
})();
