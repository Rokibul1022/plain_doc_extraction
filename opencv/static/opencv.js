/**
 * OpenCV tab: classical plate scan (corners -> straighten -> clean -> plain page) against /api/opencv/scan.
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

  function figure(src, caption, cls) {
    const f = el("figure", "cv-fig" + (cls ? " " + cls : ""));
    const im = el("img");
    im.src = src;
    im.alt = caption;
    f.appendChild(im);
    f.appendChild(el("figcaption", "", caption));
    return f;
  }

  function download(href, name) {
    const a = el("a", "btn btn-secondary btn-sm", "Download plain page");
    a.href = href;
    a.download = name;
    return a;
  }

  async function scanOne(file) {
    const block = el("div", "cv-block");
    block.appendChild(el("div", "file-title", file.name));
    const status = el("div", "no-plates");
    status.appendChild(el("span", "spinner-inline"));
    status.appendChild(document.createTextNode("Scanning…"));
    block.appendChild(status);
    $("opencvResults").prepend(block);

    const fd = new FormData();
    fd.append("file", file);
    fd.append("relief", $("opencvRelief").checked ? "force" : "auto");
    try {
      const res = await fetch("/api/opencv/scan", { method: "POST", body: fd });
      if (!res.ok) throw new Error((await res.json()).detail || `HTTP ${res.status}`);
      const d = await res.json();
      status.remove();
      const row = el("div", "cv-row");
      row.appendChild(figure(d.original, "Detected corners"));
      row.appendChild(figure(d.rectified, "Straightened plate"));
      row.appendChild(figure(d.document, "Final plain page", "cv-final"));
      block.appendChild(row);
      const meta = el("div", "scene-meta");
      meta.appendChild(el("span", "", `${d.width}×${d.height}`));
      meta.appendChild(el("span", "", `corners: ${d.method}`));
      meta.appendChild(el("span", "", `${d.elapsed_s}s`));
      if (d.relief === "force") meta.appendChild(el("span", "", "3D relief mode"));
      meta.appendChild(download(d.document, file.name.replace(/\.[^.]+$/, "") + "_plain.png"));
      block.appendChild(meta);
    } catch (err) {
      status.textContent = `Error: ${err.message}`;
    }
  }

  const zone = $("dropOpencv"), input = $("fileOpencv");
  $("pickOpencv").addEventListener("click", (e) => { e.stopPropagation(); input.click(); });
  document.querySelector(".cv-opt").addEventListener("click", (e) => e.stopPropagation());
  zone.addEventListener("click", () => input.click());
  const run = async (files) => {
    zone.classList.add("compact");
    for (const f of files) await scanOne(f);
  };
  input.addEventListener("change", (e) => { if (e.target.files.length) run([...e.target.files]); e.target.value = ""; });
  zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("dragover"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("dragover");
    if (e.dataTransfer.files.length) run([...e.dataTransfer.files]);
  });
})();
