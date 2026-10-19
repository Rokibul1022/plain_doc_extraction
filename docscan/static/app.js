/**
 * DocScan - CamScanner-Style Plain Text Document Processor
 * Lightweight, zero-dependency, modern Vanilla JS.
 */

(function () {
  "use strict";

  // Application State
  const state = {
    pages: [],
    activePageIndex: 0,
    currentView: "viewUpload",
    capabilities: { ocr: false, heif: false },
    theme: localStorage.getItem("docscan_theme") || "dark",
    crop: {
      draggingIdx: -1,
      handleRadius: 14,
      hitRadius: 28,
      scaleX: 1,
      scaleY: 1,
      offsetX: 0,
      offsetY: 0,
    },
    preview: {
      abortController: null,
      viewMode: "split",
    },
  };

  // DOM Elements
  const DOM = {
    themeToggleBtn: document.getElementById("themeToggleBtn"),
    btnNavAction: document.getElementById("btnNavAction"),
    ocrBadge: document.getElementById("ocrBadge"),
    ocrDot: document.getElementById("ocrDot"),
    ocrText: document.getElementById("ocrText"),
    tabs: {
      upload: document.getElementById("tabUpload"),
      crop: document.getElementById("tabCrop"),
      enhance: document.getElementById("tabEnhance"),
    },
    views: {
      upload: document.getElementById("viewUpload"),
      crop: document.getElementById("viewCrop"),
      enhance: document.getElementById("viewEnhance"),
    },
    dropZone: document.getElementById("dropZone"),
    fileInput: document.getElementById("fileInput"),
    selectFilesBtn: document.getElementById("selectFilesBtn"),

    // Crop View
    cropCanvas: document.getElementById("cropCanvas"),
    cropWrapper: document.getElementById("cropCanvasWrapper"),
    magnifierLoupe: document.getElementById("magnifierLoupe"),
    cropSpinner: document.getElementById("cropSpinner"),
    btnAutoDetect: document.getElementById("btnAutoDetect"),
    btnResetCorners: document.getElementById("btnResetCorners"),
    btnRotateCrop: document.getElementById("btnRotateCrop"),
    btnApplyCrop: document.getElementById("btnApplyCrop"),
    btnApplyCropToolbar: document.getElementById("btnApplyCropToolbar"),
    confidenceBadge: document.getElementById("confidenceBadge"),
    confidenceVal: document.getElementById("confidenceVal"),

    // Enhance View
    previewSpinner: document.getElementById("previewSpinner"),
    imgBefore: document.getElementById("imgBefore"),
    imgAfter: document.getElementById("imgAfter"),
    boxBefore: document.getElementById("boxBefore"),
    boxAfter: document.getElementById("boxAfter"),
    activeModeLabel: document.getElementById("activeModeLabel"),
    renderTiming: document.getElementById("renderTiming"),
    btnViewSplit: document.getElementById("btnViewSplit"),
    btnViewEnhancedOnly: document.getElementById("btnViewEnhancedOnly"),
    btnViewOriginalOnly: document.getElementById("btnViewOriginalOnly"),
    btnBackToCrop: document.getElementById("btnBackToCrop"),
    btnOpenExport: document.getElementById("btnOpenExport"),

    // Tuning Controls
    modePills: document.querySelectorAll(".mode-pill"),
    sliderStrokeWeight: document.getElementById("sliderStrokeWeight"),
    valStrokeWeight: document.getElementById("valStrokeWeight"),
    sliderBrightness: document.getElementById("sliderBrightness"),
    sliderContrast: document.getElementById("sliderContrast"),
    valBrightness: document.getElementById("valBrightness"),
    valContrast: document.getElementById("valContrast"),
    chkDeskew: document.getElementById("chkDeskew"),

    // Pages Bar
    pagesBar: document.getElementById("pagesBar"),
    pagesCountLabel: document.getElementById("pagesCountLabel"),
    pagesCarousel: document.getElementById("pagesCarousel"),
    btnAddPage: document.getElementById("btnAddPage"),
    btnBottomAction: document.getElementById("btnBottomAction"),

    // Export Modal
    exportModal: document.getElementById("exportModal"),
    btnCloseExportModal: document.getElementById("btnCloseExportModal"),
    btnCancelExport: document.getElementById("btnCancelExport"),
    btnDoExport: document.getElementById("btnDoExport"),
    exportFormat: document.getElementById("exportFormat"),
    exportPageSize: document.getElementById("exportPageSize"),
    pageSizeGroup: document.getElementById("pageSizeGroup"),
    exportQuality: document.getElementById("exportQuality"),
    valQuality: document.getElementById("valQuality"),
    chkExportOcr: document.getElementById("chkExportOcr"),
    ocrOptionText: document.getElementById("ocrOptionText"),
    exportBtnText: document.getElementById("exportBtnText"),

    toastContainer: document.getElementById("toastContainer"),
  };

  const cropCtx = DOM.cropCanvas.getContext("2d");
  const loupeCtx = DOM.magnifierLoupe.getContext("2d");
  let activeImageObj = new Image();

  function createDefaultParams() {
    return {
      mode: "plain_text",
      stroke_weight: 0.0,
      clean_background: true,
      brightness: 0.0,
      contrast: 0.0,
      deskew: false,
      rotation: 0,
      threshold_method: "plain_text",
      threshold_block_size: 35,
      threshold_c: 14.0,
      remove_speckles: true,
    };
  }

  function showToast(message, type = "info", duration = 3000) {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;
    toast.textContent = message;
    DOM.toastContainer.appendChild(toast);
    requestAnimationFrame(() => toast.classList.add("show"));
    setTimeout(() => {
      toast.classList.remove("show");
      setTimeout(() => toast.remove(), 250);
    }, duration);
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    state.theme = theme;
    localStorage.setItem("docscan_theme", theme);
  }

  DOM.themeToggleBtn.addEventListener("click", () => {
    applyTheme(state.theme === "dark" ? "light" : "dark");
  });
  applyTheme(state.theme);

  async function checkServerHealth() {
    try {
      const res = await fetch("/api/health");
      if (res.ok) {
        const data = await res.json();
        state.capabilities = data.capabilities || {};
        if (state.capabilities.ocr) {
          DOM.ocrDot.className = "status-dot";
          DOM.ocrText.textContent = "OCR Ready";
          DOM.chkExportOcr.disabled = false;
        } else {
          DOM.ocrDot.className = "status-dot inactive";
          DOM.ocrText.textContent = "OCR Offline";
          DOM.chkExportOcr.disabled = true;
        }
      }
    } catch {
      DOM.ocrText.textContent = "Local";
    }
  }
  checkServerHealth();

  // Tab & View Navigation
  function switchView(viewName) {
    state.currentView = viewName;
    Object.keys(DOM.views).forEach((k) => {
      DOM.views[k].classList.toggle("active", DOM.views[k].id === viewName);
    });

    DOM.tabs.upload.classList.toggle("active", viewName === "viewUpload");
    DOM.tabs.crop.classList.toggle("active", viewName === "viewCrop");
    DOM.tabs.enhance.classList.toggle("active", viewName === "viewEnhance");

    if (viewName === "viewCrop") {
      updateActionButtons("crop");
      loadActivePageForCropping();
    } else if (viewName === "viewEnhance") {
      updateActionButtons("enhance");
      loadActivePageForEnhancing();
    } else {
      if (DOM.pagesBar) DOM.pagesBar.style.display = "none";
      if (DOM.btnNavAction) DOM.btnNavAction.style.display = "none";
    }
  }

  function updateActionButtons(view) {
    const isCrop = view === "crop";
    const label = isCrop ? "Next: Enhance & Extract" : "Export Document";
    const icon = isCrop
      ? `<line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/>`
      : `<polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>`;
    const action = isCrop ? () => switchView("viewEnhance") : openExportModal;

    // 1. Top Nav Action Button (Always visible at top of viewport)
    if (DOM.btnNavAction) {
      DOM.btnNavAction.style.display = "inline-flex";
      DOM.btnNavAction.innerHTML = `<span>${label}</span> <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">${icon}</svg>`;
      DOM.btnNavAction.onclick = action;
    }

    // 2. Bottom Fixed Bar Button (Always visible at bottom of viewport)
    if (DOM.btnBottomAction) {
      DOM.btnBottomAction.innerHTML = `<span>${label}</span> <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">${icon}</svg>`;
      DOM.btnBottomAction.onclick = action;
    }
  }

  DOM.tabs.upload.addEventListener("click", () => switchView("viewUpload"));
  DOM.tabs.crop.addEventListener("click", () => {
    if (state.pages.length > 0) switchView("viewCrop");
  });
  DOM.tabs.enhance.addEventListener("click", () => {
    if (state.pages.length > 0) switchView("viewEnhance");
  });

  // File Upload Handlers
  DOM.selectFilesBtn.addEventListener("click", () => DOM.fileInput.click());
  DOM.fileInput.addEventListener("change", (e) => {
    handleFiles(e.target.files);
    e.target.value = "";
  });

  DOM.dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    DOM.dropZone.classList.add("dragover");
  });
  DOM.dropZone.addEventListener("dragleave", () => {
    DOM.dropZone.classList.remove("dragover");
  });
  DOM.dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    DOM.dropZone.classList.remove("dragover");
    handleFiles(e.dataTransfer.files);
  });

  async function handleFiles(files) {
    if (!files || files.length === 0) return;
    showToast(`Uploading ${files.length} document(s)...`, "info");

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/api/upload", {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const errData = await res.json();
          throw new Error(errData.detail || "Upload failed");
        }

        const data = await res.json();
        const newPage = {
          id: data.image_id,
          imageId: data.image_id,
          width: data.width,
          height: data.height,
          previewUrl: data.preview_url,
          corners: [
            [0, 0],
            [data.width - 1, 0],
            [data.width - 1, data.height - 1],
            [0, data.height - 1],
          ],
          params: createDefaultParams(),
          detected: false,
          confidence: 0,
        };

        state.pages.push(newPage);
      } catch (err) {
        showToast(`Upload error: ${err.message}`, "error");
      }
    }

    if (state.pages.length > 0) {
      DOM.pagesBar.style.display = "flex";
      renderPagesCarousel();
      state.activePageIndex = state.pages.length - 1;
      switchView("viewCrop");
    }
  }

  // Corner & Document Detection
  async function runCornerDetection(page) {
    DOM.cropSpinner.classList.add("active");
    try {
      const res = await fetch("/api/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_id: page.imageId, target: "auto" }),
      });
      if (!res.ok) throw new Error("Document detection failed");
      const data = await res.json();
      page.corners = data.corners;
      page.confidence = data.confidence;
      page.detected = data.detected;

      updateConfidenceUI(data.confidence, data.detected);
      renderCropCanvas();
      showToast("Document boundaries detected", "success");
    } catch (err) {
      showToast(err.message, "error");
    } finally {
      DOM.cropSpinner.classList.remove("active");
    }
  }

  function updateConfidenceUI(confidence, detected) {
    const pct = Math.round(confidence * 100);
    DOM.confidenceVal.textContent = detected ? `${pct}%` : "Full Frame";
    DOM.confidenceBadge.className = "confidence-indicator";
    if (!detected || confidence < 0.3) {
      DOM.confidenceBadge.classList.add("fallback");
    } else {
      DOM.confidenceBadge.classList.add("high");
    }
  }

  function loadActivePageForCropping() {
    const page = state.pages[state.activePageIndex];
    if (!page) return;

    activeImageObj = new Image();
    activeImageObj.onload = () => {
      setupCropCanvasSize();
      if (!page.detected && page.confidence === 0) {
        runCornerDetection(page);
      } else {
        updateConfidenceUI(page.confidence, page.detected);
        renderCropCanvas();
      }
    };
    activeImageObj.src = page.previewUrl;
  }

  function setupCropCanvasSize() {
    if (!activeImageObj.width) return;
    const page = state.pages[state.activePageIndex];

    const isMobile = window.innerWidth <= 1024;
    const panel = DOM.cropCanvas.closest(".canvas-panel");
    const containerW = panel && panel.clientWidth > 50
      ? panel.clientWidth - 32
      : (isMobile ? window.innerWidth - 40 : window.innerWidth - 420);
    const availableW = Math.max(260, Math.min(containerW, 880));

    // Calculate max height so the canvas + toolbar ALWAYS fits without vertical page scrolling
    const overhead = (DOM.pagesBar && DOM.pagesBar.style.display !== "none") ? 280 : 200;
    const availableH = Math.max(220, window.innerHeight - overhead);
    const maxH = Math.min(availableH, 460);

    let displayW = activeImageObj.width;
    let displayH = activeImageObj.height;

    const scale = Math.min(availableW / displayW, maxH / displayH, 1.0);
    displayW = Math.max(10, Math.round(displayW * scale));
    displayH = Math.max(10, Math.round(displayH * scale));

    const dpr = window.devicePixelRatio || 1;
    DOM.cropCanvas.width = displayW * dpr;
    DOM.cropCanvas.height = displayH * dpr;
    DOM.cropCanvas.style.width = `${displayW}px`;
    DOM.cropCanvas.style.height = `${displayH}px`;

    cropCtx.resetTransform();
    cropCtx.scale(dpr, dpr);

    state.crop.scaleX = displayW / page.width;
    state.crop.scaleY = displayH / page.height;
  }

  // Draw Quadrilateral and Interactive Corner Handles
  function renderCropCanvas() {
    const page = state.pages[state.activePageIndex];
    if (!page || !activeImageObj.width) return;

    const dW = parseFloat(DOM.cropCanvas.style.width);
    const dH = parseFloat(DOM.cropCanvas.style.height);

    cropCtx.clearRect(0, 0, dW, dH);
    cropCtx.drawImage(activeImageObj, 0, 0, dW, dH);

    const pts = page.corners.map(([x, y]) => [
      x * state.crop.scaleX,
      y * state.crop.scaleY,
    ]);

    // Semi-transparent dark mask outside document quadrilateral
    cropCtx.save();
    cropCtx.fillStyle = "rgba(0, 0, 0, 0.45)";
    cropCtx.beginPath();
    cropCtx.rect(0, 0, dW, dH);
    cropCtx.moveTo(pts[0][0], pts[0][1]);
    cropCtx.lineTo(pts[3][0], pts[3][1]);
    cropCtx.lineTo(pts[2][0], pts[2][1]);
    cropCtx.lineTo(pts[1][0], pts[1][1]);
    cropCtx.closePath();
    cropCtx.fill("evenodd");

    // Quadrilateral border line
    cropCtx.strokeStyle = "#38bdf8";
    cropCtx.lineWidth = 2.5;
    cropCtx.beginPath();
    cropCtx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < 4; i++) cropCtx.lineTo(pts[i][0], pts[i][1]);
    cropCtx.closePath();
    cropCtx.stroke();
    cropCtx.restore();

    // Corner handle circles
    pts.forEach(([cx, cy], idx) => {
      const isDragging = state.crop.draggingIdx === idx;
      cropCtx.save();
      cropCtx.shadowColor = "rgba(0,0,0,0.4)";
      cropCtx.shadowBlur = 8;
      cropCtx.beginPath();
      cropCtx.arc(cx, cy, isDragging ? 16 : 12, 0, 2 * Math.PI);
      cropCtx.fillStyle = isDragging ? "#38bdf8" : "#ffffff";
      cropCtx.fill();
      cropCtx.lineWidth = 3;
      cropCtx.strokeStyle = "#0284c7";
      cropCtx.stroke();
      cropCtx.restore();
    });
  }

  function updateMagnifierLoupe(canvasX, canvasY, origX, origY) {
    const loupe = DOM.magnifierLoupe;
    const lSize = 120;
    loupe.style.display = "block";

    let left = canvasX + 25;
    let top = canvasY - 140;
    if (top < 10) top = canvasY + 25;
    if (left + lSize > DOM.cropCanvas.width) left = canvasX - lSize - 25;

    loupe.style.left = `${left}px`;
    loupe.style.top = `${top}px`;

    loupeCtx.clearRect(0, 0, lSize, lSize);
    loupeCtx.save();
    loupeCtx.beginPath();
    loupeCtx.arc(lSize / 2, lSize / 2, lSize / 2 - 2, 0, Math.PI * 2);
    loupeCtx.clip();

    const zoom = 2.5;
    const sW = lSize / zoom;
    const sH = lSize / zoom;
    const sX = origX - sW / 2;
    const sY = origY - sH / 2;

    loupeCtx.drawImage(activeImageObj, sX, sY, sW, sH, 0, 0, lSize, lSize);

    // Crosshair target
    loupeCtx.strokeStyle = "#38bdf8";
    loupeCtx.lineWidth = 1.5;
    loupeCtx.beginPath();
    loupeCtx.moveTo(lSize / 2 - 12, lSize / 2);
    loupeCtx.lineTo(lSize / 2 + 12, lSize / 2);
    loupeCtx.moveTo(lSize / 2, lSize / 2 - 12);
    loupeCtx.lineTo(lSize / 2, lSize / 2 + 12);
    loupeCtx.stroke();
    loupeCtx.restore();
  }

  // Pointer Interaction for Dragging Corners
  function getCanvasCoords(evt) {
    const rect = DOM.cropCanvas.getBoundingClientRect();
    const clientX = evt.touches ? evt.touches[0].clientX : evt.clientX;
    const clientY = evt.touches ? evt.touches[0].clientY : evt.clientY;
    return {
      x: clientX - rect.left,
      y: clientY - rect.top,
    };
  }

  function onPointerDown(e) {
    const page = state.pages[state.activePageIndex];
    if (!page) return;
    const { x, y } = getCanvasCoords(e);

    const hitIdx = page.corners.findIndex(([cx, cy]) => {
      const sx = cx * state.crop.scaleX;
      const sy = cy * state.crop.scaleY;
      return Math.hypot(sx - x, sy - y) <= state.crop.hitRadius;
    });

    if (hitIdx !== -1) {
      e.preventDefault();
      state.crop.draggingIdx = hitIdx;
      updateMagnifierLoupe(x, y, page.corners[hitIdx][0], page.corners[hitIdx][1]);
      renderCropCanvas();
    }
  }

  function onPointerMove(e) {
    if (state.crop.draggingIdx === -1) return;
    e.preventDefault();
    const page = state.pages[state.activePageIndex];
    if (!page) return;
    const { x, y } = getCanvasCoords(e);

    const origX = Math.max(0, Math.min(page.width - 1, Math.round(x / state.crop.scaleX)));
    const origY = Math.max(0, Math.min(page.height - 1, Math.round(y / state.crop.scaleY)));

    page.corners[state.crop.draggingIdx] = [origX, origY];
    renderCropCanvas();
    updateMagnifierLoupe(x, y, origX, origY);
  }

  function onPointerUp() {
    if (state.crop.draggingIdx !== -1) {
      state.crop.draggingIdx = -1;
      DOM.magnifierLoupe.style.display = "none";
      renderCropCanvas();
    }
  }

  DOM.cropCanvas.addEventListener("mousedown", onPointerDown);
  window.addEventListener("mousemove", onPointerMove);
  window.addEventListener("mouseup", onPointerUp);

  DOM.cropCanvas.addEventListener("touchstart", onPointerDown, { passive: false });
  window.addEventListener("touchmove", onPointerMove, { passive: false });
  window.addEventListener("touchend", onPointerUp);

  DOM.btnAutoDetect.addEventListener("click", () => {
    const page = state.pages[state.activePageIndex];
    if (page) runCornerDetection(page);
  });

  DOM.btnResetCorners.addEventListener("click", () => {
    const page = state.pages[state.activePageIndex];
    if (!page) return;
    page.corners = [
      [0, 0],
      [page.width - 1, 0],
      [page.width - 1, page.height - 1],
      [0, page.height - 1],
    ];
    page.detected = false;
    page.confidence = 0.0;
    updateConfidenceUI(0.0, false);
    renderCropCanvas();
    showToast("Corners reset to full image frame", "info");
  });

  DOM.btnRotateCrop.addEventListener("click", () => {
    const page = state.pages[state.activePageIndex];
    if (!page) return;
    page.params.rotation = (page.params.rotation + 90) % 360;
    showToast(`Rotated page ${page.params.rotation}°`, "info");
  });

  const handleApplyCrop = () => {
    switchView("viewEnhance");
  };
  if (DOM.btnApplyCrop) DOM.btnApplyCrop.addEventListener("click", handleApplyCrop);
  if (DOM.btnApplyCropToolbar) DOM.btnApplyCropToolbar.addEventListener("click", handleApplyCrop);

  // Responsive redraw on window resize
  window.addEventListener("resize", () => {
    if (state.currentView === "viewCrop" && activeImageObj.width) {
      setupCropCanvasSize();
      renderCropCanvas();
    }
  });

  // Debounced Enhance Preview
  let debounceTimer = null;
  function triggerPreviewUpdate(immediate = false) {
    clearTimeout(debounceTimer);
    if (immediate) {
      fetchPreview();
    } else {
      debounceTimer = setTimeout(fetchPreview, 120);
    }
  }

  async function fetchPreview() {
    const page = state.pages[state.activePageIndex];
    if (!page) return;

    if (state.preview.abortController) {
      state.preview.abortController.abort();
    }
    state.preview.abortController = new AbortController();

    DOM.previewSpinner.classList.add("active");
    DOM.renderTiming.textContent = "Extracting plain text...";

    try {
      const payload = {
        image_id: page.imageId,
        corners: page.corners,
        params: page.params,
      };

      const res = await fetch("/api/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: state.preview.abortController.signal,
      });

      if (!res.ok) throw new Error("Preview generation failed");

      const modeUsed = res.headers.get("X-Mode-Used") || page.params.mode;
      const totalMs = res.headers.get("X-Total-Ms") || "--";
      DOM.activeModeLabel.textContent =
        modeUsed === "plain_text"
          ? "Plain Text"
          : modeUsed === "magic_color"
          ? "Magic Color"
          : modeUsed.toUpperCase();
      DOM.renderTiming.textContent = `Cleaned in ${totalMs} ms`;

      const blob = await res.blob();
      const objectUrl = URL.createObjectURL(blob);
      DOM.imgAfter.src = objectUrl;

      updatePageThumbnail(state.activePageIndex, objectUrl);
    } catch (err) {
      if (err.name !== "AbortError") {
        console.error("Preview error:", err);
        DOM.renderTiming.textContent = "Error rendering preview";
      }
    } finally {
      DOM.previewSpinner.classList.remove("active");
    }
  }

  function loadActivePageForEnhancing() {
    const page = state.pages[state.activePageIndex];
    if (!page) return;

    syncControlsFromParams(page.params);
    DOM.imgBefore.src = page.previewUrl;
    triggerPreviewUpdate(true);
  }

  function getStrokeWeightText(val) {
    if (val <= -0.4) return "Ultra Thin & Minimal";
    if (val < -0.1) return "Thin & Crisp";
    if (val <= 0.1) return "CamScanner Standard";
    if (val < 0.5) return "Medium";
    return "Bold";
  }

  function syncControlsFromParams(p) {
    DOM.modePills.forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.preset === p.mode);
    });

    const sw = typeof p.stroke_weight === "number" ? p.stroke_weight : 0.0;
    DOM.sliderStrokeWeight.value = sw;
    DOM.valStrokeWeight.textContent = getStrokeWeightText(sw);

    DOM.sliderBrightness.value = p.brightness;
    DOM.valBrightness.textContent = p.brightness;
    DOM.sliderContrast.value = p.contrast;
    DOM.valContrast.textContent = p.contrast;
    DOM.chkDeskew.checked = p.deskew;
  }

  // Preset Mode Buttons
  DOM.modePills.forEach((btn) => {
    btn.addEventListener("click", () => {
      const page = state.pages[state.activePageIndex];
      if (!page) return;
      page.params.mode = btn.dataset.preset;
      syncControlsFromParams(page.params);
      triggerPreviewUpdate(true);
    });
  });

  // Text Stroke Weight Slider
  DOM.sliderStrokeWeight.addEventListener("input", (e) => {
    const val = parseFloat(e.target.value);
    DOM.valStrokeWeight.textContent = getStrokeWeightText(val);
    const page = state.pages[state.activePageIndex];
    if (page) {
      page.params.stroke_weight = val;
      triggerPreviewUpdate();
    }
  });

  DOM.sliderBrightness.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    DOM.valBrightness.textContent = val;
    const page = state.pages[state.activePageIndex];
    if (page) {
      page.params.brightness = val;
      triggerPreviewUpdate();
    }
  });

  DOM.sliderContrast.addEventListener("input", (e) => {
    const val = parseInt(e.target.value, 10);
    DOM.valContrast.textContent = val;
    const page = state.pages[state.activePageIndex];
    if (page) {
      page.params.contrast = val;
      triggerPreviewUpdate();
    }
  });

  DOM.chkDeskew.addEventListener("change", (e) => {
    const page = state.pages[state.activePageIndex];
    if (page) {
      page.params.deskew = e.target.checked;
      triggerPreviewUpdate();
    }
  });

  // View Mode Toggles (Side-by-Side vs Enhanced Only vs Original)
  DOM.btnViewSplit.addEventListener("click", () => setPreviewViewMode("split"));
  DOM.btnViewEnhancedOnly.addEventListener("click", () => setPreviewViewMode("enhanced"));
  DOM.btnViewOriginalOnly.addEventListener("click", () => setPreviewViewMode("original"));

  function setPreviewViewMode(mode) {
    state.preview.viewMode = mode;
    DOM.btnViewSplit.classList.toggle("active", mode === "split");
    DOM.btnViewEnhancedOnly.classList.toggle("active", mode === "enhanced");
    DOM.btnViewOriginalOnly.classList.toggle("active", mode === "original");

    if (mode === "split") {
      DOM.boxBefore.style.display = "flex";
      DOM.boxAfter.style.display = "flex";
    } else if (mode === "enhanced") {
      DOM.boxBefore.style.display = "none";
      DOM.boxAfter.style.display = "flex";
    } else if (mode === "original") {
      DOM.boxBefore.style.display = "flex";
      DOM.boxAfter.style.display = "none";
    }
  }

  DOM.btnBackToCrop.addEventListener("click", () => switchView("viewCrop"));

  // Bottom Pages Carousel
  function renderPagesCarousel() {
    DOM.pagesCarousel.innerHTML = "";
    DOM.pagesCountLabel.textContent = `${state.pages.length} Page${
      state.pages.length > 1 ? "s" : ""
    }`;

    state.pages.forEach((p, idx) => {
      const card = document.createElement("div");
      card.className = `page-card ${idx === state.activePageIndex ? "active" : ""}`;
      card.dataset.index = idx;

      const img = document.createElement("img");
      img.className = "page-card-thumb";
      img.src = p.thumbUrl || p.previewUrl;
      img.alt = `Page ${idx + 1}`;

      const num = document.createElement("span");
      num.className = "page-card-num";
      num.textContent = `${idx + 1}`;

      const delBtn = document.createElement("button");
      delBtn.className = "page-card-del";
      delBtn.title = "Delete Page";
      delBtn.innerHTML = "&times;";
      delBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        deletePage(idx);
      });

      card.appendChild(img);
      card.appendChild(num);
      card.appendChild(delBtn);

      card.addEventListener("click", () => {
        state.activePageIndex = idx;
        renderPagesCarousel();
        if (state.currentView === "viewCrop") {
          loadActivePageForCropping();
        } else if (state.currentView === "viewEnhance") {
          loadActivePageForEnhancing();
        }
      });

      DOM.pagesCarousel.appendChild(card);
    });
  }

  function updatePageThumbnail(index, url) {
    if (state.pages[index]) {
      state.pages[index].thumbUrl = url;
      const card = DOM.pagesCarousel.children[index];
      if (card) {
        const thumb = card.querySelector(".page-card-thumb");
        if (thumb) thumb.src = url;
      }
    }
  }

  function deletePage(index) {
    state.pages.splice(index, 1);
    if (state.pages.length === 0) {
      DOM.pagesBar.style.display = "none";
      switchView("viewUpload");
      return;
    }
    if (state.activePageIndex >= state.pages.length) {
      state.activePageIndex = state.pages.length - 1;
    }
    renderPagesCarousel();
    if (state.currentView === "viewCrop") {
      loadActivePageForCropping();
    } else if (state.currentView === "viewEnhance") {
      loadActivePageForEnhancing();
    }
  }

  DOM.btnAddPage.addEventListener("click", () => DOM.fileInput.click());

  // Export Modal Handlers
  DOM.btnOpenExport.addEventListener("click", openExportModal);
  DOM.btnCloseExportModal.addEventListener("click", closeExportModal);
  DOM.btnCancelExport.addEventListener("click", closeExportModal);

  function openExportModal() {
    DOM.exportModal.classList.add("active");
  }

  function closeExportModal() {
    DOM.exportModal.classList.remove("active");
  }

  DOM.exportFormat.addEventListener("change", (e) => {
    const isPdf = e.target.value === "pdf";
    DOM.pageSizeGroup.style.display = isPdf ? "flex" : "none";
    DOM.exportBtnText.textContent = isPdf ? "Download PDF" : "Download Now";
  });

  DOM.btnDoExport.addEventListener("click", async () => {
    if (state.pages.length === 0) return;
    DOM.btnDoExport.disabled = true;
    DOM.exportBtnText.textContent = "Exporting...";

    try {
      const payload = {
        pages: state.pages.map((p) => ({
          image_id: p.imageId,
          corners: p.corners,
          params: p.params,
        })),
        format: DOM.exportFormat.value,
        page_size: DOM.exportPageSize.value,
        jpeg_quality: parseInt(DOM.exportQuality.value, 10),
        ocr: DOM.chkExportOcr.checked,
      };

      const res = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Export failed");
      }

      const blob = await res.blob();
      const ext = DOM.exportFormat.value === "pdf" ? "pdf" : DOM.exportFormat.value === "zip" ? "zip" : "png";
      const filename = `DocScan_${Date.now()}.${ext}`;

      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);

      closeExportModal();
      showToast("Document downloaded successfully!", "success");
    } catch (err) {
      showToast(`Export error: ${err.message}`, "error");
    } finally {
      DOM.btnDoExport.disabled = false;
      DOM.exportBtnText.textContent = "Download Now";
    }
  });

})();
