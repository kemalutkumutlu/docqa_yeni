(() => {
  const SOURCE_UI = "docqa-settings-ui";
  const SOURCE_SERVER = "docqa-settings-server";
  const DESKTOP_MIN_WIDTH = 1280;
  const PANEL_WIDTH = 430;

  const BASIC_FIELDS = [
    "runtime_preset",
    "processing_mode",
    "ocr_enabled",
    "llm_provider",
    "generation_model",
  ];

  const ADVANCED_FIELDS = [
    "embedding_model",
    "embedding_device",
    "pdf_text_backend",
    "vlm_mode",
    "ocr_backend",
    "visual_chunk_level",
    "table_structure_enabled",
    "multimodal_answer_mode",
    "visual_region_source",
    "visual_detector_backend",
    "table_structure_backend",
    "vlm_provider",
    "vlm_max_pages",
  ];

  let appState = {
    saved: null,
    draft: null,
    options: null,
    presetDefaults: {},
    summaries: { applied: "", fallback: "", documents: "" },
    activeTab: "basic",
  };

  const SUMMARY_STORAGE_KEY = "docqa-summary-collapsed";

  let panel = null;
  let basicTab = null;
  let advancedTab = null;
  let basicFields = null;
  let advancedFields = null;
  let draftSummary = null;
  let appliedSummary = null;
  let fallbackSummary = null;
  let documentsSummary = null;
  let applyButton = null;
  let resetButton = null;
  let summaryToggle = null;
  let summaryGrid = null;
  let requestTimer = null;
  let layoutFrame = null;
  let layoutObserver = null;

  const clone = (value) => JSON.parse(JSON.stringify(value));
  const normalizeText = (value) => String(value || "").replace(/\s+/g, " ").trim().toLowerCase();

  const sendToBackend = (action, payload = {}) => {
    window.postMessage({ source: SOURCE_UI, action, payload }, window.location.origin);
  };

  const normalizeToggle = (value) => {
    const text = String(value || "").trim().toLowerCase();
    return ["1", "true", "yes", "y", "on", "enabled"].includes(text) ? "on" : "off";
  };

  const getGenerationOptions = (provider) => {
    const all = (appState.options && appState.options.generation_models) || {};
    return all[String(provider || "").trim().toLowerCase()] || [];
  };

  const resolveDraft = () => appState.draft || appState.saved || {};

  const computeDisabledState = (draft) => {
    const processingMode = draft.processing_mode || "classic";
    const visualChunkLevel = draft.visual_chunk_level || "page";
    const visualRegionSource = draft.visual_region_source || "heuristic";
    const llmProvider = draft.llm_provider || "gemini";
    const embeddingModel = String(draft.embedding_model || "").trim().toLowerCase();
    const ocrActive = normalizeToggle(draft.ocr_enabled) === "on";
    const multimodalActive = processingMode === "multimodal";
    const regionModeActive = multimodalActive && visualChunkLevel === "region";
    const detectorModeActive = regionModeActive && visualRegionSource === "detector";
    const tableStageAvailable = multimodalActive;
    const tableActive = tableStageAvailable && normalizeToggle(draft.table_structure_enabled) === "on";
    const vlmActive = (draft.vlm_mode || "auto") !== "off";
    const llmActive = llmProvider !== "none";
    const llmSupportsMultimodalAnswer = llmProvider === "gemini";
    const embeddingDeviceRelevant = !embeddingModel.startsWith("gemini-embedding-");

    return {
      generation_model: !llmActive,
      ocr_backend: !ocrActive,
      visual_chunk_level: !multimodalActive,
      table_structure_enabled: !tableStageAvailable,
      multimodal_answer_mode: !(multimodalActive && llmSupportsMultimodalAnswer),
      visual_region_source: !regionModeActive,
      visual_detector_backend: !detectorModeActive,
      table_structure_backend: !tableActive,
      vlm_provider: !vlmActive,
      vlm_max_pages: !vlmActive,
      embedding_device: !embeddingDeviceRelevant,
    };
  };

  const summaryLines = (draft) => {
    const disabled = computeDisabledState(draft);
    const layoutLine = disabled.visual_detector_backend
      ? disabled.visual_region_source
        ? "Layout detector: kapali"
        : "Layout detector: detector backend secimi bekliyor"
      : `Layout detector: ${draft.visual_detector_backend || "none"}`;
    const tableLine = disabled.table_structure_backend
      ? "Table stage: kapali"
      : `Table stage: ${draft.table_structure_backend || "auto"}`;
    const generationLine = draft.llm_provider === "none"
      ? "Generation: extractive"
      : `Generation: ${draft.llm_provider || "gemini"} / ${draft.generation_model || "-"}`;
    const vlmLine = disabled.vlm_provider
      ? "VLM: kapali"
      : `VLM: ${(draft.vlm_provider || "gemini")} / ${(draft.vlm_mode || "auto")} / max_pages=${draft.vlm_max_pages ?? 25}`;
    return [
      `PDF text: ${draft.pdf_text_backend || "pymupdf"}`,
      `OCR: ${draft.ocr_enabled || "on"} / ${draft.ocr_backend || "docai"}`,
      `Processing: ${draft.processing_mode || "classic"}`,
      layoutLine,
      tableLine,
      generationLine,
      vlmLine,
    ].join("\n");
  };

  const fieldLabel = {
    runtime_preset: "Runtime Preset",
    processing_mode: "Processing Mode",
    ocr_enabled: "OCR",
    llm_provider: "LLM Provider",
    generation_model: "Generation Model",
    embedding_model: "Embedding Model",
    embedding_device: "Embedding Device",
    pdf_text_backend: "PDF Text Backend",
    vlm_mode: "VLM Mode",
    ocr_backend: "OCR Backend",
    visual_chunk_level: "Visual Chunk Level",
    table_structure_enabled: "Table Structure",
    multimodal_answer_mode: "Multimodal Answer",
    visual_region_source: "Visual Region Source",
    visual_detector_backend: "Visual Detector Backend",
    table_structure_backend: "Table Structure Backend",
    vlm_provider: "VLM Provider",
    vlm_max_pages: "VLM Max Pages",
  };

  const fieldDescription = {
    runtime_preset: "Hazir ayar kombinasyonu.",
    processing_mode: "Classic text-first veya multimodal akis.",
    ocr_enabled: "OCR katmanini acip kapatir.",
    llm_provider: "Cevap uretim backend'i.",
    generation_model: "Secili provider icin aktif model.",
    embedding_model: "Remote veya local embedding secimi.",
    embedding_device: "Sadece local embedding icin anlamli.",
    pdf_text_backend: "PDF metin çıkarma: auto=Docling bin varsa kullan, pymupdf=sadece PyMuPDF, docling=Docling zorunlu.",
    vlm_mode: "Visual extraction davranisi.",
    ocr_backend: "Secili OCR backend.",
    visual_chunk_level: "Visual granularity.",
    table_structure_enabled: "Yapisal table extraction.",
    multimodal_answer_mode: "Gemini icin visual evidence generation davranisi.",
    visual_region_source: "Region proposal kaynagi.",
    visual_detector_backend: "Aktif detector backend.",
    table_structure_backend: "Table stage backend secimi.",
    vlm_provider: "Visual model provider.",
    vlm_max_pages: "Belge basina VLM sayfa limiti.",
  };

  const ensureLayoutStyle = () => {
    if (document.getElementById("docqa-settings-layout-style")) return;
    const style = document.createElement("style");
    style.id = "docqa-settings-layout-style";
    style.textContent = `
      @media (min-width: ${DESKTOP_MIN_WIDTH}px) {
        #root {
          margin-right: var(--docqa-settings-reserved, ${PANEL_WIDTH}px) !important;
          width: auto !important;
        }
      }

      @media (max-width: ${DESKTOP_MIN_WIDTH - 1}px) {
        #root {
          margin-right: 0 !important;
          width: auto !important;
        }
      }
    `;
    document.head.appendChild(style);
  };

  const findRightRailWidth = () => {
    const minWidth = 220;
    const minHeight = Math.min(window.innerHeight * 0.35, 260);
    let maxWidth = 0;

    for (const el of Array.from(document.body.querySelectorAll("*"))) {
      if (!(el instanceof HTMLElement)) continue;
      if (el === panel || panel?.contains(el)) continue;
      if (el.id === "docqa-history-panel") continue;

      const style = window.getComputedStyle(el);
      if (!["fixed", "sticky"].includes(style.position)) continue;
      if (style.display === "none" || style.visibility === "hidden") continue;

      const rect = el.getBoundingClientRect();
      if (rect.width < minWidth || rect.height < minHeight) continue;
      if (rect.right < window.innerWidth - 4) continue;
      if (rect.left < window.innerWidth * 0.55) continue;

      const text = normalizeText(el.textContent);
      const likelySidebar =
        text.includes("belge durumu") ||
        text.includes("oturum durumu") ||
        text.includes("yüklenen belgeler") ||
        text.includes("yuklenen belgeler");

      if (!likelySidebar && rect.width < 280) continue;
      maxWidth = Math.max(maxWidth, Math.ceil(rect.width));
    }

    return maxWidth;
  };

  const applyLayoutReservation = () => {
    const root = document.documentElement;
    if (window.innerWidth < DESKTOP_MIN_WIDTH) {
      root.style.setProperty("--docqa-settings-offset", "0px");
      root.style.setProperty("--docqa-settings-reserved", "0px");
      return;
    }

    const rightRailWidth = findRightRailWidth();
    root.style.setProperty("--docqa-settings-offset", `${rightRailWidth}px`);
    root.style.setProperty("--docqa-settings-reserved", `${PANEL_WIDTH + rightRailWidth}px`);
  };

  const scheduleLayoutReservation = () => {
    if (layoutFrame !== null) {
      window.cancelAnimationFrame(layoutFrame);
    }
    layoutFrame = window.requestAnimationFrame(() => {
      layoutFrame = null;
      applyLayoutReservation();
    });
  };

  const ensureLayoutObserver = () => {
    if (layoutObserver) return;
    layoutObserver = new MutationObserver(() => {
      scheduleLayoutReservation();
    });
    layoutObserver.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ["class", "style", "data-state", "aria-hidden"],
    });
    window.addEventListener("resize", scheduleLayoutReservation);
  };

  const ensureShell = () => {
    if (panel) return;
    ensureLayoutStyle();
    ensureLayoutObserver();

    panel = document.createElement("aside");
    panel.id = "docqa-settings-panel";
    panel.innerHTML = `
      <div class="docqa-settings-head">
        <div>
          <div class="docqa-settings-kicker">Runtime</div>
          <h2>Settings</h2>
        </div>
        <div class="docqa-settings-tabbar">
          <button type="button" data-tab="basic" class="active">Basic</button>
          <button type="button" data-tab="advanced">Advanced</button>
        </div>
      </div>
      <div class="docqa-settings-summary">
        <div class="docqa-settings-summary-header">
          <span class="docqa-settings-summary-label">Pipeline Status</span>
          <button type="button" id="docqa-settings-summary-toggle">▲ Gizle</button>
        </div>
        <div class="docqa-settings-summary-grid" id="docqa-settings-summary-grid">
          <div class="docqa-settings-card">
            <div class="docqa-settings-card-title">Current Draft</div>
            <pre id="docqa-settings-draft-summary"></pre>
          </div>
          <div class="docqa-settings-card">
            <div class="docqa-settings-card-title">Applied Pipeline</div>
            <pre id="docqa-settings-applied-summary"></pre>
          </div>
          <div class="docqa-settings-card">
            <div class="docqa-settings-card-title">Fallback Notes</div>
            <pre id="docqa-settings-fallback-summary"></pre>
          </div>
          <div class="docqa-settings-card">
            <div class="docqa-settings-card-title">Document Context</div>
            <pre id="docqa-settings-documents-summary"></pre>
          </div>
        </div>
      </div>
      <div class="docqa-settings-sections">
        <section id="docqa-settings-basic" class="docqa-settings-section active"></section>
        <section id="docqa-settings-advanced" class="docqa-settings-section"></section>
      </div>
      <div class="docqa-settings-actions">
        <button type="button" id="docqa-settings-reset">Reset</button>
        <button type="button" id="docqa-settings-apply" class="accent">Apply</button>
      </div>
    `;
    document.body.appendChild(panel);

    basicTab = panel.querySelector('[data-tab="basic"]');
    advancedTab = panel.querySelector('[data-tab="advanced"]');
    basicFields = panel.querySelector("#docqa-settings-basic");
    advancedFields = panel.querySelector("#docqa-settings-advanced");
    draftSummary = panel.querySelector("#docqa-settings-draft-summary");
    appliedSummary = panel.querySelector("#docqa-settings-applied-summary");
    fallbackSummary = panel.querySelector("#docqa-settings-fallback-summary");
    documentsSummary = panel.querySelector("#docqa-settings-documents-summary");
    applyButton = panel.querySelector("#docqa-settings-apply");
    resetButton = panel.querySelector("#docqa-settings-reset");
    summaryToggle = panel.querySelector("#docqa-settings-summary-toggle");
    summaryGrid = panel.querySelector("#docqa-settings-summary-grid");

    // Restore summary collapse state from localStorage (default: collapsed)
    const storedCollapsed = localStorage.getItem(SUMMARY_STORAGE_KEY);
    const startCollapsed = storedCollapsed === null ? true : storedCollapsed === "1";

    const setSummaryCollapsed = (collapsed) => {
      summaryGrid.style.display = collapsed ? "none" : "";
      summaryToggle.textContent = collapsed ? "▼ Göster" : "▲ Gizle";
    };

    setSummaryCollapsed(startCollapsed);

    summaryToggle.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      const nowCollapsed = summaryGrid.style.display !== "none";
      setSummaryCollapsed(nowCollapsed);
      localStorage.setItem(SUMMARY_STORAGE_KEY, nowCollapsed ? "1" : "0");
    }, true);

    panel.querySelectorAll("[data-tab]").forEach((button) => {
      button.addEventListener("click", () => {
        appState.activeTab = button.dataset.tab || "basic";
        syncTabs();
      });
    });

    applyButton.addEventListener("click", () => {
      if (!appState.draft) return;
      sendToBackend("apply", normalizeForBackend(appState.draft));
    });

    resetButton.addEventListener("click", () => {
      if (!appState.saved) return;
      appState.draft = clone(appState.saved);
      render();
    });

    scheduleLayoutReservation();
  };

  const normalizeForBackend = (draft) => ({
    runtime_preset: draft.runtime_preset,
    processing_mode: draft.processing_mode,
    ocr_enabled: draft.ocr_enabled,
    ocr_backend: draft.ocr_backend,
    llm_provider: draft.llm_provider,
    generation_model: draft.generation_model,
    embedding_model: draft.embedding_model,
    embedding_device: draft.embedding_device,
    vlm_mode: draft.vlm_mode,
    vlm_provider: draft.vlm_provider,
    vlm_max_pages: Number(draft.vlm_max_pages || 0),
    visual_chunk_level: draft.visual_chunk_level,
    table_structure_enabled: draft.table_structure_enabled,
    multimodal_answer_mode: draft.multimodal_answer_mode,
    visual_region_source: draft.visual_region_source,
    visual_detector_backend: draft.visual_detector_backend,
    table_structure_backend: draft.table_structure_backend,
    pdf_text_backend: draft.pdf_text_backend,
  });

  const syncTabs = () => {
    if (!panel) return;
    const basicActive = appState.activeTab === "basic";
    basicTab.classList.toggle("active", basicActive);
    advancedTab.classList.toggle("active", !basicActive);
    basicFields.classList.toggle("active", basicActive);
    advancedFields.classList.toggle("active", !basicActive);
  };

  const createField = (field, draft, disabledMap) => {
    const wrapper = document.createElement("label");
    wrapper.className = "docqa-settings-field";
    wrapper.dataset.field = field;

    const title = document.createElement("span");
    title.className = "docqa-settings-label";
    title.textContent = fieldLabel[field] || field;
    wrapper.appendChild(title);

    let control;
    if (field === "vlm_max_pages") {
      control = document.createElement("input");
      control.type = "range";
      control.min = String((appState.options.vlm_max_pages && appState.options.vlm_max_pages.min) || 0);
      control.max = String((appState.options.vlm_max_pages && appState.options.vlm_max_pages.max) || 200);
      control.step = "1";
      control.value = String(draft[field] ?? 25);
    } else {
      control = document.createElement("select");
      const values = field === "generation_model"
        ? getGenerationOptions(draft.llm_provider)
        : appState.options[field] || [];
      values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        control.appendChild(option);
      });
      if (field === "generation_model" && values.length === 0) {
        const option = document.createElement("option");
        option.value = "";
        option.textContent = "-";
        control.appendChild(option);
      }
      control.value = String(draft[field] ?? "");
    }

    control.disabled = Boolean(disabledMap[field]);
    control.addEventListener("input", () => handleFieldChange(field, control.value));
    control.addEventListener("change", () => handleFieldChange(field, control.value));
    wrapper.appendChild(control);

    if (field === "vlm_max_pages") {
      const value = document.createElement("span");
      value.className = "docqa-settings-range-value";
      value.textContent = String(draft[field] ?? 25);
      control.addEventListener("input", () => {
        value.textContent = String(control.value);
      });
      wrapper.appendChild(value);
    }

    const description = document.createElement("small");
    description.className = "docqa-settings-help";
    description.textContent = fieldDescription[field] || "";
    wrapper.appendChild(description);

    return wrapper;
  };

  const handleFieldChange = (field, rawValue) => {
    if (!appState.draft) return;
    const next = clone(appState.draft);
    const value = field === "vlm_max_pages" ? Number(rawValue) : rawValue;

    if (field === "runtime_preset") {
      next.runtime_preset = value;
      if (value !== "custom" && appState.presetDefaults[value]) {
        Object.assign(next, clone(appState.presetDefaults[value]));
        next.runtime_preset = value;
        next.generation_model = next.generation_model_choice || next.generation_model || "";
        delete next.generation_model_choice;
        delete next.embedding_model_choice;
      }
    } else {
      next[field] = value;
      if (next.runtime_preset && next.runtime_preset !== "custom") {
        next.runtime_preset = "custom";
      }
    }

    if (field === "llm_provider") {
      const models = getGenerationOptions(value);
      next.generation_model = models[0] || "";
    }
    if (field === "embedding_model" && String(value).startsWith("gemini-embedding-")) {
      next.embedding_device = next.embedding_device || "auto";
    }
    appState.draft = next;
    render();
  };

  const renderFields = (container, fields, draft, disabledMap) => {
    container.innerHTML = "";
    fields.forEach((field) => {
      container.appendChild(createField(field, draft, disabledMap));
    });
  };

  const render = () => {
    if (!appState.saved || !appState.options) return;
    ensureShell();

    const draft = resolveDraft();
    const disabledMap = computeDisabledState(draft);

    renderFields(basicFields, BASIC_FIELDS, draft, disabledMap);
    renderFields(advancedFields, ADVANCED_FIELDS, draft, disabledMap);

    draftSummary.textContent = summaryLines(draft);
    appliedSummary.textContent = appState.summaries.applied || summaryLines(appState.saved);
    fallbackSummary.textContent = appState.summaries.fallback || "-";
    documentsSummary.textContent = appState.summaries.documents || "-";

    resetButton.disabled = JSON.stringify(draft) === JSON.stringify(appState.saved);
    syncTabs();
  };

  const receiveState = (payload) => {
    if (!payload || !payload.saved || !payload.options) return;
    appState.saved = {
      ...payload.saved,
      generation_model: payload.saved.generation_model_choice || payload.saved.generation_model || "",
      embedding_model: payload.saved.embedding_model_choice || payload.saved.embedding_model || "",
    };
    appState.draft = clone(appState.saved);
    appState.options = payload.options;
    appState.presetDefaults = payload.preset_defaults || {};
    appState.summaries = payload.summaries || { applied: "", fallback: "", documents: "" };
    render();
  };

  const handleWindowMessage = (event) => {
    const data = event && event.data;
    if (!data || data.source !== SOURCE_SERVER || data.kind !== "state") return;
    if (requestTimer) {
      clearInterval(requestTimer);
      requestTimer = null;
    }
    receiveState(data.payload);
  };

  const boot = () => {
    if (window.innerWidth < DESKTOP_MIN_WIDTH) return;
    ensureShell();
    window.addEventListener("message", handleWindowMessage);
    sendToBackend("request_state");
    requestTimer = window.setInterval(() => {
      if (!appState.saved) {
        sendToBackend("request_state");
      }
    }, 1500);
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
