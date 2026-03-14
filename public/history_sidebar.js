(() => {
  const THREADS_KEY = "docqa_threads_v1";
  const ACTIVE_KEY = "docqa_active_thread_v1";
  const MAX_THREADS = 30;
  const TITLE_LIMIT = 72;
  const DESKTOP_MIN_WIDTH = 1000;

  let rootEl = null;
  let listEl = null;
  let emptyEl = null;
  let lastCaptured = { text: "", ts: 0 };
  let suppressGlobalNewUntil = 0;
  let lastNewThreadAt = 0;
  let nativeMode = false;

  const now = () => Date.now();
  const threadTag = (id) => `<!--THREAD:${id}-->`;
  const normalizeLabel = (text) => (text || "").replace(/\s+/g, " ").trim().toLowerCase();

  const NATIVE_HISTORY_LABELS = [
    "new chat",
    "create new chat",
    "yeni sohbet",
    "thread history",
    "chat history"
  ];

  const shorten = (text, limit = TITLE_LIMIT) => {
    const clean = (text || "").replace(/\s+/g, " ").trim();
    if (clean.length <= limit) return clean;
    return `${clean.slice(0, Math.max(0, limit - 3))}...`;
  };

  const normalizeTitle = (text) => (text || "").replace(/\s+/g, " ").trim().toLowerCase();

  const sanitizeThreads = (threads) => {
    const out = [];
    const seenIds = new Set();
    let keptEmptyThread = false;
    for (const t of sortThreads(Array.isArray(threads) ? threads : [])) {
      if (!t || typeof t.id !== "string" || typeof t.title !== "string") continue;
      if (seenIds.has(t.id)) continue;
      seenIds.add(t.id);
      const isPlaceholder = normalizeTitle(t.title) === "yeni sohbet";
      if (isPlaceholder) {
        if (keptEmptyThread) continue;
        keptEmptyThread = true;
      }
      out.push(t);
      if (out.length >= MAX_THREADS) break;
    }
    return out;
  };

  const loadThreads = () => {
    try {
      const raw = localStorage.getItem(THREADS_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed.filter((t) => t && typeof t.id === "string" && typeof t.title === "string");
    } catch {
      return [];
    }
  };

  const saveThreads = (threads) => {
    try {
      localStorage.setItem(THREADS_KEY, JSON.stringify(sanitizeThreads(threads || [])));
    } catch {
      // no-op
    }
  };

  const getActiveId = () => {
    try {
      return localStorage.getItem(ACTIVE_KEY) || "";
    } catch {
      return "";
    }
  };

  const setActiveId = (id) => {
    try {
      localStorage.setItem(ACTIVE_KEY, id || "");
    } catch {
      // no-op
    }
  };

  const sortThreads = (threads) =>
    [...threads].sort((a, b) => Number(b.updatedAt || 0) - Number(a.updatedAt || 0));

  const createThread = (seedTitle = "Yeni sohbet") => ({
    id: `t_${now()}_${Math.random().toString(16).slice(2, 8)}`,
    title: shorten(seedTitle),
    updatedAt: now(),
  });

  const ensureStyles = () => {
    if (document.getElementById("docqa-history-style")) return;
    const style = document.createElement("style");
    style.id = "docqa-history-style";
    style.textContent = `
      #docqa-history-panel {
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        width: 268px;
        background: #11161d;
        border-right: 1px solid rgba(255,255,255,0.08);
        z-index: 1400;
        display: flex;
        flex-direction: column;
        color: #e5e7eb;
      }
      #docqa-history-head {
        padding: 12px;
        border-bottom: 1px solid rgba(255,255,255,0.08);
      }
      #docqa-history-title {
        font-size: 14px;
        font-weight: 700;
        margin: 0;
      }
      #docqa-history-list {
        margin: 0;
        padding: 8px;
        list-style: none;
        overflow: auto;
        flex: 1;
      }
      .docqa-history-item {
        border: 1px solid transparent;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 6px;
        background: #1a2029;
        color: #e5e7eb;
        font-size: 13px;
        cursor: pointer;
        transition: border-color .14s ease, background .14s ease;
        user-select: none;
      }
      .docqa-history-item.active {
        border-color: rgba(20,184,166,0.7);
        background: #202a33;
      }
      .docqa-history-item:hover {
        border-color: rgba(255,255,255,0.22);
      }
      .docqa-history-item:focus-visible {
        outline: 2px solid rgba(20,184,166,0.85);
        outline-offset: 2px;
      }
      #docqa-history-empty {
        color: #9ca3af;
        font-size: 13px;
        padding: 12px;
      }
      @media (min-width: ${DESKTOP_MIN_WIDTH}px) {
        #root {
          margin-left: 268px !important;
          width: calc(100% - 268px) !important;
        }
      }
      @media (max-width: ${DESKTOP_MIN_WIDTH - 1}px) {
        #docqa-history-panel {
          display: none;
        }
      }
    `;
    document.head.appendChild(style);
  };

  const ensureNativeResetStyle = () => {
    if (document.getElementById("docqa-history-native-reset")) return;
    const style = document.createElement("style");
    style.id = "docqa-history-native-reset";
    style.textContent = `
      #docqa-history-panel {
        display: none !important;
      }
      #root {
        margin-left: 0 !important;
        width: 100% !important;
      }
    `;
    document.head.appendChild(style);
  };

  const hasNativeHistoryUI = () => {
    const controls = Array.from(document.querySelectorAll('button, [role="button"], a'));
    for (const node of controls) {
      if (rootEl && rootEl.contains(node)) continue;
      const label = normalizeLabel(
        `${node.getAttribute("aria-label") || ""} ${node.getAttribute("title") || ""} ${node.textContent || ""}`
      );
      if (NATIVE_HISTORY_LABELS.some((needle) => label.includes(needle))) {
        return true;
      }
    }

    const landmarks = Array.from(document.querySelectorAll("nav, aside, [data-testid], [class]"));
    for (const node of landmarks) {
      if (rootEl && rootEl.contains(node)) continue;
      const hint = normalizeLabel(
        `${node.getAttribute("data-testid") || ""} ${node.getAttribute("class") || ""} ${node.getAttribute("aria-label") || ""}`
      );
      if (hint.includes("docqa-history")) continue;
      if (hint.includes("thread") || hint.includes("history")) {
        return true;
      }
    }
    return false;
  };

  const enableNativeMode = () => {
    nativeMode = true;
    if (rootEl) {
      rootEl.remove();
      rootEl = null;
      listEl = null;
      emptyEl = null;
    }
    ensureNativeResetStyle();
  };

  const render = () => {
    if (nativeMode) return;
    if (!listEl || !emptyEl) return;
    const threads = sortThreads(loadThreads());
    const activeId = getActiveId();
    listEl.innerHTML = "";

    if (!threads.length) {
      emptyEl.style.display = "block";
      emptyEl.textContent = "Henuz sohbet yok.";
      return;
    }

    emptyEl.style.display = "none";
    for (const t of threads) {
      const li = document.createElement("li");
      li.className = `docqa-history-item${t.id === activeId ? " active" : ""}`;
      li.textContent = t.title || "Yeni sohbet";
      li.title = t.title || "Yeni sohbet";
      li.dataset.threadId = t.id;
      li.tabIndex = 0;
      li.setAttribute("role", "button");
      li.setAttribute("aria-label", `Sohbeti ac: ${t.title || "Yeni sohbet"}`);
      listEl.appendChild(li);
    }
  };

  const openNewChatFromUI = () => {
    suppressGlobalNewUntil = now() + 1200;
    const buttons = Array.from(document.querySelectorAll('button, [role="button"], a'));
    for (const btn of buttons) {
      const label = `${btn.getAttribute("aria-label") || ""} ${btn.getAttribute("title") || ""} ${btn.textContent || ""}`.toLowerCase();
      if (label.includes("new chat") || label.includes("yeni sohbet")) {
        btn.click();
        return true;
      }
    }
    return false;
  };

  const ensureActiveThread = (seedTitle = "Yeni sohbet") => {
    let activeId = getActiveId();
    const threads = loadThreads();
    const hasActive = threads.some((t) => t.id === activeId);
    if (hasActive && activeId) return activeId;
    const t = createThread(seedTitle);
    const next = [t, ...threads].slice(0, MAX_THREADS);
    saveThreads(next);
    setActiveId(t.id);
    return t.id;
  };

  const composePayload = (threadId, text) => `${threadTag(threadId)} ${text || ""}`.trim();

  const setComposerValue = (textarea, rawText) => {
    const proto = Object.getPrototypeOf(textarea);
    const desc = Object.getOwnPropertyDescriptor(proto, "value");
    if (desc && typeof desc.set === "function") {
      desc.set.call(textarea, rawText);
    } else {
      textarea.value = rawText;
    }
    textarea.dispatchEvent(new InputEvent("input", { bubbles: true, data: rawText, inputType: "insertText" }));
    textarea.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const submitComposer = (textarea) => {
    const form = textarea.closest("form");
    if (!form) return false;

    const submitBtn = form.querySelector('button[type="submit"], button[aria-label*="Send" i], button[aria-label*="Gonder" i], button[title*="Send" i], button[title*="Gonder" i]');
    if (submitBtn && !submitBtn.disabled) {
      submitBtn.click();
      return true;
    }

    if (typeof form.requestSubmit === "function") {
      form.requestSubmit();
      return true;
    }
    return false;
  };

  const sendProgrammaticMessage = (rawText) => {
    const textarea = document.querySelector("textarea");
    if (!textarea) return false;
    setComposerValue(textarea, rawText);
    return submitComposer(textarea);
  };

  const sendProgrammaticMessageWithRetry = (rawText, remaining = 12) => {
    if (sendProgrammaticMessage(rawText)) return;
    if (remaining <= 0) return;
    setTimeout(() => sendProgrammaticMessageWithRetry(rawText, remaining - 1), 180);
  };

  const openThreadInChat = (threadId) => {
    if (!threadId) return;
    setActiveId(threadId);
    render();
    suppressGlobalNewUntil = now() + 1200;
    const opened = openNewChatFromUI();
    const delay = opened ? 240 : 40;
    setTimeout(() => {
      sendProgrammaticMessageWithRetry(`/open_thread ${threadId}`);
    }, delay);
  };

  const startNewThread = () => {
    const ts = now();
    if (ts - lastNewThreadAt < 700) return;
    lastNewThreadAt = ts;

    const threads = loadThreads();
    const t = createThread();
    const next = [t, ...threads].slice(0, MAX_THREADS);
    saveThreads(next);
    setActiveId(t.id);
    render();
  };

  const upsertThreadFromMessage = (text) => {
    const content = shorten(text || "");
    if (!content) return;

    const ts = now();
    if (content === lastCaptured.text && ts - lastCaptured.ts < 1200) return;
    lastCaptured = { text: content, ts };

    const threads = loadThreads();
    let activeId = getActiveId();
    let active = threads.find((t) => t.id === activeId);

    if (!active) {
      active = createThread(content);
      activeId = active.id;
      threads.unshift(active);
    } else {
      active.title = content;
      active.updatedAt = ts;
      const idx = threads.findIndex((t) => t.id === activeId);
      if (idx >= 0) {
        threads.splice(idx, 1);
      }
      threads.unshift(active);
    }

    saveThreads(threads);
    setActiveId(activeId);
    render();
  };

  const bindComposer = () => {
    if (nativeMode) return;
    const textarea = document.querySelector("textarea");
    if (!textarea || textarea.dataset.docqaHistoryBound === "1") return;

    textarea.dataset.docqaHistoryBound = "1";
    const getPlainInput = () => {
      const raw = (textarea.value || "").trim();
      return raw.replace(/^<!--THREAD:[^>]+-->\s*/i, "").trim();
    };

    const injectThreadMarkerIfNeeded = () => {
      const raw = (textarea.value || "").trim();
      if (!raw) return;
      if (/^<!--THREAD:[^>]+-->/i.test(raw)) return;
      const plain = getPlainInput();
      if (!plain) return;
      const activeId = ensureActiveThread(plain);
      textarea.value = composePayload(activeId, plain);
      textarea.dispatchEvent(new Event("input", { bubbles: true }));
    };

    const capture = () => {
      const plain = getPlainInput();
      if (!plain || plain.startsWith("/")) return;
      upsertThreadFromMessage(plain);
    };

    textarea.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" && !ev.shiftKey) {
        capture();
        injectThreadMarkerIfNeeded();
      }
    });

    const form = textarea.closest("form");
    if (form && form.dataset.docqaHistoryBound !== "1") {
      form.dataset.docqaHistoryBound = "1";
      form.addEventListener("submit", () => {
        capture();
        injectThreadMarkerIfNeeded();
      });
    }
  };

  const ensurePanel = () => {
    if (nativeMode) return;
    if (rootEl) return;
    rootEl = document.createElement("aside");
    rootEl.id = "docqa-history-panel";
    rootEl.innerHTML = `
      <div id="docqa-history-head">
        <div id="docqa-history-title">Gecmis Sohbetler</div>
      </div>
      <div id="docqa-history-empty">Henuz sohbet yok.</div>
      <ul id="docqa-history-list"></ul>
    `;
    document.body.appendChild(rootEl);
    listEl = rootEl.querySelector("#docqa-history-list");
    emptyEl = rootEl.querySelector("#docqa-history-empty");

    rootEl.addEventListener("click", (ev) => {
      const item = ev.target && ev.target.closest ? ev.target.closest(".docqa-history-item") : null;
      if (!item) return;
      const id = item.dataset.threadId || "";
      if (!id) return;
      openThreadInChat(id);
    });

    rootEl.addEventListener("keydown", (ev) => {
      if (ev.key !== "Enter" && ev.key !== " ") return;
      const item = ev.target && ev.target.closest ? ev.target.closest(".docqa-history-item") : null;
      if (!item) return;
      ev.preventDefault();
      const id = item.dataset.threadId || "";
      if (!id) return;
      openThreadInChat(id);
    });
  };

  const bindGlobalClicks = () => {
    if (nativeMode) return;
    if (document.body.dataset.docqaHistoryGlobalBound === "1") return;
    document.body.dataset.docqaHistoryGlobalBound = "1";

    document.addEventListener("click", (ev) => {
      const btn = ev.target && ev.target.closest ? ev.target.closest("button") : null;
      if (!btn) return;
      if (rootEl && rootEl.contains(btn)) return;
      if (now() < suppressGlobalNewUntil) return;
      const label = `${btn.getAttribute("aria-label") || ""} ${btn.getAttribute("title") || ""} ${btn.textContent || ""}`.toLowerCase();
      if (label.includes("new chat") || label.includes("yeni sohbet")) {
        startNewThread();
      }
    });
  };

  const init = () => {
    if (hasNativeHistoryUI()) {
      enableNativeMode();
      return;
    }
    ensureStyles();
    ensurePanel();
    bindGlobalClicks();
    bindComposer();
    saveThreads(loadThreads());
    render();

    // SPA surfaces are dynamic; keep bindings fresh.
    setInterval(() => {
      if (!nativeMode && hasNativeHistoryUI()) {
        enableNativeMode();
        return;
      }
      if (nativeMode) return;
      bindComposer();
      render();
    }, 1200);
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
