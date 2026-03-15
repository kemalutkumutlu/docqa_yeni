(() => {
  const loadScript = (src, id) => {
    if (document.getElementById(id)) return;
    const script = document.createElement("script");
    script.id = id;
    script.src = src;
    script.defer = true;
    document.head.appendChild(script);
  };

  const boot = () => {
    loadScript("/public/history_sidebar.js", "docqa-history-script");
    loadScript("/public/settings_sidebar.js", "docqa-settings-script");
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot, { once: true });
  } else {
    boot();
  }
})();
