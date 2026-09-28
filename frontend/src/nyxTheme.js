export const NYX_THEME_KEY = "nyxforge.theme";

export const NYX_THEMES = Object.freeze([
  { id: "solar", label: "Solar", color: "#f5d90a" },
  { id: "signal", label: "Signal", color: "#00e08a" },
  { id: "flux", label: "Flux", color: "#35c6f4" },
  { id: "plasma", label: "Plasma", color: "#8b7cf6" },
  { id: "ember", label: "Ember", color: "#ff4d5d" },
]);

export function readNyxTheme() {
  let saved = null;
  try {
    saved = window.localStorage.getItem(NYX_THEME_KEY)?.toLowerCase();
  } catch {
    // Storage can be unavailable in hardened/private browser contexts.
  }
  return NYX_THEMES.some((theme) => theme.id === saved) ? saved : "solar";
}

export function applyNyxTheme(theme) {
  const next = NYX_THEMES.some((item) => item.id === theme) ? theme : "solar";
  document.documentElement.dataset.nyxTheme = next;
  try {
    window.localStorage.setItem(NYX_THEME_KEY, next);
  } catch {
    // The active theme still applies for this session when storage is blocked.
  }
  window.dispatchEvent(new CustomEvent("nyxforge:theme", { detail: { theme: next } }));
  return next;
}
