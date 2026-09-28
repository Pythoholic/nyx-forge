import React from "react";
import ReactDOM from "react-dom/client";
import "@fontsource/jetbrains-mono/latin-400.css";
import "@fontsource/jetbrains-mono/latin-500.css";
import "@fontsource/jetbrains-mono/latin-600.css";
import "@fontsource/jetbrains-mono/latin-700.css";
import App from "./App";
import "./index.css";
import { applyNyxTheme, readNyxTheme } from "./nyxTheme";

applyNyxTheme(readNyxTheme());

const staleChunkPattern = /Failed to fetch dynamically imported module|Importing a module script failed|error loading dynamically imported module/i;
const recoverStaleBuild = (reason) => {
  const message = reason?.message ?? String(reason ?? "");
  if (!staleChunkPattern.test(message)) return;
  const key = "forge-stale-build-reload";
  const previous = Number(sessionStorage.getItem(key) ?? 0);
  if (Date.now() - previous < 15_000) return;
  sessionStorage.setItem(key, String(Date.now()));
  window.location.reload();
};
window.addEventListener("unhandledrejection", (event) => recoverStaleBuild(event.reason));
window.addEventListener("error", (event) => recoverStaleBuild(event.error ?? event.message));

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
