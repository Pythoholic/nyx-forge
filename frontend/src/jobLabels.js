// ForgeAI and ForgeIMG share one jobs table, so a single id sequence
// interleaves the two products and makes neither readable. The database id
// stays authoritative for routes and lookups; only the label is namespaced,
// using the same kind split the backend uses to pick a workspace.
const FORGEIMG_KINDS = new Set(["img2img", "character"]);
const FORGEVID_KINDS = new Set(["video_t2v", "video_i2v"]);
const FORGEBAT_KINDS = new Set(["batch_generate"]);

const WORKSPACE_LABELS = Object.freeze({
  forgeai: "ForgeAI",
  forgeimg: "ForgeIMG",
  forgevid: "ForgeVID",
  forgebat: "ForgeBAT",
});

export function workspaceForJobKind(kind) {
  if (FORGEIMG_KINDS.has(kind)) return "forgeimg";
  if (FORGEVID_KINDS.has(kind)) return "forgevid";
  if (FORGEBAT_KINDS.has(kind)) return "forgebat";
  return "forgeai";
}

export function workspaceLabel(workspace) {
  return WORKSPACE_LABELS[workspace] ?? "ForgeAI";
}

export function jobPrefix(kind) {
  if (FORGEIMG_KINDS.has(kind)) return "FIMG";
  if (FORGEVID_KINDS.has(kind)) return "FVID";
  if (FORGEBAT_KINDS.has(kind)) return "FBAT";
  return "FAI";
}

export function displayJobId(job) {
  return `${jobPrefix(job?.kind)}-${job?.id}`;
}

export function displayStyle(value) {
  if (!value?.style) return "";
  return value.style_variant ? `${value.style} · ${value.style_variant}` : value.style;
}

export function displayQuality(value) {
  const mode = value?.quality_mode
    ?? (value?.super_res ? "super" : value?.high_res ? "high" : "normal");
  return ["4k", "8k", "12k"].includes(mode) ? mode.toUpperCase() : `${mode.charAt(0).toUpperCase()}${mode.slice(1)}`;
}
