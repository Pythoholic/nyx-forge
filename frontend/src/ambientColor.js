import { useEffect, useState } from "react";

// Thumbnails are already loaded for the row preview, so the ambient colour
// costs one canvas read rather than another request. Results are cached by URL
// because the job list re-renders on every poll.
const cache = new Map();

function rgbToHsl(r, g, b) {
  const [rn, gn, bn] = [r / 255, g / 255, b / 255];
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const l = (max + min) / 2;
  if (max === min) return [0, 0, l];
  const d = max - min;
  const s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
  const h = max === rn
    ? ((gn - bn) / d + (gn < bn ? 6 : 0)) / 6
    : max === gn
      ? ((bn - rn) / d + 2) / 6
      : ((rn - gn) / d + 4) / 6;
  return [h, s, l];
}

function hslToRgb(h, s, l) {
  if (s === 0) {
    const v = Math.round(l * 255);
    return [v, v, v];
  }
  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;
  const channel = (t) => {
    let value = t;
    if (value < 0) value += 1;
    if (value > 1) value -= 1;
    if (value < 1 / 6) return p + (q - p) * 6 * value;
    if (value < 1 / 2) return q;
    if (value < 2 / 3) return p + (q - p) * (2 / 3 - value) * 6;
    return p;
  };
  return [channel(h + 1 / 3), channel(h), channel(h - 1 / 3)].map((v) => Math.round(v * 255));
}

function ambientFrom(image) {
  const canvas = document.createElement("canvas");
  canvas.width = 16;
  canvas.height = 16;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) return null;
  context.drawImage(image, 0, 0, 16, 16);
  let data;
  try {
    data = context.getImageData(0, 0, 16, 16).data;
  } catch {
    // A tainted canvas throws rather than returning anything useful.
    return null;
  }
  let r = 0;
  let g = 0;
  let b = 0;
  let weight = 0;
  for (let index = 0; index < data.length; index += 4) {
    const [pr, pg, pb] = [data[index], data[index + 1], data[index + 2]];
    // Letterboxing and blown highlights drag every average toward grey.
    if (Math.max(pr, pg, pb) < 24 || Math.min(pr, pg, pb) > 235) continue;
    // Weight by saturation so a colourful minority beats a grey majority. A
    // plain mean turns every image into the same muted brown.
    const w = 0.15 + rgbToHsl(pr, pg, pb)[1];
    r += pr * w;
    g += pg * w;
    b += pb * w;
    weight += w;
  }
  if (!weight) return null;
  const [h, s] = rgbToHsl(r / weight, g / weight, b / weight);
  return hslToRgb(h, Math.min(1, Math.max(s, 0.15) * 2.2), 0.55);
}

export function useAmbientColor(url) {
  const [color, setColor] = useState(() => (url ? cache.get(url) ?? null : null));

  useEffect(() => {
    if (!url) {
      setColor(null);
      return undefined;
    }
    if (cache.has(url)) {
      setColor(cache.get(url));
      return undefined;
    }
    let cancelled = false;
    const image = new Image();
    image.crossOrigin = "anonymous";
    image.onload = () => {
      const result = ambientFrom(image);
      cache.set(url, result);
      if (!cancelled) setColor(result);
    };
    image.onerror = () => {
      cache.set(url, null);
    };
    image.src = url;
    return () => {
      cancelled = true;
    };
  }, [url]);

  return color;
}
