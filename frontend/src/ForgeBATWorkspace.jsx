import {
  Activity, ArrowLeft, ArrowRight, CalendarDays, Camera, Check, ChevronDown, CircleUserRound,
  Cloud, Eye, Heart, Image as ImageIcon, Images, Layers3, LoaderCircle, MapPin, RotateCw,
  Search, Shirt, Sparkles, Sun, Users, Waves, X, Zap,
} from "lucide-react";
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from "@headlessui/react";
import { useEffect, useMemo, useRef, useState } from "react";

import { getBatchOptions, setFavorite } from "./api";

const FACET_ICONS = {
  age: CalendarDays,
  heritage: Users,
  body_type: CircleUserRound,
  face_features: CircleUserRound,
  hair: Waves,
  clothing: Shirt,
  pose: Activity,
  expression: CircleUserRound,
  environment: MapPin,
  camera: Camera,
  lighting: Sun,
  mood: Cloud,
  orientation: Camera,
};

export function readableId(id) {
  const tail = String(id).split(".").at(-1);
  return tail.replaceAll("_", " ").replaceAll("-", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function BatchSizeControl({ count, customCount, setCount, setCustomCount }) {
  return (
    <section className="bat-control-panel">
      <p className="bat-kicker">Batch size</p>
      <div className="mt-2 grid grid-cols-5 gap-1">
        {[4, 8, 16, 32].map((value) => (
          <button key={value} type="button" onClick={() => { setCount(value); setCustomCount(""); }} className={`bat-segment ${count === value && !customCount ? "bat-segment-active" : ""}`}>{value}</button>
        ))}
        <button type="button" onClick={() => setCustomCount(String(count))} className={`bat-segment text-[14px] ${customCount ? "bat-segment-active" : ""}`}>Custom</button>
      </div>
      {customCount !== "" && (
        <input aria-label="Custom batch size" type="number" min="1" max="1000" value={customCount} onChange={(event) => { setCustomCount(event.target.value); setCount(Math.max(1, Number(event.target.value) || 1)); }} className="bat-input mt-2" />
      )}
      <p className="mt-2 text-[14px] text-[var(--text-subtle)]">{count} image{count === 1 ? "" : "s"} will be generated in this batch.</p>
    </section>
  );
}

export function BatchModelControl({ model, models, profiles, setModel }) {
  const profile = profiles[model];
  return (
    <section className="bat-control-panel">
      <p className="bat-kicker">Model</p>
      <select value={model} onChange={(event) => setModel(event.target.value)} className="bat-input mt-2">
        {!models.length && <option value="">No supported model available</option>}
        {models.map((value) => <option key={value} value={value}>{profiles[value]?.display_name ?? value}</option>)}
      </select>
      <div className="mt-2 flex items-start gap-2 text-[14px] leading-[1.55] text-[var(--text-subtle)]">
        {profile?.recommended && <span className="rounded bg-[var(--accent-wash)] px-1.5 py-0.5 font-bold text-[var(--accent-text)]">Recommended</span>}
        <span>{profile?.description || "Structured prompt planning supported."}</span>
      </div>
    </section>
  );
}

export function BatchStyleControl({ styles, styleOptions, setStyles }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const closeOutside = (event) => { if (!rootRef.current?.contains(event.target)) setOpen(false); };
    const closeOnEscape = (event) => { if (event.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => { document.removeEventListener("pointerdown", closeOutside); document.removeEventListener("keydown", closeOnEscape); };
  }, [open]);
  const selectedLabels = styleOptions.filter((value) => styles.includes(value));
  const summary = styleOptions.length && styles.length === styleOptions.length ? "All styles" : selectedLabels.join(", ") || "No styles available";
  const toggle = (value) => {
    if (styles.includes(value)) {
      if (styles.length > 1) setStyles(styles.filter((style) => style !== value));
    } else setStyles([...styles, value]);
  };
  return <section className="bat-control-panel">
    <div ref={rootRef} className="relative text-[14px] font-semibold text-[var(--text-subtle)]">
      <span>Styles</span>
      <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)} className="bat-mini-input mt-1 flex w-full items-center justify-between gap-2 text-left"><span className="truncate">{summary}</span><ChevronDown size={14} className={`shrink-0 transition ${open ? "rotate-180" : ""}`} /></button>
      {open && <div className="absolute left-0 right-0 top-[calc(100%+5px)] z-40 rounded-[10px] bg-[var(--raised)] p-2 shadow-[var(--shadow-menu)] ring-1 ring-inset ring-[var(--divider)]">
        <p className="px-2 pb-1.5 text-[12px] font-normal leading-4 text-[var(--text-faint)]">Choose one or more styles to distribute through the batch.</p>
        {styleOptions.map((value) => <label key={value} className="flex cursor-pointer items-center gap-2 rounded-[7px] px-2 py-2 hover:bg-[var(--field-hover)]"><input type="checkbox" checked={styles.includes(value)} onChange={() => toggle(value)} className="accent-[var(--accent)]" /><span className="text-[13.5px] text-[var(--text)]">{value}</span></label>)}
        <button type="button" onClick={() => setOpen(false)} className="bat-dropdown-done mt-1">Done</button>
      </div>}
    </div>
  </section>;
}

const ORIENTATION_OPTIONS = [["front", "Front"], ["side", "Side"], ["back", "Back"]];

function OrientationControl({ orientations, setOrientations }) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef(null);
  useEffect(() => {
    if (!open) return undefined;
    const closeOutside = (event) => { if (!rootRef.current?.contains(event.target)) setOpen(false); };
    const closeOnEscape = (event) => { if (event.key === "Escape") setOpen(false); };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => { document.removeEventListener("pointerdown", closeOutside); document.removeEventListener("keydown", closeOnEscape); };
  }, [open]);
  const summary = orientations.length === 3 ? "All views" : ORIENTATION_OPTIONS.filter(([id]) => orientations.includes(id)).map(([, label]) => label).join(", ");
  const toggle = (id) => {
    if (orientations.includes(id)) {
      if (orientations.length > 1) setOrientations(orientations.filter((value) => value !== id));
    } else setOrientations([...orientations, id]);
  };
  return <div ref={rootRef} className="relative text-[14px] font-semibold text-[var(--text-subtle)]">
    <span>Orientations</span>
    <button type="button" aria-expanded={open} onClick={() => setOpen((value) => !value)} className="bat-mini-input mt-1 flex w-full items-center justify-between gap-2 text-left"><span className="truncate">{summary}</span><ChevronDown size={14} className={`shrink-0 transition ${open ? "rotate-180" : ""}`} /></button>
    {open && <div className="absolute left-0 right-0 top-[calc(100%+5px)] z-40 rounded-[10px] bg-[var(--raised)] p-2 shadow-[var(--shadow-menu)] ring-1 ring-inset ring-[var(--divider)]">
      <p className="px-2 pb-1.5 text-[12px] font-normal leading-4 text-[var(--text-faint)]">Choose one or more views to distribute through the batch.</p>
      {ORIENTATION_OPTIONS.map(([id, label]) => <label key={id} className="flex cursor-pointer items-center gap-2 rounded-[7px] px-2 py-2 hover:bg-[var(--field-hover)]"><input type="checkbox" checked={orientations.includes(id)} onChange={() => toggle(id)} className="accent-[var(--accent)]" /><span className="text-[13.5px] text-[var(--text)]">{label}</span></label>)}
      <button type="button" onClick={() => setOpen(false)} className="bat-dropdown-done mt-1">Done</button>
    </div>}
  </div>;
}

export function BatchOutputControl({ rating, orientations, qualityMode, distribution, setRating, setOrientations, setQualityMode, setDistribution }) {
  const fields = [
    ["Rating", rating, setRating, [["Safe", "Safe"]]],
    ["Quality", qualityMode, setQualityMode, [["normal", "Normal"], ["high", "High"], ["super", "Super"], ["4k", "4K"], ["8k", "8K"], ["12k", "12K"]]],
    ["Batch mix", distribution, setDistribution, [["balanced", "Balanced"], ["random", "Pure random"]]],
  ];
  return (
    <section className="bat-control-panel">
      <p className="bat-kicker">Output</p>
      <div className="mt-2 grid grid-cols-2 gap-2">
        <OrientationControl orientations={orientations} setOrientations={setOrientations} />
        {fields.map(([label, value, setter, options]) => (
          <label key={label} className="text-[14px] font-semibold text-[var(--text-subtle)]">{label}
            <select value={value} onChange={(event) => setter(event.target.value)} className="bat-mini-input mt-1">
              {options.map(([optionValue, optionLabel]) => <option key={optionValue} value={optionValue}>{optionLabel}</option>)}
            </select>
          </label>
        ))}
      </div>
    </section>
  );
}

export function FixedFacetCard({ icon: Icon, label, value }) {
  return (
    <article className="bat-facet-card opacity-90">
      <Icon size={18} className="shrink-0 text-[var(--accent-text)]" />
      <span className="min-w-0 flex-1"><span className="block text-[13.5px] font-semibold text-[var(--text-subtle)]">{label}</span><span className="mt-0.5 block truncate text-[14px] font-semibold text-[var(--text)]">{value}</span></span>
      <span title="Current model capability" className="h-1.5 w-1.5 rounded-full bg-[var(--text-faint)]" />
    </article>
  );
}

export function BatchFacetCard({ facet, context, selection, onChange, labels, rememberLabels, invalid = false, open, onOpenChange }) {
  const rootRef = useRef(null);
  const [search, setSearch] = useState("");
  const [options, setOptions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const Icon = FACET_ICONS[facet.key] ?? Layers3;

  useEffect(() => {
    if (!open) return undefined;
    const closeOutside = (event) => { if (!rootRef.current?.contains(event.target)) onOpenChange(false); };
    const closeOnEscape = (event) => { if (event.key === "Escape") onOpenChange(false); };
    document.addEventListener("pointerdown", closeOutside);
    document.addEventListener("keydown", closeOnEscape);
    return () => { document.removeEventListener("pointerdown", closeOutside); document.removeEventListener("keydown", closeOnEscape); };
  }, [onOpenChange, open]);

  useEffect(() => {
    if (!open) return undefined;
    let active = true;
    const timer = window.setTimeout(async () => {
      setLoading(true); setError("");
      try {
        const response = await getBatchOptions({ ...context, facet: facet.key, search });
        if (active) {
          const loaded = response.facets.find((item) => item.key === facet.key)?.options ?? [];
          setOptions(loaded); rememberLabels(loaded);
        }
      } catch (requestError) { if (active) setError(requestError.message); }
      finally { if (active) setLoading(false); }
    }, search ? 180 : 0);
    return () => { active = false; window.clearTimeout(timer); };
  }, [context, facet.key, open, rememberLabels, search]);

  const selected = new Set(selection.ids);
  const selectedLabels = selection.ids.map((id) => labels[id] ?? readableId(id));
  const summary = selection.mode === "all" ? "Any" : selectedLabels.length === 1 ? selectedLabels[0] : `${selectedLabels.length} selected`;
  const toggle = (id) => {
    const next = new Set(selected);
    if (next.has(id)) next.delete(id); else next.add(id);
    onChange({ mode: "selected", ids: [...next] });
  };

  return (
    <article ref={rootRef} className={`relative ${open ? "z-20" : "z-0"}`}>
      <button type="button" className={`bat-facet-card w-full text-left ${invalid ? "bat-facet-invalid" : ""}`} aria-expanded={open} aria-invalid={invalid || undefined} onClick={() => onOpenChange(!open)}>
        <Icon size={18} className="shrink-0 text-[var(--accent-text)]" />
        <span className="min-w-0 flex-1"><span className="block text-[13.5px] font-semibold text-[var(--text-subtle)]">{facet.label}</span><span className="mt-0.5 block truncate text-[14px] font-semibold text-[var(--text)]">{summary}</span></span>
        <ChevronDown size={14} className={`text-[var(--text-faint)] transition ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="absolute left-0 right-0 top-[calc(100%+5px)] rounded-[11px] bg-[var(--raised)] p-3 shadow-[var(--shadow-menu)] ring-1 ring-inset ring-[var(--divider)]">
          {facet.description && <p className="mb-2 text-[12px] leading-4 text-[var(--text-faint)]">{facet.description}</p>}
          <div className="grid grid-cols-2 gap-1.5">
            <button type="button" onClick={() => onChange({ mode: "all", ids: [] })} className={`bat-choice-mode ${selection.mode === "all" ? "bat-choice-mode-active" : ""}`}>Any</button>
            <button type="button" onClick={() => onChange({ mode: "selected", ids: selection.ids })} className={`bat-choice-mode ${selection.mode === "selected" ? "bat-choice-mode-active" : ""}`}>Selected pool</button>
          </div>
          <label className="mt-2 flex items-center gap-2 rounded-[8px] bg-[var(--well)] px-3 py-2.5 text-[var(--text-faint)]"><Search size={14} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder={`Search ${facet.label.toLowerCase()}`} className="min-w-0 flex-1 bg-transparent text-[13.5px] text-[var(--text)] outline-none" /></label>
          <div className="mt-2 flex gap-1.5 text-[14px] font-bold"><button type="button" onClick={() => onChange({ mode: "selected", ids: options.map((option) => option.id) })} className="bat-chip">Select visible</button><button type="button" onClick={() => onChange({ mode: "all", ids: [] })} className="bat-chip">Reset to Any</button></div>
          {error && <p className="mt-2 text-[13.5px] text-[var(--danger)]">{error}</p>}
          <div className="mt-2 max-h-52 overflow-y-auto scrollbar-subtle">
            {loading && <p className="py-5 text-center text-[14px] text-[var(--text-muted)]">Loading choices…</p>}
            {!loading && options.map((option) => (
              <label key={option.id} className="flex cursor-pointer items-start gap-2 rounded-[7px] px-2 py-1.5 hover:bg-[var(--field-hover)]">
                <input type="checkbox" checked={selected.has(option.id)} onChange={() => toggle(option.id)} className="mt-0.5 accent-[var(--accent)]" />
                <span><span className="block text-[13.5px] text-[var(--text)]">{option.label}</span><span className="text-[14px] text-[var(--text-faint)]">{option.group}</span></span>
              </label>
            ))}
          </div>
          <button type="button" onClick={() => onOpenChange(false)} className="bat-dropdown-done mt-2">Done</button>
        </div>
      )}
    </article>
  );
}

export function BatchFacetGrid({ facets, context, styles, selections, setSelections, labels, rememberLabels, onClear, invalidFacets = [], validationMessage = "" }) {
  const [openFacet, setOpenFacet] = useState(null);
  return (
    <section>
      <div className="flex items-end justify-between gap-4">
        <div><div className="flex items-center gap-2"><p className="bat-kicker">Recipe facets</p><span className="text-[14px] text-[var(--text-faint)]">Define the dimensions of variation.</span></div></div>
        <button type="button" onClick={onClear} className="text-[14px] font-semibold text-[var(--text-muted)] hover:text-[var(--accent-text)]">Clear all</button>
      </div>
      {validationMessage && <div className="mt-2 rounded-[9px] border border-[var(--danger)]/30 bg-[var(--danger-wash)] px-3 py-2.5 text-[14px] leading-5 text-[var(--danger)]">{validationMessage}</div>}
      <div className="mt-2 grid items-start gap-2 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
        <FixedFacetCard icon={CircleUserRound} label="Subject" value="Adult woman" />
        <FixedFacetCard icon={Users} label="Gender / Presentation" value="Woman" />
        <FixedFacetCard icon={Sparkles} label="Style" value={styles.join(", ")} />
        {facets.map((facet) => <BatchFacetCard key={facet.key} facet={facet} context={context} selection={selections[facet.key] ?? { mode: "all", ids: [] }} onChange={(selection) => setSelections((current) => ({ ...current, [facet.key]: selection }))} labels={labels} rememberLabels={rememberLabels} invalid={invalidFacets.includes(facet.key)} open={openFacet === facet.key} onOpenChange={(open) => setOpenFacet(open ? facet.key : null)} />)}
      </div>
    </section>
  );
}

function recipeValues(item, labels) {
  const values = Object.entries(item?.resolved_facets ?? {}).map(([facet, id]) => ({ facet, label: labels[id] ?? readableId(id), Icon: FACET_ICONS[facet] ?? Layers3 }));
  if (item?.orientation) values.unshift({ facet: "orientation", label: `${readableId(item.orientation)} view`, Icon: Camera });
  return values;
}

export function BatchPlanTile({ index, item, labels, status = "planned", onInspect }) {
  const values = recipeValues(item, labels);
  const thumbnail = item?.thumbnail_url;
  const state = item?.status ?? status;
  const isComplete = state === "succeeded" && thumbnail;
  return (
    <button type="button" disabled={!isComplete} onClick={isComplete ? onInspect : undefined} className={`bat-plan-tile bat-plan-tile-${state} group relative min-h-[190px] overflow-hidden rounded-[10px] bg-[var(--field)] text-left ring-1 ring-inset ring-[var(--divider)] sm:min-h-[175px] ${isComplete ? "cursor-pointer" : "cursor-default"}`}>
      {thumbnail ? <img src={thumbnail} alt={`Batch image ${index + 1}`} className="absolute inset-0 h-full w-full object-cover transition duration-200 group-hover:scale-[1.02]" /> : (
        <div className="h-full p-3 pt-8">
          {values.length ? <div className="space-y-2">{values.slice(0, 4).map(({ facet, label, Icon }) => <div key={facet} className="flex items-center gap-2"><Icon size={13} className="shrink-0 text-[var(--accent-text)]" /><span className="truncate text-[14px] text-[var(--text-2)]">{label}</span></div>)}</div> : <div className="grid h-full place-items-center"><div className="text-center"><Layers3 size={22} className="mx-auto text-[var(--accent-text)] opacity-60" /><p className="mt-2 text-[14px] text-[var(--text-faint)]">Planned recipe</p></div></div>}
        </div>
      )}
      <span className="absolute left-2 top-2 rounded-[5px] bg-black/55 px-2 py-1 text-[14px] font-bold text-white">{String(index + 1).padStart(2, "0")}</span>
      {state !== "planned" && <span className={`absolute bottom-2 right-2 flex items-center gap-1 rounded-full px-2 py-1 text-[14px] font-bold capitalize backdrop-blur ${state === "succeeded" ? "bg-[var(--ok-wash)] text-[var(--ok)]" : state === "failed" ? "bg-[var(--danger-wash)] text-[var(--danger)]" : "bg-black/60 text-white"}`}>{state === "running" && <LoaderCircle size={10} className="animate-spin" />}{state === "succeeded" && <Check size={10} />}{state === "cancelled" && <X size={10} />}{state}</span>}
    </button>
  );
}

function BatchImageViewer({ items, initialIndex, onClose }) {
  const [index, setIndex] = useState(initialIndex);
  const [favorites, setFavorites] = useState(() => Object.fromEntries(items.map((item) => [item.generation_id, Boolean(item.favorite)])));
  const [saving, setSaving] = useState(false);
  const item = items[index];
  const favorite = Boolean(favorites[item?.generation_id]);
  const move = (direction) => setIndex((current) => (current + direction + items.length) % items.length);
  const toggleFavorite = async () => {
    if (!item?.generation_id || saving) return;
    const next = !favorite;
    setFavorites((current) => ({ ...current, [item.generation_id]: next }));
    setSaving(true);
    try { await setFavorite(item.generation_id, next); }
    catch { setFavorites((current) => ({ ...current, [item.generation_id]: !next })); }
    finally { setSaving(false); }
  };
  useEffect(() => {
    const handleKey = (event) => {
      if (event.key === "ArrowLeft") move(-1);
      if (event.key === "ArrowRight") move(1);
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  });
  return <Dialog open onClose={onClose} className="relative z-[150]">
    <DialogBackdrop className="fixed inset-0 bg-black/90 backdrop-blur-md" />
    <div className="fixed inset-0 p-3 sm:p-6"><DialogPanel className="relative flex h-full w-full flex-col overflow-hidden rounded-[16px] bg-[var(--canvas)] ring-1 ring-white/10">
      <header className="flex shrink-0 items-center justify-between gap-4 border-b border-[var(--divider)] px-4 py-3 sm:px-5">
        <div><DialogTitle className="text-[16px] font-extrabold text-[var(--text)]">Batch image {index + 1} of {items.length}</DialogTitle><p className="mt-0.5 text-[13px] text-[var(--text-muted)]">Use the arrows or keyboard to browse this batch.</p></div>
        <div className="flex items-center gap-2"><button type="button" disabled={saving} onClick={toggleFavorite} aria-label={favorite ? "Remove from favourites" : "Add to favourites"} className={`bat-viewer-action ${favorite ? "is-favorite" : ""}`}><Heart size={18} fill={favorite ? "currentColor" : "none"} /><span className="hidden sm:inline">{favorite ? "Favourited" : "Favourite"}</span></button><button type="button" onClick={onClose} aria-label="Close batch viewer" className="bat-viewer-icon"><X size={20} /></button></div>
      </header>
      <div className="relative min-h-0 flex-1 overflow-hidden bg-black">
        <img src={item.full_url ?? item.thumbnail_url} alt={`Batch image ${index + 1}`} className="h-full w-full object-contain" />
        {items.length > 1 && <><button type="button" onClick={() => move(-1)} aria-label="Previous image" className="bat-viewer-nav left-3 sm:left-5"><ArrowLeft size={24} /></button><button type="button" onClick={() => move(1)} aria-label="Next image" className="bat-viewer-nav right-3 sm:right-5"><ArrowRight size={24} /></button></>}
      </div>
    </DialogPanel></div>
  </Dialog>;
}

export function BatchCanvas({ count, distribution, preview, batch, labels, onShuffle, rating, limit = 8 }) {
  const [viewerIndex, setViewerIndex] = useState(null);
  const sourceItems = batch?.items ?? preview?.items ?? [];
  const completedItems = sourceItems.filter((item) => item.status === "succeeded" && item.thumbnail_url);
  const active = batch && !["completed", "partial", "failed", "cancelled"].includes(batch.status);
  const terminal = batch && !active;
  const displayItems = terminal && completedItems.length ? completedItems : sourceItems;
  const visibleCount = Math.min(displayItems.length || count, limit);
  const hiddenCount = Math.max(0, count - visibleCount);
  const counts = batch?.counts ?? {};
  return (
    <section className={`bat-stage ${active ? "bat-stage-live" : ""}`}>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div><p className="bat-kicker">{terminal ? "Completed outputs" : "Batch canvas"}</p><p className="mt-1 text-[14px] text-[var(--text-subtle)]">{terminal ? `A representative contact sheet from ${completedItems.length} completed image${completedItems.length === 1 ? "" : "s"}.` : `Forge will generate ${count} image${count === 1 ? "" : "s"} with ${distribution === "balanced" ? "balanced variation" : "random variation"} across your selected facets.`}</p></div>
        <div className="flex items-center gap-3">{batch && <button type="button" disabled={!completedItems.length} onClick={() => setViewerIndex(0)} className="bat-small-button bat-view-batch"><Images size={14} />View batch{completedItems.length ? ` (${completedItems.length})` : ""}</button>}<button type="button" disabled={!!batch} onClick={onShuffle} className="bat-small-button"><RotateCw size={13} />Shuffle plan</button><span className="flex items-center gap-1.5 text-[14px] text-[var(--text-muted)]"><ImageIcon size={13} />{count} images</span></div>
      </div>
      {active && <div className="bat-live-banner" role="status" aria-live="polite"><span className="bat-live-icon"><LoaderCircle size={22} className="animate-spin" /></span><div className="min-w-0 flex-1"><div className="flex flex-wrap items-center gap-x-3 gap-y-1"><strong>ForgeBAT is generating your batch</strong><span>{batch.completed_count} of {batch.requested_count} complete</span></div><p>{counts.running ?? 0} generating · {counts.queued ?? 0} queued · {counts.pending ?? 0} waiting · {counts.failed ?? 0} failed</p><div className="bat-live-track"><i style={{ width: `${batch.progress_percent}%` }} /><span /></div></div></div>}
      <div className={`bat-canvas-row mt-4 ${terminal ? "is-results" : ""}`}>
        {Array.from({ length: visibleCount }, (_, index) => <BatchPlanTile key={displayItems[index]?.id ?? displayItems[index]?.request_id ?? index} index={displayItems[index]?.index ?? index} item={displayItems[index]} labels={labels} status={preview ? "planned" : "planned"} onInspect={() => { const completedIndex = completedItems.findIndex((value) => value.id === displayItems[index]?.id); if (completedIndex >= 0) setViewerIndex(completedIndex); }} />)}
      </div>
      {hiddenCount > 0 && <p className="mt-2 text-right text-[14px] text-[var(--text-faint)]">+ {hiddenCount} more {terminal ? "batch output" : "planned image"}{hiddenCount === 1 ? "" : "s"}</p>}
      {!preview && !batch && <p className="mt-3 rounded-[8px] bg-[var(--well)] px-3 py-2.5 text-[14px] text-[var(--text-subtle)]">These are recipe placeholders—not generated images. Preview the batch to resolve the deterministic plan.</p>}
      {viewerIndex !== null && completedItems.length > 0 && <BatchImageViewer items={completedItems} initialIndex={Math.min(viewerIndex, completedItems.length - 1)} onClose={() => setViewerIndex(null)} />}
    </section>
  );
}

export function BatchCoverage({ coverage, execution = false }) {
  const preferred = ["orientation", "heritage", "pose", "environment", "lighting", "hair", "clothing"];
  const entries = preferred.filter((key) => coverage?.[key]?.length).slice(0, 4).map((key) => [key, coverage[key]]);
  return (
    <section className="bat-stage">
      <div className="flex items-center gap-2"><p className="bat-kicker">Coverage overview</p><span className="text-[14px] text-[var(--text-faint)]">{execution ? "Distribution across this batch" : "Distribution from the current preview plan"}</span></div>
      {!entries.length ? <div className="mt-3 rounded-[9px] border border-dashed border-[var(--divider)] px-4 py-7 text-center text-[14px] text-[var(--text-faint)]">Preview the batch to reveal its planned coverage.</div> : (
        <div className="mt-3 grid gap-5 md:grid-cols-2 xl:grid-cols-3">
          {entries.map(([facet, values]) => {
            const sorted = [...values].sort((a, b) => b.count - a.count);
            const total = sorted.reduce((sum, value) => sum + value.count, 0);
            return <div key={facet} className="bat-coverage-group"><div className="flex items-baseline justify-between gap-3"><p className="text-[14px] font-bold text-[var(--text-2)]">{facet === "heritage" ? "Heritage" : facet === "pose" ? "Pose / Action" : readableId(facet)}</p><span className="text-[11px] text-[var(--text-faint)]">{sorted.length} variants</span></div><div className="mt-2 flex h-2 overflow-hidden rounded-full bg-[var(--well)]">{sorted.map((value, index) => <span key={value.id} title={`${value.label ?? readableId(value.id)}: ${value.count}`} className="h-full" style={{ width: `${value.count / total * 100}%`, background: ["var(--nyx-accent)", "var(--nyx-signal)", "var(--nyx-info)", "var(--nyx-warn)", "var(--nyx-alt)", "var(--nyx-fail)"][index % 6] }} />)}</div><div className="mt-2 space-y-1.5">{sorted.slice(0, 4).map((value) => <div key={value.id} className="flex items-center justify-between gap-3 text-[12px]"><span className="min-w-0 truncate text-[var(--text-muted)]">{value.label ?? readableId(value.id)}</span><span className="shrink-0 tabular-nums text-[var(--text-2)]"><b>{value.count}</b> · {Math.round(value.count / total * 100)}%</span></div>)}</div>{sorted.length > 4 && <p className="mt-2 text-[11px] text-[var(--text-faint)]">+ {sorted.length - 4} additional variants</p>}</div>;
          })}
        </div>
      )}
    </section>
  );
}

export function BatchExecutionSummary({ batch, modelLabel, styles, qualityLabel, rating, busy, error, onCancel, onRetry, onGallery, onDuplicate, onBatches }) {
  const counts = batch.counts ?? {};
  const terminal = ["completed", "partial", "failed", "cancelled"].includes(batch.status);
  const active = !terminal;
  const statusLabel = { planning: "Preparing batch", queued: "Queued", running: "Generating", completed: "Complete", partial: "Finished with errors", failed: "Failed", cancelled: "Cancelled" }[batch.status] ?? batch.status;
  const progress = Math.max(0, Math.min(100, batch.progress_percent ?? 0));
  const finishedAt = batch.finished_at ? new Date(batch.finished_at) : null;
  const startedAt = batch.started_at ? new Date(batch.started_at) : null;
  const durationMinutes = finishedAt && startedAt ? Math.max(1, Math.round((finishedAt - startedAt) / 60000)) : null;
  const metrics = terminal ? [
    ["Images", counts.succeeded ?? 0, "ok"],
    ["Failed", counts.failed ?? 0, "failed"],
  ] : [
    ["Complete", counts.succeeded ?? 0, "ok"],
    ["Generating", counts.running ?? 0, "live"],
    ["Queued", (counts.queued ?? 0) + (counts.pending ?? 0), "queued"],
    ["Failed", counts.failed ?? 0, "failed"],
  ];
  return (
    <aside className="bat-summary bat-execution-summary">
      <div className="min-h-0 flex-1 overflow-y-auto p-5 scrollbar-subtle">
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3">
            <span className={`grid h-10 w-10 place-items-center rounded-full ${active ? "bg-[var(--accent-wash)] text-[var(--accent-hi)]" : batch.status === "completed" ? "bg-[var(--ok-wash)] text-[var(--ok)]" : "bg-[var(--field)] text-[var(--text-muted)]"}`}>
              {active ? <LoaderCircle size={19} className="animate-spin" /> : batch.status === "completed" ? <Check size={18} /> : <Layers3 size={18} />}
            </span>
            <div><p className="bat-kicker">Batch execution</p><h2 className="mt-1 text-[17px] font-extrabold text-[var(--text)]">Batch {batch.id}</h2></div>
          </div>
          <span className={`rounded-full px-2.5 py-1 text-[12px] font-bold ${active ? "bg-[var(--accent-wash)] text-[var(--accent-text)]" : batch.status === "completed" ? "bg-[var(--ok-wash)] text-[var(--ok)]" : "bg-[var(--field)] text-[var(--text-muted)]"}`}>{statusLabel}</span>
        </div>

        <section className={`mt-5 rounded-[11px] p-4 ring-1 ring-inset ${batch.status === "completed" ? "bat-complete-hero" : "bg-[var(--field)] ring-[var(--divider)]"}`}>
          <div className="flex items-end justify-between gap-4"><div><p className="text-[13px] font-semibold text-[var(--text-muted)]">Overall progress</p><p className="mt-1 text-[25px] font-extrabold tabular-nums text-[var(--text)]">{Math.round(progress)}%</p></div><p className="pb-1 text-[13px] font-bold text-[var(--text-2)]">{batch.completed_count} of {batch.requested_count}</p></div>
          <div className="mt-3 h-2 overflow-hidden rounded-full bg-[var(--well)]"><div className="h-full rounded-full bg-[var(--accent)] transition-[width] duration-300" style={{ width: `${progress}%` }} /></div>
          <p className="mt-3 text-[12px] leading-5 text-[var(--text-muted)]">{active ? "This view refreshes automatically while Forge works through the queue." : batch.status === "completed" ? `${counts.succeeded ?? batch.completed_count} images are ready to review.` : `This batch finished with status: ${statusLabel.toLowerCase()}.`}</p>
        </section>

        <dl className="mt-3 grid grid-cols-2 gap-2">
          {metrics.map(([label, value, tone]) => <div key={label} className={`bat-execution-metric is-${tone}`}><dt>{label}</dt><dd>{value}</dd></div>)}
        </dl>

        <section className="mt-5 border-t border-[var(--divider)] pt-5">
          <div className="flex items-center justify-between"><h3 className="bat-kicker">Run details</h3><span className="text-[12px] text-[var(--text-faint)]">Seed {batch.planner_seed}</span></div>
          <dl className="mt-3 space-y-2.5 text-[13px]">
            <div className="flex gap-4"><dt className="w-20 shrink-0 text-[var(--text-faint)]">Model</dt><dd className="min-w-0 flex-1 break-words text-right font-semibold text-[var(--text-2)]">{modelLabel}</dd></div>
            <div className="flex gap-4"><dt className="w-20 shrink-0 text-[var(--text-faint)]">Styles</dt><dd className="min-w-0 flex-1 break-words text-right font-semibold text-[var(--text-2)]">{styles.join(", ")}</dd></div>
            <div className="flex gap-4"><dt className="w-20 shrink-0 text-[var(--text-faint)]">Output</dt><dd className="flex-1 text-right font-semibold text-[var(--text-2)]">{qualityLabel} · {rating}</dd></div>
            {durationMinutes && <div className="flex gap-4"><dt className="w-20 shrink-0 text-[var(--text-faint)]">Duration</dt><dd className="flex-1 text-right font-semibold text-[var(--text-2)]">{durationMinutes} min</dd></div>}
            {finishedAt && <div className="flex gap-4"><dt className="w-20 shrink-0 text-[var(--text-faint)]">Finished</dt><dd className="flex-1 text-right font-semibold text-[var(--text-2)]">{finishedAt.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}</dd></div>}
          </dl>
        </section>

        {error && <p className="mt-4 rounded-[8px] bg-[var(--danger-wash)] p-3 text-[13px] text-[var(--danger)]">{error}</p>}
      </div>
      <footer className="space-y-2 border-t border-[var(--divider)] p-4">
        {active && <button type="button" onClick={onCancel} disabled={!!busy} className="bat-execution-action is-danger"><X size={14} />{busy === "cancel" ? "Cancelling…" : "Cancel remaining"}</button>}
        {(counts.failed ?? 0) > 0 && <button type="button" onClick={onRetry} disabled={!!busy} className="bat-execution-action"><RotateCw size={14} />{busy === "retry" ? "Retrying…" : "Retry failed"}</button>}
        {(counts.succeeded ?? 0) > 0 && <button type="button" onClick={onGallery} className="bat-execution-action is-primary"><Images size={14} />Open completed images</button>}
        <div className="grid grid-cols-2 gap-2"><button type="button" onClick={onBatches} className="bat-execution-action">All batches</button><button type="button" onClick={onDuplicate} className="bat-execution-action">Duplicate recipe</button></div>
      </footer>
    </aside>
  );
}

export function BatchSummary({ count, modelLabel, styles, qualityLabel, rating, estimate, distribution, facets, selections, batch, busy, canSubmit, error, warnings, onPreview, onGenerate, onCancel, onRetry, onGallery, onDuplicate }) {
  const varied = facets.map((facet) => {
    const selection = selections[facet.key] ?? { mode: "all", ids: [] };
    return { ...facet, value: selection.mode === "all" ? `Any · ${facet.option_count} compatible` : selection.ids.length === 1 ? "Fixed" : `${selection.ids.length} options` };
  });
  const terminal = ["completed", "partial", "failed", "cancelled"].includes(batch?.status);
  return (
    <aside className="bat-summary">
      <div className="min-h-0 flex-1 overflow-y-auto p-5 scrollbar-subtle">
        <div className="flex items-center justify-between gap-3"><div className="flex items-center gap-2"><span className="grid h-8 w-8 place-items-center rounded-full bg-[var(--accent-wash)] text-[var(--accent-hi)]"><Zap size={16} /></span><p className="bat-kicker">Live batch summary</p></div><span className="flex items-center gap-1.5 text-[14px] font-bold text-[var(--ok)]"><span className="dot-live h-1.5 w-1.5 rounded-full bg-[var(--ok)]" />Live</span></div>
        <h2 className="mt-4 text-[16px] font-extrabold text-[var(--text)]">{batch ? `${batch.completed_count} of ${batch.requested_count} complete` : `${count} planned images`}</h2>
        <p className="mt-2 text-[13.5px] leading-[1.6] text-[var(--text-muted)]">{count} {qualityLabel.toLowerCase()} image{count === 1 ? "" : "s"} using {styles.join(", ")} with {distribution === "balanced" ? "balanced variation" : "random variation"} across {facets.length} facets.</p>
        <dl className="mt-3 grid grid-cols-2 gap-2">
          {[["Model", modelLabel], ["Quality", qualityLabel], ["Rating", rating], ["Estimated time", estimate]].map(([term, value]) => <div key={term} className="rounded-[9px] bg-[var(--field)] p-3"><dt className="text-[14px] font-semibold text-[var(--text-faint)]">{term}</dt><dd className="mt-1.5 line-clamp-2 text-[13.5px] font-bold text-[var(--text)]">{value}</dd></div>)}
        </dl>
        <section className="mt-5"><div className="flex items-center justify-between"><h3 className="bat-kicker">Varied choices</h3><span className="rounded-full bg-[var(--accent-wash)] px-2.5 py-1 text-[14px] font-bold text-[var(--accent-text)]">{distribution === "balanced" ? "Balanced" : "Random"}</span></div><div className="mt-3 space-y-2">{varied.map((facet) => { const Icon = FACET_ICONS[facet.key] ?? Layers3; return <div key={facet.key} className="flex items-center gap-2 text-[14px]"><Icon size={13} className="text-[var(--text-faint)]" /><span className="flex-1 text-[var(--text-muted)]">{facet.label}</span><span className="font-semibold text-[var(--text-2)]">{facet.value}</span></div>; })}</div></section>
        {batch && <section className="mt-4 border-t border-[var(--divider)] pt-4"><div className="flex items-center justify-between"><span className="text-[14px] font-bold text-[var(--text)]">Batch {batch.id}</span><span className="rounded-full bg-[var(--field)] px-2 py-1 text-[14px] font-bold capitalize text-[var(--text-muted)]">{batch.status}</span></div><div className="mt-2 flex flex-wrap gap-1.5">{!terminal && <button type="button" onClick={onCancel} disabled={!!busy} className="bat-chip text-[var(--danger)]">Cancel remaining</button>}{batch.counts.failed > 0 && <button type="button" onClick={onRetry} disabled={!!busy} className="bat-chip">Retry failed</button>}{batch.counts.succeeded > 0 && <button type="button" onClick={onGallery} className="bat-chip">Open Gallery</button>}<button type="button" onClick={onDuplicate} className="bat-chip">Duplicate recipe</button></div></section>}
      </div>
      <footer className="border-t border-[var(--divider)] p-5">
        <div className="mb-3 rounded-[9px] bg-[var(--field)] p-3.5 text-[14px] leading-[1.55] text-[var(--text-muted)]"><Sparkles size={14} className="mb-2 text-[var(--warn)]" />Preview the batch to review the planned variation before generating. You can refine any facet above.</div>
        {!!warnings?.length && <p className="mb-2 rounded-[8px] bg-[var(--warn-wash)] p-2 text-[14px] text-[var(--warn)]">{warnings.join(" ")}</p>}
        {error && <p className="mb-2 rounded-[8px] bg-[var(--danger-wash)] p-2 text-[14px] text-[var(--danger)]">{error}</p>}
        <button type="button" disabled={!canSubmit} onClick={onPreview} className="flex w-full items-center justify-center gap-2 rounded-[9px] border border-[var(--accent)] bg-[var(--accent-wash)] px-3 py-3 text-[14px] font-bold text-[var(--accent-text)] disabled:opacity-50">{busy === "preview" ? <LoaderCircle size={15} className="animate-spin" /> : <Eye size={15} />}{busy === "preview" ? "Planning…" : "Preview Batch"}</button>
        <button type="button" disabled={!canSubmit} onClick={onGenerate} className="mt-2 flex w-full items-center justify-center gap-2 rounded-[9px] bg-[var(--accent)] px-3 py-3.5 text-[14px] font-extrabold text-[var(--on-accent)] disabled:opacity-50"><Sparkles size={15} />{busy === "create" ? "Creating…" : `Generate ${count} Image${count === 1 ? "" : "s"}`}</button>
      </footer>
    </aside>
  );
}
