import { useEffect, useMemo, useRef, useState } from "react";
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from "@headlessui/react";
import { Heart, Search, X } from "lucide-react";
import { displayJobId } from "./jobLabels";

// Reuses a past generation's prompt without inheriting its settings, so the
// current quality mode, aspect and model still apply. Only generations from the
// selected model are offered: a Pony prompt carries score_ tags that a
// photoreal SDXL checkpoint reads as noise.
export default function PromptSourceDialog({ open, onClose, images, model, onUse, hasMore, loadingMore, onLoadMore }) {
  const sentinelRef = useRef(null);
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState(null);
  const [favoritesOnly, setFavoritesOnly] = useState(false);

  useEffect(() => {
    if (open) {
      setSelected(null);
      setQuery("");
    }
  }, [open]);

  const candidates = useMemo(() => {
    const term = query.trim().toLowerCase();
    return images
      .filter((image) => image.model === model)
      .filter((image) => !favoritesOnly || image.favorite)
      .filter((image) => {
        if (!term) return true;
        // A bare number matches the id, so typing "354" finds #354 directly.
        if (/^\d+$/.test(term)) return String(image.id).includes(term);
        return (image.positive_prompt ?? "").toLowerCase().includes(term);
      });
  }, [images, model, favoritesOnly, query]);

  // The gallery pages, so the picker would otherwise only ever see the first
  // page. Load the next one when the end of the list scrolls into view.
  useEffect(() => {
    if (!open || !hasMore || loadingMore) return undefined;
    if (!candidates.length) {
      onLoadMore?.();
      return undefined;
    }
    const node = sentinelRef.current;
    if (!node) return undefined;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) onLoadMore?.();
    }, { rootMargin: "200px" });
    observer.observe(node);
    return () => observer.disconnect();
    // onLoadMore is intentionally not a dependency: it changes identity every
    // render and would rebuild the observer on each one.
  }, [open, hasMore, loadingMore, candidates.length]); // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <Dialog open={open} onClose={onClose} className="relative z-[130]">
      <DialogBackdrop className="fixed inset-0 bg-black/70" />
      <div className="fixed inset-0 grid place-items-center p-6">
        <DialogPanel className="flex max-h-[min(720px,calc(100vh-48px))] w-full max-w-[880px] flex-col overflow-hidden rounded-[16px] bg-[var(--raised)] shadow-[0_24px_60px_rgba(0,0,0,0.6)]">
          <header className="flex items-start justify-between border-b border-[var(--divider)] px-6 py-5">
            <div>
              <DialogTitle className="text-[16px] font-bold text-[var(--text)]">Reuse a prompt</DialogTitle>
              <p className="forge-body mt-1">Only the prompt is taken. Your current model, quality and aspect ratio still apply.</p>
            </div>
            <button type="button" onClick={onClose} aria-label="Close prompt picker" className="grid h-8 w-8 place-items-center rounded-[8px] text-[var(--text-muted)] transition hover:bg-[var(--field)] hover:text-white"><X size={17} /></button>
          </header>

          <div className="flex shrink-0 items-center gap-2 px-6 pt-4">
            <div className="relative min-w-0 flex-1">
              <Search size={14} className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[var(--text-subtle)]" />
              <input
                autoFocus
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search the prompt text, or type a generation id"
                className="w-full rounded-[9px] bg-[var(--field)] py-2.5 pl-9 pr-3 text-xs text-[var(--text)] outline-none focus:ring-2 focus:ring-[var(--accent)]"
              />
            </div>
            <button type="button" aria-pressed={favoritesOnly} onClick={() => setFavoritesOnly((value) => !value)} className={`flex h-[38px] shrink-0 items-center gap-1.5 rounded-[9px] px-3 text-[11.5px] font-bold transition ${favoritesOnly ? "bg-[var(--accent-wash)] text-[var(--accent-text)] ring-1 ring-[var(--accent)]" : "bg-[var(--field)] text-[var(--text-muted)] hover:text-[var(--text-ui)]"}`}>
              <Heart size={13} fill={favoritesOnly ? "currentColor" : "none"} /> Favourites
            </button>
          </div>

          <div className="scrollbar-subtle min-h-0 flex-1 overflow-y-auto px-6 py-4">
            {candidates.length ? (
              <div className="space-y-2">
                {candidates.map((image) => (
                  <button
                    key={image.id}
                    type="button"
                    onClick={() => setSelected(image)}
                    className={`flex w-full items-start gap-3 rounded-[10px] p-2.5 text-left transition ${selected?.id === image.id ? "bg-[var(--accent-wash)] ring-1 ring-[var(--accent)]" : "bg-[var(--field)] hover:bg-[var(--field-hover)]"}`}
                  >
                    <span className="relative h-14 w-14 shrink-0 overflow-hidden rounded-[8px]">
                      <img src={image.thumbnail_url} alt="" loading="lazy" className="h-full w-full object-cover" />
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className="flex items-center gap-2">
                        <strong className="text-[12px] font-bold tabular-nums text-[var(--text-ui)]">#{image.id}</strong>
                        <span className="text-[10.5px] text-[var(--text-subtle)]">{image.style}{image.style_variant ? ` · ${image.style_variant}` : ""}</span>
                        {image.favorite && <Heart size={11} className="text-[#ff6b8a]" fill="currentColor" />}
                      </span>
                      <span className="mt-1 line-clamp-2 block text-[11px] leading-4 text-[var(--text-muted)]">{image.positive_prompt}</span>
                    </span>
                  </button>
                ))}
                <div ref={sentinelRef} className="h-px" />
                {loadingMore && <p className="py-3 text-center text-[11px] text-[var(--text-muted)]">Loading older generations…</p>}
                {!hasMore && <p className="py-3 text-center text-[10.5px] text-[var(--text-subtle)]">End of history.</p>}
              </div>
            ) : (
              <div className="grid h-full min-h-[180px] place-items-center text-center">
                <p className="text-[12.5px] text-[var(--text-muted)]">
                  {images.some((image) => image.model === model)
                    ? hasMore ? "Nothing matches yet - scroll to load older generations." : "Nothing matches that search."
                    : "No generations yet for the selected model."}
                </p>
              </div>
            )}
          </div>

          <footer className="flex shrink-0 items-center justify-end gap-2 border-t border-[var(--divider)] px-6 py-4">
            <button type="button" onClick={onClose} className="rounded-[9px] bg-[var(--field)] px-4 py-2.5 text-[12px] font-bold text-[var(--text-ui)] transition hover:bg-[var(--field-hover)]">Cancel</button>
            <button type="button" disabled={!selected} onClick={() => { onUse(selected); onClose(); }} className="forge-btn-primary disabled:opacity-40">Use this prompt</button>
          </footer>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
