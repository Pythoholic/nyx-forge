import { useCallback, useEffect, useRef, useState } from "react";
import { Heart, ThumbsDown, X } from "lucide-react";
import { Gallery, Item } from "react-photoswipe-gallery";
import { rateImage, setFavorite } from "./api";
import "photoswipe/dist/photoswipe.css";

const NEGATIVE_FEEDBACK_GROUPS = [
  {
    label: "Prompt problem",
    reasons: [{ code: "prompt_match", label: "Prompt mismatch" }],
  },
  {
    label: "Render problem",
    reasons: [
      { code: "eyes", label: "Eyes or gaze" },
      { code: "anatomy", label: "Face or anatomy" },
      { code: "composition", label: "Composition" },
      { code: "detail", label: "Detail or sharpness" },
    ],
  },
  {
    label: "Other",
    reasons: [
      { code: "style", label: "Style mismatch" },
      { code: "content_rating", label: "Rating mismatch" },
    ],
  },
];

const lucideIcon = (paths, size = 20) => `
  <svg aria-hidden="true" viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">${paths}</svg>`;
const ICONS = {
  minus: lucideIcon('<path d="M5 12h14"/>'),
  plus: lucideIcon('<path d="M5 12h14"/><path d="M12 5v14"/>'),
  download: lucideIcon('<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5"/><path d="M12 15V3"/>'),
  close: lucideIcon('<path d="M18 6 6 18"/><path d="m6 6 12 12"/>'),
  previous: lucideIcon('<path d="m15 18-6-6 6-6"/>', 18),
  next: lucideIcon('<path d="m9 18 6-6-6-6"/>', 18),
};

const OPTIONS = {
  bgOpacity: 0.94,
  wheelToZoom: true,
  initialZoomLevel: "fit",
  secondaryZoomLevel: (zoom) => Math.min(Math.max(zoom.initial * 1.8, zoom.initial + 0.25), 1.5),
  maxZoomLevel: (zoom) => Math.max(zoom.initial * 4, 2.5),
  doubleTapAction: "zoom",
  imageClickAction: false,
  bgClickAction: "close",
  tapAction: "toggle-controls",
  preload: [1, 1],
  indexIndicatorSep: " of ",
  counter: false,
  zoom: false,
  close: false,
  arrowPrev: false,
  arrowNext: false,
  // Small uniform inset only. Reserving strips for the chrome sized the image
  // to the gap between the bars, so it opened at full height and then visibly
  // shrank once the overlay mounted. The controls float over the image now.
  padding: { top: 16, bottom: 16, left: 24, right: 24 },
  showHideAnimationType: "fade",
  showAnimationDuration: 180,
  hideAnimationDuration: 140,
  closeOnVerticalDrag: false,
};

/** Favourite toggle for the open image. The prompt itself lives in Deploy. */
function FavoriteControl({ image }) {
  // Track the image id alongside the flag: PhotoSwipe swaps images under this
  // component without unmounting it, so a plain useState would show the
  // previous image's state for a frame after navigating.
  const [pending, setPending] = useState(null);
  if (!image) return null;
  const favorite = pending && pending.id === image.id ? pending.favorite : Boolean(image.favorite);

  const toggle = async () => {
    const next = !favorite;
    setPending({ id: image.id, favorite: next });
    try {
      await setFavorite(image.id, next);
      image.favorite = next;
    } catch {
      setPending({ id: image.id, favorite: !next });
    }
  };

  return (
    <button
      type="button"
      onClick={toggle}
      aria-pressed={favorite}
      aria-label={favorite ? "Remove from favourites" : "Add to favourites"}
      className={`flex items-center gap-2 rounded-full px-4 py-[10px] text-[13px] font-semibold shadow-[var(--shadow-menu)] backdrop-blur transition ${
        favorite
          ? "bg-[var(--accent)] text-[var(--on-accent)]"
          : "bg-[color-mix(in_srgb,var(--raised)_88%,transparent)] text-zinc-200 hover:text-white"
      }`}
    >
      <Heart size={16} fill={favorite ? "currentColor" : "none"} />
      {favorite ? "Favourited" : "Favourite"}
    </button>
  );
}

/** Compact low-score feedback control for the open image. */
function NotGoodControl({ image }) {
  const controlRef = useRef(null);
  const [openImageId, setOpenImageId] = useState(null);
  // Like FavoriteControl, every local value includes the image id because the
  // lightbox navigates by swapping its image prop without unmounting controls.
  const [draft, setDraft] = useState(null);
  const [pending, setPending] = useState(null);

  const displayedScore = pending?.id === image.id ? pending.score : image.user_rating;
  const displayedReasons = pending?.id === image.id
    ? pending.reasons
    : (image.feedback_reasons ?? []);
  const selectedReasons = draft?.id === image.id
    ? draft.reasons
    : new Set(displayedReasons);
  const expanded = openImageId === image.id;
  const saving = pending?.id === image.id && pending.saving;
  const markedNotGood = displayedScore != null && displayedScore <= 2;

  useEffect(() => {
    if (!expanded) return undefined;
    const closeOnOutsideClick = (event) => {
      if (!controlRef.current?.contains(event.target)) setOpenImageId(null);
    };
    const closeOnEscape = (event) => {
      if (event.key === "Escape") setOpenImageId(null);
    };
    document.addEventListener("pointerdown", closeOnOutsideClick);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnOutsideClick);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [expanded]);

  const togglePanel = () => {
    if (expanded) {
      setOpenImageId(null);
      return;
    }
    setDraft({ id: image.id, reasons: new Set(displayedReasons) });
    setOpenImageId(image.id);
  };

  const toggleReason = (code) => {
    setDraft((current) => {
      const reasons = new Set(current?.id === image.id ? current.reasons : displayedReasons);
      if (reasons.has(code)) reasons.delete(code);
      else reasons.add(code);
      return { id: image.id, reasons };
    });
  };

  const submit = async () => {
    if (selectedReasons.size === 0 || saving) return;
    const reasons = [...selectedReasons];
    const previous = {
      score: image.user_rating,
      reasons: [...(image.feedback_reasons ?? [])],
    };
    setPending({ id: image.id, score: 1, reasons, saving: true, error: "" });
    image.user_rating = 1;
    image.feedback_reasons = reasons;
    try {
      const feedback = await rateImage(image.id, 1, reasons, image.access_token ?? null);
      image.user_rating = feedback.score;
      image.feedback_reasons = feedback.reasons;
      setDraft({ id: image.id, reasons: new Set(feedback.reasons) });
      setPending({ id: image.id, score: feedback.score, reasons: feedback.reasons, saving: false, error: "" });
      setOpenImageId((current) => current === image.id ? null : current);
    } catch (error) {
      image.user_rating = previous.score;
      image.feedback_reasons = previous.reasons;
      setPending({
        id: image.id,
        score: previous.score,
        reasons: previous.reasons,
        saving: false,
        error: error instanceof Error ? error.message : "Could not save feedback.",
      });
    }
  };

  const error = pending?.id === image.id ? pending.error : "";

  return (
    <div ref={controlRef} className="relative">
      <button
        type="button"
        onClick={togglePanel}
        aria-expanded={expanded}
        aria-pressed={markedNotGood}
        aria-label={markedNotGood ? "Edit not good feedback" : "Mark image as not good"}
        className={`flex items-center gap-2 rounded-full border px-4 py-[10px] text-[13px] font-semibold shadow-[var(--shadow-menu)] backdrop-blur transition ${
          markedNotGood
            ? "border-white/20 bg-[color-mix(in_srgb,var(--raised)_94%,transparent)] text-white"
            : "border-white/10 bg-[color-mix(in_srgb,var(--raised)_78%,transparent)] text-zinc-400 hover:border-white/20 hover:text-zinc-200"
        }`}
      >
        <ThumbsDown size={16} fill={markedNotGood ? "currentColor" : "none"} />
        {markedNotGood ? "Marked not good" : "Not good"}
        {!markedNotGood && displayedScore != null && <span className="text-[10px] text-zinc-500">{displayedScore}/5</span>}
      </button>

      {expanded && (
        <div className="absolute bottom-14 right-0 z-10 w-[min(23rem,calc(100vw-2rem))] rounded-[14px] border border-white/10 bg-[color-mix(in_srgb,var(--field)_96%,transparent)] p-4 text-left shadow-[var(--shadow-menu)] backdrop-blur" role="dialog" aria-label="Why is this image not good?">
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="text-[13px] font-bold text-white">What went wrong?</p>
              <p className="mt-1 text-[11px] text-zinc-500">Choose at least one. This saves a 1/5 rating.</p>
            </div>
            <button type="button" onClick={() => setOpenImageId(null)} className="grid h-7 w-7 shrink-0 place-items-center rounded-lg text-zinc-500 transition hover:bg-white/[0.07] hover:text-white" aria-label="Close feedback panel">
              <X size={15} />
            </button>
          </div>

          <div className="mt-3 space-y-3">
            {NEGATIVE_FEEDBACK_GROUPS.map((group) => (
              <div key={group.label}>
                <p className="mb-1.5 text-[9.5px] font-bold uppercase tracking-[0.08em] text-zinc-500">{group.label}</p>
                <div className="flex flex-wrap gap-1.5">
                  {group.reasons.map((reason) => {
                    const selected = selectedReasons.has(reason.code);
                    return (
                      <button
                        key={reason.code}
                        type="button"
                        disabled={saving}
                        aria-pressed={selected}
                        onClick={() => toggleReason(reason.code)}
                        className={`rounded-full px-3 py-1.5 text-[11.5px] font-semibold transition disabled:cursor-wait ${
                          selected
                            ? "bg-white/[0.13] text-white ring-1 ring-inset ring-white/20"
                            : "bg-[var(--raised)] text-[var(--text-muted)] hover:bg-[var(--raised-hover)] hover:text-white"
                        }`}
                      >
                        {reason.label}
                      </button>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>

          {error && <p className="mt-3 text-[11px] text-[var(--danger)]" role="alert">{error}</p>}
          <button type="button" disabled={saving || selectedReasons.size === 0} onClick={submit} className="mt-3 w-full rounded-[9px] bg-white/[0.1] py-[9px] text-[12px] font-bold text-zinc-100 transition hover:bg-white/[0.16] disabled:cursor-not-allowed disabled:opacity-40">
            {saving ? "Saving…" : markedNotGood ? "Update feedback" : "Mark as not good"}
          </button>
        </div>
      )}
    </div>
  );
}

/** PhotoSwipe-backed collection that keeps the app's selected index in sync. */
export default function PhotoSwipeGallery({ entries, selectedIndex, onSelect, className = "", children }) {
  const [isOpen, setIsOpen] = useState(false);
  const [lightboxImage, setLightboxImage] = useState(null);

  const preparePhotoSwipe = useCallback((pswp) => {
    let previousButton;
    let nextButton;
    let opening = true;
    let selectionTimer;

    const updateArrowPositions = () => {
      // Pinned to the viewport edge, not to the slide. Deriving the gap from
      // the displayed image width moved the arrows on every navigation, so
      // they never stayed where the pointer expected them.
      if (!previousButton || !nextButton) return;
      previousButton.style.left = "16px";
      nextButton.style.right = "16px";
    };

    const syncSelection = () => {
      const entry = entries[pswp.currIndex];
      if (!entry) return;
      setLightboxImage(entry.image);
      if (!opening) {
        window.clearTimeout(selectionTimer);
        selectionTimer = window.setTimeout(() => {
          onSelect(entry.index);
        }, 220);
      }
      requestAnimationFrame(() => {
        const slide = pswp.currSlide;
        if (slide && slide.currZoomLevel !== slide.zoomLevels.initial) {
          slide.zoomTo(slide.zoomLevels.initial, { x: 0, y: 0 }, 0);
        }
        updateArrowPositions();
      });
    };

    pswp.on("change", syncSelection);
    pswp.on("afterInit", () => {
      // Mount the app chrome with PhotoSwipe's first painted frame. Waiting for
      // openingAnimationEnd made the image appear first and the overlay pop in
      // afterwards, which looked like a second fullscreen resize.
      const entry = entries[pswp.currIndex];
      if (entry) setLightboxImage(entry.image);
      setIsOpen(true);
    });
    pswp.on("openingAnimationEnd", () => {
      opening = false;
      const entry = entries[pswp.currIndex];
      if (entry) onSelect(entry.index);
    });
    pswp.on("close", () => {
      window.clearTimeout(selectionTimer);
      setIsOpen(false);
      setLightboxImage(null);
    });
    pswp.on("uiRegister", () => {
      pswp.ui.registerElement({
        name: "aiqg-counter",
        className: "pswp__aiqg-counter",
        order: 5,
        onInit: (element) => {
          const update = () => { element.textContent = `${pswp.currIndex + 1} of ${entries.length}`; };
          update();
          pswp.on("change", update);
        },
      });
      pswp.ui.registerElement({
        name: "aiqg-actions",
        className: "pswp__aiqg-actions",
        order: 15,
        html: `
          <button type="button" data-aiqg-action="zoom-out" aria-label="Zoom out" title="Zoom out">${ICONS.minus}</button>
          <button type="button" class="pswp__aiqg-zoom-value" data-aiqg-action="reset" aria-label="Reset zoom" title="Reset zoom">100%</button>
          <button type="button" data-aiqg-action="zoom-in" aria-label="Zoom in" title="Zoom in">${ICONS.plus}</button>
          <a data-aiqg-action="download" aria-label="Download image" title="Download image">${ICONS.download}</a>
          <button type="button" data-aiqg-action="close" aria-label="Close preview" title="Close preview">${ICONS.close}</button>
        `,
        onInit: (element) => {
          const zoomOut = element.querySelector('[data-aiqg-action="zoom-out"]');
          const zoomValue = element.querySelector('[data-aiqg-action="reset"]');
          const zoomIn = element.querySelector('[data-aiqg-action="zoom-in"]');
          const download = element.querySelector('[data-aiqg-action="download"]');
          const close = element.querySelector('[data-aiqg-action="close"]');

          const zoomTo = (target) => {
            const slide = pswp.currSlide;
            if (!slide) return;
            slide.zoomTo(
              target,
              { x: pswp.viewportSize.x / 2, y: pswp.viewportSize.y / 2 },
              180,
            );
          };
          const update = () => {
            const slide = pswp.currSlide;
            const entry = entries[pswp.currIndex];
            if (!slide || !entry) return;
            const initial = slide.zoomLevels.initial;
            const maximum = slide.zoomLevels.max;
            zoomValue.textContent = `${Math.round((slide.currZoomLevel / initial) * 100)}%`;
            zoomOut.disabled = slide.currZoomLevel <= initial + 0.001;
            zoomIn.disabled = slide.currZoomLevel >= maximum - 0.001;
            download.href = entry.image.full_url;
            download.download = `nyx-forge-${entry.image.id}.png`;
          };

          zoomOut.addEventListener("click", () => {
            const slide = pswp.currSlide;
            if (slide) zoomTo(Math.max(slide.zoomLevels.initial, slide.currZoomLevel / 1.3));
          });
          zoomIn.addEventListener("click", () => {
            const slide = pswp.currSlide;
            if (slide) zoomTo(Math.min(slide.zoomLevels.max, slide.currZoomLevel * 1.3));
          });
          zoomValue.addEventListener("click", () => {
            const slide = pswp.currSlide;
            if (slide) zoomTo(slide.zoomLevels.initial);
          });
          close.addEventListener("click", () => pswp.close());
          update();
          pswp.on("zoomPanUpdate", update);
          pswp.on("change", update);
        },
      });
      pswp.ui.registerElement({
        name: "aiqg-previous",
        className: "pswp__aiqg-nav pswp__aiqg-nav--previous",
        order: 20,
        isButton: true,
        appendTo: "wrapper",
        title: "Previous image",
        ariaLabel: "Previous image",
        html: ICONS.previous,
        onClick: () => pswp.prev(),
        onInit: (element) => {
          previousButton = element;
          element.hidden = entries.length <= 1;
          updateArrowPositions();
        },
      });
      pswp.ui.registerElement({
        name: "aiqg-next",
        className: "pswp__aiqg-nav pswp__aiqg-nav--next",
        order: 21,
        isButton: true,
        appendTo: "wrapper",
        title: "Next image",
        ariaLabel: "Next image",
        html: ICONS.next,
        onClick: () => pswp.next(),
        onInit: (element) => {
          nextButton = element;
          element.hidden = entries.length <= 1;
          updateArrowPositions();
        },
      });
    });
    pswp.on("zoomPanUpdate", updateArrowPositions);
    pswp.on("resize", updateArrowPositions);
  }, [entries, onSelect]);

  return (
    <>
      <Gallery options={OPTIONS} onBeforeOpen={preparePhotoSwipe}>
        <div className={className}>
          {entries.map((entry, position) => (
            <Item key={entry.image.id} original={entry.image.full_url} thumbnail={entry.thumbnail ?? entry.image.thumbnail_url} width={entry.image.width} height={entry.image.height}>
              {({ ref, open }) => children({ ...entry, position, ref, open })}
            </Item>
          ))}
        </div>
      </Gallery>
      {isOpen && lightboxImage && (
        <div className="fixed bottom-4 left-1/2 z-[100001] flex -translate-x-1/2 items-center gap-2">
          <FavoriteControl image={lightboxImage} />
          <NotGoodControl image={lightboxImage} />
        </div>
      )}
    </>
  );
}
