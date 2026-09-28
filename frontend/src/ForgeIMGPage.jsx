import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Dialog,
  DialogBackdrop,
  DialogPanel,
  DialogTitle,
} from "@headlessui/react";
import {
  AlertTriangle,
  Brush,
  Check,
  ChevronDown,
  Eraser,
  ImageUp,
  Link,
  LoaderCircle,
  RotateCcw,
  ShieldCheck,
  Shirt,
  Sparkles,
  Trash2,
  UserRoundPlus,
  Users,
  Wrench,
  X,
} from "lucide-react";

import {
  createIdentityProfile,
  deleteIdentityProfile,
  getIdentityProfiles,
  proposeImageMasks,
  suggestIdentityCrop,
  suggestImageTransform,
  uploadImage,
  uploadImageFromUrl,
  validateIdentityCrop,
} from "./api";
import CustomSelect from "./CustomSelect";
import ErrorBanner from "./ErrorBanner";
import ExecutionJobs from "./ExecutionJobs";

const RATINGS = ["Safe"];
const PRESETS = Object.freeze({
  outfit: {
    label: "Change the outfit",
    description: "Preserve the person and replace only their clothing",
    icon: Shirt,
    denoise: 0.65,
    prompt: "same person, identity unchanged",
    input: "garment",
    detector: "garment",
    regionOnly: "region",
    inpaintPadding: 128,
  },
  background: {
    label: "Change the background",
    description: "Keep the subject and rebuild the environment",
    icon: ImageUp,
    denoise: 0.75,
    prompt: "empty scene, no people",
    input: "background",
    detector: "person",
    invert: true,
    regionOnly: "scene",
    inpaintPadding: 128,
  },
  variation: {
    label: "New scene, same character",
    description: "Create a new canvas from a validated Face ID",
    icon: Users,
    denoise: 0.8,
    prompt: "same character",
    input: "scene",
    faceIdentity: true,
  },
  restore: {
    label: "Restore / modernise",
    description: "Sharpen an old or soft photograph",
    icon: Wrench,
    denoise: 0.35,
    prompt: "restored photograph, sharp modern detail, natural colour",
  },
  anime: {
    label: "Convert to anime",
    description: "Redraw the source in an illustrated style",
    icon: Sparkles,
    denoise: 0.55,
    prompt: "anime illustration, clean line art, expressive detail",
    style: "Anime",
  },
  repair: {
    label: "Fix a region",
    description: "Repair a hand, face, or local artifact",
    icon: Brush,
    denoise: 0.4,
    prompt:
      "repair anatomy and artifacts, preserve everything outside the selected region",
    detector: "features",
  },
});

function sourcePayload(source) {
  return source?.kind === "upload"
    ? { upload_id: source.upload_id }
    : {
        source_generation_id: source?.id,
        access_token: source?.access_token ?? null,
      };
}

function sourcePreview(source) {
  return (
    source?.full_url ??
    source?.preview_url ??
    source?.thumbnail_url ??
    (source?.upload_id ? `/api/uploads/${source.upload_id}` : "")
  );
}

function sourceName(source) {
  return source?.kind === "upload"
    ? `Uploaded · ${source.name ?? "image"}`
    : `Generation #${source?.id}`;
}

function MaskCanvas({ source, mask, onMaskChange, tool, brushSize }) {
  const canvasRef = useRef(null);
  const drawing = useRef(false);
  const lastPoint = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !source) return;
    canvas.width = source.width;
    canvas.height = source.height;
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!mask) return;
    const image = new Image();
    image.onload = () =>
      context.drawImage(image, 0, 0, canvas.width, canvas.height);
    image.src = `data:image/png;base64,${mask}`;
  }, [mask, source]);
  const pointFor = (event) => {
    const bounds = canvasRef.current.getBoundingClientRect();
    return {
      x:
        ((event.clientX - bounds.left) * canvasRef.current.width) /
        bounds.width,
      y:
        ((event.clientY - bounds.top) * canvasRef.current.height) /
        bounds.height,
    };
  };
  const draw = (event) => {
    if (!drawing.current) return;
    const canvas = canvasRef.current;
    const context = canvas.getContext("2d");
    const point = pointFor(event);
    const radius = (brushSize * canvas.width) / Math.max(canvas.clientWidth, 1);
    context.save();
    context.globalCompositeOperation =
      tool === "erase" ? "destination-out" : "source-over";
    context.strokeStyle = "white";
    context.lineWidth = radius * 2;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.beginPath();
    context.moveTo(
      lastPoint.current?.x ?? point.x,
      lastPoint.current?.y ?? point.y,
    );
    context.lineTo(point.x, point.y);
    context.stroke();
    context.restore();
    lastPoint.current = point;
  };
  const finish = () => {
    if (!drawing.current) return;
    drawing.current = false;
    lastPoint.current = null;
    onMaskChange(canvasRef.current.toDataURL("image/png").split(",")[1]);
  };
  return (
    <canvas
      ref={canvasRef}
      aria-label="Editable mask"
      className="absolute inset-0 h-full w-full cursor-crosshair touch-none opacity-55"
      onPointerDown={(event) => {
        drawing.current = true;
        event.currentTarget.setPointerCapture(event.pointerId);
        draw(event);
      }}
      onPointerMove={draw}
      onPointerUp={finish}
      onPointerCancel={finish}
    />
  );
}

function SourceCard({
  source,
  onOpen,
  mask,
  editingMask,
  onMaskChange,
  tool,
  brushSize,
}) {
  if (!source)
    return (
      <section className="nyx-forgeimg-step nyx-forgeimg-source">
        <span className="forge-eyebrow">Step 2 · Target asset</span>
        <div className="nyx-forgeimg-dropzone">
          <span className="nyx-forgeimg-dropzone-icon">
            <ImageUp size={19} />
          </span>
          <h2>Choose the full image</h2>
          <p>
            Forge needs the complete person or scene so it can preserve pixels
            outside the edit.
          </p>
          <button
            type="button"
            onClick={onOpen}
            className="forge-btn-primary mx-auto mt-4"
          >
            <ImageUp size={15} />
            Select image
          </button>
        </div>
      </section>
    );
  return (
    <section className="nyx-forgeimg-step nyx-forgeimg-source">
      <div className="nyx-forgeimg-step-heading">
        <span className="forge-eyebrow">Step 2 · Target asset</span>
        <button
          type="button"
          onClick={onOpen}
          className="nyx-forgeimg-secondary-action"
        >
          Change image
        </button>
      </div>
      <div className="nyx-forgeimg-source-preview">
        <img
          src={sourcePreview(source)}
          alt="Selected transform source"
          className="h-full w-full object-contain"
        />
        {editingMask ? (
          <MaskCanvas
            source={source}
            mask={mask}
            onMaskChange={onMaskChange}
            tool={tool}
            brushSize={brushSize}
          />
        ) : (
          mask && (
            <img
              src={`data:image/png;base64,${mask}`}
              alt="Proposed editable region"
              className="pointer-events-none absolute inset-0 h-full w-full object-contain opacity-55"
            />
          )
        )}
      </div>
      <div className="nyx-forgeimg-source-meta">
        <div>
          <strong>{sourceName(source)}</strong>
          <span>
            {source.width} × {source.height}
          </span>
        </div>
        <p>
          {editingMask
            ? "Draw over pixels Forge should regenerate."
            : mask
              ? "Accent overlay is the proposed edit area."
              : "Ready for transformation."}
        </p>
      </div>
    </section>
  );
}

function SourcePickerDialog({
  open,
  onClose,
  currentSource,
  images,
  onUse,
  onUpload,
  onUrlUpload,
  uploading,
  uploadProgress,
  hasMore,
  loadingMore,
  onLoadMore,
}) {
  const [tab, setTab] = useState("gallery");
  const [candidate, setCandidate] = useState(currentSource);
  const [url, setUrl] = useState("");
  const [error, setError] = useState("");
  const inputRef = useRef(null);
  useEffect(() => {
    if (open) {
      setCandidate(currentSource);
      setError("");
    }
  }, [currentSource, open]);
  const accept = async (work) => {
    setError("");
    try {
      setCandidate(await work);
    } catch (requestError) {
      setError(requestError.message);
    }
  };
  return (
    <Dialog
      open={open}
      onClose={uploading ? () => {} : onClose}
      className="relative z-[100]"
    >
      <DialogBackdrop className="fixed inset-0 bg-[var(--scrim)] backdrop-blur-[2px]" />
      <div className="fixed inset-0 overflow-y-auto p-4">
        <div className="flex min-h-full items-center justify-center">
          <DialogPanel className="flex h-[min(820px,calc(100vh-32px))] w-full max-w-[1180px] flex-col overflow-hidden rounded-[18px] border border-[var(--divider)] bg-[var(--raised)] shadow-[var(--shadow-modal)]">
            <header className="flex items-start justify-between border-b border-[var(--divider)] px-7 py-5">
              <div>
                <span className="forge-eyebrow">Source library</span>
                <DialogTitle className="mt-1 text-[20px] font-bold text-[var(--text)]">
                  Select an image
                </DialogTitle>
                <p className="forge-body mt-1">
                  Choose recent work or import a private source.
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="rounded-[8px] p-2 text-[var(--text-muted)] hover:bg-[var(--field-hover)]"
              >
                <X size={18} />
              </button>
            </header>
            <div className="forge-seg mx-7 mt-5 grid grid-cols-3">
              {[
                ["gallery", "Gallery"],
                ["upload", "Upload"],
                ["link", "Image link"],
              ].map(([value, label]) => (
                <button
                  key={value}
                  type="button"
                  aria-selected={tab === value}
                  onClick={() => setTab(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            <div className="grid min-h-0 flex-1 gap-5 p-7 md:grid-cols-[minmax(0,1.5fr)_minmax(300px,.8fr)]">
              <section className="min-h-0 overflow-hidden rounded-[12px] border border-[var(--divider)] bg-[var(--well)] p-5">
                {tab === "gallery" && (
                  <div className="flex h-full flex-col">
                    <div className="mb-4 flex justify-between">
                      <div>
                        <h3 className="text-[14px] font-bold text-[var(--text)]">
                          Recent generations
                        </h3>
                        <p className="mt-1 text-[10.5px] text-[var(--text-muted)]">
                          Choose one image to inspect.
                        </p>
                      </div>
                      <span className="text-[10px] text-[var(--text-subtle)]">
                        {images.length} loaded
                      </span>
                    </div>
                    <div className="min-h-0 flex-1 overflow-y-auto pr-2 scrollbar-subtle">
                      <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
                        {images.map((image) => {
                          const selected =
                            candidate?.kind !== "upload" &&
                            candidate?.id === image.id;
                          return (
                            <button
                              key={image.id}
                              type="button"
                              onClick={() => setCandidate(image)}
                              className={`relative overflow-hidden rounded-[10px] bg-[var(--field)] text-left ${selected ? "ring-2 ring-[var(--accent)]" : "hover:ring-1 hover:ring-[var(--divider)]"}`}
                            >
                              <span className="block aspect-[4/3] overflow-hidden">
                                <img
                                  src={image.thumbnail_url}
                                  alt=""
                                  className="h-full w-full object-cover"
                                  loading="lazy"
                                />
                                {selected && (
                                  <span className="absolute right-2 top-2 grid h-6 w-6 place-items-center rounded-full bg-[var(--accent)] text-white">
                                    <Check size={14} />
                                  </span>
                                )}
                                <span className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 px-2 pb-2 pt-8 text-[10px] font-bold text-white">
                                  Generation #{image.id}
                                </span>
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                    {hasMore && (
                      <button
                        type="button"
                        onClick={onLoadMore}
                        disabled={loadingMore}
                        className="mt-4 rounded-[9px] bg-[var(--field)] py-2.5 text-xs font-bold text-[var(--accent-text)]"
                      >
                        {loadingMore ? "Loading…" : "Load more"}
                      </button>
                    )}
                  </div>
                )}
                {tab === "upload" && (
                  <button
                    type="button"
                    onClick={() => inputRef.current?.click()}
                    className="flex h-full min-h-[400px] w-full flex-col items-center justify-center rounded-[12px] border border-dashed border-[color-mix(in_srgb,var(--divider)_70%,transparent)] bg-[var(--field)] p-8 text-center"
                  >
                    <span className="grid h-14 w-14 place-items-center rounded-full bg-[var(--accent-wash)] text-[var(--accent-text)]">
                      {uploading ? (
                        <LoaderCircle className="animate-spin" />
                      ) : (
                        <ImageUp />
                      )}
                    </span>
                    <strong className="mt-5 text-[14px] text-[var(--text)]">
                      {uploading
                        ? `Uploading ${uploadProgress}%`
                        : "Choose a PNG, JPEG, or WebP"}
                    </strong>
                    <span className="mt-2 text-[11px] text-[var(--text-muted)]">
                      25 MB maximum
                    </span>
                    <input
                      ref={inputRef}
                      type="file"
                      accept="image/png,image/jpeg,image/webp"
                      className="hidden"
                      onChange={(event) => {
                        const file = event.target.files?.[0];
                        if (file) accept(onUpload(file));
                        event.target.value = "";
                      }}
                    />
                  </button>
                )}
                {tab === "link" && (
                  <div className="rounded-[12px] bg-[var(--field)] p-5">
                    <span className="grid h-12 w-12 place-items-center rounded-full bg-[var(--accent-wash)] text-[var(--accent-text)]">
                      <Link size={20} />
                    </span>
                    <label className="mt-5 block text-[11px] font-semibold text-[var(--text-ui)]">
                      Public HTTPS image URL
                      <input
                        type="url"
                        value={url}
                        onChange={(event) => setUrl(event.target.value)}
                        className="mt-2 w-full rounded-[9px] bg-[var(--well)] px-3 py-3 text-xs outline-none focus:ring-2 focus:ring-[var(--accent)]"
                      />
                    </label>
                    <button
                      type="button"
                      disabled={!url.trim() || uploading}
                      onClick={() => accept(onUrlUpload(url.trim()))}
                      className="forge-btn-primary mt-3 w-full"
                    >
                      <Link size={15} />
                      Import image
                    </button>
                  </div>
                )}
                {error && (
                  <p className="mt-3 rounded-[9px] bg-[var(--danger-wash)] p-3 text-[11px] text-[var(--danger)]">
                    {error}
                  </p>
                )}
              </section>
              <section className="flex min-h-[400px] flex-col rounded-[12px] border border-[var(--divider)] bg-[var(--well)] p-5">
                <span className="forge-eyebrow">Preview</span>
                {candidate ? (
                  <>
                    <div className="mt-4 min-h-0 flex-1 overflow-hidden rounded-[10px] bg-[var(--canvas)]">
                      <img
                        src={sourcePreview(candidate)}
                        alt="Selected source preview"
                        className="h-full w-full object-contain"
                      />
                    </div>
                    <strong className="mt-3 truncate text-[12px] text-[var(--text)]">
                      {sourceName(candidate)}
                    </strong>
                    <span className="mt-1 text-[10px] text-[var(--text-muted)]">
                      {candidate.width} × {candidate.height}
                    </span>
                  </>
                ) : (
                  <div className="mt-4 grid flex-1 place-items-center rounded-[10px] border border-dashed border-[color-mix(in_srgb,var(--divider)_70%,transparent)] text-center text-[11px] text-[var(--text-muted)]">
                    Nothing selected yet
                  </div>
                )}
              </section>
            </div>
            <footer className="flex items-center justify-end gap-2 border-t border-[var(--divider)] px-7 py-4">
              <button
                type="button"
                onClick={onClose}
                className="rounded-[9px] bg-[var(--field)] px-4 py-2.5 text-xs font-bold text-[var(--text-ui)]"
              >
                Cancel
              </button>
              <button
                type="button"
                disabled={!candidate || uploading}
                onClick={() => onUse(candidate)}
                className="forge-btn-primary px-5"
              >
                <Check size={14} />
                Use this image
              </button>
            </footer>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}

function FaceCropCanvas({ source, crop, onChange }) {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const dragRef = useRef(null);
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !crop) return;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    context.drawImage(image, 0, 0);
    const x = crop.x * canvas.width;
    const y = crop.y * canvas.height;
    const width = crop.width * canvas.width;
    const height = crop.height * canvas.height;
    context.fillStyle = "rgba(5,8,11,.72)";
    context.beginPath();
    context.rect(0, 0, canvas.width, canvas.height);
    context.rect(x, y, width, height);
    context.fill("evenodd");
    context.strokeStyle =
      getComputedStyle(document.documentElement)
        .getPropertyValue("--nyx-accent")
        .trim() || "#f5d90a";
    context.lineWidth = Math.max(3, canvas.width / 320);
    context.strokeRect(x, y, width, height);
  }, [crop]);
  useEffect(() => {
    const image = new Image();
    image.onload = () => {
      imageRef.current = image;
      draw();
    };
    image.src = sourcePreview(source);
  }, [source]);
  useEffect(draw, [draw]);
  const point = (event) => {
    const rect = canvasRef.current.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / rect.width,
      y: (event.clientY - rect.top) / rect.height,
    };
  };
  return (
    <div>
      <canvas
        ref={canvasRef}
        className="mx-auto block h-auto max-h-[460px] max-w-full touch-none rounded-[12px] bg-[var(--canvas)]"
        onPointerDown={(event) => {
          const p = point(event);
          if (
            p.x < crop.x ||
            p.x > crop.x + crop.width ||
            p.y < crop.y ||
            p.y > crop.y + crop.height
          )
            return;
          dragRef.current = { point: p, crop };
          event.currentTarget.setPointerCapture(event.pointerId);
        }}
        onPointerMove={(event) => {
          if (!dragRef.current) return;
          const p = point(event);
          const dx = p.x - dragRef.current.point.x;
          const dy = p.y - dragRef.current.point.y;
          onChange({
            ...crop,
            x: Math.max(
              0,
              Math.min(1 - crop.width, dragRef.current.crop.x + dx),
            ),
            y: Math.max(
              0,
              Math.min(1 - crop.height, dragRef.current.crop.y + dy),
            ),
          });
        }}
        onPointerUp={() => {
          dragRef.current = null;
        }}
        onPointerCancel={() => {
          dragRef.current = null;
        }}
      />
      <label className="mt-4 block text-[11px] font-semibold text-[var(--text-ui)]">
        <span className="flex justify-between">
          <span>Crop size</span>
          <span>Drag the frame to position</span>
        </span>
        <input
          type="range"
          min="25"
          max="95"
          value={Math.round(
            (Math.min(crop.width * source.width, crop.height * source.height) /
              Math.min(source.width, source.height)) *
              100,
          )}
          onChange={(event) => {
            const ratio = Number(event.target.value) / 100;
            const pixels = Math.min(source.width, source.height) * ratio;
            const width = pixels / source.width;
            const height = pixels / source.height;
            const cx = crop.x + crop.width / 2;
            const cy = crop.y + crop.height / 2;
            onChange({
              x: Math.max(0, Math.min(1 - width, cx - width / 2)),
              y: Math.max(0, Math.min(1 - height, cy - height / 2)),
              width,
              height,
            });
          }}
          className="mt-3 w-full accent-[var(--accent)]"
        />
      </label>
    </div>
  );
}

function FaceProfileDialog({ open, source, initialCrop, onClose, onSaved }) {
  const [crop, setCrop] = useState(initialCrop);
  const [name, setName] = useState("");
  const [validation, setValidation] = useState(null);
  const [checking, setChecking] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    if (open) {
      setCrop(initialCrop);
      setName("");
      setValidation(null);
      setError("");
    }
  }, [initialCrop, open]);
  useEffect(() => {
    if (!open || !source || !crop) return undefined;
    setChecking(true);
    const timer = window.setTimeout(async () => {
      try {
        setValidation(
          await validateIdentityCrop({ ...sourcePayload(source), crop }),
        );
        setError("");
      } catch (requestError) {
        setValidation(null);
        setError(requestError.message);
      } finally {
        setChecking(false);
      }
    }, 650);
    return () => window.clearTimeout(timer);
  }, [crop, open, source]);
  const save = async () => {
    if (!validation?.usable || !name.trim()) return;
    setSaving(true);
    setError("");
    try {
      const profile = await createIdentityProfile({
        ...sourcePayload(source),
        crop,
        name: name.trim(),
      });
      onSaved(profile);
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setSaving(false);
    }
  };
  return (
    <Dialog
      open={open}
      onClose={saving ? () => {} : onClose}
      className="relative z-[110]"
    >
      <DialogBackdrop className="fixed inset-0 bg-[var(--scrim)] backdrop-blur-[2px]" />
      <div className="fixed inset-0 overflow-y-auto p-4">
        <div className="flex min-h-full items-center justify-center">
          <DialogPanel className="w-full max-w-[980px] rounded-[18px] border border-[var(--divider)] bg-[var(--raised)] shadow-[var(--shadow-modal)]">
            <header className="flex items-start justify-between border-b border-[var(--divider)] px-7 py-5">
              <div>
                <span className="forge-eyebrow">New Face ID</span>
                <DialogTitle className="mt-1 text-[20px] font-bold text-[var(--text)]">
                  Frame one clear identity
                </DialogTitle>
                <p className="forge-body mt-1">
                  Keep the whole face visible with a little head context.
                </p>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="rounded-[8px] p-2 text-[var(--text-muted)] hover:bg-[var(--field-hover)]"
              >
                <X size={18} />
              </button>
            </header>
            <div className="grid gap-6 p-7 md:grid-cols-[minmax(0,1.35fr)_minmax(280px,.65fr)]">
              <section>
                {source && crop && (
                  <FaceCropCanvas
                    source={source}
                    crop={crop}
                    onChange={setCrop}
                  />
                )}
              </section>
              <aside className="rounded-[12px] bg-[var(--well)] p-5">
                <span className="forge-eyebrow">Live validation</span>
                <div
                  className={`mt-4 rounded-[10px] p-4 ${checking ? "bg-[var(--field)]" : validation?.status === "ready" ? "bg-[var(--success-wash)]" : validation?.usable ? "bg-[var(--warning-wash)]" : "bg-[var(--danger-wash)]"}`}
                >
                  <div className="flex items-center gap-2">
                    {checking ? (
                      <LoaderCircle size={17} className="animate-spin" />
                    ) : validation?.status === "ready" ? (
                      <ShieldCheck size={17} />
                    ) : (
                      <AlertTriangle size={17} />
                    )}
                    <strong className="text-[12px] text-[var(--text)]">
                      {checking
                        ? "Checking face…"
                        : validation?.status === "ready"
                          ? "Excellent reference"
                          : validation?.usable
                            ? "Usable reference"
                            : "Adjust the crop"}
                    </strong>
                  </div>
                  {validation?.reasons?.length > 0 && (
                    <ul className="mt-3 space-y-1 text-[10.5px] leading-4 text-[var(--text-muted)]">
                      {validation.reasons.map((reason) => (
                        <li key={reason}>• {reason}</li>
                      ))}
                    </ul>
                  )}
                </div>
                {validation?.metrics && (
                  <dl className="mt-4 grid grid-cols-2 gap-2 text-[10px]">
                    <div className="rounded-[8px] bg-[var(--field)] p-3">
                      <dt className="text-[var(--text-subtle)]">Face pixels</dt>
                      <dd className="mt-1 font-bold text-[var(--text-ui)]">
                        {validation.metrics.face_pixel_size?.join(" × ")}
                      </dd>
                    </div>
                    <div className="rounded-[8px] bg-[var(--field)] p-3">
                      <dt className="text-[var(--text-subtle)]">Confidence</dt>
                      <dd className="mt-1 font-bold text-[var(--text-ui)]">
                        {Math.round(
                          (validation.metrics.detector_confidence ?? 0) * 100,
                        )}
                        %
                      </dd>
                    </div>
                  </dl>
                )}
                <label className="mt-5 block text-[11px] font-semibold text-[var(--text-ui)]">
                  Face ID name
                  <input
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    maxLength={60}
                    placeholder="e.g. Samantha — studio reference"
                    className="mt-2 w-full rounded-[9px] bg-[var(--field)] px-3 py-3 text-xs outline-none focus:ring-2 focus:ring-[var(--accent)]"
                  />
                </label>
                {error && (
                  <p className="mt-3 rounded-[8px] bg-[var(--danger-wash)] p-3 text-[10.5px] text-[var(--danger)]">
                    {error}
                  </p>
                )}
                <button
                  type="button"
                  disabled={!validation?.usable || !name.trim() || saving}
                  onClick={save}
                  className="forge-btn-primary mt-5 w-full"
                >
                  {saving ? (
                    <LoaderCircle size={15} className="animate-spin" />
                  ) : (
                    <ShieldCheck size={15} />
                  )}
                  {saving ? "Saving Face ID…" : "Save Face ID"}
                </button>
              </aside>
            </div>
          </DialogPanel>
        </div>
      </div>
    </Dialog>
  );
}

function PromptEngineSelector({
  settings,
  onChange,
  visionModels,
  activeVisionModel,
  cloudProviders,
  cloudCredentials,
}) {
  const activeCloud = cloudCredentials?.[settings?.cloud_provider];
  return (
    <div className="mb-3">
      <div className="forge-seg grid grid-cols-3">
        {[
          ["curated", "Curated"],
          ["ollama", "Ollama"],
          ["cloud", "Cloud API"],
        ].map(([value, label]) => (
          <button
            key={value}
            type="button"
            disabled={
              (value === "ollama" && !visionModels.length) ||
              (value === "cloud" && !cloudProviders.length)
            }
            aria-selected={settings.prompt_engine === value}
            onClick={() => onChange({ prompt_engine: value })}
            className="disabled:cursor-not-allowed disabled:opacity-40"
          >
            {label}
          </button>
        ))}
      </div>
      {settings.prompt_engine === "ollama" && visionModels.length > 0 && (
        <div className="mt-2">
          <CustomSelect
            value={activeVisionModel}
            options={visionModels}
            onChange={(ollama_model) => onChange({ ollama_model })}
            ariaLabel="ForgeIMG vision model"
          />
        </div>
      )}
      {settings.prompt_engine === "cloud" && cloudProviders.length > 0 && (
        <div className="mt-2 grid grid-cols-2 gap-2">
          <CustomSelect
            value={settings.cloud_provider}
            options={cloudProviders.map(([value]) => ({ value, label: value }))}
            onChange={(cloud_provider) => {
              const credential = cloudCredentials[cloud_provider];
              onChange({
                cloud_provider,
                cloud_model:
                  credential?.selectedModel ??
                  credential?.recommended ??
                  credential?.models?.[0] ??
                  "",
              });
            }}
            ariaLabel="Cloud provider"
          />
          {activeCloud?.models?.length > 0 && (
            <CustomSelect
              value={settings.cloud_model}
              options={activeCloud.models}
              onChange={(cloud_model) => onChange({ cloud_model })}
              ariaLabel="Cloud model"
            />
          )}
        </div>
      )}
    </div>
  );
}

export default function ForgeIMGPage({
  images,
  models,
  modelProfiles,
  connected,
  loading,
  error,
  onDismissError,
  onSubmit,
  refreshSignal,
  account,
  onNavigate,
  hasMore,
  loadingMore,
  onLoadMore,
  promptModels,
  ollamaConnected,
  cloudCredentials,
  promptSettings,
  onPromptSettingsChange,
}) {
  const [preset, setPreset] = useState(null);
  const config = preset ? PRESETS[preset] : null;
  const [source, setSource] = useState(null);
  const [profile, setProfile] = useState(null);
  const [profiles, setProfiles] = useState([]);
  const [pickerOpen, setPickerOpen] = useState(false);
  const [pickerPurpose, setPickerPurpose] = useState("source");
  const [cropDialog, setCropDialog] = useState(false);
  const [cropSource, setCropSource] = useState(null);
  const [crop, setCrop] = useState(null);
  const [model, setModel] = useState(models[0] ?? "Forge default");
  const activeProfile = modelProfiles[model];
  const [style, setStyle] = useState("Photoreal");
  const [contentRating, setContentRating] = useState("Safe");
  const [denoise, setDenoise] = useState(0.4);
  const [cfgScale, setCfgScale] = useState(null);
  const [prompt, setPrompt] = useState("");
  const [negativePrompt, setNegativePrompt] = useState("");
  const [contextInput, setContextInput] = useState("");
  const [mask, setMask] = useState(null);
  const [maskBusy, setMaskBusy] = useState(false);
  const [maskError, setMaskError] = useState("");
  const [proposals, setProposals] = useState([]);
  const [maskBlur, setMaskBlur] = useState(8);
  const [inpaintPadding, setInpaintPadding] = useState(32);
  const [inpaintFullRes, setInpaintFullRes] = useState(true);
  const [softInpainting, setSoftInpainting] = useState(false);
  const [invertMask, setInvertMask] = useState(false);
  const [tool, setTool] = useState("brush");
  const [brushSize, setBrushSize] = useState(28);
  const [moreOpen, setMoreOpen] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [suggesting, setSuggesting] = useState(false);
  const [suggestionNotice, setSuggestionNotice] = useState("");
  const [profileError, setProfileError] = useState("");
  const maskCache = useRef(new Map());
  const maskRequest = useRef(0);
  const minCfg = activeProfile?.min_cfg_scale ?? 3;
  const faceIdentityModel = useMemo(
    () =>
      models.find(
        (item) => modelProfiles[item]?.supports_face_identity === true,
      ),
    [modelProfiles, models],
  );
  const ponyModel = useMemo(
    () =>
      models.find((item) =>
        `${item} ${modelProfiles[item]?.display_name ?? ""}`
          .toLowerCase()
          .includes("pony"),
      ),
    [modelProfiles, models],
  );
  const availableRatings = RATINGS;
  const configuredCloudProviders = useMemo(
    () =>
      Object.entries(cloudCredentials ?? {}).filter(
        ([, value]) => value?.apiKey,
      ),
    [cloudCredentials],
  );
  const activeCloud = cloudCredentials?.[promptSettings?.cloud_provider];
  const visionPromptModels = useMemo(
    () =>
      promptModels.filter((name) =>
        /gemma(?:3|4)|qwen.*vl|llama.*vision|llava|minicpm.*v/i.test(name),
      ),
    [promptModels],
  );
  const activeVisionModel = visionPromptModels.includes(
    promptSettings?.ollama_model,
  )
    ? promptSettings.ollama_model
    : visionPromptModels[0];
  const loadProfiles = useCallback(async () => {
    if (!account) return;
    try {
      setProfiles(await getIdentityProfiles());
      setProfileError("");
    } catch (requestError) {
      setProfileError(requestError.message);
    }
  }, [account]);
  useEffect(() => {
    loadProfiles();
  }, [loadProfiles]);
  useEffect(() => {
    if (!models.includes(model) && models[0]) setModel(models[0]);
  }, [model, models]);
  const chooseModel = (next) => {
    const nextProfile = modelProfiles[next];
    setModel(next);
    setStyle((current) =>
      nextProfile?.styles?.includes(current)
        ? current
        : (nextProfile?.default_style ?? nextProfile?.styles?.[0] ?? current),
    );
    setCfgScale((current) =>
      current == null
        ? null
        : Math.max(current, nextProfile?.min_cfg_scale ?? 3),
    );
  };
  const fetchMask = useCallback(async (nextPreset, nextSource) => {
    const nextConfig = PRESETS[nextPreset];
    if (!nextConfig?.detector || !nextSource) return;
    const sourceKey =
      nextSource.kind === "upload"
        ? `upload:${nextSource.upload_id}`
        : `generation:${nextSource.id}`;
    const cacheKey = `${sourceKey}:${nextConfig.detector}`;
    const requestId = ++maskRequest.current;
    const apply = (items) => {
      setProposals(items);
      setMask(items[0]?.mask ?? null);
      if (!items.length)
        setMaskError(
          "No suitable region was detected. Open Edit mask to draw it manually.",
        );
    };
    const cached = maskCache.current.get(cacheKey);
    if (cached) {
      apply(cached);
      return;
    }
    setMaskBusy(true);
    setMaskError("");
    try {
      const result = await proposeImageMasks({
        ...sourcePayload(nextSource),
        detector: nextConfig.detector,
      });
      maskCache.current.set(cacheKey, result.proposals);
      if (maskRequest.current === requestId) apply(result.proposals);
    } catch (requestError) {
      if (maskRequest.current === requestId) setMaskError(requestError.message);
    } finally {
      if (maskRequest.current === requestId) setMaskBusy(false);
    }
  }, []);
  const choosePreset = (next) => {
    const nextConfig = PRESETS[next];
    setPreset(next);
    setDenoise(nextConfig.denoise);
    setPrompt(nextConfig.prompt);
    setNegativePrompt("");
    setContextInput("");
    setContentRating("Safe");
    setInvertMask(Boolean(nextConfig.invert));
    setInpaintPadding(nextConfig.inpaintPadding ?? 32);
    setMask(null);
    setProposals([]);
    setMaskError("");
    setMoreOpen(false);
    if (nextConfig.style) setStyle(nextConfig.style);
    if (nextConfig.faceIdentity && faceIdentityModel)
      chooseModel(faceIdentityModel);
    if (next === "anime" && ponyModel) chooseModel(ponyModel);
    if (nextConfig.faceIdentity) setSource(null);
    else {
      setProfile(null);
      if (source && nextConfig.detector) fetchMask(next, source);
    }
  };
  const selectPickerImage = async (next) => {
    setPickerOpen(false);
    if (pickerPurpose === "identity") {
      setCropSource(next);
      setProfileError("");
      try {
        const result = await suggestIdentityCrop(sourcePayload(next));
        setCrop(result.crop);
        setCropDialog(true);
      } catch (requestError) {
        setProfileError(requestError.message);
      }
      return;
    }
    setSource(next);
    setContentRating("Safe");
    setMask(null);
    setProposals([]);
    setMaskError("");
    if (config?.detector) fetchMask(preset, next);
  };
  const handleUpload = async (file) => {
    setUploading(true);
    setUploadProgress(0);
    try {
      const uploaded = await uploadImage(file, setUploadProgress);
      return {
        kind: "upload",
        ...uploaded,
        name: file.name,
        preview_url: `/api/uploads/${uploaded.upload_id}`,
      };
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };
  const handleUrlUpload = async (url) => {
    setUploading(true);
    try {
      const uploaded = await uploadImageFromUrl(url);
      return {
        kind: "upload",
        ...uploaded,
        name: "linked-image",
        preview_url: `/api/uploads/${uploaded.upload_id}`,
      };
    } finally {
      setUploading(false);
    }
  };
  const generateSuggestion = async () => {
    if (!config?.input) return;
    const suggestionSource = source ?? cropSource;
    if (!suggestionSource && !profile) {
      setSuggestionNotice("Choose an asset first.");
      return;
    }
    setSuggesting(true);
    setSuggestionNotice("");
    try {
      const result = await suggestImageTransform({
        ...(profile && config.faceIdentity
          ? { identity_profile_id: profile.id }
          : sourcePayload(suggestionSource)),
        kind:
          config.input === "background"
            ? "background"
            : config.input === "scene"
              ? "scene"
              : "outfit",
        prompt_engine: promptSettings.prompt_engine,
        ollama_model: activeVisionModel || null,
        cloud_provider: promptSettings.cloud_provider || null,
        cloud_model: promptSettings.cloud_model || null,
        cloud_api_key: activeCloud?.apiKey ?? null,
      });
      setContextInput(result.suggestion);
      setSuggestionNotice(result.notice ?? `Created with ${result.engine}.`);
    } catch (requestError) {
      setSuggestionNotice(requestError.message);
    } finally {
      setSuggesting(false);
    }
  };
  const needsMask = Boolean(config?.detector);
  const assetReady = config?.faceIdentity ? Boolean(profile) : Boolean(source);
  const canSubmit = Boolean(
    config &&
      assetReady &&
      connected &&
      !loading &&
      !maskBusy &&
      (!config.input || contextInput.trim()) &&
      (!needsMask || mask),
  );
  const submit = () => {
    if (!canSubmit) return;
    const contextual = config.input
      ? `${config.input === "background" ? "background changed to" : config.input === "scene" ? "new scene:" : "outfit changed to"} ${contextInput.trim()}, `
      : "";
    onSubmit({
      ...(config.faceIdentity
        ? { identity_profile_id: profile.id }
        : sourcePayload(source)),
      prompt: `${contextual}${prompt}`,
      negative_prompt: negativePrompt,
      model,
      style,
      content_rating: contentRating,
      preset,
      region_only: config.regionOnly ?? "",
      face_identity: Boolean(config.faceIdentity),
      denoising_strength: denoise,
      cfg_scale: cfgScale,
      prompt_engine: "manual",
      request_id: crypto.randomUUID(),
      mask: config.faceIdentity ? null : mask,
      mask_blur: maskBlur,
      inpaint_full_res: inpaintFullRes,
      inpaint_full_res_padding: inpaintPadding,
      invert_mask: invertMask,
      soft_inpainting: softInpainting,
    });
  };
  const invertCanvasMask = () => {
    if (!source) return;
    const canvas = document.createElement("canvas");
    canvas.width = source.width;
    canvas.height = source.height;
    const context = canvas.getContext("2d");
    context.fillStyle = "white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    if (!mask) {
      setMask(canvas.toDataURL("image/png").split(",")[1]);
      return;
    }
    const image = new Image();
    image.onload = () => {
      context.globalCompositeOperation = "destination-out";
      context.drawImage(image, 0, 0);
      setMask(canvas.toDataURL("image/png").split(",")[1]);
    };
    image.src = `data:image/png;base64,${mask}`;
  };

  return (
    <div className="nyx-forgeimg-page">
      <main className="nyx-forgeimg-layout">
        <section className="nyx-forgeimg-compose">
          <header className="nyx-forgeimg-page-heading">
            <span>
              <i />
              ForgeIMG workbench
            </span>
            <h1>
              <b>&gt;_</b> Image transforms
            </h1>
            <p>
              Choose an operation, provide only the asset it needs, and keep
              full control over the transform.
            </p>
          </header>
          <ErrorBanner
            error={error}
            onDismiss={onDismissError}
            className="mb-3"
          />
          <div className="nyx-forgeimg-form scrollbar-subtle">
            <section className="nyx-forgeimg-step">
              <div className="nyx-forgeimg-step-heading">
                <div>
                  <span className="forge-eyebrow">Step 1 · Choose action</span>
                  <h2>What should Forge change?</h2>
                </div>
                {preset && (
                  <span className="nyx-forgeimg-selected">
                    <Check size={11} />
                    Selected
                  </span>
                )}
              </div>
              <div className="nyx-forgeimg-action-grid">
                {Object.entries(PRESETS).map(([key, item]) => {
                  const Icon = item.icon;
                  const disabled = item.faceIdentity && !faceIdentityModel;
                  return (
                    <button
                      key={key}
                      type="button"
                      disabled={disabled}
                      aria-pressed={preset === key}
                      onClick={() => choosePreset(key)}
                      className={`nyx-forgeimg-action ${preset === key ? "is-selected" : ""}`}
                    >
                      <span className="nyx-forgeimg-action-icon">
                        <Icon size={16} />
                      </span>
                      <span>
                        <strong>{item.label}</strong>
                        <small>{item.description}</small>
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>
            {config &&
              (config.faceIdentity ? (
                <section className="nyx-forgeimg-step">
                  <div className="nyx-forgeimg-step-heading">
                    <div>
                      <span className="forge-eyebrow">Step 2 · Face ID</span>
                      <h2>Choose a validated identity</h2>
                    </div>
                    <button
                      type="button"
                      onClick={() => {
                        setPickerPurpose("identity");
                        setPickerOpen(true);
                      }}
                      className="forge-btn-primary"
                    >
                      <UserRoundPlus size={15} />
                      Create Face ID
                    </button>
                  </div>
                  {profileError && (
                    <p className="mt-3 rounded-[9px] bg-[var(--danger-wash)] p-3 text-[10.5px] text-[var(--danger)]">
                      {profileError}
                    </p>
                  )}
                  <div className="mt-4 grid grid-cols-2 gap-3 2xl:grid-cols-3">
                    {profiles.map((item) => (
                      <div
                        key={item.id}
                        className={`group relative overflow-hidden rounded-[10px] bg-[var(--field)] ${profile?.id === item.id ? "ring-2 ring-[var(--accent)]" : "hover:ring-1 hover:ring-[var(--divider)]"}`}
                      >
                        <button
                          type="button"
                          onClick={() => setProfile(item)}
                          className="block w-full text-left"
                        >
                          <span className="block aspect-square overflow-hidden">
                            <img
                              src={item.thumbnail_url}
                              alt=""
                              className="h-full w-full object-cover"
                            />
                          </span>
                          <span className="block p-3">
                            <strong className="block truncate text-[11.5px] text-[var(--text-ui)]">
                              {item.name}
                            </strong>
                            <span className="mt-1 flex items-center gap-1 text-[9.5px] text-[var(--success)]">
                              <ShieldCheck size={11} />
                              {item.quality?.status === "ready"
                                ? "Excellent reference"
                                : "Validated reference"}
                            </span>
                          </span>
                        </button>
                        <button
                          type="button"
                          aria-label={`Delete ${item.name}`}
                          onClick={async () => {
                            if (
                              !window.confirm(`Delete Face ID “${item.name}”?`)
                            )
                              return;
                            await deleteIdentityProfile(item.id);
                            if (profile?.id === item.id) setProfile(null);
                            loadProfiles();
                          }}
                          className="absolute right-2 top-2 rounded-[7px] bg-black/60 p-2 text-white opacity-0 transition group-hover:opacity-100 focus:opacity-100"
                        >
                          <Trash2 size={13} />
                        </button>
                        {profile?.id === item.id && (
                          <span className="absolute left-2 top-2 grid h-6 w-6 place-items-center rounded-full bg-[var(--accent)] text-white">
                            <Check size={14} />
                          </span>
                        )}
                      </div>
                    ))}
                    {!profiles.length && (
                      <div className="col-span-full rounded-[10px] border border-dashed border-[color-mix(in_srgb,var(--divider)_70%,transparent)] bg-[var(--well)] p-7 text-center">
                        <Users
                          size={22}
                          className="mx-auto text-[var(--text-subtle)]"
                        />
                        <p className="mt-3 text-[11px] text-[var(--text-muted)]">
                          No saved Face IDs yet. Create one from a clear
                          photograph.
                        </p>
                      </div>
                    )}
                  </div>
                </section>
              ) : (
                <SourceCard
                  source={source}
                  onOpen={() => {
                    setPickerPurpose("source");
                    setPickerOpen(true);
                  }}
                  mask={mask}
                  editingMask={moreOpen}
                  onMaskChange={setMask}
                  tool={tool}
                  brushSize={brushSize}
                />
              ))}
            {config && (
              <section
                className={`nyx-forgeimg-step transition ${assetReady ? "" : "pointer-events-none opacity-45"}`}
              >
                <span className="forge-eyebrow">
                  Step 3 · Direction and controls
                </span>
                {config.input && (
                  <div className="mt-4">
                    <div className="mb-2 flex items-center justify-between">
                      <label className="text-[11px] font-semibold text-[var(--text-ui)]">
                        {config.input === "background"
                          ? "New background"
                          : config.input === "scene"
                            ? "New scene"
                            : "New outfit"}
                      </label>
                      <span className="text-[9.5px] text-[var(--text-subtle)]">
                        {promptSettings.prompt_engine === "ollama"
                          ? "Vision assisted"
                          : promptSettings.prompt_engine === "cloud"
                            ? "Cloud assisted"
                            : "Local ideas"}
                      </span>
                    </div>
                    <PromptEngineSelector
                      settings={promptSettings}
                      onChange={onPromptSettingsChange}
                      visionModels={visionPromptModels}
                      activeVisionModel={activeVisionModel}
                      cloudProviders={configuredCloudProviders}
                      cloudCredentials={cloudCredentials}
                    />
                    <div className="flex gap-2">
                      <input
                        value={contextInput}
                        onChange={(event) =>
                          setContextInput(event.target.value)
                        }
                        placeholder={
                          config.input === "background"
                            ? "A quiet Tokyo street at night"
                            : config.input === "scene"
                              ? "Walking through a winter market beneath warm lanterns"
                              : "A tailored black evening dress"
                        }
                        className="min-w-0 flex-1 rounded-[9px] bg-[var(--field)] px-3 py-3 text-xs outline-none focus:ring-2 focus:ring-[var(--accent)]"
                      />
                      <button
                        type="button"
                        disabled={suggesting}
                        onClick={generateSuggestion}
                        className="flex shrink-0 items-center gap-1.5 rounded-[9px] bg-[var(--accent-wash)] px-3 text-[11px] font-bold text-[var(--accent-text)] ring-1 ring-[var(--accent)]"
                      >
                        {suggesting ? (
                          <LoaderCircle size={13} className="animate-spin" />
                        ) : (
                          <Sparkles size={13} />
                        )}
                        Generate
                      </button>
                    </div>
                    {suggestionNotice && (
                      <p className="mt-2 text-[9.5px] text-[var(--text-subtle)]">
                        {suggestionNotice}
                      </p>
                    )}
                  </div>
                )}
                <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <label className="text-[11px] font-semibold text-[var(--text-ui)]">
                    Content rating
                    <div className="mt-2">
                      <CustomSelect
                        value={contentRating}
                        options={availableRatings}
                        onChange={setContentRating}
                        ariaLabel="Transform content rating"
                      />
                    </div>
                  </label>
                  <label className="text-[11px] font-semibold text-[var(--text-ui)]">
                    <span className="flex justify-between">
                      <span>CFG scale</span>
                      <span className="text-[var(--accent-text)]">
                        {(
                          cfgScale ??
                          activeProfile?.cfg_scale ??
                          minCfg
                        ).toFixed(1)}
                      </span>
                    </span>
                    <input
                      type="range"
                      min={minCfg}
                      max="8"
                      step="0.5"
                      value={cfgScale ?? activeProfile?.cfg_scale ?? minCfg}
                      onChange={(event) =>
                        setCfgScale(Number(event.target.value))
                      }
                      className="mt-4 w-full accent-[var(--accent)]"
                    />
                  </label>
                </div>
                <button
                  type="button"
                  onClick={() => setMoreOpen((value) => !value)}
                  className="mt-4 flex w-full items-center justify-between rounded-[9px] bg-[var(--field)] px-3 py-2.5 text-[11px] font-bold text-[var(--text-ui)]"
                >
                  <span>
                    {config.faceIdentity
                      ? "More controls"
                      : "More controls · edit mask"}
                  </span>
                  <ChevronDown
                    size={15}
                    className={`transition ${moreOpen ? "rotate-180" : ""}`}
                  />
                </button>
                {moreOpen && (
                  <div className="mt-4 grid grid-cols-1 gap-4 sm:grid-cols-2">
                    <label className="text-xs text-[var(--text-ui)]">
                      Model
                      <div className="mt-2">
                        <CustomSelect
                          value={model}
                          options={models.map((value) => ({
                            value,
                            label: modelProfiles[value]?.display_name ?? value,
                          }))}
                          onChange={chooseModel}
                          ariaLabel="Transform model"
                        />
                      </div>
                    </label>
                    <label className="text-xs text-[var(--text-ui)]">
                      Style
                      <div className="mt-2">
                        <CustomSelect
                          value={style}
                          options={activeProfile?.styles ?? ["Photoreal"]}
                          onChange={setStyle}
                          ariaLabel="Transform style"
                        />
                      </div>
                    </label>
                    <label className="text-xs text-[var(--text-ui)]">
                      <span className="flex justify-between">
                        <span>Denoise</span>
                        <span>{denoise.toFixed(2)}</span>
                      </span>
                      <input
                        type="range"
                        min="0.05"
                        max="0.95"
                        step="0.05"
                        value={denoise}
                        onChange={(event) =>
                          setDenoise(Number(event.target.value))
                        }
                        className="mt-3 w-full accent-[var(--accent)]"
                      />
                    </label>
                    <label className="text-xs text-[var(--text-ui)]">
                      Padding · {inpaintPadding}px
                      <input
                        type="range"
                        min="0"
                        max="256"
                        step="16"
                        value={inpaintPadding}
                        onChange={(event) =>
                          setInpaintPadding(Number(event.target.value))
                        }
                        className="mt-3 w-full accent-[var(--accent)]"
                      />
                    </label>
                    <label className="sm:col-span-2 text-xs text-[var(--text-ui)]">
                      Positive prompt
                      <textarea
                        value={prompt}
                        onChange={(event) => setPrompt(event.target.value)}
                        rows={3}
                        className="mt-2 w-full rounded-[9px] bg-[var(--field)] p-3 text-xs outline-none focus:ring-2 focus:ring-[var(--accent)]"
                      />
                    </label>
                    <label className="sm:col-span-2 text-xs text-[var(--text-ui)]">
                      Negative prompt
                      <textarea
                        value={negativePrompt}
                        onChange={(event) =>
                          setNegativePrompt(event.target.value)
                        }
                        rows={3}
                        className="mt-2 w-full rounded-[9px] bg-[var(--field)] p-3 text-xs outline-none focus:ring-2 focus:ring-[var(--accent)]"
                      />
                    </label>
                    {!config.faceIdentity && (
                      <div className="sm:col-span-2 rounded-[10px] bg-[var(--well)] p-4">
                        <span className="forge-eyebrow">Mask tools</span>
                        <div className="mt-3 grid grid-cols-4 gap-2">
                          <button
                            type="button"
                            onClick={() => setTool("brush")}
                            className="rounded-[8px] bg-[var(--field)] p-2 text-xs"
                          >
                            <Brush size={13} className="mr-1 inline" />
                            Draw
                          </button>
                          <button
                            type="button"
                            onClick={() => setTool("erase")}
                            className="rounded-[8px] bg-[var(--field)] p-2 text-xs"
                          >
                            <Eraser size={13} className="mr-1 inline" />
                            Erase
                          </button>
                          <button
                            type="button"
                            onClick={() => setMask(null)}
                            className="rounded-[8px] bg-[var(--field)] p-2 text-xs"
                          >
                            <X size={13} className="mr-1 inline" />
                            Clear
                          </button>
                          <button
                            type="button"
                            onClick={invertCanvasMask}
                            className="rounded-[8px] bg-[var(--field)] p-2 text-xs"
                          >
                            <RotateCcw size={13} className="mr-1 inline" />
                            Invert
                          </button>
                        </div>
                        <label className="mt-3 block text-xs">
                          Brush radius · {brushSize}px
                          <input
                            type="range"
                            min="6"
                            max="80"
                            value={brushSize}
                            onChange={(event) =>
                              setBrushSize(Number(event.target.value))
                            }
                            className="mt-2 w-full accent-[var(--accent)]"
                          />
                        </label>
                        <label className="mt-3 flex items-center justify-between text-xs">
                          <span>Inpaint full resolution</span>
                          <input
                            type="checkbox"
                            checked={inpaintFullRes}
                            onChange={(event) =>
                              setInpaintFullRes(event.target.checked)
                            }
                          />
                        </label>
                        <label className="mt-3 flex items-center justify-between text-xs">
                          <span>Soft inpainting</span>
                          <input
                            type="checkbox"
                            checked={softInpainting}
                            onChange={(event) =>
                              setSoftInpainting(event.target.checked)
                            }
                          />
                        </label>
                      </div>
                    )}
                  </div>
                )}
                {maskBusy && (
                  <p className="mt-3 flex items-center gap-2 text-[10.5px] text-[var(--text-muted)]">
                    <LoaderCircle size={13} className="animate-spin" />
                    Detecting the editable region…
                  </p>
                )}
                {maskError && (
                  <p className="mt-3 rounded-[9px] bg-[var(--danger-wash)] p-3 text-[10.5px] text-[var(--danger)]">
                    {maskError}
                  </p>
                )}
                {preset === "repair" && proposals.length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-2">
                    {proposals.map((item) => (
                      <button
                        key={item.id}
                        type="button"
                        onClick={() => setMask(item.mask)}
                        className={`rounded-full px-3 py-1.5 text-[10px] font-bold ${mask === item.mask ? "bg-[var(--accent)] text-white" : "bg-[var(--field)]"}`}
                      >
                        {item.label}
                      </button>
                    ))}
                  </div>
                )}
              </section>
            )}
            {config && (
              <section
                className={`nyx-forgeimg-submit ${assetReady ? "" : "opacity-45"}`}
              >
                <button
                  type="button"
                  disabled={!canSubmit}
                  onClick={submit}
                  className="forge-btn-primary w-full"
                >
                  {loading ? (
                    <LoaderCircle size={16} className="animate-spin" />
                  ) : (
                    <ImageUp size={16} />
                  )}
                  {loading ? "Starting transformation…" : "Transform image"}
                </button>
                {!canSubmit && (
                  <p className="mt-2 text-center text-[10px] text-[var(--text-subtle)]">
                    {!assetReady
                      ? config.faceIdentity
                        ? "Choose or create a Face ID to continue."
                        : "Choose a full image to continue."
                      : !connected
                        ? "Forge is offline."
                        : config.input && !contextInput.trim()
                          ? "Describe the result you want."
                          : needsMask && !mask
                            ? "Confirm or draw the editable region."
                            : ""}
                  </p>
                )}
              </section>
            )}
          </div>
        </section>
        <section className="nyx-forgeimg-jobs-panel">
          <ExecutionJobs
            account={account}
            refreshSignal={refreshSignal}
            onNavigate={onNavigate}
            modelProfiles={modelProfiles}
            jobKinds={["img2img", "character"]}
            description="ForgeIMG transforms and identity jobs, newest first."
            variant="forgeai"
          />
        </section>
      </main>
      <SourcePickerDialog
        open={pickerOpen}
        onClose={() => setPickerOpen(false)}
        currentSource={pickerPurpose === "source" ? source : null}
        images={images}
        onUse={selectPickerImage}
        onUpload={handleUpload}
        onUrlUpload={handleUrlUpload}
        uploading={uploading}
        uploadProgress={uploadProgress}
        hasMore={hasMore}
        loadingMore={loadingMore}
        onLoadMore={onLoadMore}
      />
      <FaceProfileDialog
        open={cropDialog}
        source={cropSource}
        initialCrop={crop}
        onClose={() => setCropDialog(false)}
        onSaved={(saved) => {
          setCropDialog(false);
          setProfiles((items) => [saved, ...items]);
          setProfile(saved);
        }}
      />
    </div>
  );
}
