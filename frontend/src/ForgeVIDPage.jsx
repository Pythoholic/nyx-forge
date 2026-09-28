import { useEffect, useRef, useState } from "react";
import {
  Clapperboard,
  ImagePlus,
  LoaderCircle,
  Sparkles,
  Upload,
  X,
} from "lucide-react";
import CustomSelect from "./CustomSelect";
import ErrorBanner from "./ErrorBanner";
import ExecutionJobs from "./ExecutionJobs";
import { generateVideo, generateVideoPrompt, uploadImage } from "./api";

// Must match WanCameraEmbedding.camera_pose in ComfyUI and CAMERA_MOTIONS in
// backend/comfy.py. A value outside this list fails the whole prompt.
const CAMERAS = [
  "Static",
  "Pan Up",
  "Pan Down",
  "Pan Left",
  "Pan Right",
  "Zoom In",
  "Zoom Out",
  "Anti Clockwise (ACW)",
  "ClockWise (CW)",
];
const T2V_SIZES = {
  landscape: [832, 480],
  portrait: [480, 832],
  square: [512, 512],
};
function matchedVideoSize(width, height) {
  const ratio = Math.min(2, Math.max(0.5, (width || 1) / (height || 1)));
  let h = Math.max(16, Math.floor(Math.sqrt(262144 / ratio) / 16) * 16),
    w = Math.max(16, Math.floor((h * ratio) / 16) * 16);
  while (w * h > 262144) {
    if (w >= h) w -= 16;
    else h -= 16;
  }
  return [w, h];
}

function SourcePicker({
  images,
  onChoose,
  onClose,
  hasMore,
  loadingMore,
  onLoadMore,
}) {
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState("");
  const upload = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError("");
    try {
      const result = await uploadImage(file);
      const preview = URL.createObjectURL(file);
      const dimensions = await new Promise((resolve) => {
        const image = new window.Image();
        image.onload = () =>
          resolve({
            source_width: image.naturalWidth,
            source_height: image.naturalHeight,
          });
        image.onerror = () => resolve({});
        image.src = preview;
      });
      onChoose({
        upload_id: result.upload_id,
        preview,
        label: file.name,
        ...dimensions,
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  };
  return (
    <div
      className="fixed inset-0 z-[100] grid place-items-center bg-[var(--scrim)] p-6"
      role="dialog"
      aria-modal="true"
    >
      <div className="w-full max-w-[880px] rounded-[18px] bg-[var(--field)] p-6 shadow-[var(--shadow-modal)]">
        <div className="flex justify-between">
          <div>
            <h2 className="forge-display">Select a source image</h2>
            <p className="mt-1 forge-body">
              Choose recent work or upload a still to animate.
            </p>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-2 text-[var(--text-muted)] hover:bg-[var(--raised)]"
          >
            <X size={18} />
          </button>
        </div>
        <label className="mt-5 flex cursor-pointer items-center justify-center gap-2 rounded-[12px] border border-dashed border-[var(--divider)] bg-[var(--well)] p-4 text-[13px] font-bold text-[var(--accent-hi)]">
          <Upload size={16} />
          {uploading ? "Uploading…" : "Upload image"}
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            className="hidden"
            onChange={upload}
          />
        </label>
        {error && (
          <p className="mt-3 text-[12px] text-[var(--danger)]">{error}</p>
        )}
        <div className="mt-5 max-h-[52vh] overflow-y-auto">
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-5 md:grid-cols-7">
            {images.map((image) => (
              <button
                key={image.id}
                onClick={() =>
                  onChoose({
                    source_generation_id: image.id,
                    preview: image.thumbnail_url || image.full_url,
                    label: `Generation #${image.id}`,
                    source_prompt: image.positive_prompt || "",
                  })
                }
                className="relative aspect-square overflow-hidden rounded-[10px] bg-[var(--well)] ring-1 ring-inset ring-[var(--divider)] hover:ring-[var(--accent)]"
              >
                <img
                  src={image.thumbnail_url || image.full_url}
                  alt=""
                  className="h-full w-full object-cover"
                />
                <span className="absolute inset-x-0 bottom-0 bg-[var(--scrim)] px-2 py-1 text-[10px] font-bold text-white">
                  #{image.id}
                </span>
              </button>
            ))}
          </div>
          {hasMore && (
            <button
              type="button"
              onClick={onLoadMore}
              disabled={loadingMore}
              className="mt-3 flex w-full items-center justify-center gap-2 rounded-[10px] bg-[var(--raised)] py-2.5 text-[12px] font-bold text-[var(--text-ui)] transition-colors hover:bg-[var(--field-hover)] disabled:opacity-50"
            >
              {loadingMore && (
                <LoaderCircle size={14} className="animate-spin" />
              )}
              {loadingMore ? "Loading…" : "Load more"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

function CollapsiblePromptField({
  label,
  value,
  onChange,
  placeholder,
  singleLine = false,
}) {
  const [expanded, setExpanded] = useState(false);
  const textareaRef = useRef(null);

  useEffect(() => {
    if (!expanded) return;
    const frame = window.requestAnimationFrame(() =>
      textareaRef.current?.focus(),
    );
    return () => window.cancelAnimationFrame(frame);
  }, [expanded]);

  return (
    <div className="overflow-hidden rounded-[11px] bg-[var(--well)] ring-1 ring-inset ring-[var(--divider)]">
      <div className="flex items-center justify-between gap-3 px-3.5 py-2.5">
        <span className="flex min-w-0 items-center gap-2 text-[11.5px]">
          <strong className="shrink-0 font-semibold text-[var(--text-muted)]">
            {label}
          </strong>
          {!expanded && (
            <span className="truncate text-[var(--text-subtle)]">
              {value || placeholder}
            </span>
          )}
        </span>
        <button
          type="button"
          aria-expanded={expanded}
          onClick={() => setExpanded((current) => !current)}
          className="shrink-0 rounded-md px-2 py-1 text-[10.5px] font-bold text-[var(--accent-hi)] hover:bg-[var(--raised)]"
        >
          {expanded ? "Collapse" : "Edit"}
        </button>
      </div>
      {expanded ? (
        <div className="px-3.5 pb-3.5">
          <textarea
            ref={textareaRef}
            className={`forge-input resize-y py-2.5 ${singleLine ? "min-h-[64px]" : "min-h-[104px]"}`}
            value={value}
            onChange={(event) => onChange(event.target.value)}
            placeholder={placeholder}
          />
        </div>
      ) : null}
    </div>
  );
}

export default function ForgeVIDPage({
  images,
  account,
  refreshSignal,
  onNavigate,
  connected,
  hasMore = false,
  loadingMore = false,
  onLoadMore,
  promptModels = [],
  cloudCredentials = {},
  promptSettings = {},
}) {
  const [mode, setMode] = useState("t2v"),
    [prompt, setPrompt] = useState(""),
    [negativePrompt, setNegativePrompt] = useState(
      "low quality, blurry, distorted, morphing, flickering, jitter, unstable background, text, watermark",
    );
  const [camera, setCamera] = useState("Zoom In"),
    [seed, setSeed] = useState(""),
    [source, setSource] = useState(null),
    [sourceUploading, setSourceUploading] = useState(false),
    [picker, setPicker] = useState(false),
    [submitting, setSubmitting] = useState(false),
    [error, setError] = useState("");
  const [localRefresh, setLocalRefresh] = useState(0),
    [promptEngine, setPromptEngine] = useState("curated"),
    [promptRunId, setPromptRunId] = useState(null);
  const [interpolation, setInterpolation] = useState(2),
    [outputResolution, setOutputResolution] = useState("native"),
    [steps, setSteps] = useState(20),
    [promptLoading, setPromptLoading] = useState(false),
    [analysisNote, setAnalysisNote] = useState("");
  const [orientation, setOrientation] = useState("landscape"),
    [sourceSize, setSourceSize] = useState(null);
  useEffect(() => {
    if (!source?.preview) {
      setSourceSize(null);
      return;
    }
    const image = new window.Image();
    image.onload = () =>
      setSourceSize(matchedVideoSize(image.naturalWidth, image.naturalHeight));
    image.src = source.preview;
  }, [source]);
  const generationSize =
    mode === "t2v" ? T2V_SIZES[orientation] : sourceSize || [512, 512];
  const ready = connected && prompt.trim() && (mode === "t2v" || source);
  const promptEngineOptions = [
    { value: "curated", label: "Curated" },
    ...(promptModels.length ? [{ value: "ollama", label: "Ollama" }] : []),
    ...(account?.is_admin ? [{ value: "cloud", label: "Cloud" }] : []),
    { value: "reuse", label: "Reuse last" },
  ];
  const makePrompt = async () => {
    if (promptEngine === "reuse" && !prompt.trim() && !source?.source_prompt) {
      setError("Paste or select a prompt to reuse first.");
      return;
    }
    setPromptLoading(true);
    setError("");
    const credential = cloudCredentials[promptSettings.cloud_provider] || {};
    try {
      const result = await generateVideoPrompt({
        engine: promptEngine,
        mode,
        source_prompt: source?.source_prompt || prompt,
        source_generation_id: source?.source_generation_id ?? null,
        upload_id: source?.upload_id ?? null,
        ollama_model:
          promptSettings.ollama_model ||
          promptModels[0]?.value ||
          promptModels[0] ||
          null,
        cloud_provider: promptSettings.cloud_provider || null,
        cloud_model: promptSettings.cloud_model || null,
        cloud_api_key: credential.apiKey || null,
      });
      setPrompt(result.prompt);
      setNegativePrompt(result.negative_prompt);
      setPromptRunId(result.prompt_run_id);
      // The frame analysis picks a camera the geometry supports; keeping a
      // stale choice would put the prompt and the camera embedding back into
      // disagreement, which is what produced incoherent motion before.
      if (result.camera_motion) setCamera(result.camera_motion);
      setAnalysisNote(result.analysis_note || "");
    } catch (e) {
      setError(e.message);
    } finally {
      setPromptLoading(false);
    }
  };
  const acceptDroppedSource = async (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    if (
      !file ||
      !["image/png", "image/jpeg", "image/webp"].includes(file.type)
    ) {
      setError("Drop a PNG, JPG, or WebP image.");
      return;
    }
    setSourceUploading(true);
    setError("");
    try {
      const result = await uploadImage(file);
      setSource({
        upload_id: result.upload_id,
        preview: URL.createObjectURL(file),
        label: file.name,
      });
      setPrompt("");
      setPromptRunId(null);
    } catch (e) {
      setError(e.message);
    } finally {
      setSourceUploading(false);
    }
  };
  const submit = async () => {
    if (!ready || submitting) return;
    setSubmitting(true);
    setError("");
    try {
      await generateVideo(mode, {
        prompt: prompt.trim(),
        negative_prompt: negativePrompt.trim(),
        seed: seed === "" ? -1 : Number(seed),
        camera_motion: camera,
        source_generation_id: source?.source_generation_id ?? null,
        upload_id: source?.upload_id ?? null,
        interpolation,
        output_resolution: outputResolution,
        steps,
        prompt_engine: promptEngine,
        prompt_run_id: promptRunId,
        orientation,
      });
      setLocalRefresh((v) => v + 1);
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  };
  const orientationControl =
    mode === "t2v" ? (
      <div>
        <span className="forge-label">Orientation</span>
        <div className="forge-seg mt-2 grid grid-cols-3 rounded-[9px] bg-[var(--well)] p-1">
          {[
            ["landscape", "Landscape"],
            ["portrait", "Portrait"],
            ["square", "Square"],
          ].map(([value, label]) => (
            <button
              key={value}
              aria-selected={orientation === value}
              onClick={() => setOrientation(value)}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
    ) : null;
  const sizeNotice = (
    <div className="flex min-h-[40px] items-center rounded-[9px] bg-[var(--well)] px-3 py-2 text-[10.5px] text-[var(--text-muted)]">
      <strong className="text-[var(--text)]">
        {generationSize[0]}×{generationSize[1]}
      </strong>{" "}
      ·{" "}
      {mode === "i2v" && sourceSize
        ? `matched to your ${generationSize[0] > generationSize[1] ? "landscape" : generationSize[0] < generationSize[1] ? "portrait" : "square"} source`
        : mode === "i2v"
          ? "default until a source is selected"
          : `${orientation} generation`}
    </div>
  );
  const operations = (
    <section className="nyx-forgevid-operations rounded-[16px] bg-[var(--panel)] p-5 shadow-[var(--shadow-card)] ring-1 ring-inset ring-[var(--divider)]">
      <div className="flex items-center justify-between gap-4">
        <div>
          <h2 className="forge-display">Operations</h2>
          <p className="mt-1 forge-body">Configure the next clip.</p>
        </div>
        <span className="flex items-center gap-2 rounded-full bg-[var(--well)] px-3 py-2 text-[10.5px] font-bold text-[var(--text-muted)] ring-1 ring-inset ring-[var(--divider)]">
          <span className="h-1.5 w-1.5 rounded-full bg-[var(--accent)] shadow-[0_0_0_3px_var(--accent-wash)]" />
          {mode === "t2v" ? "Wan 2.1 T2V 1.3B" : "Wan 2.1 Fun Camera 1.3B"}
        </span>
      </div>
      <div className="forge-seg mt-4 inline-grid grid-cols-2 rounded-[11px] bg-[var(--well)] p-1 ring-1 ring-inset ring-[var(--divider)]">
        {[
          ["i2v", "Animate image"],
          ["t2v", "Text to video"],
        ].map(([v, l]) => (
          <button
            className="px-4"
            key={v}
            aria-selected={mode === v}
            onClick={() => {
              setMode(v);
              if (v === "t2v") setSource(null);
            }}
          >
            {l}
          </button>
        ))}
      </div>
      <div className="mt-4 grid gap-[18px] lg:grid-cols-[minmax(260px,330px)_minmax(0,1fr)]">
        <div className="space-y-2.5">
          <span className="forge-label">
            {mode === "i2v" ? "Source image" : "Orientation"}
          </span>
          {mode === "i2v" ? (
            <button
              onClick={() => setPicker(true)}
              onDragOver={(event) => event.preventDefault()}
              onDrop={acceptDroppedSource}
              disabled={sourceUploading}
              className="flex h-[83px] w-full items-center gap-3 rounded-[11px] border border-dashed border-[var(--divider)] bg-[var(--well)] px-4 text-left transition hover:border-[var(--accent)]"
            >
              {source ? (
                <img
                  src={source.preview}
                  alt=""
                  className="h-10 w-10 rounded-[9px] object-cover"
                />
              ) : (
                <span className="grid h-10 w-10 place-items-center rounded-[9px] bg-[var(--accent-wash)] text-[var(--accent-hi)]">
                  <ImagePlus size={19} />
                </span>
              )}
              <span className="min-w-0">
                <strong className="block truncate text-[12.5px] text-[var(--text)]">
                  {sourceUploading
                    ? "Uploading image…"
                    : source?.label || "Drop an image or browse"}
                </strong>
                <span className="mt-1 block text-[10.5px] text-[var(--text-subtle)]">
                  PNG, JPG or gallery image
                </span>
              </span>
            </button>
          ) : (
            <div className="grid grid-cols-3 gap-2">
              {[
                ["landscape", "Landscape", "h-4 w-7"],
                ["portrait", "Portrait", "h-7 w-4"],
                ["square", "Square", "h-6 w-6"],
              ].map(([value, label, shape]) => (
                <button
                  key={value}
                  aria-pressed={orientation === value}
                  onClick={() => setOrientation(value)}
                  className="flex min-h-[83px] flex-col items-center justify-center gap-2 rounded-[11px] bg-[var(--well)] text-[10.5px] font-semibold text-[var(--text-muted)] ring-1 ring-inset ring-[var(--divider)] transition aria-pressed:bg-[var(--accent-wash)] aria-pressed:text-[var(--text)] aria-pressed:ring-[var(--accent)]"
                >
                  <span
                    className={`${shape} rounded-[3px] bg-[var(--text-subtle)]`}
                  />
                  {label}
                </button>
              ))}
            </div>
          )}
          <div className="flex min-h-[42px] items-center justify-between gap-3 rounded-[10px] bg-[var(--well)] px-3 ring-1 ring-inset ring-[var(--divider)]">
            <strong className="text-[14px] tabular-nums text-[var(--text)]">
              {generationSize[0]} × {generationSize[1]}
            </strong>
            <span className="text-right text-[10.5px] text-[var(--text-subtle)]">
              {mode === "i2v"
                ? sourceSize
                  ? "matched to source"
                  : "default until a source is selected"
                : `${orientation} generation`}
            </span>
          </div>
        </div>

        <div className="space-y-2.5">
          <div className="flex items-end justify-between gap-3">
            <span className="forge-label pb-2">Motion prompt</span>
            <div className="flex items-center gap-2">
              <div className="w-[116px]">
                <CustomSelect
                  value={promptEngine}
                  onChange={setPromptEngine}
                  options={promptEngineOptions}
                  ariaLabel="Prompt engine"
                />
              </div>
              <button
                onClick={makePrompt}
                disabled={promptLoading || (mode === "i2v" && !source)}
                className="flex min-h-[38px] items-center gap-2 rounded-[8px] bg-[var(--accent)] px-3 text-[11.5px] font-bold text-white transition hover:bg-[var(--accent-hi)] disabled:opacity-50"
              >
                {promptLoading ? (
                  <LoaderCircle size={14} className="animate-spin" />
                ) : (
                  <Sparkles size={14} />
                )}
                {mode === "i2v" && !source ? "Choose image" : "Generate"}
              </button>
            </div>
          </div>
          <textarea
            className="forge-input min-h-[104px] resize-y rounded-[11px] px-3.5 py-3 text-[13px] leading-5"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="A slow dolly through fog as the light shifts…"
          />
          <CollapsiblePromptField
            label="Avoid"
            value={negativePrompt}
            onChange={setNegativePrompt}
            placeholder="Add motion or quality exclusions…"
            singleLine
          />
        </div>
      </div>

      <div
        className={`mt-4 grid gap-4 border-t border-[var(--divider)] pt-4 sm:grid-cols-2 ${mode === "i2v" ? "xl:grid-cols-5" : "xl:grid-cols-4"}`}
      >
        {mode === "i2v" && (
          <div>
            <div className="flex items-baseline gap-2">
              <span className="forge-label">Camera</span>
              {analysisNote && (
                <span className="text-[10px] text-[var(--accent-hi)]" title="Chosen from the source frame">
                  auto
                </span>
              )}
              <span className="text-[10px] text-[var(--text-subtle)]">
                move
              </span>
            </div>
            <div className="mt-2">
              <CustomSelect
                value={camera}
                onChange={setCamera}
                options={CAMERAS.map((value) => ({ value, label: value }))}
                ariaLabel="Camera motion"
              />
            </div>
          </div>
        )}
        <div>
          <div className="flex items-baseline gap-2">
            <span className="forge-label">Motion</span>
            <span className="text-[10px] text-[var(--text-subtle)]">
              fps · smoothness only
            </span>
          </div>
          <div className="forge-seg mt-2 grid grid-cols-3 rounded-[9px] bg-[var(--well)] p-1">
            {[
              [1, "16"],
              [2, "32"],
              [4, "63"],
            ].map(([v, l]) => (
              <button
                key={v}
                aria-selected={interpolation === v}
                onClick={() => setInterpolation(v)}
              >
                {l}
              </button>
            ))}
          </div>
        </div>
        <div>
          <span className="forge-label">Resolution</span>
          <div className="forge-seg mt-2 grid grid-cols-3 rounded-[9px] bg-[var(--well)] p-1">
            {[
              ["native", "Native"],
              ["720p", "720p"],
              ["1080p", "1080p"],
            ].map(([v, l]) => (
              <button
                key={v}
                aria-selected={outputResolution === v}
                onClick={() => setOutputResolution(v)}
              >
                {l}
              </button>
            ))}
          </div>
        </div>
        <div>
          <div className="flex items-baseline gap-2">
            <span className="forge-label">Steps</span>
            <span className="text-[10px] text-[var(--text-subtle)]">
              quality
            </span>
          </div>
          <div className="forge-seg mt-2 grid grid-cols-3 rounded-[9px] bg-[var(--well)] p-1">
            {[20, 25, 30].map((v) => (
              <button
                key={v}
                aria-selected={steps === v}
                onClick={() => setSteps(v)}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
        <label>
          <span className="forge-label">Seed</span>
          <span className="ml-2 text-[10px] text-[var(--text-subtle)]">
            optional
          </span>
          <input
            className="forge-input mt-2"
            type="number"
            min="0"
            value={seed}
            onChange={(e) => setSeed(e.target.value)}
            placeholder="Random"
          />
        </label>
      </div>
      <div className="mt-4 flex min-h-[42px] items-center gap-3 rounded-[11px] bg-[var(--well)] px-3.5 text-[10.5px] text-[var(--text-muted)] ring-1 ring-inset ring-[var(--divider)]">
        <Clapperboard size={14} className="shrink-0 text-[var(--accent-hi)]" />
        <span>
          <strong className="text-[var(--text)]">
            {48 * interpolation + 1} frames
          </strong>{" "}
          ·{" "}
          {interpolation === 1
            ? "16 FPS"
            : interpolation === 2
              ? "~32 FPS"
              : "~63 FPS"}
        </span>
        <span className="border-l border-[var(--divider)] pl-3">
          {outputResolution === "native"
            ? "60–150 seconds"
            : interpolation === 4
              ? "up to 5.8 minutes"
              : "upscale adds ~53 seconds"}
        </span>
        <span className="border-l border-[var(--divider)] pl-3">
          ~3 second clip · fixed by the 49-frame RTX 3080 VRAM limit
        </span>
        <span className="ml-auto text-[var(--text-subtle)]">
          {!connected
            ? "ComfyUI unavailable"
            : mode === "i2v" && !source
              ? "Waiting for a source image"
              : !prompt.trim()
                ? "Generate or enter a prompt"
                : "Ready"}
        </span>
      </div>
      <ErrorBanner error={error} onDismiss={() => setError("")} />
      {!connected && (
        <p className="mt-3 rounded-[9px] bg-[var(--danger-wash)] p-3 text-[11.5px] font-semibold text-[var(--danger)]">
          ComfyUI is unavailable. Start it in Stability Matrix.
        </p>
      )}
      <button
        onClick={submit}
        disabled={!ready || submitting}
        className={`forge-btn-primary mt-2.5 w-full ${ready && !submitting ? "shadow-[0_10px_30px_-12px_var(--accent)]" : ""}`}
      >
        {submitting ? (
          <LoaderCircle size={16} className="animate-spin" />
        ) : (
          <Sparkles size={16} />
        )}{" "}
        {submitting
          ? "Adding to queue…"
          : mode === "i2v" && !source
            ? "Choose a source to continue"
            : mode === "t2v"
              ? "Generate video"
              : "Animate image"}
      </button>
    </section>
  );
  return (
    <main className="nyx-forgevid-workspace">
      <section className="nyx-forgevid-layout">
        <div className="nyx-forgevid-jobs">
          <ExecutionJobs
            compact
            account={account}
            refreshSignal={`${refreshSignal}-${localRefresh}`}
            onNavigate={onNavigate}
            jobKinds={["video_t2v", "video_i2v"]}
            description="Current and recent ForgeVID work."
          />
        </div>
        {operations}
      </section>
      {picker && (
        <SourcePicker
          images={images}
          hasMore={hasMore}
          loadingMore={loadingMore}
          onLoadMore={onLoadMore}
          onClose={() => setPicker(false)}
          onChoose={(value) => {
            setSource(value);
            setPrompt("");
            setPromptRunId(null);
            setPicker(false);
          }}
        />
      )}
    </main>
  );
}
