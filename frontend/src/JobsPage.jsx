import { useCallback, useEffect, useMemo, useState } from "react";
import { displayJobId, displayQuality, displayStyle } from "./jobLabels";
import {
  Check,
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Image as ImageIcon,
  LoaderCircle,
  RefreshCw,
  RotateCcw,
  Square,
  X,
} from "lucide-react";

import {
  cancelJob,
  getGenerationLineage,
  getJob,
  getJobs,
  rerunJob,
} from "./api";

const COLORS = {
  background: "var(--nyx-void)",
  header: "var(--nyx-panel)",
  card: "var(--nyx-panel-2)",
  cardHover: "var(--nyx-chip)",
  sidebar: "var(--nyx-panel)",
  tile: "var(--nyx-input)",
  divider: "var(--nyx-line)",
  purple: "var(--nyx-accent)",
  purpleLight: "var(--nyx-accent-hi)",
  green: "var(--nyx-signal)",
  red: "var(--nyx-fail)",
  amber: "var(--nyx-warn)",
};

const JOB_PAGE_SIZE = 40;

const PIPELINE_ORDER = ["generate", "surprise_generate", "batch_generate", "img2img", "video_i2v", "character", "upscale", "pixel_upscale"];
const PIPELINE_TITLES = {
  generate: "Generate",
  surprise_generate: "Surprise generate",
  batch_generate: "ForgeBAT batch generation",
  img2img: "Image transform",
  video_i2v: "Image to video",
  character: "Same character",
  upscale: "Detail upscale",
  pixel_upscale: "Pixel upscale",
};

function routeState(path) {
  const parts = String(path ?? "/forge-deploy").split("/").filter(Boolean);
  if (parts[1] === "pipelines" && PIPELINE_ORDER.includes(parts[2])) {
    return { view: "list", activeKind: parts[2], selectedId: null };
  }
  if (parts[1] === "jobs" && /^\d+$/.test(parts[2] ?? "")) {
    return { view: "detail", activeKind: null, selectedId: Number(parts[2]) };
  }
  return { view: "pipelines", activeKind: null, selectedId: null };
}
const PIPELINE_SUBTITLES = {
  generate: "Prompted renders",
  surprise_generate: "Auto-prompted renders",
  batch_generate: "Generates a configured batch of Forge images.",
  img2img: "Gallery-sourced transforms",
  video_i2v: "Animates a source image into a video.",
  character: "Identity-preserving scene generation",
  upscale: "Re-renders an existing image at the same seed with a sharper detail pass. Runs on your local GPU.",
  pixel_upscale: "Resizes an existing image without another diffusion pass.",
};
const PIPELINE_ROW_SUBTITLES = {
  generate: "Prompted renders",
  surprise_generate: "Auto-prompted renders",
  batch_generate: "ForgeBAT batch renders",
  img2img: "Gallery-sourced transforms",
  video_i2v: "Source-image video animation",
  character: "InstantID scene + identity lock",
  upscale: "Same-seed detail re-render",
  pixel_upscale: "Resize without re-rendering",
};

function pipelineLabel(labels, kind) {
  return labels[kind] ?? String(kind ?? "Unknown pipeline");
}

const FILTERS = [
  ["all", "All"],
  ["active", "Active"],
  ["succeeded", "Succeeded"],
  ["failed", "Failed"],
];

const STAGE_DEFS = {
  prompt_generation: { label: "Prompt generation", detail: "Building the visual brief from the selected prompt engine." },
  prompt_ready: { label: "Prompt ready", detail: "The visual brief passed validation and is ready for Forge." },
  identity_analysis: { label: "Reference analysis", detail: "Validating the reference face and identity profile." },
  identity_generation: { label: "Scene generation", detail: "Generating a new SDXL scene with InstantID conditioning." },
  identity_lock: { label: "Identity lock", detail: "Transferring the reference identity into the generated face." },
  identity_refinement: { label: "Face refinement", detail: "Restoring swapped face texture before a conservative upscale." },
  identity_verification: { label: "Identity verification", detail: "Comparing the locked face with the reference before saving." },
  model_loading: { label: "Checkpoint", detail: "Loading the selected checkpoint and preparing inference." },
  sampling: { label: "Diffusion", detail: "Resolving the image through the model's sampling steps." },
  refining: { label: "Detail pass", detail: "Refining fine structure at the selected output quality." },
  finalizing: { label: "Pixel decode", detail: "Decoding the latent result into image pixels." },
  upscaling: { label: "Upscale", detail: "Resizing the decoded image to its final dimensions." },
  saving: { label: "Store result", detail: "Writing the final image and metadata to local storage." },
};

const PIPELINES = {
  generate: ["model_loading", "sampling", "refining", "finalizing", "upscaling", "saving"],
  surprise_generate: ["prompt_generation", "prompt_ready", "model_loading", "sampling", "refining", "finalizing", "upscaling", "saving"],
  img2img: ["model_loading", "sampling", "refining", "finalizing", "saving"],
  character: ["identity_analysis", "identity_generation", "identity_lock", "identity_refinement", "identity_verification", "saving"],
  upscale: ["model_loading", "sampling", "refining", "finalizing", "upscaling", "saving"],
  pixel_upscale: ["upscaling", "saving"],
};

const STATUS_META = {
  queued: { label: "Queued", color: COLORS.purpleLight, bg: "var(--nyx-accent-soft)" },
  running: { label: "Running", color: COLORS.amber, bg: "var(--warn-wash)" },
  succeeded: { label: "Succeeded", color: COLORS.green, bg: "var(--nyx-signal-soft)" },
  failed: { label: "Failed", color: COLORS.red, bg: "var(--danger-wash)" },
  cancelled: { label: "Cancelled", color: "var(--nyx-label)", bg: "var(--nyx-panel-2)" },
};

function parseTimestamp(value) {
  if (!value) return null;
  const parsed = new Date(`${value.replace(" ", "T")}`);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
}

function relativeTime(value) {
  const parsed = parseTimestamp(value);
  if (!parsed) return value || "—";
  const seconds = Math.max(0, Math.round((Date.now() - parsed.getTime()) / 1000));
  if (seconds < 5) return "just now";
  if (seconds < 60) return `${seconds}s ago`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.round(seconds / 3600)}h ago`;
  return `${Math.round(seconds / 86400)}d ago`;
}

function secondsBetween(startValue, endValue) {
  const start = parseTimestamp(startValue);
  const end = parseTimestamp(endValue);
  if (!start || !end) return null;
  return Math.max(0, (end.getTime() - start.getTime()) / 1000);
}

function jobDurationSeconds(job) {
  const start = job.started_at ?? job.created_at;
  const end = job.finished_at ?? new Date().toISOString().replace("T", " ").replace("Z", "");
  return secondsBetween(start, end);
}

function formatDuration(seconds, fallback = "—") {
  if (seconds == null || !Number.isFinite(seconds)) return fallback;
  const rounded = Math.max(0, Math.round(seconds));
  if (rounded < 60) return `${rounded}s`;
  return `${Math.floor(rounded / 60)}m ${rounded % 60}s`;
}

function medianDuration(jobs) {
  const values = jobs
    .filter((job) => job.status === "succeeded" || job.status === "failed")
    .map(jobDurationSeconds)
    .filter((value) => value != null)
    .sort((left, right) => left - right);
  if (!values.length) return null;
  const middle = Math.floor(values.length / 2);
  return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

function isToday(value) {
  const parsed = parseTimestamp(value);
  if (!parsed) return false;
  const now = new Date();
  return parsed.getFullYear() === now.getFullYear()
    && parsed.getMonth() === now.getMonth()
    && parsed.getDate() === now.getDate();
}

function stagesForJob(job) {
  const all = PIPELINES[job.kind] ?? ["model_loading", "sampling", "finalizing", "saving"];
  const usesDetailPass = job.kind === "upscale"
    || Boolean(job.request_payload?.high_res)
    || Boolean(job.request_payload?.super_res);
  return all.filter((stage) => {
    if (stage === "refining") return usesDetailPass;
    if (stage === "upscaling") return job.kind === "pixel_upscale" || usesDetailPass;
    return true;
  });
}

function stageStatus(job, stage, index, orderedStages) {
  if (job.status === "succeeded") return "done";
  if (job.status === "queued") return "pending";
  const observedAt = orderedStages.indexOf(job.progress_stage);
  if (observedAt === -1) return "pending";
  if (index < observedAt) return "done";
  if (index > observedAt) return job.status === "running" ? "pending" : "skipped";
  if (job.status === "running") return "active";
  if (job.status === "failed") return "failed";
  if (job.status === "cancelled") return "cancelled";
  return "skipped";
}

function statusMeta(status) {
  return STATUS_META[status] ?? STATUS_META.queued;
}

function StatusChip({ status }) {
  const meta = statusMeta(status);
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[11px] font-bold" style={{ color: meta.color, background: meta.bg }}>
      <span className={`h-1.5 w-1.5 rounded-full ${status === "running" || status === "queued" ? "animate-pulse" : ""}`} style={{ background: meta.color }} />
      {meta.label}
    </span>
  );
}

function jobConfiguration(job) {
  const payload = job.request_payload ?? {};
  if (job.kind === "character") {
    const generator = payload.character_selected_generator
      ?? payload.character_generator
      ?? "InstantID SDXL";
    return `Same character · ${generator}`;
  }
  if (job.kind === "img2img") return `${payload.preset ?? "Advanced"} · denoise ${Number(payload.denoising_strength ?? 0.4).toFixed(2)}`;
  if (job.kind === "upscale") return payload.super_res ? "Super detail pass" : "High detail pass";
  if (job.kind === "pixel_upscale") return `${payload.multiplier ?? "?"}× pixel resize`;
  const tier = displayQuality(payload);
  const resolvedStyle = displayStyle({
    style: payload.style,
    style_variant: payload.style_variant ?? job.generation?.style_variant,
  });
  const style = resolvedStyle ? `${resolvedStyle} · ` : "";
  return `${style}${tier}`;
}

function jobSource(job) {
  const sourceId = job.request_payload?.generation_id ?? job.request_payload?.source_generation_id ?? job.generation?.source_generation_id;
  const seed = job.generation?.seed;
  if (sourceId != null) return `from generation #${sourceId}${seed != null ? ` · seed ${seed}` : ""}`;
  if (seed != null) return `seed ${seed}`;
  return job.request_payload?.model ? String(job.request_payload.model) : "Forge request";
}

function progressFill(job) {
  if (job.status === "succeeded") return COLORS.green;
  if (job.status === "failed") return COLORS.red;
  if (job.status === "cancelled") return "#4d4962";
  return COLORS.amber;
}

function SummaryStat({ value, label, color = "#f2f2f5" }) {
  return (
    <div className="flex flex-col items-end gap-0.5">
      <span className="text-[19px] font-extrabold tabular-nums" style={{ color }}>{value}</span>
      <span className="whitespace-nowrap text-[11px] font-bold uppercase tracking-[0.05em] text-[#75718a]">{label}</span>
    </div>
  );
}

function PipelineBars({ jobs }) {
  const bars = [...jobs].slice(0, 12).reverse();
  if (!bars.length) return <span className="text-[11px] text-[#4d4962]">—</span>;
  return (
    <div className="flex h-8 items-end gap-1" aria-label={`Last ${bars.length} runs`}>
      {bars.map((job) => {
        const color = job.status === "failed" ? COLORS.red : job.status === "succeeded" ? COLORS.green : COLORS.amber;
        const height = job.status === "failed" ? 14 : job.status === "succeeded" ? 22 : 18;
        return <span key={job.id} className="w-1.5 rounded-[3px]" style={{ height, background: color, opacity: job.status === "succeeded" ? 0.85 : 1 }} title={`${displayJobId(job)} · ${statusMeta(job.status).label}`} />;
      })}
    </div>
  );
}

function PipelineRow({ kind, jobs, onOpen }) {
  const latest = jobs[0] ?? null;
  const latestMeta = latest ? statusMeta(latest.status) : null;
  const median = medianDuration(jobs);
  const enabled = jobs.length > 0;
  const iconColor = latestMeta?.color ?? COLORS.green;
  const iconBackground = latestMeta?.bg ?? "rgba(62,207,126,0.13)";
  return (
    <button
      type="button"
      disabled={!enabled}
      onClick={onOpen}
      className="group grid w-full min-w-0 grid-cols-[minmax(0,1.7fr)_minmax(56px,.7fr)_minmax(92px,1fr)_minmax(92px,1.1fr)_minmax(64px,.8fr)_20px] items-center gap-3 rounded-[14px] bg-[#12101a] px-4 py-5 text-left transition duration-150 enabled:hover:bg-[#181423] disabled:cursor-default disabled:opacity-60"
    >
      <span className="flex min-w-0 items-center gap-3.5">
        <span className="grid h-[30px] w-[30px] shrink-0 place-items-center rounded-[9px]" style={{ color: iconColor, background: iconBackground }}>
          {latest?.status === "running" || latest?.status === "queued"
            ? <LoaderCircle size={15} className="animate-spin" />
            : latest?.status === "failed" || latest?.status === "cancelled"
              ? <X size={15} strokeWidth={2.4} />
              : <Check size={15} strokeWidth={2.4} />}
        </span>
        <span className="min-w-0">
          <span className="block truncate text-[14.5px] font-bold text-[#f2f2f5]">{pipelineLabel(PIPELINE_TITLES, kind)}</span>
          <span className="mt-0.5 block truncate text-xs text-[#75718a]">{pipelineLabel(PIPELINE_ROW_SUBTITLES, kind)}</span>
        </span>
      </span>
      <span className="flex min-w-0 flex-col gap-0.5">
        <span className="text-sm font-bold tabular-nums text-[#e7e5ee]">{jobs.length}</span>
        <span className="text-[11px] text-[#5f5b74]">{jobs.length === 1 ? "execution" : "executions"}</span>
      </span>
      <span className="flex min-w-0 flex-col gap-1">
        <span className="text-[12.5px] font-bold" style={{ color: latestMeta?.color ?? "#4d4962" }}>{latestMeta?.label ?? "No runs yet"}</span>
        <span className="text-[11.5px] text-[#5f5b74]">{latest ? relativeTime(latest.created_at) : "—"}</span>
      </span>
      <PipelineBars jobs={jobs} />
      <span className="min-w-0 truncate text-[13px] tabular-nums text-[#c7c3d4]">{formatDuration(median)}</span>
      <span className="flex justify-end text-[#4d4962] transition group-hover:text-[#9d8cff]">{enabled && <ChevronRight size={16} />}</span>
    </button>
  );
}

function PipelineIndex({ jobsByKind, counts, onOpen }) {
  return (
    <div className="flex min-h-full min-w-0 flex-col gap-[26px] px-10 py-[34px] pb-[50px]">
      <div className="flex items-end justify-between gap-6">
        <div>
          <h1 className="text-[23px] font-extrabold tracking-[-0.015em] text-[#f2f2f5]">Pipelines</h1>
          <p className="mt-1.5 text-[13.5px] text-[#8b879a]">Each pipeline is a job type. Open one to see its executions.</p>
        </div>
        <div className="flex gap-5">
          <SummaryStat value={counts.running} label="Running" />
          <SummaryStat value={counts.queued} label="Queued" />
          <SummaryStat value={counts.failedToday} label="Failed today" />
        </div>
      </div>
      <div className="flex min-w-0 flex-col gap-2.5">
        <div className="grid min-w-0 grid-cols-[minmax(0,1.7fr)_minmax(56px,.7fr)_minmax(92px,1fr)_minmax(92px,1.1fr)_minmax(64px,.8fr)_20px] gap-3 px-4 pb-0.5 text-[10.5px] font-bold uppercase tracking-[0.06em] text-[#5f5b74]">
          <span>Pipeline</span><span>Jobs</span><span>Latest</span><span>Last 12 runs</span><span>Median</span><span />
        </div>
        {PIPELINE_ORDER.map((kind) => <PipelineRow key={kind} kind={kind} jobs={jobsByKind[kind] ?? []} onOpen={() => onOpen(kind)} />)}
      </div>
    </div>
  );
}

function filterJob(job, filter) {
  if (filter === "all") return true;
  if (filter === "active") return job.status === "running" || job.status === "queued";
  if (filter === "failed") return job.status === "failed" || job.status === "cancelled";
  return job.status === filter;
}

function PipelineJobRow({ job, onOpen }) {
  const meta = statusMeta(job.status);
  const active = job.status === "running" || job.status === "queued";
  return (
    <button type="button" onClick={onOpen} className="group grid w-full min-w-0 grid-cols-[10px_minmax(86px,.9fr)_minmax(130px,1.4fr)_minmax(84px,.75fr)_minmax(64px,.65fr)_minmax(70px,.7fr)_14px] items-center gap-3 rounded-xl bg-[#12101a] px-4 py-4 text-left transition hover:bg-[#181423]">
      <span className={`h-2 w-2 rounded-full ${active ? "animate-pulse" : ""}`} style={{ background: meta.color }} />
      <span className="min-w-0">
        <span className="block text-[13px] font-bold tabular-nums text-[#e7e5ee]">{displayJobId(job)}</span>
        <span className="mt-0.5 block text-[11.5px] font-semibold" style={{ color: meta.color }}>{meta.label}</span>
      </span>
      <span className="min-w-0">
        <span className="block truncate text-[12.5px] font-medium text-[#c7c3d4]">{jobConfiguration(job)}</span>
        <span className="mt-0.5 block truncate text-[11px] text-[#5f5b74]">{jobSource(job)}</span>
      </span>
      <span>
        <span className="block h-1 overflow-hidden rounded-[3px] bg-[#211d2c]"><span className="block h-full rounded-[3px] transition-[width] duration-500" style={{ width: `${Math.max(active ? 2 : 0, job.progress_percent)}%`, background: progressFill(job) }} /></span>
        <span className="mt-1 block text-[10.5px] text-[#5f5b74]">{job.status === "succeeded" ? "Complete" : job.status === "queued" ? "Waiting" : `${Math.round(job.progress_percent)}%`}</span>
      </span>
      <span className="text-[12.5px] tabular-nums text-[#c7c3d4]">{job.status === "queued" ? "—" : formatDuration(jobDurationSeconds(job))}</span>
      <span className="min-w-0 truncate text-[11.5px] text-[#5f5b74]">{relativeTime(job.created_at)}</span>
      <ChevronRight size={15} className="text-[#4d4962] transition group-hover:text-[#9d8cff]" />
    </button>
  );
}

function PipelineList({ kind, jobs, filter, setFilter, hasMore, loadingMore, onLoadMore, onBack, onOpen }) {
  const visible = jobs.filter((job) => filterJob(job, filter));
  const activeCount = jobs.filter((job) => job.status === "running" || job.status === "queued").length;
  const failedCount = jobs.filter((job) => job.status === "failed" || job.status === "cancelled").length;
  return (
    <div className="flex min-h-full min-w-0 flex-col gap-6 px-10 py-[30px] pb-[50px]">
      <div className="flex flex-col gap-2">
        <button type="button" onClick={onBack} className="flex w-fit items-center gap-1.5 text-[12.5px] font-semibold text-[#8b879a] transition hover:text-[#e7e5ee]"><ChevronLeft size={14} /> All pipelines</button>
        <div className="flex items-end justify-between gap-6">
          <div>
            <h1 className="text-[23px] font-extrabold tracking-[-0.015em] text-[#f2f2f5]">{pipelineLabel(PIPELINE_TITLES, kind)}</h1>
            <p className="mt-1.5 text-[13px] text-[#8b879a]">{pipelineLabel(PIPELINE_SUBTITLES, kind)}</p>
          </div>
          <div className="flex gap-5">
            <SummaryStat value={jobs.length} label="Executions" />
            <SummaryStat value={activeCount} label="Active" color={COLORS.amber} />
            <SummaryStat value={failedCount} label="Failed" color={COLORS.red} />
            <SummaryStat value={formatDuration(medianDuration(jobs))} label="Median" />
          </div>
        </div>
      </div>
      <div className="flex items-center justify-between">
        <div className="flex rounded-[6px] border-2 border-[var(--nyx-line)] bg-[var(--nyx-input)] p-1">
          {FILTERS.map(([value, label]) => <button key={value} type="button" onClick={() => setFilter(value)} aria-selected={filter === value} className={`rounded-[4px] border-2 px-3.5 py-[7px] text-[11px] font-semibold uppercase tracking-[.08em] transition ${filter === value ? "border-[var(--nyx-accent-line)] bg-[var(--nyx-accent-soft)] text-[var(--nyx-accent)]" : "border-transparent text-[var(--nyx-label)] hover:text-[var(--nyx-ink)]"}`}>{label}</button>)}
        </div>
        <span className="text-[11.5px] text-[#5f5b74]">{visible.length} of {jobs.length} executions</span>
      </div>
      <div className="flex min-w-0 flex-col gap-2">
        <div className="grid min-w-0 grid-cols-[10px_minmax(86px,.9fr)_minmax(130px,1.4fr)_minmax(84px,.75fr)_minmax(64px,.65fr)_minmax(70px,.7fr)_14px] gap-3 px-4 pb-0.5 text-[10.5px] font-bold uppercase tracking-[0.06em] text-[#5f5b74]">
          <span /><span>Job</span><span>Configuration</span><span>Progress</span><span>Duration</span><span>Started</span><span />
        </div>
        {visible.length ? visible.map((job) => <PipelineJobRow key={job.id} job={job} onOpen={() => onOpen(job.id)} />) : <div className="rounded-xl bg-[#12101a] px-5 py-10 text-center text-[13px] text-[#75718a]">No executions match this filter.</div>}
        {hasMore && <button type="button" onClick={onLoadMore} disabled={loadingMore} className="mt-2 rounded-xl bg-[#12101a] px-4 py-3 text-[12px] font-semibold text-[#8b879a] transition hover:bg-[#181423] hover:text-white disabled:cursor-wait disabled:opacity-60">{loadingMore ? "Loading older executions…" : "Load older executions"}</button>}
      </div>
    </div>
  );
}

function stageDetail(job) {
  if (job.status === "queued") return "Accepted and waiting for the Forge worker.";
  if (job.progress_stage === "sampling" && job.current_step && job.total_steps) return `Diffusion step ${job.current_step} of ${job.total_steps}`;
  return STAGE_DEFS[job.progress_stage]?.detail ?? "Processing the current pipeline stage.";
}

function MetricTile({ label, value }) {
  return <div className="flex flex-col gap-1.5 rounded-xl bg-[#0f0d15] px-[18px] py-4"><span className="text-[11px] font-bold uppercase tracking-[0.05em] text-[#5f5b74]">{label}</span><span className="text-base font-bold tabular-nums text-[#f2f2f5]">{value}</span></div>;
}

function PromptText({ label, value, tone = "positive" }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    if (!value) return;
    try {
      await navigator.clipboard.writeText(value);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  };
  return (
    <article className="min-w-0 rounded-[12px] bg-[#0d0b12] p-4 ring-1 ring-inset ring-[#1c1826]">
      <div className="mb-3 flex items-center justify-between gap-3">
        <span className={`text-[10.5px] font-bold uppercase tracking-[0.08em] ${tone === "negative" ? "text-[#f58a8a]" : "text-[#a99dff]"}`}>{label}</span>
        {value && <button type="button" onClick={copy} className="flex items-center gap-1.5 rounded-[7px] bg-[#16131e] px-2.5 py-1.5 text-[10.5px] font-semibold text-[#a9a5b8] transition hover:bg-[#211c2b] hover:text-white" aria-label={`Copy ${label.toLowerCase()}`}>{copied ? <Check size={12} /> : <Copy size={12} />}{copied ? "Copied" : "Copy"}</button>}
      </div>
      <div className="scrollbar-subtle max-h-[220px] overflow-y-auto pr-1">
        <p className={`whitespace-pre-wrap break-words text-[12.5px] leading-[1.7] ${value ? "text-[#d6d3e0]" : "italic text-[#5f5b74]"}`}>{value || `No ${label.toLowerCase()} was recorded for this execution.`}</p>
      </div>
    </article>
  );
}

function PromptRecord({ positive, negative }) {
  return (
    <section className="rounded-[14px] bg-[#12101a] p-5">
      <div className="mb-4"><h2 className="text-[15px] font-bold text-[#f2f2f5]">Prompt record</h2><p className="mt-1 text-[11.5px] leading-5 text-[#75718a]">The resolved prompts stored with this execution remain available even when rendering fails.</p></div>
      <div className="grid grid-cols-2 gap-3 max-xl:grid-cols-1"><PromptText label="Positive prompt" value={positive} /><PromptText label="Negative prompt" value={negative} tone="negative" /></div>
    </section>
  );
}

function HorizontalSteps({ job }) {
  const stages = stagesForJob(job);
  const states = stages.map((stage, index) => stageStatus(job, stage, index, stages));
  const completed = states.filter((state) => state === "done").length;
  const focusIndex = Math.max(0, states.findIndex((state) => ["active", "failed", "cancelled"].includes(state)));
  const focusStage = stages[focusIndex] ?? stages.at(-1);
  const focusState = states[focusIndex] ?? "done";
  const focusDetail = job.status === "succeeded"
    ? "Every workflow stage completed and the output was stored successfully."
    : job.status === "failed"
      ? job.error_message ?? "The workflow stopped before producing an output."
      : focusState === "active" ? stageDetail(job) : STAGE_DEFS[focusStage]?.detail;
  return (
    <section className="overflow-hidden rounded-[14px] bg-[#12101a]">
      <div className="flex items-start justify-between gap-6 px-5 pb-4 pt-5">
        <div><h2 className="text-[15px] font-bold text-[#f2f2f5]">Workflow</h2><p className="mt-1 text-[11.5px] text-[#75718a]">Execution stages from request acceptance through storage.</p></div>
        <span className="shrink-0 rounded-[8px] bg-[#0d0b12] px-3 py-2 text-[11px] font-bold tabular-nums text-[#a9a5b8]">{job.status === "succeeded" ? stages.length : completed} / {stages.length} complete</span>
      </div>
      <div className="scrollbar-subtle overflow-x-auto px-5 pb-5">
        <div className="grid min-w-[920px]" style={{ gridTemplateColumns: `repeat(${stages.length}, minmax(118px, 1fr))` }}>
          {stages.map((stage, index) => {
        const state = states[index];
        const active = state === "active";
        const color = state === "done" ? COLORS.green : state === "active" ? COLORS.amber : state === "failed" ? COLORS.red : state === "cancelled" ? "#75718a" : "#3a3648";
        return (
          <div key={stage} className="min-w-0">
            <div className="flex items-center">
              <span className="grid h-[20px] w-[20px] shrink-0 place-items-center rounded-full bg-white/[0.045]"><span className={`h-[9px] w-[9px] rounded-full ${active ? "animate-pulse" : ""}`} style={{ background: color }} /></span>
              {index < stages.length - 1 && <span className="h-0.5 flex-1" style={{ background: state === "done" ? "rgba(62,207,126,0.38)" : "#211d2c" }} />}
            </div>
            <div className="pr-4 pt-2.5">
              <span className={`block truncate text-[12.5px] font-bold ${state === "pending" || state === "skipped" || state === "cancelled" ? "text-[#6a6580]" : "text-[#f2f2f5]"}`}>{STAGE_DEFS[stage]?.label ?? stage}</span>
              <span className="mt-1 block text-[10px] font-bold uppercase tracking-[0.06em]" style={{ color }}>{state}</span>
            </div>
          </div>
        );
          })}
        </div>
      </div>
      <div className="flex items-center gap-3 border-t border-[#1a1724] bg-[#0f0d15] px-5 py-3.5">
        <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: job.status === "succeeded" ? COLORS.green : focusState === "failed" ? COLORS.red : focusState === "active" ? COLORS.amber : "#75718a" }} />
        <span className="text-[11px] font-bold uppercase tracking-[0.06em] text-[#8b879a]">{job.status === "succeeded" ? "Completed" : STAGE_DEFS[focusStage]?.label ?? focusStage}</span>
        <span className="min-w-0 truncate text-[12px] text-[#a9a5b8]">{focusDetail}</span>
      </div>
    </section>
  );
}

function parameterRows(job) {
  const payload = job.request_payload ?? {};
  const generation = job.generation;
  const rows = [
    ["Pipeline", pipelineLabel(PIPELINE_TITLES, job.kind)],
    ["Preset", jobConfiguration(job)],
    ["Input", (payload.generation_id ?? payload.source_generation_id) != null ? `Generation #${payload.generation_id ?? payload.source_generation_id}` : null],
    ["Model", generation?.model ?? payload.model],
    ["Seed", generation?.seed],
    ["Sampler", generation ? `${generation.sampler_name}${generation.scheduler ? ` · ${generation.scheduler}` : ""}` : null],
    ["Steps", generation?.steps],
    ["Guidance", generation?.cfg_scale],
    ["Identity score", job.kind === "character" ? payload.character_identity_score : null],
    ["Identity provider", job.kind === "character" ? payload.identity_lock_provider : null],
    ["Denoise", job.kind === "img2img" ? payload.denoising_strength : null],
    ["Multiplier", job.kind === "pixel_upscale" ? payload.multiplier : null],
    ["Output", generation ? `${generation.width} × ${generation.height}` : null],
  ];
  return rows.filter(([, value]) => value !== null && value !== undefined && value !== "");
}

function observedCandidates(job) {
  const candidates = [{ key: "created", time: job.created_at, message: "Job accepted and queued." }];
  if (job.started_at) candidates.push({ key: "started", time: job.started_at, message: "Forge worker started the job." });
  if (job.progress_stage) candidates.push({ key: `stage:${job.progress_stage}`, time: null, message: `Observed stage: ${STAGE_DEFS[job.progress_stage]?.label ?? job.progress_stage}.` });
  if (job.current_step && job.total_steps) candidates.push({ key: `step:${job.current_step}`, time: null, message: `Sampling step ${job.current_step} of ${job.total_steps}.` });
  if (job.status === "succeeded") candidates.push({ key: "succeeded", time: job.finished_at, message: `Generation #${job.generation_id} stored successfully.`, tone: "success" });
  if (job.status === "failed") candidates.push({ key: "failed", time: job.finished_at, message: job.error_message ?? "The job failed.", tone: "error" });
  if (job.status === "cancelled") candidates.push({ key: "cancelled", time: job.finished_at, message: "The job was cancelled.", tone: "muted" });
  return candidates;
}

function clockLabel(value) {
  const date = value ? parseTimestamp(value) : new Date();
  return date ? date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }) : "--:--:--";
}

function JobLog({ job, entries }) {
  const live = job.status === "running" || job.status === "queued";
  return (
    <section className="flex flex-col gap-3">
      <div className="flex items-center justify-between"><h3 className="text-xs font-bold uppercase tracking-[0.05em] text-[#8b879a]">Activity</h3><span className={`flex items-center gap-1.5 text-[11.5px] font-semibold ${live ? "text-[#3ecf7e]" : "text-[#5f5b74]"}`}><span className={`h-1.5 w-1.5 rounded-full ${live ? "animate-pulse bg-[#3ecf7e]" : "bg-[#3f3b52]"}`} />{live ? "Live" : "Archived"}</span></div>
      <div className="scrollbar-subtle flex max-h-[260px] flex-col gap-1.5 overflow-y-auto rounded-[11px] bg-[#0a080f] px-[15px] py-3.5">
        {entries.map((entry) => <div key={entry.key} className="flex gap-2.5 font-mono text-[11.5px] leading-[1.5]"><span className="shrink-0 text-[#3f3b52]">{entry.clock}</span><span className={entry.tone === "error" ? "text-[#ff8f8f]" : entry.tone === "success" ? "text-[#3ecf7e]" : entry.tone === "muted" ? "text-[#5f5b74]" : "text-[#a9a5b8]"}>{entry.message}</span></div>)}
      </div>
      <p className="text-[10.5px] leading-4 text-[#4d4962]">Activity is reconstructed from real job snapshots observed by this browser; it is not a persistent server log.</p>
    </section>
  );
}

function lineageTitle(node) {
  if (node.pipeline_kind === "surprise_generate") return "Surprise generate";
  if (node.pipeline_kind === "img2img") return "Image transform";
  if (node.pipeline_kind === "character") return "Same character";
  if (node.pipeline_kind === "generate") return "Generate";
  if (node.pipeline_kind === "pixel_upscale") return "Pixel upscale";
  if (node.pipeline_kind === "upscale") return node.super_res ? "Super detail pass" : "High detail pass";
  return node.source_generation_id == null ? "Generate" : node.super_res ? "Super detail pass" : node.high_res ? "High detail pass" : "Derived image";
}

function Lineage({ job, nodes, loading, error }) {
  if (!job.generation_id) {
    return <section className="flex flex-col gap-3"><h3 className="text-xs font-bold uppercase tracking-[0.05em] text-[#8b879a]">Lineage</h3><p className="text-[11.5px] leading-5 text-[#5f5b74]">Lineage appears after this job produces an image.</p></section>;
  }
  return (
    <section className="flex flex-col gap-3">
      <h3 className="text-xs font-bold uppercase tracking-[0.05em] text-[#8b879a]">Lineage</h3>
      {loading ? <div className="h-[66px] animate-pulse rounded-[11px] bg-[#12101a]" />
        : error ? <p className="text-[11.5px] leading-5 text-[#f08a8a]">{error}</p>
          : <div className="flex flex-col gap-2">{nodes.map((node, index) => {
            const current = index === nodes.length - 1;
            const tag = current ? "current" : index === 0 ? "source" : "parent";
            return <div key={node.id} className={`relative flex items-center gap-3 rounded-[11px] px-[13px] py-[11px] ${current ? "bg-[#1a1526]" : "bg-[#12101a]"}`}>
              {index < nodes.length - 1 && <span className="absolute -bottom-2 left-[27px] h-2 w-px bg-[#2a2536]" />}
              <span className="grid h-11 w-[30px] shrink-0 place-items-center overflow-hidden rounded-md bg-gradient-to-br from-[#2a2438] to-[#171320]">{node.thumbnail_url ? <img src={node.thumbnail_url} alt="" loading="lazy" className="h-full w-full object-cover" /> : <ImageIcon size={13} className="text-[#5f5b74]" />}</span>
              <span className="min-w-0 flex-1"><span className="block truncate text-[12.5px] font-bold text-[#e7e5ee]">{lineageTitle(node)}</span><span className="mt-0.5 block text-[11px] text-[#75718a]">Generation #{node.id} · {node.width} × {node.height}</span></span>
              <span className={`rounded-full px-2 py-1 text-[10px] font-bold ${current ? "bg-[#8b7bff]/20 text-[#c9bfff]" : "bg-[#1c1826] text-[#75718a]"}`}>{tag}</span>
            </div>;
          })}</div>}
    </section>
  );
}

function JobLineage({ job }) {
  const [state, setState] = useState({ nodes: [], loading: Boolean(job.generation_id), error: "" });
  useEffect(() => {
    if (!job.generation_id) {
      setState({ nodes: [], loading: false, error: "" });
      return undefined;
    }
    let cancelled = false;
    setState((current) => ({ ...current, loading: true, error: "" }));
    getGenerationLineage(job.generation_id, job.generation?.access_token ?? null)
      .then((response) => {
        if (!cancelled) setState({ nodes: response.items ?? [], loading: false, error: "" });
      })
      .catch((requestError) => {
        if (!cancelled) setState({ nodes: [], loading: false, error: requestError.message });
      });
    return () => { cancelled = true; };
  }, [job.generation_id, job.generation?.access_token]);
  return <Lineage job={job} nodes={state.nodes} loading={state.loading} error={state.error} />;
}

function JobDetail({ job, logs, onBack, onRerun, onCancel, actionBusy, actionError }) {
  if (!job) return <div className="grid h-full place-items-center text-sm text-[#75718a]">This execution is no longer available.</div>;
  const active = job.status === "queued" || job.status === "running";
  const meta = statusMeta(job.status);
  const queueSeconds = job.started_at ? secondsBetween(job.created_at, job.started_at) : null;
  const elapsed = jobDurationSeconds(job);
  const generation = job.generation;
  const hasRenderSplit = generation
    && (generation.diffusion_duration_seconds != null || generation.upscale_duration_seconds != null);
  const split = hasRenderSplit
    ? `${formatDuration(generation.diffusion_duration_seconds)} diffusion · ${formatDuration(generation.upscale_duration_seconds)} upscale`
    : active ? "In progress" : "Not measured";
  const params = parameterRows(job);
  const prompt = job.positive_prompt ?? generation?.positive_prompt ?? job.request_payload?.resolved_positive_prompt ?? job.request_payload?.prompt;
  const negativePrompt = job.negative_prompt ?? generation?.negative_prompt ?? job.request_payload?.resolved_negative_prompt ?? job.request_payload?.negative_prompt;
  return (
    <div className="grid h-full min-h-0 grid-cols-[minmax(0,1fr)_minmax(340px,400px)] max-lg:grid-cols-1 max-lg:overflow-y-auto">
      <section className="scrollbar-subtle min-w-0 overflow-y-auto px-[34px] py-[30px] pb-11 max-lg:overflow-visible">
        <div className="flex flex-col gap-[26px]">
          <div className="flex items-start justify-between gap-6">
            <div className="flex flex-col gap-2">
              <button type="button" onClick={onBack} className="flex w-fit items-center gap-1.5 text-[12.5px] font-semibold text-[#8b879a] transition hover:text-[#e7e5ee]"><ChevronLeft size={14} /> All {pipelineLabel(PIPELINE_TITLES, job.kind).toLowerCase()} jobs</button>
              <div className="flex items-center gap-3"><h1 className="text-[23px] font-extrabold tracking-[-0.015em] tabular-nums text-[#f2f2f5]">{displayJobId(job)}</h1><StatusChip status={job.status} /></div>
              <p className="text-[13px] text-[#8b879a]">{pipelineLabel(PIPELINE_TITLES, job.kind)} · {jobConfiguration(job)} · {jobSource(job)}</p>
            </div>
            <div className="flex shrink-0 items-center gap-2">
              <button type="button" onClick={onRerun} disabled={Boolean(actionBusy)} className="flex items-center gap-2 rounded-[9px] bg-[#16131e] px-3.5 py-2.5 text-[13px] font-semibold text-[#d6d3e0] transition hover:bg-[#1f1a29] hover:text-white disabled:cursor-wait disabled:opacity-55">
                {actionBusy === "rerun" ? <LoaderCircle size={14} className="animate-spin" /> : <RotateCcw size={14} />} Re-run
              </button>
              {active && <button type="button" onClick={onCancel} disabled={Boolean(actionBusy)} className="flex items-center gap-2 rounded-[9px] bg-[#f05a5a]/10 px-3.5 py-2.5 text-[13px] font-semibold text-[#f58a8a] transition hover:bg-[#f05a5a]/20 hover:text-[#ffaaaa] disabled:cursor-wait disabled:opacity-55">
                {actionBusy === "cancel" ? <LoaderCircle size={14} className="animate-spin" /> : <Square size={13} fill="currentColor" />} Cancel
              </button>}
              {generation && <a href={generation.full_url} download={`nyx-forge-${generation.id}.png`} className="flex items-center gap-2 rounded-[9px] bg-[#16131e] px-3.5 py-2.5 text-[13px] font-semibold text-[#d6d3e0] transition hover:bg-[#1f1a29] hover:text-white"><Download size={14} /> Download</a>}
            </div>
          </div>

          {actionError && <div className="rounded-xl bg-[#f05a5a]/10 px-4 py-3 text-[13px] leading-5 text-[#ff8f8f]">{actionError}</div>}

          <div className="rounded-[14px] bg-[#12101a] px-[26px] py-6">
            <div className="flex items-end justify-between gap-5">
              <div className="min-w-0 flex-1"><span className="text-xs font-bold uppercase tracking-[0.05em] text-[#8b879a]">{job.status === "queued" ? "Waiting for worker" : STAGE_DEFS[job.progress_stage]?.label ?? job.progress_stage}</span><p className="mt-1.5 text-[15px] font-semibold text-[#f2f2f5]">{stageDetail(job)}</p></div>
              {job.current_image && <img src={job.current_image} alt="Live Forge preview" className="h-20 w-20 shrink-0 rounded-lg bg-[var(--well)] object-contain" />}
              <span className="flex shrink-0 items-baseline gap-1"><span className="text-[30px] font-extrabold leading-none tabular-nums text-[#f2f2f5]">{Math.round(job.progress_percent)}</span><span className="text-[15px] font-bold text-[#75718a]">%</span></span>
            </div>
            <div className="mt-[18px] h-[7px] overflow-hidden rounded bg-[#211d2c]"><span className="block h-full rounded transition-[width] duration-500" style={{ width: `${Math.max(active ? 2 : 0, job.progress_percent)}%`, background: progressFill(job) }} /></div>
            <div className="mt-[18px] flex justify-between text-[12.5px] text-[#75718a]"><span>{formatDuration(elapsed)} elapsed</span><span>{job.eta_seconds != null ? `~${Math.ceil(job.eta_seconds)}s remaining` : active ? "Calculating ETA" : meta.label}</span></div>
          </div>

          <div className="grid grid-cols-4 gap-2.5 max-xl:grid-cols-2">
            <MetricTile label="Queue wait" value={job.status === "queued" ? "Waiting" : formatDuration(queueSeconds)} />
            <MetricTile label="Total elapsed" value={formatDuration(elapsed)} />
            <MetricTile label="Output" value={generation ? `${generation.width} × ${generation.height}` : "Pending"} />
            <MetricTile label="Render split" value={split} />
          </div>

          {job.status === "failed" && <div className="rounded-xl bg-[#f05a5a]/10 px-4 py-3 text-[13px] leading-5 text-[#ff8f8f]">{job.error_message ?? "This execution failed before producing an image."}</div>}

          <PromptRecord positive={prompt} negative={negativePrompt} />

          <HorizontalSteps job={job} />
        </div>
      </section>

      <aside className="scrollbar-subtle overflow-y-auto bg-[#0d0b12] px-[26px] py-[30px] max-lg:overflow-visible">
        <div className="flex flex-col gap-[26px]">
          <section className="flex flex-col gap-3"><h3 className="text-xs font-bold uppercase tracking-[0.05em] text-[#8b879a]">Parameters</h3><div>{params.map(([key, value]) => <div key={key} className="flex items-start justify-between gap-3 py-2.5"><span className="text-[12.5px] text-[#75718a]">{key}</span><span className="max-w-[230px] break-words text-right text-[12.5px] font-semibold tabular-nums text-[#e7e5ee]">{String(value)}</span></div>)}</div></section>
          <div className="h-px bg-[#16131e]" />
          <JobLog job={job} entries={logs} />
          <div className="h-px bg-[#16131e]" />
          <JobLineage job={job} />
        </div>
      </aside>
    </div>
  );
}

export default function JobsPage({ routePath, onNavigate, account, guestHandles = [], onRememberJob }) {
  const initialRoute = routeState(routePath);
  const [view, setView] = useState(initialRoute.view);
  const [activeKind, setActiveKind] = useState(initialRoute.activeKind);
  const [filter, setFilter] = useState("all");
  const [jobs, setJobs] = useState([]);
  const [detailJob, setDetailJob] = useState(null);
  const [selectedId, setSelectedId] = useState(initialRoute.selectedId);
  const [nextCursor, setNextCursor] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState("");
  const [observedLogs, setObservedLogs] = useState({});
  const [actionBusy, setActionBusy] = useState("");
  const [actionError, setActionError] = useState("");

  useEffect(() => {
    const next = routeState(routePath);
    setView(next.view);
    setActiveKind(next.activeKind);
    setSelectedId(next.selectedId);
    setActionError("");
  }, [routePath]);

  const load = useCallback(async (beforeId = null, mode = "replace") => {
    // The pipeline index needs an unfiltered sample to compute honest
    // per-kind stats. "Active" and "Failed" are UI groups spanning two
    // persisted statuses, so keep those client-side; the API intentionally
    // accepts only one concrete status at a time.
    const statusFilter = view === "list" && filter === "succeeded" ? filter : null;
    try {
      const page = account
        ? await getJobs(beforeId, JOB_PAGE_SIZE, statusFilter)
        : {
            items: (await Promise.all(guestHandles.map((handle) => getJob(handle.job_id, handle.access_token).catch(() => null))))
              .filter(Boolean)
              .filter((job) => !statusFilter || filterJob(job, filter))
              .sort((left, right) => right.id - left.id),
            next_cursor: null,
          };
      setJobs((current) => {
        if (mode === "replace") return page.items;
        if (mode === "append") {
          const known = new Set(current.map((job) => job.id));
          return [...current, ...page.items.filter((job) => !known.has(job.id))];
        }
        const incoming = new Set(page.items.map((job) => job.id));
        return [...page.items, ...current.filter((job) => !incoming.has(job.id))];
      });
      if (mode !== "refresh") setNextCursor(page.next_cursor);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
      setLoadingMore(false);
    }
  }, [account, guestHandles, view, filter]);

  useEffect(() => {
    setNextCursor(null);
    load(null, "replace");
  }, [load]);

  const hasActive = jobs.some((job) => job.status === "running" || job.status === "queued");
  useEffect(() => {
    if (!hasActive) return undefined;
    const timer = window.setInterval(() => load(null, "refresh"), 900);
    return () => window.clearInterval(timer);
  }, [hasActive, load]);

  useEffect(() => {
    setObservedLogs((current) => {
      const next = { ...current };
      for (const job of jobs) {
        const entries = [...(next[job.id] ?? [])];
        const known = new Set(entries.map((entry) => entry.key));
        for (const candidate of observedCandidates(job)) {
          if (known.has(candidate.key)) continue;
          entries.push({ ...candidate, clock: clockLabel(candidate.time) });
          known.add(candidate.key);
        }
        next[job.id] = entries.slice(-48);
      }
      return next;
    });
  }, [jobs]);

  const jobsByKind = useMemo(() => {
    const grouped = {};
    for (const job of jobs) (grouped[job.kind] ??= []).push(job);
    return grouped;
  }, [jobs]);
  const scopedJobs = useMemo(() => jobs.filter((job) => job.kind === activeKind), [jobs, activeKind]);
  const listedSelection = useMemo(() => jobs.find((job) => job.id === selectedId) ?? null, [jobs, selectedId]);
  const selected = detailJob?.id === selectedId ? detailJob : listedSelection;

  useEffect(() => {
    if (view !== "detail" || !selectedId) {
      setDetailJob(null);
      return undefined;
    }
    let cancelled = false;
    const access = guestHandles.find((handle) => handle.job_id === selectedId)?.access_token ?? null;
    getJob(selectedId, access)
      .then((job) => {
        if (!cancelled) {
          setDetailJob(job);
          setActiveKind(job.kind);
        }
      })
      .catch((requestError) => {
        if (!cancelled) setError(requestError.message);
      });
    return () => { cancelled = true; };
  }, [guestHandles, selectedId, view]);
  const counts = useMemo(() => ({
    running: jobs.filter((job) => job.status === "running").length,
    queued: jobs.filter((job) => job.status === "queued").length,
    failedToday: jobs.filter((job) => job.status === "failed" && isToday(job.finished_at ?? job.created_at)).length,
  }), [jobs]);

  const refresh = () => {
    setLoading(view !== "detail");
    load(null, "replace");
  };
  const openPipeline = (kind) => {
    setActiveKind(kind);
    setFilter("all");
    setView("list");
    onNavigate(`/forge-deploy/pipelines/${kind}`);
  };
  const openDetail = (id) => {
    setActionError("");
    setSelectedId(id);
    setView("detail");
    onNavigate(`/forge-deploy/jobs/${id}`);
  };
  const showPipelines = () => {
    setView("pipelines");
    setActiveKind(null);
    setSelectedId(null);
    setActionError("");
    onNavigate("/forge-deploy");
  };
  const showList = () => {
    const kind = activeKind ?? selected?.kind;
    if (!kind) return showPipelines();
    setActiveKind(kind);
    setView("list");
    setSelectedId(null);
    setActionError("");
    onNavigate(`/forge-deploy/pipelines/${kind}`);
  };
  const loadMore = () => {
    if (!nextCursor || loadingMore) return;
    setLoadingMore(true);
    load(nextCursor, "append");
  };
  const selectedAccess = selected
    ? guestHandles.find((handle) => handle.job_id === selected.id)?.access_token ?? null
    : null;
  const rerunSelected = async () => {
    if (!selected || actionBusy) return;
    setActionBusy("rerun");
    setActionError("");
    try {
      const handle = await rerunJob(selected.id, selectedAccess);
      onRememberJob?.(handle, selected.kind);
      const fresh = await getJob(handle.job_id, handle.access_token ?? null);
      setJobs((current) => [fresh, ...current.filter((job) => job.id !== fresh.id)]);
      setDetailJob(fresh);
      setSelectedId(fresh.id);
      onNavigate(`/forge-deploy/jobs/${fresh.id}`);
    } catch (requestError) {
      setActionError(requestError.message);
    } finally {
      setActionBusy("");
    }
  };
  const cancelSelected = async () => {
    if (!selected || actionBusy) return;
    setActionBusy("cancel");
    setActionError("");
    try {
      const updated = await cancelJob(selected.id, selectedAccess);
      setJobs((current) => current.map((job) => job.id === updated.id ? updated : job));
      setDetailJob(updated);
    } catch (requestError) {
      setActionError(requestError.message);
    } finally {
      setActionBusy("");
    }
  };

  return (
    <main className="nyx-deploy-page forge-page flex h-[calc(100vh-88px)] flex-col overflow-hidden">
      <div className="flex shrink-0 items-start justify-between gap-[18px] px-8 pb-[18px] pt-[22px]">
        <div className="flex flex-col gap-[6px]">
          <h1 className="forge-display">{view === "pipelines" ? "Pipelines" : view === "list" ? pipelineLabel(PIPELINE_TITLES, activeKind) : selected ? displayJobId(selected) : `#${selectedId}`}</h1>
          <p className="forge-body">{view === "pipelines" ? "Monitor every generation and delivery pipeline." : view === "list" ? "Review executions, outcomes, and queue performance." : "Inspect execution stages, parameters, and output lineage."}</p>
        </div>
        <div className="flex items-center gap-[10px]">
          <div className="flex items-center gap-2 rounded-[8px] bg-[var(--panel)] px-3 py-[7px]">
            <span className={`h-[7px] w-[7px] rounded-full ${counts.running || counts.queued ? "dot-live bg-[var(--warn)]" : "bg-[var(--ok)]"}`} />
            <span className="text-[12.5px] font-semibold text-[var(--text-ui)] tabular-nums">{counts.running} running · {counts.queued} queued</span>
          </div>
          <button type="button" onClick={refresh} className="grid h-8 w-8 place-items-center rounded-[8px] bg-[var(--field)] text-[var(--text-muted)] transition-colors hover:bg-[var(--field-hover)] hover:text-[var(--text)]" aria-label="Refresh jobs">
            <RefreshCw size={14} />
          </button>
        </div>
      </div>
      <div className={`scrollbar-subtle min-h-0 min-w-0 flex-1 ${view === "detail" ? "overflow-hidden" : "overflow-x-hidden overflow-y-auto"}`}>
        {loading ? <div className="flex items-center gap-2 px-10 py-8 text-[13px] text-[#75718a]"><LoaderCircle size={16} className="animate-spin text-[#9d8cff]" /> Loading executions…</div>
          : error ? <div className="m-10 rounded-xl bg-[#f05a5a]/10 px-4 py-3 text-[13px] text-[#ff8f8f]">{error}</div>
            : view === "pipelines" ? <PipelineIndex jobsByKind={jobsByKind} counts={counts} onOpen={openPipeline} />
              : view === "list" ? <PipelineList kind={activeKind} jobs={scopedJobs} filter={filter} setFilter={setFilter} hasMore={Boolean(nextCursor)} loadingMore={loadingMore} onLoadMore={loadMore} onBack={showPipelines} onOpen={openDetail} />
                : <JobDetail job={selected} logs={observedLogs[selected?.id] ?? []} onBack={showList} onRerun={rerunSelected} onCancel={cancelSelected} actionBusy={actionBusy} actionError={actionError} />}
      </div>
    </main>
  );
}
