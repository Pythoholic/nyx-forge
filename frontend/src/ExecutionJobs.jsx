import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { displayJobId, displayQuality, displayStyle } from "./jobLabels";
import { cancelJob, deleteJobs, rerunJob } from "./api";
import { useAmbientColor } from "./ambientColor";
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle, Menu, MenuButton, MenuItem, MenuItems } from "@headlessui/react";
import { ChevronDown, ExternalLink, Image as ImageIcon, LoaderCircle, Maximize2, Minus, Play, Plus, RotateCcw, Trash2, X } from "lucide-react";

import { getJobs } from "./api";
import PhotoSwipeGallery from "./PhotoSwipeGallery";

const COLORS = {
  green: "var(--nyx-signal)",
  red: "var(--nyx-fail)",
  amber: "var(--nyx-warn)",
  purple: "var(--nyx-accent)",
  muted: "var(--nyx-label)",
  pending: "var(--nyx-line-strong)",
};

const FILTERS = [
  ["all", "All"],
  ["active", "Active"],
  ["succeeded", "Succeeded"],
  ["failed", "Failed"],
];

const JOB_PAGE_SIZE = 40;

function jobsAreUnchanged(current, next) {
  // Poll responses contain fresh objects, including nested payload/output data.
  // Compare their values so an identical response preserves React state identity.
  return JSON.stringify(current) === JSON.stringify(next);
}

const STATUS = {
  queued: { label: "Queued", color: COLORS.purple },
  running: { label: "Running", color: COLORS.amber },
  succeeded: { label: "Succeeded", color: COLORS.green },
  failed: { label: "Failed", color: COLORS.red },
  cancelled: { label: "Cancelled", color: COLORS.muted },
};

const STAGES = {
  queued: { label: "Queued", detail: "Accepted and waiting for the Forge worker." },
  prompt_generation: { label: "Prompt generation", detail: "Building the visual brief from the selected prompt engine." },
  prompt_ready: { label: "Prompt ready", detail: "The visual brief passed validation and is ready for Forge." },
  identity_analysis: { label: "Reference analysis", detail: "Validating the reference face and identity profile." },
  identity_generation: { label: "Scene generation", detail: "Generating a new SDXL scene with InstantID conditioning." },
  identity_lock: { label: "Identity lock", detail: "Transferring the reference identity into the generated face." },
  identity_refinement: { label: "Face refinement", detail: "Restoring swapped face texture before a conservative upscale." },
  identity_verification: { label: "Identity verification", detail: "Comparing the locked face with the reference before saving." },
  model_loading: { label: "Checkpoint", detail: "Loading the selected checkpoint and preparing inference." },
  sampling: { label: "Diffusion", detail: "Resolving the image through the model sampling steps." },
  interpolating: { label: "Interpolation", detail: "RIFE is creating smooth in-between frames without discarding any output." },
  refining: { label: "Detail pass", detail: "Refining fine structure at the selected output quality." },
  finalizing: { label: "Pixel decode", detail: "Decoding the latent result into image pixels." },
  decoding: { label: "Pixel decode", detail: "Decoding the latent result into image pixels." },
  upscaling: { label: "Upscale", detail: "Resizing the decoded image to its final dimensions." },
  encoding: { label: "Encoding", detail: "Packaging the completed frame sequence as an MP4." },
  saving: { label: "Store result", detail: "Writing the final image and metadata to local storage." },
};

const PIPELINES = {
  generate: ["queued", "model_loading", "sampling", "refining", "finalizing", "upscaling", "saving"],
  surprise_generate: ["queued", "prompt_generation", "prompt_ready", "model_loading", "sampling", "refining", "finalizing", "upscaling", "saving"],
  img2img: ["queued", "model_loading", "sampling", "refining", "finalizing", "saving"],
  character: ["queued", "identity_analysis", "identity_generation", "identity_lock", "identity_refinement", "identity_verification", "saving"],
  upscale: ["queued", "model_loading", "sampling", "refining", "finalizing", "upscaling", "saving"],
  pixel_upscale: ["queued", "upscaling", "saving"],
  video_t2v: ["queued", "submitted", "sampling", "interpolating", "upscaling", "encoding", "saving"],
  video_i2v: ["queued", "submitted", "sampling", "interpolating", "upscaling", "encoding", "saving"],
};
const IMG_PRESET_LABELS = {
  restore: "Restore / modernize",
  background: "Background change",
  style: "Style shift",
  variation: "New scene, same character",
};

function parseTimestamp(value) {
  if (!value) return null;
  const parsed = new Date(`${String(value).replace(" ", "T")}`);
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

function durationSeconds(job) {
  const start = parseTimestamp(job.started_at ?? job.created_at);
  const end = job.finished_at ? parseTimestamp(job.finished_at) : new Date();
  return start && end ? Math.max(0, (end.getTime() - start.getTime()) / 1000) : null;
}

function formatDuration(seconds) {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const rounded = Math.max(0, Math.round(seconds));
  if (rounded < 60) return `${rounded}s`;
  return `${Math.floor(rounded / 60)}m ${rounded % 60}s`;
}

function medianDuration(jobs) {
  const values = jobs
    .filter((job) => job.status === "succeeded" || job.status === "failed")
    .map(durationSeconds)
    .filter((value) => value != null)
    .sort((left, right) => left - right);
  if (!values.length) return null;
  const middle = Math.floor(values.length / 2);
  return values.length % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

function stagesForJob(job) {
  const pipeline = PIPELINES[job.kind] ?? PIPELINES.generate;
  const detailPass = job.kind === "upscale" || job.request_payload?.high_res || job.request_payload?.super_res;
  return pipeline.filter((stage) => {
    if (stage === "refining") return Boolean(detailPass);
    if (stage === "upscaling") return job.kind === "pixel_upscale" || Boolean(detailPass);
    return true;
  });
}

function stageState(job, stage, index, stages) {
  if (job.status === "succeeded") return "done";
  if (job.status === "queued") return stage === "queued" ? "active" : "pending";
  const current = stages.indexOf(job.progress_stage);
  const observed = current < 0 ? (job.status === "running" ? 1 : 0) : current;
  if (index < observed) return "done";
  if (index > observed) return job.status === "running" ? "pending" : "skipped";
  if (job.status === "running") return "active";
  if (job.status === "failed") return "failed";
  if (job.status === "cancelled") return "cancelled";
  return "pending";
}

function stateColor(state) {
  if (state === "done") return COLORS.green;
  if (state === "active") return COLORS.amber;
  if (state === "failed") return COLORS.red;
  if (state === "cancelled") return COLORS.muted;
  return COLORS.pending;
}

function modelName(job, modelProfiles) {
  if (job.kind === "video_t2v") return "Wan 2.1 T2V 1.3B";
  if (job.kind === "video_i2v") return "Wan 2.1 Fun Camera 1.3B";
  const model = job.request_payload?.model;
  if (!model) return "Forge default";
  // The profile carries the human name; fall back to the bare file stem rather
  // than the full "sd\name.safetensors [hash]" path.
  const profile = modelProfiles?.[model];
  if (profile?.display_name) return profile.display_name;
  return String(model).split(/[\/]/).pop().replace(/\.safetensors.*$/, "");
}

function timestamp(value) {
  if (!value) return "—";
  const date = new Date(value.replace(" ", "T"));
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleString(undefined, {
    month: "short", day: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit",
  });
}

function configuration(job) {
  const payload = job.request_payload ?? {};
  if (job.kind === "character") return `Same character · ${payload.character_generator ?? "InstantID SDXL"}`;
  if (job.kind === "img2img") return `${payload.preset ? IMG_PRESET_LABELS[payload.preset] ?? payload.preset : "Advanced transform"} · denoise ${Number(payload.denoising_strength ?? 0.4).toFixed(2)}`;
  if (job.kind === "upscale") return payload.super_res ? "Super detail pass" : "High detail pass";
  if (job.kind === "pixel_upscale") return `${payload.multiplier ?? "?"}× pixel resize`;
  if (job.kind === "video_t2v") return `Text to video · 832×480 · 3.1s`;
  if (job.kind === "video_i2v") return `${payload.camera_motion ?? "Animate image"} · 512×512 · 3.1s`;
  const quality = displayQuality(payload);
  const style = displayStyle({
    style: payload.style,
    style_variant: payload.style_variant ?? job.generation?.style_variant,
  });
  return `${style ? `${style} · ` : ""}${quality}`;
}

function source(job) {
  if (job.kind === "video_t2v") return "ComfyUI · 49 frames";
  if (job.kind === "video_i2v") return job.request_payload?.source_generation_id ? `from generation #${job.request_payload.source_generation_id}` : "uploaded source · 49 frames";
  const generationId = job.request_payload?.generation_id ?? job.request_payload?.source_generation_id ?? job.generation?.source_generation_id;
  const seed = job.generation?.seed;
  if (generationId != null) return `from generation #${generationId}${seed != null ? ` · seed ${seed}` : ""}`;
  if (seed != null) return `seed ${seed}`;
  // A raw checkpoint path repeated on every queued row carries no
  // information. Prefer the output size, and fall back to a bare model name
  // rather than the full "sd\name.safetensors [hash]" string.
  const payload = job.request_payload ?? {};
  if (payload.aspect_ratio) return String(payload.aspect_ratio);
  const model = payload.model;
  if (!model) return "Forge request";
  return String(model).split(/[\/]/).pop().replace(/\.safetensors.*$/, "");
}

function matchesFilter(job, filter) {
  if (filter === "all") return true;
  if (filter === "active") return job.status === "queued" || job.status === "running";
  if (filter === "failed") return job.status === "failed" || job.status === "cancelled";
  return job.status === filter;
}

function progressColor(job) {
  if (job.status === "succeeded") return COLORS.green;
  if (job.status === "failed") return COLORS.red;
  if (job.status === "cancelled") return COLORS.muted;
  return job.status === "queued" ? COLORS.purple : COLORS.amber;
}

function previewSource(job) {
  if (job.generation?.thumbnail_url) return job.generation.thumbnail_url;
  if (!job.current_image) return null;
  return job.current_image.startsWith("data:") ? job.current_image : `data:image/jpeg;base64,${job.current_image}`;
}

function fullPreviewSource(job) {
  if (job.generation?.full_url) return job.generation.full_url;
  return previewSource(job);
}

function PreviewDialog({ preview, onClose }) {
  return (
    <Dialog open={Boolean(preview)} onClose={onClose} className="relative z-[130]">
      <DialogBackdrop className="fixed inset-0 bg-black/90" />
      <div className="fixed inset-0 overflow-y-auto p-5"><div className="flex min-h-full items-center justify-center">
        <DialogPanel className="relative max-h-[94vh] max-w-[94vw]">
          <DialogTitle className="sr-only">{preview?.title ?? "Execution output preview"}</DialogTitle>
          <button type="button" onClick={onClose} aria-label="Close preview" className="absolute -right-3 -top-3 z-10 grid h-9 w-9 place-items-center rounded-full bg-[#211d2c] text-white shadow-xl hover:bg-[#302a40]"><X size={17} /></button>
          {preview && <img src={preview.url} alt={preview.title} className="max-h-[90vh] max-w-[90vw] rounded-[14px] object-contain shadow-2xl" />}
        </DialogPanel>
      </div></div>
    </Dialog>
  );
}

function Workflow({ job }) {
  const stages = stagesForJob(job);
  return (
    <div className="overflow-x-auto scrollbar-subtle">
      <div className="grid min-w-[830px]" style={{ gridTemplateColumns: `repeat(${stages.length}, minmax(120px, 1fr))` }}>
        {stages.map((stage, index) => {
          const state = stageState(job, stage, index, stages);
          const active = state === "active";
          const detail = active && job.current_step && job.total_steps
            ? `${STAGES[stage]?.detail} Step ${job.current_step} of ${job.total_steps}.`
            : STAGES[stage]?.detail;
          return (
            <div key={stage} className="min-w-0">
              <div className="flex items-center">
                <span className="grid h-[18px] w-[18px] shrink-0 place-items-center rounded-full bg-white/[0.04]"><span className={`h-[9px] w-[9px] rounded-full ${active ? "animate-pulse" : ""}`} style={{ background: stateColor(state) }} /></span>
                <span className={`h-0.5 flex-1 ${index === stages.length - 1 ? "bg-transparent" : "bg-[#1c1a25]"}`} />
              </div>
              <div className="pr-4 pt-2.5">
                <p className={`text-[12.5px] font-bold ${["pending", "skipped"].includes(state) ? "text-[#75718a]" : "text-[#f2f2f5]"}`}>{STAGES[stage]?.label ?? stage}</p>
                <p className="mt-1 text-[10.5px] font-semibold capitalize text-[#5f5b74]">{state}</p>
                <p className="mt-1 text-[11px] leading-[1.45] text-[#75718a]">{detail}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ExecutionRow({ job, expanded, onToggle, onPreview, modelProfiles, onCancel, onRerun, busy, checked, onToggleSelect }) {
  const meta = STATUS[job.status] ?? STATUS.queued;
  const active = job.status === "queued" || job.status === "running";
  const thumbnail = previewSource(job);
  const ambient = useAmbientColor(thumbnail);
  const fullPreview = fullPreviewSource(job);
  const progress = job.status === "succeeded" ? 100 : Math.max(active ? 2 : 0, job.progress_percent ?? 0);
  const outputTitle = job.status === "succeeded" ? "Output ready" : job.status === "running" ? "Rendering" : job.status === "queued" ? "No output yet" : "No output";
  const outputCaption = job.generation
    ? `${job.generation.width} × ${job.generation.height} · ${configuration(job)}`
    : job.status === "running" ? STAGES[job.progress_stage]?.detail ?? "Forge is processing this execution."
      : job.error_message ?? (job.status === "queued" ? "Waiting for a free worker." : `${meta.label} at ${Math.round(progress)}%.`);
  return (
    <article
      className={`relative shrink-0 overflow-hidden rounded-[12px] transition-colors ${expanded ? "bg-[#181423]" : "bg-[#12101a]"}`}
      style={ambient ? { backgroundImage: `linear-gradient(100deg, rgba(${ambient.join(",")},0.30) 0%, rgba(${ambient.join(",")},0.12) 45%, rgba(${ambient.join(",")},0) 78%)` } : undefined}
    >
      <button type="button" onClick={onToggle} aria-expanded={expanded} className="grid w-full min-w-[830px] grid-cols-[28px_92px_150px_210px_100px_minmax(160px,1fr)_78px_86px_64px] items-center gap-3.5 px-5 py-4 text-left transition hover:bg-[#181423]">
        <span className="flex items-center" onClick={(event) => event.stopPropagation()}>
          <input type="checkbox" aria-label={`Select ${displayJobId(job)}`} checked={checked} onChange={() => onToggleSelect(job.id)} className="h-3.5 w-3.5 accent-[#9d8cff]" />
        </span>
        <span className="min-w-0"><span className="block truncate text-[13px] font-bold tabular-nums text-[#f2f2f5]">{displayJobId(job)}</span></span>
        <span className="min-w-0 text-[11.5px] tabular-nums text-[#8b869e]">{timestamp(job.created_at)}</span>
        <span className="min-w-0"><span className="block truncate text-[12.5px] text-[#c7c3d4]">{modelName(job, modelProfiles)}</span><span className="mt-0.5 block truncate text-[11px] text-[#5f5b74]">{source(job)}</span></span>
        <span className="min-w-0 truncate text-[12px] font-semibold" style={{ color: meta.color }}>{meta.label}</span>
        <span className="flex items-center gap-2.5">
          <span className="block h-1.5 min-w-0 flex-1 overflow-hidden rounded-full bg-[#211d2c]"><span className="block h-full rounded-full transition-[width] duration-500" style={{ width: `${progress}%`, background: progressColor(job) }} /></span>
          <span className="w-9 shrink-0 text-right text-[11px] font-semibold tabular-nums" style={{ color: job.status === "running" ? progressColor(job) : "transparent" }}>{Math.round(progress)}%</span>
        </span>
        <span className="text-right text-[12.5px] tabular-nums text-[#c7c3d4]">{job.status === "queued" ? "—" : formatDuration(durationSeconds(job))}</span>
        <span className="text-right text-[11.5px] tabular-nums text-[#5f5b74]">{relativeTime(job.created_at)}</span>
        <span className="flex items-center justify-end gap-1">
          {(job.status === "queued" || job.status === "running") && (
            <button type="button" title={`Cancel ${displayJobId(job)}`} aria-label={`Cancel ${displayJobId(job)}`} disabled={busy} onClick={(event) => { event.stopPropagation(); onCancel(job); }} className="rounded-[6px] p-1.5 text-[#75718a] transition hover:bg-[#2a1f26] hover:text-[#ff8f8f] disabled:opacity-40"><X size={14} /></button>
          )}
          {(job.status === "failed" || job.status === "cancelled") && (
            <button type="button" title={`Re-run ${displayJobId(job)}`} aria-label={`Re-run ${displayJobId(job)}`} disabled={busy} onClick={(event) => { event.stopPropagation(); onRerun(job); }} className="rounded-[6px] p-1.5 text-[#75718a] transition hover:bg-[#1f2333] hover:text-[#9d8cff] disabled:opacity-40"><RotateCcw size={14} /></button>
          )}
          <ChevronDown size={15} className={`text-[#5f5b74] transition-transform ${expanded ? "rotate-180" : ""}`} />
        </span>
      </button>
      {expanded && <div className="bg-[#0f0d15]">
        <section className="px-5 pb-5 pt-[18px]"><p className="mb-3.5 text-[10.5px] font-bold uppercase tracking-[0.07em] text-[#5f5b74]">Workflow progress</p><Workflow job={job} /></section>
        <div className="h-px bg-[#1a1724]" />
        <section className="flex items-center gap-3.5 px-5 py-3.5">
          {job.generation ? (
            <PhotoSwipeGallery entries={[{ image: job.generation, index: 0, thumbnail }]} selectedIndex={0} onSelect={() => {}} className="contents">
              {({ ref, open }) => <>
                <button ref={ref} type="button" onClick={open} className="grid h-[60px] w-11 shrink-0 place-items-center overflow-hidden rounded-[8px] bg-gradient-to-br from-[#2c2431] to-[#17121a]" aria-label={`Preview ${displayJobId(job)} output`}>
                  <img src={thumbnail} alt="" className="h-full w-full object-cover" />
                </button>
                <span className="min-w-0 flex-1"><span className="block text-[12.5px] font-bold text-[#e7e5ee]">{outputTitle}</span><span className="mt-1 block truncate text-[11.5px] text-[#75718a]">{outputCaption}</span></span>
                <button type="button" onClick={open} className="flex shrink-0 items-center gap-2 rounded-[9px] bg-[#16131e] px-3.5 py-2.5 text-[12.5px] font-semibold text-[#d6d3e0] transition hover:bg-[#211c2b] hover:text-white"><Maximize2 size={13} /> Full preview</button>
                <a href={`/forge-deploy/jobs/${job.id}`} target="_blank" rel="noopener noreferrer" className="flex shrink-0 items-center gap-2 rounded-[9px] bg-[#16131e] px-3.5 py-2.5 text-[12.5px] font-semibold text-[#e7e5ee] transition hover:bg-[#211c2b] hover:text-white">Open deployment <ExternalLink size={13} /></a>
              </>}
            </PhotoSwipeGallery>
          ) : <>
            <button type="button" disabled={!fullPreview} onClick={() => fullPreview && onPreview({ url: fullPreview, title: `${displayJobId(job)} output` })} className="grid h-[60px] w-11 shrink-0 place-items-center overflow-hidden rounded-[8px] bg-gradient-to-br from-[#2c2431] to-[#17121a] disabled:cursor-default">
              {thumbnail ? <img src={thumbnail} alt="" className="h-full w-full object-cover" /> : active ? <LoaderCircle size={15} className="animate-spin text-[#75718a]" /> : <ImageIcon size={15} className="text-[#5f5b74]" />}
            </button>
            <span className="min-w-0 flex-1"><span className="block text-[12.5px] font-bold text-[#e7e5ee]">{outputTitle}</span><span className="mt-1 block truncate text-[11.5px] text-[#75718a]">{outputCaption}</span></span>
            <a href={`/forge-deploy/jobs/${job.id}`} target="_blank" rel="noopener noreferrer" className="flex shrink-0 items-center gap-2 rounded-[9px] bg-[#16131e] px-3.5 py-2.5 text-[12.5px] font-semibold text-[#e7e5ee] transition hover:bg-[#211c2b] hover:text-white">Open deployment <ExternalLink size={13} /></a>
          </>}
        </section>
      </div>}
    </article>
  );
}

function ForgeExecutionRow({ job, expanded, onToggle, onPreview, modelProfiles, onCancel, onRerun, busy, checked, onToggleSelect }) {
  const status = STATUS[job.status] ?? STATUS.queued;
  const active = job.status === "queued" || job.status === "running";
  const progress = job.status === "succeeded" ? 100 : Math.max(active ? 2 : 0, job.progress_percent ?? 0);
  const thumbnail = previewSource(job);
  const fullPreview = fullPreviewSource(job);
  const stage = STAGES[job.progress_stage] ?? { label: job.progress_stage || "Waiting", detail: "Preparing the job." };
  return (
    <article className={`nyx-forge-job${expanded ? " is-expanded" : ""}`}>
      <div className="nyx-forge-job-main">
        <input type="checkbox" aria-label={`Select ${displayJobId(job)}`} checked={checked} onChange={() => onToggleSelect(job.id)} />
        <button type="button" className="nyx-forge-job-id" onClick={onToggle} aria-expanded={expanded}>
          <strong>{displayJobId(job)}</strong>
          <small>{timestamp(job.created_at)} · {modelName(job, modelProfiles)}</small>
        </button>
        <button type="button" className={`nyx-forge-job-progress is-${job.status}`} onClick={onToggle} aria-expanded={expanded}>
          <span><b>{status.label}</b><i>{Math.round(progress)}%</i></span>
          <span className="nyx-forge-progress-track"><i style={{ width: `${progress}%`, background: progressColor(job) }} /></span>
        </button>
        <button type="button" className="nyx-forge-job-time" onClick={onToggle} aria-expanded={expanded}>
          <strong>{job.status === "queued" ? "—" : formatDuration(durationSeconds(job))}</strong>
          <small>{relativeTime(job.created_at)}</small>
        </button>
        <span className="nyx-forge-job-actions">
          {active && <button type="button" title={`Cancel ${displayJobId(job)}`} disabled={busy} onClick={() => onCancel(job)}><X size={14} /></button>}
          {(job.status === "failed" || job.status === "cancelled") && <button type="button" title={`Re-run ${displayJobId(job)}`} disabled={busy} onClick={() => onRerun(job)}><RotateCcw size={14} /></button>}
          <button type="button" title={expanded ? "Collapse details" : "Show details"} onClick={onToggle}>{expanded ? <Minus size={14} /> : <Plus size={14} />}</button>
        </span>
      </div>
      {expanded && <div className="nyx-forge-job-detail">
        <section className="nyx-forge-workflow"><h3>Workflow progress</h3><Workflow job={job} /></section>
        <div className="nyx-forge-job-facts">
          <span><small>Model</small><strong>{modelName(job, modelProfiles)}</strong></span>
          <span><small>Source</small><strong>{source(job)}</strong></span>
          <span><small>Stage</small><strong>{stage.label}</strong></span>
          <span><small>Configuration</small><strong>{configuration(job)}</strong></span>
        </div>
        <p>{stage.detail}</p>
        <section className="nyx-forge-output">
          {job.generation ? <PhotoSwipeGallery entries={[{ image: job.generation, index: 0, thumbnail }]} selectedIndex={0} onSelect={() => {}} className="contents">
            {({ ref, open }) => <>
              <button ref={ref} type="button" onClick={open} className="nyx-forge-output-image" aria-label={`Preview ${displayJobId(job)} output`}><img src={thumbnail} alt="" /></button>
              <span className="nyx-forge-output-copy"><strong>Output ready</strong><small>{job.generation.width} × {job.generation.height} · {configuration(job)}</small></span>
              <button type="button" onClick={open} className="nyx-forge-output-action"><Maximize2 size={12} /> Full preview</button>
            </>}
          </PhotoSwipeGallery> : <>
            <button type="button" disabled={!fullPreview} onClick={() => fullPreview && onPreview({ url: fullPreview, title: `${displayJobId(job)} live output` })} className="nyx-forge-output-image" aria-label={`Preview ${displayJobId(job)} live output`}>
              {thumbnail ? <img src={thumbnail} alt="" /> : active ? <LoaderCircle size={15} className="animate-spin" /> : <ImageIcon size={15} />}
            </button>
            <span className="nyx-forge-output-copy"><strong>{active ? "Live render" : "No output"}</strong><small>{active ? stage.detail : job.error_message ?? "This execution did not create an image."}</small></span>
          </>}
          <a href={`/forge-deploy/jobs/${job.id}`} target="_blank" rel="noopener noreferrer" className="nyx-forge-output-action">Open deployment <ExternalLink size={12} /></a>
        </section>
      </div>}
    </article>
  );
}

export default function ExecutionJobs({ account, refreshSignal, onNavigate, modelProfiles = {}, jobKinds = null, description = "Every persisted job sent to Forge, newest first.", compact = false, variant = "default" }) {
  const [jobs, setJobs] = useState([]);
  const [filter, setFilter] = useState("all");
  const [expandedId, setExpandedId] = useState(null);
  const [preview, setPreview] = useState(null);
  const [busyJobId, setBusyJobId] = useState(null);
  const [bulkBusy, setBulkBusy] = useState(false);
  const [selected, setSelected] = useState(() => new Set());
  const [actionError, setActionError] = useState("");
  const [actionNotice, setActionNotice] = useState("");
  const [confirmDelete, setConfirmDelete] = useState(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [nextCursor, setNextCursor] = useState(null);
  const [error, setError] = useState("");
  const initialized = useRef(false);
  const newestSeen = useRef(0);
  const loadMoreRef = useRef(null);
  const jobKindsKey = jobKinds?.join(",") ?? "";

  const refreshLatest = useCallback(async ({ replace = false } = {}) => {
    if (!account) return;
    try {
      const page = await getJobs(null, JOB_PAGE_SIZE, null, jobKindsKey ? jobKindsKey.split(",") : null);
      setJobs((current) => {
        if (replace) {
          const next = [...page.items].sort((a, b) => b.id - a.id);
          return jobsAreUnchanged(current, next) ? current : next;
        }
        const incoming = new Set(page.items.map((job) => job.id));
        const next = [...page.items, ...current.filter((job) => !incoming.has(job.id))].sort((a, b) => b.id - a.id);
        return jobsAreUnchanged(current, next) ? current : next;
      });
      if (replace) setNextCursor(page.next_cursor);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, [account, jobKindsKey]);
  const runJobAction = async (job, action) => {
    setActionError("");
    setBusyJobId(job.id);
    try {
      await action(job.id, job.access_token ?? null);
      await refreshLatest();
    } catch (requestError) {
      setActionError(requestError.message);
    } finally {
      setBusyJobId(null);
    }
  };
  const handleCancel = (job) => runJobAction(job, cancelJob);
  const handleRerun = (job) => runJobAction(job, rerunJob);
  const loadOlder = useCallback(async () => {
    if (!account || !nextCursor || loadingMore) return;
    setLoadingMore(true);
    try {
      const page = await getJobs(nextCursor, JOB_PAGE_SIZE, null, jobKindsKey ? jobKindsKey.split(",") : null);
      setJobs((current) => {
        const known = new Set(current.map((job) => job.id));
        return [...current, ...page.items.filter((job) => !known.has(job.id))];
      });
      setNextCursor(page.next_cursor);
      setError("");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoadingMore(false);
    }
  }, [account, jobKindsKey, loadingMore, nextCursor]);
  // Re-run everything that did not finish, oldest first so the queue keeps the
  // order the user originally created.
  const [, setTick] = useState(0);
  const visible = useMemo(() => jobs.filter((job) => matchesFilter(job, filter)), [filter, jobs]);
  // Selection drives bulk actions. A job is selectable when the API would
  // accept the action for it, so the dropdown never offers an impossible run.
  const canRerun = (job) => job.status === "failed" || job.status === "cancelled";
  const canCancel = (job) => job.status === "queued" || job.status === "running";
  const canDelete = (job) => job.status === "succeeded" || job.status === "failed" || job.status === "cancelled";
  const selectedJobs = useMemo(
    () => visible.filter((job) => selected.has(job.id)),
    [visible, selected],
  );
  const toggleSelected = (jobId) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(jobId)) next.delete(jobId);
      else next.add(jobId);
      return next;
    });
  };
  const allVisibleSelected = visible.length > 0 && visible.every((job) => selected.has(job.id));
  const toggleSelectAll = () => {
    setSelected((current) => {
      const next = new Set(current);
      if (allVisibleSelected) visible.forEach((job) => next.delete(job.id));
      else visible.forEach((job) => next.add(job.id));
      return next;
    });
  };
  // A success notice should not need dismissing; clear it on its own.
  useEffect(() => {
    if (!actionNotice) return undefined;
    const timer = window.setTimeout(() => setActionNotice(""), 5000);
    return () => window.clearTimeout(timer);
  }, [actionNotice]);

  const requestDelete = () => {
    const targets = selectedJobs.filter(canDelete);
    if (targets.length) setConfirmDelete(targets);
  };
  const confirmDeleteSelected = async () => {
    const targets = confirmDelete ?? [];
    setConfirmDelete(null);
    if (!targets.length) return;
    setActionError("");
    setActionNotice("");
    setBulkBusy(true);
    try {
      const result = await deleteJobs(targets.map((job) => job.id));
      setSelected(new Set());
      // Replace rather than merge: the deleted rows are gone server-side and
      // a merge would keep showing them until a full reload.
      await refreshLatest({ replace: true });
      setActionNotice(`Deleted ${result.deleted} execution record${result.deleted === 1 ? "" : "s"}.`);
    } catch (requestError) {
      setActionError(requestError.message);
    } finally {
      setBulkBusy(false);
    }
  };
  const runBulk = async (action, eligible, label) => {
    const targets = selectedJobs.filter(eligible);
    if (!targets.length) return;
    setActionError("");
    setBulkBusy(true);
    const failures = [];
    // Oldest first so the queue keeps the order the jobs were created in.
    for (const job of targets.slice().reverse()) {
      try {
        await action(job.id, job.access_token ?? null);
      } catch (requestError) {
        failures.push(`${displayJobId(job)}: ${requestError.message}`);
      }
    }
    setBulkBusy(false);
    setSelected(new Set());
    if (failures.length) setActionError(`${failures.length} of ${targets.length} could not ${label} · ${failures[0]}`);
    await refreshLatest();
  };



  useEffect(() => {
    if (!account) {
      setJobs([]);
      setLoading(false);
      return undefined;
    }
    let cancelled = false;
    setLoading(true);
    (async () => {
      try {
        const page = await getJobs(null, JOB_PAGE_SIZE, null, jobKindsKey ? jobKindsKey.split(",") : null);
        if (!cancelled) {
          setJobs(page.items);
          setNextCursor(page.next_cursor);
          setError("");
          setLoading(false);
          initialized.current = false;
          newestSeen.current = 0;
        }
      } catch (requestError) {
        if (!cancelled) {
          setError(requestError.message);
          setLoading(false);
        }
      }
    })();
    return () => { cancelled = true; };
  }, [account, jobKindsKey]);

  useEffect(() => {
    const target = loadMoreRef.current;
    if (!target || !nextCursor || loadingMore) return undefined;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) loadOlder();
    }, { rootMargin: "240px" });
    observer.observe(target);
    return () => observer.disconnect();
  }, [loadOlder, loadingMore, nextCursor]);

  const hasActive = jobs.some((job) => job.status === "queued" || job.status === "running");
  useEffect(() => {
    if (!account) return undefined;
    refreshLatest();
    const delay = hasActive ? 900 : 4000;
    const timer = window.setInterval(refreshLatest, delay);
    return () => window.clearInterval(timer);
  }, [account, hasActive, refreshLatest, refreshSignal]);

  // Elapsed time for a running job is measured against "now", which only moves
  // when this component re-renders. Without a tick the duration sits still
  // between polls and jumps at the end.
  useEffect(() => {
    if (!hasActive) return undefined;
    const timer = window.setInterval(() => setTick((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [hasActive]);

  useEffect(() => {
    if (!jobs.length) return;
    const newestActive = jobs.find((job) => job.status === "queued" || job.status === "running");
    if (!initialized.current) {
      setExpandedId(newestActive?.id ?? null);
      newestSeen.current = jobs[0].id;
      initialized.current = true;
      return;
    }
    if (jobs[0].id > newestSeen.current) {
      newestSeen.current = jobs[0].id;
      if (jobs[0].status === "queued" || jobs[0].status === "running") setExpandedId(jobs[0].id);
    }
  }, [jobs]);

  const activeCount = jobs.filter((job) => job.status === "queued" || job.status === "running").length;
  const failedCount = jobs.filter((job) => job.status === "failed").length;

  if (variant === "forgeai") return (
    <section className="nyx-forge-jobs">
      <header className="nyx-forge-jobs-title"><h1><b>&gt;_</b> Execution jobs</h1><span>{description}</span></header>
      <div className="nyx-forge-job-summary">
        {[[jobs.length, "Total", ""], [activeCount, "Active", "is-active"], [failedCount, "Failed", "is-failed"], [formatDuration(medianDuration(jobs)), "Median", ""]].map(([value, label, tone]) => <span key={label} className={tone}><strong>{value}</strong><small>{label}</small></span>)}
      </div>
      <div className="nyx-forge-job-filterbar">
        <div className="forge-seg" role="tablist" aria-label="Execution status">
          {FILTERS.map(([value, label]) => <button key={value} type="button" role="tab" aria-selected={filter === value} onClick={() => setFilter(value)}>{label}</button>)}
        </div>
        <div className="nyx-forge-job-tools">
          {selected.size > 0 && <Menu as="div" className="relative">
            <MenuButton disabled={bulkBusy} className="nyx-forge-bulk-button">{bulkBusy ? <LoaderCircle size={12} className="animate-spin" /> : <Play size={12} />}{selected.size} selected <ChevronDown size={12} /></MenuButton>
            <MenuItems anchor="bottom end" className="nyx-menu [--anchor-gap:6px]">
              <MenuItem><button type="button" disabled={!selectedJobs.some(canRerun)} onClick={() => runBulk(rerunJob, canRerun, "re-run")} className="nyx-menu-item"><RotateCcw size={13} /> Re-run {selectedJobs.filter(canRerun).length}</button></MenuItem>
              <MenuItem><button type="button" disabled={!selectedJobs.some(canCancel)} onClick={() => runBulk(cancelJob, canCancel, "cancel")} className="nyx-menu-item is-danger"><X size={13} /> Cancel {selectedJobs.filter(canCancel).length}</button></MenuItem>
              <MenuItem><button type="button" disabled={!selectedJobs.some(canDelete)} onClick={requestDelete} className="nyx-menu-item is-danger"><Trash2 size={13} /> Delete {selectedJobs.filter(canDelete).length}</button></MenuItem>
              <MenuItem><button type="button" onClick={() => setSelected(new Set())} className="nyx-menu-item">Clear selection</button></MenuItem>
            </MenuItems>
          </Menu>}
          <span>{visible.length} of {jobs.length}</span>
        </div>
      </div>
      {actionNotice && <div className="nyx-forge-notice is-success"><span>{actionNotice}</span><button type="button" onClick={() => setActionNotice("")}><X size={12} /></button></div>}
      {actionError && <div className="nyx-forge-notice is-error"><span>{actionError}</span><button type="button" onClick={() => setActionError("")}><X size={12} /></button></div>}
      <div className="nyx-forge-job-list scrollbar-subtle">
        {loading ? <div className="nyx-forge-jobs-state"><LoaderCircle size={15} className="animate-spin" /> Loading persisted executions…</div>
          : error ? <div className="nyx-forge-jobs-state is-error">{error}</div>
            : visible.length ? visible.map((job) => <ForgeExecutionRow key={job.id} job={job} modelProfiles={modelProfiles} onCancel={handleCancel} onRerun={handleRerun} onPreview={setPreview} busy={busyJobId === job.id || bulkBusy} checked={selected.has(job.id)} onToggleSelect={toggleSelected} expanded={expandedId === job.id} onToggle={() => setExpandedId((current) => current === job.id ? null : job.id)} />)
              : <div className="nyx-forge-jobs-state"><p>No executions match this filter.</p>{filter !== "all" && <button type="button" onClick={() => setFilter("all")}>Show all executions</button>}</div>}
        {nextCursor && <div ref={loadMoreRef} className="nyx-forge-load-more">{loadingMore ? "Loading older executions…" : "Scroll for older executions"}</div>}
      </div>
      <PreviewDialog preview={preview} onClose={() => setPreview(null)} />
      <Dialog open={Boolean(confirmDelete)} onClose={() => setConfirmDelete(null)} className="relative z-[130]">
        <DialogBackdrop className="fixed inset-0 bg-black/70" />
        <div className="fixed inset-0 grid place-items-center p-6">
          <DialogPanel className="nyx-switch-panel w-full max-w-[420px] p-6">
            <DialogTitle className="text-[16px] font-bold text-[var(--nyx-ink)]">Delete {confirmDelete?.length ?? 0} execution record{(confirmDelete?.length ?? 0) === 1 ? "" : "s"}?</DialogTitle>
            <p className="mt-2 text-[12px] leading-5 text-[var(--nyx-muted)]">This removes the job history entries only. Generated images remain in the gallery.</p>
            <div className="mt-5 flex justify-end gap-2"><button type="button" onClick={() => setConfirmDelete(null)} className="nyx-forge-dialog-button">Cancel</button><button type="button" onClick={confirmDeleteSelected} className="nyx-forge-dialog-button is-danger"><Trash2 size={13} /> Delete</button></div>
          </DialogPanel>
        </div>
      </Dialog>
    </section>
  );

  if (compact) return (
    <section className="nyx-execution-jobs is-compact flex min-h-[420px] flex-col rounded-[8px] bg-[var(--panel)] p-4">
      <div className="flex items-start justify-between gap-4"><div><h2 className="forge-display">Execution jobs</h2><p className="mt-1 forge-body">{description}</p></div><span className="rounded-[8px] bg-[var(--well)] px-2.5 py-1.5 text-[10.5px] font-bold text-[var(--text-muted)]">{activeCount} active</span></div>
      <div className="forge-seg mt-4 grid grid-cols-4 rounded-[9px] bg-[var(--well)] p-1">{FILTERS.map(([value,label])=><button key={value} type="button" aria-selected={filter===value} onClick={()=>setFilter(value)}>{label}</button>)}</div>
      <div className="scrollbar-subtle mt-4 min-h-0 flex-1 space-y-2 overflow-y-auto">
        {loading ? <div className="flex items-center gap-2 rounded-[11px] bg-[var(--well)] p-4 text-[12px] text-[var(--text-muted)]"><LoaderCircle size={14} className="animate-spin"/>Loading executions…</div>
          : error ? <div className="rounded-[11px] bg-[var(--danger-wash)] p-4 text-[12px] text-[var(--danger)]">{error}</div>
          : visible.length ? visible.map((job) => {
            const stage = STAGES[job.progress_stage] ?? { label: job.progress_stage || "Waiting", detail: "Preparing the job." };
            const status = STATUS[job.status] ?? STATUS.queued;
            return <article key={job.id} className="rounded-[11px] bg-[var(--well)] p-3 ring-1 ring-inset ring-[var(--divider)]">
              <div className="flex items-center justify-between gap-3"><div className="min-w-0"><div className="flex items-center gap-2"><span className="text-[12px] font-extrabold text-[var(--text)]">{displayJobId(job)}</span><span className="text-[10.5px] font-bold" style={{color:status.color}}>{status.label}</span></div><p className="mt-1 truncate text-[11px] font-semibold text-[var(--text-muted)]">{stage.label} · {stage.detail}</p></div><span className="shrink-0 text-[11px] font-bold tabular-nums text-[var(--text-muted)]">{Math.round(job.progress_percent || 0)}%</span></div>
              <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[var(--raised)]"><div className="h-full rounded-full bg-[var(--accent)] transition-[width]" style={{width:`${Math.max(0,Math.min(100,job.progress_percent||0))}%`}}/></div>
              {(job.status === "queued" || job.status === "running") && <button type="button" disabled={busyJobId===job.id} onClick={()=>handleCancel(job)} className="mt-2 text-[10.5px] font-bold text-[var(--danger)] disabled:opacity-50">Cancel</button>}
            </article>;
          }) : <div className="rounded-[11px] bg-[var(--well)] p-8 text-center text-[12px] text-[var(--text-muted)]">No executions match this filter.</div>}
        {nextCursor && <div ref={loadMoreRef} className="py-2 text-center text-[11px] text-[var(--text-muted)]">{loadingMore ? "Loading older executions…" : "Scroll for older executions"}</div>}
      </div>
    </section>
  );

  return (
    <section className="nyx-execution-jobs flex min-h-0 flex-1 flex-col">
      <div className="flex shrink-0 items-end justify-between gap-6 pb-[18px]">
        <div><h1 className="forge-display">Execution jobs</h1><p className="mt-1.5 forge-body">{description}</p></div>
        <div className="flex gap-5">
          {[[jobs.length, "Executions", "#f2f2f5"], [activeCount, "Active", COLORS.amber], [failedCount, "Failed", COLORS.red], [formatDuration(medianDuration(jobs)), "Median", "#f2f2f5"]].map(([value, label, color]) => <span key={label} className="flex flex-col items-end"><span className="text-[19px] font-extrabold tabular-nums" style={{ color }}>{value}</span><span className="text-[10.5px] font-bold uppercase tracking-[0.07em] text-[#5f5b74]">{label}</span></span>)}
        </div>
      </div>
      <div className="flex shrink-0 items-center justify-between gap-4 pb-4">
        <div className="flex rounded-[6px] border-2 border-[var(--nyx-line)] bg-[var(--nyx-input)] p-1">{FILTERS.map(([value, label]) => <button key={value} type="button" onClick={() => setFilter(value)} aria-selected={filter === value} className={`rounded-[4px] border-2 px-3.5 py-2 text-[11px] font-semibold uppercase tracking-[.08em] transition ${filter === value ? "border-[var(--nyx-accent-line)] bg-[var(--nyx-accent-soft)] text-[var(--nyx-accent)]" : "border-transparent text-[var(--nyx-label)] hover:text-[var(--nyx-ink)]"}`}>{label}</button>)}</div>
        <div className="flex items-center gap-3">
          {selected.size > 0 && (
            <Menu as="div" className="relative">
              <MenuButton disabled={bulkBusy} className="flex items-center gap-1.5 rounded-[8px] bg-[#1c1830] px-3 py-2 text-[11.5px] font-bold text-[#c4b9ff] transition hover:bg-[#241f3d] hover:text-white disabled:opacity-50">
                {bulkBusy ? <LoaderCircle size={13} className="animate-spin" /> : <Play size={13} />}
                {selected.size} selected
                <ChevronDown size={13} />
              </MenuButton>
              <MenuItems anchor="bottom end" className="z-50 mt-1 w-56 rounded-[10px] bg-[#181423] p-1 shadow-[0_18px_44px_rgba(0,0,0,0.55)] focus:outline-none">
                <MenuItem>
                  <button type="button" disabled={!selectedJobs.some(canRerun)} onClick={() => runBulk(rerunJob, canRerun, "re-run")} className="flex w-full items-center gap-2 rounded-[7px] px-3 py-2.5 text-left text-[12px] font-semibold text-[#e7e5ee] data-[focus]:bg-[#241f3d] disabled:opacity-35">
                    <RotateCcw size={13} /> Re-run {selectedJobs.filter(canRerun).length} job{selectedJobs.filter(canRerun).length === 1 ? "" : "s"}
                  </button>
                </MenuItem>
                <MenuItem>
                  <button type="button" disabled={!selectedJobs.some(canCancel)} onClick={() => runBulk(cancelJob, canCancel, "cancel")} className="flex w-full items-center gap-2 rounded-[7px] px-3 py-2.5 text-left text-[12px] font-semibold text-[#ff9d9d] data-[focus]:bg-[#2a1f26] disabled:opacity-35">
                    <X size={13} /> Cancel {selectedJobs.filter(canCancel).length} job{selectedJobs.filter(canCancel).length === 1 ? "" : "s"}
                  </button>
                </MenuItem>
                <MenuItem>
                  <button type="button" disabled={!selectedJobs.some(canDelete)} onClick={requestDelete} className="flex w-full items-center gap-2 rounded-[7px] px-3 py-2.5 text-left text-[12px] font-semibold text-[#ff9d9d] data-[focus]:bg-[#2a1f26] disabled:opacity-35">
                    <Trash2 size={13} /> Delete {selectedJobs.filter(canDelete).length} record{selectedJobs.filter(canDelete).length === 1 ? "" : "s"}
                  </button>
                </MenuItem>
                <MenuItem>
                  <button type="button" onClick={() => setSelected(new Set())} className="flex w-full items-center gap-2 rounded-[7px] px-3 py-2.5 text-left text-[12px] font-semibold text-[#8b879a] data-[focus]:bg-[#241f3d]">
                    Clear selection
                  </button>
                </MenuItem>
              </MenuItems>
            </Menu>
          )}
          <span className="text-[11.5px] tabular-nums text-[#75718a]">{visible.length} of {jobs.length} executions</span>
        </div>
      </div>
      {actionNotice && <div className="mb-3 flex shrink-0 items-start gap-3 rounded-[9px] bg-[#3ddc97]/10 px-[13px] py-[10px] text-[12px] font-medium text-[#7ce0b4]"><span className="min-w-0 flex-1">{actionNotice}</span><button type="button" onClick={() => setActionNotice("")} aria-label="Dismiss notice" className="shrink-0 rounded-[6px] p-1 hover:text-white"><X size={13} /></button></div>}
      {actionError && <div className="mb-3 flex shrink-0 items-start gap-3 rounded-[9px] bg-[#f05a5a]/10 px-[13px] py-[10px] text-[12px] font-medium text-[#ff9d9d]"><span className="min-w-0 flex-1">{actionError}</span><button type="button" onClick={() => setActionError("")} aria-label="Dismiss error" className="shrink-0 rounded-[6px] p-1 hover:text-white"><X size={13} /></button></div>}
      <div className="scrollbar-subtle min-h-0 flex-1 overflow-auto">
        <div className="min-w-[940px]">
          <div className="grid min-w-[830px] grid-cols-[28px_92px_150px_210px_100px_minmax(160px,1fr)_78px_86px_64px] gap-4 px-5 pb-2 text-[10.5px] font-bold uppercase tracking-[0.07em] text-[#5f5b74]"><span className="flex items-center"><input type="checkbox" aria-label="Select all executions" checked={allVisibleSelected} ref={(node) => { if (node) node.indeterminate = visible.some((job) => selected.has(job.id)) && !allVisibleSelected; }} onChange={toggleSelectAll} className="h-3.5 w-3.5 accent-[#9d8cff]" /></span><span>Job ID</span><span>Timestamp</span><span>Model</span><span>Status</span><span>Progress</span><span className="text-right">Duration</span><span className="text-right">Started</span><span /></div>
          <div className="flex flex-col gap-2">
            {loading ? <div className="flex items-center gap-2 rounded-[12px] bg-[#12101a] px-5 py-8 text-[12.5px] text-[#75718a]"><LoaderCircle size={15} className="animate-spin text-[#9d8cff]" /> Loading persisted executions…</div>
              : error ? <div className="rounded-[12px] bg-[#f05a5a]/10 px-5 py-4 text-[12.5px] text-[#ff8f8f]">{error}</div>
                : visible.length ? visible.map((job) => <ExecutionRow key={job.id} job={job} modelProfiles={modelProfiles} onCancel={handleCancel} onRerun={handleRerun} busy={busyJobId === job.id || bulkBusy} checked={selected.has(job.id)} onToggleSelect={toggleSelected} expanded={expandedId === job.id} onToggle={() => setExpandedId((current) => current === job.id ? null : job.id)} onPreview={setPreview} />)
                  : <div className="rounded-[6px] border-2 border-[var(--nyx-line)] bg-[var(--nyx-panel-2)] px-5 py-10 text-center"><p className="text-[13px] font-medium text-[var(--nyx-muted)]">No executions match this filter.</p>{filter !== "all" && <button type="button" onClick={() => setFilter("all")} className="mt-2 text-[11px] font-semibold uppercase tracking-[.08em] text-[var(--nyx-accent)]">Show all executions</button>}</div>}
            {nextCursor && <div ref={loadMoreRef} className="py-4 text-center text-[11px] text-[#75718a]">{loadingMore ? "Loading older executions…" : "Scroll for older executions"}</div>}
          </div>
        </div>
      </div>
      <PreviewDialog preview={preview} onClose={() => setPreview(null)} />
      <Dialog open={Boolean(confirmDelete)} onClose={() => setConfirmDelete(null)} className="relative z-[130]">
        <DialogBackdrop className="fixed inset-0 bg-black/70" />
        <div className="fixed inset-0 grid place-items-center p-6">
          <DialogPanel className="w-full max-w-[420px] rounded-[14px] bg-[#181423] p-6 shadow-[0_24px_60px_rgba(0,0,0,0.6)]">
            <DialogTitle className="text-[16px] font-bold text-[#f2f2f5]">Delete {confirmDelete?.length ?? 0} execution record{(confirmDelete?.length ?? 0) === 1 ? "" : "s"}?</DialogTitle>
            <p className="mt-2 text-[12.5px] leading-5 text-[#8b879a]">This removes the job history entries only. The generated images stay in your gallery.</p>
            <div className="mt-5 flex justify-end gap-2">
              <button type="button" onClick={() => setConfirmDelete(null)} className="rounded-[9px] bg-[#221d30] px-4 py-2.5 text-[12px] font-bold text-[#c7c3d4] transition hover:bg-[#2b2440]">Cancel</button>
              <button type="button" onClick={confirmDeleteSelected} className="flex items-center gap-1.5 rounded-[9px] bg-[#7a2230] px-4 py-2.5 text-[12px] font-bold text-white transition hover:bg-[#93293a]"><Trash2 size={13} /> Delete</button>
            </div>
          </DialogPanel>
        </div>
      </Dialog>
    </section>
  );
}
