import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from "@headlessui/react";
import { Clock3, Layers3, LoaderCircle, Sparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { cancelBatch, createBatch, getBatch, getBatchOptions, previewBatch, retryBatchFailures } from "./api";
import { BatchCanvas, BatchCoverage, BatchExecutionSummary, BatchFacetGrid, BatchModelControl, BatchOutputControl, BatchSizeControl, BatchStyleControl, BatchSummary } from "./ForgeBATWorkspace";
import "./ForgeBAT.css";

const QUALITY_LABELS = { normal: "Normal", high: "High", super: "Super", "4k": "4K", "8k": "8K", "12k": "12K" };

function formatDuration(imageCount) {
  const minutes = Math.max(1, Math.round((imageCount * 72) / 60));
  if (minutes < 60) return `~${minutes} minute${minutes === 1 ? "" : "s"}`;
  const hours = Math.floor(minutes / 60);
  const remainder = minutes % 60;
  return `~${hours}h${remainder ? ` ${remainder}m` : ""}`;
}

function selectionIssueFor(requestError, selections, facets, orientations) {
  if (requestError.detail?.reason !== "BUDGET_UNRECOVERABLE") return null;
  const selectedKeys = Object.entries(selections)
    .filter(([, selection]) => selection.mode === "selected" && selection.ids.length)
    .map(([key]) => key);
  const preferred = ["camera", "expression", "mood", "face_features", "lighting", "clothing", "body_type", "heritage"];
  const suggestions = preferred.filter((key) => selectedKeys.includes(key)).slice(0, 4);
  const labelByKey = Object.fromEntries(facets.map((facet) => [facet.key, facet.label]));
  const suggestionLabels = suggestions.map((key) => labelByKey[key] ?? key);
  return {
    facets: selectedKeys,
    message: `The selected controls cannot all fit within this model's prompt budget.${orientations.length > 1 ? " Try fewer orientations" : ""}${suggestionLabels.length ? `${orientations.length > 1 ? " or set" : " Set"} one selected facet to Any: ${suggestionLabels.join(", ")}.` : "."}`,
  };
}

function stylesForModel(values, model, profiles) {
  const profile = profiles[model];
  const supported = profile?.styles ?? [];
  const uniqueValues = [...new Set(values)];
  if (!supported.length) return uniqueValues.length ? uniqueValues : ["Photoreal"];
  const compatible = uniqueValues.filter((value) => supported.includes(value));
  return compatible.length ? compatible : [profile.default_style ?? supported[0]];
}

export default function ForgeBATPage({ models = [], modelProfiles = {}, onNavigate, batchId = null }) {
  const [count, setCount] = useState(16);
  const [customCount, setCustomCount] = useState("");
  const [model, setModel] = useState(models[0] ?? "");
  const [styles, setStyles] = useState(() => stylesForModel([], models[0] ?? "", modelProfiles));
  const [rating, setRating] = useState("Safe");
  const [orientations, setOrientations] = useState(["front"]);
  const [qualityMode, setQualityMode] = useState("high");
  const [distribution, setDistribution] = useState("balanced");
  const [batchSeed, setBatchSeed] = useState(() => Math.floor(Math.random() * 2 ** 31));
  const [requestId, setRequestId] = useState(() => crypto.randomUUID());
  const [facets, setFacets] = useState([]);
  const [selections, setSelections] = useState({});
  const [labels, setLabels] = useState({});
  const [error, setError] = useState("");
  const [busy, setBusy] = useState("");
  const [preview, setPreview] = useState(null);
  const [previewKey, setPreviewKey] = useState("");
  const [batch, setBatch] = useState(null);
  const [unsupportedIds, setUnsupportedIds] = useState([]);
  const [selectionIssue, setSelectionIssue] = useState(null);
  const [confirmationOpen, setConfirmationOpen] = useState(false);

  useEffect(() => {
    if (!batchId) return undefined;
    let active = true;
    setBusy("load");
    getBatch(batchId).then((saved) => {
      if (!active) return;
      const config = saved.config ?? {};
      const savedModel = config.model ?? "";
      setCount(config.count ?? saved.requested_count ?? 16);
      setModel(savedModel);
      setStyles(config.styles?.length ? config.styles : [config.style ?? "Photoreal"]);
      setRating("Safe");
      setOrientations(config.orientations?.length ? config.orientations : [config.orientation ?? "front"]);
      setQualityMode(config.quality_mode ?? "high");
      setDistribution(config.distribution ?? "balanced");
      setBatchSeed(config.batch_seed ?? saved.planner_seed);
      setRequestId(config.request_id ?? saved.request_id);
      setSelections(config.facets ?? {});
      setBatch(saved);
      setError("");
    }).catch((requestError) => active && setError(requestError.message)).finally(() => active && setBusy(""));
    return () => { active = false; };
  }, [batchId]);

  useEffect(() => {
    if (!model && models.length) {
      setModel(models[0]);
      setStyles((current) => stylesForModel(current, models[0], modelProfiles));
    }
  }, [model, modelProfiles, models]);
  useEffect(() => {
    setStyles((current) => {
      const next = stylesForModel(current, model, modelProfiles);
      return next.length === current.length && next.every((value, index) => value === current[index]) ? current : next;
    });
  }, [model, modelProfiles]);

  const context = useMemo(() => ({ model: model || undefined, content_rating: rating, style: styles[0], orientations: orientations.join(",") }), [model, orientations, rating, styles]);
  useEffect(() => {
    let active = true;
    getBatchOptions(context).then((response) => {
      if (!active) return;
      setFacets(response.facets);
      setSelections((current) => Object.fromEntries(response.facets.map((facet) => [facet.key, current[facet.key] ?? { mode: "all", ids: [] }])));
      setError("");
    }).catch((requestError) => active && setError(requestError.message));
    return () => { active = false; };
  }, [context]);

  useEffect(() => {
    const selectedFacets = Object.entries(selections).filter(([, selection]) => selection.mode === "selected" && selection.ids.length);
    if (!selectedFacets.length) { setUnsupportedIds([]); return undefined; }
    let active = true;
    Promise.all(selectedFacets.map(async ([facet, selection]) => {
      const response = await getBatchOptions({ ...context, facet });
      const available = new Set(response.facets.find((item) => item.key === facet)?.options.map((option) => option.id) ?? []);
      return selection.ids.filter((id) => !available.has(id));
    })).then((values) => { if (active) setUnsupportedIds(values.flat()); }).catch(() => {});
    return () => { active = false; };
  }, [context, selections]);

  const rememberLabels = useMemo(() => (options) => setLabels((current) => ({ ...current, ...Object.fromEntries(options.map((option) => [option.id, option.label])) })), []);
  const profile = modelProfiles[model];
  const modelLabel = profile?.display_name ?? model;
  const aspectRatio = profile?.aspect_ratios?.[0] ?? "Portrait (832x1216)";
  const hasEmptySelection = Object.values(selections).some((selection) => selection.mode === "selected" && !selection.ids.length);
  const payload = useMemo(() => ({ count, model, style: styles[0], styles, content_rating: rating, quality_mode: qualityMode, aspect_ratio: aspectRatio, orientation: orientations[0], orientations, distribution, batch_seed: Number(batchSeed), request_id: requestId, facets: selections }), [aspectRatio, batchSeed, count, distribution, model, orientations, qualityMode, rating, requestId, selections, styles]);
  const payloadKey = useMemo(() => JSON.stringify(payload), [payload]);
  useEffect(() => { setSelectionIssue(null); }, [payloadKey]);
  const activePreview = previewKey === payloadKey ? preview : null;
  const planLabels = useMemo(() => {
    const coverageLabels = Object.values(activePreview?.coverage ?? {}).flat();
    return { ...labels, ...Object.fromEntries(coverageLabels.map((value) => [value.id, value.label])) };
  }, [activePreview, labels]);
  const liveCoverage = useMemo(() => {
    if (activePreview?.coverage) return activePreview.coverage;
    const result = {};
    for (const item of batch?.items ?? []) {
      for (const [facet, id] of Object.entries(item.resolved_facets ?? {})) {
        const values = result[facet] ?? (result[facet] = []);
        const existing = values.find((value) => value.id === id);
        if (existing) existing.count += 1;
        else values.push({ id, label: labels[id] ?? undefined, count: 1 });
      }
    }
    return result;
  }, [activePreview, batch, labels]);

  const runPreview = async () => {
    setBusy("preview"); setError(""); setSelectionIssue(null);
    try { const value = await previewBatch(payload); setPreview(value); setPreviewKey(payloadKey); }
    catch (requestError) {
      const issue = selectionIssueFor(requestError, selections, facets, orientations);
      if (issue) setSelectionIssue(issue); else setError(requestError.message);
    }
    finally { setBusy(""); }
  };
  const runBatch = async () => {
    setConfirmationOpen(false); setBusy("create"); setError("");
    try { const created = await createBatch(payload); setBatch(await getBatch(created.batch_id)); onNavigate?.(`/forge-bat/batches/${created.batch_id}`); }
    catch (requestError) { setError(requestError.message); }
    finally { setBusy(""); }
  };
  const cancelCurrentBatch = async () => { setBusy("cancel"); setError(""); try { setBatch(await cancelBatch(batch.id)); } catch (requestError) { setError(requestError.message); } finally { setBusy(""); } };
  const retryFailures = async () => { setBusy("retry"); setError(""); try { setBatch(await retryBatchFailures(batch.id)); } catch (requestError) { setError(requestError.message); } finally { setBusy(""); } };
  const shufflePlan = () => { setBatchSeed(Math.floor(Math.random() * 2 ** 31)); setRequestId(crypto.randomUUID()); setPreview(null); setPreviewKey(""); };
  const clearFacets = () => setSelections((current) => Object.fromEntries(Object.keys(current).map((key) => [key, { mode: "all", ids: [] }])));
  const duplicateRecipe = () => {
    const config = batch?.config ?? payload;
    const duplicateModel = config.model ?? model;
    setCount(config.count ?? count); setModel(duplicateModel); setStyles(stylesForModel(config.styles?.length ? config.styles : [config.style ?? "Photoreal"], duplicateModel, modelProfiles)); setRating(config.content_rating ?? rating); setOrientations(config.orientations?.length ? config.orientations : [config.orientation ?? orientations[0]]); setQualityMode(config.quality_mode ?? qualityMode); setDistribution(config.distribution ?? distribution); setSelections(config.facets ?? selections);
    shufflePlan(); setBatch(null); setError(""); onNavigate?.("/forge-bat/new");
  };

  useEffect(() => {
    if (!batch?.id || ["completed", "partial", "failed", "cancelled"].includes(batch.status)) return undefined;
    let active = true;
    const timer = window.setTimeout(() => getBatch(batch.id).then((value) => active && setBatch(value)).catch((requestError) => active && setError(requestError.message)), 2000);
    return () => { active = false; window.clearTimeout(timer); };
  }, [batch]);

  const validationError = hasEmptySelection ? "Choose at least one value in every facet set to Selected." : unsupportedIds.length ? `${unsupportedIds.length} selected choice${unsupportedIds.length === 1 ? " is" : "s are"} incompatible with the current setup.` : "";
  const canSubmit = Boolean(model) && !batch && !hasEmptySelection && !unsupportedIds.length && !selectionIssue && !busy;
  const selectModel = (nextModel) => {
    setModel(nextModel);
    setStyles((current) => stylesForModel(current, nextModel, modelProfiles));
  };
  const executionView = Boolean(batchId || batch);

  return (
    <>
      <main className="bat-page min-w-0 px-4 pb-7 pt-5 sm:px-6 2xl:px-8">
        {executionView && !batch ? (
          <section className="grid min-h-[55vh] place-items-center rounded-[14px] bg-[var(--well)] ring-1 ring-inset ring-[var(--divider)]">
            {error ? <div className="max-w-md px-6 text-center"><Layers3 className="mx-auto text-[var(--danger)]" /><h1 className="mt-4 text-[18px] font-extrabold text-[var(--text)]">Could not load this batch</h1><p className="mt-2 text-[14px] leading-6 text-[var(--text-muted)]">{error}</p><button type="button" onClick={() => onNavigate?.("/forge-bat")} className="bat-small-button mx-auto mt-5">Return to batches</button></div> : <div className="text-center"><LoaderCircle className="mx-auto animate-spin text-[var(--accent-text)]" /><h1 className="mt-4 text-[18px] font-extrabold text-[var(--text)]">Loading batch execution</h1><p className="mt-2 text-[14px] text-[var(--text-muted)]">Restoring the canvas and current queue state…</p></div>}
          </section>
        ) : executionView ? (
          <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(350px,0.25fr)]">
            <section className="min-w-0 space-y-3">
              <header className="flex flex-col gap-3 px-1 pb-1 sm:flex-row sm:items-end sm:justify-between"><div><p className="bat-kicker">{batch.status === "completed" ? "ForgeBAT result" : "ForgeBAT processing"}</p><h1 className="mt-1 text-[22px] font-extrabold text-[var(--text)]">Batch {batch.id}</h1><p className="mt-1 text-[14px] text-[var(--text-muted)]">{batch.status === "completed" ? `${batch.completed_count} images generated successfully. Review the contact sheet and coverage below.` : "Watch completed images arrive and verify coverage while the queue runs."}</p></div><span className={`w-fit rounded-full px-3 py-1.5 text-[12px] font-bold capitalize ${batch.status === "completed" ? "bg-[var(--ok-wash)] text-[var(--ok)]" : "bg-[var(--accent-wash)] text-[var(--accent-text)]"}`}>{batch.status}</span></header>
              <BatchCanvas count={batch.requested_count} distribution={batch.config?.distribution ?? distribution} batch={batch} labels={planLabels} onShuffle={shufflePlan} rating={batch.config?.content_rating ?? rating} limit={12} />
              <BatchCoverage coverage={liveCoverage} execution />
            </section>
            <BatchExecutionSummary batch={batch} modelLabel={modelProfiles[batch.config?.model]?.display_name ?? batch.config?.model ?? modelLabel ?? "Unavailable"} styles={batch.config?.styles?.length ? batch.config.styles : [batch.config?.style ?? styles[0]]} qualityLabel={QUALITY_LABELS[batch.config?.quality_mode] ?? QUALITY_LABELS[qualityMode]} rating={batch.config?.content_rating ?? rating} busy={busy} error={error} onCancel={cancelCurrentBatch} onRetry={retryFailures} onGallery={() => onNavigate?.(`/forge-ai/gallery?batch=${batch.id}`)} onDuplicate={duplicateRecipe} onBatches={() => onNavigate?.("/forge-bat")} />
          </div>
        ) : (
        <div className="grid min-w-0 gap-5 xl:grid-cols-[minmax(0,1fr)_minmax(350px,0.25fr)]">
          <section className="min-w-0 space-y-3">
            <header className="px-1 pb-1"><h1 className="text-[20px] font-extrabold text-[var(--text)]">Batch studio</h1><p className="mt-1 text-[14px] text-[var(--text-muted)]">Choose the space you want Forge to explore.</p></header>
            <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-[1.05fr_1fr_1.3fr]">
              <BatchSizeControl count={count} customCount={customCount} setCount={setCount} setCustomCount={setCustomCount} />
              <div className="space-y-3">
                <BatchModelControl model={model} models={models} profiles={modelProfiles} setModel={selectModel} />
                <BatchStyleControl styles={styles} styleOptions={profile?.styles ?? []} setStyles={setStyles} />
              </div>
              <div className="md:col-span-2 xl:col-span-1"><BatchOutputControl rating={rating} orientations={orientations} qualityMode={qualityMode} distribution={distribution} setRating={setRating} setOrientations={setOrientations} setQualityMode={setQualityMode} setDistribution={setDistribution} /></div>
            </div>
            <BatchFacetGrid facets={facets} context={context} styles={styles} selections={selections} setSelections={setSelections} labels={labels} rememberLabels={rememberLabels} onClear={clearFacets} invalidFacets={selectionIssue?.facets} validationMessage={selectionIssue?.message} />
            <details className="rounded-[10px] bg-[var(--field)] px-4 py-3 text-[14px] text-[var(--text-muted)] ring-1 ring-inset ring-[var(--divider)]"><summary className="cursor-pointer font-semibold">Advanced settings</summary><label className="mt-2 block max-w-xs">Batch seed<input type="number" min="0" value={batchSeed} onChange={(event) => setBatchSeed(event.target.value)} className="bat-input mt-1" /></label></details>
            <BatchCanvas count={count} distribution={distribution} preview={activePreview} batch={batch} labels={planLabels} onShuffle={shufflePlan} rating={rating} />
            <BatchCoverage coverage={liveCoverage} />
          </section>
          <BatchSummary count={count} modelLabel={modelLabel || "Unavailable"} styles={styles} qualityLabel={QUALITY_LABELS[qualityMode]} rating={rating} estimate={formatDuration(count)} distribution={distribution} facets={facets} selections={selections} batch={batch} busy={busy} canSubmit={canSubmit} error={selectionIssue?.message || validationError || error} warnings={activePreview?.warnings} onPreview={runPreview} onGenerate={() => setConfirmationOpen(true)} onCancel={cancelCurrentBatch} onRetry={retryFailures} onGallery={() => onNavigate?.(`/forge-ai/gallery?batch=${batch.id}`)} onDuplicate={duplicateRecipe} />
        </div>
        )}
      </main>

      <Dialog open={confirmationOpen} onClose={busy ? () => {} : () => setConfirmationOpen(false)} className="relative z-[130]">
        <DialogBackdrop className="fixed inset-0 bg-[var(--scrim)] backdrop-blur-[2px]" />
        <div className="fixed inset-0 overflow-y-auto p-4 sm:p-8"><div className="flex min-h-full items-center justify-center"><DialogPanel className="w-full max-w-[560px] rounded-[18px] bg-[var(--field)] p-6 shadow-[var(--shadow-modal)] ring-1 ring-inset ring-[var(--divider)]">
          <p className="bat-kicker">Confirm GPU request</p><DialogTitle className="mt-2 text-[18px] font-extrabold text-[var(--text)]">Generate this {count}-image batch?</DialogTitle><p className="mt-2 text-[14px] leading-5 text-[var(--text-muted)]">Forge will persist this exact recipe and begin queueing its planned images.</p>
          <dl className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">{[["Images", count], ["Model", modelLabel || "Unavailable"], ["Quality", QUALITY_LABELS[qualityMode]], ["Rating", rating]].map(([term, value]) => <div key={term} className="rounded-[9px] bg-[var(--well)] p-3"><dt className="text-[14px] font-bold uppercase text-[var(--text-faint)]">{term}</dt><dd className="mt-1 break-words text-[14px] font-semibold text-[var(--text)]">{value}</dd></div>)}</dl>
          <div className="mt-4 flex gap-3 rounded-[10px] bg-[var(--accent-wash)] p-3"><Clock3 size={15} className="mt-0.5 shrink-0 text-[var(--accent-text)]" /><div><p className="text-[14px] font-bold text-[var(--text)]">Expected duration: {formatDuration(count)}</p><p className="mt-1 text-[13.5px] leading-4 text-[var(--text-muted)]">Estimate uses the existing measured rate; queue load and quality mode can change the actual time.</p></div></div>
          <div className="mt-5 flex justify-end gap-2"><button type="button" onClick={() => setConfirmationOpen(false)} className="rounded-[9px] bg-[var(--well)] px-4 py-2.5 text-[14px] font-bold text-[var(--text-muted)]">Cancel</button><button type="button" onClick={runBatch} className="flex items-center gap-2 rounded-[9px] bg-[var(--accent)] px-4 py-2.5 text-[14px] font-bold text-[var(--on-accent)]"><Sparkles size={13} />Confirm and generate</button></div>
        </DialogPanel></div></div>
      </Dialog>
    </>
  );
}
