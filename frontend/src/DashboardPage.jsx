import { useEffect, useMemo, useState } from "react";
import { Dialog, DialogBackdrop, DialogPanel, DialogTitle } from "@headlessui/react";
import {
  BarChart3,
  Clock3,
  Database,
  Gauge,
  Image as ImageIcon,
  Info,
  Lightbulb,
  RefreshCw,
  ShieldCheck,
  Sparkles,
  Star,
  TimerReset,
  TriangleAlert,
  WandSparkles,
  X,
  Zap,
} from "lucide-react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { getAnalytics } from "./api";

const ACCENT = "var(--nyx-accent)";
const GRID = "var(--nyx-line)";
const TICK = "var(--nyx-label)";
const WINDOWS = [
  { label: "7 days", value: 7 },
  { label: "30 days", value: 30 },
  { label: "90 days", value: 90 },
  { label: "All time", value: 0 },
];
const TABS = [
  { id: "recommendations", label: "Recommendations", icon: Lightbulb },
  { id: "overview", label: "Overview", icon: BarChart3 },
  { id: "performance", label: "Performance", icon: Gauge },
  { id: "prompts", label: "Prompts & recipes", icon: WandSparkles },
];
const RECOMMENDATION_ICONS = {
  quality: Star,
  speed: Zap,
  prompt: WandSparkles,
  data: Database,
  reliability: ShieldCheck,
};

const seconds = (value) => value == null ? "Not measured" : `${Number(value).toFixed(value >= 100 ? 0 : 1)}s`;
const rating = (value) => value == null ? "Not rated" : `${Number(value).toFixed(2)} / 5`;
const percent = (value) => value == null ? "No attempts" : `${Number(value).toFixed(1)}%`;
const qualityLabel = (value) => ({ standard: "Normal", normal: "Normal", high: "High", super: "Super", "4k": "4K", "8k": "8K", "12k": "12K" }[value] ?? value);

function modelName(value, profiles) {
  if (!value) return "Unknown model";
  if (profiles[value]?.display_name) return profiles[value].display_name;
  return value.split("\\").pop().replace(".safetensors", "").replaceAll("_", " ");
}

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded-[12px] bg-[var(--field)] px-3 py-2 text-xs shadow-[var(--shadow-menu)]">
      <p className="mb-1.5 font-semibold text-zinc-200">{label}</p>
      {payload.map((item) => (
        <p key={item.dataKey} className="mt-1 flex items-center justify-between gap-5 text-zinc-400">
          <span>{item.name}</span>
          <span className="font-mono text-zinc-100">
            {item.dataKey.includes("rating") ? Number(item.value).toFixed(2) : item.dataKey.includes("seconds") ? seconds(item.value) : item.value}
          </span>
        </p>
      ))}
    </div>
  );
}

function MetricCard({ icon: Icon, label, value, detail }) {
  return (
    <article className="min-w-0 rounded-[12px] bg-[var(--panel)] px-[18px] py-4">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate text-[10px] font-semibold uppercase tracking-[.14em] text-zinc-500">{label}</p>
          <p className="mt-1.5 text-2xl font-semibold tracking-tight text-zinc-50">{value}</p>
        </div>
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-[9px] bg-[var(--accent-wash)] text-[var(--accent-hi)]"><Icon size={16} /></span>
      </div>
      <p className="mt-1.5 truncate text-[11px] text-zinc-600" title={detail}>{detail}</p>
    </article>
  );
}

function Panel({ title, subtitle, children, className = "", bodyClassName = "" }) {
  return (
    <section className={`flex min-h-0 flex-col rounded-[14px] bg-[var(--panel)] px-[26px] py-[22px] ${className}`}>
      <div className="mb-4 shrink-0">
        <h2 className="text-sm font-semibold text-zinc-100">{title}</h2>
        <p className="mt-1 text-xs text-zinc-600">{subtitle}</p>
      </div>
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

function DashboardSkeleton() {
  return (
    <div className="flex h-full animate-pulse flex-col gap-4">
      <div className="grid grid-cols-5 gap-3">
        {Array.from({ length: 5 }, (_, index) => <div key={index} className="h-24 rounded-2xl bg-zinc-900" />)}
      </div>
      <div className="min-h-0 flex-1 rounded-2xl bg-zinc-900" />
    </div>
  );
}

function AnalyticsInfoDialog({ open, onClose }) {
  const items = [
    [Clock3, "Speed", "Median is a typical render. P90 is the slower edge: 90% of measured renders finished within that time. Seconds per megapixel makes unlike resolutions comparable."],
    [Star, "Image quality", "Your star ratings are the quality signal. More ratings, including weak results, make comparisons substantially more trustworthy."],
    [WandSparkles, "Prompt intelligence", "This compares which prompt engine led to better-rated images and how long prompt creation took. It does not pretend to judge the artwork automatically."],
    [Database, "Clean baseline", "Only runs created by the current prompt compiler are compared. Older images stay in your gallery, but their longer legacy prompts do not distort the new recommendations."],
    [Lightbulb, "Recommendations", "Advice is derived only from your local runs. A confidence label shows how much repeat evidence exists; it is guidance, not a guarantee."],
    [ShieldCheck, "Privacy", "Generation metadata stays local. API keys and live in-progress previews are not stored in analytics."],
  ];
  return (
    <Dialog open={open} onClose={onClose} className="relative z-[100]">
      <DialogBackdrop className="fixed inset-0 bg-[var(--scrim)]" />
      <div className="fixed inset-0 grid place-items-center overflow-y-auto p-5">
        <DialogPanel className="w-full max-w-[660px] rounded-[18px] bg-[var(--field)] px-7 pb-6 pt-[26px] shadow-[var(--shadow-modal)]">
          <div className="flex items-start justify-between gap-5">
            <div>
              <p className="forge-eyebrow text-[var(--accent-hi)]">Analysis guide</p>
              <DialogTitle className="mt-2 text-xl font-semibold text-zinc-50">What these numbers tell you</DialogTitle>
              <p className="mt-2 text-sm leading-6 text-zinc-500">Use this page to find repeatable settings that balance visual quality, prompt variety, render time, and reliability.</p>
            </div>
            <button type="button" onClick={onClose} className="grid h-[30px] w-[30px] shrink-0 place-items-center rounded-[8px] bg-[var(--field-hover)] text-[var(--text-muted)] transition-colors hover:bg-[var(--raised-hover)] hover:text-[var(--text)]" aria-label="Close guide"><X size={15} /></button>
          </div>
          <div className="mt-6 grid gap-3 sm:grid-cols-2">
            {items.map(([Icon, title, copy]) => (
              <article key={title} className="rounded-2xl bg-[#17181f] p-4">
                <Icon size={16} className="text-[var(--accent-hi)]" />
                <h3 className="mt-3 text-sm font-semibold text-zinc-200">{title}</h3>
                <p className="mt-1.5 text-xs leading-5 text-zinc-500">{copy}</p>
              </article>
            ))}
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}

function RecommendationCard({ item, modelProfiles, className = "" }) {
  const Icon = RECOMMENDATION_ICONS[item.category] ?? Sparkles;
  const settings = Object.entries(item.settings ?? {});
  return (
    <article className={`flex min-w-0 flex-col rounded-[14px] bg-[var(--panel)] px-[26px] py-[22px] ${className}`}>
      <div className="flex items-start justify-between gap-4">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-[9px] bg-[var(--accent-wash)] text-[var(--accent-hi)]"><Icon size={18} /></span>
        <span className={`shrink-0 whitespace-nowrap rounded-[20px] px-[9px] py-[3px] text-[10.5px] font-bold ${item.confidence === "high" ? "bg-[var(--ok-wash)] text-[var(--ok)]" : "bg-[var(--queued-wash)] text-[var(--queued)]"}`}>{item.confidence} confidence</span>
      </div>
      <p className="forge-eyebrow mt-4">{item.category}</p>
      <h3 className="mt-1.5 text-base font-semibold leading-snug text-zinc-100">{item.title}</h3>
      <p className="mt-2 break-words text-xs leading-5 text-zinc-500">{item.summary}</p>
      <div className="mt-4 rounded-xl bg-[#0e0f14] px-3.5 py-3">
        <p className="text-[10px] font-semibold uppercase tracking-[.13em] text-zinc-600">Recommended next move</p>
        <p className="mt-1.5 break-words text-xs leading-5 text-zinc-300">{item.action}</p>
      </div>
      {settings.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {settings.map(([key, value]) => (
            <span key={key} className="max-w-full truncate rounded-lg bg-white/[0.04] px-2 py-1 text-[10px] text-zinc-400" title={String(value)}>
              {key === "model" ? modelName(value, modelProfiles) : key === "quality_mode" ? qualityLabel(value) : String(value)}
            </span>
          ))}
        </div>
      )}
      <div className="mt-auto grid gap-1.5 pt-4 text-[10px] text-zinc-600">
        {(item.evidence ?? []).map((evidence) => <p key={evidence} className="flex min-w-0 items-start gap-2"><span className="shrink-0 text-[var(--accent-hi)]">/</span><span className="min-w-0 break-words">{evidence}</span></p>)}
      </div>
    </article>
  );
}

function recommendationColumnSpan(index, count) {
  if (count === 1) return "2xl:col-span-6";
  if (count === 2 || count === 4) return "2xl:col-span-3";
  if (count === 5 && index >= 3) return "2xl:col-span-3";
  return "2xl:col-span-2";
}

function RecommendationsTab({ data, modelProfiles }) {
  const recommendations = data.recommendations ?? [];
  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="mb-4 flex shrink-0 items-end justify-between gap-5">
        <div><h2 className="text-lg font-semibold text-zinc-100">Your next best moves</h2><p className="mt-1 text-xs text-zinc-500">Ranked from repeat local evidence, with small samples clearly marked.</p></div>
        <p className="hidden text-right text-[11px] leading-5 text-zinc-600 lg:block">Ratings teach quality. Timing teaches speed.<br />More repeated runs increase confidence.</p>
      </div>
      {recommendations.length ? (
        <div className="grid min-h-0 flex-1 content-start items-stretch gap-4 overflow-y-auto p-px lg:grid-cols-2 2xl:grid-cols-6">
          {recommendations.map((item, index) => (
            <RecommendationCard
              key={item.id}
              item={item}
              modelProfiles={modelProfiles}
              className={`min-h-[300px] ${recommendationColumnSpan(index, recommendations.length)}`}
            />
          ))}
        </div>
      ) : (
        <div className="grid min-h-0 flex-1 place-items-center rounded-[14px] bg-[var(--well)]">
          <div className="max-w-md text-center"><Lightbulb className="mx-auto text-zinc-700" size={30} /><h3 className="mt-3 font-semibold">Still learning your preferences</h3><p className="mt-1 text-sm leading-6 text-zinc-500">Generate and rate a few repeated combinations. Recommendations appear once there is enough evidence to avoid guessing.</p></div>
        </div>
      )}
    </div>
  );
}

function OverviewTab({ data, summary }) {
  return (
    <div className="flex h-full min-h-0 flex-col gap-4">
      <section className="grid shrink-0 grid-cols-2 gap-3 lg:grid-cols-3 xl:grid-cols-5">
        <MetricCard icon={ImageIcon} label="Generations" value={summary.total_generations} detail={`${summary.measured_generations} include render timing`} />
        <MetricCard icon={Clock3} label="Median render" value={seconds(summary.median_seconds)} detail={`Average ${seconds(summary.average_seconds)}`} />
        <MetricCard icon={TimerReset} label="P90 render" value={seconds(summary.p90_seconds)} detail="90% completed within this time" />
        <MetricCard icon={Star} label="Average rating" value={rating(summary.average_rating)} detail={`${summary.rated_count} of ${summary.total_generations} rated`} />
        <MetricCard icon={Gauge} label="Success rate" value={percent(summary.success_rate)} detail={`${summary.failed_attempts} failed terminal attempts`} />
      </section>
      <section className="grid min-h-0 flex-1 gap-4 xl:grid-cols-3">
        <Panel title="Generation trend" subtitle="Daily volume and measured average render time" className="xl:col-span-2" bodyClassName="aiqg-chart min-h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={data.timeline} margin={{ top: 8, right: 8, bottom: 0, left: -16 }}>
              <defs><linearGradient id="durationFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={ACCENT} stopOpacity={0.28} /><stop offset="100%" stopColor={ACCENT} stopOpacity={0} /></linearGradient></defs>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="date" tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={false} minTickGap={30} />
              <YAxis yAxisId="time" tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}s`} />
              <YAxis yAxisId="count" orientation="right" allowDecimals={false} tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip content={<ChartTooltip />} cursor={{ stroke: ACCENT, strokeOpacity: 0.22, strokeWidth: 1 }} />
              <Area yAxisId="time" type="monotone" dataKey="average_seconds" name="Average render" stroke={ACCENT} strokeWidth={2} fill="url(#durationFill)" connectNulls />
              <Bar yAxisId="count" dataKey="count" name="Generations" fill="rgba(255,255,255,.12)" radius={[4, 4, 0, 0]} barSize={16} />
            </ComposedChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Ratings" subtitle="Quality feedback distribution" bodyClassName="aiqg-chart min-h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data.rating_distribution} margin={{ top: 8, right: 4, bottom: 0, left: -24 }}>
              <CartesianGrid stroke={GRID} vertical={false} />
              <XAxis dataKey="score" tick={{ fill: TICK, fontSize: 11 }} tickFormatter={(value) => `${value} star`} tickLine={false} axisLine={false} />
              <YAxis allowDecimals={false} tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255, 77, 77, 0.045)" }} />
              <Bar dataKey="count" name="Images" radius={[6, 6, 0, 0]} activeBar={{ fill: ACCENT, fillOpacity: 0.9 }}>{data.rating_distribution.map((item) => <Cell key={item.score} fill={ACCENT} fillOpacity={0.25 + item.score * 0.13} />)}</Bar>
            </BarChart>
          </ResponsiveContainer>
        </Panel>
      </section>
    </div>
  );
}

function PerformanceTab({ qualityData, modelData }) {
  return (
    <div className="grid h-full min-h-0 gap-4 xl:grid-cols-[0.9fr_1.35fr]">
      <Panel title="Output-quality cost" subtitle="Render time and throughput by Normal, High, and Super" bodyClassName="flex flex-col">
        <div className="aiqg-chart min-h-[260px] flex-1">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={qualityData} layout="vertical" margin={{ top: 4, right: 16, bottom: 0, left: 4 }}>
              <CartesianGrid stroke={GRID} horizontal={false} />
              <XAxis type="number" tick={{ fill: TICK, fontSize: 11 }} tickLine={false} axisLine={false} tickFormatter={(value) => `${value}s`} />
              <YAxis type="category" dataKey="label" width={62} tick={{ fill: "#a1a1aa", fontSize: 11 }} tickLine={false} axisLine={false} />
              <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255, 77, 77, 0.045)" }} />
              <Bar dataKey="median_seconds" name="Median render" fill={ACCENT} radius={[0, 6, 6, 0]} barSize={18} activeBar={{ fill: ACCENT, fillOpacity: 0.9 }} />
              <Bar dataKey="seconds_per_megapixel" name="Seconds / MP" fill="rgba(255,255,255,.16)" radius={[0, 6, 6, 0]} barSize={18} activeBar={{ fill: "rgba(255,255,255,.28)" }} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="mt-3 grid shrink-0 gap-2 sm:grid-cols-3">
          {qualityData.map((item) => (
            <div key={item.label} className="rounded-lg bg-[#17181f] px-3 py-2 text-[10px] text-zinc-500">
              <span className="font-semibold text-zinc-300">{item.label}</span>
              <span className="mt-1 block font-mono">{item.average_diffusion_seconds == null ? "Phase timing starts with new runs" : `Diffuse ${seconds(item.average_diffusion_seconds)} / upscale ${seconds(item.average_upscale_seconds)}`}</span>
            </div>
          ))}
        </div>
      </Panel>
      <Panel title="Model performance" subtitle="Read sample count beside speed and rating; one run is not a reliable winner" bodyClassName="overflow-auto">
        <table className="w-full min-w-[760px] text-left text-xs">
          <thead className="sticky top-0 bg-[#111218] text-[9px] uppercase tracking-[.12em] text-zinc-600"><tr><th className="pb-3 font-semibold">Model</th><th className="pb-3 font-semibold">Runs</th><th className="pb-3 font-semibold">Median</th><th className="pb-3 font-semibold">P90</th><th className="pb-3 font-semibold">Seconds / MP</th><th className="pb-3 font-semibold">Rating</th><th className="pb-3 font-semibold">Rated</th></tr></thead>
          <tbody className="divide-y divide-white/[0.05]">{modelData.map((item) => <tr key={item.label}><td className="max-w-[300px] py-3 pr-4 font-medium text-zinc-200"><span className="block truncate">{item.label}</span></td><td className="py-3 font-mono text-zinc-400">{item.count}</td><td className="py-3 font-mono text-zinc-400">{seconds(item.median_seconds)}</td><td className="py-3 font-mono text-zinc-400">{seconds(item.p90_seconds)}</td><td className="py-3 font-mono text-zinc-400">{seconds(item.seconds_per_megapixel)}</td><td className="py-3 font-mono text-zinc-400">{item.average_rating == null ? "--" : Number(item.average_rating).toFixed(2)}</td><td className="py-3 font-mono text-zinc-500">{item.rated_count}/{item.count}</td></tr>)}</tbody>
        </table>
      </Panel>
    </div>
  );
}

function PromptsTab({ data, modelProfiles }) {
  const topIssue = data.quality_signals?.issues?.[0];
  const topStrength = data.quality_signals?.strengths?.[0];
  const reasonLabel = (value) => value ? value.replaceAll("_", " ") : "Still learning";
  return (
    <div className="grid h-full min-h-0 gap-4 xl:grid-cols-[0.8fr_1.4fr]">
      <Panel title="Prompt intelligence" subtitle="Outcome and prompt-phase coverage by engine" bodyClassName="flex flex-col">
        <div className="min-h-0 flex-1 space-y-2 overflow-auto pr-1">
          {data.by_prompt_engine.map((item) => (
            <div key={item.label} className="flex items-center justify-between gap-5 rounded-xl bg-[#17181f] px-4 py-3">
              <div className="min-w-0"><p className="truncate text-sm font-medium text-zinc-200">{item.label}</p><p className="mt-1 text-[11px] text-zinc-600">{item.count} generations / {item.rated_count} rated</p></div>
              <div className="shrink-0 text-right"><p className="font-mono text-sm text-zinc-200">{item.average_rating == null ? "--" : `${Number(item.average_rating).toFixed(2)} / 5`}</p><p className="mt-1 text-[10px] text-zinc-600">{seconds(item.average_prompt_seconds)} prompt · {item.duplicate_risk_count ?? 0} similar</p></div>
            </div>
          ))}
        </div>
        <div className="mt-3 flex shrink-0 items-start gap-2 rounded-[12px] bg-[var(--accent-wash)] px-3 py-2 text-[11px] leading-5 text-[var(--text-muted)]"><Sparkles className="mt-0.5 shrink-0 text-[var(--accent-hi)]" size={14} /><span>Older runs without prompt timing remain useful for image-quality comparisons and are clearly marked.</span></div>
        <div className="mt-3 grid shrink-0 grid-cols-3 gap-2 text-[10px]">
          <div className="rounded-xl bg-[#17181f] p-3"><p className="text-zinc-600">Most reported issue</p><p className="mt-1 capitalize text-zinc-300">{reasonLabel(topIssue?.reason)}{topIssue ? ` · ${topIssue.count}` : ""}</p></div>
          <div className="rounded-xl bg-[#17181f] p-3"><p className="text-zinc-600">Strongest signal</p><p className="mt-1 capitalize text-zinc-300">{reasonLabel(topStrength?.reason)}{topStrength ? ` · ${topStrength.count}` : ""}</p></div>
          <div className="rounded-xl bg-[#17181f] p-3"><p className="text-zinc-600">Near-duplicate risk</p><p className="mt-1 font-mono text-zinc-300">{data.summary.duplicate_risk_count ?? 0} / {data.summary.similarity_count ?? 0}</p></div>
        </div>
      </Panel>
      <Panel title="Top parameter recipes" subtitle="Repeatable combinations ranked by rated evidence, then quality and sample count" bodyClassName="overflow-auto pr-1">
        <div className="grid gap-3 lg:grid-cols-2 2xl:grid-cols-3">
          {data.top_combinations.map((item, index) => (
            <article key={`${item.model}-${item.style}-${item.content_rating}-${item.quality_mode}-${item.aspect_ratio}-${index}`} className="rounded-xl bg-[#17181f] p-4">
              <div className="flex items-start justify-between gap-3"><div className="min-w-0"><p className="truncate text-sm font-semibold text-zinc-200">{modelName(item.model, modelProfiles)}</p><p className="mt-1 truncate text-[11px] text-zinc-500">{item.style} · {item.content_rating} · {qualityLabel(item.quality_mode)} · {item.aspect_ratio}</p></div><span className="shrink-0 rounded-[20px] bg-[var(--accent-wash)] px-[9px] py-[3px] font-mono text-[10.5px] text-[var(--accent-text)]">{item.count} runs</span></div>
              <div className="mt-4 grid grid-cols-3 gap-2 text-[11px]"><div><p className="text-zinc-600">Median</p><p className="mt-1 font-mono text-zinc-300">{seconds(item.median_seconds)}</p></div><div><p className="text-zinc-600">Rating</p><p className="mt-1 font-mono text-zinc-300">{item.average_rating == null ? "--" : Number(item.average_rating).toFixed(2)}</p></div><div><p className="text-zinc-600">Rated</p><p className="mt-1 font-mono text-zinc-300">{item.rated_count}/{item.count}</p></div></div>
              <p className="mt-3 truncate text-[10px] text-zinc-600" title={`${item.prompt_engine} / ${item.sampler_name} / ${item.scheduler}`}>{item.prompt_engine} / {item.sampler_name} / {item.scheduler} / {item.steps} steps / CFG {item.cfg_scale}</p>
            </article>
          ))}
        </div>
      </Panel>
    </div>
  );
}

export default function DashboardPage({ modelProfiles }) {
  const [days, setDays] = useState(30);
  const [activeTab, setActiveTab] = useState("recommendations");
  const [infoOpen, setInfoOpen] = useState(false);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      setData(await getAnalytics(days));
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, [days]); // eslint-disable-line react-hooks/exhaustive-deps

  const qualityData = useMemo(() => (data?.by_quality ?? []).map((item) => ({ ...item, label: qualityLabel(item.label) })), [data]);
  const modelData = useMemo(() => (data?.by_model ?? []).map((item) => ({ ...item, label: modelName(item.label, modelProfiles) })), [data, modelProfiles]);
  const summary = data?.summary;

  return (
    <main className="nyx-analytics-page forge-page flex h-[calc(100vh-88px)] min-h-[700px] flex-col overflow-hidden">
      <section className="flex min-h-0 flex-1 flex-col px-8 pb-[22px]">
        <div className="shrink-0 py-[22px]">
          <div className="mb-[18px] flex items-start justify-between gap-[18px]">
            <div className="flex flex-col gap-[6px]"><h1 className="forge-display">Analysis</h1><p className="forge-body">Turn local generation history into repeatable quality and performance decisions.</p></div>
            <div className="flex items-center gap-[10px]">
              <button type="button" onClick={() => setInfoOpen(true)} className="forge-btn flex items-center gap-2"><Info size={15} /><span className="hidden sm:inline">Read the metrics</span></button>
              <button type="button" onClick={load} disabled={loading} className="grid h-8 w-8 place-items-center rounded-[8px] bg-[var(--field)] text-[var(--text-muted)] transition-colors hover:bg-[var(--field-hover)] hover:text-[var(--text)] disabled:text-[var(--text-disabled)]" aria-label="Refresh analytics"><RefreshCw size={14} /></button>
            </div>
          </div>
          <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="forge-eyebrow">Local telemetry</p>
              <p className="mt-[6px] text-[12px] font-medium leading-[1.45] text-[var(--text-subtle)]">
                {summary
                  ? `Current prompt compiler: ${summary.compiler_version} · ${summary.total_generations} comparable runs${summary.legacy_generations_excluded ? ` · ${summary.legacy_generations_excluded} legacy runs preserved but excluded` : ""}`
                  : "Turn your own speed, rating, and reliability history into better generation choices."}
              </p>
            </div>
            <div className="forge-seg w-fit">
              {WINDOWS.map((item) => <button key={item.value} type="button" onClick={() => setDays(item.value)} aria-selected={days === item.value}>{item.label}</button>)}
            </div>
          </div>
          <nav className="forge-seg mt-[18px] w-fit max-w-full overflow-x-auto" aria-label="Analytics views">
            {TABS.map(({ id, label, icon: Icon }) => (
              <button key={id} type="button" onClick={() => setActiveTab(id)} className="flex shrink-0 items-center gap-2 px-[14px]" aria-selected={activeTab === id}>
                <Icon size={14} />{label}
              </button>
            ))}
          </nav>
        </div>

        <div className="min-h-0 flex-1">
          {error ? (
            <div className="grid h-full place-items-center rounded-[14px] bg-[var(--danger-wash)]"><div className="text-center"><TriangleAlert className="mx-auto text-[var(--danger)]" /><h2 className="mt-3 font-semibold">Analytics could not be loaded</h2><p className="mt-1 text-sm text-zinc-500">{error}</p></div></div>
          ) : loading ? <DashboardSkeleton /> : !summary?.total_generations ? (
            <div className="grid h-full place-items-center rounded-[14px] bg-[var(--well)]"><div className="text-center"><h2 className="text-[14.5px] font-bold text-[var(--text-muted)]">No telemetry in this period</h2><p className="mt-2 text-[12.5px] font-semibold text-[var(--accent-hi)]">Generate an image or choose a wider time window.</p></div></div>
          ) : activeTab === "recommendations" ? <RecommendationsTab data={data} modelProfiles={modelProfiles} />
            : activeTab === "overview" ? <OverviewTab data={data} summary={summary} />
              : activeTab === "performance" ? <PerformanceTab qualityData={qualityData} modelData={modelData} />
                : <PromptsTab data={data} modelProfiles={modelProfiles} />}
        </div>
      </section>
      <AnalyticsInfoDialog open={infoOpen} onClose={() => setInfoOpen(false)} />
    </main>
  );
}
