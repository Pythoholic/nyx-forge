import { Check, Clock3, Layers3, LoaderCircle, Plus, Search, X } from "lucide-react";
import { useCallback, useEffect, useMemo, useState } from "react";

import { getBatch, getBatches } from "./api";
import "./ForgeBAT.css";

const ACTIVE = new Set(["planning", "queued", "running"]);
const STATUS_LABELS = { completed: "Complete", partial: "Finished with errors", failed: "Failed", cancelled: "Cancelled", running: "Generating", queued: "Queued", planning: "Planning" };

function BatchCard({ batch, onOpen }) {
  const detail = batch.detail;
  const thumbnails = detail?.items?.filter((item) => item.thumbnail_url).slice(0, 6) ?? [];
  const status = batch.status;
  const StatusIcon = status === "completed" ? Check : ACTIVE.has(status) ? LoaderCircle : status === "failed" || status === "cancelled" ? X : Clock3;
  const config = batch.config ?? {};
  const model = String(config.model ?? "ForgeBAT model").split("\\").at(-1).replace(/\.safetensors.*$/i, "").replaceAll("_", " ");
  return <button type="button" onClick={onOpen} className="bat-library-card">
    <div className="bat-library-cover">{thumbnails.length ? thumbnails.map((item) => <img key={item.id} src={item.thumbnail_url} alt="" />) : Array.from({ length: 6 }, (_, index) => <span key={index}><Layers3 size={16} /></span>)}</div>
    <div className="bat-library-copy"><h2>Batch {batch.id}</h2><p>{batch.requested_count} images · {model}</p><div className={`bat-library-status is-${status}`}><span><StatusIcon size={12} className={ACTIVE.has(status) ? "animate-spin" : ""} />{STATUS_LABELS[status] ?? status}</span><time>{batch.updated_at ? new Date(batch.updated_at).toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" }) : ""}</time></div>{ACTIVE.has(status) && <div className="bat-library-progress"><i style={{ width: `${batch.progress_percent ?? 0}%` }} /></div>}</div>
  </button>;
}

export default function ForgeBATHome({ onNavigate }) {
  const [batches, setBatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("all");
  const [sort, setSort] = useState("newest");

  const load = useCallback(async () => {
    try {
      const response = await getBatches(100);
      const rows = response.batches ?? [];
      const details = await Promise.all(rows.slice(0, 24).map((batch) => getBatch(batch.id).catch(() => null)));
      setBatches(rows.map((batch, index) => ({ ...batch, detail: details[index] ?? null })));
      setError("");
    } catch (requestError) { setError(requestError.message); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => { load(); }, [load]);
  useEffect(() => { if (!batches.some((batch) => ACTIVE.has(batch.status))) return undefined; const timer = window.setInterval(load, 4000); return () => window.clearInterval(timer); }, [batches, load]);

  const visible = useMemo(() => {
    const needle = search.trim().toLowerCase();
    const filtered = batches.filter((batch) => (status === "all" || (status === "active" ? ACTIVE.has(batch.status) : batch.status === status)) && (!needle || `batch ${batch.id} ${batch.config?.model ?? ""}`.toLowerCase().includes(needle)));
    return [...filtered].sort((left, right) => sort === "oldest" ? left.id - right.id : right.id - left.id);
  }, [batches, search, sort, status]);

  return <main className="bat-library-page"><header className="bat-library-header"><div><h1>Batches</h1><p>{batches.length} saved batch{batches.length === 1 ? "" : "es"}</p></div><div className="bat-library-tools"><label><Search size={14} /><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search batches…" /></label><select value={status} onChange={(event) => setStatus(event.target.value)}><option value="all">All</option><option value="active">Active</option><option value="completed">Complete</option><option value="failed">Failed</option><option value="cancelled">Cancelled</option></select><select value={sort} onChange={(event) => setSort(event.target.value)}><option value="newest">Newest</option><option value="oldest">Oldest</option></select><button type="button" onClick={() => onNavigate("/forge-bat/new")}><Plus size={15} />New batch</button></div></header>
    {error && <p className="bat-error mt-4">{error}</p>}{loading ? <div className="bat-library-empty"><LoaderCircle className="animate-spin" />Loading saved batches…</div> : visible.length ? <section className="bat-library-grid">{visible.map((batch) => <BatchCard key={batch.id} batch={batch} onOpen={() => onNavigate(`/forge-bat/batches/${batch.id}`)} />)}</section> : <div className="bat-library-empty"><Layers3 size={28} /><h2>{batches.length ? "No matching batches" : "Create your first batch"}</h2><p>Define a recipe, preview its deterministic plan, and monitor generation here.</p><button type="button" onClick={() => onNavigate("/forge-bat/new")}><Plus size={14} />New batch</button></div>}
  </main>;
}
