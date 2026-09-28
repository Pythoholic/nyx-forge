import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { getBatches, getJobs } from "./api";
import { displayJobId, workspaceForJobKind, workspaceLabel } from "./jobLabels";

const ACTIVE_STATUSES = new Set(["queued", "running"]);
const TERMINAL_STATUSES = new Set(["succeeded", "failed", "cancelled"]);

function operationKey(job) {
  const batchId = job.kind === "batch_generate" ? job.request_payload?.batch_id : null;
  return batchId == null ? `job:${job.id}` : `batch:${batchId}`;
}

function summarizeOperations(jobs, batches = []) {
  const operations = new Map();
  for (const job of jobs) {
    // ForgeBAT is represented by its durable parent below. Child jobs are an
    // implementation detail and must not inflate counts or emit toast storms.
    if (job.kind === "batch_generate") continue;
    const key = operationKey(job);
    const current = operations.get(key);
    if (!current) {
      operations.set(key, {
        key,
        workspace: workspaceForJobKind(job.kind),
        label: key.startsWith("batch:") ? `Batch #${job.request_payload.batch_id}` : displayJobId(job),
        status: job.status,
        error: job.error_message ?? "",
        jobs: [job],
      });
      continue;
    }
    current.jobs.push(job);
    const statuses = new Set(current.jobs.map((item) => item.status));
    current.status = statuses.has("running")
      ? "running"
      : statuses.has("queued")
        ? "queued"
        : statuses.has("failed")
          ? "failed"
          : statuses.has("cancelled")
            ? "cancelled"
            : "succeeded";
    current.error ||= job.error_message ?? "";
  }
  for (const batch of batches) {
    operations.set(`batch:${batch.id}`, {
      key: `batch:${batch.id}`,
      workspace: "forgebat",
      label: `Batch #${batch.id}`,
      status: batch.status === "pending" ? "queued" : batch.status === "completed" ? "succeeded" : batch.status,
      error: batch.counts?.failed ? `${batch.counts.failed} image${batch.counts.failed === 1 ? "" : "s"} failed.` : "",
      jobs: [],
    });
  }
  return operations;
}

function transitionToast(operation) {
  const product = workspaceLabel(operation.workspace);
  if (operation.status === "succeeded") {
    return { tone: "success", title: `${product} operation complete`, detail: `${operation.label} is ready.` };
  }
  if (operation.status === "failed") {
    return { tone: "danger", title: `${product} operation failed`, detail: operation.error || `${operation.label} needs attention.` };
  }
  return { tone: "neutral", title: `${product} operation cancelled`, detail: `${operation.label} was cancelled.` };
}

export function useOperationsMonitor(account) {
  const accountId = account?.id ?? null;
  const [jobs, setJobs] = useState([]);
  const [batches, setBatches] = useState([]);
  const [toasts, setToasts] = useState([]);
  const previous = useRef(new Map());
  const initialized = useRef(false);
  const nextToastId = useRef(1);

  const pushToast = useCallback((toast) => {
    const id = nextToastId.current++;
    setToasts((current) => [...current.slice(-3), { id, ...toast }]);
    return id;
  }, []);

  const dismissToast = useCallback((id) => {
    setToasts((current) => current.filter((toast) => toast.id !== id));
  }, []);

  useEffect(() => {
    if (!accountId) {
      setJobs([]);
      setBatches([]);
      setToasts([]);
      previous.current = new Map();
      initialized.current = false;
      return undefined;
    }
    let cancelled = false;
    let timer = null;

    const refresh = async () => {
      let delay = 5000;
      try {
        const [page, batchPage] = await Promise.all([
          getJobs(null, 100, null, null, true),
          getBatches(20),
        ]);
        if (cancelled) return;
        const nextJobs = page.items ?? [];
        const nextBatches = batchPage.batches ?? [];
        const next = summarizeOperations(nextJobs, nextBatches);
        if (initialized.current) {
          for (const [key, operation] of next) {
            const old = previous.current.get(key);
            if (old && ACTIVE_STATUSES.has(old.status) && TERMINAL_STATUSES.has(operation.status)) {
              pushToast(transitionToast(operation));
            }
          }
        } else {
          initialized.current = true;
        }
        previous.current = next;
        setJobs((current) => JSON.stringify(current) === JSON.stringify(nextJobs) ? current : nextJobs);
        setBatches((current) => JSON.stringify(current) === JSON.stringify(nextBatches) ? current : nextBatches);
        delay = [...next.values()].some((operation) => ACTIVE_STATUSES.has(operation.status)) ? 1000 : 5000;
      } catch {
        // Keep the last reliable operation snapshot through transient API failures.
      }
      if (!cancelled) timer = window.setTimeout(refresh, delay);
    };

    refresh();
    const onFocus = () => {
      if (timer) window.clearTimeout(timer);
      refresh();
    };
    window.addEventListener("focus", onFocus);
    return () => {
      cancelled = true;
      if (timer) window.clearTimeout(timer);
      window.removeEventListener("focus", onFocus);
    };
  }, [accountId, pushToast]);

  const operations = useMemo(() => [...summarizeOperations(jobs, batches).values()], [batches, jobs]);
  const activeOperations = useMemo(
    () => operations.filter((operation) => ACTIVE_STATUSES.has(operation.status)),
    [operations],
  );
  const summary = useMemo(() => {
    const byWorkspace = { forgeai: 0, forgeimg: 0, forgevid: 0, forgebat: 0 };
    let queued = 0;
    let running = 0;
    for (const operation of activeOperations) {
      byWorkspace[operation.workspace] += 1;
      if (operation.status === "running") running += 1;
      else queued += 1;
    }
    return { active: activeOperations.length, queued, running, byWorkspace };
  }, [activeOperations]);

  return { jobs, summary, toasts, pushToast, dismissToast };
}
