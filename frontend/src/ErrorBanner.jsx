import { X } from "lucide-react";

// A page-level error stays until something replaces it, so a transient one -
// "queue limit reached" being the common case - lingers long after the queue
// has drained. Dismissing clears the state rather than only hiding the banner,
// so the same error can be shown again if it recurs.
export default function ErrorBanner({ error, onDismiss, className = "" }) {
  if (!error) return null;
  return (
    <div
      role="alert"
      className={`flex items-start gap-3 rounded-[9px] bg-[var(--danger-wash)] px-[13px] py-[10px] text-[12px] font-medium leading-[1.45] text-[#ff9d9d] ${className}`}
    >
      <span className="min-w-0 flex-1">{error}</span>
      {onDismiss && (
        <button
          type="button"
          onClick={onDismiss}
          aria-label="Dismiss error"
          className="-mr-1 -mt-0.5 shrink-0 rounded-[6px] p-1 text-[#ff9d9d] transition hover:bg-[color-mix(in_srgb,var(--danger)_20%,transparent)] hover:text-white"
        >
          <X size={14} />
        </button>
      )}
    </div>
  );
}
