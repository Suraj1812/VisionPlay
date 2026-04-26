import { useEffect } from "react";
import { createPortal } from "react-dom";
import { AlertTriangle, CheckCircle2 } from "lucide-react";

export default function ConfirmDialog({
  open,
  title,
  description,
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  tone = "default",
  busy = false,
  onConfirm,
  onClose
}) {
  useEffect(() => {
    if (!open) {
      return undefined;
    }

    function handleEscape(event) {
      if (!busy && event.key === "Escape") {
        onClose?.();
      }
    }

    window.addEventListener("keydown", handleEscape);
    return () => {
      window.removeEventListener("keydown", handleEscape);
    };
  }, [busy, onClose, open]);

  if (!open) {
    return null;
  }

  const dialog = (
    <div
      className="confirm-dialog-backdrop"
      role="presentation"
      onClick={(event) => {
        if (!busy && event.target === event.currentTarget) {
          onClose?.();
        }
      }}
    >
      <div className="confirm-dialog" role="dialog" aria-modal="true" aria-labelledby="confirm-title">
        <span className={`confirm-dialog__icon ${tone === "danger" ? "confirm-dialog__icon--danger" : ""}`}>
          {tone === "danger" ? (
            <AlertTriangle size={18} strokeWidth={2.1} aria-hidden="true" />
          ) : (
            <CheckCircle2 size={18} strokeWidth={2.1} aria-hidden="true" />
          )}
        </span>
        <span className="section-kicker">Confirm</span>
        <h3 id="confirm-title">{title}</h3>
        <p>{description}</p>

        <div className="confirm-dialog__actions">
          <button type="button" className="secondary-button" onClick={onClose} disabled={busy}>
            {cancelLabel}
          </button>
          <button
            type="button"
            className={tone === "danger" ? "danger-button" : "primary-button"}
            onClick={onConfirm}
            disabled={busy}
          >
            {busy ? "Working..." : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );

  if (typeof document === "undefined") {
    return dialog;
  }

  return createPortal(dialog, document.body);
}
