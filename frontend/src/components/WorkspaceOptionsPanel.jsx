import { useEffect, useState } from "react";
import { ChevronRight, Database, RefreshCw, Trash2 } from "lucide-react";
import ConfirmDialog from "./ConfirmDialog";
import { SkeletonBlock, SkeletonMetric } from "./Skeleton";
import { useVisionPlay } from "../context/VisionPlayContext";

export default function WorkspaceOptionsPanel({ loading = false }) {
  const { sessions, refreshSessions, deleteOldSessions, deleteAllSessions } = useVisionPlay();
  const [busyAction, setBusyAction] = useState("");
  const [actionMessage, setActionMessage] = useState("");
  const [actionError, setActionError] = useState("");
  const [confirmAction, setConfirmAction] = useState("");

  const processingCount = sessions.filter((session) => session.status === "processing").length;
  const oldSessionsCount = sessions.filter(
    (session) => session.status === "completed" || session.status === "failed"
  ).length;

  if (loading) {
    return (
      <section className="panel control-panel control-panel--loading" data-aos="fade-up">
        <div className="section-header">
          <div className="section-header__copy">
            <SkeletonBlock className="skeleton-block--title" />
          </div>
          <div className="control-panel__counts control-panel__counts--loading">
            <SkeletonMetric />
          </div>
        </div>
        <div className="control-panel__actions">
          <SkeletonBlock className="skeleton-block--action" />
          <SkeletonBlock className="skeleton-block--action" />
          <SkeletonBlock className="skeleton-block--action" />
        </div>
      </section>
    );
  }

  useEffect(() => {
    if (!actionMessage) {
      return undefined;
    }

    const timeoutId = window.setTimeout(() => {
      setActionMessage("");
    }, 2400);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [actionMessage]);

  async function runAction(actionKey, action, successMessage) {
    setBusyAction(actionKey);
    setActionError("");
    setActionMessage("");

    try {
      const response = await action();
      setActionMessage(response?.message || successMessage);
    } catch (requestError) {
      setActionError(requestError.message || "Unable to complete the request.");
    } finally {
      setBusyAction("");
      setConfirmAction("");
    }
  }

  function handleRefreshSessions() {
    return runAction(
      "refresh",
      async () => {
        const refreshedCount = await refreshSessions();
        return {
          message: refreshedCount ? "Workspace updated" : "Already up to date"
        };
      },
      "Workspace updated"
    );
  }

  function handleConfirmAction() {
    if (confirmAction === "delete-old") {
      void runAction("delete-old", deleteOldSessions, "Old sessions deleted");
      return;
    }

    if (confirmAction === "delete-all") {
      void runAction("delete-all", deleteAllSessions, "All session data deleted");
    }
  }

  return (
    <section className="panel control-panel" data-aos="fade-up">
      <div className="section-header">
        <div className="section-header__copy">
          <h2>Workspace</h2>
        </div>
        <div className="control-panel__counts">
          <span>Stored {sessions.length}</span>
          <span>Active {processingCount}</span>
          <span>Old {oldSessionsCount}</span>
        </div>
      </div>

      <div className="control-panel__actions">
        <button
          type="button"
          className="control-action"
          onClick={handleRefreshSessions}
          disabled={busyAction === "refresh"}
        >
          <span className="control-action__title">
            <RefreshCw size={16} strokeWidth={2.1} aria-hidden="true" />
            {busyAction === "refresh" ? "Refreshing..." : "Refresh"}
          </span>
          <span className="control-action__meta">Sync sessions</span>
          <ChevronRight size={16} strokeWidth={2.1} aria-hidden="true" className="control-action__chevron" />
        </button>

        <button
          type="button"
          className="control-action"
          onClick={() => setConfirmAction("delete-old")}
          disabled={!oldSessionsCount || busyAction === "delete-old"}
        >
          <span className="control-action__title">
            <Trash2 size={16} strokeWidth={2.1} aria-hidden="true" />
            {busyAction === "delete-old" ? "Deleting..." : "Delete old"}
          </span>
          <span className="control-action__meta">Completed and failed</span>
          <ChevronRight size={16} strokeWidth={2.1} aria-hidden="true" className="control-action__chevron" />
        </button>

        <button
          type="button"
          className="control-action control-action--danger"
          onClick={() => setConfirmAction("delete-all")}
          disabled={!sessions.length || busyAction === "delete-all"}
        >
          <span className="control-action__title">
            <Database size={16} strokeWidth={2.1} aria-hidden="true" />
            {busyAction === "delete-all" ? "Deleting..." : "Delete all data"}
          </span>
          <span className="control-action__meta">Everything in workspace</span>
          <ChevronRight size={16} strokeWidth={2.1} aria-hidden="true" className="control-action__chevron" />
        </button>
      </div>

      {actionMessage ? <div className="control-panel__feedback control-panel__feedback--success">{actionMessage}</div> : null}
      {actionError ? <div className="control-panel__feedback control-panel__feedback--error">{actionError}</div> : null}

      <ConfirmDialog
        open={Boolean(confirmAction)}
        title={confirmAction === "delete-all" ? "Delete all data?" : "Delete old sessions?"}
        description={
          confirmAction === "delete-all"
            ? "This removes every stored session and uploaded media file from VisionPlay."
            : "This removes completed and failed sessions while keeping active work untouched."
        }
        confirmLabel={confirmAction === "delete-all" ? "Delete everything" : "Delete old sessions"}
        tone="danger"
        busy={busyAction === "delete-old" || busyAction === "delete-all"}
        onClose={() => {
          if (!busyAction) {
            setConfirmAction("");
          }
        }}
        onConfirm={handleConfirmAction}
      />
    </section>
  );
}
