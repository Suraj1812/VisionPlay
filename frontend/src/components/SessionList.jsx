import { useState } from "react";
import { Link } from "react-router-dom";
import { FolderOpen, Trash2 } from "lucide-react";
import ConfirmDialog from "./ConfirmDialog";
import { SkeletonBlock, SkeletonLines } from "./Skeleton";
import {
  formatDateTime,
  formatNumber,
  getProcessingProgress,
  getSessionDisplayTitle,
  getSummarySnapshot,
  getSessionHeadline,
  getStatusLabel,
  getStatusTone
} from "../utils/formatters";
import { useVisionPlay } from "../context/VisionPlayContext";

export default function SessionList({
  sessions,
  title,
  emptyTitle,
  emptyCopy,
  emptyAction = null,
  loading = false,
  compact = false,
  actions = null
}) {
  const { deleteSession } = useVisionPlay();
  const [deletingVideoId, setDeletingVideoId] = useState("");
  const [actionError, setActionError] = useState("");
  const [pendingDeleteSession, setPendingDeleteSession] = useState(null);

  async function handleConfirmDelete() {
    if (!pendingDeleteSession) {
      return;
    }

    setDeletingVideoId(pendingDeleteSession.videoId);
    setActionError("");

    try {
      await deleteSession(pendingDeleteSession.videoId);
      setPendingDeleteSession(null);
    } catch (requestError) {
      setActionError(requestError.message || "Unable to delete the session.");
    } finally {
      setDeletingVideoId("");
    }
  }

  return (
    <section className="panel section-panel section-panel--sessions" data-aos="fade-up">
      <div className="section-header">
        <div className="section-header__copy">
          <span className="section-kicker">Sessions</span>
          <h2>{title}</h2>
        </div>
        {actions ? <div className="section-header__actions">{actions}</div> : null}
      </div>

      {actionError ? <div className="error-banner">{actionError}</div> : null}

      <div className={compact ? "session-list session-list--compact" : "session-list"}>
        {loading ? (
          Array.from({ length: compact ? 2 : 3 }).map((_, index) => (
            <article key={index} className={compact ? "session-row session-row--compact" : "session-row"}>
              <div className="session-row__content session-row__content--static">
                <div className="session-row__main">
                  <div className="session-row__top">
                    <SkeletonLines lines={1} className="session-row__skeleton-title" />
                    <SkeletonBlock className="skeleton-block--chip" />
                  </div>
                  <div className="session-row__meta session-row__meta--skeleton">
                    <SkeletonBlock className="skeleton-block--label" />
                    <SkeletonBlock className="skeleton-block--label" />
                    <SkeletonBlock className="skeleton-block--label" />
                  </div>
                </div>
              </div>
              <div className="session-row__actions">
                <SkeletonBlock className="skeleton-block--button" />
              </div>
            </article>
          ))
        ) : sessions.length ? (
          sessions.map((session) => {
            const progress = getProcessingProgress(session);
            const rawTitle = getSessionHeadline(session);
            const displayTitle = getSessionDisplayTitle(session);

            return (
              <article
                key={session.videoId}
                className={compact ? "session-row session-row--compact" : "session-row"}
              >
                <Link className="session-row__content" to={`/workspace/results/${session.videoId}`}>
                  <div className="session-row__main">
                    <div className="session-row__top">
                      <div className="session-row__titleline">
                        <FolderOpen size={16} strokeWidth={2.1} aria-hidden="true" />
                        <strong className="session-row__title" title={rawTitle}>
                          {displayTitle}
                        </strong>
                      </div>
                      <span className={`status-pill status-pill--${getStatusTone(session.status)}`}>
                        {getStatusLabel(session.status)}
                      </span>
                    </div>

                    <div className="session-row__meta">
                      <span>{formatDateTime(session.uploadedAt)}</span>
                      <span>{formatNumber(session.summary?.tracked_objects || 0)} tracks</span>
                      {getSummarySnapshot(session.summary) ? (
                        <span>{getSummarySnapshot(session.summary)}</span>
                      ) : null}
                    </div>

                    {session.status === "pending" || session.status === "processing" ? (
                      <div className="session-row__progress">
                        <span style={{ width: `${progress}%` }} />
                      </div>
                    ) : null}
                  </div>
                </Link>

                <div className="session-row__actions">
                  <button
                    type="button"
                    className="session-card__delete secondary-button secondary-button--small secondary-button--icon"
                    onClick={() => setPendingDeleteSession(session)}
                    disabled={deletingVideoId === session.videoId}
                  >
                    <Trash2 size={15} strokeWidth={2.1} aria-hidden="true" />
                    {deletingVideoId === session.videoId ? "Deleting..." : "Delete"}
                  </button>
                </div>
              </article>
            );
          })
        ) : (
          <div className="empty-box empty-box--minimal">
            <div className="empty-box__copy">
              <strong>{emptyTitle}</strong>
              <p>{emptyCopy}</p>
            </div>
            {emptyAction ? <div className="empty-box__actions">{emptyAction}</div> : null}
          </div>
        )}
      </div>

      <ConfirmDialog
        open={Boolean(pendingDeleteSession)}
        title="Delete session?"
        description={
          pendingDeleteSession
            ? `This removes "${getSessionHeadline(
                pendingDeleteSession
              )}" and all stored files for that analysis.`
            : ""
        }
        confirmLabel="Delete session"
        tone="danger"
        busy={Boolean(deletingVideoId)}
        onClose={() => {
          if (!deletingVideoId) {
            setPendingDeleteSession(null);
          }
        }}
        onConfirm={handleConfirmDelete}
      />
    </section>
  );
}
