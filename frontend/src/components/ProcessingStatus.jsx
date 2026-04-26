import { Link } from "react-router-dom";
import { ArrowUpRight, CheckCircle2, Clock3, Radar, Sparkles, UploadCloud } from "lucide-react";
import { SkeletonBlock, SkeletonLines, SkeletonMetric } from "./Skeleton";
import {
  formatDateTime,
  formatNumber,
  getProcessingProgress,
  getSessionDisplayTitle,
  getSessionHeadline,
  getStatusLabel,
  getStatusTone
} from "../utils/formatters";

const STATUS_COPY = {
  pending: "Waiting to start",
  processing: "Processing now",
  completed: "Ready to review",
  failed: "Needs attention"
};

export default function ProcessingStatus({ session, loading = false }) {
  if (loading) {
    return (
      <section className="panel status-panel status-panel--loading" data-aos="fade-up">
        <div className="status-panel__header">
          <div className="status-panel__copy">
            <div className="status-panel__eyebrow">
              <SkeletonBlock className="skeleton-block--label" />
              <SkeletonBlock className="skeleton-block--chip" />
            </div>
            <SkeletonLines lines={2} />
          </div>
          <SkeletonBlock className="skeleton-block--button" />
        </div>
        <div className="status-facts status-facts--loading">
          <SkeletonMetric />
          <SkeletonMetric />
          <SkeletonMetric />
        </div>
        <SkeletonBlock className="skeleton-block--progress" />
      </section>
    );
  }

  if (!session) {
    return (
      <section className="panel status-panel status-panel--empty" data-aos="fade-up">
        <div className="status-panel__header status-panel__header--empty">
          <div className="status-panel__copy">
            <div className="status-panel__eyebrow">
              <span className="section-kicker">Current session</span>
            </div>
            <h2 className="status-panel__title">No active analysis</h2>
            <p className="empty-state">Start with an upload.</p>
          </div>

          <div className="status-panel__summary">
            <Link className="primary-button primary-button--icon" to="/workspace/upload">
              <UploadCloud size={16} strokeWidth={2.1} aria-hidden="true" />
              Upload video
            </Link>
          </div>
        </div>
      </section>
    );
  }

  const tone = getStatusTone(session.status);
  const progress = getProcessingProgress(session);
  const rawTitle = getSessionHeadline(session);
  const displayTitle = getSessionDisplayTitle(session);
  const detailItems = [
    { label: "Uploaded", value: formatDateTime(session.uploadedAt), icon: Clock3 },
    {
      label: session.completedAt ? "Finished" : "Started",
      value: formatDateTime(session.completedAt || session.startedAt),
      icon: session.completedAt ? CheckCircle2 : Sparkles
    },
    { label: "Tracks", value: `${formatNumber(session.summary?.tracked_objects || 0)} tracks`, icon: Radar }
  ];

  if (Number(session.summary?.frame_errors || 0) > 0) {
    detailItems.push({
      label: "Errors",
      value: `${formatNumber(session.summary?.frame_errors || 0)} errors`,
      icon: Sparkles
    });
  }

  return (
    <section className="panel status-panel" data-aos="fade-up">
      <div className="status-panel__header">
        <div className="status-panel__copy">
          <div className="status-panel__eyebrow">
            <span className="section-kicker">Current session</span>
            <span className={`status-pill status-pill--${tone}`}>{getStatusLabel(session.status)}</span>
          </div>

          <div className="status-panel__title-row">
            <h2 className="status-panel__title" title={rawTitle}>
              {displayTitle}
            </h2>
            <span className="status-panel__progress-inline">{progress}%</span>
          </div>

          <p className="status-panel__note">{STATUS_COPY[session.status] || STATUS_COPY.pending}</p>
        </div>

        <div className="status-panel__summary">
          <Link className="secondary-button secondary-button--small secondary-button--icon" to={`/workspace/results/${session.videoId}`}>
            {session.status === "completed" ? "Open results" : "View session"}
            <ArrowUpRight size={16} strokeWidth={2.1} aria-hidden="true" />
          </Link>
        </div>
      </div>

      <div className="status-facts">
        {detailItems.map((item) => (
          <div key={item.label} className="status-facts__item">
            <span>
              {item.icon ? <item.icon size={14} strokeWidth={2.1} aria-hidden="true" /> : null}
              {item.label}
            </span>
            <strong>{item.value}</strong>
          </div>
        ))}
      </div>

      <div className="status-progress" aria-hidden="true">
        <span className="status-progress__bar" style={{ width: `${progress}%` }} />
      </div>

      {session.errorMessage ? <div className="error-banner">{session.errorMessage}</div> : null}
    </section>
  );
}
