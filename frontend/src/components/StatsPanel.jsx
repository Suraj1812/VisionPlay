import { Activity, ListTree, Radar } from "lucide-react";
import { SkeletonBlock, SkeletonLines } from "./Skeleton";
import { formatDurationMs, formatNumber, formatObjectLabel } from "../utils/formatters";

function buildTrackPath(track) {
  if (!track.path?.length) {
    return "";
  }

  const xs = track.path.map((point) => point.x);
  const ys = track.path.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = maxX - minX || 1;
  const height = maxY - minY || 1;

  return track.path
    .map((point, index) => {
      const x = ((point.x - minX) / width) * 220 + 10;
      const y = ((point.y - minY) / height) * 70 + 10;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function formatEventTitle(event) {
  if (event.event_type === "interaction" && Array.isArray(event.details?.participants)) {
    return event.details.participants.map((item) => formatObjectLabel(item)).join(" + ");
  }

  return formatObjectLabel(event.event_type);
}

function formatEventCopy(event) {
  if (event.event_type === "interaction") {
    const trackingIds = (event.tracking_ids || []).map((trackingId) => `#${trackingId}`).join(", ");
    return trackingIds || "Tracked interaction";
  }

  return `${formatObjectLabel(event.object_type)} #${event.tracking_ids?.[0] ?? "NA"}`;
}

function getConfidencePercent(value) {
  const numericValue = Number(value || 0);
  if (!Number.isFinite(numericValue) || numericValue <= 0) {
    return 0;
  }

  if (numericValue <= 1) {
    return numericValue * 100;
  }

  return Math.min(100, numericValue);
}

export default function StatsPanel({ summary = {}, tracks = [], events = [], loading = false }) {
  if (loading) {
    return (
      <section className="stats-shell">
        <div className="results-sections">
          {Array.from({ length: 3 }).map((_, index) => (
            <section key={index} className="panel results-section">
              <div className="section-header section-header--stacked">
                <div className="section-header__copy">
                  <SkeletonBlock className="skeleton-block--label" />
                  <SkeletonBlock className="skeleton-block--title" />
                </div>
              </div>
              <SkeletonLines lines={4} />
            </section>
          ))}
        </div>
      </section>
    );
  }

  const objectTypes = Array.isArray(summary.object_types)
    ? summary.object_types
    : Object.keys(summary.tracks_by_type || {});
  const breakdownRows = objectTypes.map((objectType) => ({
    objectType,
    detections: summary.detections_by_type?.[objectType] || 0,
    tracks: summary.tracks_by_type?.[objectType] || 0
  }));

  const primaryTracks = [...tracks]
    .sort((left, right) => {
      if (Boolean(left.is_primary) !== Boolean(right.is_primary)) {
        return left.is_primary ? -1 : 1;
      }
      return (right.primary_score || 0) - (left.primary_score || 0);
    })
    .slice(0, 4);

  const recentEvents = [...events]
    .sort((left, right) => left.start_frame - right.start_frame)
    .slice(0, 5);

  return (
    <section className="stats-shell">
      <div className="results-sections">
        <section className="panel results-section">
          <div className="section-header section-header--stacked">
            <div className="section-header__copy">
              <span className="section-kicker">Object mix</span>
              <h2><Activity size={18} strokeWidth={2.1} aria-hidden="true" /> Detected classes</h2>
            </div>
          </div>

          <div className="results-list">
            {breakdownRows.length ? (
              breakdownRows.map((row) => (
                <article key={row.objectType} className="results-list__item">
                  <div className="results-list__main">
                    <strong>{formatObjectLabel(row.objectType)}</strong>
                    <span>{formatNumber(row.detections)} detections</span>
                  </div>
                  <span className="results-list__value">{formatNumber(row.tracks)} tracks</span>
                </article>
              ))
            ) : (
              <p className="empty-state">No tracked classes available yet.</p>
            )}
          </div>
        </section>

        <section className="panel results-section">
          <div className="section-header section-header--stacked">
            <div className="section-header__copy">
              <span className="section-kicker">Timeline</span>
              <h2><ListTree size={18} strokeWidth={2.1} aria-hidden="true" /> Recent events</h2>
            </div>
          </div>

          <div className="results-list">
            {recentEvents.length ? (
              recentEvents.map((event, index) => (
                <article key={`${event.event_type}-${event.start_frame}-${index}`} className="results-list__item">
                  <div className="results-list__main">
                    <strong>{formatEventTitle(event)}</strong>
                    <span>{formatEventCopy(event)}</span>
                  </div>
                  <span className="results-list__value">
                    {event.start_frame}-{event.end_frame}
                  </span>
                </article>
              ))
            ) : (
              <p className="empty-state">No events available yet.</p>
            )}
          </div>
        </section>

        <section className="panel results-section">
          <div className="section-header section-header--stacked">
            <div className="section-header__copy">
              <span className="section-kicker">Top tracks</span>
              <h2><Radar size={18} strokeWidth={2.1} aria-hidden="true" /> Most important tracks</h2>
            </div>
          </div>

          <div className="track-grid track-grid--compact">
            {primaryTracks.length ? (
              primaryTracks.map((track) => (
                <article key={`${track.object_type}-${track.tracking_id}`} className="track-card track-card--compact">
                  <div className="track-card__meta">
                    <strong>
                      {formatObjectLabel(track.object_type)} #{track.tracking_id}
                    </strong>
                    <span>{formatDurationMs(track.duration_ms || track.time_in_frame_ms || 0)} in frame</span>
                    <span>{formatNumber(track.distance_px || 0)} px travelled</span>
                    <span>{formatNumber(getConfidencePercent(track.avg_confidence || 0))}% confidence</span>
                  </div>
                  <svg viewBox="0 0 240 90" className="track-map" preserveAspectRatio="none">
                    <path d={buildTrackPath(track)} />
                  </svg>
                </article>
              ))
            ) : (
              <p className="empty-state">No stable tracks yet.</p>
            )}
          </div>
        </section>
      </div>
    </section>
  );
}
