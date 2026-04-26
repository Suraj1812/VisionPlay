import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { Download, Library, UploadCloud } from "lucide-react";
import StatsPanel from "../components/StatsPanel";
import CommentaryPanel from "../components/CommentaryPanel";
import CricketDashboard from "../components/CricketDashboard";
import { SkeletonBlock, SkeletonLines, SkeletonMetric } from "../components/Skeleton";
import VideoOverlayPlayer from "../components/VideoOverlayPlayer";
import { useVisionPlay } from "../context/VisionPlayContext";
import {
  formatNumber,
  getProcessingProgress,
  getSessionHeadline,
  getStatusLabel
} from "../utils/formatters";

export default function ResultsPage() {
  const { videoId } = useParams();
  const [pageError, setPageError] = useState("");
  const [loading, setLoading] = useState(true);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);
  const { isBooting, ensureVideo, getResultsById, getSessionById } = useVisionPlay();

  const session = getSessionById(videoId);
  const results = getResultsById(videoId);
  const progress = getProcessingProgress(session);
  const summary = results?.summary || session?.summary || {};
  const cricket = results?.cricket || {};
  const objectTypeCount = Array.isArray(summary.object_types)
    ? summary.object_types.length
    : Object.keys(summary.tracks_by_type || {}).length;
  const stableTracks = summary.tracked_objects || results?.tracks?.length || 0;
  const eventCount = cricket?.deliveries?.length || results?.events?.length || summary.interaction_count || 0;
  const sessionTitle = getSessionHeadline(session);
  const profileLabel = cricket?.profile === "cricket_end_on_action_cam_v1" ? "End-on cricket" : cricket?.profile;

  useEffect(() => {
    let isActive = true;
    setLoading(true);
    setPageError("");

    ensureVideo(videoId)
      .catch((requestError) => {
        if (isActive) {
          setPageError(requestError.message);
        }
      })
      .finally(() => {
        if (isActive) {
          setLoading(false);
        }
      });

    return () => {
      isActive = false;
    };
  }, [videoId]);

  function handleExportJson() {
    if (!results) {
      return;
    }

    const blob = new Blob([JSON.stringify(results, null, 2)], { type: "application/json" });
    const objectUrl = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = `${sessionTitle || videoId || "visionplay-results"}.json`;
    link.click();
    URL.revokeObjectURL(objectUrl);
  }

  return (
    <div className="page-stack page-stack--results">
      {pageError ? <div className="error-banner">{pageError}</div> : null}

      <section className="results-header panel" data-aos="fade-up">
        {loading || isBooting ? (
          <>
            <div className="results-header__copy">
              <SkeletonBlock className="skeleton-block--label" />
              <SkeletonLines lines={2} />
              <div className="results-header__meta results-header__meta--loading">
                <SkeletonBlock className="skeleton-block--chip" />
                <SkeletonBlock className="skeleton-block--chip" />
                <SkeletonBlock className="skeleton-block--chip" />
              </div>
            </div>
            <div className="results-header__actions">
              <SkeletonBlock className="skeleton-block--button" />
              <SkeletonBlock className="skeleton-block--button" />
            </div>
          </>
        ) : (
          <>
            <div className="results-header__copy">
              <span className="section-kicker">Results</span>
              <h1 className="results-header__title" title={sessionTitle}>
                {sessionTitle}
              </h1>
              <div className="results-header__meta">
                <span>{session ? getStatusLabel(session.status) : "Loading"}</span>
                <span>{formatNumber(stableTracks)} tracks</span>
                <span>{formatNumber(eventCount)} deliveries</span>
                <span>{formatNumber(objectTypeCount)} detected types</span>
                {profileLabel ? <span>{profileLabel}</span> : null}
              </div>
            </div>

            <div className="results-header__actions">
              <Link className="secondary-button secondary-button--icon" to="/workspace/library">
                <Library size={17} strokeWidth={2.1} aria-hidden="true" />
                Library
              </Link>
              <Link className="primary-button primary-button--icon" to="/workspace/upload">
                <UploadCloud size={17} strokeWidth={2.1} aria-hidden="true" />
                Upload
              </Link>
              {results ? (
                <button type="button" className="secondary-button secondary-button--icon" onClick={handleExportJson}>
                  <Download size={17} strokeWidth={2.1} aria-hidden="true" />
                  Export
                </button>
              ) : null}
            </div>
          </>
        )}
      </section>

      {loading || isBooting ? (
        <>
          <section className="panel video-panel" data-aos="fade-up" data-aos-delay="80">
            <div className="video-panel__header">
              <SkeletonLines lines={1} />
              <div className="video-panel__toolbar">
                <SkeletonBlock className="skeleton-block--button" />
                <SkeletonBlock className="skeleton-block--button" />
              </div>
            </div>
            <div className="skeleton-stage" />
            <div className="video-meta-inline video-meta-inline--loading">
              <SkeletonBlock className="skeleton-block--label" />
              <SkeletonBlock className="skeleton-block--label" />
              <SkeletonBlock className="skeleton-block--label" />
            </div>
          </section>
          <StatsPanel loading />
        </>
      ) : results?.video ? (
        <>
          <div data-aos="fade-up" data-aos-delay="80">
            <VideoOverlayPlayer
              video={results.video}
              frames={results.frames || []}
              cricket={cricket}
              onTimeUpdate={setCurrentTimeMs}
            />
          </div>

          {(cricket.timeline?.length > 0 || cricket.commentary?.length > 0) && (
            <div data-aos="fade-up" data-aos-delay="100">
              <CommentaryPanel
                cricket={cricket}
                currentTimeMs={currentTimeMs}
                onSeekTo={(ms) => {
                  const videoEl = document.querySelector(".video-element");
                  if (videoEl) {
                    videoEl.currentTime = ms / 1000;
                    setCurrentTimeMs(ms);
                  }
                }}
              />
            </div>
          )}

          {Object.keys(cricket).length > 0 && (
            <div data-aos="fade-up" data-aos-delay="140">
              <CricketDashboard cricket={cricket} />
            </div>
          )}

          <div data-aos="fade-up" data-aos-delay="160">
            <StatsPanel summary={results.summary} tracks={results.tracks || []} events={results.events || []} />
          </div>
        </>
      ) : (
        <section className="panel empty-panel empty-panel--results" data-aos="fade-up" data-aos-delay="140">
          <span className="section-kicker">Results pending</span>
          <h2>
            {loading || session?.status === "pending" || session?.status === "processing"
              ? `${progress}% complete`
              : "Results are not available yet"}
          </h2>
          <p className="empty-state">
            {loading || session?.status === "pending" || session?.status === "processing"
              ? "The engine is still processing detections, tracks, and event summaries."
              : "No result payload is available for this session yet."}
          </p>
        </section>
      )}
    </div>
  );
}
