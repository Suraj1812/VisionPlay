import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { API_BASE_URL } from "../api";
import { formatObjectLabel, formatNumber } from "../utils/formatters";

const OVERLAY_PALETTE = [
  { border: "#8db2ff", badge: "#1b49ff" },
  { border: "#62d3c6", badge: "#0f9384" },
  { border: "#ffb86b", badge: "#d97706" },
  { border: "#ff8f8f", badge: "#dc2626" },
  { border: "#c6a4ff", badge: "#7c3aed" },
  { border: "#9fe7ae", badge: "#059669" }
];

function getClosestFrameData(frameLookup, currentFrame, fps) {
  if (frameLookup[currentFrame]) {
    return frameLookup[currentFrame];
  }

  const maxFrameGap = Math.max(2, Math.round((fps || 25) / 6));
  for (let offset = 1; offset <= maxFrameGap; offset += 1) {
    if (frameLookup[currentFrame - offset]) {
      return frameLookup[currentFrame - offset];
    }
    if (frameLookup[currentFrame + offset]) {
      return frameLookup[currentFrame + offset];
    }
  }

  return null;
}

function paletteForLabel(label) {
  const source = String(label || "unknown");
  let hash = 0;
  for (let index = 0; index < source.length; index += 1) {
    hash = source.charCodeAt(index) + ((hash << 5) - hash);
  }

  return OVERLAY_PALETTE[Math.abs(hash) % OVERLAY_PALETTE.length];
}

function getActiveCaption(cricket, currentTimeMs) {
  if (!cricket) return null;
  const subtitleCues = cricket.subtitles?.cues || [];
  for (let i = subtitleCues.length - 1; i >= 0; i -= 1) {
    const cue = subtitleCues[i];
    const start = cue.start_ms || 0;
    const end = cue.end_ms || start + 2500;
    if (currentTimeMs >= start && currentTimeMs <= end) {
      return {
        text: cue.text || "",
        detail: cue.source === "speech" ? "Speech caption" : "Match event",
        type: cue.source === "speech" ? "normal" : "event",
        over: cue.over || "",
        score: cricket.scorecard?.score || null,
      };
    }
  }
  const timeline = cricket.timeline || [];
  const events = cricket.events || [];
  const scorecard = cricket.scorecard || {};

  for (let i = timeline.length - 1; i >= 0; i--) {
    const entry = timeline[i];
    const start = entry.ts_start || 0;
    const end = entry.ts_end || start + 4000;
    if (currentTimeMs >= start && currentTimeMs <= end + 2000) {
      let detail = "";
      if (entry.shot !== "unknown" && entry.shot) {
        detail = entry.shot.replace(/_/g, " ");
        if (entry.zone > 0) detail += ` · zone ${entry.zone}`;
      }
      return {
        text: entry.commentary || `Over ${entry.over}`,
        detail,
        type: entry.wicket ? "wicket" : entry.six ? "six" : entry.boundary ? "four" : entry.dot ? "dot" : "normal",
        over: entry.over,
        score: scorecard.score || null,
      };
    }
  }

  for (let i = events.length - 1; i >= 0; i--) {
    const ev = events[i];
    const evTime = ev.timestamp_ms || 0;
    if (currentTimeMs >= evTime && currentTimeMs <= evTime + 3000) {
      const labels = {
        four: "FOUR!",
        six: "SIX!",
        wicket: "WICKET!",
        bat_impact: "Shot played",
        ball_released: "Ball released",
        ball_bounced: "Ball pitched",
        dot_ball: "Dot ball",
        over_complete: "End of over",
      };
      const label = labels[ev.event_type];
      if (!label) continue;
      const d = ev.details || {};
      let detail = "";
      if (ev.event_type === "bat_impact" && d.shot_type) {
        detail = d.shot_type.replace(/_/g, " ");
        if (d.power && d.power !== "unknown") detail += ` · ${d.power}`;
        if (d.wagon_zone) detail += ` · zone ${d.wagon_zone}`;
      } else if (ev.event_type === "ball_bounced" && d.length) {
        detail = `${d.length.replace(/_/g, " ")}`;
        if (d.line) detail += `, ${d.line.replace(/_/g, " ")}`;
      } else if ((ev.event_type === "four" || ev.event_type === "six") && d.runs) {
        detail = `${d.runs} runs`;
      }
      return {
        text: label,
        detail,
        type: ev.event_type === "wicket" ? "wicket" : ev.event_type === "six" ? "six" : ev.event_type === "four" ? "four" : "event",
        details: d,
        score: scorecard.score || null,
      };
    }
  }

  return null;
}

function buildSubtitleTrack(cricket) {
  if (!cricket) return "";
  const subtitleCues = Array.isArray(cricket.subtitles?.cues) ? cricket.subtitles.cues : [];
  if (subtitleCues.length) {
    const formatCueTime = (valueMs) => {
      const totalMs = Math.max(0, Math.floor(valueMs));
      const hours = Math.floor(totalMs / 3600000);
      const minutes = Math.floor((totalMs % 3600000) / 60000);
      const seconds = Math.floor((totalMs % 60000) / 1000);
      const milliseconds = totalMs % 1000;
      return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(milliseconds).padStart(3, "0")}`;
    };

    return [
      "WEBVTT",
      "",
      ...subtitleCues.flatMap((cue, index) => [
        String(index + 1),
        `${formatCueTime(cue.start_ms || 0)} --> ${formatCueTime(cue.end_ms || (cue.start_ms || 0) + 1000)}`,
        String(cue.text || "").trim(),
        "",
      ]),
    ].join("\n");
  }
  const timeline = Array.isArray(cricket.timeline) ? cricket.timeline : [];
  const events = Array.isArray(cricket.events) ? cricket.events : [];
  const cues = [];

  for (const entry of timeline) {
    const start = Number(entry.ts_start || 0);
    const end = Math.max(Number(entry.ts_end || start + 3000), start + 1000);
    const text = String(entry.commentary || "").trim();
    if (!text) continue;
    const detailParts = [];
    if (entry.shot && entry.shot !== "unknown") detailParts.push(String(entry.shot).replace(/_/g, " "));
    if (entry.length && entry.length !== "unknown") detailParts.push(String(entry.length).replace(/_/g, " "));
    if (entry.line && entry.line !== "unknown") detailParts.push(String(entry.line).replace(/_/g, " "));
    const detail = detailParts.join(" · ");
    cues.push({ start, end, text: detail ? `${text}\n${detail}` : text });
  }

  if (!cues.length) {
    const labels = {
      four: "FOUR!",
      six: "SIX!",
      wicket: "WICKET!",
      bat_impact: "Shot played",
      ball_released: "Ball released",
      ball_bounced: "Ball pitched",
      dot_ball: "Dot ball",
    };
    for (const event of events) {
      const text = labels[event.event_type];
      if (!text) continue;
      const start = Number(event.timestamp_ms || 0);
      cues.push({ start, end: start + 2200, text });
    }
  }

  if (!cues.length) return "";

  const formatCueTime = (valueMs) => {
    const totalMs = Math.max(0, Math.floor(valueMs));
    const hours = Math.floor(totalMs / 3600000);
    const minutes = Math.floor((totalMs % 3600000) / 60000);
    const seconds = Math.floor((totalMs % 60000) / 1000);
    const milliseconds = totalMs % 1000;
    return `${String(hours).padStart(2, "0")}:${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}.${String(milliseconds).padStart(3, "0")}`;
  };

  return [
    "WEBVTT",
    "",
    ...cues.flatMap((cue, index) => [
      String(index + 1),
      `${formatCueTime(cue.start)} --> ${formatCueTime(cue.end)}`,
      cue.text,
      "",
    ]),
  ].join("\n");
}

function EventTimelineBar({ events, videoDurationMs, currentTimeMs, onSeek }) {
  if (!events?.length || !videoDurationMs) return null;

  const markers = events
    .filter((ev) => ["four", "six", "wicket", "bat_impact", "dot_ball"].includes(ev.event_type))
    .map((ev) => ({
      position: Math.min(100, Math.max(0, ((ev.timestamp_ms || 0) / videoDurationMs) * 100)),
      type: ev.event_type,
      ts: ev.timestamp_ms || 0,
    }));

  if (!markers.length) return null;

  const colorMap = {
    four: "#4ade80",
    six: "#fbbf24",
    wicket: "#f87171",
    bat_impact: "#8ea5ff",
    dot_ball: "rgba(255,255,255,0.2)",
  };

  const sizeMap = {
    four: 8,
    six: 10,
    wicket: 10,
    bat_impact: 5,
    dot_ball: 4,
  };

  const currentPct = Math.min(100, Math.max(0, (currentTimeMs / videoDurationMs) * 100));

  return (
    <div className="event-timeline-bar" onClick={(e) => {
      const rect = e.currentTarget.getBoundingClientRect();
      const pct = (e.clientX - rect.left) / rect.width;
      onSeek?.(pct * videoDurationMs);
    }}>
      <div className="event-timeline-bar__track">
        <div className="event-timeline-bar__progress" style={{ width: `${currentPct}%` }} />
        {markers.map((m, i) => (
          <div
            key={i}
            className={`event-timeline-marker event-timeline-marker--${m.type}`}
            style={{
              left: `${m.position}%`,
              width: `${sizeMap[m.type] || 6}px`,
              height: `${sizeMap[m.type] || 6}px`,
              backgroundColor: colorMap[m.type] || "#8ea5ff",
            }}
            title={`${m.type.replace(/_/g, " ")} at ${(m.ts / 1000).toFixed(1)}s`}
          />
        ))}
      </div>
      <div className="event-timeline-bar__legend">
        <span className="event-legend-dot" style={{ background: "#4ade80" }} /> 4
        <span className="event-legend-dot" style={{ background: "#fbbf24" }} /> 6
        <span className="event-legend-dot" style={{ background: "#f87171" }} /> W
        <span className="event-legend-dot" style={{ background: "#8ea5ff" }} /> Shot
      </div>
    </div>
  );
}

function ScoreTicker({ scorecard, deliverySummary }) {
  if (!scorecard?.score) return null;
  return (
    <div className="score-ticker">
      <div className="score-ticker__main">
        <span className="score-ticker__score">{scorecard.score}</span>
        <span className="score-ticker__overs">{scorecard.overs} ov</span>
      </div>
      <div className="score-ticker__stats">
        <span>RR {scorecard.run_rate || 0}</span>
        {scorecard.partnership?.balls > 0 && (
          <span>P'ship {scorecard.partnership.runs}({scorecard.partnership.balls})</span>
        )}
      </div>
    </div>
  );
}

function BallTrajectoryOverlay({ trajectory, videoWidth, videoHeight }) {
  if (!trajectory?.length || trajectory.length < 3 || !videoWidth || !videoHeight) return null;

  const points = trajectory.map((p) => ({
    x: ((p.x || p[0] || 0) / videoWidth) * 100,
    y: ((p.y || p[1] || 0) / videoHeight) * 100,
  }));

  const visible = points.slice(-60);
  if (visible.length < 2) return null;

  const pathD = visible
    .map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(2)},${p.y.toFixed(2)}`)
    .join(" ");

  return (
    <svg className="ball-trajectory-overlay" viewBox="0 0 100 100" preserveAspectRatio="none">
      <defs>
        <linearGradient id="ball-trail" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="rgba(251,191,36,0)" />
          <stop offset="40%" stopColor="rgba(251,191,36,0.3)" />
          <stop offset="100%" stopColor="rgba(251,191,36,0.8)" />
        </linearGradient>
      </defs>
      <path d={pathD} fill="none" stroke="url(#ball-trail)" strokeWidth="0.4" strokeLinecap="round" />
      
      <circle
        cx={visible[visible.length - 1].x}
        cy={visible[visible.length - 1].y}
        r="0.6"
        fill="#fbbf24"
        opacity={0.9}
      >
        <animate attributeName="r" values="0.4;0.8;0.4" dur="1.2s" repeatCount="indefinite" />
      </circle>
    </svg>
  );
}

export default function VideoOverlayPlayer({ video, frames, cricket, onTimeUpdate }) {
  const videoRef = useRef(null);
  const stageRef = useRef(null);
  const [viewMode, setViewMode] = useState("overlay");
  const [currentFrame, setCurrentFrame] = useState(0);
  const [frameLookup, setFrameLookup] = useState({});
  const [captionsEnabled, setCaptionsEnabled] = useState(true);
  const [showTrajectory, setShowTrajectory] = useState(true);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);
  const [subtitleTrackUrl, setSubtitleTrackUrl] = useState("");
  const hasCricket = cricket && (
    cricket.timeline?.length > 0
    || cricket.events?.length > 0
    || cricket.deliveries?.length > 0
    || cricket.subtitles?.cues?.length > 0
  );
  const activeCaption = captionsEnabled && hasCricket ? getActiveCaption(cricket, currentTimeMs) : null;
  const subtitleTrackText = useMemo(() => buildSubtitleTrack(cricket), [cricket]);

  const handleTimeMs = useCallback(
    (ms) => {
      setCurrentTimeMs(ms);
      if (onTimeUpdate) onTimeUpdate(ms);
    },
    [onTimeUpdate]
  );

  useEffect(() => {
    function handleKeyDown(e) {
      if (!hasCricket || !videoRef.current) return;
      const events = cricket.events || [];
      if (!events.length) return;

      if (e.key === "ArrowRight" && e.shiftKey) {
        e.preventDefault();
        const next = events.find((ev) => (ev.timestamp_ms || 0) > currentTimeMs + 200);
        if (next) {
          const t = (next.timestamp_ms || 0) / 1000;
          videoRef.current.currentTime = Math.max(0, t - 0.5);
        }
      } else if (e.key === "ArrowLeft" && e.shiftKey) {
        e.preventDefault();
        const prev = [...events].reverse().find((ev) => (ev.timestamp_ms || 0) < currentTimeMs - 500);
        if (prev) {
          const t = (prev.timestamp_ms || 0) / 1000;
          videoRef.current.currentTime = Math.max(0, t - 0.5);
        }
      } else if (e.key === "c" && !e.ctrlKey && !e.metaKey) {
        if (document.activeElement?.tagName === "INPUT" || document.activeElement?.tagName === "TEXTAREA") return;
        setCaptionsEnabled((v) => !v);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [hasCricket, cricket, currentTimeMs]);

  useEffect(() => {
    const nextLookup = {};
    for (const frame of frames) {
      nextLookup[frame.frame_id] = frame;
    }
    setFrameLookup(nextLookup);
  }, [frames]);

  useEffect(() => {
    setCurrentFrame(0);
    if (videoRef.current) {
      videoRef.current.pause();
      videoRef.current.currentTime = 0;
    }
  }, [video.id, video.source_video_url, video.processed_video_url, frames]);

  useEffect(() => {
    if (!subtitleTrackText) {
      setSubtitleTrackUrl("");
      return undefined;
    }
    const blob = new Blob([subtitleTrackText], { type: "text/vtt" });
    const nextUrl = URL.createObjectURL(blob);
    setSubtitleTrackUrl(nextUrl);
    return () => URL.revokeObjectURL(nextUrl);
  }, [subtitleTrackText]);

  useEffect(() => {
    if (!videoRef.current) return;
    for (const track of Array.from(videoRef.current.textTracks || [])) {
      track.mode = captionsEnabled ? "showing" : "hidden";
    }
  }, [captionsEnabled, subtitleTrackUrl]);

  function resolveMediaUrl(url) {
    if (!url) return "";
    if (url.startsWith("http://") || url.startsWith("https://")) return url;
    return `${API_BASE_URL}${url}`;
  }

  async function handleFullscreen() {
    if (!stageRef.current?.requestFullscreen) return;
    try {
      await stageRef.current.requestFullscreen();
    } catch (_) {
      return;
    }
  }

  function seekToMs(ms) {
    if (videoRef.current) {
      videoRef.current.currentTime = ms / 1000;
      handleTimeMs(ms);
    }
  }

  const videoDurationMs = useMemo(() => {
    if (!video.fps || !video.frame_count) return 0;
    return (video.frame_count / video.fps) * 1000;
  }, [video.fps, video.frame_count]);

  const activeFrame = getClosestFrameData(frameLookup, currentFrame, video.fps || 25);
  const activeDetections = activeFrame?.detections || [];
  const frameTimestampMs =
    activeFrame?.timestamp_ms ?? Math.round((currentFrame / Math.max(video.fps || 25, 1)) * 1000);
  const visibleTypes = Object.entries(
    activeDetections.reduce((counts, detection) => {
      const key = detection.object_type || "unknown";
      return { ...counts, [key]: (counts[key] || 0) + 1 };
    }, {})
  )
    .sort((left, right) => right[1] - left[1])
    .map(([objectType, count]) => `${count} ${formatObjectLabel(objectType)}`)
    .join(" · ");

  const sourceUrl = resolveMediaUrl(
    viewMode === "processed" && video.processed_video_url
      ? video.processed_video_url
      : video.source_video_url
  );

  const scorecard = cricket?.scorecard || {};
  const deliverySummary = cricket?.delivery_summary || {};
  const profile = cricket?.profile || cricket?.profile_report?.profile || "generic";
  const subtitleMode = cricket?.subtitles?.mode || cricket?.speech?.mode || "event-led";
  const timelineEvents = Array.isArray(cricket?.deliveries) && cricket.deliveries.length
    ? cricket.deliveries.map((delivery) => ({
        event_type: delivery.result === "wicket" ? "wicket" : delivery.result === "six" ? "six" : delivery.result === "four" ? "four" : "bat_impact",
        timestamp_ms: delivery.ts_start,
      }))
    : cricket?.events;

  return (
    <section className="panel video-panel" data-aos="fade-up">
      <div className="video-panel__header">
        <div className="video-panel__title-block">
          <span className="section-kicker">Playback</span>
        </div>

        <div className="video-panel__toolbar">
          <div className="segmented-control">
            <button
              type="button"
              className={viewMode === "overlay" ? "is-active" : ""}
              onClick={() => setViewMode("overlay")}
            >
              Overlay
            </button>
            <button
              type="button"
              className={viewMode === "processed" ? "is-active" : ""}
              onClick={() => setViewMode("processed")}
            >
              Processed
            </button>
            {hasCricket && (
              <button
                type="button"
                className={captionsEnabled ? "is-active" : ""}
                onClick={() => setCaptionsEnabled((v) => !v)}
                title="Toggle captions (C)"
              >
                CC
              </button>
            )}
            {(cricket?.ball_path?.points?.length > 0 || cricket?.ball_trajectory?.length > 0) && (
              <button
                type="button"
                className={showTrajectory ? "is-active" : ""}
                onClick={() => setShowTrajectory((v) => !v)}
                title="Toggle ball trajectory"
              >
                Trail
              </button>
            )}
          </div>

          <button type="button" className="secondary-button secondary-button--small" onClick={handleFullscreen}>
            Fullscreen
          </button>
        </div>
      </div>

      <div
        ref={stageRef}
        className="video-stage"
        style={{
          aspectRatio: `${video.width || 16} / ${video.height || 9}`
        }}
      >
        {sourceUrl ? (
          <>
            <video
              key={sourceUrl}
              ref={videoRef}
              className="video-element"
              src={sourceUrl}
              controls
              playsInline
              onLoadedMetadata={(event) => {
                const t = event.currentTarget.currentTime;
                const nextFrame = Math.max(0, Math.floor(t * (video.fps || 25)));
                setCurrentFrame(nextFrame);
                handleTimeMs(Math.round(t * 1000));
              }}
              onTimeUpdate={(event) => {
                const t = event.currentTarget.currentTime;
                const nextFrame = Math.max(0, Math.floor(t * (video.fps || 25)));
                setCurrentFrame(nextFrame);
                handleTimeMs(Math.round(t * 1000));
              }}
            >
              {subtitleTrackUrl ? (
                <track
                  kind="captions"
                  srcLang="en"
                  label="VisionPlay captions"
                  src={subtitleTrackUrl}
                  default
                />
              ) : null}
            </video>

            {viewMode === "overlay" && video.width && video.height ? (
              <div className="overlay-layer">
                {activeDetections.map((detection, index) => {
                  const [x1, y1, x2, y2] = detection.bbox;
                  const width = x2 - x1;
                  const height = y2 - y1;
                  const palette = paletteForLabel(detection.object_type);

                  return (
                    <div
                      key={`${detection.object_type}-${detection.tracking_id}-${index}`}
                      className="overlay-box"
                      style={{
                        left: `${(x1 / video.width) * 100}%`,
                        top: `${(y1 / video.height) * 100}%`,
                        width: `${(width / video.width) * 100}%`,
                        height: `${(height / video.height) * 100}%`,
                        borderColor: palette.border,
                        boxShadow: `0 18px 42px ${palette.badge}30`
                      }}
                    >
                      <span
                        className="overlay-label"
                        style={{
                          backgroundColor: `${palette.badge}E6`,
                          borderColor: `${palette.border}55`
                        }}
                      >
                        {formatObjectLabel(detection.object_type)}
                        {detection.tracking_id != null ? ` · #${detection.tracking_id}` : ""}
                      </span>
                    </div>
                  );
                })}
              </div>
            ) : null}

            
            {showTrajectory && viewMode === "overlay" && (
              <BallTrajectoryOverlay
                trajectory={cricket?.ball_path?.points || cricket?.ball_trajectory}
                videoWidth={video.width}
                videoHeight={video.height}
              />
            )}

            
            {captionsEnabled && hasCricket && scorecard.score && (
              <ScoreTicker scorecard={scorecard} deliverySummary={deliverySummary} />
            )}

            
            {activeCaption && (
              <div
                key={`${activeCaption.text}-${activeCaption.over || ""}`}
                className={[
                  "video-caption",
                  `video-caption--${activeCaption.type}`,
                  (activeCaption.type === "four" || activeCaption.type === "six" || activeCaption.type === "wicket")
                    && "video-caption--pulse",
                ].filter(Boolean).join(" ")}
              >
                {activeCaption.score && (
                  <span className="video-caption__score">{activeCaption.score}</span>
                )}
                <div className="video-caption__content">
                  <span className="video-caption__text">{activeCaption.text}</span>
                  {activeCaption.detail && (
                    <span className="video-caption__detail">{activeCaption.detail}</span>
                  )}
                </div>
                {activeCaption.over && (
                  <span className="video-caption__over">{activeCaption.over}</span>
                )}
              </div>
            )}
          </>
        ) : (
          <div className="video-stage__empty">
            <strong>Video preview unavailable</strong>
            <p>VisionPlay could not resolve a playable media URL for this analysis.</p>
          </div>
        )}
      </div>

      
      {hasCricket && (
        <EventTimelineBar
          events={timelineEvents}
          videoDurationMs={videoDurationMs}
          currentTimeMs={currentTimeMs}
          onSeek={seekToMs}
        />
      )}

      <div className="video-panel__footer">
        <div className="video-meta-inline">
          <span>{activeFrame?.frame_id ?? currentFrame} frame</span>
          <span>{formatNumber(activeDetections.length)} objects</span>
          <span>{visibleTypes || "No labels"}</span>
          <span>{`${(frameTimestampMs / 1000).toFixed(2)}s`}</span>
          {hasCricket && <span>{formatObjectLabel(profile)}</span>}
          {hasCricket && <span>{subtitleMode === "speech-led" ? "Speech CC" : "Event CC"}</span>}
          <span>
            {formatNumber(video.width || 0)} × {formatNumber(video.height || 0)} ·{" "}
            {formatNumber(video.fps || 0)} fps
          </span>
          {hasCricket && (
            <span className="video-meta-shortcut">Shift+← → jump events · C toggle captions</span>
          )}
        </div>

        <div className="video-panel__footer-actions">
          {video.processed_video_url ? (
            <a
              className="secondary-button secondary-button--small"
              href={resolveMediaUrl(video.processed_video_url)}
              target="_blank"
              rel="noreferrer"
            >
              Open video
            </a>
          ) : null}
        </div>
      </div>
    </section>
  );
}
