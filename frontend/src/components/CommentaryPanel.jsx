import { useEffect, useMemo, useRef, useState } from "react";
import { MessageCircle, ChevronDown, ChevronUp } from "lucide-react";

function formatTimestamp(ms) {
  if (!ms || ms <= 0) return "0:00";
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

function getBallBadge(entry) {
  if (entry.wicket) return { label: "W", className: "commentary-badge--wicket" };
  if (entry.six) return { label: "6", className: "commentary-badge--six" };
  if (entry.boundary) return { label: "4", className: "commentary-badge--four" };
  if (entry.dot) return { label: "•", className: "commentary-badge--dot" };
  if (entry.runs === 1) return { label: "1", className: "commentary-badge--run" };
  if (entry.runs === 2) return { label: "2", className: "commentary-badge--run" };
  if (entry.runs === 3) return { label: "3", className: "commentary-badge--run" };
  if (entry.runs > 0) return { label: String(entry.runs), className: "commentary-badge--run" };
  return { label: "•", className: "commentary-badge--dot" };
}

function getShotIcon(shot) {
  const icons = {
    cover_drive: "🏏",
    drive: "🏏",
    lofted_drive: "🚀",
    cut: "⚔️",
    pull: "💪",
    sweep: "🧹",
    reverse_sweep: "🔄",
    flick: "👋",
    upper_cut: "⬆️",
    defensive: "🛡️",
  };
  return icons[shot] || "";
}

function ThisOverStrip({ timeline }) {
  const currentOverBalls = useMemo(() => {
    if (!timeline.length) return [];
    const lastEntry = timeline[timeline.length - 1];
    const lastOverNum = lastEntry.over ? lastEntry.over.split(".")[0] : null;
    if (lastOverNum === null) return [];
    return timeline.filter((e) => e.over && e.over.split(".")[0] === lastOverNum);
  }, [timeline]);

  if (!currentOverBalls.length) return null;

  const overNum = Number(currentOverBalls[0].over.split(".")[0]) + 1;
  const overRuns = currentOverBalls.reduce((s, b) => s + (b.runs || 0), 0);

  return (
    <div className="this-over-strip">
      <span className="this-over-strip__label">Over {overNum}</span>
      <div className="this-over-strip__balls">
        {currentOverBalls.map((ball, i) => {
          const badge = getBallBadge(ball);
          return (
            <span key={i} className={`this-over-dot ${badge.className}`}>
              {badge.label}
            </span>
          );
        })}
        
        {Array.from({ length: Math.max(0, 6 - currentOverBalls.length) }).map((_, i) => (
          <span key={`e${i}`} className="this-over-dot this-over-dot--empty" />
        ))}
      </div>
      <span className="this-over-strip__runs">{overRuns} runs</span>
    </div>
  );
}

function RecentOversChart({ overBreakdown }) {
  const recent = useMemo(() => {
    if (!overBreakdown?.length) return [];
    return overBreakdown.slice(-8);
  }, [overBreakdown]);

  if (recent.length < 2) return null;

  const maxRuns = Math.max(1, ...recent.map((o) => o.runs));

  return (
    <div className="recent-overs-chart">
      <span className="recent-overs-chart__label">Last {recent.length} overs</span>
      <div className="recent-overs-chart__bars">
        {recent.map((ov) => (
          <div key={ov.over} className="recent-overs-chart__bar-wrap">
            <div
              className={[
                "recent-overs-chart__bar",
                ov.wickets > 0 && "recent-overs-chart__bar--wicket",
                ov.runs >= 10 && "recent-overs-chart__bar--high",
              ].filter(Boolean).join(" ")}
              style={{ height: `${Math.max(12, (ov.runs / maxRuns) * 100)}%` }}
            />
            <span className="recent-overs-chart__val">{ov.runs}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function CommentaryPanel({
  cricket = {},
  currentTimeMs = 0,
  onSeekTo,
}) {
  const [expanded, setExpanded] = useState(true);
  const [autoScroll, setAutoScroll] = useState(true);
  const feedRef = useRef(null);
  const activeRef = useRef(null);

  const timeline = cricket.timeline || [];
  const scorecard = cricket.scorecard || {};
  const commentary = cricket.commentary || [];
  const overBreakdown = cricket.over_breakdown || [];
  const warnings = cricket.warnings || [];
  const mode = cricket.mode || "fallback";
  const feedSource = cricket.capabilities?.feed_source || "unavailable";
  const speech = cricket.speech || {};
  const isSpeechLed = cricket.subtitles?.mode === "speech-led" || speech.mode === "speech-led";
  const hasData = timeline.length > 0 || commentary.length > 0;

  const activeIndex = timeline.findIndex(
    (entry) => currentTimeMs >= entry.ts_start && currentTimeMs <= (entry.ts_end || entry.ts_start + 5000)
  );

  useEffect(() => {
    if (autoScroll && activeRef.current && feedRef.current) {
      activeRef.current.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [activeIndex, autoScroll]);

  if (!hasData) {
    return null;
  }

  return (
    <section className="panel commentary-panel" data-aos="fade-up">
      <div className="commentary-panel__header">
        <div className="commentary-panel__title-block">
          <span className="section-kicker">{isSpeechLed ? "Caption Feed" : mode === "specialized" ? "Delivery Feed" : "Match Feed"}</span>
          <h2>
            <MessageCircle size={18} strokeWidth={2.1} aria-hidden="true" />
            {isSpeechLed ? "Timed Captions" : mode === "specialized" ? "Cricket Feed" : "Match Feed"}
          </h2>
        </div>

        <div className="commentary-panel__controls">
          {scorecard.score && (
            <div className="commentary-scorecard">
              <span className="commentary-scorecard__score">{scorecard.score}</span>
              <span className="commentary-scorecard__overs">({scorecard.overs} ov)</span>
              {scorecard.run_rate > 0 && (
                <span className="commentary-scorecard__rr">RR {scorecard.run_rate}</span>
              )}
            </div>
          )}

          <button
            type="button"
            className="secondary-button secondary-button--small"
            onClick={() => setAutoScroll((v) => !v)}
            title={autoScroll ? "Disable auto-scroll" : "Enable auto-scroll"}
          >
            {autoScroll ? "Auto" : "Manual"}
          </button>

          <button
            type="button"
            className="secondary-button secondary-button--small commentary-panel__toggle"
            onClick={() => setExpanded((v) => !v)}
            aria-label={expanded ? "Collapse commentary" : "Expand commentary"}
          >
            {expanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
          </button>
        </div>
      </div>

      {expanded && (
        <div className="commentary-panel__notice">
          <span>
            {isSpeechLed
              ? "Captions are aligned from real detected speech and delivery windows."
              : mode === "specialized"
                ? "Feed follows specialized delivery analysis for end-on cricket clips."
                : "Feed follows fallback visual cricket analysis."}
          </span>
          {warnings[0] ? <span>{warnings[0]}</span> : null}
        </div>
      )}

      
      {expanded && timeline.length > 0 && mode === "specialized" && (
        <div className="commentary-panel__ticker">
          <ThisOverStrip timeline={timeline} />
          <RecentOversChart overBreakdown={overBreakdown} />
        </div>
      )}

      {expanded && timeline.length > 0 && mode !== "specialized" && overBreakdown.length > 1 && (
        <div className="commentary-panel__ticker">
          <RecentOversChart overBreakdown={overBreakdown} />
        </div>
      )}

      {expanded && (
        <div className="commentary-feed" ref={feedRef}>
          {timeline.length > 0 ? (
            (() => {
              let lastOver = null;
              return timeline.map((entry, index) => {
                const badge = getBallBadge(entry);
                const isActive = index === activeIndex;
                const isPast = entry.ts_end < currentTimeMs;
                const shotIcon = getShotIcon(entry.shot);
                const currentOver = entry.over ? entry.over.split(".")[0] : null;
                const showOverHeader = currentOver !== lastOver;
                if (showOverHeader) lastOver = currentOver;

                let overRuns = 0;
                let overWickets = 0;
                let overBalls = [];
                if (showOverHeader) {
                  for (const e of timeline) {
                    if (e.over && e.over.split(".")[0] === currentOver) {
                      overRuns += e.runs || 0;
                      if (e.wicket) overWickets += 1;
                      overBalls.push(e);
                    }
                  }
                }

                return (
                  <div key={`${entry.ball}-${entry.over}`}>
                    {showOverHeader && currentOver !== null && (
                      <div className="commentary-over-header">
                        <span className="commentary-over-header__label">
                          Over {Number(currentOver) + 1}
                        </span>
                        <div className="commentary-over-header__balls">
                          {overBalls.map((b, bi) => {
                            const bb = getBallBadge(b);
                            return (
                              <span key={bi} className={`over-header-dot ${bb.className}`}>
                                {bb.label}
                              </span>
                            );
                          })}
                        </div>
                        <span className="commentary-over-header__stats">
                          {overRuns} runs{overWickets > 0 ? ` · ${overWickets}W` : ""}
                        </span>
                        <div className="commentary-over-header__line" />
                      </div>
                    )}
                    <article
                      ref={isActive ? activeRef : undefined}
                      className={[
                        "commentary-entry",
                        isActive && "commentary-entry--active",
                        isPast && !isActive && "commentary-entry--past",
                        entry.wicket && "commentary-entry--wicket",
                        entry.six && "commentary-entry--six",
                        entry.boundary && !entry.six && "commentary-entry--four",
                      ].filter(Boolean).join(" ")}
                      onClick={() => onSeekTo?.(entry.ts_start)}
                      role="button"
                      tabIndex={0}
                      onKeyDown={(e) => e.key === "Enter" && onSeekTo?.(entry.ts_start)}
                    >
                      <div className="commentary-entry__gutter">
                        <span className={`commentary-badge ${badge.className}`}>
                          {badge.label}
                        </span>
                        <span className="commentary-entry__over">{entry.over}</span>
                      </div>

                      <div className="commentary-entry__body">
                        <p className="commentary-entry__text">
                          {shotIcon && <span className="commentary-entry__shot-icon">{shotIcon}</span>}
                          {entry.commentary || `Over ${entry.over}`}
                        </p>
                        <div className="commentary-entry__meta">
                          {entry.length !== "unknown" && (
                            <span className="commentary-chip">{entry.length.replace(/_/g, " ")}</span>
                          )}
                          {entry.line !== "unknown" && (
                            <span className="commentary-chip">{entry.line.replace(/_/g, " ")}</span>
                          )}
                          {entry.shot !== "unknown" && (
                            <span className="commentary-chip commentary-chip--shot">
                              {entry.shot.replace(/_/g, " ")}
                            </span>
                          )}
                          {entry.zone > 0 && mode === "specialized" && (
                            <span className="commentary-chip commentary-chip--zone">
                              Zone {entry.zone}
                            </span>
                          )}
                          {entry.score && (
                            <span className="commentary-chip commentary-chip--score">{entry.score}</span>
                          )}
                          {entry.detail && (
                            <span className="commentary-chip commentary-chip--detail">{entry.detail}</span>
                          )}
                        </div>
                      </div>

                      <span className="commentary-entry__timestamp">
                        {formatTimestamp(entry.ts_start)}
                      </span>
                    </article>
                  </div>
                );
              });
            })()
          ) : (
            
            commentary.map((text, index) => (
              <article key={index} className="commentary-entry commentary-entry--flat">
                <div className="commentary-entry__gutter">
                  <span className="commentary-badge commentary-badge--dot">
                    <MessageCircle size={12} />
                  </span>
                </div>
                <div className="commentary-entry__body">
                  <p className="commentary-entry__text">{text}</p>
                </div>
              </article>
            ))
          )}
        </div>
      )}

      {expanded && scorecard.total_balls > 0 && (
        <div className="commentary-panel__footer">
          <div className="commentary-stats-bar">
            {scorecard.fours > 0 && <span>{scorecard.fours} fours</span>}
            {scorecard.sixes > 0 && <span>{scorecard.sixes} sixes</span>}
            {scorecard.dot_balls > 0 && <span>{scorecard.dot_balls} dots</span>}
            {scorecard.boundary_pct > 0 && (
              <span>Boundary% {scorecard.boundary_pct}%</span>
            )}
            {scorecard.dot_pct > 0 && (
              <span>Dot% {scorecard.dot_pct}%</span>
            )}
          </div>
        </div>
      )}
    </section>
  );
}
