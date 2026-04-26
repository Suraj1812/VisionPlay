import { useMemo, useState } from "react";
import { Target, TrendingUp, BarChart3, Crosshair, Award, Zap, Info } from "lucide-react";

function WagonWheel({ data = [] }) {
  const zones = useMemo(() => {
    const z = Array.from({ length: 8 }, () => ({ runs: 0, shots: 0, boundaries: 0 }));
    for (const d of data) {
      const idx = Math.max(0, Math.min(7, (d.zone || 1) - 1));
      z[idx].runs += d.runs || 0;
      z[idx].shots += 1;
      if (d.boundary || d.six) z[idx].boundaries += 1;
    }
    return z;
  }, [data]);

  const maxRuns = Math.max(1, ...zones.map((z) => z.runs));
  const cx = 120, cy = 120, maxR = 95;

  return (
    <div className="cricket-viz">
      <div className="cricket-viz__header">
        <Crosshair size={15} strokeWidth={2.2} />
        <span>Wagon Wheel</span>
      </div>
      <svg viewBox="0 0 240 240" className="wagon-wheel-svg">
        
        {[0.25, 0.5, 0.75, 1].map((r) => (
          <circle key={r} cx={cx} cy={cy} r={maxR * r}
            fill="none" stroke="currentColor" opacity={0.08} strokeWidth={0.8} />
        ))}
        
        {Array.from({ length: 8 }).map((_, i) => {
          const angle = (i * 45 - 112.5) * (Math.PI / 180);
          return (
            <line key={i}
              x1={cx} y1={cy}
              x2={cx + Math.cos(angle) * maxR}
              y2={cy + Math.sin(angle) * maxR}
              stroke="currentColor" opacity={0.06} strokeWidth={0.6}
            />
          );
        })}
        
        {zones.map((zone, i) => {
          const startAngle = (i * 45 - 112.5) * (Math.PI / 180);
          const endAngle = ((i + 1) * 45 - 112.5) * (Math.PI / 180);
          const r = maxR * Math.max(0.08, zone.runs / maxRuns);
          const x1 = cx + Math.cos(startAngle) * r;
          const y1 = cy + Math.sin(startAngle) * r;
          const x2 = cx + Math.cos(endAngle) * r;
          const y2 = cy + Math.sin(endAngle) * r;
          const fill = zone.boundaries > 0
            ? "rgba(74, 222, 128, 0.35)"
            : zone.runs > 0
              ? "rgba(142, 165, 255, 0.25)"
              : "rgba(255, 255, 255, 0.04)";

          return (
            <path key={i}
              d={`M${cx},${cy} L${x1},${y1} A${r},${r} 0 0,1 ${x2},${y2} Z`}
              fill={fill} stroke="rgba(255,255,255,0.12)" strokeWidth={0.6}
            />
          );
        })}
        
        {zones.map((zone, i) => {
          if (zone.runs === 0) return null;
          const midAngle = ((i + 0.5) * 45 - 112.5) * (Math.PI / 180);
          const labelR = maxR * Math.max(0.22, zone.runs / maxRuns) * 0.55;
          const lx = cx + Math.cos(midAngle) * labelR;
          const ly = cy + Math.sin(midAngle) * labelR;
          return (
            <text key={i} x={lx} y={ly}
              textAnchor="middle" dominantBaseline="central"
              fill="currentColor" fontSize="10" fontWeight="700" opacity={0.7}>
              {zone.runs}
            </text>
          );
        })}
        
        <circle cx={cx} cy={cy} r={4} fill="var(--accent)" opacity={0.5} />
      </svg>
      <div className="cricket-viz__legend">
        {data.length > 0 ? (
          <>
            <span>{data.length} shots</span>
            <span>{data.filter((d) => d.boundary || d.six).length} boundaries</span>
          </>
        ) : (
          <span className="muted-text">No shot data</span>
        )}
      </div>
    </div>
  );
}

function PitchMap({ data = [] }) {
  return (
    <div className="cricket-viz">
      <div className="cricket-viz__header">
        <Target size={15} strokeWidth={2.2} />
        <span>Pitch Map</span>
      </div>
      <svg viewBox="0 0 160 280" className="pitch-map-svg">
        
        <rect x="30" y="10" width="100" height="260" rx="4"
          fill="rgba(139, 169, 96, 0.12)" stroke="rgba(139, 169, 96, 0.2)" strokeWidth={0.8} />
        
        <line x1="35" y1="50" x2="125" y2="50" stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
        <line x1="35" y1="230" x2="125" y2="230" stroke="rgba(255,255,255,0.15)" strokeWidth={1} />
        
        {[70, 80, 90].map((x) => (
          <rect key={`t${x}`} x={x - 1} y="46" width={2} height={8} rx={1}
            fill="rgba(255,255,255,0.25)" />
        ))}
        {[70, 80, 90].map((x) => (
          <rect key={`b${x}`} x={x - 1} y="226" width={2} height={8} rx={1}
            fill="rgba(255,255,255,0.25)" />
        ))}
        
        {[
          { y: 60, h: 30, label: "Yorker", opacity: 0.04 },
          { y: 90, h: 35, label: "Full", opacity: 0.03 },
          { y: 125, h: 35, label: "Good", opacity: 0.04 },
          { y: 160, h: 35, label: "Short", opacity: 0.03 },
          { y: 195, h: 30, label: "Bouncer", opacity: 0.04 },
        ].map((zone) => (
          <g key={zone.label}>
            <rect x="30" y={zone.y} width="100" height={zone.h}
              fill={`rgba(255,255,255,${zone.opacity})`} />
            <text x="128" y={zone.y + zone.h / 2}
              textAnchor="start" dominantBaseline="central"
              fill="currentColor" fontSize="7" fontWeight="600" opacity={0.25}>
            </text>
          </g>
        ))}
        
        {data.map((d, i) => {
          if (!d.bx && !d.by) return null;
          const px = 30 + (d.bx / 1920) * 100;
          const py = 60 + (d.by / 1080) * 160;
          const color = d.wicket
            ? "rgba(248, 113, 113, 0.9)"
            : d.runs >= 4
              ? "rgba(74, 222, 128, 0.8)"
              : d.runs > 0
                ? "rgba(142, 165, 255, 0.7)"
                : "rgba(255, 255, 255, 0.4)";
          return (
            <circle key={i} cx={Math.min(Math.max(px, 34), 126)} cy={Math.min(Math.max(py, 62), 218)}
              r={d.wicket ? 5 : d.runs >= 4 ? 4.5 : 3.5}
              fill={color} opacity={0.85}
              stroke="rgba(0,0,0,0.3)" strokeWidth={0.5}
            />
          );
        })}
      </svg>
      <div className="cricket-viz__legend">
        {data.length > 0 ? (
          <>
            <span>{data.length} deliveries</span>
            <span className="pitch-legend-dot pitch-legend-dot--wicket" />
            <span className="pitch-legend-dot pitch-legend-dot--boundary" />
            <span className="pitch-legend-dot pitch-legend-dot--run" />
          </>
        ) : (
          <span className="muted-text">No pitch data</span>
        )}
      </div>
    </div>
  );
}

function MomentumGraph({ data = [] }) {
  if (data.length < 2) {
    return (
      <div className="cricket-viz cricket-viz--wide">
        <div className="cricket-viz__header">
          <TrendingUp size={15} strokeWidth={2.2} />
          <span>Run Momentum</span>
        </div>
        <div className="cricket-viz__empty">Not enough data for momentum graph</div>
      </div>
    );
  }

  const maxRuns = Math.max(1, ...data.map((d) => d.cum_runs));
  const w = 400, h = 120, pad = 8;
  const points = data.map((d, i) => {
    const x = pad + ((w - pad * 2) * i) / Math.max(data.length - 1, 1);
    const y = h - pad - ((h - pad * 2) * d.cum_runs) / maxRuns;
    return { x, y, d };
  });
  const pathD = points.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const areaD = `${pathD} L${points[points.length - 1].x.toFixed(1)},${h - pad} L${pad},${h - pad} Z`;

  return (
    <div className="cricket-viz cricket-viz--wide">
      <div className="cricket-viz__header">
        <TrendingUp size={15} strokeWidth={2.2} />
        <span>Run Momentum</span>
        <span className="cricket-viz__value">{maxRuns} runs</span>
      </div>
      <svg viewBox={`0 0 ${w} ${h}`} className="momentum-svg" preserveAspectRatio="none">
        <defs>
          <linearGradient id="momentum-fill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(142, 165, 255, 0.3)" />
            <stop offset="100%" stopColor="rgba(142, 165, 255, 0.02)" />
          </linearGradient>
        </defs>
        
        {[0.25, 0.5, 0.75].map((r) => (
          <line key={r}
            x1={pad} y1={h - pad - (h - pad * 2) * r}
            x2={w - pad} y2={h - pad - (h - pad * 2) * r}
            stroke="currentColor" opacity={0.06} strokeWidth={0.5}
          />
        ))}
        <path d={areaD} fill="url(#momentum-fill)" />
        <path d={pathD} fill="none" stroke="var(--accent)" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" />
        
        {points.map((p, i) =>
          p.d.this_ball >= 4 ? (
            <circle key={i} cx={p.x} cy={p.y} r={3.5}
              fill={p.d.this_ball >= 6 ? "#fbbf24" : "#4ade80"}
              stroke="rgba(0,0,0,0.3)" strokeWidth={0.5} />
          ) : null
        )}
      </svg>
    </div>
  );
}

function OverBreakdown({ data = [] }) {
  if (!data.length) return null;
  const maxRuns = Math.max(1, ...data.map((d) => d.runs));

  return (
    <div className="cricket-viz cricket-viz--wide">
      <div className="cricket-viz__header">
        <BarChart3 size={15} strokeWidth={2.2} />
        <span>Over-by-Over</span>
      </div>
      <div className="over-breakdown">
        {data.map((ov) => (
          <div key={ov.over} className="over-bar-group">
            <div className="over-bar-container">
              <div
                className={`over-bar ${ov.wickets > 0 ? "over-bar--wicket" : ov.runs >= 10 ? "over-bar--high" : ""}`}
                style={{ height: `${Math.max(8, (ov.runs / maxRuns) * 100)}%` }}
              >
                <span className="over-bar__value">{ov.runs}</span>
              </div>
            </div>
            <span className="over-bar__label">{ov.over}</span>
            {ov.wickets > 0 && <span className="over-bar__wicket">W</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function CricketDashboard({ cricket = {} }) {
  const [activeTab, setActiveTab] = useState("overview");

  const scorecard = cricket.scorecard || {};
  const wagonWheel = cricket.wagon_wheel || [];
  const pitchMap = cricket.pitch_map || [];
  const momentum = cricket.momentum || [];
  const overBreakdown = cricket.over_breakdown || [];
  const deliverySummary = cricket.delivery_summary || {};
  const teamAnalysis = cricket.team_analysis || {};
  const pressureIndex = cricket.pressure_index || [];
  const capabilities = cricket.capabilities || {};
  const warnings = cricket.warnings || [];
  const speech = cricket.speech || {};
  const profile = cricket.profile || cricket.profile_report?.profile || "generic";

  const hasData =
    scorecard.total_balls > 0 ||
    wagonWheel.length > 0 ||
    pitchMap.length > 0 ||
    momentum.length > 0 ||
    warnings.length > 0;

  if (!hasData) return null;

  const partnership = scorecard.partnership || {};
  const avgPressure =
    pressureIndex.length > 0
      ? Math.round(pressureIndex.reduce((s, p) => s + p.pressure_index, 0) / pressureIndex.length)
      : 0;

  return (
    <section className="panel cricket-dashboard" data-aos="fade-up">
      <div className="cricket-dashboard__header">
        <div className="commentary-panel__title-block">
          <span className="section-kicker">Cricket Intelligence</span>
          <h2>
            <Award size={18} strokeWidth={2.1} aria-hidden="true" />
            Match Analysis
          </h2>
        </div>

        <div className="cricket-dashboard__tabs">
          {[
            { key: "overview", label: "Overview" },
            { key: "visuals", label: "Visuals" },
            { key: "details", label: "Details" },
          ].map((tab) => (
            <button
              key={tab.key}
              type="button"
              className={`cricket-tab ${activeTab === tab.key ? "cricket-tab--active" : ""}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      
      {activeTab === "overview" && (
        <div className="cricket-dashboard__body">
          {(capabilities.score_source || warnings.length > 0) && (
            <div className="cricket-status-strip">
              {capabilities.score_source && (
                <span>
                  Source <strong>{capabilities.score_source === "scoreboard" ? "Scoreboard" : "Visual estimate"}</strong>
                </span>
              )}
              {profile && (
                <span>
                  Profile <strong>{profile === "cricket_end_on_action_cam_v1" ? "End-on cricket" : profile}</strong>
                </span>
              )}
              {capabilities.mode && (
                <span>
                  Mode <strong>{capabilities.mode === "specialized" ? "Specialized" : "Fallback"}</strong>
                </span>
              )}
              {capabilities.feed_source && (
                <span>
                  Feed <strong>{capabilities.feed_source === "delivery-timeline" ? "Delivery timeline" : capabilities.feed_source === "event-feed" ? "Event feed" : capabilities.feed_source === "scoreboard" ? "Scoreboard" : capabilities.feed_source}</strong>
                </span>
              )}
              <span>
                Captions <strong>{speech.mode === "speech-led" ? "Speech-led" : "Event-led"}</strong>
              </span>
              {warnings[0] && <span>{warnings[0]}</span>}
            </div>
          )}

          
          {scorecard.total_balls > 0 && (
            <div className="cricket-scorecard-strip">
              <div className="cricket-stat-card cricket-stat-card--hero">
                <span className="cricket-stat-card__label">Score</span>
                <strong>{scorecard.score || "0/0"}</strong>
                <span className="cricket-stat-card__sub">{scorecard.overs || "0.0"} overs</span>
              </div>
              <div className="cricket-stat-card">
                <span className="cricket-stat-card__label">Run Rate</span>
                <strong>{scorecard.run_rate || 0}</strong>
              </div>
              <div className="cricket-stat-card">
                <span className="cricket-stat-card__label">Boundaries</span>
                <strong>{(scorecard.fours || 0) + (scorecard.sixes || 0)}</strong>
                <span className="cricket-stat-card__sub">
                  {scorecard.fours || 0}×4 · {scorecard.sixes || 0}×6
                </span>
              </div>
              <div className="cricket-stat-card">
                <span className="cricket-stat-card__label">Dot Balls</span>
                <strong>{scorecard.dot_balls || 0}</strong>
                <span className="cricket-stat-card__sub">{scorecard.dot_pct || 0}%</span>
              </div>
              {partnership.balls > 0 && (
                <div className="cricket-stat-card">
                  <span className="cricket-stat-card__label">Partnership</span>
                  <strong>{partnership.runs}</strong>
                  <span className="cricket-stat-card__sub">{partnership.balls} balls</span>
                </div>
              )}
              {avgPressure > 0 && (
                <div className="cricket-stat-card">
                  <span className="cricket-stat-card__label">
                    <Zap size={12} /> Pressure
                  </span>
                  <strong>{avgPressure}%</strong>
                </div>
              )}
            </div>
          )}

          
          {deliverySummary.total_deliveries > 0 && (
            <div className="cricket-delivery-metrics">
              <span>
                <strong>{deliverySummary.total_deliveries}</strong> deliveries ·{" "}
                <strong>{deliverySummary.estimated_overs}</strong> overs
              </span>
              <span>Boundary% <strong>{deliverySummary.boundary_pct}%</strong></span>
              <span>Dot% <strong>{deliverySummary.dot_pct}%</strong></span>
              {deliverySummary.bat_impacts > 0 && (
                <span>{deliverySummary.bat_impacts} bat impacts</span>
              )}
            </div>
          )}

          
          <MomentumGraph data={momentum} />

          
          <OverBreakdown data={overBreakdown} />
        </div>
      )}

      
      {activeTab === "visuals" && (
        <div className="cricket-dashboard__body">
          <div className="cricket-viz-grid">
            <WagonWheel data={wagonWheel} />
            <PitchMap data={pitchMap} />
          </div>
          <MomentumGraph data={momentum} />
        </div>
      )}

      
      {activeTab === "details" && (
        <div className="cricket-dashboard__body">
          <OverBreakdown data={overBreakdown} />

          {(warnings.length > 0 || capabilities.supported_features?.length > 0) && (
            <div className="cricket-viz cricket-viz--wide">
              <div className="cricket-viz__header">
                <Info size={15} strokeWidth={2.2} />
                <span>Capability Notes</span>
              </div>
              <div className="cricket-capability-grid">
                {warnings.length > 0 && (
                  <div className="cricket-capability-list">
                    <span className="cricket-capability-list__label">Current limits</span>
                    {warnings.map((warning) => (
                      <span key={warning} className="cricket-capability-chip cricket-capability-chip--warning">
                        {warning}
                      </span>
                    ))}
                  </div>
                )}

                {capabilities.supported_features?.length > 0 && (
                  <div className="cricket-capability-list">
                    <span className="cricket-capability-list__label">What works now</span>
                    {capabilities.supported_features.map((feature) => (
                      <span key={feature} className="cricket-capability-chip">
                        {feature}
                      </span>
                    ))}
                  </div>
                )}

                {capabilities.unavailable_features?.length > 0 && (
                  <div className="cricket-capability-list">
                    <span className="cricket-capability-list__label">Still unavailable</span>
                    {capabilities.unavailable_features.map((feature) => (
                      <span key={feature} className="cricket-capability-chip cricket-capability-chip--muted">
                        {feature}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          
          {teamAnalysis.teams && Object.keys(teamAnalysis.teams).length > 0 && (
            <div className="cricket-viz cricket-viz--wide">
              <div className="cricket-viz__header">
                <Target size={15} strokeWidth={2.2} />
                <span>Team Analysis</span>
              </div>
              <div className="cricket-team-list">
                {Object.entries(teamAnalysis.teams).map(([teamId, info]) => (
                  <div key={teamId} className="cricket-team-item">
                    <div
                      className="cricket-team-swatch"
                      style={{ background: info.color || "var(--muted)" }}
                    />
                    <span className="cricket-team-name">
                      {info.label || `Team ${teamId}`}
                    </span>
                    <span className="cricket-team-count">
                      {info.player_count || 0} players
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          
          {pressureIndex.length > 3 && (
            <div className="cricket-viz cricket-viz--wide">
              <div className="cricket-viz__header">
                <Zap size={15} strokeWidth={2.2} />
                <span>Pressure Index</span>
              </div>
              <svg viewBox="0 0 400 80" className="pressure-svg" preserveAspectRatio="none">
                <defs>
                  <linearGradient id="pressure-fill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="rgba(248, 113, 113, 0.4)" />
                    <stop offset="100%" stopColor="rgba(248, 113, 113, 0.02)" />
                  </linearGradient>
                </defs>
                {(() => {
                  const pts = pressureIndex.map((p, i) => ({
                    x: 8 + ((400 - 16) * i) / Math.max(pressureIndex.length - 1, 1),
                    y: 72 - ((72 - 8) * p.pressure_index) / 100,
                  }));
                  const line = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
                  const area = `${line} L${pts[pts.length - 1].x.toFixed(1)},72 L8,72 Z`;
                  return (
                    <>
                      <path d={area} fill="url(#pressure-fill)" />
                      <path d={line} fill="none" stroke="#f87171" strokeWidth={1.8} strokeLinecap="round" />
                    </>
                  );
                })()}
              </svg>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
