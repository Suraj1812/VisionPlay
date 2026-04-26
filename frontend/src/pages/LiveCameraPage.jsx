import { useEffect, useMemo, useRef, useState } from "react";
import { Camera, Play, Square } from "lucide-react";
import { detectLiveFrame } from "../api";
import { formatNumber, formatObjectLabel } from "../utils/formatters";
import { SkeletonBlock } from "../components/Skeleton";

const TARGET_CAPTURE_INTERVAL_MS = 110;
const MAX_CAPTURE_WIDTH = 1280;
const MAX_VISIBLE_OVERLAYS = 3;
const DERIVED_LIVE_LABELS = new Set(["phone_like_device", "handheld_item", "face", "hand"]);

const OVERLAY_PALETTE = [
  { border: "#4F8A41", badge: "#2F6B3F" },
  { border: "#6FA8DC", badge: "#355f85" },
  { border: "#D9A441", badge: "#7A5C3E" },
  { border: "#A8C3A0", badge: "#476b4a" },
  { border: "#7A5C3E", badge: "#5d4631" },
  { border: "#86B86B", badge: "#3e6a3a" }
];

function createSessionId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }

  return `live-${Date.now()}-${Math.round(Math.random() * 1_000_000)}`;
}

function paletteForLabel(label) {
  const source = String(label || "unknown");
  let hash = 0;
  for (let index = 0; index < source.length; index += 1) {
    hash = source.charCodeAt(index) + ((hash << 5) - hash);
  }

  return OVERLAY_PALETTE[Math.abs(hash) % OVERLAY_PALETTE.length];
}

function formatOverlayMeta(objectType, details) {
  if (!details) {
    return "";
  }

  if (
    objectType === "face" &&
    details.emotion &&
    !["none", "unknown"].includes(String(details.emotion).toLowerCase())
  ) {
    return ` · ${formatObjectLabel(details.emotion)}`;
  }

  if (
    objectType === "hand" &&
    details.gesture &&
    !["none", "unknown"].includes(String(details.gesture).toLowerCase())
  ) {
    return ` · ${formatObjectLabel(details.gesture)}`;
  }

  if (objectType === "hand" && details.handedness) {
    return ` · ${formatObjectLabel(details.handedness)}`;
  }

  return "";
}

function shouldShowTrackingId(item) {
  return item?.tracking_id != null && !DERIVED_LIVE_LABELS.has(item.object_type);
}

function getOverlayPriority(item, liveResult) {
  const isFocus =
    item.object_type === liveResult?.focus_label &&
    (item.tracking_id ?? null) === (liveResult?.focus_tracking_id ?? null);

  if (isFocus) {
    return 100;
  }

  const priorities = {
    phone_like_device: 90,
    handheld_item: 84,
    face: 76,
    laptop: 72,
    bottle: 66,
    book: 64,
    person: 52,
    hand: 40
  };

  return priorities[item.object_type] || 48;
}

export default function LiveCameraPage() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const captureTimerRef = useRef(null);
  const reconnectTimerRef = useRef(null);
  const previewMonitorRef = useRef(null);
  const requestInFlightRef = useRef(false);
  const captureInProgressRef = useRef(false);
  const queuedFrameRef = useRef(null);
  const requestControllerRef = useRef(null);
  const pipelineNonceRef = useRef(0);
  const reconnectAttemptsRef = useRef(0);
  const sessionIdRef = useRef(createSessionId());
  const mountedRef = useRef(true);

  const [cameraState, setCameraState] = useState("loading");
  const [cameraError, setCameraError] = useState("");
  const [liveResult, setLiveResult] = useState(null);
  const [previewFps, setPreviewFps] = useState(0);

  const objects = liveResult?.objects || [];
  const metrics = liveResult?.metrics || {};
  const reactions = liveResult?.reactions?.length ? liveResult.reactions : ["Camera active"];
  const liveGestureLabel =
    metrics.gesture
      ? formatObjectLabel(metrics.gesture)
      : metrics.handedness
        ? `${formatObjectLabel(metrics.handedness)} hand`
        : null;
  const objectCountEntries = useMemo(
    () => Object.entries(liveResult?.object_counts || {}).sort((left, right) => right[1] - left[1]),
    [liveResult]
  );
  const focusLabel = liveResult?.focus_label ? formatObjectLabel(liveResult.focus_label) : "Scanning scene";
  const focusTrackingId = liveResult?.focus_tracking_id ?? null;
  const focusObject = useMemo(
    () =>
      objects.find(
        (item) =>
          item.object_type === liveResult?.focus_label &&
          (item.tracking_id ?? null) === (liveResult?.focus_tracking_id ?? null)
      ) || null,
    [liveResult, objects]
  );
  const overlayObjects = useMemo(() => {
    const derivedSignalsPresent = objects.some((item) =>
      ["phone_like_device", "handheld_item", "face"].includes(item.object_type)
    );
    const sorted = [...objects]
      .filter((item) => !(item.is_predicted && item.confidence < 0.46))
      .filter((item) => {
        if (item.object_type !== "hand") {
          return true;
        }

        const gesture = String(item.details?.gesture || "").trim().toLowerCase();
        return (
          (gesture && !["none", "unknown"].includes(gesture)) ||
          Boolean(item.details?.handedness) ||
          (item.confidence || 0) >= 0.72
        );
      })
      .filter((item) => !(derivedSignalsPresent && item.object_type === "person"))
      .sort((left, right) => {
        const priorityDelta = getOverlayPriority(right, liveResult) - getOverlayPriority(left, liveResult);
        if (priorityDelta !== 0) {
          return priorityDelta;
        }

        const areaLeft = Math.max((left.bbox[2] - left.bbox[0]) * (left.bbox[3] - left.bbox[1]), 0);
        const areaRight = Math.max((right.bbox[2] - right.bbox[0]) * (right.bbox[3] - right.bbox[1]), 0);
        if (areaRight !== areaLeft) {
          return areaRight - areaLeft;
        }

        return (right.confidence || 0) - (left.confidence || 0);
      });

    const selected = [];
    let faceCount = 0;
    let handCount = 0;

    for (const item of sorted) {
      if (item.object_type === "face") {
        if (faceCount >= 1) {
          continue;
        }
        faceCount += 1;
      }

      if (item.object_type === "hand") {
        if (handCount >= 1) {
          continue;
        }
        handCount += 1;
      }

      selected.push(item);
      if (selected.length >= MAX_VISIBLE_OVERLAYS) {
        break;
      }
    }

    return selected;
  }, [liveResult, objects]);
  const summaryItemsLabel = objectCountEntries.length
    ? objectCountEntries
        .slice(0, 3)
        .map(([objectType, count]) => `${formatNumber(count)} ${formatObjectLabel(objectType)}`)
        .join(" · ")
    : "No stable objects";

  useEffect(() => {
    mountedRef.current = true;
    window.scrollTo(0, 0);
    startCamera();

    return () => {
      mountedRef.current = false;
      requestInFlightRef.current = false;
      captureInProgressRef.current = false;
      queuedFrameRef.current = null;
      if (requestControllerRef.current) {
        requestControllerRef.current.abort();
        requestControllerRef.current = null;
      }
      stopLoopingWork();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  function stopLoopingWork() {
    if (captureTimerRef.current) {
      window.clearTimeout(captureTimerRef.current);
      captureTimerRef.current = null;
    }
    if (reconnectTimerRef.current) {
      window.clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (previewMonitorRef.current) {
      window.cancelAnimationFrame(previewMonitorRef.current);
      previewMonitorRef.current = null;
    }
  }

  function startPreviewMonitor() {
    if (!mountedRef.current) {
      return;
    }

    if (previewMonitorRef.current) {
      window.cancelAnimationFrame(previewMonitorRef.current);
    }

    let frameCount = 0;
    let windowStartedAt = performance.now();

    const tick = (timestamp) => {
      if (!mountedRef.current) {
        return;
      }

      frameCount += 1;
      const elapsed = timestamp - windowStartedAt;
      if (elapsed >= 1000) {
        setPreviewFps(Math.max(1, Math.round((frameCount * 1000) / elapsed)));
        frameCount = 0;
        windowStartedAt = timestamp;
      }

      previewMonitorRef.current = window.requestAnimationFrame(tick);
    };

    previewMonitorRef.current = window.requestAnimationFrame(tick);
  }

  function scheduleNextCapture(delayMs = TARGET_CAPTURE_INTERVAL_MS) {
    if (!mountedRef.current) {
      return;
    }

    if (captureTimerRef.current) {
      window.clearTimeout(captureTimerRef.current);
    }

    captureTimerRef.current = window.setTimeout(() => {
      void captureFrame();
    }, delayMs);
  }

  function scheduleReconnect(message) {
    if (!mountedRef.current) {
      return;
    }

    if (reconnectAttemptsRef.current >= 1) {
      setCameraState("error");
      setCameraError(message);
      return;
    }

    reconnectAttemptsRef.current += 1;
    setCameraState("loading");
    setCameraError(message);
    reconnectTimerRef.current = window.setTimeout(() => {
      if (mountedRef.current) {
        void startCamera();
      }
    }, 700);
  }

  function flushQueuedFrame() {
    if (requestInFlightRef.current) {
      return;
    }

    const nextFrame = queuedFrameRef.current;
    if (!nextFrame || nextFrame.pipelineNonce !== pipelineNonceRef.current) {
      queuedFrameRef.current = null;
      return;
    }

    queuedFrameRef.current = null;
    void sendFrame(nextFrame);
  }

  async function sendFrame(frameJob) {
    requestInFlightRef.current = true;
    const controller = new AbortController();
    requestControllerRef.current = controller;

    try {
      const response = await detectLiveFrame(frameJob.blob, frameJob.sessionId, {
        signal: controller.signal
      });
      if (
        !mountedRef.current ||
        frameJob.pipelineNonce !== pipelineNonceRef.current ||
        frameJob.sessionId !== sessionIdRef.current
      ) {
        return;
      }

      setLiveResult(response);
      setCameraError("");
    } catch (error) {
      if (error?.name === "AbortError") {
        return;
      }

      if (mountedRef.current && frameJob.pipelineNonce === pipelineNonceRef.current) {
        setCameraError(error.message || "Live detection failed.");
      }
    } finally {
      if (requestControllerRef.current === controller) {
        requestControllerRef.current = null;
      }
      requestInFlightRef.current = false;
      flushQueuedFrame();
    }
  }

  async function startCamera() {
    if (!navigator.mediaDevices?.getUserMedia) {
      if (mountedRef.current) {
        setCameraState("error");
        setCameraError("This browser does not support live camera capture.");
      }
      return;
    }

    pipelineNonceRef.current += 1;
    stopLoopingWork();
    captureInProgressRef.current = false;
    queuedFrameRef.current = null;
    requestInFlightRef.current = false;
    reconnectAttemptsRef.current = 0;
    if (requestControllerRef.current) {
      requestControllerRef.current.abort();
      requestControllerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    sessionIdRef.current = createSessionId();

    setLiveResult(null);
    setPreviewFps(0);
    setCameraState("loading");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: {
          width: { ideal: 1280 },
          height: { ideal: 720 },
          facingMode: "user"
        }
      });

      if (!mountedRef.current) {
        stream.getTracks().forEach((track) => track.stop());
        return;
      }

      const pipelineNonce = pipelineNonceRef.current;
      for (const track of stream.getVideoTracks()) {
        track.onended = () => {
          if (mountedRef.current && pipelineNonce === pipelineNonceRef.current) {
            scheduleReconnect("Camera disconnected. Reconnecting...");
          }
        };
      }

      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setCameraError("");
      setCameraState("ready");
      startPreviewMonitor();
      scheduleNextCapture(0);
    } catch (error) {
      if (mountedRef.current) {
        setCameraState("error");
        setCameraError(error.message || "Unable to start live camera mode.");
      }
    }
  }

  async function captureFrame() {
    if (captureInProgressRef.current) {
      scheduleNextCapture(24);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas || video.readyState < 2 || !video.videoWidth || !video.videoHeight) {
      scheduleNextCapture(140);
      return;
    }

    captureInProgressRef.current = true;
    const startedAt = performance.now();
    const pipelineNonce = pipelineNonceRef.current;

    try {
      const scale = Math.min(1, MAX_CAPTURE_WIDTH / video.videoWidth);
      canvas.width = Math.round(video.videoWidth * scale);
      canvas.height = Math.round(video.videoHeight * scale);

      const context = canvas.getContext("2d", { alpha: false });
      if (!context) {
        throw new Error("Unable to capture the live camera frame.");
      }

      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise((resolve) => {
        canvas.toBlob(resolve, "image/jpeg", 0.78);
      });

      if (!blob) {
        throw new Error("Unable to compress the live camera frame.");
      }

      const frameJob = {
        blob,
        pipelineNonce,
        sessionId: sessionIdRef.current
      };

      if (requestInFlightRef.current) {
        queuedFrameRef.current = frameJob;
      } else {
        void sendFrame(frameJob);
      }
    } catch (error) {
      if (mountedRef.current) {
        setCameraError(error.message || "Live detection failed.");
      }
    } finally {
      captureInProgressRef.current = false;
      const elapsed = performance.now() - startedAt;
      scheduleNextCapture(Math.max(0, TARGET_CAPTURE_INTERVAL_MS - elapsed));
    }
  }

  function stopCamera() {
    pipelineNonceRef.current += 1;
    stopLoopingWork();
    captureInProgressRef.current = false;
    queuedFrameRef.current = null;
    requestInFlightRef.current = false;
    if (requestControllerRef.current) {
      requestControllerRef.current.abort();
      requestControllerRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    setCameraError("");
    setLiveResult(null);
    setCameraState("stopped");
  }

  const statusTone =
    cameraState === "ready" ? "completed" : cameraState === "error" ? "failed" : "pending";

  return (
    <div className="page-stack">
      <section className="panel live-camera-panel" data-aos="fade-up">
        <div className="live-camera-panel__header">
          <div className="live-camera-panel__title-block">
            <span className="eyebrow-pill">Live camera</span>
            <h1>Realtime detection</h1>
          </div>

          <div className="live-camera-panel__actions">
            <span className={`status-pill status-pill--${statusTone}`}>
              {cameraState === "ready"
                ? "Live"
                : cameraState === "stopped"
                  ? "Stopped"
                  : cameraState === "error"
                    ? "Error"
                    : "Starting"}
            </span>
            <div className="live-camera-panel__stat">
              <span className="metric-label">Preview</span>
              <strong>{previewFps || "--"} FPS</strong>
            </div>
            <div className="live-camera-panel__stat">
              <span className="metric-label">Inference</span>
              <strong>{metrics.inference_fps || "--"} FPS</strong>
            </div>
            {cameraState === "ready" ? (
              <button className="secondary-button secondary-button--icon" type="button" onClick={stopCamera}>
                <Square size={16} strokeWidth={2.1} aria-hidden="true" />
                Stop
              </button>
            ) : (
              <button className="primary-button primary-button--icon" type="button" onClick={startCamera}>
                <Play size={16} strokeWidth={2.1} aria-hidden="true" />
                Start
              </button>
            )}
          </div>
        </div>

        <div className="live-camera-stage" style={{ aspectRatio: "16 / 9" }}>
          <video ref={videoRef} className="video-element" muted playsInline autoPlay />
          <canvas ref={canvasRef} className="live-camera-stage__canvas" />

          {cameraState === "ready" ? (
            <>
              <div className="overlay-layer">
                {overlayObjects.map((item, index) => {
                  const palette = paletteForLabel(item.object_type);
                  const frameWidth = liveResult?.frame_width || 1;
                  const frameHeight = liveResult?.frame_height || 1;
                  const [x1, y1, x2, y2] = item.bbox;
                  const isFocus =
                    item.object_type === liveResult?.focus_label &&
                    (item.tracking_id ?? null) === (liveResult?.focus_tracking_id ?? null);

                  return (
                    <div
                      key={`${item.object_type}-${item.tracking_id}-${index}`}
                      className={`overlay-box ${isFocus ? "overlay-box--focus" : ""}`}
                      style={{
                        left: `${(x1 / frameWidth) * 100}%`,
                        top: `${(y1 / frameHeight) * 100}%`,
                        width: `${((x2 - x1) / frameWidth) * 100}%`,
                        height: `${((y2 - y1) / frameHeight) * 100}%`,
                        borderColor: palette.border,
                        boxShadow: `0 0 0 1px rgba(255, 255, 255, 0.16), 0 12px 28px ${palette.badge}40`
                      }}
                    >
                      <span
                        className="overlay-label"
                        style={{
                          backgroundColor: `${palette.badge}EE`,
                          borderColor: `${palette.border}66`
                        }}
                      >
                        {formatObjectLabel(item.object_type)}
                        {shouldShowTrackingId(item) ? ` · #${item.tracking_id}` : ""}
                        {formatOverlayMeta(item.object_type, item.details)}
                      </span>
                    </div>
                  );
                })}
              </div>

              <div className="live-camera-stage__hud">
                <span>Latency {metrics.processing_latency_ms || "--"} ms</span>
                <span>Aux {metrics.auxiliary_vision_enabled ? "on" : "off"}</span>
                {metrics.emotion ? <span>Emotion {formatObjectLabel(metrics.emotion)}</span> : null}
                {liveGestureLabel ? <span>Gesture {liveGestureLabel}</span> : null}
              </div>
            </>
          ) : null}

          {cameraState !== "ready" ? (
            <div className="live-camera-stage__empty">
              {cameraState === "loading" ? (
                <div className="live-camera-stage__placeholder" aria-hidden="true">
                  <SkeletonBlock className="skeleton-block--camera" />
                  <div className="live-camera-stage__placeholder-copy">
                    <SkeletonBlock className="skeleton-block--title" />
                    <SkeletonBlock className="skeleton-block--label" />
                  </div>
                </div>
              ) : (
                <>
                  <strong>
                    <Camera size={18} strokeWidth={2.1} aria-hidden="true" />
                    {cameraState === "error" ? "Camera unavailable" : "Preparing live camera"}
                  </strong>
                  <p>{cameraError || "Allow camera access to start realtime detection."}</p>
                </>
              )}
            </div>
          ) : null}
        </div>

        <div className="live-camera-ribbon live-camera-ribbon--three">
          <div className="live-camera-ribbon__card live-camera-ribbon__card--focus">
            <span className="metric-label">Focus</span>
            <strong>
              {focusLabel}
              {focusTrackingId != null ? ` #${focusTrackingId}` : ""}
            </strong>
            <span>
              {focusObject?.details?.emotion
                ? `${formatObjectLabel(focusObject.details.emotion)} face`
                : focusObject?.details?.gesture
                  ? `${formatObjectLabel(focusObject.details.gesture)} gesture`
                  : focusObject?.details?.handedness
                    ? `${formatObjectLabel(focusObject.details.handedness)} hand`
                  : liveResult?.lighting
                    ? `${formatObjectLabel(liveResult.lighting)} lighting`
                    : "Lighting unknown"}
            </span>
          </div>

          <div className="live-camera-ribbon__card">
            <span className="metric-label">Visible</span>
            <strong>{summaryItemsLabel}</strong>
            <span>{objects.length ? `${objects.length} objects on frame` : "Waiting for frame"}</span>
          </div>

          <div className="live-camera-ribbon__card">
            <span className="metric-label">Reactions</span>
            <div className="live-camera-reaction-list">
              {reactions.map((reaction) => (
                <span key={reaction} className="live-camera-chip live-camera-chip--reaction">
                  {reaction}
                </span>
              ))}
            </div>
          </div>
        </div>

        {cameraError && cameraState === "ready" ? <div className="error-banner">{cameraError}</div> : null}
      </section>
    </div>
  );
}
