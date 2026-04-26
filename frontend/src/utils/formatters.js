const LABEL_ALIASES = {
  face: "Face",
  hand: "Hand",
  handheld_item: "Handheld",
  phone_like_device: "Phone",
  "cell phone": "Phone",
  remote: "Remote",
  "sports ball": "Ball",
  striker: "Striker",
  non_striker: "Non-Striker",
  wicketkeeper: "Wicketkeeper",
  close_fielder: "Close Fielder"
};

export function formatDateTime(value) {
  if (!value) {
    return "Not available";
  }

  try {
    return new Intl.DateTimeFormat("en-IN", {
      dateStyle: "medium",
      timeStyle: "short"
    }).format(new Date(value));
  } catch (_error) {
    return value;
  }
}

export function formatNumber(value) {
  return new Intl.NumberFormat("en-IN", {
    maximumFractionDigits: 1
  }).format(value ?? 0);
}

export function formatObjectLabel(value) {
  if (!value) {
    return "Unknown";
  }

  const normalized = String(value).trim().toLowerCase();
  if (LABEL_ALIASES[normalized]) {
    return LABEL_ALIASES[normalized];
  }

  return String(value)
    .replaceAll(/[_-]+/g, " ")
    .split(" ")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function formatDurationMs(value) {
  const durationMs = Number(value ?? 0);
  if (!Number.isFinite(durationMs) || durationMs <= 0) {
    return "0s";
  }

  if (durationMs < 1000) {
    return `${Math.round(durationMs)} ms`;
  }

  const totalSeconds = durationMs / 1000;
  if (totalSeconds < 60) {
    return `${totalSeconds.toFixed(totalSeconds >= 10 ? 0 : 1)}s`;
  }

  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.round(totalSeconds % 60);
  return `${minutes}m ${seconds}s`;
}

export function getSummarySnapshot(summary) {
  if (!summary) {
    return "";
  }

  const cricket = summary.cricket || {};
  const deliverySummary = cricket.delivery_summary || {};
  if (deliverySummary.total_deliveries > 0) {
    const parts = [
      `${formatNumber(deliverySummary.total_deliveries)} deliveries`
    ];
    if (deliverySummary.wickets > 0) {
      parts.push(`${formatNumber(deliverySummary.wickets)} wickets`);
    }
    if (cricket.speech?.speech_present) {
      parts.push("speech captions");
    }
    return parts.join(" · ");
  }

  const tracksByType = Object.entries(summary.tracks_by_type || {})
    .sort((left, right) => right[1] - left[1])
    .slice(0, 2);

  if (tracksByType.length) {
    return tracksByType
      .map(([objectType, count]) => `${formatNumber(count)} ${formatObjectLabel(objectType)}`)
      .join(" · ");
  }

  if (summary.peak_speed_px_s) {
    return `Peak ${formatNumber(summary.peak_speed_px_s)} px/s`;
  }

  return "";
}

export function getProcessingProgress(session) {
  if (!session) {
    return 0;
  }

  if (session.status === "completed") {
    return 100;
  }

  const progress = Number(session.processingProgress ?? 0);
  if (!Number.isFinite(progress)) {
    return 0;
  }

  return Math.max(0, Math.min(100, Math.round(progress)));
}

export function getStatusLabel(status) {
  const labels = {
    pending: "Queued",
    processing: "Processing",
    completed: "Completed",
    failed: "Failed"
  };

  return labels[status] || status || "Unknown";
}

export function getStatusTone(status) {
  const tones = {
    pending: "pending",
    processing: "processing",
    completed: "completed",
    failed: "failed"
  };

  return tones[status] || "pending";
}

export function getSessionHeadline(session) {
  if (!session) {
    return "No analysis selected";
  }

  if (session.filename) {
    return session.filename;
  }

  return `Analysis ${session.videoId}`;
}

export function getSessionDisplayTitle(session) {
  const headline = getSessionHeadline(session);

  return headline
    .replace(/\.[a-z0-9]{2,5}$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}
