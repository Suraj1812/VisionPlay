const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:8000").replace(/\/$/, "");

function parsePayload(body, contentType) {
  if (contentType.includes("application/json")) {
    try {
      return body ? JSON.parse(body) : {};
    } catch (_error) {
      return body;
    }
  }

  return body;
}

function getErrorMessage(payload) {
  if (typeof payload === "string") {
    return payload;
  }

  return payload.detail || payload.message || "Request failed";
}

async function parseResponse(response) {
  const contentType = response.headers.get("content-type") || "";
  const payload = parsePayload(await response.text(), contentType);

  if (!response.ok) {
    throw new Error(getErrorMessage(payload));
  }

  return payload;
}

export async function uploadVideo(file, options = {}) {
  const formData = new FormData();
  formData.append("file", file);

  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", `${API_BASE_URL}/upload-video`);

    request.upload.addEventListener("progress", (event) => {
      if (typeof options.onProgress !== "function") {
        return;
      }

      const total = event.total || file.size || 0;
      const percent = total > 0 ? Math.min(100, Math.round((event.loaded / total) * 100)) : 0;

      options.onProgress({
        loaded: event.loaded,
        total,
        percent
      });
    });

    request.addEventListener("load", () => {
      const contentType = request.getResponseHeader("content-type") || "";
      const payload = parsePayload(request.responseText || "", contentType);

      if (request.status >= 200 && request.status < 300) {
        resolve(payload);
        return;
      }

      reject(new Error(getErrorMessage(payload)));
    });

    request.addEventListener("error", () => {
      reject(new Error("Network error. Please try again."));
    });

    request.addEventListener("abort", () => {
      reject(new Error("Upload cancelled."));
    });

    request.send(formData);
  });
}

export class RateLimitError extends Error {
  constructor(retryAfterSeconds) {
    super("Rate limit exceeded");
    this.name = "RateLimitError";
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

export class NotFoundError extends Error {
  constructor(message = "Resource not found") {
    super(message);
    this.name = "NotFoundError";
  }
}

export async function getStatus(videoId) {
  const response = await fetch(`${API_BASE_URL}/status/${videoId}`);
  if (response.status === 429) {
    const retryAfter = parseInt(response.headers.get("Retry-After") || "10", 10);
    throw new RateLimitError(retryAfter);
  }
  if (response.status === 404) {
    const payload = parsePayload(await response.text(), response.headers.get("content-type") || "");
    throw new NotFoundError(getErrorMessage(payload));
  }
  return parseResponse(response);
}

export async function getResults(videoId) {
  const response = await fetch(`${API_BASE_URL}/results/${videoId}`);
  if (response.status === 404) {
    const payload = parsePayload(await response.text(), response.headers.get("content-type") || "");
    throw new NotFoundError(getErrorMessage(payload));
  }
  return parseResponse(response);
}

export async function detectLiveFrame(frameBlob, sessionId, options = {}) {
  const formData = new FormData();
  formData.append("file", frameBlob, "live-frame.jpg");
  if (sessionId) {
    formData.append("session_id", sessionId);
  }

  const response = await fetch(`${API_BASE_URL}/detect-live-frame`, {
    method: "POST",
    body: formData,
    signal: options.signal
  });

  return parseResponse(response);
}

export async function deleteVideo(videoId) {
  const response = await fetch(`${API_BASE_URL}/videos/${videoId}`, {
    method: "DELETE"
  });
  return parseResponse(response);
}

export async function deleteOldSessions() {
  const response = await fetch(`${API_BASE_URL}/sessions/delete-old`, {
    method: "POST"
  });
  return parseResponse(response);
}

export async function deleteAllSessions() {
  const response = await fetch(`${API_BASE_URL}/sessions`, {
    method: "DELETE"
  });
  return parseResponse(response);
}

export { API_BASE_URL };
