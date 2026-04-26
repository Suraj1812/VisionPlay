import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import {
  NotFoundError,
  RateLimitError,
  deleteAllSessions,
  deleteOldSessions,
  deleteVideo,
  getResults,
  getStatus,
  uploadVideo
} from "../api";

const VisionPlayContext = createContext(null);
const STORAGE_KEY = "visionplay:sessions:v16";
const LEGACY_STORAGE_KEYS = [
  "visionplay:sessions",
  "visionplay:sessions:v10",
  "visionplay:sessions:v11",
  "visionplay:sessions:v12",
  "visionplay:sessions:v13",
  "visionplay:sessions:v14",
  "visionplay:sessions:v15"
];

function normalizeSession(session) {
  return {
    videoId: session.videoId,
    filename: session.filename || "",
    status: session.status || "pending",
    processingProgress: Number.isFinite(Number(session.processingProgress))
      ? Math.max(0, Math.min(100, Math.round(Number(session.processingProgress))))
      : 0,
    uploadedAt: session.uploadedAt || null,
    startedAt: session.startedAt || null,
    completedAt: session.completedAt || null,
    errorMessage: session.errorMessage || "",
    updatedAt: session.updatedAt || session.uploadedAt || new Date().toISOString(),
    sourceVideoUrl: session.sourceVideoUrl || "",
    processedVideoUrl: session.processedVideoUrl || "",
    summary: session.summary || null
  };
}

function sortSessions(sessions) {
  return [...sessions].sort((left, right) => {
    const leftTime = new Date(left.updatedAt || left.uploadedAt || 0).getTime();
    const rightTime = new Date(right.updatedAt || right.uploadedAt || 0).getTime();
    return rightTime - leftTime;
  });
}

function readStoredSessions() {
  if (typeof window === "undefined") {
    return [];
  }

  try {
    for (const legacyKey of LEGACY_STORAGE_KEYS) {
      window.localStorage.removeItem(legacyKey);
    }

    const rawValue = window.localStorage.getItem(STORAGE_KEY);
    if (!rawValue) {
      return [];
    }

    const parsed = JSON.parse(rawValue);
    if (!Array.isArray(parsed)) {
      return [];
    }

    return sortSessions(
      parsed
        .filter((item) => item && typeof item.videoId === "string")
        .map((item) => normalizeSession(item))
    );
  } catch (_error) {
    return [];
  }
}

function mergeSessions(currentSessions, partialSession) {
  const existingIndex = currentSessions.findIndex(
    (session) => session.videoId === partialSession.videoId
  );

  if (existingIndex === -1) {
    return sortSessions([
      normalizeSession({
        ...partialSession,
        updatedAt: new Date().toISOString()
      }),
      ...currentSessions
    ]);
  }

  const mergedSession = normalizeSession({
    ...currentSessions[existingIndex],
    ...partialSession,
    updatedAt: new Date().toISOString()
  });

  const nextSessions = [...currentSessions];
  nextSessions[existingIndex] = mergedSession;
  return sortSessions(nextSessions);
}

function removeSessionsById(currentSessions, deletedIds) {
  const deletedSet = new Set(deletedIds);
  return currentSessions.filter((session) => !deletedSet.has(session.videoId));
}

export function VisionPlayProvider({ children }) {
  const [sessions, setSessions] = useState([]);
  const [isBooting, setIsBooting] = useState(true);
  const [resultsByVideoId, setResultsByVideoId] = useState({});
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploadFilename, setUploadFilename] = useState("");
  const [uploadError, setUploadError] = useState("");
  const syncingStatusIds = useRef(new Set());
  const syncingResultIds = useRef(new Set());

  useEffect(() => {
    setSessions(readStoredSessions());
    setIsBooting(false);
  }, []);

  useEffect(() => {
    if (isBooting) {
      return;
    }
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    } catch (_error) {
    }
  }, [isBooting, sessions]);

  const upsertSession = useCallback((partialSession) => {
    setSessions((currentSessions) => mergeSessions(currentSessions, partialSession));
  }, []);

  const removeSessions = useCallback((deletedIds) => {
    if (!deletedIds.length) {
      return;
    }

    for (const videoId of deletedIds) {
      syncingStatusIds.current.delete(videoId);
      syncingResultIds.current.delete(videoId);
    }

    setSessions((currentSessions) => removeSessionsById(currentSessions, deletedIds));
    setResultsByVideoId((currentResults) => {
      const nextResults = { ...currentResults };
      for (const videoId of deletedIds) {
        delete nextResults[videoId];
      }
      return nextResults;
    });
  }, []);

  const fetchResultsForVideo = useCallback(async (videoId) => {
    if (!videoId || syncingResultIds.current.has(videoId)) {
      return resultsByVideoId[videoId] || null;
    }

    if (resultsByVideoId[videoId]) {
      return resultsByVideoId[videoId];
    }

    syncingResultIds.current.add(videoId);

    try {
      const results = await getResults(videoId);
      setResultsByVideoId((currentResults) => ({
        ...currentResults,
        [videoId]: results
      }));

      upsertSession({
        videoId,
        filename: results.video?.filename || "",
        sourceVideoUrl: results.video?.source_video_url || "",
        processedVideoUrl: results.video?.processed_video_url || "",
        summary: results.summary || null,
        status: "completed",
        processingProgress: 100
      });

      return results;
    } catch (error) {
      if (error instanceof NotFoundError) {
        removeSessions([videoId]);
        return null;
      }
      throw error;
    } finally {
      syncingResultIds.current.delete(videoId);
    }
  }, [removeSessions, resultsByVideoId, upsertSession]);

  const syncSession = useCallback(async (videoId) => {
    if (!videoId || syncingStatusIds.current.has(videoId)) {
      return null;
    }

    syncingStatusIds.current.add(videoId);

    try {
      const statusData = await getStatus(videoId);
      upsertSession({
        videoId,
        status: statusData.status,
        processingProgress: statusData.processing_progress,
        uploadedAt: statusData.uploaded_at,
        startedAt: statusData.started_at,
        completedAt: statusData.completed_at,
        errorMessage: statusData.error_message || ""
      });

      if (statusData.status === "completed") {
        await fetchResultsForVideo(videoId);
      }

      return statusData;
    } catch (error) {
      if (error instanceof NotFoundError) {
        removeSessions([videoId]);
        return null;
      }
      throw error;
    } finally {
      syncingStatusIds.current.delete(videoId);
    }
  }, [fetchResultsForVideo, removeSessions, upsertSession]);

  const ensureVideo = useCallback(async (videoId) => {
    const session = sessions.find((item) => item.videoId === videoId);
    if (session?.status === "completed" && !resultsByVideoId[videoId]) {
      await fetchResultsForVideo(videoId);
      return;
    }

    await syncSession(videoId);
  }, [fetchResultsForVideo, resultsByVideoId, sessions, syncSession]);

  const createUpload = useCallback(async (file) => {
    setUploading(true);
    setUploadProgress(0);
    setUploadFilename(file?.name || "");
    setUploadError("");

    try {
      const response = await uploadVideo(file, {
        onProgress(progressEvent) {
          setUploadProgress((currentProgress) =>
            progressEvent.percent > currentProgress ? progressEvent.percent : currentProgress
          );
        }
      });
      setUploadProgress(100);

      upsertSession({
        videoId: response.video_id,
        filename: file.name,
        status: response.status,
        processingProgress: 0,
        uploadedAt: new Date().toISOString(),
        errorMessage: ""
      });

      await syncSession(response.video_id);
      return response.video_id;
    } catch (requestError) {
      setUploadError(requestError.message);
      throw requestError;
    } finally {
      setUploading(false);
      setUploadProgress(0);
      setUploadFilename("");
    }
  }, [syncSession, upsertSession]);

  const clearUploadError = useCallback(() => {
    setUploadError("");
  }, []);

  const refreshSessions = useCallback(async () => {
    const uniqueIds = [...new Set(sessions.map((session) => session.videoId).filter(Boolean))];
    await Promise.all(uniqueIds.map((videoId) => syncSession(videoId).catch(() => null)));
    return uniqueIds.length;
  }, [sessions, syncSession]);

  const removeSession = useCallback(async (videoId) => {
    const response = await deleteVideo(videoId);
    removeSessions(response.deleted_video_ids?.length ? response.deleted_video_ids : [videoId]);
    return response;
  }, [removeSessions]);

  const removeOldSessions = useCallback(async () => {
    const response = await deleteOldSessions();
    removeSessions(response.deleted_video_ids || []);
    return response;
  }, [removeSessions]);

  const removeAllSessions = useCallback(async () => {
    const response = await deleteAllSessions();
    removeSessions(response.deleted_video_ids || sessions.map((session) => session.videoId));
    return response;
  }, [removeSessions, sessions]);

  const getSessionById = useCallback(
    (videoId) => sessions.find((session) => session.videoId === videoId) || null,
    [sessions]
  );

  const getResultsById = useCallback(
    (videoId) => resultsByVideoId[videoId] || null,
    [resultsByVideoId]
  );

  const syncSessionRef = useRef(syncSession);
  syncSessionRef.current = syncSession;

  const pendingIdsKey = useMemo(
    () => {
      const ids = sessions
        .filter((s) => s.status === "pending" || s.status === "processing")
        .map((s) => s.videoId);
      return JSON.stringify([...ids].sort());
    },
    [sessions]
  );

  useEffect(() => {
    const ids = JSON.parse(pendingIdsKey);
    if (!ids.length) {
      return undefined;
    }

    const BASE_POLL_MS = 5000;
    const MAX_POLL_MS = 30000;
    let isActive = true;
    let currentDelay = BASE_POLL_MS;
    let timeoutId;

    async function pollOnce() {
      if (!isActive) {
        return;
      }

      let hitRateLimit = false;

      for (const videoId of ids) {
        if (!isActive) {
          return;
        }

        try {
          await syncSessionRef.current(videoId);
          currentDelay = BASE_POLL_MS;
        } catch (error) {
          if (error instanceof RateLimitError) {
            hitRateLimit = true;
            const serverRetry = (error.retryAfterSeconds || 10) * 1000;
            currentDelay = Math.min(MAX_POLL_MS, Math.max(currentDelay * 2, serverRetry));
          }
        }
      }

      if (hitRateLimit) {
        currentDelay = Math.min(MAX_POLL_MS, currentDelay * 1.5);
      }

      if (isActive) {
        timeoutId = window.setTimeout(pollOnce, currentDelay);
      }
    }

    timeoutId = window.setTimeout(pollOnce, 1500);

    return () => {
      isActive = false;
      window.clearTimeout(timeoutId);
    };
  }, [pendingIdsKey]);

  const latestActiveSession = useMemo(
    () =>
      sessions.find((session) => session.status === "processing") ||
      sessions.find((session) => session.status === "pending") ||
      null,
    [sessions]
  );

  const latestCompletedSession = useMemo(
    () => sessions.find((session) => session.status === "completed") || null,
    [sessions]
  );

  const value = useMemo(
    () => ({
      sessions,
      isBooting,
      uploading,
      uploadProgress,
      uploadFilename,
      uploadError,
      latestActiveSession,
      latestCompletedSession,
      createUpload,
      clearUploadError,
      refreshSessions,
      deleteSession: removeSession,
      deleteOldSessions: removeOldSessions,
      deleteAllSessions: removeAllSessions,
      ensureVideo,
      getSessionById,
      getResultsById
    }),
    [
      sessions,
      isBooting,
      uploading,
      uploadProgress,
      uploadFilename,
      uploadError,
      latestActiveSession,
      latestCompletedSession,
      createUpload,
      clearUploadError,
      refreshSessions,
      removeSession,
      removeOldSessions,
      removeAllSessions,
      ensureVideo,
      getSessionById,
      getResultsById
    ]
  );

  return <VisionPlayContext.Provider value={value}>{children}</VisionPlayContext.Provider>;
}

export function useVisionPlay() {
  const context = useContext(VisionPlayContext);

  if (!context) {
    throw new Error("useVisionPlay must be used within VisionPlayProvider");
  }

  return context;
}
