from __future__ import annotations

import logging
from pathlib import Path
from threading import Event, Lock, Thread
import time
from typing import TYPE_CHECKING

from backend.database.entities import VideoStatus
from backend.database.session import SessionLocal
from backend.services.storage_service import storage_service
from backend.services.video_service import VideoService
from backend.utils.config import settings

if TYPE_CHECKING:
    from ai.pipeline.video_processor import VideoProcessor


logger = logging.getLogger(__name__)


class ProcessingCancelledError(RuntimeError):
    pass


class ProcessingService:
    def __init__(self) -> None:
        self._processor: VideoProcessor | None = None
        self._processor_lock = Lock()
        self._progress_by_video_id: dict[str, int] = {}
        self._progress_lock = Lock()
        self._active_video_ids: set[str] = set()
        self._active_video_ids_lock = Lock()
        self._cancelled_video_ids: set[str] = set()
        self._cancelled_video_ids_lock = Lock()
        self._worker_threads: list[Thread] = []
        self._worker_control_lock = Lock()
        self._wake_event = Event()
        self._stop_event = Event()
        self._last_cleanup_at = 0.0

    @property
    def processor(self) -> VideoProcessor:
        if self._processor is None:
            with self._processor_lock:
                if self._processor is None:
                    from ai.pipeline.video_processor import VideoProcessor

                    self._processor = VideoProcessor()
        return self._processor

    @property
    def worker_started(self) -> bool:
        with self._worker_control_lock:
            return any(thread.is_alive() for thread in self._worker_threads)

    def start(self) -> None:
        with self._worker_control_lock:
            active_threads = [thread for thread in self._worker_threads if thread.is_alive()]
            if active_threads:
                self._worker_threads = active_threads
                return

            self._stop_event.clear()
            self._wake_event.clear()
            self._worker_threads = []

            recovered_ids = self._recover_orphaned_jobs()
            if recovered_ids:
                logger.warning(
                    "Recovered %s stale processing job(s) after startup restart",
                    len(recovered_ids),
                )

            for index in range(settings.processing_worker_count):
                worker = Thread(
                    target=self._worker_loop,
                    name=f"visionplay-worker-{index + 1}",
                    daemon=True,
                )
                worker.start()
                self._worker_threads.append(worker)

            self._wake_event.set()

    def stop(self, join_timeout_seconds: float = 5.0) -> None:
        with self._worker_control_lock:
            threads = [thread for thread in self._worker_threads if thread.is_alive()]
            if not threads:
                self._worker_threads = []
                return

            self._stop_event.set()
            self._wake_event.set()
            self._worker_threads = []

        for thread in threads:
            thread.join(timeout=join_timeout_seconds)

        self._stop_event.clear()
        self._wake_event.clear()

    def wake(self) -> None:
        self._wake_event.set()

    def set_progress(self, video_id: str, progress: int) -> None:
        clamped = max(0, min(int(progress), 100))
        with self._progress_lock:
            self._progress_by_video_id[video_id] = clamped

    def get_progress(self, video_id: str, status: str | None = None) -> int:
        if status == "completed":
            return 100
        if status == "pending":
            return 0

        with self._progress_lock:
            return self._progress_by_video_id.get(video_id, 0)

    def clear_progress(self, video_id: str) -> None:
        with self._progress_lock:
            self._progress_by_video_id.pop(video_id, None)

    def cancel(self, video_id: str) -> None:
        with self._cancelled_video_ids_lock:
            self._cancelled_video_ids.add(video_id)
        self.clear_progress(video_id)
        self._wake_event.set()

    def process_video(self, video_id: str) -> None:
        db = SessionLocal()
        video_service = VideoService(db)
        try:
            video = video_service.require_video(video_id)
            if video.status == VideoStatus.PENDING:
                video_service.mark_processing(video)
                db.commit()
            elif video.status != VideoStatus.PROCESSING:
                logger.warning(
                    "Ignoring direct processing request for %s with status=%s",
                    video_id,
                    video.status.value,
                )
                return
        finally:
            db.close()

        self._process_claimed_video(video_id)

    def _worker_loop(self) -> None:
        poll_timeout = max(settings.processing_poll_interval_ms / 1000, 0.1)

        while not self._stop_event.is_set():
            self._run_periodic_cleanup()
            claimed_video_id = self._claim_next_pending_video_id()
            if claimed_video_id is None:
                self._wake_event.wait(timeout=poll_timeout)
                self._wake_event.clear()
                continue

            self._process_claimed_video(claimed_video_id)

    def _claim_next_pending_video_id(self) -> str | None:
        db = SessionLocal()
        try:
            video_service = VideoService(db)
            claimed_video = video_service.claim_next_pending_video()
            return claimed_video.id if claimed_video is not None else None
        except Exception:
            logger.exception("Unable to claim the next pending video for processing")
            return None
        finally:
            db.close()

    def _recover_orphaned_jobs(self) -> list[str]:
        db = SessionLocal()
        try:
            video_service = VideoService(db)
            return video_service.recover_stale_processing_videos(
                settings.processing_stale_job_seconds
            )
        except Exception:
            logger.exception("Unable to recover stale processing jobs at startup")
            return []
        finally:
            db.close()

    def _run_periodic_cleanup(self) -> None:
        now = time.monotonic()
        if now - self._last_cleanup_at < settings.cleanup_scan_interval_seconds:
            return

        self._last_cleanup_at = now
        db = SessionLocal()
        try:
            video_service = VideoService(db)
            deleted_ids = video_service.delete_expired_videos()
            for video_id in deleted_ids:
                self.clear_progress(video_id)
            if deleted_ids:
                logger.info("Deleted %s expired video session(s)", len(deleted_ids))
        except Exception:
            logger.exception("Unable to clean up expired video sessions")
        finally:
            db.close()

    def _process_claimed_video(self, video_id: str) -> None:
        with self._active_video_ids_lock:
            if video_id in self._active_video_ids:
                logger.warning("Skipping duplicate processing request for video %s", video_id)
                return
            self._active_video_ids.add(video_id)

        db = SessionLocal()
        video_service = VideoService(db)
        output_path = None

        try:
            if self._is_cancelled(video_id):
                raise ProcessingCancelledError("Processing cancelled before start")

            self.set_progress(video_id, 0)
            video = video_service.require_video(video_id)
            if video.status != VideoStatus.PROCESSING:
                logger.info("Video %s is no longer in processing state; skipping worker run", video_id)
                return
            if not video.original_path or not Path(video.original_path).exists():
                raise FileNotFoundError("Uploaded source video could not be found")

            output_path = storage_service.build_output_path(video.id, video.filename)
            processing_result = self.processor.process_video(
                video_path=video.original_path,
                output_path=str(output_path),
                progress_callback=lambda processed_frames, total_frames: self._handle_progress(
                    video_id=video.id,
                    processed_frames=processed_frames,
                    total_frames=total_frames,
                ),
            )

            if self._is_cancelled(video.id):
                raise ProcessingCancelledError("Processing cancelled")

            video_service.store_processing_results(
                video_id=video.id,
                processed_path=str(output_path),
                processing_result=processing_result,
            )
            db.commit()
            self.set_progress(video.id, 100)
            logger.info("Completed processing for video %s", video_id)
        except ProcessingCancelledError:
            logger.info("Cancelled processing for video %s", video_id)
            db.rollback()
            if output_path is not None:
                storage_service.delete_file(output_path)

            video = video_service.get_video(video_id)
            if video is not None and video.status == VideoStatus.PROCESSING:
                video_service.mark_failed(video, "Processing cancelled")
                db.commit()
        except Exception as exc:
            logger.exception("Video processing failed for %s", video_id)
            db.rollback()
            if output_path is not None:
                storage_service.delete_file(output_path)
            video = video_service.get_video(video_id)
            if video is not None:
                video_service.mark_failed(video, str(exc))
                db.commit()
        finally:
            self.clear_progress(video_id)
            self._clear_cancellation(video_id)
            with self._active_video_ids_lock:
                self._active_video_ids.discard(video_id)
            db.close()

    def _handle_progress(self, video_id: str, processed_frames: int, total_frames: int) -> None:
        if self._is_cancelled(video_id):
            raise ProcessingCancelledError("Processing cancelled")

        if total_frames <= 0:
            return

        percent = int((max(processed_frames, 0) / total_frames) * 100)
        if processed_frames < total_frames:
            percent = min(percent, 99)
        self.set_progress(video_id, percent)

    def _is_cancelled(self, video_id: str) -> bool:
        with self._cancelled_video_ids_lock:
            return video_id in self._cancelled_video_ids

    def _clear_cancellation(self, video_id: str) -> None:
        with self._cancelled_video_ids_lock:
            self._cancelled_video_ids.discard(video_id)


processing_service = ProcessingService()
