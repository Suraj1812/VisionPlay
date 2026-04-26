from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.database.entities import VideoStatus
from backend.models.schemas import (
    LiveDetectionResponse,
    ResultsResponse,
    SessionActionResponse,
    StatusResponse,
    UploadResponse,
)
from backend.services.live_detection_service import live_detection_service
from backend.services.processing_service import processing_service
from backend.services.storage_service import storage_service
from backend.services.video_service import VideoService


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/upload-video", response_model=UploadResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_video(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadResponse:
    try:
        storage_service.validate_upload(file)
        storage_service.validate_content_length(request.headers.get("content-length"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    video_service = VideoService(db)
    video = video_service.create_video(filename=file.filename, original_path="")
    destination = None

    try:
        destination = storage_service.build_input_path(video.id, file.filename)
        saved_path, _ = await storage_service.save_upload_file(file, destination)
        video.original_path = str(saved_path)
        db.commit()
        db.refresh(video)
    except ValueError as exc:
        db.rollback()
        if destination is not None:
            storage_service.delete_file(destination)
        logger.warning("Rejected upload for %s: %s", file.filename, exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        db.rollback()
        if destination is not None:
            storage_service.delete_file(destination)
        logger.exception("Failed to save uploaded video")
        raise HTTPException(status_code=500, detail="Unable to save uploaded video") from exc

    logger.info("Queued video %s for processing", video.id)
    processing_service.wake()

    return video_service.build_upload_response(video)


@router.get("/status/{video_id}", response_model=StatusResponse)
def get_status(video_id: str, db: Session = Depends(get_db)) -> StatusResponse:
    video_service = VideoService(db)
    video = video_service.require_video(video_id)
    return video_service.build_status_response(
        video,
        processing_progress=processing_service.get_progress(video.id, video.status.value),
    )


@router.get("/results/{video_id}", response_model=ResultsResponse)
def get_results(video_id: str, db: Session = Depends(get_db)) -> ResultsResponse:
    video_service = VideoService(db)
    return video_service.build_results_response(video_id)


@router.post("/detect-live-frame", response_model=LiveDetectionResponse)
async def detect_live_frame(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
) -> LiveDetectionResponse:
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image frame")

    try:
        payload = await file.read()
        return LiveDetectionResponse.model_validate(
            live_detection_service.detect_frame_bytes(payload, session_id=session_id)
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Live camera detection failed")
        raise HTTPException(status_code=500, detail="Unable to analyze live camera frame") from exc


@router.delete("/videos/{video_id}", response_model=SessionActionResponse)
def delete_video(video_id: str, db: Session = Depends(get_db)) -> SessionActionResponse:
    video_service = VideoService(db)
    processing_service.cancel(video_id)

    try:
        deleted_video_id = video_service.delete_video(video_id)
        processing_service.clear_progress(deleted_video_id)
        db.commit()
    except HTTPException:
        db.rollback()
        raise
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to delete video %s", video_id)
        raise HTTPException(status_code=500, detail="Unable to delete the session") from exc

    return SessionActionResponse(
        deleted_count=1,
        deleted_video_ids=[deleted_video_id],
        message="Session deleted",
    )


@router.post("/sessions/delete-old", response_model=SessionActionResponse)
def delete_old_sessions(db: Session = Depends(get_db)) -> SessionActionResponse:
    video_service = VideoService(db)

    try:
        deleted_video_ids = video_service.delete_videos_by_statuses(
            {VideoStatus.COMPLETED, VideoStatus.FAILED}
        )
        for video_id in deleted_video_ids:
            processing_service.cancel(video_id)
            processing_service.clear_progress(video_id)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to delete old sessions")
        raise HTTPException(status_code=500, detail="Unable to delete old sessions") from exc

    return SessionActionResponse(
        deleted_count=len(deleted_video_ids),
        deleted_video_ids=deleted_video_ids,
        message=(
            f"Deleted {len(deleted_video_ids)} old session"
            f"{'' if len(deleted_video_ids) == 1 else 's'}"
        ),
    )


@router.delete("/sessions", response_model=SessionActionResponse)
def delete_all_sessions(db: Session = Depends(get_db)) -> SessionActionResponse:
    video_service = VideoService(db)

    try:
        deleted_video_ids = video_service.delete_all_videos()
        for video_id in deleted_video_ids:
            processing_service.cancel(video_id)
            processing_service.clear_progress(video_id)
        storage_service.clear_upload_directories()
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.exception("Failed to delete all sessions")
        raise HTTPException(status_code=500, detail="Unable to delete all session data") from exc

    return SessionActionResponse(
        deleted_count=len(deleted_video_ids),
        deleted_video_ids=deleted_video_ids,
        message=(
            "All session data deleted"
            if deleted_video_ids
            else "No session data was stored"
        ),
    )
