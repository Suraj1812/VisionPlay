from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

class UploadResponse(BaseModel):
    video_id: str
    status: str

class SessionActionResponse(BaseModel):
    deleted_count: int = 0
    deleted_video_ids: list[str] = Field(default_factory=list)
    message: str

class StatusResponse(BaseModel):
    video_id: str
    status: str
    processing_progress: int = 0
    error_message: Optional[str] = None
    uploaded_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

class DetectionResponse(BaseModel):
    frame_id: int
    timestamp_ms: int
    object_type: str
    class_id: int = -1
    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float
    tracking_id: Optional[int] = None

class FrameResultResponse(BaseModel):
    frame_id: int
    timestamp_ms: int
    detections: list[DetectionResponse]

class TrackPointResponse(BaseModel):
    frame_id: int
    timestamp_ms: int
    x: float
    y: float
    confidence: float = 0.0
    scale: float = 0.0

class TrackResponse(BaseModel):
    tracking_id: int
    object_type: str
    class_id: int = -1
    frame_start: int
    frame_end: int
    duration_frames: int = 0
    duration_ms: float = 0.0
    time_in_frame_ms: float = 0.0
    detection_count: int = 0
    avg_confidence: float = 0.0
    max_confidence: float = 0.0
    path: list[TrackPointResponse]
    distance_px: float
    avg_speed_px_s: float
    max_speed_px_s: float
    primary_score: float = 0.0
    is_primary: bool = False

class EventResponse(BaseModel):
    event_type: str
    start_frame: int
    end_frame: int
    duration_frames: int
    object_type: str
    tracking_ids: list[int] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)

class VideoMetadataResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    filename: str
    status: str
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None
    uploaded_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    source_video_url: Optional[str] = None
    processed_video_url: Optional[str] = None

class ResultsResponse(BaseModel):
    video: VideoMetadataResponse
    summary: dict[str, Any]
    events: list[EventResponse]
    frames: list[FrameResultResponse]
    tracks: list[TrackResponse]
    cricket: dict[str, Any] = Field(default_factory=dict)

class LiveDetectionObjectResponse(BaseModel):
    object_type: str
    class_id: int = -1
    bbox: list[float] = Field(min_length=4, max_length=4)
    confidence: float
    tracking_id: Optional[int] = None
    is_predicted: bool = False
    details: dict[str, Any] = Field(default_factory=dict)

class LiveDetectionResponse(BaseModel):
    session_id: str
    frame_index: int
    frame_width: int
    frame_height: int
    focus_label: Optional[str] = None
    focus_tracking_id: Optional[int] = None
    lighting: str = "balanced"
    reactions: list[str] = Field(default_factory=list)
    object_counts: dict[str, int] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    objects: list[LiveDetectionObjectResponse] = Field(default_factory=list)
