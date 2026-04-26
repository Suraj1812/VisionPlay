from __future__ import annotations

from pathlib import Path

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_SQLITE_PATH = ROOT_DIR / "backend" / "database" / "visionplay.db"


class Settings(BaseSettings):
    app_name: str = "VisionPlay"
    environment: str = "development"
    debug: bool = False
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    allowed_hosts: str = "localhost,127.0.0.1,testserver,*.railway.app"
    cors_origins: str = "http://localhost:5173,http://localhost:4173"
    database_url: str = f"sqlite:///{DEFAULT_SQLITE_PATH}"
    media_root: str = "backend/uploads"
    media_mount_path: str = "/media"
    gzip_minimum_size_bytes: int = Field(default=1024, ge=256, le=1048576)
    request_id_header_name: str = "X-Request-ID"
    slow_request_threshold_ms: int = Field(default=1200, ge=100, le=60000)
    global_rate_limit_per_minute: int = Field(default=600, ge=0, le=10000)
    upload_rate_limit_per_minute: int = Field(default=12, ge=0, le=1000)
    live_rate_limit_per_minute: int = Field(default=1800, ge=0, le=10000)
    status_rate_limit_per_minute: int = Field(default=120, ge=0, le=10000)
    database_pool_size: int = Field(default=5, ge=1, le=50)
    database_max_overflow: int = Field(default=10, ge=0, le=100)
    database_pool_timeout_seconds: int = Field(default=30, ge=1, le=300)
    database_pool_recycle_seconds: int = Field(default=1800, ge=30, le=86400)
    database_connect_timeout_seconds: int = Field(default=5, ge=1, le=60)
    sqlite_busy_timeout_ms: int = Field(default=30000, ge=1000, le=120000)
    upload_chunk_size_bytes: int = Field(default=1024 * 1024, ge=65536, le=16 * 1024 * 1024)
    processing_worker_count: int = Field(default=1, ge=1, le=4)
    processing_poll_interval_ms: int = Field(default=1500, ge=100, le=60000)
    processing_stale_job_seconds: int = Field(default=180, ge=30, le=86400)
    retention_completed_days: int = Field(default=14, ge=0, le=365)
    retention_failed_days: int = Field(default=3, ge=0, le=365)
    cleanup_scan_interval_seconds: int = Field(default=3600, ge=60, le=86400)
    yolo_model_path: str = "auto"
    yolo_auto_model_candidates: str = "yolov8n.pt,yolov8s.pt"
    yolo_device: str = "auto"
    model_cache_dir: str = "backend/model_cache"
    auto_download_auxiliary_models: bool = True
    yolo_confidence: float = Field(default=0.26, ge=0.0, le=1.0)
    yolo_image_size: int = Field(default=640, ge=320, le=2048)
    target_detection_fps: int = Field(default=5, ge=4, le=30)
    target_cricket_analysis_fps: int = Field(default=5, ge=4, le=30)
    target_scoreboard_analysis_fps: int = Field(default=1, ge=1, le=15)
    adaptive_roi_refinement_enabled: bool = False
    adaptive_roi_refinement_frame_stride: int = Field(default=2, ge=1, le=12)
    adaptive_roi_refinement_sparse_detection_count: int = Field(default=3, ge=0, le=20)
    adaptive_roi_refinement_max_regions: int = Field(default=5, ge=1, le=12)
    adaptive_roi_refinement_expand_ratio: float = Field(default=0.2, ge=0.0, le=0.8)
    adaptive_roi_refinement_min_area_ratio: float = Field(default=0.015, ge=0.001, le=0.4)
    adaptive_roi_refinement_max_area_ratio: float = Field(default=0.42, ge=0.05, le=0.95)
    adaptive_roi_refinement_image_size: int = Field(default=1280, ge=320, le=2048)
    cricket_profile_min_confidence: float = Field(default=0.58, ge=0.0, le=1.0)
    cricket_specialized_detection_fps: int = Field(default=10, ge=4, le=30)
    cricket_specialized_analysis_fps: int = Field(default=12, ge=4, le=30)
    cricket_specialized_ball_tile_stride: int = Field(default=2, ge=1, le=12)
    cricket_team_color_clusters: int = Field(default=3, ge=2, le=5)
    audio_transcription_enabled: bool = True
    audio_transcription_model_size: str = "tiny.en"
    audio_transcription_compute_type: str = "int8"
    audio_transcription_language: str = "en"
    audio_transcription_cpu_threads: int = Field(default=4, ge=1, le=32)
    audio_transcription_min_confidence: float = Field(default=0.44, ge=0.0, le=1.0)
    tracked_class_names: str = "person,sports ball,baseball bat"
    small_object_class_names: str = "sports ball"
    class_confidence_thresholds: str = (
        "sports ball=0.25,baseball bat=0.35,"
        "person=0.30"
    )
    class_min_box_sizes_px: str = "sports ball=6,baseball bat=15,person=18"
    class_max_area_ratios: str = "sports ball=0.015,person=0.28"
    class_aspect_ratio_ranges: str = (
        "sports ball=0.4:2.5,"
        "person=0.15:1.2,baseball bat=0.08:0.6"
    )
    small_object_tile_grid_size: int = Field(default=1, ge=1, le=4)
    small_object_tile_overlap_ratio: float = Field(default=0.30, ge=0.0, lt=0.5)
    small_object_tile_image_size: int = Field(default=640, ge=320, le=2048)
    small_object_tile_frame_stride: int = Field(default=6, ge=1, le=12)
    cricket_focus_roi_enabled: bool = True
    cricket_focus_roi_frame_stride: int = Field(default=24, ge=1, le=60)
    cricket_focus_roi_image_size: int = Field(default=768, ge=320, le=2048)
    small_object_track_confidence_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    detection_nms_iou_threshold: float = Field(default=0.40, ge=0.0, le=1.0)
    tracker_activation_threshold: float = Field(default=0.20, ge=0.0, le=1.0)
    tracker_max_stale_frames: int = Field(default=20, ge=1, le=240)
    small_object_stale_track_multiplier: float = Field(default=2.2, gt=1.0, le=4.0)
    detection_frame_stride: int = Field(default=3, ge=1, le=8)
    tracker_step_px: float = Field(default=280.0, gt=0.0)
    small_object_tracker_step_multiplier: float = Field(default=1.8, gt=0.5, le=4.0)
    tracker_min_iou: float = Field(default=0.08, ge=0.0, le=1.0)
    tracker_hard_reject_distance_multiplier: float = Field(default=1.55, gt=1.0, le=4.0)
    tracker_motion_inconsistency_penalty: float = Field(default=0.35, ge=0.0, le=2.0)
    tracker_appearance_similarity_bonus: float = Field(default=0.8, ge=0.0, le=2.0)
    tracker_appearance_reject_threshold: float = Field(default=0.42, ge=0.0, le=1.0)
    tracker_appearance_blend_alpha: float = Field(default=0.35, ge=0.0, le=1.0)
    max_tracker_size_ratio_delta: float = Field(default=3.2, gt=1.0)
    tracker_box_smoothing_factor: float = Field(default=0.28, ge=0.0, le=1.0)
    max_upload_size_mb: int = Field(default=512, ge=1)
    max_error_message_length: int = Field(default=500, ge=64)
    min_detection_box_size_px: float = Field(default=10.0, gt=0.0)
    max_detection_area_ratio: float = Field(default=0.45, gt=0.0, lt=1.0)
    min_detection_confidence: float = Field(default=0.3, ge=0.0, le=1.0)
    edge_ignore_margin_ratio: float = Field(default=0.02, ge=0.0, lt=0.2)
    edge_low_confidence_reject_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    instant_track_confidence: float = Field(default=0.84, ge=0.0, le=1.0)
    min_track_hits: int = Field(default=3, ge=1, le=20)
    min_track_points: int = Field(default=3, ge=1, le=100)
    min_track_distance_px: float = Field(default=8.0, ge=0.0)
    min_track_avg_confidence: float = Field(default=0.42, ge=0.0, le=1.0)
    min_small_object_track_avg_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    max_segment_speed_px_s: float = Field(default=4000.0, gt=0.0)
    small_object_max_segment_speed_px_s: float = Field(default=5200.0, gt=0.0)
    max_track_gap_frames: int = Field(default=10, ge=1, le=240)
    scene_cut_mean_diff_threshold: float = Field(default=28.0, ge=0.0, le=255.0)
    overlay_ignore_top_ratio: float = Field(default=0.1, ge=0.0, lt=0.4)
    overlay_ignore_bottom_ratio: float = Field(default=0.16, ge=0.0, lt=0.5)
    overlay_ignore_side_ratio: float = Field(default=0.035, ge=0.0, lt=0.2)
    overlay_overlap_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    live_tracker_frame_rate: int = Field(default=12, ge=1, le=60)
    live_detection_frame_stride: int = Field(default=2, ge=1, le=6)
    live_session_ttl_seconds: int = Field(default=45, ge=5, le=3600)
    live_aux_frame_stride: int = Field(default=2, ge=1, le=8)
    live_aux_max_dimension: int = Field(default=960, ge=256, le=2048)
    live_aux_track_ttl_frames: int = Field(default=6, ge=1, le=60)
    live_aux_track_min_hits: int = Field(default=2, ge=1, le=12)
    live_aux_track_min_iou: float = Field(default=0.16, ge=0.0, le=1.0)
    live_aux_track_center_distance_ratio: float = Field(default=0.18, ge=0.01, le=1.0)
    live_aux_track_smoothing_factor: float = Field(default=0.32, ge=0.0, le=1.0)
    primary_track_count: int = Field(default=5, ge=1, le=20)
    trajectory_smoothing_factor: float = Field(default=0.35, ge=0.0, le=1.0)
    small_object_trajectory_smoothing_factor: float = Field(default=0.5, ge=0.0, le=1.0)
    interaction_distance_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    interaction_min_consecutive_frames: int = Field(default=2, ge=1, le=60)
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @computed_field  # type: ignore[misc]
    @property
    def media_root_path(self) -> Path:
        return (ROOT_DIR / self.media_root).resolve()

    @computed_field  # type: ignore[misc]
    @property
    def upload_input_dir(self) -> Path:
        return self.media_root_path / "input"

    @computed_field  # type: ignore[misc]
    @property
    def upload_output_dir(self) -> Path:
        return self.media_root_path / "output"

    @computed_field  # type: ignore[misc]
    @property
    def model_cache_path(self) -> Path:
        return (ROOT_DIR / self.model_cache_dir).resolve()

    @computed_field  # type: ignore[misc]
    @property
    def resolved_yolo_model_path(self) -> str:
        configured_path = self.yolo_model_path.strip()
        if configured_path and configured_path.lower() != "auto":
            candidate = Path(configured_path)
            if candidate.exists():
                return str(candidate.resolve())
            root_candidate = (ROOT_DIR / configured_path).resolve()
            if root_candidate.exists():
                return str(root_candidate)
            return configured_path

        for candidate_name in self.yolo_auto_model_candidate_list:
            candidate_path = (ROOT_DIR / candidate_name).resolve()
            if candidate_path.exists():
                return str(candidate_path)

        auto_candidates = self.yolo_auto_model_candidate_list
        return auto_candidates[0] if auto_candidates else "yolov8n.pt"

    @computed_field  # type: ignore[misc]
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def trusted_hosts(self) -> list[str]:
        return [host.strip() for host in self.allowed_hosts.split(",") if host.strip()]

    @property
    def tracked_class_name_list(self) -> list[str]:
        return [value.strip().lower() for value in self.tracked_class_names.split(",") if value.strip()]

    @property
    def small_object_class_name_list(self) -> list[str]:
        return [value.strip().lower() for value in self.small_object_class_names.split(",") if value.strip()]

    @property
    def class_confidence_threshold_map(self) -> dict[str, float]:
        return self._parse_float_map(self.class_confidence_thresholds)

    @property
    def yolo_auto_model_candidate_list(self) -> list[str]:
        return [value.strip() for value in self.yolo_auto_model_candidates.split(",") if value.strip()]

    @property
    def class_min_box_size_map(self) -> dict[str, float]:
        return self._parse_float_map(self.class_min_box_sizes_px)

    @property
    def class_max_area_ratio_map(self) -> dict[str, float]:
        return self._parse_float_map(self.class_max_area_ratios)

    @property
    def class_aspect_ratio_range_map(self) -> dict[str, tuple[float, float]]:
        ranges: dict[str, tuple[float, float]] = {}
        for item in self.class_aspect_ratio_ranges.split(","):
            if "=" not in item or ":" not in item:
                continue
            label, raw_range = item.split("=", 1)
            lower_raw, upper_raw = raw_range.split(":", 1)
            try:
                lower = float(lower_raw.strip())
                upper = float(upper_raw.strip())
            except ValueError:
                continue
            key = label.strip().lower()
            if not key or lower <= 0 or upper <= 0:
                continue
            ranges[key] = (min(lower, upper), max(lower, upper))
        return ranges

    def ensure_directories(self) -> None:
        self.media_root_path.mkdir(parents=True, exist_ok=True)
        self.upload_input_dir.mkdir(parents=True, exist_ok=True)
        self.upload_output_dir.mkdir(parents=True, exist_ok=True)
        self.model_cache_path.mkdir(parents=True, exist_ok=True)
        sqlite_path = self.sqlite_file_path
        if sqlite_path is not None:
            sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def sqlite_file_path(self) -> Path | None:
        if not self.database_url.startswith("sqlite:///"):
            return None
        raw_path = self.database_url.replace("sqlite:///", "", 1)
        if raw_path.startswith("/"):
            return Path(raw_path)
        return (ROOT_DIR / raw_path).resolve()

    @staticmethod
    def _parse_float_map(value: str) -> dict[str, float]:
        parsed: dict[str, float] = {}
        for item in value.split(","):
            if "=" not in item:
                continue
            key, raw_number = item.split("=", 1)
            try:
                parsed_value = float(raw_number.strip())
            except ValueError:
                continue
            normalized_key = key.strip().lower()
            if not normalized_key:
                continue
            parsed[normalized_key] = parsed_value
        return parsed


settings = Settings()
