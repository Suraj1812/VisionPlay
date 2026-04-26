from __future__ import annotations

import re
import shutil
from contextlib import suppress
from pathlib import Path

from fastapi import UploadFile

from backend.utils.config import settings


class StorageService:
    supported_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    supported_content_types = {
        "application/octet-stream",
        "binary/octet-stream",
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/x-matroska",
        "video/webm",
        "video/x-m4v",
        "video/mpeg",
    }

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        clean_name = Path(filename).name
        sanitized = re.sub(r"[^A-Za-z0-9._-]", "_", clean_name)
        stem = Path(sanitized).stem[:180]
        suffix = Path(sanitized).suffix[:20]
        return f"{stem}{suffix}" if stem else f"upload{suffix or '.mp4'}"

    def is_supported_video(self, filename: str) -> bool:
        return Path(filename).suffix.lower() in self.supported_extensions

    def is_supported_content_type(self, content_type: str | None) -> bool:
        if not content_type:
            return True
        normalized = content_type.split(";", 1)[0].strip().lower()
        return normalized in self.supported_content_types

    def validate_upload(self, upload_file: UploadFile) -> None:
        if not upload_file.filename:
            raise ValueError("Missing video filename")
        if not self.is_supported_video(upload_file.filename):
            raise ValueError("Unsupported video format")
        if not self.is_supported_content_type(upload_file.content_type):
            raise ValueError("Unsupported video content type")

    def validate_content_length(self, raw_content_length: str | None) -> None:
        if not raw_content_length:
            return

        try:
            content_length = int(raw_content_length)
        except ValueError as exc:
            raise ValueError("Invalid upload content length") from exc

        if content_length <= 0:
            raise ValueError("Uploaded video is empty")
        if content_length > settings.max_upload_size_bytes:
            raise ValueError(
                f"Uploaded video exceeds the {settings.max_upload_size_mb} MB limit"
            )

    def build_input_path(self, video_id: str, filename: str) -> Path:
        safe_name = self.sanitize_filename(filename)
        return settings.upload_input_dir / f"{video_id}_{safe_name}"

    def build_output_path(self, video_id: str, filename: str) -> Path:
        safe_name = self.sanitize_filename(filename)
        return settings.upload_output_dir / f"{video_id}_{Path(safe_name).stem}_processed.mp4"

    async def save_upload_file(self, upload_file: UploadFile, destination: Path) -> tuple[Path, int]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_destination = destination.with_name(f"{destination.name}.part")
        bytes_written = 0

        try:
            with temp_destination.open("wb") as buffer:
                while chunk := await upload_file.read(settings.upload_chunk_size_bytes):
                    bytes_written += len(chunk)
                    if bytes_written > settings.max_upload_size_bytes:
                        raise ValueError(
                            f"Uploaded video exceeds the {settings.max_upload_size_mb} MB limit"
                        )
                    buffer.write(chunk)

            if bytes_written <= 0:
                raise ValueError("Uploaded video is empty")

            temp_destination.replace(destination)
            return destination, bytes_written
        except Exception:
            with suppress(FileNotFoundError):
                temp_destination.unlink()
            raise
        finally:
            await upload_file.close()

    @staticmethod
    def delete_file(file_path: str | Path | None) -> None:
        if not file_path:
            return
        with suppress(FileNotFoundError):
            Path(file_path).unlink()

    def media_url_for(self, file_path: str | Path | None) -> str | None:
        if file_path is None:
            return None
        absolute_path = Path(file_path).resolve()
        try:
            relative_path = absolute_path.relative_to(settings.media_root_path)
        except ValueError:
            return None
        return f"{settings.media_mount_path}/{relative_path.as_posix()}"

    def clear_upload_directories(self) -> None:
        for directory in (settings.upload_input_dir, settings.upload_output_dir):
            if not directory.exists():
                directory.mkdir(parents=True, exist_ok=True)
                continue

            for child in directory.iterdir():
                if child.is_dir():
                    shutil.rmtree(child, ignore_errors=True)
                    continue
                with suppress(FileNotFoundError):
                    child.unlink()


storage_service = StorageService()
