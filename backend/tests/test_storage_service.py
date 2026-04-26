from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from fastapi import UploadFile
from starlette.datastructures import Headers

from backend.services.storage_service import StorageService
from backend.utils.config import settings

class StorageServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_upload_file_rejects_empty_upload(self) -> None:
        service = StorageService()

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "clip.mp4"
            upload = UploadFile(file=io.BytesIO(b""), filename="clip.mp4")

            with self.assertRaisesRegex(ValueError, "empty"):
                await service.save_upload_file(upload, destination)

            self.assertFalse(destination.exists())
            self.assertFalse((Path(temp_dir) / "clip.mp4.part").exists())

    async def test_save_upload_file_rejects_oversized_upload(self) -> None:
        service = StorageService()

        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "clip.mp4"
            upload = UploadFile(file=io.BytesIO(b"0123456789"), filename="clip.mp4")

            with patch.object(
                type(settings),
                "max_upload_size_bytes",
                new_callable=PropertyMock,
                return_value=4,
            ):
                with self.assertRaisesRegex(ValueError, "exceeds"):
                    await service.save_upload_file(upload, destination)

            self.assertFalse(destination.exists())
            self.assertFalse((Path(temp_dir) / "clip.mp4.part").exists())

    def test_validate_upload_checks_extension_and_content_type(self) -> None:
        service = StorageService()

        valid_upload = UploadFile(
            file=io.BytesIO(b"data"),
            filename="clip.mp4",
            headers=Headers({"content-type": "video/mp4"}),
        )
        service.validate_upload(valid_upload)

        invalid_upload = UploadFile(
            file=io.BytesIO(b"data"),
            filename="clip.txt",
            headers=Headers({"content-type": "text/plain"}),
        )
        with self.assertRaisesRegex(ValueError, "Unsupported video format"):
            service.validate_upload(invalid_upload)

    def test_validate_content_length_rejects_oversized_payloads(self) -> None:
        service = StorageService()

        with patch.object(
            type(settings),
            "max_upload_size_bytes",
            new_callable=PropertyMock,
            return_value=10,
        ):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                service.validate_content_length("11")

        service.validate_content_length("10")

    def test_clear_upload_directories_removes_stored_files(self) -> None:
        service = StorageService()

        with tempfile.TemporaryDirectory() as temp_dir:
            input_dir = Path(temp_dir) / "input"
            output_dir = Path(temp_dir) / "output"
            input_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            (input_dir / "clip.mp4").write_bytes(b"input")
            (output_dir / "clip_processed.mp4").write_bytes(b"output")

            with patch.object(
                type(settings),
                "upload_input_dir",
                new_callable=PropertyMock,
                return_value=input_dir,
            ), patch.object(
                type(settings),
                "upload_output_dir",
                new_callable=PropertyMock,
                return_value=output_dir,
            ):
                service.clear_upload_directories()

            self.assertEqual(list(input_dir.iterdir()), [])
            self.assertEqual(list(output_dir.iterdir()), [])
