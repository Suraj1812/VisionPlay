from __future__ import annotations

import unittest

from backend.services.processing_service import ProcessingService

class ProcessingServiceTests(unittest.TestCase):
    def test_progress_defaults_and_completion_state(self) -> None:
        service = ProcessingService()

        self.assertEqual(service.get_progress("video-1", "pending"), 0)
        self.assertEqual(service.get_progress("video-1", "completed"), 100)

        service.set_progress("video-1", 42)
        self.assertEqual(service.get_progress("video-1", "processing"), 42)

        service.clear_progress("video-1")
        self.assertEqual(service.get_progress("video-1", "processing"), 0)

if __name__ == "__main__":
    unittest.main()
