from __future__ import annotations

import json
import unittest
from unittest.mock import PropertyMock, patch

from backend.main import health_check, processing_service, readiness_check

class HealthCheckTests(unittest.TestCase):
    def test_health_endpoint_returns_liveness_payload(self) -> None:
        with patch.object(
            type(processing_service),
            "worker_started",
            new_callable=PropertyMock,
            return_value=True,
        ):
            response = health_check()

        payload = json.loads(response.body.decode("utf-8"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["worker"], "ok")
        self.assertEqual(payload["version"], "1.0.0")

    def test_ready_endpoint_returns_ok_when_dependencies_are_ready(self) -> None:
        with patch.object(
            type(processing_service),
            "worker_started",
            new_callable=PropertyMock,
            return_value=True,
        ):
            response = readiness_check()

        payload = json.loads(response.body.decode("utf-8"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["database"], "ok")
        self.assertEqual(payload["storage"], "ok")
        self.assertEqual(payload["worker"], "ok")

    def test_ready_endpoint_returns_503_when_database_check_fails(self) -> None:
        with patch.object(
            type(processing_service),
            "worker_started",
            new_callable=PropertyMock,
            return_value=True,
        ), patch("backend.main.engine.connect", side_effect=RuntimeError("db down")):
            response = readiness_check()

        payload = json.loads(response.body.decode("utf-8"))

        self.assertEqual(response.status_code, 503)
        self.assertEqual(payload["status"], "degraded")
        self.assertEqual(payload["database"], "error")

if __name__ == "__main__":
    unittest.main()
