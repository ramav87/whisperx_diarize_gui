import types
import unittest
from unittest.mock import patch

from diarize_gui import local_server


class FakeServer:
    def __init__(self, config):
        self.config = config
        self.should_exit = False
        self.ran = False

    def run(self):
        self.ran = True


class LocalServerTests(unittest.TestCase):
    def test_auto_start_can_be_disabled(self):
        with patch.dict("os.environ", {"DIARIZE_GUI_START_SERVER": "0"}):
            self.assertIsNone(local_server.start_local_server_if_enabled())

    def test_existing_healthy_server_is_reused(self):
        with patch.object(local_server, "_health_ok", return_value=True):
            handle = local_server.start_local_server_if_enabled()

        self.assertIsNotNone(handle)
        self.assertFalse(handle.owned)
        self.assertEqual(handle.url, "http://127.0.0.1:8000")

    def test_server_starts_in_background_when_port_is_available(self):
        fake_uvicorn = types.SimpleNamespace(Config=lambda *args, **kwargs: kwargs, Server=FakeServer)
        with patch.dict("sys.modules", {"uvicorn": fake_uvicorn}), patch.object(
            local_server, "_health_ok", return_value=False
        ), patch.object(local_server, "_port_is_open", return_value=False), patch.object(
            local_server, "_wait_for_health", return_value=True
        ):
            handle = local_server.start_local_server_if_enabled()

        self.assertIsNotNone(handle)
        self.assertTrue(handle.owned)
        handle.stop()
        self.assertTrue(handle.server.should_exit)


if __name__ == "__main__":
    unittest.main()
