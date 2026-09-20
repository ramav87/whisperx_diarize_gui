from __future__ import annotations

import logging
import os
import socket
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

logger = logging.getLogger("diarize_gui.local_server")


@dataclass
class LocalServerHandle:
    host: str
    port: int
    url: str
    server: object
    thread: Optional[threading.Thread]
    owned: bool

    def stop(self, timeout: float = 5.0) -> None:
        if not self.owned:
            return
        if hasattr(self.server, "should_exit"):
            setattr(self.server, "should_exit", True)
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=timeout)


def default_local_server_host() -> str:
    return os.environ.get("DIARIZE_GUI_SERVER_HOST") or os.environ.get("DIARIZE_SERVER_HOST") or "127.0.0.1"


def default_local_server_port() -> int:
    raw = os.environ.get("DIARIZE_GUI_SERVER_PORT") or os.environ.get("DIARIZE_SERVER_PORT") or "8000"
    try:
        return int(raw)
    except ValueError:
        logger.warning("invalid server port %r; falling back to 8000", raw)
        return 8000


def default_local_server_url() -> str:
    return f"http://{default_local_server_host()}:{default_local_server_port()}"


def start_local_server_if_enabled() -> Optional[LocalServerHandle]:
    if os.environ.get("DIARIZE_GUI_START_SERVER", "1").strip().lower() in {"0", "false", "no", "off"}:
        logger.info("local server auto-start disabled")
        return None

    host = default_local_server_host()
    port = default_local_server_port()
    url = f"http://{host}:{port}"

    if _health_ok(url):
        logger.info("using existing diarize server at %s", url)
        return LocalServerHandle(host=host, port=port, url=url, server=None, thread=None, owned=False)
    if _port_is_open(host, port):
        logger.warning("port %s is already in use, but %s/api/health did not respond as diarize", port, url)
        return None

    try:
        import uvicorn

        from .api import app
    except ModuleNotFoundError as exc:
        logger.exception(
            "could not import server runtime; install server dependencies with "
            "`python -m pip install -e .` or `python -m pip install fastapi 'uvicorn[standard]' python-multipart`"
        )
        print(
            "Could not start local diarize server: missing Python package "
            f"{exc.name!r}. Install server dependencies with:\n"
            "  python -m pip install -e .\n"
            "or:\n"
            "  python -m pip install fastapi 'uvicorn[standard]' python-multipart"
        )
        return None
    except Exception:
        logger.exception("could not import server runtime")
        return None

    # uvloop can wedge while accepting requests when it runs in a secondary
    # thread alongside Tk on macOS. The stdlib loop is stable for the embedded
    # server; standalone deployments may still choose uvloop themselves.
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        loop="asyncio",
        log_level=os.environ.get("DIARIZE_LOG_LEVEL", "warning").lower(),
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="diarize-local-server", daemon=True)
    thread.start()

    if _wait_for_health(url):
        logger.info("started local diarize server at %s", url)
    else:
        logger.warning("local diarize server was started but did not pass health check at %s", url)
    return LocalServerHandle(host=host, port=port, url=url, server=server, thread=thread, owned=True)


def _wait_for_health(url: str, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _health_ok(url):
            return True
        time.sleep(0.2)
    return False


def _health_ok(url: str) -> bool:
    try:
        with urlopen(f"{url.rstrip('/')}/api/health", timeout=0.5) as response:
            return response.status == 200 and b'"ok"' in response.read(512)
    except (OSError, URLError, TimeoutError):
        return False


def _port_is_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.3):
            return True
    except OSError:
        return False
