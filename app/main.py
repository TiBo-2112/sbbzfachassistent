"""Desktop entrypoint: runs the FastAPI backend in a background thread and
opens it in a native pywebview window (no browser tab, no Electron/Node)."""

import threading
import subprocess

import uvicorn
import webview

from app.config import SERVER_HOST, SERVER_PORT
from app.server import app as fastapi_app


class ClipboardApi:
    """Small native bridge for clipboard access inside pywebview."""

    def get_clipboard_text(self) -> str:
        try:
            result = subprocess.run(
                ["pbpaste"],
                capture_output=True,
                text=True,
                check=True,
            )
            return result.stdout
        except Exception:
            return ""



def _run_server() -> None:
    uvicorn.run(fastapi_app, host=SERVER_HOST, port=SERVER_PORT, log_level="warning")


def main() -> None:
    thread = threading.Thread(target=_run_server, daemon=True)
    thread.start()
    # pywebview blocks <a download> / navigations-that-become-downloads by
    # default on every backend (Cocoa, EdgeChromium, GTK, Qt) — without this,
    # the "Herunterladen" link is a silent no-op.
    webview.settings["ALLOW_DOWNLOADS"] = True
    webview.create_window(
        "AnonyMeister",
        f"http://{SERVER_HOST}:{SERVER_PORT}",
        width=1180,
        height=900,
        min_size=(800, 600),
        js_api=ClipboardApi(),
    )
    webview.start()


if __name__ == "__main__":
    main()
