"""Checks the local components the pipeline depends on.

LM Studio is the primary local LLM backend.
Legacy Ollama helpers are temporarily retained during migration.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Callable

from app.config import LMSTUDIO_BASE_URL, OLLAMA_HOST, SPACY_MODELS
from app.schemas import DependencyStatus
from app.settings import (
    get_lmstudio_model,
    get_ollama_model,
    get_whisper_model_size,
)

_OLLAMA_HTTP_TIMEOUT = 2
_OLLAMA_PULL_TIMEOUT = 600
_LMSTUDIO_HTTP_TIMEOUT = 2
_SPACY_DOWNLOAD_TIMEOUT = 600


# ---------------------------------------------------------------------------
# LM Studio
# ---------------------------------------------------------------------------

def _lmstudio_models() -> list[str] | None:
    """Return model IDs reported by LM Studio, or None if unreachable."""
    try:
        with urllib.request.urlopen(
            f"{LMSTUDIO_BASE_URL}/models",
            timeout=_LMSTUDIO_HTTP_TIMEOUT,
        ) as response:
            payload = json.loads(response.read())
    except (
        urllib.error.URLError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None

    return [
        item.get("id", "")
        for item in payload.get("data", [])
        if item.get("id")
    ]


def list_lmstudio_models() -> list[dict]:
    """Return models currently exposed by LM Studio."""
    models = _lmstudio_models()

    if models is None:
        return []

    return [
        {"name": model, "size": ""}
        for model in models
    ]


def _check_lmstudio() -> DependencyStatus:
    models = _lmstudio_models()

    if models is not None:
        return DependencyStatus(
            name="lmstudio",
            available=True,
            detail=f"LM Studio is reachable at {LMSTUDIO_BASE_URL}.",
        )

    return DependencyStatus(
        name="lmstudio",
        available=False,
        detail=f"LM Studio is not reachable at {LMSTUDIO_BASE_URL}.",
        install_hint=(
            "Start LM Studio and enable the local API server "
            "on 127.0.0.1:1234."
        ),
    )


def _check_lmstudio_model() -> DependencyStatus:
    model = get_lmstudio_model()
    name = f"lmstudio model {model}"
    models = _lmstudio_models()

    if models is None:
        return DependencyStatus(
            name=name,
            available=False,
            detail="Could not reach LM Studio to check available models.",
            install_hint="Start LM Studio and enable the local API server.",
        )

    if model in models:
        return DependencyStatus(
            name=name,
            available=True,
            detail=f"Model '{model}' is available in LM Studio.",
        )

    return DependencyStatus(
        name=name,
        available=False,
        detail=f"Model '{model}' is not available in LM Studio.",
        install_hint="Load or download the model in LM Studio.",
    )


# ---------------------------------------------------------------------------
# Legacy Ollama support
# ---------------------------------------------------------------------------

def _ollama_tags_payload() -> list[dict] | None:
    try:
        with urllib.request.urlopen(
            f"{OLLAMA_HOST}/api/tags",
            timeout=_OLLAMA_HTTP_TIMEOUT,
        ) as response:
            payload = json.loads(response.read())
    except (
        urllib.error.URLError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ):
        return None

    return payload.get("models", [])


def _ollama_tags() -> list[str] | None:
    models = _ollama_tags_payload()

    if models is None:
        return None

    return [
        model.get("name", "")
        for model in models
    ]


def _format_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or unit == "TB":
            return (
                f"{n:.0f} {unit}"
                if unit == "B"
                else f"{n:.1f} {unit}"
            )
        n /= 1000

    return f"{n:.1f} TB"


def _progress_bar(percent: float, width: int = 20) -> str:
    filled = max(
        0,
        min(width, round(width * percent / 100)),
    )

    return "▕" + "█" * filled + "░" * (width - filled) + "▏"


def list_ollama_models() -> list[dict]:
    models = _ollama_tags_payload()

    if models is None:
        return []

    return [
        {
            "name": model["name"],
            "size": _format_bytes(model.get("size", 0)),
        }
        for model in models
        if model.get("name")
    ]


def _check_ollama() -> DependencyStatus:
    if _ollama_tags() is not None:
        return DependencyStatus(
            name="ollama",
            available=True,
            detail=f"Ollama is reachable at {OLLAMA_HOST}.",
        )

    if shutil.which("ollama") is None:
        return DependencyStatus(
            name="ollama",
            available=False,
            detail=f"Ollama is not reachable at {OLLAMA_HOST}.",
            install_hint="https://ollama.com/download",
        )

    hint = (
        "open -a Ollama"
        if platform.system() == "Darwin"
        else "ollama serve"
    )

    return DependencyStatus(
        name="ollama",
        available=False,
        detail=(
            "Ollama is installed but not running "
            f"(no response from {OLLAMA_HOST})."
        ),
        install_hint=hint,
    )


def _check_ollama_model() -> DependencyStatus:
    model = get_ollama_model()
    name = f"ollama model {model}"
    tags = _ollama_tags()

    if tags is None:
        return DependencyStatus(
            name=name,
            available=False,
            detail=(
                f"Could not reach Ollama at {OLLAMA_HOST} "
                "to check installed models."
            ),
            install_hint=f"ollama pull {model}",
        )

    if model in tags:
        return DependencyStatus(
            name=name,
            available=True,
            detail=f"Model '{model}' is available on the Ollama server.",
        )

    return DependencyStatus(
        name=name,
        available=False,
        detail=(
            f"Model '{model}' was not found on the "
            f"Ollama server ({OLLAMA_HOST})."
        ),
        install_hint=f"ollama pull {model}",
    )


def _ollama_pull_via_http(
    model: str,
    on_pull_progress: Callable[
        [list[str], float | None],
        None,
    ]
    | None = None,
) -> bool:
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/pull",
        data=json.dumps({"name": model}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    lines: list[str] = []
    line_index: dict[str, int] = {}
    layer_progress: dict[str, tuple[int, int]] = {}

    def _set_line(key: str, text: str) -> None:
        if key in line_index:
            lines[line_index[key]] = text
        else:
            line_index[key] = len(lines)
            lines.append(text)

    def _overall_percent() -> float | None:
        if not layer_progress:
            return None

        completed = sum(
            completed
            for completed, _ in layer_progress.values()
        )

        total = sum(
            total
            for _, total in layer_progress.values()
        )

        if total <= 0:
            return None

        return round(
            min(100.0, completed / total * 100),
            1,
        )

    saw_success = False

    try:
        with urllib.request.urlopen(
            request,
            timeout=_OLLAMA_PULL_TIMEOUT,
        ) as response:
            for raw_line in response:
                if not raw_line.strip():
                    continue

                try:
                    event = json.loads(raw_line)
                except (
                    json.JSONDecodeError,
                    UnicodeDecodeError,
                ):
                    continue

                status = event.get("status", "")
                digest = event.get("digest")

                if digest:
                    total = event.get("total", 0)
                    completed = event.get("completed", 0)

                    layer_progress[digest] = (
                        completed,
                        total,
                    )

                    short_digest = digest.removeprefix("sha256:")[:12]

                    if total > 0:
                        percent = min(
                            100.0,
                            completed / total * 100,
                        )

                        text = (
                            f"pulling {short_digest}: "
                            f"{_progress_bar(percent)} "
                            f"{percent:5.1f}% "
                            f"({_format_bytes(completed)}/"
                            f"{_format_bytes(total)})"
                        )
                    else:
                        text = (
                            f"pulling {short_digest}: {status}"
                        )

                    _set_line(digest, text)

                elif status:
                    _set_line(status, status)

                    if status == "success":
                        saw_success = True

                if on_pull_progress:
                    on_pull_progress(
                        list(lines),
                        _overall_percent(),
                    )

        return saw_success

    except (
        urllib.error.URLError,
        OSError,
    ):
        return False


# ---------------------------------------------------------------------------
# spaCy
# ---------------------------------------------------------------------------

def _check_spacy_models() -> list[DependencyStatus]:
    try:
        import spacy.util
    except ImportError:
        return [
            DependencyStatus(
                name=f"spaCy model {model_name}",
                available=False,
                detail="spaCy is not installed.",
                install_hint="pip install spacy",
            )
            for model_name in SPACY_MODELS.values()
        ]

    statuses = []

    for model_name in SPACY_MODELS.values():
        name = f"spaCy model {model_name}"

        if spacy.util.is_package(model_name):
            statuses.append(
                DependencyStatus(
                    name=name,
                    available=True,
                    detail=f"spaCy model '{model_name}' is installed.",
                )
            )
        else:
            statuses.append(
                DependencyStatus(
                    name=name,
                    available=False,
                    detail=f"spaCy model '{model_name}' is not installed.",
                    install_hint=f"python -m spacy download {model_name}",
                )
            )

    return statuses


# ---------------------------------------------------------------------------
# Whisper
# ---------------------------------------------------------------------------

def _check_whisper_cache() -> DependencyStatus:
    size = get_whisper_model_size()

    return DependencyStatus(
        name="faster-whisper model cache",
        available=True,
        detail=(
            f"faster-whisper downloads the '{size}' model automatically "
            "on first transcription and caches it locally afterwards; "
            "no manual setup is required."
        ),
    )


# ---------------------------------------------------------------------------
# Public dependency API
# ---------------------------------------------------------------------------

def check_dependencies() -> list[DependencyStatus]:
    statuses = [
        _check_lmstudio(),
        _check_lmstudio_model(),
    ]

    statuses.extend(_check_spacy_models())
    statuses.append(_check_whisper_cache())

    return statuses


def attempt_auto_install(
    name: str,
    on_pull_progress: Callable[
        [list[str], float | None],
        None,
    ]
    | None = None,
) -> DependencyStatus:

    if name == "lmstudio":
        return _check_lmstudio()

    if name.startswith("lmstudio model "):
        return _check_lmstudio_model()

    if name == f"ollama model {get_ollama_model()}":
        pulled = _ollama_pull_via_http(
            get_ollama_model(),
            on_pull_progress=on_pull_progress,
        )

        status = _check_ollama_model()

        if not pulled and not status.available:
            return DependencyStatus(
                name=status.name,
                available=False,
                detail=(
                    "Der Download wurde unterbrochen, "
                    "bevor er abgeschlossen war."
                ),
                install_hint=status.install_hint,
            )

        return status

    if name.startswith("spaCy model "):
        model_name = name.removeprefix("spaCy model ")

        if model_name not in SPACY_MODELS.values():
            raise ValueError(
                f"Unrecognized dependency name: {name!r}"
            )

        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "spacy",
                    "download",
                    model_name,
                ],
                capture_output=True,
                text=True,
                timeout=_SPACY_DOWNLOAD_TIMEOUT,
            )
        except (
            FileNotFoundError,
            subprocess.TimeoutExpired,
            OSError,
        ):
            pass

        return next(
            status
            for status in _check_spacy_models()
            if status.name == name
        )

    if name == "ollama":
        return _check_ollama()

    if name == "faster-whisper model cache":
        return _check_whisper_cache()

    raise ValueError(
        f"Unrecognized dependency name: {name!r}"
    )
