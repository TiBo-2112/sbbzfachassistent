"""Checks the local components the pipeline depends on.

LM Studio is this app's only local LLM backend (see app/llm/client.py) — the
Ollama-specific dependency checks/model-pull helpers this module used to have
(HTTP calls against OLLAMA_HOST, NDJSON pull-progress parsing) were removed
once nothing in the pipeline called Ollama anymore; keeping them around would
have left a live, working code path for a backend this app no longer uses,
which is exactly the kind of thing that could mislead a future reader about
which service document text actually goes to.
"""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Callable

from app.config import LMSTUDIO_BASE_URL, SPACY_MODELS
from app.schemas import DependencyStatus
from app.settings import get_lmstudio_model, get_whisper_model_size

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
    # on_pull_progress is unused for every branch below — LM Studio has no
    # HTTP pull API to stream progress from (models are downloaded through
    # its own app), unlike Ollama's /api/pull. Kept as a parameter since
    # app.server's caller still passes one generically for every dependency
    # kind; it simply has nothing to call right now.

    if name == "lmstudio":
        return _check_lmstudio()

    if name.startswith("lmstudio model "):
        return _check_lmstudio_model()

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

    if name == "faster-whisper model cache":
        return _check_whisper_cache()

    raise ValueError(
        f"Unrecognized dependency name: {name!r}"
    )
