"""Thin wrapper around the local LM Studio server.

Privacy invariant: local-only document processing is a hard requirement of
this app (see CLAUDE.md), not a preference — no stage may send raw or
anonymized content to a network API. LM Studio is this app's only local-LLM
backend (deep-check, transcript correction, and summarization all route
through generate() below), which is only acceptable because it is guaranteed
to run on this machine, never over the network. That guarantee used to rest
on Ollama always being "the local, never-networked instance" purely by
deployment convention; `_assert_local_base_url()` makes it a real code-level
check instead — generate() refuses to call LM Studio at all if
LMSTUDIO_BASE_URL has been pointed at anything other than loopback, so a
misconfigured env var can't silently turn this into a network call.
"""

from __future__ import annotations

from urllib.parse import urlparse

from openai import OpenAI

from app.config import LMSTUDIO_BASE_URL
from app.settings import get_lmstudio_model

_LOOPBACK_HOSTNAMES = {"127.0.0.1", "localhost", "::1"}


def _assert_local_base_url(base_url: str) -> None:
    hostname = urlparse(base_url).hostname
    if hostname not in _LOOPBACK_HOSTNAMES:
        raise RuntimeError(
            f"LMSTUDIO_BASE_URL is set to '{base_url}', which is not a "
            "loopback address. This app never sends document content to a "
            "non-local host, so it refuses to call LM Studio there. Point "
            "LMSTUDIO_BASE_URL at 127.0.0.1/localhost, or run LM Studio on "
            "this machine."
        )


def generate(
    prompt: str,
    system: str | None = None,
    temperature: float | None = None,
) -> str:
    _assert_local_base_url(LMSTUDIO_BASE_URL)

    messages: list[dict[str, str]] = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    model = get_lmstudio_model()

    kwargs = {
        "model": model,
        "messages": messages,
    }

    if temperature is not None:
        kwargs["temperature"] = temperature

    try:
        client = OpenAI(
            base_url=LMSTUDIO_BASE_URL,
            api_key="lm-studio",
        )

        response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content

        if not content:
            raise RuntimeError("LM Studio returned an empty response.")

    except Exception as exc:
        raise RuntimeError(
            f"Could not reach the local LM Studio model "
            f"'{model}' at {LMSTUDIO_BASE_URL}. "
            "Make sure LM Studio is running, the local server is started, "
            "and the selected model is available."
        ) from exc

    return content
