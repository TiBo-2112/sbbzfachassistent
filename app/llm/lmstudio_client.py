"""Thin wrapper around the local LM Studio server."""

from __future__ import annotations

from openai import OpenAI

from app.config import LMSTUDIO_BASE_URL
from app.settings import get_lmstudio_model


def generate(
    prompt: str,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int = 2000,
) -> str:
    messages: list[dict[str, str]] = []

    if system:
        messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    model = get_lmstudio_model()

    kwargs = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
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
