"""Unified local LLM interface.

All application code should import generate() from this module.
The concrete backend can be changed here without touching the pipelines —
but whatever backend is wired in here must be local-only (see
app/llm/lmstudio_client.py's docstring for the privacy invariant this
app relies on).
"""

from __future__ import annotations

from app.llm.lmstudio_client import generate

__all__ = ["generate"]
