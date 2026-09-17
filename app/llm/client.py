"""Unified local LLM interface.

All application code should import generate() from this module.
The concrete backend can be changed here without touching the pipelines.
"""

from __future__ import annotations

from app.llm.lmstudio_client import generate

__all__ = ["generate"]
