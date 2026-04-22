"""Styling assets for the Gradio UI.

Two siblings sit next to this module:

    apple.css  — design tokens + Gradio component overrides
    apple.js   — cursor-tracking 3D tilt + glare on .tilt-card elements

Both are read at import time by ``app.ui`` and injected into ``gr.Blocks``
via the ``css=`` and ``head=`` parameters. Keeping them as separate plain
files (instead of inline string literals) lets editors syntax-highlight
them and lets the browser dev-tools show meaningful sources.
"""
from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve().parent


def load_css() -> str:
    return (_HERE / "apple.css").read_text(encoding="utf-8")


def load_js() -> str:
    return (_HERE / "apple.js").read_text(encoding="utf-8")
