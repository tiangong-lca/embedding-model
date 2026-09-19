"""Canonical TIDAS process-to-Markdown adapter."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tidas_sdk import TidasProcess


def tidas_process_to_markdown(process: "TidasProcess", lang: str = "en") -> str:
    """Render a process with the installed TIDAS SDK's canonical formatter."""

    return process.to_markdown(lang=lang)
