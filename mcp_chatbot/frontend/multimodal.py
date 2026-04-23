from __future__ import annotations

import base64
import logging
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp"}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".json", ".csv", ".yaml", ".yml",
    ".toml", ".xml", ".html", ".js", ".ts", ".css", ".sh",
    ".env", ".ini", ".cfg", ".log",
}
MIME_MAP = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


def build_content_array(message: str, files: list[str]) -> str | list[dict]:
    """
    Build an OpenAI-compatible content value from a text message and file paths.

    Returns a plain string when no files are present (preserves existing behavior).
    Returns a list[dict] content array when files are present.

    Images are base64-encoded with a data URI.
    Text files are read and injected as text parts with a filename header.
    Unknown extensions are skipped silently.
    """
    if not files:
        return message

    content: list[dict] = []

    if message.strip():
        content.append({"type": "text", "text": message})

    for file_path in files:
        ext = Path(file_path).suffix.lower()
        try:
            if ext in IMAGE_EXTENSIONS:
                mime = MIME_MAP.get(ext, "image/jpeg")
                with open(file_path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            elif ext in TEXT_EXTENSIONS:
                with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                    file_content = f.read()
                filename = Path(file_path).name
                content.append({
                    "type": "text",
                    "text": f"--- {filename} ---\n{file_content}",
                })
            # Unknown extensions: skip silently
        except Exception as e:
            logging.warning("Skipping file %s: %s", file_path, e)

    # If all files were unknown extensions and message was blank, fall back to plain string
    if not content:
        return message
    return content
