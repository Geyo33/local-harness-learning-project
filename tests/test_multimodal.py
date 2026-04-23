import base64
import os
import tempfile
from pathlib import Path

import pytest

from mcp_chatbot.frontend.multimodal import build_content_array, IMAGE_EXTENSIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_temp_image(suffix=".jpg") -> str:
    """Write a minimal valid JPEG to a temp file and return its path."""
    jpeg_bytes = bytes([
        0xFF,0xD8,0xFF,0xE0,0x00,0x10,0x4A,0x46,0x49,0x46,0x00,0x01,
        0x01,0x00,0x00,0x01,0x00,0x01,0x00,0x00,0xFF,0xDB,0x00,0x43,
        0x00,0x08,0x06,0x06,0x07,0x06,0x05,0x08,0x07,0x07,0x07,0x09,
        0x09,0x08,0x0A,0x0C,0x14,0x0D,0x0C,0x0B,0x0B,0x0C,0x19,0x12,
        0x13,0x0F,0x14,0x1D,0x1A,0x1F,0x1E,0x1D,0x1A,0x1C,0x1C,0x20,
        0x24,0x2E,0x27,0x20,0x22,0x2C,0x23,0x1C,0x1C,0x28,0x37,0x29,
        0x2C,0x30,0x31,0x34,0x34,0x34,0x1F,0x27,0x39,0x3D,0x38,0x32,
        0x3C,0x2E,0x33,0x34,0x32,0xFF,0xC0,0x00,0x0B,0x08,0x00,0x01,
        0x00,0x01,0x01,0x01,0x11,0x00,0xFF,0xC4,0x00,0x1F,0x00,0x00,
        0x01,0x05,0x01,0x01,0x01,0x01,0x01,0x01,0x00,0x00,0x00,0x00,
        0x00,0x00,0x00,0x00,0x01,0x02,0x03,0x04,0x05,0x06,0x07,0x08,
        0x09,0x0A,0x0B,0xFF,0xDA,0x00,0x08,0x01,0x01,0x00,0x00,0x3F,
        0x00,0xFB,0x26,0xAE,0xAF,0xFF,0xD9
    ])
    f = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    f.write(jpeg_bytes)
    f.close()
    return f.name


def _make_temp_text(content="hello world", suffix=".txt") -> str:
    f = tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False, encoding="utf-8")
    f.write(content)
    f.close()
    return f.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBuildContentArray:
    def test_no_files_returns_plain_string(self):
        assert build_content_array("hello", []) == "hello"

    def test_empty_message_no_files_returns_empty_string(self):
        assert build_content_array("", []) == ""

    def test_image_returns_list(self):
        path = _make_temp_image(".jpg")
        try:
            result = build_content_array("look", [path])
            assert isinstance(result, list)
        finally:
            os.unlink(path)

    def test_image_has_text_and_image_url_parts(self):
        path = _make_temp_image(".jpg")
        try:
            result = build_content_array("look", [path])
            types = [p["type"] for p in result]
            assert "text" in types
            assert "image_url" in types
        finally:
            os.unlink(path)

    def test_image_base64_matches_file(self):
        path = _make_temp_image(".png")
        try:
            with open(path, "rb") as f:
                expected_b64 = base64.b64encode(f.read()).decode()
            result = build_content_array("pic", [path])
            img = next(p for p in result if p["type"] == "image_url")
            assert expected_b64 in img["image_url"]["url"]
        finally:
            os.unlink(path)

    def test_png_image_url_has_correct_mime(self):
        path = _make_temp_image(".png")
        try:
            result = build_content_array("pic", [path])
            img = next(p for p in result if p["type"] == "image_url")
            assert img["image_url"]["url"].startswith("data:image/png;base64,")
        finally:
            os.unlink(path)

    def test_jpeg_image_url_has_correct_mime(self):
        path = _make_temp_image(".jpg")
        try:
            result = build_content_array("pic", [path])
            img = next(p for p in result if p["type"] == "image_url")
            assert img["image_url"]["url"].startswith("data:image/jpeg;base64,")
        finally:
            os.unlink(path)

    def test_text_file_content_in_result(self):
        path = _make_temp_text("secret contents")
        try:
            result = build_content_array("read this", [path])
            assert isinstance(result, list)
            combined = " ".join(p["text"] for p in result if p["type"] == "text")
            assert "secret contents" in combined
        finally:
            os.unlink(path)

    def test_text_file_includes_filename_header(self):
        path = _make_temp_text("data", suffix=".py")
        filename = Path(path).name
        try:
            result = build_content_array("read", [path])
            combined = " ".join(p["text"] for p in result if p["type"] == "text")
            assert filename in combined
        finally:
            os.unlink(path)

    def test_unknown_extension_skipped(self):
        f = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
        f.write(b"data")
        f.close()
        try:
            result = build_content_array("msg", [f.name])
            if isinstance(result, list):
                assert all(p["type"] == "text" for p in result)
                assert not any(".xyz" in p.get("text", "") for p in result)
            else:
                assert result == "msg"
        finally:
            os.unlink(f.name)

    def test_empty_message_with_image_has_no_text_part(self):
        path = _make_temp_image(".jpg")
        try:
            result = build_content_array("", [path])
            assert isinstance(result, list)
            assert all(p["type"] != "text" for p in result)
        finally:
            os.unlink(path)

    def test_multiple_images(self):
        p1 = _make_temp_image(".jpg")
        p2 = _make_temp_image(".png")
        try:
            result = build_content_array("two", [p1, p2])
            imgs = [p for p in result if p["type"] == "image_url"]
            assert len(imgs) == 2
        finally:
            os.unlink(p1)
            os.unlink(p2)

    def test_empty_message_with_unknown_extension_returns_string(self):
        f = tempfile.NamedTemporaryFile(suffix=".xyz", delete=False)
        f.write(b"data")
        f.close()
        try:
            result = build_content_array("", [f.name])
            # Should return empty string (plain string fallback), not empty list
            assert result == ""
            assert not isinstance(result, list)
        finally:
            os.unlink(f.name)

    def test_mixed_image_and_text_file(self):
        img = _make_temp_image(".jpg")
        txt = _make_temp_text("notes")
        try:
            result = build_content_array("here", [img, txt])
            assert sum(1 for p in result if p["type"] == "image_url") == 1
            assert sum(1 for p in result if p["type"] == "text") >= 2
        finally:
            os.unlink(img)
            os.unlink(txt)
