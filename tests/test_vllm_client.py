"""Tests for vllm_client helpers (no live endpoint required)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.benchmarking.vllm_client import _image_mime, _build_messages


class TestImageMime:
    def test_jpeg(self):
        assert _image_mime("photo.jpg") == "image/jpeg"
        assert _image_mime("photo.jpeg") == "image/jpeg"

    def test_png(self):
        assert _image_mime("image.png") == "image/png"

    def test_gif(self):
        assert _image_mime("anim.gif") == "image/gif"

    def test_webp(self):
        assert _image_mime("pic.webp") == "image/webp"

    def test_unknown_defaults_to_jpeg(self):
        assert _image_mime("file.unknownext") == "image/jpeg"
        assert _image_mime("noextension") == "image/jpeg"


class TestBuildMessages:
    def test_text_only(self):
        msgs = _build_messages("Hello", [], [])
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
        content = msgs[0]["content"]
        assert len(content) == 1
        assert content[0] == {"type": "text", "text": "Hello"}

    def test_image_entry_structure(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x89PNG\r\n")
        msgs = _build_messages("Describe.", [str(img)], [])
        content = msgs[0]["content"]
        img_entry = content[0]
        assert img_entry["type"] == "image_url"
        assert img_entry["image_url"]["url"].startswith("data:image/png;base64,")

    def test_video_entry_structure(self, tmp_path):
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"\x00\x00\x00\x18ftyp")
        msgs = _build_messages("Describe.", [], [str(vid)])
        content = msgs[0]["content"]
        vid_entry = content[0]
        assert vid_entry["type"] == "video_url"
        assert vid_entry["video_url"]["url"].startswith("data:video/mp4;base64,")

    def test_text_is_last_content_item(self, tmp_path):
        img = tmp_path / "a.jpg"
        img.write_bytes(b"fake")
        msgs = _build_messages("Q?", [str(img)], [])
        content = msgs[0]["content"]
        assert content[-1] == {"type": "text", "text": "Q?"}
