"""Tests for data-generation utilities: prompt_generator and json_utils."""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.prompt_generator import get_image_prompt, get_video_prompt, IMAGE_PROMPTS, VIDEO_PROMPTS
from src.utils.json_utils import read_jsonl, write_jsonl, read_json, write_json


class TestPromptGenerator:
    def test_get_image_prompt_returns_string(self):
        prompt = get_image_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_image_prompt_from_pool(self):
        assert get_image_prompt() in IMAGE_PROMPTS

    def test_get_video_prompt_returns_string(self):
        prompt = get_video_prompt()
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_get_video_prompt_from_pool(self):
        assert get_video_prompt() in VIDEO_PROMPTS


class TestJsonUtils:
    def test_write_and_read_jsonl(self):
        records = [
            {"prompt": "Describe this.", "image_files": ["/tmp/a.jpg"]},
            {"prompt": "What is this?", "video_files": ["/tmp/b.mp4"]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            write_jsonl(records, path)
            result = list(read_jsonl(path))
        assert result == records

    def test_write_jsonl_creates_parent_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "test.jsonl"
            write_jsonl([{"key": "val"}], path)
            assert path.exists()

    def test_read_jsonl_skips_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            path.write_text('{"a": 1}\n\n{"b": 2}\n', encoding="utf-8")
            result = list(read_jsonl(path))
        assert result == [{"a": 1}, {"b": 2}]

    def test_write_and_read_json(self):
        data = {"model": "test", "metrics": {"ttft": 100.0}}
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            write_json(data, path)
            result = read_json(path)
        assert result == data

    def test_write_json_is_pretty_printed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            write_json({"key": "value"}, path)
            content = path.read_text(encoding="utf-8")
        # indent=2 means the file will have newlines
        assert "\n" in content
