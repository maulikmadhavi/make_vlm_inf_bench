"""Generates JSONL benchmark datasets from real downloaded media."""
from pathlib import Path

from src.data_generation.image_downloader import download_images
from src.data_generation.video_downloader import download_videos
from src.data_generation.prompt_generator import get_image_prompt, get_video_prompt
from src.utils.config import DATA_DIR
from src.utils.json_utils import write_jsonl
from src.utils.logger import get_logger

log = get_logger(__name__)


def generate_image_dataset(count: int, output_path: Path | None = None) -> Path:
    """Download `count` images and write an image JSONL dataset."""
    output_path = output_path or DATA_DIR / "images_dataset.jsonl"
    images = download_images(count)

    records = [
        {"prompt": get_image_prompt(), "image_files": [str(p.resolve())]}
        for p in images
    ]
    write_jsonl(records, output_path)
    log.info(f"Image dataset written to {output_path} ({len(records)} records)")
    return output_path


def generate_video_dataset(count: int, output_path: Path | None = None) -> Path:
    """Download `count` videos and write a video JSONL dataset."""
    output_path = output_path or DATA_DIR / "videos_dataset.jsonl"
    videos = download_videos(count)

    records = [
        {"prompt": get_video_prompt(), "video_files": [str(p.resolve())]}
        for p in videos
    ]
    write_jsonl(records, output_path)
    log.info(f"Video dataset written to {output_path} ({len(records)} records)")
    return output_path
