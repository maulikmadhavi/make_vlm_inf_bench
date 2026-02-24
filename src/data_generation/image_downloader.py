"""Downloads real images from the COCO val2017 dataset (no API key required)."""
import json
import random
import zipfile
import io
from pathlib import Path

import requests
from tqdm import tqdm

from src.utils.config import COCO_ANNOTATIONS_URL, IMAGES_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

COCO_BASE_URL = "http://images.cocodataset.org/val2017/"


def _fetch_coco_filenames() -> list[str]:
    """Download COCO val2017 image info and return list of filenames."""
    log.info("Fetching COCO val2017 image list...")
    response = requests.get(COCO_ANNOTATIONS_URL, stream=True, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        with z.open("image_info_val2017.json") as f:
            data = json.load(f)

    return [img["file_name"] for img in data["images"]]


def download_images(count: int, output_dir: Path = IMAGES_DIR) -> list[Path]:
    """Download `count` random images from COCO val2017. Returns list of local paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    filenames = _fetch_coco_filenames()
    selected = random.sample(filenames, min(count, len(filenames)))

    downloaded: list[Path] = []
    for fname in tqdm(selected, desc="Downloading images", unit="img"):
        dest = output_dir / fname
        if dest.exists():
            downloaded.append(dest)
            continue
        try:
            r = requests.get(COCO_BASE_URL + fname, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
            downloaded.append(dest)
        except Exception as e:
            log.warning(f"Failed to download {fname}: {e}")

    log.info(f"Downloaded {len(downloaded)} images to {output_dir}")
    return downloaded
