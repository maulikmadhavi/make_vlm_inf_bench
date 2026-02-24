"""Downloads real short videos from the Pexels API (free API key required)."""
from pathlib import Path

import requests
from tqdm import tqdm

from src.utils.config import PEXELS_API_KEY, VIDEOS_DIR
from src.utils.logger import get_logger

log = get_logger(__name__)

PEXELS_API_URL = "https://api.pexels.com/videos/search"
SEARCH_QUERIES = [
    "nature", "city", "people", "animals", "sports",
    "food", "technology", "travel", "ocean", "mountains",
]


def _search_pexels(query: str, per_page: int = 10, page: int = 1) -> list[dict]:
    if not PEXELS_API_KEY:
        raise EnvironmentError(
            "PEXELS_API_KEY environment variable is not set. "
            "Get a free key at https://www.pexels.com/api/"
        )
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": per_page, "page": page}
    r = requests.get(PEXELS_API_URL, headers=headers, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("videos", [])


def _best_video_url(video: dict, max_height: int = 720) -> str | None:
    """Pick the smallest file that is ≤ max_height resolution."""
    files = sorted(video.get("video_files", []), key=lambda f: f.get("height", 0))
    for f in files:
        if f.get("height", 0) <= max_height and f.get("file_type") == "video/mp4":
            return f["link"]
    return None


def download_videos(count: int, output_dir: Path = VIDEOS_DIR) -> list[Path]:
    """Download `count` random videos from Pexels. Returns list of local paths."""
    output_dir.mkdir(parents=True, exist_ok=True)

    video_urls: list[tuple[str, str]] = []  # (url, filename)
    per_query = max(1, count // len(SEARCH_QUERIES) + 1)

    for query in SEARCH_QUERIES:
        if len(video_urls) >= count:
            break
        try:
            videos = _search_pexels(query, per_page=per_query)
            for v in videos:
                url = _best_video_url(v)
                if url:
                    fname = f"{query}_{v['id']}.mp4"
                    video_urls.append((url, fname))
        except Exception as e:
            log.warning(f"Pexels search failed for '{query}': {e}")

    video_urls = video_urls[:count]
    downloaded: list[Path] = []

    for url, fname in tqdm(video_urls, desc="Downloading videos", unit="vid"):
        dest = output_dir / fname
        if dest.exists():
            downloaded.append(dest)
            continue
        try:
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(dest)
        except Exception as e:
            log.warning(f"Failed to download video {fname}: {e}")

    log.info(f"Downloaded {len(downloaded)} videos to {output_dir}")
    return downloaded
