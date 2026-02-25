import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Directories
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", ROOT_DIR / "results"))
IMAGES_DIR = DATA_DIR / "images"
VIDEOS_DIR = DATA_DIR / "videos"

# vllm service
VLLM_ENDPOINT = os.getenv("VLLM_ENDPOINT", "http://localhost:8000/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-8B-Instruct")

# Public media sources
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "")  # Required for video downloads
COCO_ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/image_info_val2017.zip"

# Dataset defaults
DEFAULT_IMAGE_COUNT = 100
DEFAULT_VIDEO_COUNT = 100

# Benchmark defaults
REQUEST_TIMEOUT = 120       # seconds per request
MAX_TOKENS = 256            # max tokens to generate per request
CONCURRENCY = 1             # sequential by default
GPU_POLL_INTERVAL = 0.25    # seconds between GPU metric samples
