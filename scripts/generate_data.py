"""CLI: Download real media and generate JSONL benchmark datasets."""
import argparse
import sys
from pathlib import Path

# Allow running from repo root: python scripts/generate_data.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_generation.dataset_generator import generate_image_dataset, generate_video_dataset
from src.utils.config import DEFAULT_IMAGE_COUNT, DEFAULT_VIDEO_COUNT, DATA_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate JSONL datasets from real images/videos for VLM benchmarking."
    )
    parser.add_argument("--images", type=int, default=0,
                        help=f"Number of images to download (default: {DEFAULT_IMAGE_COUNT})")
    parser.add_argument("--videos", type=int, default=0,
                        help=f"Number of videos to download (default: {DEFAULT_VIDEO_COUNT})")
    parser.add_argument("--image-output", type=Path, default=DATA_DIR / "images_dataset.jsonl",
                        help="Output path for image JSONL dataset")
    parser.add_argument("--video-output", type=Path, default=DATA_DIR / "videos_dataset.jsonl",
                        help="Output path for video JSONL dataset")
    args = parser.parse_args()

    if args.images == 0 and args.videos == 0:
        # Default: generate both
        args.images = DEFAULT_IMAGE_COUNT
        args.videos = DEFAULT_VIDEO_COUNT

    if args.images > 0:
        path = generate_image_dataset(args.images, args.image_output)
        print(f"Image dataset: {path}")

    if args.videos > 0:
        path = generate_video_dataset(args.videos, args.video_output)
        print(f"Video dataset: {path}")


if __name__ == "__main__":
    main()
