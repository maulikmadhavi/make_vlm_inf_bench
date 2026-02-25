"""OpenAI-compatible streaming client for vllm, capturing TTFT per request."""
import base64
import mimetypes
import time
from pathlib import Path

from openai import OpenAI

from src.utils.config import VLLM_ENDPOINT, MODEL_NAME, MAX_TOKENS, REQUEST_TIMEOUT
from src.utils.logger import get_logger

log = get_logger(__name__)


def _encode_media(path: str) -> str:
    """Base64-encode any media file."""
    return base64.b64encode(Path(path).read_bytes()).decode("utf-8")


def _image_mime(path: str) -> str:
    """Detect image MIME type from file extension; default to image/jpeg."""
    mime, _ = mimetypes.guess_type(path)
    return mime if mime and mime.startswith("image/") else "image/jpeg"


def _build_messages(prompt: str, image_files: list[str], video_files: list[str]) -> list[dict]:
    content: list[dict] = []

    for img_path in image_files:
        b64 = _encode_media(img_path)
        mime = _image_mime(img_path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })

    for vid_path in video_files:
        b64 = _encode_media(vid_path)
        content.append({
            "type": "video_url",
            "video_url": {"url": f"data:video/mp4;base64,{b64}"},
        })

    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


class VLMClient:
    def __init__(
        self,
        endpoint: str = VLLM_ENDPOINT,
        model: str = MODEL_NAME,
    ):
        self.model = model
        self.client = OpenAI(base_url=endpoint, api_key="EMPTY")

    def infer(
        self,
        prompt: str,
        image_files: list[str] | None = None,
        video_files: list[str] | None = None,
    ) -> dict:
        """
        Send a streaming request and return timing + token metrics.

        Returns:
            {
                "ttft_ms": float,          # time to first token (ms)
                "total_time_ms": float,    # total request time (ms)
                "output_tokens": int,      # number of tokens generated
                "tokens_per_second": float,
                "text": str,               # full generated text
            }
        """
        image_files = image_files or []
        video_files = video_files or []
        messages = _build_messages(prompt, image_files, video_files)

        t_start = time.perf_counter()
        ttft_ms: float | None = None
        chunks: list[str] = []
        output_tokens: int | None = None  # filled from usage data if available

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=MAX_TOKENS,
            stream=True,
            stream_options={"include_usage": True},
            timeout=REQUEST_TIMEOUT,
        )

        chunk_count = 0
        for chunk in stream:
            if ttft_ms is None and chunk.choices:
                ttft_ms = (time.perf_counter() - t_start) * 1000

            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                chunks.append(delta)
                chunk_count += 1

            # The final chunk carries accurate token usage when include_usage=True
            if chunk.usage is not None:
                output_tokens = chunk.usage.completion_tokens

        total_time_ms = (time.perf_counter() - t_start) * 1000
        total_time_s = total_time_ms / 1000

        # Fall back to chunk count if server didn't return usage
        if output_tokens is None:
            output_tokens = chunk_count

        # If no content tokens arrived treat TTFT as the full request time
        if ttft_ms is None:
            ttft_ms = total_time_ms

        return {
            "ttft_ms": round(ttft_ms, 2),
            "total_time_ms": round(total_time_ms, 2),
            "output_tokens": output_tokens,
            "tokens_per_second": round(output_tokens / total_time_s, 2) if total_time_s > 0 else 0,
            "text": "".join(chunks),
        }
